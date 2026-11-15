# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""The seam between the two phones' pictures, for follow.py's pass A: a
lowest-cost path down the overlap, searched for every frame.

The search is a loop over rows in Python (16 ms a frame on a cloud
instance, more than the GPU needs for the rest of the frame), and each
frame's search starts from the last frame's seam, so it can't be spread
over frames. Run as a script it is that search as a process of its own,
which follow.py feeds and reads through pipes (`Seams`), so the frames'
GPU work doesn't wait for it.

    seam.py <lam>     stdin:  int32 rows, int32 cols, uint8 alone, rows x cols float32 cost; rows = 0 ends
                      stdout: int32 rows, rows int32 columns
"""

from __future__ import annotations

import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np


def seam_dp(cost: np.ndarray, prev: np.ndarray | None, lam: float) -> np.ndarray:
    """Lowest-cost top-to-bottom path, one column per row, moving at most
    two columns per row; `lam` per column per row pulls it towards the
    previous frame's path so the seam doesn't jump."""
    H, W = cost.shape
    if prev is not None:
        cost = cost + lam * np.abs(np.arange(W)[None, :] - prev[:, None])
    acc = cost[0].copy()
    back = np.zeros((H, W), np.int8)
    big = np.full(2, np.inf)
    for r in range(1, H):
        padded = np.concatenate([big, acc, big])
        cand = np.stack([padded[2 + k:2 + k + W] for k in (-2, -1, 0, 1, 2)])
        k = np.argmin(cand, 0)
        back[r] = k - 2
        acc = cost[r] + cand[k, np.arange(W)]
    path = np.empty(H, np.int32)
    path[-1] = int(np.argmin(acc))
    for r in range(H - 1, 0, -1):
        path[r - 1] = path[r] + back[r, path[r]]
    return path


# The same search for an NVIDIA GPU: one block, a thread per column, the
# rows in turn. CELL is the arithmetic for one cell, in double precision as
# seam_dp's is, shared with a CPU build that tests/test_seam.py holds to
# seam_dp.
CELL = r"""
/* The cost of (r, c): the frame's, plus the pull towards the last seam. */
SEAM_FN double seam_cost(const float *cost, const int *prev, int r, int c, int W, double lam, int pulled)
{
    double v = (double)cost[r * W + c];
    if (pulled) {
        int d = c - prev[r];
        v = v + lam * (double)(d < 0 ? -d : d);
    }
    return v;
}

/* The cheapest of the five cells above (r, c) (two columns either way; the
   first of equals, as argmin gives): its cost, and which in *k (-2..2). */
SEAM_FN double seam_best(const double *acc, int c, int W, signed char *k)
{
    double best = 0.0;
    int have = 0, j;
    *k = -2;
    for (j = -2; j <= 2; j++) {
        int cc = c + j;
        if (cc < 0 || cc >= W) continue;
        if (!have || acc[cc] < best) { best = acc[cc]; *k = (signed char)j; have = 1; }
    }
    return best;
}
"""

CPU = "#define SEAM_FN static\n" + CELL + r"""
void seam_frame(const float *cost, const int *prev, int *path, signed char *back,
                int H, int W, double lam, int pulled)
{
    double acc[2][4096];
    int r, c, cur = 0;
    for (c = 0; c < W; c++) acc[0][c] = seam_cost(cost, prev, 0, c, W, lam, pulled);
    for (r = 1; r < H; r++) {
        for (c = 0; c < W; c++) {
            signed char k;
            double v = seam_best(acc[cur], c, W, &k);
            back[r * W + c] = k;
            acc[1 - cur][c] = seam_cost(cost, prev, r, c, W, lam, pulled) + v;
        }
        cur = 1 - cur;
    }
    int at = 0;
    for (c = 1; c < W; c++) if (acc[cur][c] < acc[cur][at]) at = c;
    path[H - 1] = at;
    for (r = H - 1; r > 0; r--) { at = at + back[r * W + at]; path[r - 1] = at; }
}
"""

MAX_COLS = 1024  # a block's threads

CUDA = "#define SEAM_FN __device__ __forceinline__\n" + CELL + r"""
extern "C" __global__
void seam_frame(const float *cost, const int *prev, int *path, signed char *back,
                int H, int W, double lam, int pulled)
{
    __shared__ double acc[2][1024];
    int c = threadIdx.x, cur = 0, r;
    if (c < W) acc[0][c] = seam_cost(cost, prev, 0, c, W, lam, pulled);
    __syncthreads();
    for (r = 1; r < H; r++) {
        if (c < W) {
            signed char k;
            double v = seam_best(acc[cur], c, W, &k);
            back[r * W + c] = k;
            acc[1 - cur][c] = seam_cost(cost, prev, r, c, W, lam, pulled) + v;
        }
        __syncthreads();
        cur = 1 - cur;
    }
    if (c == 0) {
        int at = 0, i;
        for (i = 1; i < W; i++) if (acc[cur][i] < acc[cur][at]) at = i;
        path[H - 1] = at;
        for (r = H - 1; r > 0; r--) { at = at + back[r * W + at]; path[r - 1] = at; }
    }
}
"""


class GpuSeams:
    """`Seams` for costs that are on an NVIDIA GPU: each frame's search is
    one kernel there, so no cost comes back to the CPU and nothing waits
    for a search in Python (15 ms a frame, the slowest thing in pass A once
    the background model was a kernel). `check`: also search the old way
    and say in `results` how many frames' seams differ."""

    def __init__(self, lam: float, check: bool = False):
        import cupy as cp

        self.cp = cp
        self.kernel = cp.RawKernel(CUDA, "seam_frame", options=("--fmad=false",))
        self.lam, self.prev, self.out, self.back = lam, None, [], None
        self.twin = Seams(lam, worker=True) if check else None

    def submit(self, cost, alone: bool) -> None:
        import torch

        if self.twin is not None:
            self.twin.submit(cost.cpu().numpy(), alone)
        if not alone or self.prev is None:
            H, W = cost.shape
            if W > MAX_COLS:
                raise ValueError(f"the overlap is {W} columns; the GPU search takes {MAX_COLS}")
            cost = cost.contiguous().float()
            if self.back is None:
                self.back = torch.empty(H * W, dtype=torch.int8, device=cost.device)
            path = torch.empty(H, dtype=torch.int32, device=cost.device)
            cp = self.cp
            self.kernel((1,), (MAX_COLS,),
                        (cp.asarray(cost), cp.asarray(self.prev if self.prev is not None else path),
                         cp.asarray(path), cp.asarray(self.back), np.int32(H), np.int32(W),
                         np.float64(self.lam), np.int32(self.prev is not None)))
            self.prev = path
        self.out.append(self.prev)

    def results(self) -> list[np.ndarray]:
        import torch

        out = list(torch.stack(self.out).cpu().numpy()) if self.out else []
        if self.twin is not None:
            want = self.twin.results()
            bad = sum(not np.array_equal(a, b) for a, b in zip(out, want))
            print(f"seam check: {bad} of {len(out)} frames' seams differ from the search in Python", flush=True)
        return out


class Seams:
    """Each frame's seam, searched as the frames come. `submit` a frame's
    cost (`alone`: Secondary has no footage for it, so the last seam is
    kept); `results` ends the search and returns every frame's seam, in
    order. `worker`: search in a process of its own instead of here."""

    def __init__(self, lam: float, worker: bool):
        self.lam, self.prev, self.out = lam, None, []
        self.proc = None
        if worker:
            self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), repr(float(lam))],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            # Bounded, so a search that falls behind slows the frames down
            # instead of holding thousands of costs in memory.
            self.q: queue.Queue = queue.Queue(maxsize=64)
            self.sent = 0
            self.threads = [threading.Thread(target=self._write, daemon=True), threading.Thread(target=self._read, daemon=True)]
            for t in self.threads:
                t.start()

    def submit(self, cost: np.ndarray, alone: bool) -> None:
        if self.proc is None:
            if not alone or self.prev is None:
                self.prev = seam_dp(cost, self.prev, self.lam)
            self.out.append(self.prev)
        else:
            self.sent += 1
            self.q.put((np.ascontiguousarray(cost, np.float32), alone))

    def _write(self) -> None:
        try:
            while (item := self.q.get()) is not None:
                cost, alone = item
                self.proc.stdin.write(struct.pack("<iiB", *cost.shape, int(alone)) + cost.tobytes())
            self.proc.stdin.write(struct.pack("<iiB", 0, 0, 0))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # The worker died: results() says so. Keep emptying the queue, or
            # the frames would wait on it forever.
            while self.q.get() is not None:
                pass

    def _read(self) -> None:
        f = self.proc.stdout
        while len(head := f.read(4)) == 4:
            n = struct.unpack("<i", head)[0]
            self.out.append(np.frombuffer(f.read(4 * n), np.int32).copy())

    def results(self) -> list[np.ndarray]:
        if self.proc is not None:
            self.q.put(None)
            self.threads[1].join()
            self.proc.wait()
            if len(self.out) != self.sent:
                raise RuntimeError(f"the seam search stopped after {len(self.out)} of {self.sent} frames")
        return self.out


def main() -> None:
    lam = float(sys.argv[1])
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    prev = None
    while True:
        head = stdin.read(9)
        if len(head) < 9:
            return
        rows, cols, alone = struct.unpack("<iiB", head)
        if rows == 0:
            return
        cost = np.frombuffer(stdin.read(4 * rows * cols), np.float32).reshape(rows, cols)
        if not alone or prev is None:
            prev = seam_dp(cost, prev, lam)
        stdout.write(struct.pack("<i", len(prev)) + prev.astype(np.int32).tobytes())
        stdout.flush()


if __name__ == "__main__":
    main()
