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
