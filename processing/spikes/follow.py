#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Phase 0 spike: a players-only follow cam rendered at 1080p.

Run after phase0.py (it reads phase0/report.json for the audio offsets):

    uv run processing/spikes/follow.py footage/rec3 <segment> <start s> <seconds> [play|players] [features]

features: comma-separated, e.g. goalkick,goalframe (see play_targets).
Best so far on the night game: "play goalframe", and "play
goalkick,goalframe,unsure" (catches goal kicks and clearances, but wider).
Goal positions come from `pitch.py <recording> goals` when it has run.

1. Geometry: the panorama stitcher's cameras (focal length, principal
   point, rotation), scaled to the source resolution, map any direction
   to a pixel in each phone's frame. Estimated once; assumes a rigid mount.
2. Pass A: a low-resolution panorama of the field, background subtraction
   (the mount is fixed, so moving blobs on the turf are players), blob
   positions as yaw/pitch angles.
3. Path: per frame, aim at where play is and zoom to fit it; then smooth
   over seconds (offline, so it can use future frames) and cap the pan
   speed. "play" (default, needs pitch.py's pitch.json): players' feet on
   the pitch, tracked, aim at their speed-weighted centroid (play.py), zoom
   to the spread of the running players. "players": the median of all
   player blobs and their spread.
4. Pass B: a 1920x1080 virtual camera along the path, sampled straight from
   both phones' full-resolution frames. Each pixel comes from one phone:
   the two never capture the same instant (free-running 30 fps clocks,
   opposite rolling shutters), so blending them doubles moving players.
   The switch follows a per-frame seam through the overlap that avoids
   players and places the phones disagree, and sticks between frames.

The stitcher's cameras are refined at 1920 px on matches from frames
across the game (focal lengths, two radial terms, Secondary's rotation),
which roughly halves the alignment error at 4K.

Both passes warp and blend on the GPU (gpu.py: Apple's GPU locally, CUDA
in the cloud), in NV12 from the hardware decoder to the hardware encoder.

Writes <recording>/follow/: follow.mp4, path.json, background.jpg and a
detections overlay. Stays inside one segment for now.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import gpu  # noqa: E402
import phase0  # noqa: E402

FPS = 30
CAL_W = 960            # width the cameras are estimated at (phase0.PREVIEW_W)
REF_W = 1920           # width the camera model is refined and held at
SRC_W, SRC_H = 3840, 2160
OUT_W, OUT_H = 1920, 1080
PANO_DEG_PER_PX = 0.08  # pass A panorama resolution
WARMUP_S = 20           # background model warm-up before the excerpt
MIN_HFOV, MAX_HFOV = 30.0, 70.0
MARGIN_DEG = 4.0        # framing margin on each side of the players
SMOOTH_S = 1.5          # Gaussian sigma for the path
MAX_PAN_DEG_S = 20.0    # pan speed cap
SEAM_FEATHER_DEG = 0.3  # detail blend width either side of the seam
TONE_FEATHER_DEG = 6.0  # brightness/colour blend width across the seam
SEAM_LAMBDA = 0.003     # seam stickiness between frames, per column per row
PLAYER_COST = 3.0       # seam cost on a detected player, per pixel


# --------------------------------------------------------------- geometry

class Camera:
    """One phone: world direction (y down) <-> pixel, at a given width.

    Pinhole plus two radial distortion terms. Parameters are held at
    REF_W and scaled to the width asked for.
    """

    def __init__(self, f, cx, cy, k1, k2, R, width=REF_W):
        s = width / REF_W
        self.f, self.cx, self.cy = f * s, cx * s, cy * s
        self.k1, self.k2, self.R = k1, k2, R
        self.w, self.h = width, round(width * 9 / 16)
        self.ref = (f, cx, cy, k1, k2, R)
        # The radial model folds back beyond the image: directions far
        # outside would land inside it. Only trust radii up to the corners.
        corners = np.array([[0, 0], [self.w - 1, 0], [0, self.h - 1], [self.w - 1, self.h - 1]], float)
        c = self.unproject(corners) @ self.R
        self.r2_max = 1.05 * float(((c[:, 0] / c[:, 2]) ** 2 + (c[:, 1] / c[:, 2]) ** 2).max())

    @classmethod
    def from_stitcher(cls, cam):
        """From an OpenCV CameraParams estimated at CAL_W."""
        K = cam.K().astype(np.float64) * (REF_W / CAL_W)
        return cls(K[0, 0], K[0, 2], K[1, 2], 0.0, 0.0, cam.R.astype(np.float64))

    def at(self, width):
        return Camera(*self.ref, width=width)

    def project(self, d: np.ndarray):
        """World directions (..., 3) -> pixel maps and validity."""
        c = d @ self.R  # R^T d, row-vector form
        z = c[..., 2]
        ok = z > 1e-6
        zz = np.where(ok, z, 1)
        x, y = c[..., 0] / zz, c[..., 1] / zz
        r2 = x * x + y * y
        ok &= r2 < self.r2_max
        g = 1 + self.k1 * r2 + self.k2 * r2 * r2
        px = np.where(ok, self.f * x * g + self.cx, -1)
        py = np.where(ok, self.f * y * g + self.cy, -1)
        ok &= (px >= 0) & (px <= self.w - 1) & (py >= 0) & (py <= self.h - 1)
        return px.astype(np.float32), py.astype(np.float32), ok

    def unproject(self, px: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> world directions (N, 3)."""
        xd = (px[:, 0] - self.cx) / self.f
        yd = (px[:, 1] - self.cy) / self.f
        x, y = xd, yd
        for _ in range(8):  # invert the radial model by fixed-point iteration
            r2 = x * x + y * y
            g = 1 + self.k1 * r2 + self.k2 * r2 * r2
            x, y = xd / g, yd / g
        return np.stack([x, y, np.ones_like(x)], -1) @ self.R.T

    def edge_weight(self, x, y, ok):
        """Feather weight for the detection panorama: distance to the
        nearest image edge."""
        d = np.minimum(np.minimum(x, self.w - 1 - x), np.minimum(y, self.h - 1 - y))
        return np.where(ok, np.clip(d / (0.05 * self.w), 0, 1) ** 2, 0).astype(np.float32)


def match_pairs(pairs):
    """SIFT matches (secondary px, main px) pooled over frame pairs."""
    sift = cv2.SIFT_create(nfeatures=12000)
    out = []
    for m, s in pairs:
        km, dm = sift.detectAndCompute(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), None)
        ks, ds = sift.detectAndCompute(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), None)
        good = [a for a, b in (x for x in cv2.BFMatcher().knnMatch(ds, dm, k=2) if len(x) == 2)
                if a.distance < 0.7 * b.distance]
        out.append((np.float64([ks[g.queryIdx].pt for g in good]).reshape(-1, 2),
                    np.float64([km[g.trainIdx].pt for g in good]).reshape(-1, 2)))
    return out


def transfer_error(cm: Camera, cs: Camera, ps, pm):
    """Pixel distance in Main between each match and Secondary's point
    carried through the model."""
    x, y, ok = cm.project(cs.unproject(ps))
    e = np.hypot(x - pm[:, 0], y - pm[:, 1])
    return np.where(ok, e, np.inf)


def refine(cm: Camera, cs: Camera, matches, gate=40.0):
    """Least-squares refinement of both focal lengths, radial distortion
    and Secondary's rotation (Main's stays as the world reference), on
    matches the starting model already roughly agrees with."""
    ps = np.concatenate([m[0] for m in matches])
    pm = np.concatenate([m[1] for m in matches])
    keep = transfer_error(cm, cs, ps, pm) < gate
    ps, pm = ps[keep], pm[keep]
    fm, cxm, cym = cm.ref[:3]
    fs, cxs, cys = cs.ref[:3]
    R0 = cs.R

    def cams(v):
        rs = cv2.Rodrigues(v[6:9])[0] @ R0
        return (Camera(v[0], cxm, cym, v[2], v[4], cm.R),
                Camera(v[1], cxs, cys, v[3], v[5], rs))

    def resid(v):
        a, b = cams(v)
        xm, ym, _ = a.project(b.unproject(ps))
        xs, ys, _ = b.project(a.unproject(pm))
        prior = 20 * np.array([v[2], v[3], v[4], v[5]])  # keep distortion modest
        return np.concatenate([xm - pm[:, 0], ym - pm[:, 1], xs - ps[:, 0], ys - ps[:, 1], prior])

    v0 = np.array([fm, fs, 0, 0, 0, 0, 0, 0, 0], float)
    sol = least_squares(resid, v0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    return cams(sol.x), int(keep.sum())


def direction(yaw, pitch):
    """World direction for yaw (right +) and pitch (down +), radians."""
    return np.stack([np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                     np.cos(yaw) * np.cos(pitch)], -1)


def view_directions(yaw, pitch, hfov, w, h):
    """Directions through each pixel of a pinhole virtual camera."""
    f = (w / 2) / np.tan(hfov / 2)
    u, v = np.meshgrid((np.arange(w) - w / 2 + 0.5) / f, (np.arange(h) - h / 2 + 0.5) / f)
    z = direction(yaw, pitch)
    x = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    y = np.cross(z, x)
    return u[..., None] * x + v[..., None] * y + z


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


class Stopwatch:
    """Accumulates wall time per stage: each lap charges the time since the
    previous lap to `name`."""

    def __init__(self):
        self.t = time.perf_counter()
        self.total: dict[str, float] = defaultdict(float)

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.total[name] += now - self.t
        self.t = now

    def report(self, frames: dict[str, int]) -> str:
        lines = []
        for name, sec in self.total.items():
            n = frames.get(name.split(":")[0], 0)
            per = f"{1000 * sec / n:7.1f} ms/frame" if n and ":" in name else ""
            lines.append(f"  {name:22s} {sec:7.1f} s {per}")
        return "\n".join(lines)


# --------------------------------------------------------------- media io

def frame_stream(path: Path, start: float, seconds: float, width: int, fps: float = FPS):
    """Yields raw NV12 frames (flat uint8) at constant `fps`, decoded in
    hardware, as the phone recorded them (no rotation applied)."""
    height = round(width * 9 / 16)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-noautorotate", "-hwaccel", "videotoolbox", "-ss", f"{start:.4f}",
         "-i", str(path), "-t", f"{seconds:.3f}", "-vf", f"fps={fps},scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * 3 // 2
    while True:
        buf = bytearray(size)  # writable, so torch can share it without a copy
        if proc.stdout.readinto(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8)
    proc.wait()


def prefetch(gen, depth: int = 6):
    """Runs a frame generator on its own thread, `depth` frames ahead, so
    reading the decoder's pipe overlaps with GPU work."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def run():
        for item in gen:
            q.put(item)
        q.put(done)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not done:
        yield item


class NV12Encoder:
    """Hardware H.264 from NV12 frames."""

    def __init__(self, path: Path, w: int, h: int, fps: int = FPS, bitrate: str = "16M"):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "nv12",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "h264_videotoolbox", "-b:v", bitrate,
             "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
             "-color_trc", "bt709", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def setup_cameras(rec: Path, report: dict, devs: dict, flip: list[bool]) -> tuple[Camera, Camera]:
    """Main's and Secondary's cameras, for upright frames: the panorama
    stitcher on segment 0, refined at REF_W on matches from frames across
    the game. Cached in phase0/cameras.json (delete it to recompute)."""
    cache = rec / "phase0" / "cameras.json"
    if cache.exists():
        j = json.loads(cache.read_text())
        return tuple(Camera(*c[:5], np.array(c[5])) for c in (j["main"], j["secondary"]))
    s0 = report["segments"][0]
    m0 = devs["main"]["segments"][0]
    sec0 = devs["secondary"]["segments"][0]
    dur = min(m0["durationMs"], sec0["durationMs"]) / 1000 - abs(s0["offsetS"]) - 1
    t0 = max(0.0, -s0["offsetS"]) + 0.5
    pairs = [(phase0.frames_at(rec / m0["objectKey"], t, 1, FPS, CAL_W, flip[0])[0],
              phase0.frames_at(rec / sec0["objectKey"], t + s0["offsetS"], 1, FPS, CAL_W, flip[1])[0])
             for t in np.linspace(t0, t0 + dur - 0.5, phase0.CALIB_FRAMES)]
    st, cal = phase0.calibrate(pairs)
    if st is None:
        sys.exit("calibration failed")
    cams = st.cameras()  # order [secondary, main]
    print(f"stitcher: {cal['angleBetweenDeg']} deg apart, focal {cal['focalPx']} @ {CAL_W}")

    cm, cs = Camera.from_stitcher(cams[1]), Camera.from_stitcher(cams[0])
    nseg = len(report["segments"])
    fit_pairs = []
    for sg in sorted({round(x) for x in np.linspace(0, nseg - 1, 6)}):
        o = report["segments"][sg]["offsetS"]
        for t in (40.0, 140.0):
            fit_pairs.append((
                phase0.frames_at(rec / devs["main"]["segments"][sg]["objectKey"], t, 1, FPS, REF_W, flip[0])[0],
                phase0.frames_at(rec / devs["secondary"]["segments"][sg]["objectKey"], t + o, 1, FPS, REF_W,
                                 flip[1])[0]))
    matches = match_pairs(fit_pairs)
    ps = np.concatenate([x[0] for x in matches]); pm = np.concatenate([x[1] for x in matches])
    before = transfer_error(cm, cs, ps, pm)
    (cm, cs), n = refine(cm, cs, matches)
    after = transfer_error(cm, cs, ps, pm)
    near = before < 40
    print(f"refined on {n} matches: median error {2 * np.median(before[near]):.1f} ->"
          f" {2 * np.median(after[near]):.1f} px at 4K; k1 {cm.k1:.4f}/{cs.k1:.4f}")
    cache.write_text(json.dumps({k: [*c.ref[:5], c.R.tolist()] for k, c in (("main", cm), ("secondary", cs))}))
    return cm, cs


AIM_UP = 0.1             # the aim point (on the grass) sits this fraction of the view below centre
LULL_RUNNERS = 1         # at most this many running players: play has stopped
LULL_MIN_S = 1.0         # ... for at least this long
HOLD_BACK_S = 1.0        # in a lull, hold the aim from this long before it began
STILL_SPEED = 1.0        # m/s: a restart taker stands about still
TAKER_ALONE_M = 12.0     # ... this far from everyone else (goal kicks, corners)
THROW_ALONE_M = 4.0      # throw-ins happen closer to other players
RESTART_MAX_HFOV = 95.0  # a restart may need a wider view: taker and receivers
LOITER_CELL_M = 4.0      # occupancy grid for finding where non-players stand
LOITER_EDGE_M = 4.0      # ... within this far of a touchline or goal line
LOITER_FRAC = 0.10       # ... occupied this often: subs, bench, spectators


HOT_WINDOW_S = 0.5      # pool players this long either side of a frame
HOT_BANDWIDTH_M = 10.0  # how far one running player's pull reaches
HOT_SWITCH = 1.5        # another spot must be this much stronger to move to it
HOT_PERSIST_S = 0.7     # ... for this long
TRACK_SMOOTH_S = 0.5    # position smoothing before velocity (far-side jitter)


def hotspots(pos, vel, fps):
    """Per frame, where running is most concentrated: players pooled over
    +-HOT_WINDOW_S, weighted by speed, smoothed with a Gaussian of
    HOT_BANDWIDTH_M; the peak is found by mean shift. The camera stays with
    the previous frame's peak unless another is HOT_SWITCH times stronger,
    for HOT_PERSIST_S, so a lone runner elsewhere can't make it hop."""
    import play

    n = len(pos)
    k = int(HOT_WINDOW_S * fps)
    h2 = 2 * HOT_BANDWIDTH_M ** 2
    spots = np.full((n, 2), np.nan)
    cur, challenger, streak = None, None, 0
    persist = int(HOT_PERSIST_S * fps)
    for i in range(n):
        xs = [pos[j] for j in range(max(0, i - k), min(n, i + k + 1)) if len(pos[j])]
        vs = [vel[j] for j in range(max(0, i - k), min(n, i + k + 1)) if len(vel[j])]
        if not xs:
            spots[i] = cur if cur is not None else np.nan
            continue
        x = np.vstack(xs)
        # Linear in speed: several moving players outweigh one sprinter.
        w = np.clip(np.hypot(*np.vstack(vs).T), 0.5, 8.0)

        def density(p):
            return float((w * np.exp(-((x - p) ** 2).sum(1) / h2)).sum())

        def climb(p):
            for _ in range(20):
                g = w * np.exp(-((x - p) ** 2).sum(1) / h2)
                q = (g[:, None] * x).sum(0) / max(g.sum(), 1e-9)
                if np.hypot(*(q - p)) < 0.05:
                    return q
                p = q
            return p

        # Candidate peaks: from where we are, and from the fastest runners.
        fast = x[np.argsort(-w)[:5]]
        cands = [climb(f) for f in fast]
        best = max(cands, key=density)
        if cur is None:
            cur = best
        else:
            cur = climb(cur)
            if density(best) > HOT_SWITCH * density(cur) and np.hypot(*(best - cur)) > HOT_BANDWIDTH_M:
                # A different, stronger spot: move only if it stays stronger.
                same = challenger is not None and np.hypot(*(best - challenger)) < HOT_BANDWIDTH_M
                streak = streak + 1 if same else 1
                challenger = best
                if streak >= persist:
                    cur, challenger, streak = best, None, 0
            else:
                challenger, streak = None, 0
        spots[i] = cur
    return spots


def loiter_cells(frames_xz, pose):
    """Grid cells near the pitch's edge that someone occupies in at least
    LOITER_FRAC of the frames: where subs, the bench and spectators stand,
    not where play goes. Returns a set of (ix, iz), dilated by one cell."""
    hl, hw = pose.L / 2, pose.W / 2
    counts: dict[tuple[int, int], int] = {}
    for xz in frames_xz:
        cells = {(int(np.floor(x / LOITER_CELL_M)), int(np.floor(z / LOITER_CELL_M))) for x, z in xz
                 if abs(x) >= hl - LOITER_EDGE_M or abs(z) >= hw - LOITER_EDGE_M}
        for c in cells:
            counts[c] = counts.get(c, 0) + 1
    busy = {c for c, n in counts.items() if n >= LOITER_FRAC * len(frames_xz)}
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print("loiter: busiest edge cells", [((a * LOITER_CELL_M, b * LOITER_CELL_M), round(n / len(frames_xz), 2))
                                          for (a, b), n in top])
    return {(a + da, b + db) for a, b in busy for da in (-1, 0, 1) for db in (-1, 0, 1)}


def restart_taker(pos, vel, pose):
    """During a lull: a still player alone in a goal area, near a corner, or
    on a touchline. Returns (kind, position) or None."""
    import pitch as P

    if len(pos) < 3:
        return None
    speed = np.hypot(*vel.T)
    hl, hw = pose.L / 2, pose.W / 2
    goal_x = hl - pose.k * P.GOAL_DEPTH - 3.0
    goal_z = pose.k * P.GOAL_HALF + 3.0
    best = None
    for j, (x, z) in enumerate(pos):
        if speed[j] > STILL_SPEED:
            continue
        others = np.delete(pos, j, axis=0)
        alone = np.hypot(*(others - pos[j]).T).min()
        if abs(x) >= goal_x and abs(z) <= goal_z and alone >= TAKER_ALONE_M:
            kind = "goal kick"
        elif abs(x) >= hl - 4 and abs(z) >= hw - 4 and alone >= TAKER_ALONE_M:
            kind = "corner"
        elif abs(z) >= hw - 1.5 and alone >= THROW_ALONE_M:
            kind = "throw-in"
        else:
            continue
        if best is None or alone > best[2]:
            best = (kind, pos[j], alone)
    return None if best is None else (best[0], best[1])


GK_RECENT_S = 15.0       # goal kick: play was near that goal this recently
GK_NEAR_GOAL_M = 30.0    # ... meaning the hotspot within this of the goal line
GK_ALONE_M = 15.0        # the taker is this far from everyone
GK_GROUP_M = 20.0        # the others' median is at least this far away
GK_STILL = 2.0           # m/s: the taker stands about still
GK_ZONE_EXTRA_M = 5.0    # zone: the penalty area plus this (keepers place the ball, walk back)
GK_WINDOW_S = 1.0        # the pattern shows in GK_MIN_FRAC of the frames within +-this
GK_MIN_FRAC = 0.3        # (a keeper 50 m away is only tracked on and off)
GK_FOLLOW_S = 2.0        # keep the framing this long after play resumes (the kick's flight)
GK_RESUME_RUNNERS = 3    # play has resumed when this many run...
GK_RESUME_S = 0.5        # ... for this long
GK_MAX_S = 12.0          # a restart framing lasts at most this long
GOAL_ZONE_M = 20.0       # goalframe: players within this of a goal line
GOAL_MIN_PLAYERS = 4     # ... at least this many
GOAL_MAX_HFOV = 75.0     # the union must fit in this
GOAL_RECENT_S = 10.0     # if it doesn't fit, prefer the goal end if play was there this recently
UNSURE_HFOV = ((1, 55.0), (2, 42.0))  # unsure: with at most N runners near the hotspot, at least this wide
LEAD_S = 1.0             # lead: aim where the hotspot will be this much later (processing is offline)


def play_targets(rec: Path, dets, extent, start: float = 0.0, features: tuple = ()):
    """Per-frame aim (yaw, pitch) and width from player motion: blob feet
    placed on the pitch, tracked (play.py), aim at the hotspot of running
    (see hotspots), width from the players around it.

    When play stops (a lull: at most LULL_RUNNERS running for LULL_MIN_S),
    hold the aim from just before it rather than drifting to where the
    players wait; and if a restart taker stands alone in a goal area, at a
    corner or on a touchline, frame the taker and the waiting group.

    Optional `features`: "goalkick" (a lone still player in a penalty area
    soon after play was at that end: frame the taker and the group),
    "goalframe" (several players near a goal while the hotspot is
    elsewhere: frame both, and the goal mouth), "unsure" (few runners near
    the hotspot: don't zoom in tight, the ball may be elsewhere) and "lead"
    (aim where the hotspot will be LEAD_S later)."""
    import pitch as P
    import play

    pose = P.load_pose(rec)
    y0, y1, p0, p1 = extent
    # Aim only from people inside the lines, and not where people stand
    # around all game long (subs, bench, spectators by the touchline).
    hl, hw = pose.L / 2, pose.W / 2
    inside = []
    for fd in dets:
        xz = []
        for yw, _, _, feet in fd:
            x, z = pose.to_pitch(np.degrees(yw), np.degrees(feet))
            if not np.isnan(x) and abs(x) <= hl and abs(z) <= hw:
                xz.append((float(x), float(z)))
        inside.append(xz)
    loiter = loiter_cells(inside, pose)
    frames, dropped = [], 0
    for xz in inside:
        keep = [(x, z) for x, z in xz
                if (int(np.floor(x / LOITER_CELL_M)), int(np.floor(z / LOITER_CELL_M))) not in loiter]
        dropped += len(xz) - len(keep)
        frames.append({"detections": [{"cls": "person", "onPitch": True, "xz": [x, z]} for x, z in keep]})
    print(f"play: {sum(map(len, inside))} detections inside the lines; {len(loiter)} loiter cells"
          f" drop {dropped} ({100 * dropped / max(1, sum(map(len, inside))):.0f}%)")
    tracks = play.track_players(frames, FPS, TRACK_SMOOTH_S)
    pos, vel = play.per_frame(tracks, len(frames))
    T = len(dets)

    def target(aim_xz, who):
        """Aim at a pitch point; width to cover `who` (pitch points) around it."""
        aim = np.radians(pose.angles(np.asarray(aim_xz)[None])[0])
        yw = np.radians(pose.angles(who)[:, 0])
        lo_, hi_ = np.percentile(yw, [15, 85]) if len(yw) > 2 else (yw.min(), yw.max())
        spread = max(hi_ - aim[0], aim[0] - lo_) * 2 + 2 * np.radians(MARGIN_DEG)
        return aim, spread

    def finish(aim, spread, max_hfov=MAX_HFOV):
        hfov = np.clip(spread, np.radians(MIN_HFOV), np.radians(max_hfov))
        vfov = 2 * np.arctan(np.tan(hfov / 2) * OUT_H / OUT_W)
        return aim[0], aim[1] - AIM_UP * vfov, hfov

    tyaw, tpitch, thfov = np.zeros(T), np.zeros(T), np.zeros(T)
    running = np.array([(np.hypot(*v.T) > play.RUN_SPEED).sum() if len(v) else 0 for v in vel])
    spots = hotspots(pos, vel, FPS)
    lead = int(LEAD_S * FPS) if "lead" in features else 0
    last = (0.5 * (y0 + y1), 0.5 * (p0 + p1), np.radians(MAX_HFOV))
    near_runners = np.zeros(T)
    for i in range(T):
        j = min(T - 1, i + lead)  # with "lead", the hotspot and players LEAD_S ahead
        if len(pos[j]) >= 2 and not np.isnan(spots[j, 0]):
            # Frame the players around the hotspot (running ones if enough).
            near = np.hypot(*(pos[j] - spots[j]).T) < 2 * HOT_BANDWIDTH_M
            run = near & (np.hypot(*vel[j].T) > play.RUN_SPEED)
            near_runners[i] = run.sum()
            who = pos[j][run] if run.sum() >= 2 else pos[j][near] if near.sum() >= 2 else pos[j]
            last = finish(*target(spots[j], who))
        tyaw[i], tpitch[i], thfov[i] = last
    if "unsure" in features:
        # Few runners near the hotspot: the evidence is thin, so stay wide.
        n = np.convolve(near_runners, np.ones(int(FPS)) / int(FPS), mode="same")
        widened = 0
        for i in range(T):
            for most, wide in UNSURE_HFOV:
                if n[i] <= most and thfov[i] < np.radians(wide):
                    vfov_old = 2 * np.arctan(np.tan(thfov[i] / 2) * OUT_H / OUT_W)
                    thfov[i] = np.radians(wide)
                    vfov_new = 2 * np.arctan(np.tan(thfov[i] / 2) * OUT_H / OUT_W)
                    tpitch[i] -= AIM_UP * (vfov_new - vfov_old)  # keep the aim point where it was in the view
                    widened += 1
                    break
        print(f"unsure: {widened} frames widened (few runners near the hotspot)")

    # Lulls: few runners for long enough (median-filtered so a lone jogger
    # doesn't break one up).
    lull = median_filter((running <= LULL_RUNNERS).astype(np.uint8), int(FPS) | 1) > 0
    events, i = [], 0
    while i < T:
        if not lull[i]:
            i += 1
            continue
        j = i
        while j < T and lull[j]:
            j += 1
        if (j - i) / FPS >= LULL_MIN_S:
            k = max(0, i - int(HOLD_BACK_S * FPS))
            held = (tyaw[k], tpitch[k], thfov[k])
            takers = 0
            for f in range(i, j):
                tk = restart_taker(pos[f], vel[f], pose)
                if tk is not None:
                    kind, at = tk
                    others = pos[f][np.hypot(*(pos[f] - at).T) > 1.0]
                    group = np.median(others, axis=0)
                    # Aim between taker and the group; cover both.
                    aim, spread = target((at + group) / 2, np.vstack([at[None], others]))
                    tyaw[f], tpitch[f], thfov[f] = finish(aim, spread, RESTART_MAX_HFOV)
                    takers += 1
                    last_kind = kind
                else:
                    tyaw[f], tpitch[f], thfov[f] = held
            events.append({"startS": round(start + i / FPS, 2), "endS": round(start + j / FPS, 2),
                           "restartFrames": takers, "restart": last_kind if takers else None})
        i = j
    hl, hw = pose.L / 2, pose.W / 2
    # The on-pitch goals from pitch.py's once-per-game detection, else the
    # pitch model's goal-line centres.
    detected = P.load_goals(rec)
    goal_at = {g["side"]: np.array([g["x"], g["z"]]) for g in detected} if detected else {}
    for side in (-1, 1):
        goal_at.setdefault(side, np.array([side * hl, 0.0]))
    print(f"goals: {'detected' if detected else 'from the pitch model'} at "
          f"{[tuple(np.round(goal_at[s_], 1)) for s_ in (-1, 1)]}")

    def span(points, max_hfov):
        """Aim and width to show all `points` (pitch xz), or None if they
        don't fit in max_hfov."""
        ang = np.radians(pose.angles(np.asarray(points)))
        lo_, hi_ = ang[:, 0].min(), ang[:, 0].max()
        spread = hi_ - lo_ + 2 * np.radians(MARGIN_DEG)
        if spread > np.radians(max_hfov):
            return None
        centre = np.radians(pose.angles(np.mean(points, axis=0)[None])[0])
        return finish(np.array([(lo_ + hi_) / 2, centre[1]]), spread, max_hfov)

    def near_goal_recently(i, side, secs, metres):
        a = max(0, i - int(secs * FPS))
        sp = spots[a:i + 1]
        sp = sp[~np.isnan(sp[:, 0])]
        return bool(np.any(np.hypot(*(sp - goal_at[side]).T) <= metres)) if len(sp) else False

    if "goalkick" in features:
        pen_depth = pose.k * P.PEN_DEPTH + GK_ZONE_EXTRA_M
        pen_z = pose.k * P.PEN_HALF + GK_ZONE_EXTRA_M
        flag = np.zeros(T, bool)
        frame_for = {}
        for i in range(T):
            if len(pos[i]) < 4:
                continue
            speed = np.hypot(*vel[i].T)
            for j, (x, z) in enumerate(pos[i]):
                side = 1 if x > 0 else -1
                gx, gz = goal_at[side]
                if side * (gx - x) > pen_depth or abs(z - gz) > pen_z or speed[j] > GK_STILL:
                    continue
                others = np.delete(pos[i], j, axis=0)
                d = np.hypot(*(others - pos[i][j]).T)
                if d.min() < GK_ALONE_M or np.median(d) < GK_GROUP_M:
                    continue
                if not near_goal_recently(i, side, GK_RECENT_S, GK_NEAR_GOAL_M):
                    continue
                fr = span(np.vstack([pos[i][j][None], others]), RESTART_MAX_HFOV)
                if fr is None:  # too wide: taker and the group's middle
                    aim, spread = target((pos[i][j] + np.median(others, axis=0)) / 2,
                                         np.vstack([pos[i][j][None], others]))
                    fr = finish(aim, spread, RESTART_MAX_HFOV)
                flag[i], frame_for[i] = True, fr
                break
        w = int(GK_WINDOW_S * FPS)
        frac = np.convolve(flag.astype(float), np.ones(2 * w + 1) / (2 * w + 1), mode="same")
        keep = frac >= GK_MIN_FRAC
        # A restart is a state: entered when the pattern shows, held (with the
        # latest framing) through the taker's run-up and kick, left when play
        # has clearly resumed or after GK_MAX_S, plus GK_FOLLOW_S.
        resumed = np.convolve((running >= GK_RESUME_RUNNERS).astype(float),
                              np.ones(int(GK_RESUME_S * FPS)) / int(GK_RESUME_S * FPS), mode="same") >= 0.99
        n_kicks, held, state, since, follow, last_fr = 0, 0, False, 0, 0, None
        for i in range(T):
            if keep[i] and i in frame_for:
                last_fr = frame_for[i]
                if not state:
                    n_kicks += 1
                    state, since = True, i
            if state and ((resumed[i] and i - since > FPS) or i - since > GK_MAX_S * FPS):
                state, follow = False, int(GK_FOLLOW_S * FPS)
            if state or follow > 0:
                tyaw[i], tpitch[i], thfov[i] = last_fr
                held += 1
                if not state:
                    follow -= 1
        print(f"goalkick: {n_kicks} restart(s), {held} frames framed on taker and group")

    if "goalframe" in features:
        flag = np.zeros(T, bool)
        frame_for = {}
        for i in range(T):
            if len(pos[i]) < 4 or np.isnan(spots[i, 0]):
                continue
            for side in (-1, 1):
                mouth = goal_at[side]
                if side * (mouth[0] - spots[i, 0]) <= GOAL_ZONE_M + 5:
                    continue  # the hotspot is already at this end
                g = pos[i][side * (mouth[0] - pos[i][:, 0]) <= GOAL_ZONE_M]
                if len(g) < GOAL_MIN_PLAYERS:
                    continue
                cl = pos[i][np.hypot(*(pos[i] - spots[i]).T) < 2 * HOT_BANDWIDTH_M]
                pts = np.vstack([cl, g, mouth[None]])
                fr = span(pts, GOAL_MAX_HFOV)
                if fr is None and near_goal_recently(i, side, GOAL_RECENT_S, GOAL_ZONE_M + 5):
                    fr = span(np.vstack([g, mouth[None]]), GOAL_MAX_HFOV)
                if fr is not None:
                    flag[i], frame_for[i] = True, fr
                    break
        keep = median_filter(flag.astype(np.uint8), int(FPS) | 1) > 0
        n = 0
        for i in range(T):
            if keep[i] and i in frame_for:
                tyaw[i], tpitch[i], thfov[i] = frame_for[i]
                n += 1
        print(f"goalframe: {n} frames widened to include a goal")

    print(f"play: {len(tracks)} player tracks; running per frame median {int(np.median(running))}")
    for e in events:
        print(f"  lull {e['startS']:.1f}-{e['endS']:.1f} s"
              + (f": {e['restart']} in {e['restartFrames']} frames" if e["restart"] else ": held"))
    (rec / "follow" / "lulls.json").write_text(json.dumps(
        {"events": events, "running": running.tolist(), "startS": start, "fps": FPS}))
    # Per-frame players and aim, for diagnosing what the camera followed.
    (rec / "follow" / "play_debug.json").write_text(json.dumps({"startS": start, "fps": FPS, "frames": [
        {"pos": np.round(pos[i], 1).tolist(), "vel": np.round(vel[i], 1).tolist(),
         "aimYawDeg": round(float(np.degrees(tyaw[i])), 2), "hfovDeg": round(float(np.degrees(thfov[i])), 1)}
        for i in range(T)]}))
    return tyaw, tpitch, thfov


# ------------------------------------------------------------------- main

def main(rec: Path, seg: int, start: float, seconds: float, aim: str = "play", features: tuple = ()) -> None:
    sw = Stopwatch()
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    out = rec / "follow"
    out.mkdir(exist_ok=True)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    files = [rec / devs["main"]["segments"][seg]["objectKey"],
             rec / devs["secondary"]["segments"][seg]["objectKey"]]
    offsets = [0.0, report["segments"][seg]["offsetS"]]

    cm, cs = setup_cameras(rec, report, devs, flip)
    sw.lap("setup cameras")

    # Pass A panorama: equirectangular over everything either phone sees.
    lo = [cm.at(CAL_W), cs.at(CAL_W)]
    edge = []
    for c in lo:
        xs = np.linspace(0, c.w - 1, 40)
        ys = np.linspace(0, c.h - 1, 24)
        px = np.r_[np.c_[xs, 0 * xs], np.c_[xs, 0 * xs + c.h - 1], np.c_[0 * ys, ys], np.c_[0 * ys + c.w - 1, ys]]
        edge.append(c.unproject(px))
    rays = np.concatenate(edge)
    yaws = np.arctan2(rays[:, 0], rays[:, 2])
    pitches = np.arctan2(rays[:, 1], np.hypot(rays[:, 0], rays[:, 2]))
    step = np.radians(PANO_DEG_PER_PX)
    y0, y1, p0, p1 = yaws.min(), yaws.max(), pitches.min(), pitches.max()
    PW, PH = int((y1 - y0) / step), int((p1 - p0) / step)
    YY, PP = np.meshgrid(y0 + np.arange(PW) * step, p0 + np.arange(PH) * step)
    pano_dirs = direction(YY, PP)
    pmaps, pweights = [], []
    for c in lo:
        x, y, ok = c.project(pano_dirs)
        pmaps.append((x, y))
        pweights.append(c.edge_weight(x, y, ok))
    print(f"pass A panorama {PW}x{PH}, yaw {np.degrees(y0):.0f}..{np.degrees(y1):.0f} deg")
    # Columns where both phones see something; the seam runs through them.
    both = (pweights[0] > 0) & (pweights[1] > 0)
    ocols = np.flatnonzero(both.any(0))
    c0, c1 = int(ocols.min()), int(ocols.max()) + 1
    main_right = YY[0][pweights[0].any(0)].mean() > YY[0][pweights[1].any(0)].mean()
    print(f"overlap yaw {np.degrees(y0 + c0 * step):.1f}..{np.degrees(y0 + c1 * step):.1f} deg,"
          f" Main on the {'right' if main_right else 'left'}")

    # Exposure: match Secondary to Main in the overlap, mid segment 0.
    s0 = report["segments"][0]
    t_mid = 0.5 * min(devs["main"]["segments"][0]["durationMs"], devs["secondary"]["segments"][0]["durationMs"]) / 1000
    a = phase0.frames_at(rec / devs["main"]["segments"][0]["objectKey"], t_mid, 1, FPS, CAL_W, flip[0])[0]
    b = phase0.frames_at(rec / devs["secondary"]["segments"][0]["objectKey"], t_mid + s0["offsetS"], 1, FPS,
                         CAL_W, flip[1])[0]
    wa, wb = pweights
    ov = (wa > 0) & (wb > 0)
    ia = cv2.remap(a, *pmaps[0], cv2.INTER_LINEAR)[ov].astype(np.float32)
    ib = cv2.remap(b, *pmaps[1], cv2.INTER_LINEAR)[ov].astype(np.float32)
    gain_s = (ia.mean(0) + 1) / (ib.mean(0) + 1)
    gains = [np.float32(1.0), np.float32(gain_s.mean())]
    print(f"secondary gain {gains[1]:.3f}")

    dev = gpu.device()
    lo_h = round(CAL_W * 9 / 16)
    warper = gpu.PanoWarper(pmaps, pweights, flip, CAL_W, lo_h, gains, c0, c1, dev)
    print(f"GPU: {dev}")
    sw.lap("setup refine+maps")
    # Pass A: detect moving blobs on the turf.
    warm = min(WARMUP_S, start)
    a_start = start - warm
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=True)
    streams = [prefetch(frame_stream(f, a_start + o, warm + seconds, CAL_W))
               for f, o in zip(files, offsets)]
    dets: list[list[tuple[float, float, float]]] = []
    field = None
    n_warm = int(warm * FPS)
    kernel = np.ones((3, 3), np.uint8)
    sample = None
    seams, seam_h = [], None
    nA = 0
    for i, (fa, fb) in enumerate(zip(*streams)):
        nA += 1
        sw.lap("A: decode")
        pano, cost = warper.warp([gpu.upload_nv12(f, CAL_W, lo_h, dev) for f in (fa, fb)])
        sw.lap("A: warp+blend")
        # After the warm-up, learn the background slowly (~a minute), so a
        # player standing still (a keeper waiting to take a goal kick) stays
        # foreground instead of fading in after ~10 s.
        fg = mog.apply(pano, learningRate=-1 if i < n_warm else 1 / (60 * FPS))
        sw.lap("A: background")
        # Seam: avoid players and places where the two phones disagree.
        # Searched on row pairs; it moves slowly, so half the rows is plenty.
        players = cv2.dilate((fg[:, c0:c1] == 255).astype(np.uint8), np.ones((7, 7), np.uint8))
        cost += PLAYER_COST * players
        cost[~both[:, c0:c1]] = 50.0
        half = cost[: PH // 2 * 2].reshape(PH // 2, 2, -1).mean(1)
        seam_h = seam_dp(half, seam_h, 2 * SEAM_LAMBDA)
        seam = np.repeat(seam_h, 2)
        seam = np.r_[seam, seam[-1:]][:PH]
        sw.lap("A: seam")
        if i == n_warm:
            bg = mog.getBackgroundImage()
            cv2.imwrite(str(out / "background.jpg"), bg)
            hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
            turf = ((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40)).astype(np.uint8) * 255
            turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(turf)
            field = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        if i < n_warm:
            continue
        fg = cv2.morphologyEx((fg == 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
        frame_dets = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            feet_x, feet_y = x + w // 2, min(y + h, PH - 1)
            if area < 12 or h < 5 or w > 3 * h or not field[feet_y, feet_x]:
                continue
            frame_dets.append((float(YY[0, int(cent[k][0])]), float(PP[int(cent[k][1]), 0]), float(area),
                               float(PP[feet_y, 0])))
        dets.append(frame_dets)
        sw.lap("A: blobs")
        seams.append(y0 + (c0 + seam) * step)
        if len(dets) == int(seconds * FPS / 2):
            sample = pano.copy()
            for yw, pt, *_ in frame_dets:
                cv2.circle(sample, (int((yw - y0) / step), int((pt - p0) / step)), 6, (0, 0, 255), 2)
            cv2.polylines(sample, [np.c_[c0 + seam, np.arange(PH)].astype(np.int32)], False, (255, 255, 0), 1)
    if sample is not None:
        cv2.imwrite(str(out / "detections.jpg"), sample)
    sy = np.degrees(np.array(seams))
    print(f"seam yaw at the horizon row: {np.percentile(sy[:, PH // 2], [5, 50, 95]).round(1)} deg")
    counts = [len(d) for d in dets]
    print(f"pass A: {len(dets)} frames, players detected per frame median {int(np.median(counts))}"
          f" (range {min(counts)}-{max(counts)})")

    # Path: where play is and how wide, then smooth.
    T = len(dets)
    if aim == "play":
        tyaw, tpitch, thfov = play_targets(rec, dets, (y0, y1, p0, p1), start, features)
    else:
        tyaw, tpitch, thfov = np.zeros(T), np.zeros(T), np.zeros(T)
        last = (0.5 * (y0 + y1), 0.5 * (p0 + p1), np.radians(MAX_HFOV))
        for i, fd in enumerate(dets):
            if len(fd) >= 2:
                yw = np.array([d[0] for d in fd])
                pt = np.array([d[1] for d in fd])
                lo_, hi_ = np.percentile(yw, [15, 85])
                spread = hi_ - lo_ + 2 * np.radians(MARGIN_DEG)
                last = (np.median(yw), np.median(pt),
                        np.clip(spread, np.radians(MIN_HFOV), np.radians(MAX_HFOV)))
            tyaw[i], tpitch[i], thfov[i] = last
    win = int(FPS) | 1
    sig = SMOOTH_S * FPS
    yaw = gaussian_filter1d(median_filter(tyaw, win, mode="nearest"), sig, mode="nearest")
    pitch = gaussian_filter1d(median_filter(tpitch, win, mode="nearest"), 2 * sig, mode="nearest")
    hfov = gaussian_filter1d(median_filter(thfov, 3 * win, mode="nearest"), 2 * sig, mode="nearest")
    vmax = np.radians(MAX_PAN_DEG_S) / FPS
    for i in range(1, T):
        yaw[i] = yaw[i - 1] + np.clip(yaw[i] - yaw[i - 1], -vmax, vmax)
    # Keep the view inside what the phones saw vertically.
    vfov = 2 * np.arctan(np.tan(hfov / 2) * OUT_H / OUT_W)
    pitch = np.clip(pitch, p0 + vfov / 2, p1 - vfov / 2)
    tag = "_".join([aim, *features])
    (out / f"path_{tag}.json").write_text(json.dumps({
        "segment": seg, "startS": start, "fps": FPS, "aim": aim,
        "yawDeg": np.round(np.degrees(yaw), 2).tolist(),
        "pitchDeg": np.round(np.degrees(pitch), 2).tolist(),
        "hfovDeg": np.round(np.degrees(hfov), 2).tolist()}))
    print(f"path: yaw {np.degrees(yaw.min()):.0f}..{np.degrees(yaw.max()):.0f} deg,"
          f" hfov {np.degrees(hfov.min()):.0f}..{np.degrees(hfov.max()):.0f} deg")

    sw.lap("path")
    if os.environ.get("OSB_NO_RENDER"):
        print("OSB_NO_RENDER set: stopping before pass B")
        return
    # Pass B: render the virtual camera from full-resolution frames. Each
    # pixel comes from one phone, switching at that frame's seam.
    renderer = gpu.ViewRenderer([cm, cs], flip, SRC_W, OUT_W, OUT_H, gains, main_right,
                                p0, step, np.radians(SEAM_FEATHER_DEG),
                                np.radians(TONE_FEATHER_DEG), dev)
    seams_t = torch.tensor(np.array(seams), dtype=torch.float32, device=dev)
    streams = [prefetch(frame_stream(f, start + o, seconds, SRC_W)) for f, o in zip(files, offsets)]
    enc = NV12Encoder(out / f"follow_{tag}.mp4", OUT_W, OUT_H)
    for i, (fa, fb) in enumerate(zip(*streams)):
        sw.lap("B: decode")
        if i >= T:
            break
        frames = [gpu.upload_nv12(f, SRC_W, SRC_H, dev) for f in (fa, fb)]
        frame = renderer.render(frames, yaw[i], pitch[i], hfov[i], seams_t[i])
        sw.lap("B: render")
        enc.write(frame)
        sw.lap("B: encode")
        if i % 300 == 0:
            print(f"pass B: frame {i}/{T}", flush=True)
    enc.close()
    sw.lap("B: encode")
    print(f"wrote {out / f'follow_{tag}.mp4'}")
    print("time by stage:\n" + sw.report({"A": nA, "B": T}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
         sys.argv[5] if len(sys.argv) > 5 else "play",
         tuple(f for f in sys.argv[6].split(",") if f) if len(sys.argv) > 6 else ())
