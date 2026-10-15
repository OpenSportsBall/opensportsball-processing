#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Phase 0 spike: where is play, from how the players move?

Run after detect.py, pitch.py and ball.py:

    uv run processing/spikes/play.py footage/rec3

Tracks the players on the pitch (positions in metres from pitch.py),
smooths them and takes velocities, then estimates a "play point" per
frame several ways:

- centroid / median of all players (what the follow cam aims at today)
- speed-weighted centroid: running players count far more than walkers
- flow lead: that centroid moved ahead along the fast players' mean
  velocity (several players running the same way: play is going there)
- convergence: the point fast players are running towards (closest to
  their heading lines, ignoring players running away from it)
- blend of flow lead and convergence

Each is scored against ball.py's ball estimate, which is right most of the
time it exists. Writes <recording>/play/: play.json, scores, minimap.mp4.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
import pitch as P  # noqa: E402

GATE_M = 3.0          # a detection this close to a track's prediction can join it
MAX_MISSED_S = 0.5
MIN_TRACK_S = 1.0     # shorter tracks are duplicates or noise
SMOOTH_S = 0.25       # Gaussian sigma on positions before taking velocity
RUN_SPEED = 2.5       # m/s: above this a player counts as running
LEAD_S = 1.0          # how far ahead the flow lead looks
CONV_PRIOR = 0.05     # pull of the convergence point towards the weighted centroid


# ----------------------------------------------------------------- tracking

def track_players(frames, fps):
    """Hungarian matching of on-pitch people to constant-velocity
    predictions. Returns {track id: (frame indices, positions)}."""
    tracks: dict[int, dict] = {}
    alive: list[int] = []
    nid = 0
    for i, f in enumerate(frames):
        det = np.array([d["xz"] for d in f["detections"] if d["cls"] == "person" and d["onPitch"]]).reshape(-1, 2)
        alive = [k for k in alive if (i - tracks[k]["t"][-1]) / fps <= MAX_MISSED_S]
        if alive and len(det):
            pred = np.array([tracks[k]["x"][-1] + tracks[k]["v"] * (i - tracks[k]["t"][-1]) / fps for k in alive])
            cost = np.hypot(*(pred[:, None, :] - det[None, :, :]).transpose(2, 0, 1))
            rows, cols = linear_sum_assignment(cost)
            matched = set()
            for r, c in zip(rows, cols):
                if cost[r, c] > GATE_M:
                    continue
                k = alive[r]
                tr = tracks[k]
                dt = (i - tr["t"][-1]) / fps
                tr["v"] = 0.7 * tr["v"] + 0.3 * (det[c] - tr["x"][-1]) / dt
                tr["t"].append(i)
                tr["x"].append(det[c])
                matched.add(c)
        else:
            matched = set()
        for c in range(len(det)):
            if c not in matched:
                tracks[nid] = {"t": [i], "x": [det[c]], "v": np.zeros(2)}
                alive.append(nid)
                nid += 1
    out = {}
    for k, tr in tracks.items():
        if (tr["t"][-1] - tr["t"][0]) / fps < MIN_TRACK_S:
            continue
        t = np.array(tr["t"])
        x = np.array(tr["x"])
        # Fill missed frames linearly, then smooth and differentiate.
        tt = np.arange(t[0], t[-1] + 1)
        xx = np.c_[np.interp(tt, t, x[:, 0]), np.interp(tt, t, x[:, 1])]
        xs = gaussian_filter1d(xx, SMOOTH_S * fps, axis=0, mode="nearest")
        v = np.gradient(xs, axis=0) * fps
        out[k] = (tt, xs, v)
    return out


def per_frame(tracks, n):
    """For each frame: arrays of positions and velocities of tracked players."""
    pos = [[] for _ in range(n)]
    vel = [[] for _ in range(n)]
    for tt, xs, v in tracks.values():
        for i, x, u in zip(tt, xs, v):
            pos[i].append(x)
            vel[i].append(u)
    return [np.array(p).reshape(-1, 2) for p in pos], [np.array(u).reshape(-1, 2) for u in vel]


# --------------------------------------------------------------- play point

def convergence(x, v, w, prior, iters=3):
    """Point closest (weighted) to the heading lines of moving players,
    pulled lightly towards `prior`; players for whom it lies behind them
    are dropped and it is solved again."""
    speed = np.hypot(v[:, 0], v[:, 1])
    keep = speed > RUN_SPEED
    p = prior
    for _ in range(iters):
        if keep.sum() < 2:
            return prior
        u = v[keep] / speed[keep, None]
        A = CONV_PRIOR * np.eye(2) * w[keep].sum()
        b = CONV_PRIOR * w[keep].sum() * prior
        for xi, ui, wi in zip(x[keep], u, w[keep]):
            M = np.eye(2) - np.outer(ui, ui)
            A += wi * M
            b += wi * M @ xi
        p = np.linalg.solve(A, b)
        ahead = ((p - x) * (v / np.maximum(speed, 1e-6)[:, None])).sum(1) > 0
        new = keep & ahead
        if new.sum() == keep.sum():
            break
        keep = new
    return p


def play_points(pos, vel):
    """All candidate play points for one frame."""
    if len(pos) == 0:
        return None
    speed = np.hypot(vel[:, 0], vel[:, 1])
    w = np.clip(speed, 0.5, 8.0) ** 2
    weighted = (pos * w[:, None]).sum(0) / w.sum()
    run = speed > RUN_SPEED
    flow = (vel[run] * w[run, None]).sum(0) / w[run].sum() if run.any() else np.zeros(2)
    lead = weighted + LEAD_S * flow
    conv = convergence(pos, vel, w, weighted)
    return {
        "centroid": pos.mean(0),
        "median": np.median(pos, axis=0),
        "speedWeighted": weighted,
        "flowLead": lead,
        "convergence": conv,
        "blend": 0.5 * (lead + conv),
    }


CAMERA_SMOOTH_S = 1.5   # the follow cam smooths its aim about this much
VIEW_MARGIN = 0.8       # ball counts as in view inside this fraction of the half-width


def evaluate(points, ball, detected, pose, fps, names):
    """Score each play point the way the follow cam would use it: smoothed
    over time, turned into an aim direction, compared with the ball's
    direction. Reports the median yaw error and how often the ball would be
    inside a 50 and a 60 degree wide view."""
    n = len(points)
    ok = [i for i in range(n) if points[i] and ball[i] is not None]
    bang = {i: pose.angles(ball[i][None])[0] for i in ok}
    scores = {}
    print(f"{'play point':14s} {'metres':>7s} {'yaw err':>8s} {'in 50 deg':>10s} {'in 60 deg':>10s}"
          f" {'in 60, detected':>16s}")
    for name in names:
        tr = np.array([points[i][name] if points[i] else [np.nan, np.nan] for i in range(n)])
        # Fill gaps, then smooth like the camera path.
        idx = np.arange(n)
        good = ~np.isnan(tr[:, 0])
        tr = np.c_[np.interp(idx, idx[good], tr[good, 0]), np.interp(idx, idx[good], tr[good, 1])]
        tr = gaussian_filter1d(tr, CAMERA_SMOOTH_S * fps, axis=0, mode="nearest")
        aim = pose.angles(tr)
        dm = np.array([np.hypot(*(tr[i] - ball[i])) for i in ok])
        dy = np.array([abs(aim[i, 0] - bang[i][0]) for i in ok])
        det = np.array([detected[i] for i in ok])
        r = {"medianM": float(np.median(dm)), "medianYawErrDeg": float(np.median(dy))}
        for hfov in (50, 60):
            r[f"inView{hfov}"] = float(np.mean(dy < VIEW_MARGIN * hfov / 2))
        r["inView60Detected"] = float(np.mean(dy[det] < VIEW_MARGIN * 30))
        scores[name] = r
        print(f"{name:14s} {r['medianM']:6.1f}m {r['medianYawErrDeg']:7.1f}d {100 * r['inView50']:9.0f}%"
              f" {100 * r['inView60']:9.0f}% {100 * r['inView60Detected']:15.0f}%")
    return scores


# ------------------------------------------------------------------- output

def minimap_video(path, pose, pos, vel, points, ball, fps, show=("median", "flowLead", "convergence", "blend")):
    s, pad = 8, 40
    W_, H_ = int(pose.L * s) + 2 * pad, int(pose.W * s) + 2 * pad
    W_, H_ = W_ // 2 * 2, H_ // 2 * 2

    def px(p):
        return int(pad + (p[0] + pose.L / 2) * s), int(pad + (pose.W / 2 - p[1]) * s)

    base = np.full((H_, W_, 3), (60, 120, 60), np.uint8)
    for x, z in P.pitch_lines(pose.L, pose.W, pose.k, step=0.1):
        cv2.circle(base, px((x, z)), 1, (255, 255, 255), -1)
    colours = {"median": (200, 200, 200), "flowLead": (0, 200, 255), "convergence": (255, 0, 255),
               "blend": (0, 255, 255), "speedWeighted": (255, 160, 0), "centroid": (128, 128, 128)}
    y = 18
    for name in show:
        cv2.putText(base, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours[name], 1)
        y += 18
    cv2.putText(base, "ball", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W_}x{H_}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "23",
                             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for i in range(len(pos)):
        img = base.copy()
        for x, v in zip(pos[i], vel[i]):
            sp = np.hypot(*v)
            col = (255, 255, 255) if sp < RUN_SPEED else (0, 165, 255)
            cv2.circle(img, px(x), 4, col, -1)
            cv2.arrowedLine(img, px(x), px(x + v * 0.6), col, 2, tipLength=0.3)
        if points[i]:
            for name in show:
                cv2.drawMarker(img, px(points[i][name]), colours[name], cv2.MARKER_CROSS, 18, 3)
        if ball[i] is not None:
            cv2.circle(img, px(ball[i]), 6, (0, 0, 255), 2)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def main(rec: Path) -> None:
    out = rec / "play"
    out.mkdir(exist_ok=True)
    pose = P.load_pose(rec)
    data = json.loads((rec / "detect" / "detections_pitch.json").read_text())
    fps, frames = data["fps"], data["frames"]
    b = json.loads((rec / "ball" / "ball.json").read_text())["frames"]
    ball = [np.array(f["xz"]) if f["xz"] is not None else None for f in b]
    detected = [f["detected"] for f in b]

    tracks = track_players(frames, fps)
    pos, vel = per_frame(tracks, len(frames))
    points = [play_points(p, v) for p, v in zip(pos, vel)]
    lens = [len(t[0]) / fps for t in tracks.values()]
    print(f"{len(tracks)} player tracks (median {np.median(lens):.1f} s); tracked per frame median"
          f" {int(np.median([len(p) for p in pos]))}; running (> {RUN_SPEED} m/s) median"
          f" {int(np.median([(np.hypot(*v.T) > RUN_SPEED).sum() if len(v) else 0 for v in vel]))}")

    names = list(next(p for p in points if p).keys())
    scores = evaluate(points, ball, detected, pose, fps, names)

    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    (out / "play.json").write_text(json.dumps({"fps": fps, "frames": [
        {"t": f["t"], **({k: [round(float(c), 2) for c in v] for k, v in p.items()} if p else {})}
        for f, p in zip(frames, points)]}))
    minimap_video(out / "minimap.mp4", pose, pos, vel, points, ball, fps)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
