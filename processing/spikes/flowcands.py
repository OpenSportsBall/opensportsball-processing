#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Ball candidates from motion alone, and goal-net ripple, along a clip.

    uv run processing/spikes/flowcands.py footage/rec3 <segment> <start s> <seconds> [fps]

At each candidate frame (every 30/fps source frames) and in each phone,
the frames just before and after give a double difference that isolates
what moved at that instant; blobs about the size a ball would be at that
spot, with consistent motion over the three frames (the same step before
and after), are candidates. They don't depend on a detector seeing a
5 px ball; a ball at rest gives nothing (the detector covers those).
Each candidate is scored by balljudge.py's crop judge.

Also, from the same frames: the share of the goal-mouth area (the goals
from pitch.py, projected into each phone) that changed between frames,
per candidate frame. The net ripples for a second or two when the ball
hits it: a goal or side-netting event.

Writes <recording>/detect/ball_<segment>_<start>_<seconds>.flow.json
(ballclip.py's format, "src": "flow") and net_<...>.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import balljudge  # noqa: E402
import detect  # noqa: E402
import follow  # noqa: E402
import pitch as P  # noqa: E402

BAND_TOP = 700          # rows above this are sky and trees
DIFF_K = 4.0            # moved: above this many times the frame's median difference ...
MIN_DIFF = 10           # ... and at least this (grey levels)
SIZE_RANGE = (0.25, 8.0)  # blob area vs a ball's area there
MAX_ELONG = 6.0
MIN_STEP_PX = 1.5       # must move at least this per frame (else it is not the ball in play)
MAX_STEP_PX = 120.0
CONSIST = 0.35          # |step before - step after| <= CONSIST x |step| + 3 px
NET_THR = 12            # net ripple: pixels changed by more than this


def grid_diam(rec: Path, w: int, h: int, step: int = 64):
    """Expected ball diameter (px) on a grid of each phone's frame."""
    diam = balljudge.expected_at(rec)
    out = {}
    for role in ("main", "secondary"):
        g = np.zeros((h // step + 1, w // step + 1), np.float32)
        for iy in range(g.shape[0]):
            for ix in range(g.shape[1]):
                g[iy, ix] = diam(role, min(w - 1, ix * step), min(h - 1, iy * step))
        out[role] = g
    return out, step


def band_diff(a, b):
    """|a - b| over the field band, uint8 (OpenCV, fast)."""
    return cv2.absdiff(a[BAND_TOP:], b[BAND_TOP:])


def blobs(d1, d2, diam_grid, step):
    """Small moving blobs where a frame differs from both neighbours
    (d1, d2: band differences): centroid, area, expected diameter."""
    d = cv2.min(d1, d2)
    thr = max(MIN_DIFF, DIFF_K * float(np.median(d1[::8, ::8])))
    fg = (d > thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n_, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
    out = []
    for k in range(1, n_):
        x, y, w, h, area = stats[k]
        cx, cy = float(cent[k][0]), float(cent[k][1]) + BAND_TOP
        s = float(diam_grid[min(diam_grid.shape[0] - 1, int(cy) // step), min(diam_grid.shape[1] - 1, int(cx) // step)])
        expect = np.pi * (s / 2) ** 2
        if not (SIZE_RANGE[0] * expect <= area <= SIZE_RANGE[1] * expect) or max(w, h) / max(1, min(w, h)) > MAX_ELONG:
            continue
        out.append((cx, cy, float(area), s))
    return out


def consistent(mid, before, after):
    """Mid blobs with a partner before and after on a straight, even path."""
    out = []
    for cx, cy, area, s in mid:
        best = None
        for bx, by, _, _ in before:
            d1 = np.array([cx - bx, cy - by])
            n1 = np.hypot(*d1)
            if not (MIN_STEP_PX <= n1 <= MAX_STEP_PX):
                continue
            for ax, ay, _, _ in after:
                d2 = np.array([ax - cx, ay - cy])
                if abs(np.hypot(*(d2 - d1))) <= CONSIST * n1 + 3:
                    score = 1 - np.hypot(*(d2 - d1)) / (CONSIST * n1 + 3)
                    if best is None or score > best[0]:
                        best = (score, n1)
        if best:
            out.append((cx, cy, area, s, best[0], best[1]))
    return out


def goal_masks(rec: Path, cams, pose, w: int, h: int):
    """Per phone and goal side, the goal-mouth polygon in frame pixels (or None)."""
    goals = P.load_goals(rec)
    masks = {}
    for phone, cam in enumerate(cams):
        for g in goals:
            x, z, hw_, ht = g["x"], g["z"], g["width"] / 2, g["height"]
            pts = np.array([[x, z - hw_], [x, z + hw_], [x, z + hw_], [x, z - hw_]])
            hs = [0.0, 0.0, ht, ht]
            ang = np.radians(pose.angles3(pts, hs))
            px, py, ok = cam.project(follow.direction(ang[:, 0], ang[:, 1]))
            if ok.all():
                masks[(phone, g["side"])] = np.c_[px, py].astype(np.int32)
    return masks


def main(rec: Path, seg: int, start: float, seconds: float, fps: float) -> None:
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    roles = ("main", "secondary")
    pose = P.load_pose(rec)
    hl, hw = pose.L / 2 + P.ON_PITCH_MARGIN, pose.W / 2 + P.ON_PITCH_MARGIN
    W, H = follow.SRC_W, follow.SRC_H
    diam_grids, gstep = grid_diam(rec, W, H)
    judge = balljudge.load_judge(rec)
    masks = goal_masks(rec, cams, pose, W, H)
    mask_img = {}
    for key, poly in masks.items():
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [poly], 1)
        mask_img[key] = m.astype(bool)
    print(f"goal mouths in view: {sorted(masks)}", flush=True)

    sub = int(round(follow.FPS / fps))  # source frames per candidate frame
    n_out = int(round(seconds * fps))
    per_frame = [[] for _ in range(n_out)]
    net = [[0.0, 0.0] for _ in range(n_out)]  # per frame: [left goal, right goal] max over phones
    t0 = time.time()
    for phone in (0, 1):
        stream = follow.game_stream(rec, devs, report, phone, seg, start, seconds, W, fps=follow.FPS)
        hist = []  # last 5 grey frames (Y plane), upright
        for i, buf in enumerate(stream):
            y = np.asarray(buf[:W * H]).reshape(H, W)
            if flip[phone]:
                y = y[::-1, ::-1]
            hist.append(y)
            if len(hist) > 5:
                hist.pop(0)
            # Candidate frame k corresponds to source frame k * sub; process when its +2 neighbour has arrived.
            if i < 4 or (i - 2) % sub != 0:
                continue
            k = (i - 2) // sub
            if k >= n_out:
                break
            f0, f1, f2, f3, f4 = hist[-5:]
            d12, d23 = band_diff(f1, f2), band_diff(f2, f3)
            mid = blobs(d12, d23, diam_grids[roles[phone]], gstep)
            if mid:
                before = blobs(band_diff(f0, f1), d12, diam_grids[roles[phone]], gstep)
                after = blobs(d23, band_diff(f3, f4), diam_grids[roles[phone]], gstep)
                cons = consistent(mid, before, after)
            else:
                cons = []
            if cons:
                bgr = cv2.cvtColor(np.asarray(buf).reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_NV12)
                if flip[phone]:
                    bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                # (buf is frame i = mid + 2; the judge crops come from the mid frame's colour
                # only approximately: use this frame, the ball has moved < step px.)
                crops = np.stack([balljudge.crop_at(bgr, cx, cy, s) for cx, cy, _, s, _, _ in cons])
                js = judge(crops)
                for (cx, cy, area, s, score, stp), j in zip(cons, js):
                    yaw, pit = detect.to_angles(cams[phone], np.array([[cx, cy]]))[0]
                    x, z = pose.to_pitch(yaw, pit)
                    xz = None if np.isnan(x) else [round(float(x), 2), round(float(z), 2)]
                    per_frame[k].append({"cls": "ball", "phone": phone, "yaw": round(float(yaw), 3),
                                         "pitch": round(float(pit), 3), "conf": round(float(score), 3),
                                         "judge": round(float(j), 3), "xz": xz,
                                         "onPitch": bool(xz is not None and abs(x) <= hl and abs(z) <= hw),
                                         "box": [round(cx - s / 2), round(cy - s / 2), round(cx + s / 2), round(cy + s / 2)],
                                         "stepPx": round(float(stp), 1), "src": "flow"})
            # Net ripple: share of the goal mouth that changed between the mid frame and the one before.
            for (ph, side), m in mask_img.items():
                if ph != phone:
                    continue
                d = cv2.absdiff(f2, f1)[m]
                net[k][0 if side < 0 else 1] = max(net[k][0 if side < 0 else 1], float((d > NET_THR).mean()))
            if (k + 1) % 200 == 0:
                print(f"phone {phone}: {k + 1}/{n_out} frames, {(time.time() - t0) / 60:.1f} min", flush=True)
    frames_out = [{"t": round(start + k / fps, 3), "detections": per_frame[k]} for k in range(n_out)]
    path = rec / "detect" / f"ball_{seg}_{start:g}_{seconds:g}.flow.json"
    path.write_text(json.dumps({"segment": seg, "fps": fps, "mode": "flow", "frames": frames_out}))
    (rec / "detect" / f"net_{seg}_{start:g}_{seconds:g}.json").write_text(json.dumps({"fps": fps, "net": net}))
    n = sum(len(f) for f in per_frame)
    print(f"{n_out} frames, {n} motion candidates ({n / n_out:.1f} per frame), {(time.time() - t0) / 60:.1f} min -> {path.name}")


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), int(a[2]), float(a[3]), float(a[4]), float(a[5]) if len(a) > 5 else 10.0)
