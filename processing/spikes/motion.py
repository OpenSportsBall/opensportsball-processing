#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Ball candidates from motion, scored against the ball labels.

    uv run processing/spikes/motion.py footage/rec3

The mount is fixed, so whatever moved stands out against the scene's
recent median. For each labelled frame (labelset.py + the labelling page):
the median of frames +-1 s around it is the background; the frame's
difference from it, confirmed by a difference from the frames just
before and after (so it moved now, not earlier), gives moving blobs.
Blobs about the size a ball would be at that spot (pitch map and camera:
a 22 cm ball, allowing for a blur streak) are candidates; much bigger
ones are players. Spare balls lying still are invisible to this by design.

Writes <recording>/labels/motion.json (candidates per frame) and
motion_examples.jpg, and prints recall of the match ball alone and
combined with the detector's suggestions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import phase0  # noqa: E402
import pitch as P  # noqa: E402

BAND_TOP = 760
BG_SECONDS, BG_FPS = 1.0, 5      # background: median of frames +-1 s, every 0.2 s
DIFF_K = 4.0                     # foreground: above this many times the frame's median difference
MIN_DIFF = 12                    # ... and at least this (8-bit grey levels)
SIZE_RANGE = (0.3, 6.0)          # blob area vs a still ball's area at that spot (a streak is longer)
MAX_ELONG = 6.0                  # longest/shortest side of a blob's box
HIT_PX = 15                      # a candidate this close to a label counts


def balls_of(v: dict) -> list[dict]:
    if "balls" in v:
        return v["balls"]
    return [{"x": v["x"], "y": v["y"], "inPlay": True}] if v.get("status") == "ball" and "x" in v else []


def grey(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]


def native_frames(path: Path, start: float, count: int, rotate180: bool) -> list[np.ndarray]:
    """`count` consecutive frames as recorded, greyscale, from `start`. Not
    resampled: a constant-rate filter right after a seek repeats the first
    frame, which would hide all motion between neighbours."""
    import subprocess

    w, h = 3840, 2160
    vf = "hflip,vflip,format=gray" if rotate180 else "format=gray"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-noautorotate", "-ss", f"{start:.4f}", "-i", str(path),
                          "-vf", vf, "-fps_mode", "passthrough", "-frames:v", str(count), "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"],
                         check=True, capture_output=True).stdout
    n = len(raw) // (w * h)
    return list(np.frombuffer(raw[: n * w * h], np.uint8).reshape(n, h, w))


def candidates(cur, prev, nxt, bg, ball_px_at) -> list[dict]:
    """Moving blobs of about ball size in the field band of `cur`."""
    band = slice(BAND_TOP, cur.shape[0])
    c, p, n, b = (a[band].astype(np.int16) for a in (cur, prev, nxt, bg))
    d_bg = np.abs(c - b)
    d_now = np.minimum(np.abs(c - p), np.abs(c - n))
    thr = max(MIN_DIFF, DIFF_K * float(np.median(d_bg)))
    fg = ((d_bg > thr) & (d_now > thr * 0.5)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    fg = cv2.dilate(fg, np.ones((3, 3), np.uint8))
    n_, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
    out = []
    for k in range(1, n_):
        x, y, w, h, area = stats[k]
        cx, cy = float(cent[k][0]), float(cent[k][1]) + BAND_TOP
        s = ball_px_at(cx, cy)
        expect = np.pi * (s / 2) ** 2
        if not (SIZE_RANGE[0] * expect <= area <= SIZE_RANGE[1] * expect):
            continue
        if max(w, h) / max(1, min(w, h)) > MAX_ELONG:
            continue
        contrast = float(d_bg[lab == k].mean()) / thr
        out.append({"x": round(cx, 1), "y": round(cy, 1), "area": int(area), "expect": round(expect, 1),
                    "score": round(contrast, 2)})
    return sorted(out, key=lambda c: -c["score"])


def main(rec: Path) -> None:
    lab_dir = rec / "labels"
    items = json.loads((lab_dir / "items.json").read_text())
    labels = json.loads((lab_dir / "labels.json").read_text())
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = {"main": report["upsideDown"]["main"], "secondary": report["upsideDown"]["secondary"]}
    cm, cs = follow.setup_cameras(rec, report, devs, [flip["main"], flip["secondary"]])
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    pose = P.load_pose(rec)

    def ball_px(phone):
        cam = cams[phone]

        def at(x, y):
            d = cam.unproject(np.array([[x, y]], float))[0]
            yaw = np.degrees(np.arctan2(d[0], d[2]))
            pit = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
            gx, gz = pose.to_pitch(yaw, pit)
            if np.isnan(gx):
                return 6.0
            dist = float(np.sqrt((gx - pose.cx) ** 2 + (gz - pose.cz) ** 2 + pose.h ** 2))
            return float(np.clip(0.22 / dist * cam.f, 4, 160))
        return at

    frames = [(it, f) for it in items for f in (it["frames"] if it.get("kind") == "pair" else [it])]
    frames = [(it, f) for it, f in frames if f["id"] in labels and labels[f["id"]]["status"] != "unsure"]
    out_path = lab_dir / "motion.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    t_start = time.time()
    for k, (it, f) in enumerate(frames):
        if f["id"] in results:
            continue
        seg, role = it["segment"], f["phone"]
        off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
        path = rec / devs[role]["segments"][seg]["objectKey"]
        t = it["t"] + off
        near = native_frames(path, t - 1 / 30, 3, flip[role])
        if len(near) < 3:
            continue
        step = int(30 / BG_FPS)
        bgf = native_frames(path, max(0.0, t - BG_SECONDS), int(2 * BG_SECONDS * 30) + 1, flip[role])[::step]
        bg = np.median(np.stack(bgf), axis=0).astype(np.uint8)
        results[f["id"]] = candidates(near[1], near[0], near[2], bg, ball_px(role))
        if (k + 1) % 25 == 0:
            out_path.write_text(json.dumps(results))
            print(f"{k + 1}/{len(frames)} frames, {(time.time() - t_start) / 60:.1f} min", flush=True)
    out_path.write_text(json.dumps(results))
    evaluate(frames, labels, results, lab_dir)


def evaluate(frames, labels, results, lab_dir: Path) -> None:
    def hit(cands, b):
        return any(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) < HIT_PX for c in cands)

    rows = []
    for _, f in frames:
        if f["id"] not in results:
            continue
        v = labels[f["id"]]
        m = [b for b in balls_of(v) if b["inPlay"]]
        rows.append((f, m[0] if m else None, results[f["id"]]))
    n_match = sum(r[1] is not None for r in rows)
    print(f"\n{len(rows)} labelled frames scored, {n_match} with the match ball")
    for top in (3, 5, 10, None):
        mo = lambda r: r[2][:top] if top else r[2]  # noqa: E731
        det = lambda r: [c for c in r[0]["candidates"] if c["conf"] >= 0.2]  # noqa: E731
        m_hit = sum(hit(mo(r), r[1]) for r in rows if r[1])
        both = sum(hit(mo(r) + det(r), r[1]) for r in rows if r[1])
        per = np.mean([len(mo(r)) for r in rows])
        print(f"motion top {top or 'all'}: match ball {100 * m_hit / n_match:.0f}%, {per:.1f} candidates/frame;"
              f" with the detector (conf >= 0.2): {100 * both / n_match:.0f}%")
    d_only = sum(hit([c for c in r[0]["candidates"] if c["conf"] >= 0.2], r[1]) for r in rows if r[1])
    print(f"detector alone (conf >= 0.2): match ball {100 * d_only / n_match:.0f}%")

    # Examples: motion found the match ball where the detector didn't.
    tiles = []
    for f, b, mo in rows:
        if b is None or not hit(mo[:5], b) or hit([c for c in f["candidates"] if c["conf"] >= 0.2], b):
            continue
        img = cv2.imread(str(lab_dir / f["image"]))
        x, y = int(b["x"]), int(b["y"])
        c = img[max(0, y - 80):y + 80, max(0, x - 120):x + 120].copy()
        c = cv2.copyMakeBorder(c, 0, 160 - c.shape[0], 0, 240 - c.shape[1], cv2.BORDER_CONSTANT)
        cv2.circle(c, (x - max(0, x - 120), y - max(0, y - 80)), 14, (0, 0, 255), 1)
        tiles.append(cv2.resize(c, (360, 240)))
        if len(tiles) == 12:
            break
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(lab_dir / "motion_examples.jpg"), np.vstack([np.hstack(tiles[i:i + 4])
                                                                      for i in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
