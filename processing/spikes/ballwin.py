#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""Ball candidates for a clip with the detector run in search windows.

    uv run processing/spikes/ballwin.py footage/rec3 <segment> <start s> <seconds> <checkpoint.pth> [fps]

ballclip.py scans ~28 tiles of 640 px per 4K frame. This keeps a simple
online track of the ball on the pitch (last accepted detection plus its
velocity) and, in each phone's frame, runs the detector only on a window
around the predicted position: one native 640 px window, plus a 2x zoom
window (a 320 px crop upscaled) when the ball is expected to be small.
A full scan runs every FULL_EVERY_S, or once the ball has been lost for
LOST_S. Every candidate is scored by balljudge.py's crop judge inline.

Writes <recording>/detect/ball_<segment>_<start>_<seconds>.win.json in
ballclip.py's format (plus "src": scan / win / zoom per candidate) and
prints the tiles per frame against a full scan.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import balljudge  # noqa: E402
import detect  # noqa: E402
import follow  # noqa: E402
import gpu  # noqa: E402
import pitch as P  # noqa: E402

FLOOR = 0.1
MERGE_DEG = 0.4         # the same ball seen by both phones
FULL_EVERY_S = 2.0      # a full scan this often regardless
LOST_S = 1.0            # ... and once the ball has been lost this long, every LOST_SCAN_S
LOST_SCAN_S = 0.5
ZOOM_BELOW_PX = 10.0    # 2x zoom window when the expected ball diameter is under this
ACCEPT_CONF = 0.3       # a detection at least this sure, and judged at least ACCEPT_JUDGE, updates the track ...
ACCEPT_JUDGE = 0.3
GATE_M = 3.0            # ... if within this + MAX_SPEED x dt of the prediction
MAX_SPEED = 35.0
WIN = 640
ZOOM_CROP = 320


def main(rec: Path, seg: int, start: float, seconds: float, ckpt: Path, fps: float) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    roles = ("main", "secondary")
    pose = P.load_pose(rec)
    hl, hw = pose.L / 2 + P.ON_PITCH_MARGIN, pose.W / 2 + P.ON_PITCH_MARGIN
    diam_at = balljudge.expected_at(rec)
    judge = balljudge.load_judge(rec)

    dev = gpu.device()
    model = rfdetr.RFDETRMedium(pretrain_weights=str(ckpt))
    W, H = follow.SRC_W, follow.SRC_H
    streams = [follow.prefetch(follow.game_stream(rec, devs, report, ph, seg, start, seconds, W, fps=fps))
               for ph in (0, 1)]

    def to_pixel(phone, xz):
        """Predicted pitch position -> this phone's pixel, or None if out of frame."""
        yaw, pit = np.radians(pose.angles(np.asarray(xz)[None])[0])
        x, y, ok = cams[phone].project(follow.direction(yaw, pit)[None])
        return (float(x[0]), float(y[0])) if ok[0] and 0 <= x[0] < W and 0 <= y[0] < H else None

    def run(img, boxes_xy, scale=1.0):
        """Detector on crops of `img` (CHW float) at the given top-left
        corners; returns rows [x1, y1, x2, y2, conf] in frame pixels."""
        crops = []
        for x, y in boxes_xy:
            side = int(WIN / scale)
            c = img[:, y:y + side, x:x + side]
            if scale != 1.0:
                c = F.interpolate(c[None], size=(WIN, WIN), mode="bilinear", align_corners=False)[0]
            crops.append(c)
        res = model.predict(crops, threshold=FLOOR)
        rows = []
        for (x, y), det in zip(boxes_xy, res):
            for box, sc in zip(det.xyxy, det.confidence):
                b = np.array(box) / scale + [x, y, x, y]
                rows.append(np.r_[b, sc])
        return rows

    frames_out, t0 = [], time.time()
    ball_xz, ball_v, last_seen, last_full = None, np.zeros(2), -10 ** 9, -10 ** 9
    tiles_total, n_full = 0, 0
    for i, (fa, fb) in enumerate(zip(*streams)):
        t = i / fps
        lost = (i - last_seen) / fps
        since_full = (i - last_full) / fps
        full = since_full >= FULL_EVERY_S or ((ball_xz is None or lost >= LOST_S) and since_full >= LOST_SCAN_S)
        pred = ball_xz + ball_v * ((i - last_seen) / fps) if ball_xz is not None else None
        found = []
        for phone, (buf, fl) in enumerate(((fa, flip[0]), (fb, flip[1]))):
            img = detect.nv12_to_rgb(*gpu.upload_nv12(buf, W, H, dev))
            img = img.flip(1, 2) if fl else img
            rows = []
            if full:
                tl = detect.tiles(H, W)
                rows += [(r, "scan") for r in run(img, tl)]
                tiles_total += len(tl)
            if pred is not None:
                px = to_pixel(phone, pred)
                if px is not None:
                    x0 = int(np.clip(px[0] - WIN / 2, 0, W - WIN)); y0 = int(np.clip(px[1] - WIN / 2, 0, H - WIN))
                    rows += [(r, "win") for r in run(img, [(x0, y0)])]
                    tiles_total += 1
                    if diam_at(roles[phone], *px) < ZOOM_BELOW_PX:
                        x0 = int(np.clip(px[0] - ZOOM_CROP / 2, 0, W - ZOOM_CROP))
                        y0 = int(np.clip(px[1] - ZOOM_CROP / 2, 0, H - ZOOM_CROP))
                        rows += [(r, "zoom") for r in run(img, [(x0, y0)], scale=2.0)]
                        tiles_total += 1
            if not rows:
                continue
            kept = detect.nms(np.array([r for r, _ in rows]))
            src_of = {tuple(np.round(r[:4])): s for r, s in rows}
            bgr = (img.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
            crops = np.stack([balljudge.crop_at(bgr, (x1 + x2) / 2, (y1 + y2) / 2,
                                                diam_at(roles[phone], (x1 + x2) / 2, (y1 + y2) / 2))
                              for x1, y1, x2, y2, _ in kept])
            scores = judge(crops)
            for (x1, y1, x2, y2, sc), js in zip(kept, scores):
                yaw, pit = detect.to_angles(cams[phone], np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
                x, z = pose.to_pitch(yaw, pit)
                xz = None if np.isnan(x) else [round(float(x), 2), round(float(z), 2)]
                found.append({"cls": "ball", "phone": phone, "yaw": round(float(yaw), 3),
                              "pitch": round(float(pit), 3), "conf": round(float(sc), 3),
                              "judge": round(float(js), 3), "xz": xz,
                              "onPitch": bool(xz is not None and abs(x) <= hl and abs(z) <= hw),
                              "box": [round(float(v)) for v in (x1, y1, x2, y2)],
                              "src": src_of.get(tuple(np.round([x1, y1, x2, y2])), "scan")})
        if full:
            last_full, n_full = i, n_full + 1
        merged = []
        for d in sorted(found, key=lambda d: -d["conf"] * d["judge"]):
            if not any(m["phone"] != d["phone"] and np.hypot(m["yaw"] - d["yaw"], m["pitch"] - d["pitch"]) < MERGE_DEG
                       for m in merged):
                merged.append(d)
        # Online track: the surest on-pitch candidate near the prediction (or anywhere after a scan).
        best = None
        for d in merged:
            if not d["onPitch"] or d["conf"] < ACCEPT_CONF or d["judge"] < ACCEPT_JUDGE:
                continue
            if pred is not None:
                gate = GATE_M + MAX_SPEED * (i - last_seen) / fps
                if np.hypot(*(np.array(d["xz"]) - pred)) > gate:
                    continue
            if best is None or d["conf"] * d["judge"] > best["conf"] * best["judge"]:
                best = d
        if best is not None:
            xz = np.array(best["xz"])
            if ball_xz is not None and i > last_seen:
                v = (xz - ball_xz) * fps / (i - last_seen)
                ball_v = 0.5 * ball_v + 0.5 * v
            ball_xz, last_seen = xz, i
        frames_out.append({"t": round(start + t, 3), "detections": merged})
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{int(seconds * fps)} frames, {el / 60:.1f} min, {tiles_total / (i + 1):.1f} tiles/frame"
                  f" ({n_full} full scans)", flush=True)
    path = rec / "detect" / f"ball_{seg}_{start:g}_{seconds:g}.win.json"
    path.write_text(json.dumps({"segment": seg, "fps": fps, "model": str(ckpt), "mode": "window", "frames": frames_out}))
    n = len(frames_out)
    print(f"{n} frame pairs in {(time.time() - t0) / 60:.1f} min; {tiles_total / n:.1f} tiles per frame pair"
          f" vs {2 * len(detect.tiles(H, W))} for full scans; {n_full} full scans -> {path.name}")


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), int(a[2]), float(a[3]), float(a[4]), Path(a[5]), float(a[6]) if len(a) > 6 else 10.0)
