#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""Ball candidates for a clip from the fine-tuned detector (balltrain.py).

    uv run processing/spikes/ballclip.py footage/rec3 <segment> <start s> <seconds> <checkpoint.pth> [fps]

Both phones' full-resolution frames, tiled as in detect.py; every
suggestion at or above FLOOR is kept with its confidence, placed on the
pitch (pitch.json) like detect.py + pitch.py do. The tuned model has one
class, so everything it finds is a ball candidate.

Writes <recording>/detect/ball_<segment>_<start>_<seconds>.json, read by
follow.py's "ball" feature.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import detect  # noqa: E402
import follow  # noqa: E402
import gpu  # noqa: E402
import pitch as P  # noqa: E402

FLOOR = 0.1
MERGE_DEG = 0.4     # the same ball seen by both phones


def clip_path(rec: Path, seg: int, start: float, seconds: float) -> Path:
    return rec / "detect" / f"ball_{seg}_{start:g}_{seconds:g}.json"


def main(rec: Path, seg: int, start: float, seconds: float, ckpt: Path, fps: float) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    files = [rec / devs["main"]["segments"][seg]["objectKey"],
             rec / devs["secondary"]["segments"][seg]["objectKey"]]
    offsets = [0.0, report["segments"][seg]["offsetS"]]
    pose = P.load_pose(rec)
    hl, hw = pose.L / 2 + P.ON_PITCH_MARGIN, pose.W / 2 + P.ON_PITCH_MARGIN

    dev = gpu.device()
    model = rfdetr.RFDETRMedium(pretrain_weights=str(ckpt))
    W, H = follow.SRC_W, follow.SRC_H
    streams = [follow.prefetch(follow.frame_stream(f, start + o, seconds, W, fps=fps))
               for f, o in zip(files, offsets)]
    frames_out, t0 = [], time.time()
    for i, (fa, fb) in enumerate(zip(*streams)):
        found = []
        for phone, (buf, fl) in enumerate(((fa, flip[0]), (fb, flip[1]))):
            img = detect.nv12_to_rgb(*gpu.upload_nv12(buf, W, H, dev))
            img = img.flip(1, 2) if fl else img
            tl = detect.tiles(H, W)
            res = model.predict([img[:, y:y + detect.TILE, x:x + detect.TILE] for x, y in tl], threshold=FLOOR)
            rows = [np.r_[box + [x, y, x, y], sc] for (x, y), det in zip(tl, res)
                    for box, sc in zip(det.xyxy, det.confidence)]
            if not rows:
                continue
            for x1, y1, x2, y2, sc in detect.nms(np.array(rows)):
                yaw, pit = detect.to_angles(cams[phone], np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
                x, z = pose.to_pitch(yaw, pit)
                xz = None if np.isnan(x) else [round(float(x), 2), round(float(z), 2)]
                found.append({"cls": "ball", "phone": phone, "yaw": round(float(yaw), 3),
                              "pitch": round(float(pit), 3), "conf": round(float(sc), 3), "xz": xz,
                              "onPitch": bool(xz is not None and abs(x) <= hl and abs(z) <= hw),
                              "box": [round(float(v)) for v in (x1, y1, x2, y2)]})
        merged = []
        for d in sorted(found, key=lambda d: -d["conf"]):
            if not any(m["phone"] != d["phone"] and np.hypot(m["yaw"] - d["yaw"], m["pitch"] - d["pitch"]) < MERGE_DEG
                       for m in merged):
                merged.append(d)
        frames_out.append({"t": round(start + i / fps, 3), "detections": merged})
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{int(seconds * fps)} frame pairs, {el / 60:.1f} min,"
                  f" ~{el / (i + 1) * (seconds * fps - i - 1) / 60:.0f} min left", flush=True)
    path = clip_path(rec, seg, start, seconds)
    path.write_text(json.dumps({"segment": seg, "fps": fps, "model": str(ckpt), "frames": frames_out}))
    n = sum(len(f["detections"]) for f in frames_out)
    print(f"{len(frames_out)} frame pairs, {n} candidates -> {path}")


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), int(a[2]), float(a[3]), float(a[4]), Path(a[5]), float(a[6]) if len(a) > 6 else 10.0)
