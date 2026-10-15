#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""Phase 0 spike: players and ball from an off-the-shelf detector.

Run after phase0.py (audio offsets) and follow.py (cached cameras):

    uv run processing/spikes/detect.py footage/rec3 <segment> <start s> <seconds> [fps]

RF-DETR (Apache-2.0, COCO-trained: "person" and "sports ball") on tiles
of each phone's full-resolution frame: the ball is ~5 px wide at the far
side, so the frame can't be shrunk first. Detections from overlapping
tiles are merged, then mapped to world directions (yaw, pitch) with the
phones' cameras, and the two phones' views of the same object merged.

Writes <recording>/detect/: detections.json (per frame), a timeline of
ball candidates by yaw over time (a ball in play should draw a continuous
trail; spare balls on the sideline draw straight lines) and a few frames
with detections drawn.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import gpu  # noqa: E402

PERSON, BALL = 1, 37        # COCO class ids in RF-DETR's output
TILE, STRIDE = 640, 576     # 64 px overlap between tiles
BAND_TOP = 760              # first frame row tiled (4K); above it is sky and trees
THRESH = {PERSON: 0.35, BALL: 0.15}
MERGE_DEG = {PERSON: 0.6, BALL: 0.4}  # same object seen by both phones


def nv12_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """NV12 planes (1x1xHxW, 1x2xH/2xW/2) -> RGB 3xHxW in [0, 1]."""
    h, w = y.shape[-2:]
    uv = F.interpolate(uv.float(), size=(h, w), mode="bilinear", align_corners=False)[0]
    bgr = gpu.nv_to_bgr(y[0, 0].float(), uv)
    return bgr.flip(-1).permute(2, 0, 1).float() / 255


def tiles(h: int, w: int):
    ys = list(range(BAND_TOP, h - TILE, 530)) + [h - TILE]
    xs = list(range(0, w - TILE, STRIDE)) + [w - TILE]
    return [(x, y) for y in ys for x in xs]


def nms(boxes: np.ndarray, iou: float = 0.5, contain: float = 0.8) -> np.ndarray:
    """Greedy suppression on (x1, y1, x2, y2, score) rows: drops boxes that
    overlap a better one, or sit mostly inside it (a tile edge cut it)."""
    keep = []
    for i in np.argsort(-boxes[:, 4]):
        b = boxes[i]
        ok = True
        for k in keep:
            a = boxes[k]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = (a[2] - a[0]) * (a[3] - a[1]) + area_b - inter
            if inter / union > iou or inter / max(area_b, 1e-6) > contain:
                ok = False
                break
        if ok:
            keep.append(i)
    return boxes[keep]


def to_angles(cam: follow.Camera, pts: np.ndarray) -> np.ndarray:
    d = cam.unproject(pts)
    return np.degrees(np.c_[np.arctan2(d[:, 0], d[:, 2]), np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))])


def detect_frame(model, frames_rgb, cams):
    """Detections for one frame pair, merged across tiles and phones."""
    found = []
    for phone, (img, cam) in enumerate(zip(frames_rgb, cams)):
        h, w = img.shape[-2:]
        tl = tiles(h, w)
        res = model.predict([img[:, y:y + TILE, x:x + TILE] for x, y in tl],
                            threshold=min(THRESH.values()))
        for cls in (PERSON, BALL):
            rows = [np.r_[box + [x, y, x, y], sc]
                    for (x, y), det in zip(tl, res)
                    for box, c, sc in zip(det.xyxy, det.class_id, det.confidence)
                    if c == cls and sc >= THRESH[cls]]
            if not rows:
                continue
            for x1, y1, x2, y2, sc in nms(np.array(rows)):
                # People are placed by their feet; the ball by its centre.
                anchor = [(x1 + x2) / 2, y2 if cls == PERSON else (y1 + y2) / 2]
                yaw, pitch = to_angles(cam, np.array([anchor]))[0]
                size = float(np.degrees((y2 - y1) / cam.f))
                found.append({"cls": "person" if cls == PERSON else "ball", "phone": phone,
                              "yaw": round(float(yaw), 3), "pitch": round(float(pitch), 3),
                              "conf": round(float(sc), 3), "sizeDeg": round(size, 3),
                              "box": [round(float(v)) for v in (x1, y1, x2, y2)]})
    # The same object seen by both phones in the overlap: keep the surer one.
    merged = []
    for d in sorted(found, key=lambda d: -d["conf"]):
        if not any(m["cls"] == d["cls"] and m["phone"] != d["phone"]
                   and np.hypot(m["yaw"] - d["yaw"], m["pitch"] - d["pitch"]) < MERGE_DEG[
                       PERSON if d["cls"] == "person" else BALL]
                   for m in merged):
            merged.append(d)
    return merged


def main(rec: Path, seg: int, start: float, seconds: float, fps: float) -> None:
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
    out = rec / "detect"
    out.mkdir(exist_ok=True)

    dev = gpu.device()
    model = rfdetr.RFDETRMedium()
    W, H = follow.SRC_W, follow.SRC_H
    streams = [follow.prefetch(follow.frame_stream(f, start + o, seconds, W, fps=fps))
               for f, o in zip(files, offsets)]
    frames_out, t_det, n = [], 0.0, 0
    snapshots = {int(seconds * fps * k / 4) for k in (1, 2, 3)}
    for i, (fa, fb) in enumerate(zip(*streams)):
        rgb = []
        for buf, fl in ((fa, flip[0]), (fb, flip[1])):
            img = nv12_to_rgb(*gpu.upload_nv12(buf, W, H, dev))
            rgb.append(img.flip(1, 2) if fl else img)  # detectors expect people upright
        t0 = time.perf_counter()
        dets = detect_frame(model, rgb, cams)
        t_det += time.perf_counter() - t0
        n += 1
        frames_out.append({"t": round(start + i / fps, 3), "detections": dets})
        if i in snapshots:
            vis = [(im.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy() for im in rgb]
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                col = (0, 255, 0) if d["cls"] == "person" else (0, 0, 255)
                pad = 0 if d["cls"] == "person" else 10
                cv2.rectangle(vis[d["phone"]], (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, 3)
                if d["cls"] == "ball":
                    cv2.putText(vis[d["phone"]], f"{d['conf']:.2f}", (x1, y1 - 16), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, col, 3)
            cv2.imwrite(str(out / f"frame_{i:04d}.jpg"),
                        cv2.resize(np.hstack([vis[1], vis[0]]), (2 * 1280, 720)))
        if i % 50 == 0:
            print(f"frame {i}: {sum(d['cls'] == 'person' for d in dets)} people,"
                  f" {sum(d['cls'] == 'ball' for d in dets)} ball candidates", flush=True)
    (out / "detections.json").write_text(json.dumps({"segment": seg, "fps": fps, "frames": frames_out}))
    print(f"{n} frame pairs, detection {1000 * t_det / max(n, 1):.0f} ms per pair")
    timeline(frames_out, out / "timeline.png", fps)


def timeline(frames, path: Path, fps: float, px_per_deg: int = 6, px_per_frame: int = 3) -> None:
    """Yaw across, time down: people in grey, ball candidates by confidence
    (brighter red is surer)."""
    y0, y1 = -100, 100
    img = np.full((len(frames) * px_per_frame + 20, (y1 - y0) * px_per_deg, 3), 255, np.uint8)
    for i, f in enumerate(frames):
        row = 10 + i * px_per_frame
        for d in f["detections"]:
            x = int((d["yaw"] - y0) * px_per_deg)
            if d["cls"] == "person":
                cv2.circle(img, (x, row), 1, (190, 190, 190), -1)
        for d in f["detections"]:
            if d["cls"] == "ball":
                x = int((d["yaw"] - y0) * px_per_deg)
                c = int(255 * (1 - min(1.0, d["conf"] / 0.6)))
                cv2.circle(img, (x, row), 3, (c, c, 255), -1)
    for deg in range(y0, y1 + 1, 20):
        x = (deg - y0) * px_per_deg
        cv2.line(img, (x, 0), (x, 8), (0, 0, 0), 1)
        cv2.putText(img, f"{deg}", (x + 2, 8), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
         float(sys.argv[5]) if len(sys.argv) > 5 else 10.0)
