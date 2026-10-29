#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Exports the labelled balls as a COCO dataset for fine-tuning RF-DETR.

    uv run processing/spikes/balldata.py footage/rec3 [out dir]

From <recording>/labels (labelset.py + the labelling page): 640 px tiles
of the full-resolution frames, one class "ball" (match and spare balls
alike; telling the match ball apart is the tracker's job). Each ball gets
a tile at a random offset, its box sized from its distance (pitch map and
camera: a 22 cm ball, plus margin for blur). Negatives: empty field tiles
from frames without a ball, and hard negatives where the off-the-shelf
detector's suggestions were wrong (kit, paint, lights). "Unsure" frames
are left out. Moments are split 80/20 into train and valid (both phones
of a moment on the same side); test is a copy of valid.

Writes <out>/{train,valid,test}/ with images and _annotations.coco.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import phase0  # noqa: E402
import pitch as P  # noqa: E402

TILE = 640
BAND_TOP = 760          # field band starts here (4K rows)
BALL_M = 0.22
BOX_MARGIN = 1.4        # boxes a bit bigger than the ball: blur, labelling error
BOX_PX = (8, 160)       # clamp (a ball a couple of metres from the camera is ~150 px)
VALID_FRAC = 0.2
NEG_PER_EMPTY = 2       # empty field tiles per frame without a ball
HARD_PER_FRAME = 2      # tiles at wrong suggestions per frame
SEED = 7


def balls_of(v: dict) -> list[dict]:
    if "balls" in v:
        return v["balls"]
    return [{"x": v["x"], "y": v["y"], "inPlay": True}] if v.get("status") == "ball" and "x" in v else []


def split_of(moment: str) -> str:
    h = int(hashlib.sha1(moment.encode()).hexdigest(), 16) % 1000
    return "valid" if h < VALID_FRAC * 1000 else "train"


def main(rec: Path, out: Path) -> None:
    lab_dir = rec / "labels"
    items = json.loads((lab_dir / "items.json").read_text())
    labels = json.loads((lab_dir / "labels.json").read_text())
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    pose = P.load_pose(rec)

    def box_px(phone: str, x: float, y: float) -> float:
        """Ball diameter in pixels at this pixel, if it is on the ground."""
        cam = cams[phone]
        d = cam.unproject(np.array([[x, y]], float))[0]
        yaw = np.degrees(np.arctan2(d[0], d[2]))
        pit = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
        gx, gz = pose.to_pitch(yaw, pit)
        if np.isnan(gx):
            return BOX_PX[0]
        dist = float(np.sqrt((gx - pose.cx) ** 2 + (gz - pose.cz) ** 2 + pose.h ** 2))
        return float(np.clip(BALL_M / dist * cam.f * BOX_MARGIN, *BOX_PX))

    rng = np.random.default_rng(SEED)
    if out.exists():
        shutil.rmtree(out)
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "ball", "supercategory": "ball"}]}
            for s in ("train", "valid")}
    counts = {s: {"tiles": 0, "balls": 0, "empty": 0, "hard": 0} for s in coco}
    for s in coco:
        (out / s).mkdir(parents=True)

    def add_tile(split, img, x0, y0, balls, phone, kind):
        tile = img[y0:y0 + TILE, x0:x0 + TILE]
        iid = len(coco[split]["images"]) + 1
        name = f"{split}_{iid:05d}.jpg"
        cv2.imwrite(str(out / split / name), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
        coco[split]["images"].append({"id": iid, "file_name": name, "width": TILE, "height": TILE})
        for b in balls:
            bx, by = b["x"] - x0, b["y"] - y0
            if not (0 <= bx < TILE and 0 <= by < TILE):
                continue
            s_ = box_px(phone, b["x"], b["y"])
            x1, y1 = max(0.0, bx - s_ / 2), max(0.0, by - s_ / 2)
            x2, y2 = min(TILE, bx + s_ / 2), min(TILE, by + s_ / 2)
            coco[split]["annotations"].append({
                "id": len(coco[split]["annotations"]) + 1, "image_id": iid, "category_id": 1,
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0})
            counts[split]["balls"] += 1
        counts[split]["tiles"] += 1
        counts[split][kind] = counts[split].get(kind, 0) + 1

    for it in items:
        moment = it["id"] if it.get("kind") == "pair" else it["id"].rsplit("_", 1)[0]
        split = split_of(moment)
        for f in (it["frames"] if it.get("kind") == "pair" else [it]):
            v = labels.get(f["id"])
            if not v or v["status"] == "unsure":
                continue
            img = cv2.imread(str(lab_dir / f["image"]))
            h, w = img.shape[:2]
            balls = balls_of(v)

            def clamp(x0, y0):
                return int(np.clip(x0, 0, w - TILE)), int(np.clip(y0, BAND_TOP - TILE // 2, h - TILE))

            for b in balls:
                ox, oy = rng.integers(64, TILE - 64, size=2)
                add_tile(split, img, *clamp(b["x"] - ox, b["y"] - oy), balls, f["phone"], "ball")
            # Hard negatives: suggestions that weren't a ball.
            wrong = [c for c in f["candidates"] if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > 20 for b in balls)]
            for c in wrong[:HARD_PER_FRAME]:
                ox, oy = rng.integers(64, TILE - 64, size=2)
                x0, y0 = clamp(c["x"] - ox, c["y"] - oy)
                if any(x0 <= b["x"] < x0 + TILE and y0 <= b["y"] < y0 + TILE for b in balls):
                    continue  # would contain a ball after all; covered by the ball tiles
                add_tile(split, img, x0, y0, [], f["phone"], "hard")
            if not balls:
                for _ in range(NEG_PER_EMPTY):
                    x0, y0 = clamp(rng.integers(0, w - TILE), rng.integers(BAND_TOP, h - TILE))
                    add_tile(split, img, x0, y0, [], f["phone"], "empty")

    for s, d in coco.items():
        (out / s / "_annotations.coco.json").write_text(json.dumps(d))
    shutil.copytree(out / "valid", out / "test")
    for s, c in counts.items():
        print(f"{s}: {c['tiles']} tiles ({c.get('ball', 0)} around balls, {c['empty']} empty, {c['hard']} hard negatives),"
              f" {c['balls']} ball boxes")


if __name__ == "__main__":
    rec = Path(sys.argv[1])
    main(rec, Path(sys.argv[2]) if len(sys.argv) > 2 else rec / "balldata")
