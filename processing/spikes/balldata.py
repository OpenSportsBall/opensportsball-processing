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
detector's suggestions were wrong (kit, paint, lights). A suggestion nobody
marked that lies outside the pitch lines may be someone else's ball (a
warm-up, the next pitch, a spare by the bench), so it is neither a ball nor
"not a ball": no hard negative is made of it and no tile may show it (a
tile is moved off it; failing that a negative tile is dropped and a ball's
tile has the spot painted over from its surroundings). The tracker drops candidates off the
pitch anyway (ball.usable). OSB_OFF_PITCH_NEG=1 keeps them as negatives,
as before 2026-11-11, to compare. "Unsure" frames
are left out, and so are moments in held-out time (holdout.py: test
labels no model trains on). Moments are split 80/20 into train and valid (both phones
of a moment on the same side); test is a copy of valid.

Writes <out>/{train,valid,test}/ with images and _annotations.coco.json,
and <out>/holdout.json (the held-out time left out; ballmerge.py and
balltrain.py check it).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import holdout  # noqa: E402
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
OFF_PITCH_M = 0.0       # unmarked suggestions this far outside the lines are left out of training
KEEP_OFF_PITCH = os.environ.get("OSB_OFF_PITCH_NEG") == "1"  # the old way: they are negatives
TILE_PAD = 24           # a tile is clear of a left-out suggestion this far beyond its edges too (px)
TILE_TRIES = 20         # placements tried to keep a tile clear before it is dropped or painted
SEED = 7


def balls_of(v: dict) -> list[dict]:
    if "balls" in v:
        return v["balls"]
    return [{"x": v["x"], "y": v["y"], "inPlay": True}] if v.get("status") == "ball" and "x" in v else []


def off_pitch(pose, gx: float, gz: float) -> bool:
    """A ground point (pitch metres) outside the lines. A point above the
    horizon (NaN: lights, stands) is not: it is no ball on another pitch."""
    if np.isnan(gx) or np.isnan(gz):
        return False
    return bool(abs(gx) > pose.L / 2 + OFF_PITCH_M or abs(gz) > pose.W / 2 + OFF_PITCH_M)


def ground_xz(cam, pose, x: float, y: float):
    """The pitch point under a 4K frame pixel (NaN above the horizon)."""
    d = cam.unproject(np.array([[x, y]], float))[0]
    yaw = np.degrees(np.arctan2(d[0], d[2]))
    pit = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
    return pose.to_pitch(yaw, pit)


def off_pitch_at(rec: Path):
    """(phone, x, y) -> whether that 4K frame pixel's ground point is
    outside the pitch lines, from the recording's cameras and pitch map."""
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    cm, cs = follow.setup_cameras(rec, report, devs, [report["upsideDown"]["main"], report["upsideDown"]["secondary"]])
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    pose = P.load_pose(rec)
    return lambda phone, x, y: off_pitch(pose, *ground_xz(cams[phone], pose, x, y))


def left_out(cands: list[dict], balls: list[dict], is_off, near: float) -> list[dict]:
    """The suggestions nobody marked (none of `balls` within `near` px)
    that are off the pitch: `is_off(x, y)`."""
    if KEEP_OFF_PITCH:
        return []
    return [c for c in cands if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > near for b in balls)
            and is_off(c["x"], c["y"])]


def tile_clear(x0: int, y0: int, pts: list[dict]) -> bool:
    """No left-out suggestion in the tile at (x0, y0) or just beyond its edges."""
    return not any(x0 - TILE_PAD <= c["x"] < x0 + TILE + TILE_PAD and y0 - TILE_PAD <= c["y"] < y0 + TILE + TILE_PAD
                   for c in pts)


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
        gx, gz = ground_xz(cam, pose, x, y)
        if np.isnan(gx):
            return BOX_PX[0]
        dist = float(np.sqrt((gx - pose.cx) ** 2 + (gz - pose.cz) ** 2 + pose.h ** 2))
        return float(np.clip(BALL_M / dist * cam.f * BOX_MARGIN, *BOX_PX))

    rng = np.random.default_rng(SEED)
    held = holdout.Holdout(rec)
    if out.exists():
        shutil.rmtree(out)
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "ball", "supercategory": "ball"}]}
            for s in ("train", "valid")}
    counts = {s: {"tiles": 0, "balls": 0, "empty": 0, "hard": 0, "left_out": 0, "dropped": 0, "painted": 0} for s in coco}
    for s in coco:
        (out / s).mkdir(parents=True)

    def add_tile(split, img, x0, y0, balls, phone, kind, hide=()):
        tile = img[y0:y0 + TILE, x0:x0 + TILE]
        if hide:  # left-out suggestions the tile couldn't avoid: filled in from around them
            mask = np.zeros(tile.shape[:2], np.uint8)
            for c in hide:
                r = int(box_px(phone, c["x"], c["y"]))
                for b in balls:  # never onto a marked ball's box
                    gap = max(abs(c["x"] - b["x"]), abs(c["y"] - b["y"])) - box_px(phone, b["x"], b["y"]) / 2
                    r = int(max(2, min(r, gap - 2)))
                cv2.rectangle(mask, (int(c["x"]) - x0 - r, int(c["y"]) - y0 - r),
                              (int(c["x"]) - x0 + r, int(c["y"]) - y0 + r), 255, -1)
            tile = cv2.inpaint(tile, mask, 5, cv2.INPAINT_TELEA)
            counts[split]["painted"] += 1
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

    n_held = 0
    for it in items:
        moment = it["id"] if it.get("kind") == "pair" else it["id"].rsplit("_", 1)[0]
        if held.held(it["segment"], it["t"]):
            n_held += 1
            continue
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

            # Unmarked suggestions off the pitch: perhaps another game's ball, so in no tile.
            skip = left_out(f["candidates"], balls,
                            lambda x, y: off_pitch(pose, *ground_xz(cams[f["phone"]], pose, x, y)), 20)
            counts[split]["left_out"] += len(skip)

            def place(corner):
                """A tile from corner() (a fresh random placement each call)
                clear of `skip`; None when none is found. With nothing
                left out the first try is taken, as before."""
                for _ in range(TILE_TRIES):
                    x0, y0 = clamp(*corner())
                    if tile_clear(x0, y0, skip):
                        return x0, y0
                return None

            def around(p):
                def corner():
                    ox, oy = rng.integers(64, TILE - 64, size=2)
                    return p["x"] - ox, p["y"] - oy
                return corner

            for b in balls:
                at = place(around(b))
                if at:
                    add_tile(split, img, *at, balls, f["phone"], "ball")
                else:  # a ball is too rare to drop: its tile as it falls, the left-out spots painted over
                    x0, y0 = clamp(*around(b)())
                    add_tile(split, img, x0, y0, balls, f["phone"], "ball",
                             hide=[c for c in skip if not tile_clear(x0, y0, [c])])
            # Hard negatives: suggestions on the pitch that weren't a ball.
            wrong = [c for c in f["candidates"] if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > 20 for b in balls)
                     and not any(c is k for k in skip)]
            for c in wrong[:HARD_PER_FRAME]:
                at = place(around(c))
                if not at:
                    counts[split]["dropped"] += 1
                    continue
                x0, y0 = at
                if any(x0 <= b["x"] < x0 + TILE and y0 <= b["y"] < y0 + TILE for b in balls):
                    continue  # would contain a ball after all; covered by the ball tiles
                add_tile(split, img, x0, y0, [], f["phone"], "hard")
            if not balls:
                for _ in range(NEG_PER_EMPTY):
                    at = place(lambda: (rng.integers(0, w - TILE), rng.integers(BAND_TOP, h - TILE)))
                    if at:
                        add_tile(split, img, *at, [], f["phone"], "empty")
                    else:
                        counts[split]["dropped"] += 1

    for s, d in coco.items():
        (out / s / "_annotations.coco.json").write_text(json.dumps(d))
    shutil.copytree(out / "valid", out / "test")
    (out / "holdout.json").write_text(json.dumps(held.stamp()))
    if n_held:
        print(f"{n_held} labelled moments in held-out time {held.windows} left out")
    for s, c in counts.items():
        print(f"{s}: {c['tiles']} tiles ({c.get('ball', 0)} around balls, {c['empty']} empty, {c['hard']} hard negatives),"
              f" {c['balls']} ball boxes; {c['left_out']} unmarked suggestions off the pitch left out,"
              f" {c['dropped']} negative tiles dropped and {c['painted']} ball tiles painted to stay clear of them")


if __name__ == "__main__":
    rec = Path(sys.argv[1])
    main(rec, Path(sys.argv[2]) if len(sys.argv) > 2 else rec / "balldata")
