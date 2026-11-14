#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Exports the labelled balls as a COCO dataset for fine-tuning RF-DETR.

    uv run processing/spikes/balldata.py footage/rec3 [out dir] [label set ...]

From <recording>/labels (labelset.py + the labelling page; or the label
sets named, e.g. a clip set `labels_seg6`): 640 px tiles
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
pitch anyway (ball.usable). A suggestion triaged with offpitch.py is no
longer left out once its verdict is reviewed: "ball" makes it a ball like
any marked one, "not" a negative again (<set>/offpitch.json; "unsure" and
unreviewed drafts stay left out). The phones don't move, so one object is
triaged in many frames: each spot counts 10 times at most, and its balls
stay on one side of the train/valid split (plan_verdicts). OSB_OFF_PITCH_NEG=1 keeps them as negatives,
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
VERDICT_PX = 3.0        # a triage verdict belongs to the suggestion within this of its pixel
SPOT_PX = 8.0           # verdicts of one phone this close together are one spot: the same object, frame after frame
SPOT_CAP = 10           # frames of one spot used as positives ("ball") or as hard negatives ("not")
SPOT_PERSIST = 5        # a ball spot marked this often lies there all game: hidden where it has no verdict
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


def left_out(cands: list[dict], balls: list[dict], is_off, near: float, cleared: list[dict] = ()) -> list[dict]:
    """The suggestions nobody marked (none of `balls` within `near` px)
    that are off the pitch: `is_off(x, y)`. Those in `cleared` were looked
    at and are not a ball (plan_verdicts), so they stay negatives."""
    if KEEP_OFF_PITCH:
        return []
    return [c for c in cands if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > near for b in balls)
            and not any(c is k for k in cleared) and is_off(c["x"], c["y"])]


def load_verdicts(lab_dir: Path) -> dict[str, list[dict]]:
    """The triage of a label set's unmarked off-pitch suggestions
    (offpitch.py): {frame id: [{"x", "y", "verdict", "by"}]}."""
    p = lab_dir / "offpitch.json"
    return json.loads(p.read_text()) if p.exists() else {}


def plan_verdicts(verdicts: dict[str, list[dict]], frames: list[tuple[str, str, str]]) -> dict[str, dict]:
    """What each frame's triage verdicts mean for training, given the set's
    trainable frames as (frame id, phone, split). The phones don't move, so
    one object (a spare ball by the bags, a lamp) gets a verdict in frame
    after frame: verdicts of one phone within SPOT_PX of each other are one
    spot, and a spot counts SPOT_CAP times at most, spread over its frames.

    Returns {frame id: {"ball", "not", "hard", "hide"}}, lists of {"x", "y"}:
      ball  a positive in this frame. A spot's balls all come from one side
            of the train/valid split, so valid never shows a ball train saw.
      not   not a ball: a negative again (never left out)
      hard  those of "not" to make a hard negative of (the spot's SPOT_CAP)
      hide  a ball that isn't a positive here: the spot's other frames, and,
            for a spot marked SPOT_PERSIST times or more (it lies there all
            game), every frame of that phone without a verdict on it, where
            it would otherwise sit in a tile as background.
    "unsure" and no verdict change nothing (still left out, see left_out).
    Only verdicts a person reviewed count ("by": "reviewed"): the drafts
    gave one object different verdicts from frame to frame."""
    plan = {fid: {"ball": [], "not": [], "hard": [], "hide": []} for fid, _, _ in frames}
    if KEEP_OFF_PITCH:
        return plan
    spots: list[dict] = []
    for fid, phone, split in sorted(frames):
        for v in verdicts.get(fid, []):
            if v.get("by") != "reviewed" or v["verdict"] not in ("ball", "not"):
                continue
            sp = next((sp for sp in spots if sp["phone"] == phone and abs(sp["x"] - v["x"]) <= SPOT_PX
                       and abs(sp["y"] - v["y"]) <= SPOT_PX), None)
            if sp is None:
                sp = {"phone": phone, "x": v["x"], "y": v["y"], "ball": [], "not": []}
                spots.append(sp)
            sp[v["verdict"]].append((fid, split, {"x": v["x"], "y": v["y"]}))

    def spread(rows: list) -> list:
        """SPOT_CAP of rows at most, evenly over them (they are in frame order)."""
        if len(rows) <= SPOT_CAP:
            return rows
        return [rows[i] for i in sorted({int(round(i)) for i in np.linspace(0, len(rows) - 1, SPOT_CAP)})]

    for sp in spots:
        if sp["ball"]:
            side = split_of(f"spot {sp['phone']} {sp['x']:.0f} {sp['y']:.0f}")
            if not any(split == side for _, split, _ in sp["ball"]):
                side = sp["ball"][0][1]
            chosen = spread([r for r in sp["ball"] if r[1] == side])
            for r in sp["ball"]:
                plan[r[0]]["ball" if any(r is c for c in chosen) else "hide"].append(r[2])
            if len(sp["ball"]) >= SPOT_PERSIST:
                seen = {r[0] for r in sp["ball"] + sp["not"]}
                at = {"x": float(np.mean([r[2]["x"] for r in sp["ball"]])),
                      "y": float(np.mean([r[2]["y"] for r in sp["ball"]]))}
                for fid, phone, _ in frames:
                    if phone == sp["phone"] and fid not in seen:
                        plan[fid]["hide"].append(at)
        chosen = spread(sp["not"])
        for r in sp["not"]:
            plan[r[0]]["not"].append(r[2])
            if any(r is c for c in chosen):
                plan[r[0]]["hard"].append(r[2])
    return plan


def with_verdicts(cands: list[dict], balls: list[dict], plan: dict | None):
    """A frame's balls and suggestions after its triage (plan_verdicts):
    (balls with the triaged ones added as spares, the suggestions that are
    not a ball, those of them to leave alone rather than make a hard
    negative of, extra spots no tile may show). A verdict belongs to the
    suggestion within VERDICT_PX of its pixel."""
    if not plan:
        return balls, [], [], []

    def at(c: dict, pts: list[dict]) -> bool:
        return any(np.hypot(p["x"] - c["x"], p["y"] - c["y"]) <= VERDICT_PX for p in pts)

    balls = list(balls) + [{"x": p["x"], "y": p["y"], "inPlay": False} for p in plan["ball"]]
    cleared = [c for c in cands if at(c, plan["not"])]
    unused = [c for c in cleared if not at(c, plan["hard"])]
    hide = [p for p in plan["hide"] if not any(at(c, [p]) for c in cands)]  # a suggestion there is left out anyway
    return balls, cleared, unused, hide


def tile_clear(x0: int, y0: int, pts: list[dict]) -> bool:
    """No left-out suggestion in the tile at (x0, y0) or just beyond its edges."""
    return not any(x0 - TILE_PAD <= c["x"] < x0 + TILE + TILE_PAD and y0 - TILE_PAD <= c["y"] < y0 + TILE + TILE_PAD
                   for c in pts)


def split_of(moment: str) -> str:
    h = int(hashlib.sha1(moment.encode()).hexdigest(), 16) % 1000
    return "valid" if h < VALID_FRAC * 1000 else "train"


def set_frames(items: list[dict], held) -> list[tuple[dict, str]]:
    """(frame, split) of a label set's trainable frames: moments in
    held-out time are left out, the rest split by moment."""
    out = []
    for it in items:
        if held.held(it["segment"], it["t"]):
            continue
        split = split_of(it["id"] if it.get("kind") == "pair" else it["id"].rsplit("_", 1)[0])
        out += [(f, split) for f in (it["frames"] if it.get("kind") == "pair" else [it])]
    return out


def main(rec: Path, out: Path, sets: list[str] = ("labels",)) -> None:
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
    counts = {s: {"tiles": 0, "balls": 0, "empty": 0, "hard": 0, "left_out": 0, "dropped": 0, "painted": 0,
                  "triaged_ball": 0, "triaged_not": 0, "hidden": 0} for s in coco}
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
    for name in sets:
        held.refuse_training_set(name)
        lab_dir = rec / name
        items = json.loads((lab_dir / "items.json").read_text())
        labels = json.loads((lab_dir / "labels.json").read_text())
        n_held += sum(held.held(it["segment"], it["t"]) for it in items)
        frames = set_frames(items, held)
        plan = plan_verdicts(load_verdicts(lab_dir), [(f["id"], f["phone"], split) for f, split in frames])
        for f, split in frames:
            v = labels.get(f["id"])
            if not v or v["status"] == "unsure":
                continue
            img = cv2.imread(str(lab_dir / f["image"]))
            h, w = img.shape[:2]
            balls, cleared, unused, hide = with_verdicts(f["candidates"], balls_of(v), plan[f["id"]])
            counts[split]["triaged_ball"] += len(balls) - len(balls_of(v))
            counts[split]["triaged_not"] += len(cleared) - len(unused)

            def clamp(x0, y0):
                return int(np.clip(x0, 0, w - TILE)), int(np.clip(y0, BAND_TOP - TILE // 2, h - TILE))

            # Unmarked suggestions off the pitch: perhaps another game's ball, so in no tile.
            skip = left_out(f["candidates"], balls,
                            lambda x, y: off_pitch(pose, *ground_xz(cams[f["phone"]], pose, x, y)), 20, cleared)
            counts[split]["left_out"] += len(skip)
            skip = skip + hide  # and a triaged ball that is no positive in this frame (plan_verdicts)
            counts[split]["hidden"] += len(hide)

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
                     and not any(c is k for k in skip) and not any(c is k for k in unused)]
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
              f" {c['dropped']} negative tiles dropped and {c['painted']} ball tiles painted to stay clear of them;"
              f" triaged (offpitch.py, {SPOT_CAP} a spot at most): {c['triaged_ball']} balls,"
              f" {c['triaged_not']} not a ball, {c['hidden']} more spots of a triaged ball kept out of tiles")


if __name__ == "__main__":
    rec = Path(sys.argv[1])
    main(rec, Path(sys.argv[2]) if len(sys.argv) > 2 else rec / "balldata", sys.argv[3:] or ["labels"])
