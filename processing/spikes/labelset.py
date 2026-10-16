#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""Builds a set of frames to label the ball in, with suggestions.

    uv run processing/spikes/labelset.py footage/rec3 [count] [seed]
    uv run processing/spikes/labelset.py footage/rec3 pano
    uv run processing/spikes/labelset.py footage/rec3 clip <name> <segment> <start s> <seconds> [step s]
    uv run processing/spikes/labelset.py footage/rec3 pairs

Picks `count` random moments across all segments of the recording, and
for each one phone's frame (upright, full resolution), saved as a JPEG in
<recording>/labels/images/. RF-DETR suggests up to five ball candidates
per frame (low threshold, none on players' bodies above the feet), so most
labels are one keypress. Writes <recording>/labels/items.json; label with
processing/spikes/label/server.py.

"pano" adds a stitched panorama of both phones at each item's moment
(pano/<id>.jpg), so the ball is easier to find on the field, and
pano_maps.json: per phone, panorama <-> frame pixel lookups on coarse grids
(the cameras are fixed, so one table per phone serves every item).

"pairs" turns every unlabelled item into a moment with both phones'
frames (kind "pair"), so the ball is clicked once on the panorama and
labelled in whichever phone sees it. Each half keeps the single-frame id
scheme (<moment>_main, <moment>_secondary), so labels made before or during
the conversion stay attached. Labelled items are left as they are.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ball  # noqa: E402
import detect  # noqa: E402
import phase0  # noqa: E402

SUGGEST_THRESH = 0.1
MAX_SUGGESTIONS = 5


def candidates(model, bgr: np.ndarray):
    """Ball suggestions (x, y, conf) in full-frame pixels, best first."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    tl = detect.tiles(h, w)
    res = model.predict([rgb[y:y + detect.TILE, x:x + detect.TILE] for x, y in tl], threshold=SUGGEST_THRESH)
    rows = {detect.PERSON: [], detect.BALL: []}
    for (x, y), det in zip(tl, res):
        for box, c, sc in zip(det.xyxy, det.class_id, det.confidence):
            if c in rows and (c == detect.BALL or sc >= detect.THRESH[detect.PERSON]):
                rows[c].append(np.r_[box + [x, y, x, y], sc])
    people = [{"phone": 0, "box": list(b[:4])} for b in detect.nms(np.array(rows[detect.PERSON]))] \
        if rows[detect.PERSON] else []
    out = []
    if rows[detect.BALL]:
        for x1, y1, x2, y2, sc in detect.nms(np.array(rows[detect.BALL])):
            if ball.on_body({"phone": 0, "box": [x1, y1, x2, y2]}, people):
                continue
            out.append({"x": round(float((x1 + x2) / 2), 1), "y": round(float((y1 + y2) / 2), 1),
                        "conf": round(float(sc), 3)})
    # Tiny boxes from overlapping tiles barely overlap, so NMS can keep the
    # same spot twice: merge suggestions within 12 px of a better one.
    kept = []
    for c in sorted(out, key=lambda c: -c["conf"]):
        if all(np.hypot(c["x"] - k["x"], c["y"] - k["y"]) >= 12 for k in kept):
            kept.append(c)
    return kept[:MAX_SUGGESTIONS]


PANO_DEG = 0.06              # panorama resolution
PANO_YAW = (-100.0, 100.0)   # degrees
PANO_PITCH = (-6.0, 36.0)    # degrees, down is +
PANO_STEP = 6                # panorama -> frame lookup every this many pixels
FRAME_STEP = 40              # frame -> panorama lookup every this many pixels


def panoramas(rec: Path, out: Path | None = None) -> None:
    """Stitched panoramas for every item, and the lookup tables the page
    uses to go between a panorama click and the item's own frame."""
    import follow

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    out = out or rec / "labels"
    (out / "pano").mkdir(exist_ok=True)
    W = int((PANO_YAW[1] - PANO_YAW[0]) / PANO_DEG)
    H = int((PANO_PITCH[1] - PANO_PITCH[0]) / PANO_DEG)
    yaw = np.radians(PANO_YAW[0] + np.arange(W) * PANO_DEG)
    pit = np.radians(PANO_PITCH[0] + np.arange(H) * PANO_DEG)
    dirs = follow.direction(*np.meshgrid(yaw, pit))
    src_w = 1920  # panorama sampled from half-resolution frames
    cams = {"main": cm, "secondary": cs}
    maps, edge = {}, {}
    for role, cam in cams.items():
        c = cam.at(src_w)
        x, y, ok = c.project(dirs)
        maps[role] = (x, y, ok)
        edge[role] = np.where(ok, np.minimum(np.minimum(x, c.w - 1 - x), np.minimum(y, c.h - 1 - y)), -1)
    use_main = edge["main"] >= edge["secondary"]
    seen = (edge["main"] >= 0) | (edge["secondary"] >= 0)
    outline = {}
    for role in cams:
        m = maps[role][2].astype(np.uint8)
        outline[role] = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0

    # Lookup tables at full frame resolution (3840 wide, upright).
    scale = 3840 / src_w
    tables = {"panoWidth": W, "panoHeight": H, "panoStep": PANO_STEP, "frameStep": FRAME_STEP}
    for role, cam in cams.items():
        x, y, ok = maps[role]
        fx = np.where(ok, x * scale, -1)[::PANO_STEP, ::PANO_STEP]
        fy = np.where(ok, y * scale, -1)[::PANO_STEP, ::PANO_STEP]
        c4 = cam.at(3840)
        gx, gy = np.meshgrid(np.arange(0, c4.w + 1, FRAME_STEP), np.arange(0, c4.h + 1, FRAME_STEP))
        d = c4.unproject(np.c_[gx.ravel(), gy.ravel()].astype(float))
        py = (np.degrees(np.arctan2(d[:, 0], d[:, 2])) - PANO_YAW[0]) / PANO_DEG
        pp = (np.degrees(np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))) - PANO_PITCH[0]) / PANO_DEG
        tables[role] = {
            "toFrameX": np.round(fx, 1).tolist(), "toFrameY": np.round(fy, 1).tolist(),
            "toPanoX": np.round(py.reshape(gx.shape), 1).tolist(),
            "toPanoY": np.round(pp.reshape(gx.shape), 1).tolist(),
        }
    (out / "pano_maps.json").write_text(json.dumps(tables))

    items = json.loads((out / "items.json").read_text())
    for k, it in enumerate(items):
        path = out / "pano" / f"{it['id']}.jpg"
        if it.get("kind") == "pair" and "clipT" in it:  # clip sets: use the clip's own segment time
            pass
        if path.exists():
            continue
        seg, t = it["segment"], it["t"]
        a = phase0.frames_at(rec / devs["main"]["segments"][seg]["objectKey"], t, 1, 30, src_w, flip[0])[0]
        b = phase0.frames_at(rec / devs["secondary"]["segments"][seg]["objectKey"],
                             t + report["segments"][seg]["offsetS"], 1, 30, src_w, flip[1])[0]
        wa = cv2.remap(a, maps["main"][0], maps["main"][1], cv2.INTER_LINEAR)
        wb = cv2.remap(b, maps["secondary"][0], maps["secondary"][1], cv2.INTER_LINEAR)
        pano = np.where(use_main[..., None], wa, wb)
        pano[~seen] = 0
        if it.get("kind") != "pair":
            pano[outline[it["phone"]]] = (255, 220, 0)  # this item's phone, outlined
        cv2.imwrite(str(path), pano, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(items)} panoramas", flush=True)
    print(f"{len(items)} panoramas, {W}x{H}")


def pairs(rec: Path) -> None:
    import rfdetr

    out = rec / "labels"
    items_path = out / "items.json"
    items = json.loads(items_path.read_text())
    labels_path = out / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = {"main": report["upsideDown"]["main"], "secondary": report["upsideDown"]["secondary"]}
    (out / "items.single.json").write_text(json.dumps(items, indent=1))  # backup

    model = None
    new = []
    for it in items:
        if it.get("kind") == "pair" or it["id"] in labels:
            new.append(it)
            continue
        base = it["id"].rsplit("_", 1)[0]
        halves = {it["phone"]: {k: it[k] for k in ("id", "phone", "image", "width", "height", "candidates")}}
        other = "main" if it["phone"] == "secondary" else "secondary"
        oid = f"{base}_{other}"
        seg, t = it["segment"], it["t"]
        off = 0.0 if other == "main" else report["segments"][seg]["offsetS"]
        img = phase0.frames_at(rec / devs[other]["segments"][seg]["objectKey"], t + off, 1, 30, 3840, flip[other])[0]
        cv2.imwrite(str(out / "images" / f"{oid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if model is None:
            model = rfdetr.RFDETRMedium()
        halves[other] = {"id": oid, "phone": other, "image": f"images/{oid}.jpg", "width": img.shape[1],
                         "height": img.shape[0], "candidates": candidates(model, img)}
        new.append({"id": base, "kind": "pair", "segment": seg, "t": t,
                    "frames": [halves["secondary"], halves["main"]]})  # left to right, as in the panorama
        if len(new) % 25 == 0:
            print(f"{len(new)}/{len(items)} items", flush=True)
    items_path.write_text(json.dumps(new, indent=1))
    n_pairs = sum(it.get("kind") == "pair" for it in new)
    print(f"{len(new)} items: {n_pairs} moments with both phones, {len(new) - n_pairs} single frames (labelled)")
    panoramas(rec)


def clip(rec: Path, name: str, seg0: int, start: float, seconds: float, step: float) -> None:
    """A set of both-phone moments every `step` s along a clip (clip times
    as in follow.py: from `start` s into `seg0`, running on into the next
    segments), in <recording>/labels_<name>/, for ball ground truth along
    the clip. Suggestions come from ballclip.py's candidates for the clip
    when they exist (the fine-tuned detector, with balljudge.py's score),
    else the stock detector."""
    import playeval

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = {"main": report["upsideDown"]["main"], "secondary": report["upsideDown"]["secondary"]}
    out = rec / f"labels_{name}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    cands_path = rec / "detect" / f"ball_{seg0}_{start:g}_{seconds:g}.json"
    clip_cands = json.loads(cands_path.read_text()) if cands_path.exists() else None
    model = None if clip_cands else __import__("rfdetr").RFDETRMedium()
    items = []
    for clip_t in np.arange(0.0, seconds, step):
        seg, t = playeval.segment_time(devs, seg0, start + clip_t)
        # A phone's file can end just before a segment switch: step back a little.
        for role in ("secondary", "main"):
            off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
            while t + off > devs[role]["segments"][seg]["durationMs"] / 1000 - 1.5:
                t, clip_t = t - 0.2, clip_t - 0.2
        base = f"c{seg0}_{start:g}_t{clip_t:06.2f}"
        halves = {}
        for role in ("secondary", "main"):
            iid = f"{base}_{role}"
            off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
            frames = []
            for back in range(8):  # the decoder returns nothing right at a file's end: step back
                frames = phase0.frames_at(rec / devs[role]["segments"][seg]["objectKey"], t + off - 0.25 * back,
                                          1, 30, 3840, flip[role])
                if frames:
                    break
            if not frames:
                raise SystemExit(f"no frame for {role} segment {seg} at {t + off:.2f} s (clip {clip_t:.1f} s)")
            if back:
                print(f"{iid}: stepped back {0.25 * back:.2f} s at the end of the file", flush=True)
            img = frames[0]
            cv2.imwrite(str(out / "images" / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if clip_cands and int(round(clip_t * clip_cands["fps"])) < len(clip_cands["frames"]):
                k = int(round(clip_t * clip_cands["fps"]))
                phone = 0 if role == "main" else 1
                cs = [{"x": round((d["box"][0] + d["box"][2]) / 2, 1), "y": round((d["box"][1] + d["box"][3]) / 2, 1),
                       "conf": d["conf"], "judge": d.get("judge")}
                      for d in clip_cands["frames"][k]["detections"] if d["phone"] == phone]
                cs = sorted(cs, key=lambda c: -(c["conf"] * (c["judge"] if c["judge"] is not None else 1)))[:MAX_SUGGESTIONS]
            else:
                if model is None:
                    model = __import__("rfdetr").RFDETRMedium()
                cs = candidates(model, img)
            halves[role] = {"id": iid, "phone": role, "image": f"images/{iid}.jpg", "width": img.shape[1],
                            "height": img.shape[0], "candidates": cs}
        items.append({"id": base, "kind": "pair", "segment": seg, "t": round(t, 3), "clipT": float(clip_t),
                      "frames": [halves["secondary"], halves["main"]]})
        if len(items) % 10 == 0:
            print(f"{len(items)} moments", flush=True)
    (out / "items.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} moments in {out}")
    panoramas(rec, out)


def main(rec: Path, count: int, seed: int) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = {"main": report["upsideDown"]["main"], "secondary": report["upsideDown"]["secondary"]}
    out = rec / "labels"
    (out / "images").mkdir(parents=True, exist_ok=True)
    items_path = out / "items.json"
    items = json.loads(items_path.read_text()) if items_path.exists() else []
    done = {it["id"] for it in items}

    rng = np.random.default_rng(seed)
    nseg = len(report["segments"])
    picks = []
    for _ in range(count):
        seg = int(rng.integers(nseg))
        dur = min(devs[r]["segments"][seg]["durationMs"] for r in ("main", "secondary")) / 1000
        t = float(rng.uniform(5, max(6, dur - 5)))
        role = "main" if rng.random() < 0.5 else "secondary"
        picks.append((seg, round(t, 2), role))

    model = rfdetr.RFDETRMedium()
    for k, (seg, t, role) in enumerate(picks):
        iid = f"s{seg:02d}_t{t:07.2f}_{role}"
        if iid in done:
            continue
        off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
        img = phase0.frames_at(rec / devs[role]["segments"][seg]["objectKey"], t + off, 1, 30, 3840, flip[role])[0]
        cv2.imwrite(str(out / "images" / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items.append({"id": iid, "image": f"images/{iid}.jpg", "segment": seg, "t": t, "phone": role,
                      "width": img.shape[1], "height": img.shape[0], "candidates": candidates(model, img)})
        if (k + 1) % 25 == 0 or k + 1 == len(picks):
            items_path.write_text(json.dumps(items, indent=1))
            print(f"{k + 1}/{len(picks)} frames", flush=True)
    items_path.write_text(json.dumps(items, indent=1))
    with_c = sum(bool(it["candidates"]) for it in items)
    print(f"{len(items)} frames, {with_c} with at least one suggestion")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "pano":
        panoramas(Path(sys.argv[1]))
    elif len(sys.argv) > 2 and sys.argv[2] == "pairs":
        pairs(Path(sys.argv[1]))
    elif len(sys.argv) > 6 and sys.argv[2] == "clip":
        clip(Path(sys.argv[1]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),
             float(sys.argv[7]) if len(sys.argv) > 7 else 2.0)
    else:
        main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 300,
             int(sys.argv[3]) if len(sys.argv) > 3 else 1)
