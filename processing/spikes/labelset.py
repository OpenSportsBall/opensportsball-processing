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
    uv run processing/spikes/labelset.py footage/rec5-game ball <candidates file in detect/> [count] [phone] [seed]

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

"ball" picks `count` frames of one phone ("main" by default) where the
ball is likely in view, from a ballwin.py candidates file (detect/...,
e.g. a .nojudge.json one: see below) and the ball path ballpath.py makes
from it: half where the path rests on that phone's detection (its
candidates there are the suggestions), half where the path bridges a gap
the detector missed (often the hard cases: small, blurred, half hidden),
with the path's position projected into the frame as the first suggestion
("fromPath"). Frames are at least BALL_GAP_S apart. For a new ball the
models haven't seen (the 2026-10-28 game's orange one, which the judge
rejects): its candidates without the judge still find it.

"pairs" turns every unlabelled item into a moment with both phones'
frames (kind "pair"), so the ball is clicked once on the panorama and
labelled in whichever phone sees it. Each half keeps the single-frame id
scheme (<moment>_main, <moment>_secondary), so labels made before or during
the conversion stay attached. Labelled items are left as they are.

Every frame is grabbed at its item's time ("t", plus the audio offset for
the secondary phone) through `grab`, which decodes with a preroll; a plain
seek (phase0.frames_at) returns a frame up to 0.9 s later than asked. A
frame's "dtS" is its image's offset from that time: 0.0 for sets made
since 2026-10-18; the earlier clip sets carry the offset ballheat.py
measured. The random set (labels/) was grabbed the old way and has no
dtS: its labels sit on its images, so nothing reads its times except
motion.py, which seeks the same way and so lands on the same frame.
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


def grab(path: Path, t: float, flip: bool, width: int = 3840) -> np.ndarray | None:
    """The frame at file time `t`, upright. Decoded with a preroll
    (phase0.frame_at): a plain seek returns a frame up to 0.9 s late. Near
    a file's end the decoder returns nothing, so the time steps back in
    0.25 s steps (up to 2 s); None when there is still no frame."""
    for back in range(9):
        if t - 0.25 * back < 0:
            break  # the file's start: nothing earlier to try
        for pre in (phase0.FRAME_PREROLL_S, 3.0):
            img = phase0.frame_at(path, t - 0.25 * back, width, flip, pre)
            if img is not None:
                if back:
                    print(f"{path.name} {t:.2f} s: stepped back {0.25 * back:.2f} s at the end of the file", flush=True)
                return img
    return None


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
    devs = phase0.paired_devices(manifest, report)
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
        if path.exists():
            continue
        seg, t = it["segment"], it["t"]
        a = grab(rec / devs["main"]["segments"][seg]["objectKey"], t, flip[0], src_w)
        # Phones with different segment lengths: the Secondary file that has this moment (phase0's pieces).
        piece = next((p for p in report["segments"][seg].get("secondaryPieces", [])
                      if 0 <= t + p["offsetS"] < p["durationMs"] / 1000), None)
        b = grab(rec / piece["objectKey"], t + piece["offsetS"], flip[1], src_w) if piece else \
            grab(rec / devs["secondary"]["segments"][seg]["objectKey"], t + report["segments"][seg]["offsetS"],
                 flip[1], src_w)
        if a is None or b is None:
            print(f"{it['id']}: no frame for the panorama at {t:.2f} s, skipped", flush=True)
            continue
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
    devs = phase0.paired_devices(manifest, report)
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
        img = grab(rec / devs[other]["segments"][seg]["objectKey"], t + off, flip[other])
        if img is None:
            print(f"{oid}: no frame at {t + off:.2f} s, left as a single frame", flush=True)
            new.append(it)
            continue
        cv2.imwrite(str(out / "images" / f"{oid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if model is None:
            model = rfdetr.RFDETRMedium()
        halves[other] = {"id": oid, "phone": other, "image": f"images/{oid}.jpg", "width": img.shape[1],
                         "height": img.shape[0], "candidates": candidates(model, img), "dtS": 0.0}
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
    else the stock detector. Frames are grabbed at the asked time (see
    `grab`) and each carries "dtS": 0.0, its image's offset from the
    moment's time; sets made before 2026-10-18 were grabbed up to 0.9 s
    late and have their measured offset there (ballheat.py <set> dt)."""
    import playeval

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
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
            img = grab(rec / devs[role]["segments"][seg]["objectKey"], t + off, flip[role])
            if img is None:
                raise SystemExit(f"no frame for {role} segment {seg} at {t + off:.2f} s (clip {clip_t:.1f} s)")
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
                            "height": img.shape[0], "candidates": cs, "dtS": 0.0}
        items.append({"id": base, "kind": "pair", "segment": seg, "t": round(t, 3), "clipT": float(clip_t),
                      "frames": [halves["secondary"], halves["main"]]})
        if len(items) % 10 == 0:
            print(f"{len(items)} moments", flush=True)
    (out / "items.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} moments in {out}")
    panoramas(rec, out)


BALL_GAP_S = 1.5  # "ball" set frames at least this far apart in game time


def ball_set(rec: Path, cands_name: str, count: int, role: str, seed: int) -> None:
    import ball as B
    import ballpath
    import follow
    import pitch as P

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = {"main": report["upsideDown"]["main"], "secondary": report["upsideDown"]["secondary"]}
    phone = 0 if role == "main" else 1
    if phone and any("secondaryPieces" in s for s in report["segments"]):
        sys.exit("ball: secondary frames of a mixed-length recording aren't supported; use main")
    cam = follow.setup_cameras(rec, report, devs, [flip["main"], flip["secondary"]])[phone].at(3840)
    pose = P.load_pose(rec)
    data = json.loads((rec / "detect" / cands_name).read_text())
    fps, frames = data["fps"], data["frames"]
    n = len(frames)
    cands = [[d for d in f["detections"] if B.usable(d, pose)] for f in frames]
    spots = B.static_spots(cands, fps, pose=pose)
    keep = [[d for d in fr if all(np.hypot(*(np.array(d["xz"]) - sp)) > B.STATIC_RADIUS for sp in spots)]
            for fr in cands]
    est, src = ballpath.track(keep, [np.zeros((0, 2))] * n, n, fps)

    def path_px(i):
        yaw, pitch = pose.angles(np.array([est[i]], float))[0]
        x, y, ok = cam.project(follow.direction(np.radians(yaw), np.radians(pitch)))
        return (float(x), float(y)) if ok else None

    def chosen(i):
        return next((c for c in src[i] if isinstance(c, dict)), None) if src[i] is not None else None

    seen = [i for i in range(n) if chosen(i) is not None and chosen(i)["phone"] == phone]
    bridged = [i for i in range(n) if est[i] is not None and src[i] is None and path_px(i) is not None]
    rng = np.random.default_rng(seed)
    picks: list[tuple[int, str]] = []

    def take(pool, kind, want):
        got = 0
        for i in rng.permutation(pool):
            if got >= want:
                break
            if all(abs(frames[i]["t"] - frames[j]["t"]) >= BALL_GAP_S for j, _ in picks):
                picks.append((int(i), kind))
                got += 1
        return got

    half = take(bridged, "bridged", count // 2)
    take(seen, "detected", count - half)
    if len(picks) < count:
        take(bridged, "bridged", count - len(picks))
    print(f"{len(seen)} frames with a {role} detection on the path, {len(bridged)} bridged in {role}'s view;"
          f" picked {sum(k == 'detected' for _, k in picks)} + {sum(k == 'bridged' for _, k in picks)}", flush=True)

    out = rec / "labels"
    (out / "images").mkdir(parents=True, exist_ok=True)
    items_path = out / "items.json"
    items = json.loads(items_path.read_text()) if items_path.exists() else []
    done = {it["id"] for it in items}
    segs = devs["main"]["segments"]
    for k, (i, kind) in enumerate(sorted(picks)):
        seg, t = 0, frames[i]["t"]
        while seg < len(segs) - 1 and t >= segs[seg]["durationMs"] / 1000:
            t -= segs[seg]["durationMs"] / 1000
            seg += 1
        t = round(t, 2)
        iid = f"b{seg:02d}_t{t:07.2f}_{role}"
        if iid in done:
            continue
        print(f"{k + 1}/{len(picks)} grabbing {iid}", flush=True)
        off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
        img = grab(rec / devs[role]["segments"][seg]["objectKey"], t + off, flip[role])
        if img is None:
            print(f"{iid}: no frame at {t + off:.2f} s, skipped", flush=True)
            continue
        cv2.imwrite(str(out / "images" / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        sugg = [{"x": round((d["box"][0] + d["box"][2]) / 2, 1), "y": round((d["box"][1] + d["box"][3]) / 2, 1),
                 "conf": round(d["conf"], 3)} for d in sorted(frames[i]["detections"], key=lambda d: -d["conf"])
                if d["phone"] == phone][:MAX_SUGGESTIONS]
        if kind == "bridged":
            x, y = path_px(i)
            sugg = [{"x": round(x, 1), "y": round(y, 1), "conf": 0.0, "fromPath": True}] + sugg[:MAX_SUGGESTIONS - 1]
        items.append({"id": iid, "image": f"images/{iid}.jpg", "segment": seg, "t": t, "phone": role,
                      "width": img.shape[1], "height": img.shape[0], "candidates": sugg, "dtS": 0.0,
                      "pick": kind, "source": cands_name})
        items_path.write_text(json.dumps(items, indent=1))  # every frame: a long run can be stopped
        print(f"{k + 1}/{len(picks)} {iid}", flush=True)
    items_path.write_text(json.dumps(items, indent=1))
    print(f"{len(items)} frames in {out}")


def main(rec: Path, count: int, seed: int) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
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
        img = grab(rec / devs[role]["segments"][seg]["objectKey"], t + off, flip[role])
        if img is None:
            print(f"{iid}: no frame at {t + off:.2f} s, skipped", flush=True)
            continue
        cv2.imwrite(str(out / "images" / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items.append({"id": iid, "image": f"images/{iid}.jpg", "segment": seg, "t": t, "phone": role,
                      "width": img.shape[1], "height": img.shape[0], "candidates": candidates(model, img),
                      "dtS": 0.0})
        if (k + 1) % 25 == 0 or k + 1 == len(picks):
            items_path.write_text(json.dumps(items, indent=1))
            print(f"{k + 1}/{len(picks)} frames", flush=True)
    items_path.write_text(json.dumps(items, indent=1))
    with_c = sum(bool(it["candidates"]) for it in items)
    print(f"{len(items)} frames, {with_c} with at least one suggestion")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "pano":
        panoramas(Path(sys.argv[1]))
    elif len(sys.argv) > 3 and sys.argv[2] == "ball":
        ball_set(Path(sys.argv[1]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 100,
                 sys.argv[5] if len(sys.argv) > 5 else "main", int(sys.argv[6]) if len(sys.argv) > 6 else 1)
    elif len(sys.argv) > 2 and sys.argv[2] == "pairs":
        pairs(Path(sys.argv[1]))
    elif len(sys.argv) > 6 and sys.argv[2] == "clip":
        clip(Path(sys.argv[1]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),
             float(sys.argv[7]) if len(sys.argv) > 7 else 2.0)
    else:
        main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 300,
             int(sys.argv[3]) if len(sys.argv) > 3 else 1)
