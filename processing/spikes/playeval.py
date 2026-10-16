#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Labelled moments for judging play following without watching videos.

    uv run processing/spikes/playeval.py make footage/rec3 <set> <segment> <start s> <clip t> [<clip t> ...]
    uv run processing/spikes/playeval.py score footage/rec3 <set> <path.json> [<path.json> ...]

"make" writes <recording>/playeval/<set>/: a stitched panorama of both
phones at each moment (clip times in seconds from `start` in `segment`;
a clip may run on into the next segments, as in follow.py), and
items.json. A reviewer then draws, on the page in playeval/ (server.py),
the "must-see" box for each moment: the part of the field a follow cam
has to show for the play to be followed. The page saves boxes.json, in
panorama degrees (yaw right +, pitch down +), so boxes don't depend on
any render.

"score" reads follow.py path files (path_<tag>.json, for the same clip)
and reports, per moment, how much of the box the virtual camera showed
and how wide it was, and a summary per path: moments fully captured, and
the median width (narrower is better when captured).

Footage shows people: everything stays under footage/ (not in git).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402

PANO_DEG = 0.04              # panorama resolution: zoomable, ~2 4K pixels per panorama pixel at the far side
PANO_YAW = (-100.0, 100.0)   # degrees
PANO_PITCH = (-6.0, 36.0)    # degrees, down is +
CAPTURED = 0.98              # a moment counts as captured when this much of its box is in view


def segment_time(devs: dict, seg: int, t: float) -> tuple[int, float]:
    """Segment and time within it on Main's clock, for `t` s from the start
    of `seg` (running on into the next segments)."""
    segs = devs["main"]["segments"]
    while t >= segs[seg]["durationMs"] / 1000:
        t -= segs[seg]["durationMs"] / 1000
        seg += 1
    return seg, t


def grab(rec: Path, devs: dict, report: dict, phone: int, seg: int, t: float, flip: bool) -> np.ndarray:
    """One full-resolution BGR frame of a phone at Main's time `t` in `seg`."""
    role = ("main", "secondary")[phone]
    local = t + (0.0 if phone == 0 else report["segments"][seg]["offsetS"])
    path = rec / devs[role]["segments"][seg]["objectKey"]
    W, H = follow.SRC_W, follow.SRC_H
    buf = next(follow.frame_stream(path, max(0.0, local), 1 / follow.FPS, W))
    bgr = cv2.cvtColor(buf.reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_NV12)
    return cv2.rotate(bgr, cv2.ROTATE_180) if flip else bgr


def make(rec: Path, name: str, seg0: int, start: float, times: list[float]) -> None:
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    out = rec / "playeval" / name
    (out / "pano").mkdir(parents=True, exist_ok=True)

    W = int((PANO_YAW[1] - PANO_YAW[0]) / PANO_DEG)
    H = int((PANO_PITCH[1] - PANO_PITCH[0]) / PANO_DEG)
    yaw = np.radians(PANO_YAW[0] + (np.arange(W) + 0.5) * PANO_DEG)
    pit = np.radians(PANO_PITCH[0] + (np.arange(H) + 0.5) * PANO_DEG)
    dirs = follow.direction(*np.meshgrid(yaw, pit))
    maps, edge = [], []
    for cam in (cm, cs):
        c = cam.at(follow.SRC_W)
        x, y, ok = c.project(dirs)
        maps.append((x.astype(np.float32), y.astype(np.float32)))
        edge.append(np.where(ok, np.minimum(np.minimum(x, c.w - 1 - x), np.minimum(y, c.h - 1 - y)), -1))
    use_main = edge[0] >= edge[1]
    seen = (edge[0] >= 0) | (edge[1] >= 0)

    items_path = out / "items.json"
    items = {it["id"]: it for it in json.loads(items_path.read_text())} if items_path.exists() else {}
    for clip_t in times:
        seg, t = segment_time(devs, seg0, start + clip_t)
        iid = f"s{seg0}_{start:g}_t{clip_t:g}"
        frames = [grab(rec, devs, report, ph, seg, t, flip[ph]) for ph in (0, 1)]
        warped = [cv2.remap(f, mx, my, cv2.INTER_LINEAR) for f, (mx, my) in zip(frames, maps)]
        pano = np.where(use_main[..., None], warped[0], warped[1])
        pano[~seen] = 0
        cv2.imwrite(str(out / "pano" / f"{iid}.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items[iid] = {"id": iid, "clipSegment": seg0, "clipStartS": start, "clipT": clip_t,
                      "frame": int(round(clip_t * follow.FPS)), "segment": seg, "segmentT": round(t, 3),
                      "image": f"pano/{iid}.jpg"}
        print(f"{iid}: segment {seg} at {t:.2f} s")
    meta = {"panoDeg": PANO_DEG, "yawDeg": PANO_YAW, "pitchDeg": PANO_PITCH, "width": W, "height": H}
    items_path.write_text(json.dumps(sorted(items.values(), key=lambda i: i["clipT"]), indent=1))
    (out / "pano.json").write_text(json.dumps(meta))
    print(f"{len(items)} moments in {out}")


def coverage(box: dict, yaw: float, pitch: float, hfov: float, n: int = 21) -> float:
    """Share of the box (degrees) inside follow.py's 16:9 virtual camera."""
    by = np.radians(np.linspace(box["yaw0"], box["yaw1"], n))
    bp = np.radians(np.linspace(box["pitch0"], box["pitch1"], n))
    d = follow.direction(*np.meshgrid(by, bp)).reshape(-1, 3)
    z = follow.direction(yaw, pitch)
    x = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    y = np.cross(z, x)
    dz = d @ z
    u, v = (d @ x) / np.maximum(dz, 1e-9), (d @ y) / np.maximum(dz, 1e-9)
    th = np.tan(hfov / 2)
    inside = (dz > 0) & (np.abs(u) <= th) & (np.abs(v) <= th * follow.OUT_H / follow.OUT_W)
    return float(inside.mean())


def score(rec: Path, name: str, paths: list[Path]) -> None:
    out = rec / "playeval" / name
    items = json.loads((out / "items.json").read_text())
    boxes = json.loads((out / "boxes.json").read_text()) if (out / "boxes.json").exists() else {}
    labelled = [it for it in items if it["id"] in boxes]
    if not labelled:
        raise SystemExit(f"no boxes yet in {out / 'boxes.json'}")
    print(f"{len(labelled)} moments with boxes ({sum(boxes[i['id']].get('reviewed', False) for i in labelled)}"
          " reviewed)")
    for p in paths:
        path = json.loads(p.read_text())
        rows, caught, widths = [], 0, []
        for it in labelled:
            if path.get("segment") != it["clipSegment"] or abs(path.get("startS", -1) - it["clipStartS"]) > 1e-6:
                raise SystemExit(f"{p.name} is a different clip from {it['id']}")
            k = it["frame"]
            yaw, pitch, hfov = (np.radians(path[key][k]) for key in ("yawDeg", "pitchDeg", "hfovDeg"))
            cov = coverage(boxes[it["id"]], yaw, pitch, hfov)
            caught += cov >= CAPTURED
            widths.append(np.degrees(hfov))
            rows.append(f"{it['clipT']:6.1f}s {100 * cov:4.0f}% at {np.degrees(hfov):3.0f} deg")
        print(f"\n{p.name}: {caught}/{len(labelled)} captured, median width {np.median(widths):.0f} deg")
        for r in rows:
            print("  " + r)


if __name__ == "__main__":
    a = sys.argv
    if len(a) > 6 and a[1] == "make":
        make(Path(a[2]), a[3], int(a[4]), float(a[5]), [float(x) for x in a[6:]])
    elif len(a) > 4 and a[1] == "score":
        score(Path(a[2]), a[3], [Path(x) for x in a[4:]])
    else:
        raise SystemExit(__doc__)
