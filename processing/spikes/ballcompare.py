#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""Off-the-shelf vs fine-tuned ball detector on the labelled full frames.

    uv run processing/spikes/ballcompare.py footage/rec3 <checkpoint.pth>

Both models run the same way (detect.py's tiles of the 4K frame, NMS,
suggestions within 12 px merged), with no cap on suggestions. Scored per
split (balldata.py's moments: "valid" was held back from training):
share of match balls found, of all balls found, and wrong suggestions per
frame, at several confidence cut-offs. Detections are cached in
<recording>/labels/compare.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import balldata  # noqa: E402
import detect  # noqa: E402

FLOOR = 0.05
CUTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
HIT_PX = 15


def suggestions(model, rgb, ball_ids) -> list[dict]:
    h, w = rgb.shape[:2]
    tl = detect.tiles(h, w)
    res = model.predict([rgb[y:y + detect.TILE, x:x + detect.TILE] for x, y in tl], threshold=FLOOR)
    rows = [np.r_[box + [x, y, x, y], sc] for (x, y), det in zip(tl, res)
            for box, c, sc in zip(det.xyxy, det.class_id, det.confidence) if ball_ids is None or c in ball_ids]
    if not rows:
        return []
    out = [{"x": float((x1 + x2) / 2), "y": float((y1 + y2) / 2), "conf": float(sc)}
           for x1, y1, x2, y2, sc in detect.nms(np.array(rows))]
    kept = []
    for c in sorted(out, key=lambda c: -c["conf"]):
        if all(np.hypot(c["x"] - k["x"], c["y"] - k["y"]) >= 12 for k in kept):
            kept.append({k: round(v, 3) for k, v in c.items()})
    return kept


def main(rec: Path, ckpt: Path) -> None:
    import rfdetr

    lab_dir = rec / "labels"
    items = json.loads((lab_dir / "items.json").read_text())
    labels = json.loads((lab_dir / "labels.json").read_text())
    frames = []
    for it in items:
        moment = it["id"] if it.get("kind") == "pair" else it["id"].rsplit("_", 1)[0]
        for f in (it["frames"] if it.get("kind") == "pair" else [it]):
            v = labels.get(f["id"])
            if v and v["status"] != "unsure":
                frames.append((balldata.split_of(moment), f, balldata.balls_of(v)))
    frames.sort(key=lambda r: r[0] != "valid")  # held-back frames first
    cache_path = lab_dir / "compare.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    models = {"stock": (lambda: rfdetr.RFDETRMedium(), {detect.BALL}),
              "tuned": (lambda: rfdetr.RFDETRMedium(pretrain_weights=str(ckpt)), None)}
    for name, (make, ids) in models.items():
        todo = [f for _, f, _ in frames if f["id"] not in cache.get(name, {})]
        if not todo:
            continue
        model = make()
        cache.setdefault(name, {})
        t0 = time.time()
        for k, f in enumerate(todo):
            rgb = cv2.cvtColor(cv2.imread(str(lab_dir / f["image"])), cv2.COLOR_BGR2RGB)
            cache[name][f["id"]] = suggestions(model, rgb, ids)
            if (k + 1) % 20 == 0:
                cache_path.write_text(json.dumps(cache))
                print(f"{name}: {k + 1}/{len(todo)} frames, {(time.time() - t0) / 60:.1f} min", flush=True)
        cache_path.write_text(json.dumps(cache))
    report(frames, cache)


def report(frames, cache) -> None:
    def near(s, b):
        return np.hypot(s["x"] - b["x"], s["y"] - b["y"]) < HIT_PX

    for split in ("valid", "train"):
        rows = [(f, balls) for s, f, balls in frames if s == split]
        n_match = sum(any(b["inPlay"] for b in balls) for _, balls in rows)
        n_any = sum(len(balls) for _, balls in rows)
        print(f"\n{split}: {len(rows)} frames, {n_match} match balls, {n_any} balls in all")
        print(f"{'cut':>5} | {'stock: match  any  wrong/frame':>30} | {'tuned: match  any  wrong/frame':>30}")
        for cut in CUTS:
            cols = []
            for name in ("stock", "tuned"):
                m = a = wrong = 0
                for f, balls in rows:
                    sug = [s for s in cache[name][f["id"]] if s["conf"] >= cut]
                    m += any(b["inPlay"] and any(near(s, b) for s in sug) for b in balls)
                    a += sum(any(near(s, b) for s in sug) for b in balls)
                    wrong += sum(not any(near(s, b) for b in balls) for s in sug)
                cols.append(f"{100 * m / max(1, n_match):11.0f}% {100 * a / max(1, n_any):4.0f}% {wrong / len(rows):11.2f}")
            print(f"{cut:5.1f} | {cols[0]:>30} | {cols[1]:>30}")
        # Where is the match ball in each model's ranking?
        for name in ("stock", "tuned"):
            ranks = []
            for f, balls in rows:
                mb = [b for b in balls if b["inPlay"]]
                if not mb:
                    continue
                sug = cache[name][f["id"]]
                r = next((i for i, s in enumerate(sug) if near(s, mb[0])), None)
                ranks.append(r)
            top = {k: 100 * sum(r is not None and r < k for r in ranks) / max(1, len(ranks)) for k in (1, 3, 5)}
            print(f"{name}: match ball is the top suggestion {top[1]:.0f}%, in the top 3 {top[3]:.0f}%,"
                  f" top 5 {top[5]:.0f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
