#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "scikit-learn>=1.4", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""Which players are on which team: kit colour, clustered per game.

    uv run processing/spikes/teams.py footage/rec3 <set> [components]

People are found with the stock RF-DETR (class "person") on the full
frames of a clip label set (both phones). For each, the torso patch (the
middle of the upper body) is white-balanced against the turf beside the
player (floodlights cast differently across the pitch), and described by
its colour: mean Lab chroma, mean saturation and value, and a
saturation-weighted hue histogram. A Gaussian mixture over all the
patches gives the kits: the two largest components are the outfield
teams, the small distinct ones are keepers and the referee.

Writes <recording>/teams/<set>/: people.json (per frame: boxes, features,
component), model.npz (the mixture, to assign new detections), and
sheet.jpg (torso patches grouped by component, to judge purity by eye).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import detect  # noqa: E402
import follow  # noqa: E402
import pitch as P  # noqa: E402

PERSON_THRESH = 0.4
MIN_HEIGHT_PX = 28         # people shorter than this are skipped entirely
STAT_MIN_HEIGHT_PX = 45    # ... and only taller ones shape the clusters and the headcounts (a readable torso)
TORSO = (0.25, 0.75, 0.18, 0.50)   # x0, x1, y0, y1 as fractions of the person box
HUE_BINS = 12
KIT_MIN_PER_MOMENT = 2.5   # a kit component shows this many people on the pitch per labelled moment (both phones)
SOLO_MAX_PER_MOMENT = 1.6  # a keeper or referee component: about one person on the pitch per moment
SOLO_MIN_SAT = 25.0        # ... with a saturated colour (mean saturation, 0-100)
KIT_MIN_SAT = 24.0         # a kit component is coloured (grass-dominated far torsos, skin and bench are not) ...
KIT_MAX_SPREAD = 15.0      # ... and tight in chroma (Lab a, b spread within the component)



def person_boxes(model, rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    tl = detect.tiles(h, w)
    res = model.predict([rgb[y:y + detect.TILE, x:x + detect.TILE] for x, y in tl], threshold=PERSON_THRESH)
    rows = [np.r_[box + [x, y, x, y], sc] for (x, y), det in zip(tl, res)
            for box, c, sc in zip(det.xyxy, det.class_id, det.confidence) if c == detect.PERSON]
    if not rows:
        return np.zeros((0, 5))
    return detect.nms(np.array(rows))


def turf_reference(bgr: np.ndarray, box) -> np.ndarray | None:
    """Mean colour of the turf just below and beside the player."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    h = y2 - y1
    strips = [bgr[y2:min(bgr.shape[0], y2 + h // 4), max(0, x1 - (x2 - x1)):min(bgr.shape[1], x2 + (x2 - x1))]]
    px = np.concatenate([s.reshape(-1, 3) for s in strips if s.size]) if any(s.size for s in strips) else None
    if px is None or len(px) < 50:
        return None
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 95) & (hsv[:, 1] > 40)  # green-ish
    if grass.sum() < 30:
        return None
    return px[grass].mean(0)


def features(bgr: np.ndarray, box, turf_global: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    if h < MIN_HEIGHT_PX:
        return None
    tx0, tx1, ty0, ty1 = (x1 + int(TORSO[0] * w), x1 + int(TORSO[1] * w), y1 + int(TORSO[2] * h), y1 + int(TORSO[3] * h))
    patch = bgr[ty0:ty1, tx0:tx1].astype(np.float32)
    if patch.size == 0:
        return None
    local = turf_reference(bgr, box)
    if local is not None:
        patch = np.clip(patch * (turf_global / np.maximum(local, 1)), 0, 255)
    p8 = patch.astype(np.uint8)
    hsv = cv2.cvtColor(p8, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(p8, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    sat = hsv[:, 1] / 255
    hist = np.histogram(hsv[:, 0], bins=HUE_BINS, range=(0, 180), weights=sat)[0]
    hist = hist / max(hist.sum(), 1e-6)
    f = np.r_[lab[:, 1].mean() - 128, lab[:, 2].mean() - 128, sat.mean() * 100, hsv[:, 2].mean() / 255 * 100,
              hist * 100]
    return f, cv2.resize(p8, (24, 32))


def sides(out: Path, people, X, comp, n_moments: int, k: int) -> None:
    """Which kit components make up each side: headcount. A side has a
    fixed number of players, so the two-way split of the kit components
    whose sides are most equal in number, moment by moment, is the teams
    (a kit worn by part of a team adds up with the rest of its team; a
    genuine third group would leave a side short). Components with about
    one person per moment and a saturated colour are keeper or referee
    candidates; the rest is not a kit."""
    from itertools import combinations

    comp = np.asarray(comp)
    moments = sorted({p["frame"].rsplit("_", 1)[0] for p in people})
    m_index = {m: i for i, m in enumerate(moments)}
    counts = np.zeros((k, len(moments)))
    for p, c in zip(people, comp):
        if p["onPitch"] and p["box"][3] - p["box"][1] >= STAT_MIN_HEIGHT_PX:  # inside the lines, torso readable
            counts[c, m_index[p["frame"].rsplit("_", 1)[0]]] += 1
    per_moment = counts.mean(1)
    sat = np.array([X[comp == c, 2].mean() if (comp == c).any() else 0 for c in range(k)])
    spread = np.array([np.sqrt(X[comp == c, :2].var(0).sum()) if (comp == c).sum() > 1 else 99 for c in range(k)])
    kits = [c for c in range(k) if per_moment[c] >= KIT_MIN_PER_MOMENT and sat[c] >= KIT_MIN_SAT and spread[c] <= KIT_MAX_SPREAD]
    solos = [c for c in range(k) if 0.4 <= per_moment[c] <= SOLO_MAX_PER_MOMENT and sat[c] >= SOLO_MIN_SAT and c not in kits]
    print("components (on-pitch people): " + "; ".join(f"{c}: {per_moment[c]:.1f}/moment, sat {sat[c]:.0f}, spread {spread[c]:.0f}" for c in range(k)))
    print(f"kit components {kits}; keeper/referee candidates {solos}; other {[c for c in range(k) if c not in kits + solos]}")
    best = None
    for r in range(1, len(kits)):
        for a in combinations(kits, r):
            if kits[0] not in a:  # each split once
                continue
            b = [c for c in kits if c not in a]
            ca, cb = counts[list(a)].sum(0), counts[b].sum(0)
            cost = float(np.mean(np.abs(ca - cb)))
            if best is None or cost < best[0]:
                best = (cost, list(a), b, ca, cb)
    if best is None:
        print("sides: fewer than two kit components")
        return
    cost, a, b, ca, cb = best
    # Confidence: how much worse the next-best split is, relative.
    others = []
    for r in range(1, len(kits)):
        for a2 in combinations(kits, r):
            if kits[0] in a2 and sorted(a2) != sorted(a):
                b2 = [c for c in kits if c not in a2]
                others.append(float(np.mean(np.abs(counts[list(a2)].sum(0) - counts[b2].sum(0)))))
    margin = (min(others) - cost) / max(cost, 1e-6) if others else float("inf")
    print(f"sides: A = components {a} ({ca.mean():.1f}/moment), B = components {b} ({cb.mean():.1f}/moment);"
          f" mean |A-B| {cost:.2f}; next-best split {min(others) if others else float('nan'):.2f} (margin {margin:.0%})")
    (out / "sides.json").write_text(json.dumps({"A": a, "B": b, "keeperOrReferee": solos, "kitComponents": kits,
                                                "perMoment": per_moment.round(2).tolist(), "saturation": sat.round(1).tolist(),
                                                "meanAbsDiff": round(cost, 3), "nextBest": round(min(others), 3) if others else None}))


def main(rec: Path, name: str, k: int) -> None:
    import rfdetr
    from sklearn.mixture import GaussianMixture

    lab = rec / f"labels_{name}"
    items = json.loads((lab / "items.json").read_text())
    out = rec / "teams" / name
    out.mkdir(parents=True, exist_ok=True)
    model = rfdetr.RFDETRMedium()
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    pose = P.load_pose(rec)
    hl, hw = pose.L / 2 + 1, pose.W / 2 + 1

    def on_pitch(role, box):
        d = cams[role].unproject(np.array([[(box[0] + box[2]) / 2, box[3]]], float))[0]
        x, z = pose.to_pitch(np.degrees(np.arctan2(d[0], d[2])), np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2]))))
        return (None if np.isnan(x) else [round(float(x), 1), round(float(z), 1)]), bool(not np.isnan(x) and abs(x) <= hl and abs(z) <= hw)

    people, feats, thumbs, t0 = [], [], [], time.time()
    for n, it in enumerate(items):
        for f in it["frames"]:
            bgr = cv2.imread(str(lab / f["image"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes = person_boxes(model, rgb)
            # Global turf reference: the median of green pixels over the field band.
            band = bgr[760::16, ::16].reshape(-1, 3)
            hsv = cv2.cvtColor(band.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            g = band[(hsv[:, 0] > 30) & (hsv[:, 0] < 95) & (hsv[:, 1] > 40)]
            turf_global = g.mean(0) if len(g) else np.array([60, 120, 60], float)
            for b in boxes:
                r = features(bgr, b, turf_global)
                if r is None:
                    continue
                xz, inside = on_pitch(f["phone"], b)
                people.append({"frame": f["id"], "phone": f["phone"], "box": [round(float(v)) for v in b[:4]],
                               "conf": round(float(b[4]), 3), "xz": xz, "onPitch": inside})
                feats.append(r[0]); thumbs.append(r[1])
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(items)} moments, {len(people)} people, {(time.time() - t0) / 60:.1f} min", flush=True)
    X = np.array(feats, dtype=np.float64)
    tall = np.array([p["box"][3] - p["box"][1] >= STAT_MIN_HEIGHT_PX and p["onPitch"] for p in people])
    mu, sd = X[tall].mean(0), X[tall].std(0) + 1e-3
    Z = (X - mu) / sd
    gmm = GaussianMixture(n_components=k, covariance_type="diag", n_init=5, reg_covar=1e-2, random_state=0).fit(Z[tall])
    comp = gmm.predict(Z)
    prob = gmm.predict_proba(Z).max(1)
    order = np.argsort(-np.bincount(comp, minlength=k))
    rank = {int(c): i for i, c in enumerate(order)}
    for p, c, pr in zip(people, comp, prob):
        p["component"] = rank[int(c)]
        p["prob"] = round(float(pr), 3)
    (out / "people.json").write_text(json.dumps(people))
    np.savez(out / "model.npz", means=gmm.means_, covs=gmm.covariances_, weights=gmm.weights_, mu=mu, sd=sd,
             rank=np.array([rank[i] for i in range(k)]))
    sizes = np.bincount([p["component"] for p in people], minlength=k)
    print(f"{len(people)} people from {len(items)} moments; components by size: {sizes.tolist()};"
          f" mean assignment probability {prob.mean():.2f}")
    sides(out, people, X, [p["component"] for p in people], len(items), k)
    # Contact sheet: one row band per component, the patches sorted by probability.
    rows = []
    per_row = 60
    for ci in range(k):
        idx = [i for i in range(len(people)) if people[i]["component"] == ci]
        idx.sort(key=lambda i: -people[i]["prob"])
        sel = idx[:: max(1, len(idx) // (per_row * 3))][:per_row * 3]
        tiles = [thumbs[i] for i in sel]
        while len(tiles) % per_row:
            tiles.append(np.zeros((32, 24, 3), np.uint8))
        band = np.vstack([np.hstack(tiles[r:r + per_row]) for r in range(0, len(tiles), per_row)])
        label = np.zeros((20, band.shape[1], 3), np.uint8)
        cv2.putText(label, f"component {ci}: {len(idx)} people", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows += [label, band, np.full((4, band.shape[1], 3), 90, np.uint8)]
    cv2.imwrite(str(out / "sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"-> {out / 'sheet.jpg'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
