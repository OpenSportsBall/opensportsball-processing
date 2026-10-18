#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "scikit-learn>=1.4", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""People along a clip with a role: side A, side B, keeper, referee, other.

    uv run processing/spikes/roles.py footage/rec3 <segment> <start s> <seconds> <teams set> [fps]

The stock RF-DETR finds people on both phones' frames at half
resolution (people are big enough; the ball isn't the point here), every
1/fps s. Each person's torso colour is assigned to the kit mixture of
teams.py's <teams set> (model.npz, sides.json): a kit component gives
side A or B; a keeper/referee component gives "keeper" when the person
stands in a goal area (within KEEPER_X_M of a goal line, KEEPER_Z_M of
its centre line) and "referee" otherwise; everything else is "other".

Writes <recording>/detect/roles_<segment>_<start>_<seconds>.json: per
frame, people as {xz, role, conf}. follow.py's "roles" feature reads it.
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
import teams  # noqa: E402

HALF = 1920
KEEPER_X_M = 22.0
KEEPER_Z_M = 25.0
FAR_MIN_PX = 12            # near a goal, people this tall still get a colour (keepers are far and small) ...
FAR_SAT = 30.0             # ... and a saturated torso there (not a kit colour) is the keeper


def main(rec: Path, seg: int, start: float, seconds: float, tset: str, fps: float) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    pose = P.load_pose(rec)
    goals = {g["side"]: np.array([g["x"], g["z"]]) for g in P.load_goals(rec)}
    hl, hw = pose.L / 2 + 1, pose.W / 2 + 1
    tdir = rec / "teams" / tset
    m = np.load(tdir / "model.npz")
    sides = json.loads((tdir / "sides.json").read_text())
    means, covs, weights, mu, sd, rank = m["means"], m["covs"], m["weights"], m["mu"], m["sd"], m["rank"]
    role_of_comp = {}
    for c in range(len(rank)):
        r = int(rank[c])
        role_of_comp[c] = "A" if r in sides["A"] else "B" if r in sides["B"] else "solo" if r in sides["keeperOrReferee"] else "other"

    def assign(f):
        z = (f - mu) / sd
        ll = -0.5 * (((z - means) ** 2) / covs).sum(1) - 0.5 * np.log(covs).sum(1) + np.log(weights)
        return int(np.argmax(ll))

    model = rfdetr.RFDETRMedium()
    W, H = follow.SRC_W, follow.SRC_H
    streams = [follow.prefetch(follow.game_stream(rec, devs, report, ph, seg, start, seconds, W, fps=fps))
               for ph in (0, 1)]
    frames_out, t0 = [], time.time()
    from collections import Counter
    tally = Counter()
    for i, (fa, fb) in enumerate(zip(*streams)):
        people = []
        for phone, (buf, fl) in enumerate(((fa, flip[0]), (fb, flip[1]))):
            bgr = cv2.cvtColor(np.asarray(buf).reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_NV12)
            if fl:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            small = cv2.resize(bgr, (HALF, HALF * 9 // 16), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            boxes = teams.person_boxes(model, rgb, band_top=detect.BAND_TOP // 2)
            band = bgr[760::16, ::16].reshape(-1, 3)
            hsv = cv2.cvtColor(band.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            g = band[(hsv[:, 0] > 30) & (hsv[:, 0] < 95) & (hsv[:, 1] > 40)]
            turf = g.mean(0) if len(g) else np.array([60, 120, 60], float)
            for b in boxes:
                box = b[:4] * 2  # back to full resolution
                d = cams[phone].unproject(np.array([[(box[0] + box[2]) / 2, box[3]]], float))[0]
                x, zz = pose.to_pitch(np.degrees(np.arctan2(d[0], d[2])), np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2]))))
                if np.isnan(x) or abs(x) > hl or abs(zz) > hw:
                    continue
                side = 1 if x > 0 else -1
                gx, gz = goals.get(side, (side * pose.L / 2, 0.0))
                in_area = side * (gx - x) <= KEEPER_X_M and abs(zz - gz) <= KEEPER_Z_M
                saved = teams.MIN_HEIGHT_PX
                if in_area:
                    teams.MIN_HEIGHT_PX = FAR_MIN_PX
                r = teams.features(bgr, box, turf)
                teams.MIN_HEIGHT_PX = saved
                role = "other"
                if r is not None:
                    comp = assign(r[0])
                    role = role_of_comp[comp]
                    if role == "solo":
                        role = "keeper" if in_area else "referee"
                    elif in_area and role == "other" and r[0][2] >= FAR_SAT:
                        role = "keeper"  # saturated but not a kit colour, in a goal area: the keeper
                people.append({"xz": [round(float(x), 2), round(float(zz), 2)], "role": role, "phone": phone,
                               "conf": round(float(b[4]), 3), "h": int(box[3] - box[1])})
                tally[role] += 1
        # The same person seen by both phones: keep one (nearest pair within 1.5 m, prefer the taller box).
        merged = []
        for p in sorted(people, key=lambda p: -p["h"]):
            if not any(q["phone"] != p["phone"] and np.hypot(*(np.subtract(q["xz"], p["xz"]))) < 1.5 for q in merged):
                merged.append(p)
        frames_out.append({"t": round(start + i / fps, 3), "people": merged})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{int(seconds * fps)} frames, {(time.time() - t0) / 60:.1f} min, roles so far {dict(tally)}", flush=True)
    path = rec / "detect" / f"roles_{seg}_{start:g}_{seconds:g}.json"
    path.write_text(json.dumps({"segment": seg, "fps": fps, "teams": tset, "frames": frames_out}))
    print(f"{len(frames_out)} frames in {(time.time() - t0) / 60:.1f} min; roles {dict(tally)} -> {path.name}")


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), int(a[2]), float(a[3]), float(a[4]), a[5], float(a[6]) if len(a) > 6 else 10.0)
