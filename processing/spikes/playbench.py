#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Runs follow-cam variants on the labelled clips and scores them against
the reviewed must-see boxes (playeval.py).

    uv run processing/spikes/playbench.py footage/rec3 [variant ...]

Each clip's player detections (follow.py pass A) are cached after the
first run, so a variant plans its path in seconds. Paths are kept in
<recording>/playeval/<set>/paths/<variant>.json. Variants are feature
strings for follow.py (see play_targets); named ones below, or any
feature string on the command line.

Score per variant: moments captured (>= 98% of the box in view), the mean
share of the box in view, the median view width over the clip, and the
width at the captured moments (narrower is better once captured).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import playeval  # noqa: E402

CLIPS = [("seg9-60", 9, 60.0, 120.0), ("seg3-225", 3, 225.0, 120.0), ("seg0-0", 0, 0.0, 600.0)]
A = "goalkick,goalframe,unsure"
VARIANTS = {
    "A": A,
    "A+weak": A + ",weak",
    "A+weak60": A + ",weak,WEAK_HFOV=50:60",
    "A+gk2": A + ",gk2",
    "A+gk2+weak": A + ",gk2,weak",
    "A+allnear": A + ",allnear",
    "A+flow": A + ",flow",
    "A+nowalk": A + ",nowalk",
    "A+nolull": A + ",LULL_RUNNERS=-1",
    "A+bw15": A + ",HOT_BANDWIDTH_M=15",
    "A+ball": A + ",ball",
    "A+gk2+weak+allnear": A + ",gk2,weak,allnear",
    "A+goalin": A + ",goalin",
    "A+lone": A + ",lone",
    "A+goalin+lone": A + ",goalin,lone",
    "A+in3": A + ",INSIDE_M=3",
    "A+goalin+lone+in3": A + ",goalin,lone,INSIDE_M=3",
    "A+nowalk+goalin+lone": A + ",nowalk,goalin,lone",
    "A+ball+goalin+lone": A + ",ball,goalin,lone",
    "A+ball+goalin": A + ",ball,goalin",
    "A+ball+nowalk": A + ",ball,nowalk",
    "A+ball+hold": A + ",ball,ballhold",
    "A+ball+hold+goalin": A + ",ball,ballhold,goalin",
    "A+ball+hold+goalin+lone": A + ",ball,ballhold,goalin,lone",
    "A+ball+hold+nowalk": A + ",ball,ballhold,nowalk",
    "A+ball+hold+goalin+in3": A + ",ball,ballhold,goalin,INSIDE_M=3",
    "A+ball+goalin+nowalk1": A + ",ball,goalin,nowalk,WALK_SPEED=1.0",
}


def run_variant(rec: Path, name: str, features: str) -> dict[str, Path]:
    paths = {}
    for set_name, seg, start, seconds in CLIPS:
        dest = rec / "playeval" / set_name / "paths" / f"{name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OSB_NO_RENDER="1")
        r = subprocess.run(["uv", "run", "-q", str(Path(__file__).parent / "follow.py"), str(rec), str(seg),
                            f"{start:g}", f"{seconds:g}", "play", features], env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:])
            raise SystemExit(f"{name} failed on {set_name}")
        for line in r.stdout.splitlines():
            if line.startswith("ball:"):
                print(f"  [{set_name}] {line[line.index('aim at the ball'):]}", flush=True)
        flags = [f for f in features.split(",") if "=" not in f]
        src = rec / "follow" / ("path_" + "_".join(["play", *flags]) + ".json")
        shutil.copy(src, dest)
        shutil.copy(rec / "follow" / "play_debug.json", dest.with_name(f"{name}_debug.json"))
        paths[set_name] = dest
    return paths


def score_variant(rec: Path, name: str, paths: dict[str, Path]) -> dict:
    caps, covs, widths, cap_widths, per = 0, [], [], [], []
    n = 0
    for set_name, *_ in CLIPS:
        out = rec / "playeval" / set_name
        items = json.loads((out / "items.json").read_text())
        boxes = json.loads((out / "boxes.json").read_text())
        path = json.loads(paths[set_name].read_text())
        widths += path["hfovDeg"]
        for it in sorted(items, key=lambda i: i["clipT"]):
            b = boxes.get(it["id"])
            if not b or not b.get("reviewed"):
                continue
            k = it["frame"]
            yaw, pitch, hfov = (np.radians(path[key][k]) for key in ("yawDeg", "pitchDeg", "hfovDeg"))
            cov = playeval.coverage(b, yaw, pitch, hfov)
            n += 1
            covs.append(cov)
            if cov >= playeval.CAPTURED:
                caps += 1
                cap_widths.append(np.degrees(hfov))
            per.append(f"{100 * cov:3.0f}")
    return {"name": name, "captured": caps, "n": n, "meanCov": float(np.mean(covs)),
            "medianWidth": float(np.median(widths)),
            "capWidth": float(np.median(cap_widths)) if cap_widths else float("nan"), "per": per}


def main(rec: Path, names: list[str]) -> None:
    todo = {n: VARIANTS.get(n, n) for n in (names or VARIANTS)}
    results = []
    for name, feats in todo.items():
        paths = run_variant(rec, name, feats)
        r = score_variant(rec, name, paths)
        results.append(r)
        print(f"{name:22s} captured {r['captured']:2d}/{r['n']}  mean cov {100 * r['meanCov']:3.0f}%"
              f"  width median {r['medianWidth']:3.0f} / at captures {r['capWidth']:3.0f}   {' '.join(r['per'])}",
              flush=True)
    (rec / "playeval" / "bench.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
