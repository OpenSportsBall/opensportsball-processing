#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Scores pitch fits against a marked-field benchmark.

    uv run processing/spikes/pitchbench.py footage/fixtures/pitch/<name> <pitch.json> [<pitch.json> ...]
                                           [--goals <goals.json>] [--all]

A benchmark fixture holds background.png, the cameras.json it was built with,
and benchmark.json: every pitch marking as complete polylines in panorama
degrees (yaw right +, pitch down +) with a per-point "traced" flag (on paint
seen in the image, or completed), the tapped keypoints and the goal frames
(goalFramesDeg). A fit is a pitch.json (pitch.py's PitchPose, optionally with
"circleRadiusM" for a centre circle of its own size) made on that same
background panorama; a fit made with other cameras is in another panorama
and can't be scored here.

Per marking: how far the benchmark's traced points are from the fit's
marking (median and 90th percentile, panorama px; --all also scores the
completed parts). Per tap: how far the fit puts that keypoint. With --goals,
how far that goals.json's frames (posts and crossbar) are from the
benchmark's. One panorama px is 0.03 deg.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pitch as P  # noqa: E402

# benchmark.json's marking names, in pitch.py's _elements order
NAMES = ["near touchline", "far touchline", "left goal line", "right goal line", "halfway line", "centre circle"]
for _e in ("left", "right"):
    NAMES += [f"{_e} 18-yard front", f"{_e} 18-yard near side", f"{_e} 18-yard far side",
              f"{_e} 6-yard front", f"{_e} 6-yard near side", f"{_e} 6-yard far side", f"{_e} penalty arc"]
GROUPS = {"near touchline": ["near touchline"], "far touchline": ["far touchline"],
          "goal lines": ["left goal line", "right goal line"], "halfway": ["halfway line"], "centre circle": ["centre circle"],
          "left end boxes": [n for n in NAMES if n.startswith("left ") and "goal line" not in n],
          "right end boxes": [n for n in NAMES if n.startswith("right ") and "goal line" not in n]}


def px(yp) -> np.ndarray:
    yp = np.asarray(yp, float).reshape(-1, 2)
    return np.c_[(yp[:, 0] - P.YAW_RANGE[0]) / P.DEG_PER_PX, (yp[:, 1] - P.PITCH_RANGE[0]) / P.DEG_PER_PX]


def nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each point of a, the distance to the nearest point of b (chunked)."""
    out = np.empty(len(a))
    for i in range(0, len(a), 2000):
        d = a[i:i + 2000, None, :] - b[None, :, :]
        out[i:i + 2000] = np.sqrt((d ** 2).sum(-1)).min(1)
    return out


def fit_marking(pose: P.PitchPose, i: int, rc: float | None) -> np.ndarray:
    if i == P.CENTRE_CIRCLE and rc is not None:
        t = np.linspace(0, 2 * np.pi, int(rc * 2 * np.pi / 0.05))
        xz = np.c_[rc * np.cos(t), rc * np.sin(t)]
    else:
        xz = P.pitch_lines(pose.L, pose.W, pose.k, step=0.05, only=(i,))
    a = pose.angles(xz)
    p = px(a)
    ok = (p[:, 0] >= 0) & (p[:, 0] < (P.YAW_RANGE[1] - P.YAW_RANGE[0]) / P.DEG_PER_PX) & \
         (p[:, 1] >= 0) & (p[:, 1] < (P.PITCH_RANGE[1] - P.PITCH_RANGE[0]) / P.DEG_PER_PX)
    return p[ok]


def tap_target(pose: P.PitchPose, rc: float | None, name: str):
    tg = P.tap_targets(pose)
    r = rc if rc is not None else P.CIRCLE_R * pose.k
    tg.update({"halfway far end": (0.0, pose.W / 2), "circle near crossing": (0.0, -r), "circle far crossing": (0.0, r)})
    return tg.get(name)


def goal_frame_px(pose: P.PitchPose, g: dict) -> np.ndarray:
    th = math.radians(g.get("angleToGoalLineDeg", 0.0))
    d = np.array([math.sin(th), math.cos(th)])
    c = np.array([g["x"], g["z"]])
    a_, b_ = c - g["width"] / 2 * d, c + g["width"] / 2 * d
    ph, cz = np.linspace(0, g["height"], 40), np.linspace(0, 1, 120)
    xz = np.r_[np.tile(a_, (40, 1)), np.tile(b_, (40, 1)), a_ + np.outer(cz, b_ - a_)]
    hh = np.r_[ph, ph, np.full(120, g["height"])]
    return px(pose.angles3(xz, hh))


def score(bench: dict, pj: Path, goals_path: Path | None, use_all: bool) -> dict:
    j = json.loads(pj.read_text())
    pose, rc = P.pose_from_json(j), j.get("circleRadiusM")
    res = {"markings": {}, "taps": {}, "goals": {}}
    for i, n in enumerate(NAMES):
        m = bench["markings"].get(n)
        if not m:
            continue
        pts = [q for p in m["pieces"] for q, t in zip(p["polyline"], p["traced"]) if use_all or t]
        model = fit_marking(pose, i, rc)
        if not pts or not len(model):
            continue
        d = nearest(px(pts), model)
        res["markings"][n] = (float(np.median(d)), float(np.percentile(d, 90)), len(d))
    for n, t in bench["taps"].items():
        if t.get("skipped") or n.endswith("post base"):
            continue
        xz = tap_target(pose, rc, n)
        if xz is None:
            continue
        res["taps"][n] = float(np.hypot(*(px(pose.angles(np.array([xz])))[0] - px([[t["yaw"], t["pitch"]]])[0])))
    if goals_path is not None:
        gs = {("left" if g["side"] < 0 else "right"): g for g in json.loads(goals_path.read_text())["goals"]}
        for e, fr in bench["goalFramesDeg"].items():
            if e in gs:
                ref = px([q for part in fr.values() for q in part])
                d = nearest(goal_frame_px(pose, gs[e]), ref)
                res["goals"][e] = (float(np.median(d)), float(np.percentile(d, 90)))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", type=Path)
    ap.add_argument("fits", type=Path, nargs="+")
    ap.add_argument("--goals", type=Path)
    ap.add_argument("--all", action="store_true", help="also score the completed (not traced) parts")
    a = ap.parse_args()
    bench = json.loads((a.fixture / "benchmark.json").read_text())
    results = {f: score(bench, f, a.goals, a.all) for f in a.fits}
    names = [str(f.parent.name if f.name == "pitch.json" else f.stem) for f in a.fits]
    w = max(18, *(len(n) for n in names)) + 2
    print(f"benchmark {a.fixture.name} ({'all points' if a.all else 'traced points only'}): median / 90th percentile, panorama px (0.03 deg)\n")
    print(f"{'':22s}" + "".join(f"{n:>{w}s}" for n in names))
    for g, members in GROUPS.items():
        row = f"{g:22s}"
        for f in a.fits:
            ms = [results[f]["markings"][m] for m in members if m in results[f]["markings"]]
            if ms:
                med = float(np.average([x[0] for x in ms], weights=[x[2] for x in ms]))
                p90 = float(np.average([x[1] for x in ms], weights=[x[2] for x in ms]))
                row += f"{f'{med:5.1f} / {p90:5.1f}':>{w}s}"
            else:
                row += f"{'-':>{w}s}"
        print(row)
    row = f"{'taps (median / worst)':22s}"
    for f in a.fits:
        t = list(results[f]["taps"].values())
        row += f"{f'{np.median(t):5.1f} / {max(t):5.1f}':>{w}s}" if t else f"{'-':>{w}s}"
    print(row)
    if a.goals:
        for e in ("left", "right"):
            row = f"{e + ' goal frame':22s}"
            for f in a.fits:
                g = results[f]["goals"].get(e)
                row += f"{f'{g[0]:5.1f} / {g[1]:5.1f}':>{w}s}" if g else f"{'-':>{w}s}"
            print(row)
    print("\nper tap (px):")
    taps = sorted({n for f in a.fits for n in results[f]["taps"]})
    for n in taps:
        print(f"  {n:30s}" + "".join(f"{results[f]['taps'].get(n, float('nan')):{w}.1f}" for f in a.fits))


if __name__ == "__main__":
    main()
