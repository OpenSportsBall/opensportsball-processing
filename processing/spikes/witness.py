#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Ball "witnesses": where the ball is, from motion other than the ball's.

    uv run processing/spikes/witness.py footage/rec3 <set> keepers|fuse [tracker overrides]

Each witness gives a per-frame estimate of the ball on the pitch (or
none) at the clip candidates' frame rate, scored against the clip's ball
labels like the tracker (balleval.py):

- keepers: a keeper stands on the line from the ball to the goal centre,
  and comes off the line as the ball comes nearer. Each goal's keeper
  (the player deepest in its goal area) gives a bearing from the goal
  centre; two bearings intersect; one bearing plus the keeper's depth
  gives a point. Coarse: tens of metres across the ball's distance, a
  few metres sideways.
- fuse: the tracker's chain (ball.py over the clip's candidates) with
  the witness filling frames the chain leaves empty, and with candidates
  far from the witness (> FUSE_GATE_M) left out of the tracker.
"""

from __future__ import annotations

import io
import contextlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ball as B  # noqa: E402
import balleval as E  # noqa: E402
import follow  # noqa: E402
import pitch as P  # noqa: E402

KEEPER_ZONE_X = 20.0     # the keeper: the player within this of the goal line ...
KEEPER_ZONE_Z = 22.0     # ... and this of the goal centre line, nearest the goal
KEEPER_OFF_M = 1.5       # keepers stand a little off the ball-goal line: ignore offsets under this
DEPTH_GAIN = 6.0         # one bearing: ball distance from the goal ~ DEPTH_GAIN x the keeper's depth
FUSE_GATE_M = 25.0       # fuse: candidates this far from the witness are left out


def players_per_frame(rec: Path, seg0: int, start: float, seconds: float, fps: float, pose):
    """Pitch positions of the tracked people at each candidate frame, from
    follow.py's cached pass A detections (30 fps) at the candidate rate."""
    cache = follow.passa_cache(rec / "follow", seg0, start, seconds)
    dets = json.loads(cache.read_text())["dets"]
    n = int(round(seconds * fps))
    out = []
    hl, hw = pose.L / 2 + 1, pose.W / 2 + 1
    for i in range(n):
        j = min(len(dets) - 1, int(round(i * follow.FPS / fps)))
        pts = [pose.to_pitch(np.degrees(yw), np.degrees(feet)) for yw, _, _, feet in dets[j]]
        out.append(np.array([p for p in pts if not np.isnan(p[0]) and abs(p[0]) <= hl and abs(p[1]) <= hw]).reshape(-1, 2))
    return out


def keepers(rec: Path, players, pose):
    goals = P.load_goals(rec)
    goal_at = {g["side"]: np.array([g["x"], g["z"]]) for g in goals}
    est = []
    for pts in players:
        rays = []
        for side, g in goal_at.items():
            zone = pts[(side * (g[0] - pts[:, 0]) <= KEEPER_ZONE_X) & (side * (g[0] - pts[:, 0]) >= -2)
                       & (np.abs(pts[:, 1] - g[1]) <= KEEPER_ZONE_Z)] if len(pts) else pts
            if not len(zone):
                continue
            k = zone[np.argmin(side * (g[0] - zone[:, 0]))]  # deepest: nearest the goal line
            d = k - g
            if np.hypot(*d) < KEEPER_OFF_M:
                continue
            rays.append((g, d / np.hypot(*d), float(np.hypot(*d))))
        if len(rays) == 2:
            # Least-squares intersection of the two rays.
            (g1, u1, _), (g2, u2, _) = rays
            A = np.c_[u1, -u2]
            if abs(np.linalg.det(A)) < 0.05:  # near-parallel: fall back to the depth rule
                rays = rays[:1]
            else:
                t = np.linalg.solve(A, g2 - g1)
                if t[0] > 0 and t[1] > 0:
                    est.append(g1 + u1 * t[0])
                    continue
                rays = rays[:1]
        if len(rays) == 1:
            g, u, depth = rays[0]
            est.append(g + u * DEPTH_GAIN * depth)
        else:
            est.append(None)
    est = [None if e is None else np.clip(e, [-pose.L / 2, -pose.W / 2], [pose.L / 2, pose.W / 2]) for e in est]
    return est


def fuse(data, people, pose, witness):
    """The tracker's chain over candidates within FUSE_GATE_M of the
    witness (where there is one), then the witness where the chain is
    empty."""
    fps, n = data["fps"], len(data["frames"])
    cands = []
    for i, f in enumerate(data["frames"]):
        w = witness[i] if i < len(witness) else None
        cands.append([d for d in f["detections"] if B.usable(d, pose)
                      and (w is None or np.hypot(*(np.array(d["xz"]) - w)) <= FUSE_GATE_M)])
    spots = B.static_spots(cands, fps)
    keep = [[d for d in fr if all(np.hypot(*(np.array(d["xz"]) - s)) > B.STATIC_RADIUS for s in spots)]
            for fr in cands]
    tracks = B.link(keep, fps)
    est, src, _ = B.choose(tracks, people, n, fps)
    filled = [e if e is not None else (witness[i] if i < len(witness) else None) for i, e in enumerate(est)]
    return est, filled


def main(rec: Path, name: str, mode: str, runs: list[str]) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        data, moments, people, cam_xz, pose = E.setup(rec, name)
    items = json.loads((rec / f"labels_{name}" / "items.json").read_text())
    seg0, start = int(items[0]["id"].split("_")[0][1:]), float(items[0]["id"].split("_")[1])
    seconds = len(data["frames"]) / data["fps"]
    players = players_per_frame(rec, seg0, start, seconds, data["fps"], pose)
    n_ball = sum(s == "ball" and xz is not None for _, _, s, xz in moments)
    print(f"{name}: {n_ball} labelled moments with the ball")
    witness = keepers(rec, players, pose)
    f, w, m, ne, nn, cov = E.score(witness, moments, cam_xz)
    print(f"keepers alone: found {f}/{n_ball} ({100 * f / n_ball:.0f}%) wrong {w} missed {m}; coverage {100 * cov:.0f}%")
    # How far off, when it has an answer?
    errs = [np.hypot(*(witness[k] - xz)) for t, k, s, xz in moments if s == "ball" and xz is not None and witness[k] is not None]
    print(f"  distance to the labelled ball: median {np.median(errs):.1f} m, 75th {np.percentile(errs, 75):.1f} m")
    if mode == "fuse":
        for run in runs or [""]:
            saved = {}
            for ov in filter(None, run.split(",")):
                nm, val = ov.split("=")
                mod = sys.modules[__name__] if nm in globals() else B
                saved[(mod, nm)] = getattr(mod, nm)
                setattr(mod, nm, float(val))
            with contextlib.redirect_stdout(io.StringIO()):
                est, filled = fuse(data, people, pose, witness)
                est0, _ = fuse(data, people, pose, [None] * len(witness))
            for label, e in (("tracker alone", est0), ("tracker gated by keepers", est), ("gated + keepers fill", filled)):
                f, w, m, ne, nn, cov = E.score(e, moments, cam_xz)
                print(f"{run or 'default':40s} {label:26s} found {f:2d}/{n_ball} ({100 * f / n_ball:3.0f}%) wrong {w:2d} missed {m:2d}"
                      f" | coverage {100 * cov:3.0f}%")
            for (mod, nm), v in saved.items():
                setattr(mod, nm, v)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "keepers", sys.argv[4:])
