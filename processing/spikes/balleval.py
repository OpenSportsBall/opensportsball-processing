#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Scores ball tracking along a clip against the clip's ball labels.

    uv run processing/spikes/balleval.py footage/rec3 <set> ["NAME=value,..." ...]

<set> is a labels_<set> made by `labelset.py clip` and labelled on the
page. The clip's candidates (ballclip.py, with balljudge.py scores if
run) go through ball.py's filters and ballpath.py's offline path as
follow.py's "ball" feature does; at each labelled moment the estimate is
compared with the label placed on the pitch. OSB_BALL_CHAIN=1 links with
ball.py's online chain instead (same candidates and filters), as
follow.py does under the same variable. Each extra argument is a tracker
setting to override
("ball.MAX_MISSED_S=1.0,ball.MIN_JUDGE=0.5" or "ballpath.NONE_COST=3"),
one run per argument, so settings can be swept in one call.

Per run: found (estimate within HIT_M of the labelled ball), wrong
(estimate elsewhere), missed (no estimate); coverage over the clip; and
estimates at moments labelled "none" (ball not visible: usually in the
keeper's hands or in flight, so not counted as wrong).

A moment's estimate is read at the time of the label image that placed
the ball: a frame's "dtS" in items.json (its offset from the nominal
clipT; the two phones of one moment differ, so the offset of the phone
whose image was labelled is used). OSB_NOMINAL=1 ignores the offsets, as
runs before 2026-10-18 did. OSB_BALL_TAG picks the candidate file ("" is
ballclip.py's full scan, "win" ballwin.py's windows).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ball as B  # noqa: E402
import follow  # noqa: E402
import pitch as P  # noqa: E402

HIT_M = float(os.environ.get("OSB_HIT_M", 3.0))        # a hit: within this plus HIT_PER_M x the ball's distance ...
HIT_PER_M = float(os.environ.get("OSB_HIT_PER_M", 0.05))  # ... from the camera (a far ball's depth on the pitch is uncertain)
NOMINAL = bool(os.environ.get("OSB_NOMINAL"))          # compare at the nominal label time, ignoring the frames' dtS


def cands_file(rec: Path, seg0: int, start: float, tag: str) -> Path:
    """The clip's candidate file with exactly this tag: ball_<seg>_<start>_<seconds>[.<tag>].json."""
    pat = re.compile(rf"ball_{seg0}_{start:g}_[\d.]+?{re.escape('.' + tag) if tag else ''}\.json$")
    paths = sorted(p for p in (rec / "detect").glob(f"ball_{seg0}_{start:g}_*.json") if pat.fullmatch(p.name))
    if not paths:
        raise SystemExit(f"no candidate file ball_{seg0}_{start:g}_*{'.' + tag if tag else ''}.json in {rec / 'detect'}")
    return paths[0]


def setup(rec: Path, name: str):
    lab = rec / f"labels_{name}"
    items = json.loads((lab / "items.json").read_text())
    labels = json.loads((lab / "labels.json").read_text())
    seg0, start = int(items[0]["id"].split("_")[0][1:]), float(items[0]["id"].split("_")[1])
    cands_path = cands_file(rec, seg0, start, os.environ.get("OSB_BALL_TAG", ""))
    print(f"candidates: {cands_path.name}")
    data = json.loads(cands_path.read_text())
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    pose = P.load_pose(rec)
    cam_xz = np.array([pose.cx, pose.cz])

    def on_pitch(role, x, y):
        d = cams[role].unproject(np.array([[x, y]], float))[0]
        yaw = np.degrees(np.arctan2(d[0], d[2]))
        pit = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
        return pose.to_pitch(yaw, pit)

    # Labelled moments: the match ball on the pitch, or the status; each at
    # the time of the label image it came from (clipT + that frame's dtS).
    moments, dts = [], []
    for it in items:
        xz, status, dt = None, None, 0.0
        for f in it["frames"]:
            v = labels.get(f["id"])
            if not v:
                continue
            if status is None:
                status, dt = v["status"], f.get("dtS", 0.0)
            for b in v.get("balls", []):
                if b["inPlay"] and xz is None:
                    x, z = on_pitch(f["phone"], b["x"], b["y"])
                    xz = None if np.isnan(x) else np.array([x, z])
                    status, dt = "ball", f.get("dtS", 0.0)
        if status:
            dts.append(dt)
            k = int(round((it["clipT"] + (0.0 if NOMINAL else dt)) * data["fps"]))
            moments.append((it["clipT"], k, status, xz))
    n_dt = sum(f.get("dtS") is not None for it in items for f in it["frames"])
    n_fr = sum(len(it["frames"]) for it in items)
    if n_dt < n_fr:
        print(f"only {n_dt}/{n_fr} frames carry dtS (label time offset): the rest are compared at the nominal time"
              f" (ballheat.py <set> dt measures it)")
    when = "nominal clipT (OSB_NOMINAL)" if NOMINAL else "clipT + the labelled frame's dtS"
    print(f"label times: {when}, offsets {np.mean(dts):+.2f} s mean, {np.min(dts):+.2f} to {np.max(dts):+.2f}")
    # Players per candidate frame, for the tracker's "near a player" score.
    seconds = len(data["frames"]) / data["fps"]
    cache = follow.passa_cache(rec / "follow", seg0, start, seconds)
    people = [np.zeros((0, 2))] * len(data["frames"])
    if cache.exists():
        dets = json.loads(cache.read_text())["dets"]
        for i in range(len(data["frames"])):
            j = min(len(dets) - 1, int(round(i * follow.FPS / data["fps"])))
            pts = [pose.to_pitch(np.degrees(yw), np.degrees(feet)) for yw, _, _, feet in dets[j]]
            people[i] = np.array([p for p in pts if not np.isnan(p[0])]).reshape(-1, 2)
    return data, moments, people, cam_xz, pose


def track(data, people, pose):
    fps, n = data["fps"], len(data["frames"])
    cands = [[d for d in f["detections"] if B.usable(d, pose)] for f in data["frames"]]
    spots = B.static_spots(cands, fps, pose=pose)
    keep = [[d for d in fr if all(np.hypot(*(np.array(d["xz"]) - s)) > B.STATIC_RADIUS for s in spots)]
            for fr in cands]
    if os.environ.get("OSB_BALL_CHAIN"):  # ball.py's online chain instead of the offline path
        tracks = B.link(keep, fps)
        est, src, _ = B.choose(tracks, people, n, fps)
        return est
    import ballpath
    est, _ = ballpath.track(keep, people, n, fps)
    return est


def score(est, moments, cam_xz):
    found = wrong = missed = none_est = none_n = 0
    for t, k, status, xz in moments:
        e = est[k] if k < len(est) else None
        if status == "ball" and xz is not None:
            if e is None:
                missed += 1
            elif np.hypot(*(e - xz)) <= HIT_M + HIT_PER_M * np.hypot(*(xz - cam_xz)):
                found += 1
            else:
                wrong += 1
        elif status == "none":
            none_n += 1
            none_est += e is not None
    cov = np.mean([e is not None for e in est])
    return found, wrong, missed, none_est, none_n, cov


def main(rec: Path, name: str, runs: list[str]) -> None:
    data, moments, people, cam_xz, pose = setup(rec, name)
    n_ball = sum(s == "ball" and xz is not None for _, _, s, xz in moments)
    print(f"{name}: {len(moments)} labelled moments, {n_ball} with the match ball on the pitch")
    defaults = {}
    for run in runs or [""]:
        for (mod, attr), v in list(defaults.items()):  # restore
            setattr(importlib.import_module(mod), attr, v)
        for ov in filter(None, run.split(",")):
            nm, val = ov.split("=")
            mod, attr = nm.split(".") if "." in nm else ("ball", nm)
            m = importlib.import_module(mod)
            cur = getattr(m, attr)
            defaults.setdefault((mod, attr), cur)
            setattr(m, attr, int(val) if isinstance(cur, int) else float(val))
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            est = track(data, people, pose)
        f, w, m, ne, nn, cov = score(est, moments, cam_xz)
        print(f"{run or 'default':55s} found {f:2d}/{n_ball} ({100 * f / n_ball:3.0f}%)  wrong {w:2d}  missed {m:2d}"
              f"  | estimate at {ne}/{nn} 'none' moments | coverage {100 * cov:3.0f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
