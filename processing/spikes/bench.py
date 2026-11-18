#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""Runs the standard benchmarks on the bench store and compares them with
the recorded baselines.

    uv run processing/spikes/bench.py check              # is everything a run needs here, and are the caches current?
    uv run processing/spikes/bench.py run [play] [ball]  # run (all by default) and compare with the baselines
    uv run processing/spikes/bench.py stamp              # record the inputs the caches were made from

The store is footage/bench/ (docs/DATA.md): what the benchmarks read and
nothing can remake. footage/bench-cache/ holds what is made from it and
only saves time (pass A caches, ball candidates, roles); footage/bench-out/
what the runs write (recpaths.py). Baselines are in
processing/bench_baselines.json, in git: change them in the same commit
as the change that moves a score, with the reason in the message.

"check" lists, per recording, the inputs and caches each benchmark in the
baselines reads and whether they are there, and compares the inputs the
caches were made from (bench-cache/<rec>/inputs.json, written by "stamp")
with the inputs now: a cache made from another camera fit, pitch map or
model gives scores that mean nothing. The video is reported, not
required: the benchmarks run from the caches, and a cache that is missing
is remade from the video (pass A by the play bench itself; ball
candidates by ballwin.py, roles by roles.py: docs/TESTING-PROCESSING.md).

"run" exits 1 if a score is below its baseline or a run fails. A score
above its baseline is reported so the baseline can be raised.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recpaths  # noqa: E402

ROOT = HERE.parents[1]
STORE = ROOT / "footage" / "bench"
MODELS = ROOT / "footage" / "models"
BASELINES = HERE.parent / "bench_baselines.json"
# What a recording's caches are made from: a change in any of these makes them stale.
REC_INPUTS = ["phase0/cameras.json", "phase0/report.json", "pitch/pitch.json", "pitch/goals.json"]
MODEL_INPUTS = ["ball_detector.pth", "ball_judge.pt"]


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()[:16]


def inputs_now(rec: Path) -> dict:
    return {**{p: sha(rec / p) for p in REC_INPUTS}, **{f"models/{m}": sha(MODELS / m) for m in MODEL_INPUTS}}


def play_needs(rec: Path, features: str) -> list[tuple[str, Path, bool]]:
    """(what, path, there) for every file the play bench reads for `rec`."""
    out = [("input", rec / p, (rec / p).exists()) for p in ("manifest.json", "phase0/report.json", "phase0/cameras.json",
                                                             "pitch/pitch.json")]
    cache = recpaths.cache(rec)
    for clip in sorted(rec.glob("playeval/*/clip.json")):
        c = json.loads(clip.read_text())
        for name in ("items.json", "boxes.json"):
            out.append(("input", clip.parent / name, (clip.parent / name).exists()))
        passa = cache / "follow" / f"passA_{int(c['segment'])}_{float(c['startS']):g}_{float(c['seconds']):g}.json"
        out.append(("cache", passa, passa.exists()))
        for feature, pattern in (("ball", f"ball_*_{float(c['startS']):g}_*.win.json"),
                                 ("roles", f"roles_*_{float(c['startS']):g}_*.json")):
            if feature in features.split(","):
                found = sorted((cache / "detect").glob(pattern))
                out.append(("cache", found[0] if found else cache / "detect" / pattern, bool(found)))
    return out


def ball_needs(rec: Path, name: str, tag: str) -> list[tuple[str, Path, bool]]:
    lab = rec / f"labels_{name}"
    out = [("input", lab / f, (lab / f).exists()) for f in ("items.json", "labels.json")]
    out += [("input", rec / p, (rec / p).exists()) for p in ("manifest.json", "phase0/report.json", "phase0/cameras.json",
                                                              "pitch/pitch.json")]
    if out[0][2]:
        it = json.loads((lab / "items.json").read_text())[0]
        pattern = f"ball_{int(it['segment'])}_{float(it['t']) - float(it.get('clipT', 0)):g}_*{'.' + tag if tag else ''}.json"
        found = sorted((recpaths.cache(rec) / "detect").glob(pattern))
        out.append(("cache", found[0] if found else recpaths.cache(rec) / "detect" / pattern, bool(found)))
    return out


def video_here(rec: Path) -> tuple[int, int]:
    """(segments here at the manifest's size, segments in the manifest)."""
    here = total = 0
    for d in json.loads((rec / "manifest.json").read_text())["devices"]:
        for s in d["segments"]:
            total += 1
            p = rec / s["objectKey"]
            here += p.exists() and p.stat().st_size == s["bytes"]
    return here, total


def check(base: dict) -> bool:
    ok = True
    recs = sorted({*base["play"]["recordings"], *(k.split("/")[0] for k in base["ball"])})
    for name in recs:
        rec = STORE / name
        needs = []
        if name in base["play"]["recordings"]:
            needs += play_needs(rec, base["play"]["features"])
        for key, b in base["ball"].items():
            if key.split("/")[0] == name:
                needs += ball_needs(rec, key.split("/")[1], b.get("tag", ""))
        missing = [(what, p) for what, p, there in needs if not there]
        stamp_file = recpaths.cache(rec) / "inputs.json"
        stale = []
        if stamp_file.exists():
            made, now = json.loads(stamp_file.read_text())["inputs"], inputs_now(rec)
            stale = [k for k in now if made.get(k) != now[k]]
        here, total = video_here(rec) if (rec / "manifest.json").exists() else (0, 0)
        state = "ok" if not missing and not stale and stamp_file.exists() else "PROBLEMS"
        print(f"{name}: {state}; {len(needs) - len(missing)}/{len(needs)} files; {here} of the recording's {total} video segments here "
              f"(benchstore.py checks the ones the sets need)")
        for what, p in missing:
            print(f"  missing {what}: {p.relative_to(ROOT)}")
        if not stamp_file.exists():
            print(f"  caches not stamped ({stamp_file.relative_to(ROOT)}): run `bench.py stamp` once they are known to be current")
        for k in stale:
            print(f"  caches are stale: {k} changed since they were made (remake them from the video, then `bench.py stamp`)")
        ok &= state == "ok"
    print("the store has what the baselines' benchmarks read" if ok else "PROBLEMS above")
    return ok


def stamp(base: dict) -> None:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    for name in sorted({*base["play"]["recordings"], *(k.split("/")[0] for k in base["ball"])}):
        rec = STORE / name
        f = recpaths.cache(rec) / "inputs.json"
        f.write_text(json.dumps({"inputs": inputs_now(rec), "stampedAtCommit": commit,
                                 "note": "the inputs these caches match; bench.py check compares them with the store"}, indent=1) + "\n")
        print(f"stamped {f.relative_to(ROOT)}")


def run_play(base: dict) -> bool:
    b = base["play"]
    recs = [STORE / r for r in b["recordings"]]
    r = subprocess.run(["uv", "run", "-q", str(HERE / "playbench.py"), *map(str, recs), b["features"]])
    if r.returncode:
        print("play bench: FAILED to run")
        return False
    ok = True
    for rec in recs:
        got = json.loads((recpaths.out(rec) / "playeval" / "bench.json").read_text())[0]
        want = b["recordings"][rec.name]
        verdict = "ok" if [got["captured"], got["n"]] == want else "BELOW" if got["captured"] < want[0] or got["n"] != want[1] else "above"
        print(f"play {rec.name}: {got['captured']}/{got['n']} captured (baseline {want[0]}/{want[1]}) {verdict}")
        ok &= verdict != "BELOW"
    return ok


def run_ball(base: dict) -> bool:
    import os

    ok = True
    for key, b in base["ball"].items():
        name, set_name = key.split("/")
        r = subprocess.run(["uv", "run", "-q", str(HERE / "balleval.py"), str(STORE / name), set_name],
                           env=dict(os.environ, OSB_BALL_TAG=b.get("tag", "")), capture_output=True, text=True)
        m = re.search(r"found\s+(\d+)/(\d+).*?wrong\s+(\d+)", r.stdout)
        if r.returncode or not m:
            print(f"ball {key}: FAILED to run\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
            ok = False
            continue
        found, of, wrong = map(int, m.groups())
        below = found < b["found"] or of != b["of"] or wrong > b["wrong"]
        same = (found, of, wrong) == (b["found"], b["of"], b["wrong"])
        print(f"ball {key}: found {found}/{of}, wrong {wrong} (baseline {b['found']}/{b['of']}, wrong {b['wrong']}) "
              f"{'ok' if same else 'BELOW' if below else 'above'}")
        ok &= not below
    return ok


def main() -> None:
    base = json.loads(BASELINES.read_text())
    match sys.argv[1:2]:
        case ["check"]:
            sys.exit(0 if check(base) else 1)
        case ["stamp"]:
            stamp(base)
        case ["run"]:
            which = sys.argv[2:] or ["play", "ball"]
            ok = all([{"play": run_play, "ball": run_ball}[w](base) for w in which])
            print("all at or above the baselines" if ok else "BELOW the baselines (or failed) above")
            sys.exit(0 if ok else 1)
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
