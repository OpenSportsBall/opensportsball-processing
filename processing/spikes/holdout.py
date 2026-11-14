#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Held-out ball labels: which stretches of a recording no model may train on.

    python3 processing/spikes/holdout.py footage/rec3                          # what is held out, and what it excludes
    python3 processing/spikes/holdout.py footage/rec3 mark labels_seg9         # a whole set
    python3 processing/spikes/holdout.py footage/rec3 mark labels segments 9,15  # whole segments of a set
    python3 processing/spikes/holdout.py footage/rec7 mark labels_seg6 seconds 6 180 220  # seconds 180-220 of segment 6

Every time balls are labelled, part of the labels is held back for testing
(docs/TESTING-PROCESSING.md, "Held-out ball labels"). A set says so with a
heldout.json in its folder:

    {"all": true}                 the whole set (a test clip)
    {"segments": [9, 15]}         these whole segments of the recording
    {"gameS": [[600, 1200]]}      these stretches of the game, in seconds
                                  from the recording's start (the engine's
                                  `label ball` sets)

The held-out time belongs to the recording, not the set: a labelled frame
from any set inside it (or within MARGIN_S of a held-out set's frames) is
test data. The training-data builders (balljudge.py data, balldata.py,
heattrain.py) leave those frames out and refuse a held-out set as a
training set, and stamp their output with the held-out time they applied;
balljudge.py train / merge and balltrain.py / ballmerge.py refuse data
whose stamp is missing or no longer matches (a set marked after the data
was built), so a model can't be trained on held-out labels by accident.
Marks are only added: to undo one, delete the heldout.json by hand, and
only if nothing was ever scored on it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MARGIN_S = 20.0  # frames this close to a held-out set's frames are held out too (the same ball, players and light)
MARK = "heldout.json"


class HeldOut(SystemExit):
    """A held-out label would have gone into training."""


def set_dirs(rec: Path) -> list[Path]:
    """Every ball label set of a recording: labels*/ and the engine's engine/labels/ball-*/."""
    dirs = [d for d in sorted(rec.glob("labels*")) if (d / "items.json").exists()]
    dirs += [d for d in sorted((rec / "engine" / "labels").glob("ball-*")) if (d / "items.json").exists()]
    return dirs


def _starts(rec: Path) -> list[float]:
    """Game seconds at which each of Main's segments starts."""
    main = next(d for d in json.loads((rec / "manifest.json").read_text())["devices"] if d["role"] == "main")
    out, t = [], 0.0
    for s in main["segments"]:
        out.append(t)
        t += s["durationMs"] / 1000
    out.append(t)
    return out


def item_times(items: list[dict]) -> list[tuple[int, float]]:
    """(segment, seconds into it) of each labelled moment."""
    return [(int(it["segment"]), float(it["t"])) for it in items if "segment" in it and "t" in it]


def windows(rec: Path) -> list[list[float]]:
    """The recording's held-out stretches, in game seconds, merged and sorted."""
    starts = _starts(rec)
    spans = []
    for d in set_dirs(rec):
        if not (d / MARK).exists():
            continue
        m = json.loads((d / MARK).read_text())
        if m.get("all"):
            ts = sorted(starts[s] + t for s, t in item_times(json.loads((d / "items.json").read_text())))
            for t in ts:  # each moment +- the margin, then merged
                spans.append([t - MARGIN_S, t + MARGIN_S])
        for s in m.get("segments", []):
            spans.append([starts[s], starts[s + 1]])
        spans += [list(map(float, w)) for w in m.get("gameS", [])]
    spans.sort()
    out: list[list[float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([round(a, 3), round(b, 3)])
    return out


class Holdout:
    """A recording's held-out time, for the training-data builders."""

    def __init__(self, rec: Path):
        self.rec = rec
        self.starts = _starts(rec)
        self.windows = windows(rec)

    def held(self, segment: int, t: float) -> bool:
        g = self.starts[segment] + t
        return any(a <= g <= b for a, b in self.windows)

    def stamp(self) -> dict:
        """What a builder writes next to its output: the held-out time it left out."""
        return {str(self.rec.resolve()): self.windows}

    def refuse_training_set(self, name: str) -> None:
        d = self.rec / name
        if (d / MARK).exists() and json.loads((d / MARK).read_text()).get("all"):
            raise HeldOut(f"{d} is held out for testing (heldout.json): it can't be a training set")


def check_stamp(stamp: dict | None, what: str) -> None:
    """Refuses training data built before the current held-out marks."""
    if not stamp:
        raise HeldOut(f"{what} carries no held-out stamp (built before held-out labels were checked): "
                      f"rebuild it (balljudge.py data / balldata.py) so held-out labels are left out")
    for rec, built in stamp.items():
        p = Path(rec)
        if not p.exists():
            continue  # another machine's path: checked where it was built
        now = windows(p)
        if now != built:
            raise HeldOut(f"{what}: {p.name}'s held-out time changed since it was built "
                          f"({built} -> {now}): rebuild it")


def mark(rec: Path, name: str, how: dict) -> None:
    d = rec / name
    if not (d / "items.json").exists():
        sys.exit(f"no label set {d}")
    if (d / MARK).exists():
        old = json.loads((d / MARK).read_text())
        if old.get("all") or how.get("all"):
            how = {"all": True}
        else:  # marks are only added: whole segments and stretches of the game both accumulate
            how = {k: v for k, v in {
                "segments": sorted(set(old.get("segments", [])) | set(how.get("segments", []))),
                "gameS": sorted(old.get("gameS", []) + how.get("gameS", []))}.items() if v}
    before = Holdout(rec)
    (d / MARK).write_text(json.dumps({**how, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1) + "\n")
    report(rec, before)


def report(rec: Path, before: Holdout | None = None) -> None:
    h = Holdout(rec)
    print(f"{rec.name}: held out {h.windows or 'nothing'} (game seconds)")
    for d in set_dirs(rec):
        items = json.loads((d / "items.json").read_text())
        times = item_times(items)
        n = sum(h.held(s, t) for s, t in times)
        m = json.loads((d / MARK).read_text()) if (d / MARK).exists() else None
        what = "held out" if m and m.get("all") else f"{n} held out, the rest training" if n else "training"
        unl = "" if (d / "labels.json").exists() else " (not labelled)"
        print(f"  {d.relative_to(rec)}: {len(times)} moments{unl}, {what}")
        if before is not None:
            newly = sum(h.held(s, t) and not before.held(s, t) for s, t in times)
            if newly and not (m and m.get("all")):
                print(f"    {newly} of them newly held out: left out of training from now on"
                      f" (models trained before may have learnt them)")


if __name__ == "__main__":
    a = sys.argv
    if len(a) == 2:
        report(Path(a[1]))
    elif len(a) == 4 and a[2] == "mark":
        mark(Path(a[1]), a[3], {"all": True})
    elif len(a) == 6 and a[2] == "mark" and a[4] == "segments":
        mark(Path(a[1]), a[3], {"segments": [int(x) for x in a[5].split(",")]})
    elif len(a) == 8 and a[2] == "mark" and a[4] == "seconds":  # a stretch of one segment, e.g. the end of a clip
        start = _starts(Path(a[1]))[int(a[5])]
        mark(Path(a[1]), a[3], {"gameS": [[round(start + float(a[6]), 3), round(start + float(a[7]), 3)]]})
    else:
        sys.exit(__doc__)
