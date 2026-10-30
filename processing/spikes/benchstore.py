#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.10"]  # phase0.py's
# ///
"""Checks the bench store: the video segments every benchmark reads are
present and match their recording's manifest byte for byte.

    uv run processing/spikes/benchstore.py [footage/bench/rec3 ...] [--quick]

With no recording, checks every one under footage/bench/. The needed
segments are worked out from the recording itself: each must-see set's
clip.json (playbench.py) and each clip label set's items.json
(balleval.py), on both phones as phase0 paired them. --quick compares
sizes only, not sha256.

The store and its S3 mirror are described in docs/TESTING-PROCESSING.md
(section 6).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase0  # noqa: E402

STORE = Path("footage/bench")


def main_segments(segs: list[dict], seg: int, start: float, seconds: float) -> set[int]:
    """Main segments a clip from `start` s into `seg`, `seconds` long, reads
    (it carries on into the following segments, as follow.game_stream does)."""
    out, t, k, left = set(), start, seg, seconds
    while left > 0 and k < len(segs):
        dur = segs[k]["durationMs"] / 1000
        if t < dur:
            out.add(k)
            left -= dur - t
            t = 0.0
        else:
            t -= dur
        k += 1
    return out


def needed(rec: Path) -> dict[int, list[str]]:
    """Main segment index -> the clips that read it."""
    segs = json.loads((rec / "manifest.json").read_text())
    segs = {d["role"]: d for d in segs["devices"]}["main"]["segments"]
    out: dict[int, list[str]] = {}
    for f in sorted(rec.glob("playeval/*/clip.json")):
        c = json.loads(f.read_text())
        for k in main_segments(segs, int(c["segment"]), float(c["startS"]), float(c["seconds"])):
            out.setdefault(k, []).append(f.parent.name)
    for f in sorted(rec.glob("labels_*/items.json")):
        for it in json.loads(f.read_text()):
            for k in main_segments(segs, int(it["segment"]), float(it["t"]), 1.0):
                if f.parent.name not in out.setdefault(k, []):
                    out[k].append(f.parent.name)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def check(rec: Path, quick: bool) -> bool:
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    need = needed(rec)
    ok = True
    print(f"{rec}: Main segments {sorted(need)}")
    for k in sorted(need):
        for role in ("main", "secondary"):
            s = devs[role]["segments"][k]
            path = rec / s["objectKey"]
            if not path.exists():
                status, ok = "MISSING", False
            elif path.stat().st_size != s["bytes"]:
                status, ok = f"SIZE {path.stat().st_size} != {s['bytes']}", False
            elif not quick and sha256(path) != s["sha256"]:
                status, ok = "SHA256 MISMATCH", False
            else:
                status = "ok"
            print(f"  {k:2d} {role:9s} {path.name}  {status}  ({', '.join(need[k])})", flush=True)
    return ok


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    recs = [Path(a) for a in args] or sorted(p for p in STORE.iterdir() if (p / "manifest.json").exists())
    ok = all([check(rec, "--quick" in sys.argv) for rec in recs])
    print("all present and matching" if ok else "PROBLEMS above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
