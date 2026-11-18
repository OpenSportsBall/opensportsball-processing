#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.10"]  # phase0.py's
# ///
"""Checks the bench store: the video segments every benchmark reads are
present and match their recording's manifest byte for byte. A file the
sets read only the start of may be cut there (the manifest's
"trimmedToMs", with the upload's own size and hash kept as
"originalBytes" and "originalSha256"; "durationMs" stays the upload's, so
the game's clock is unchanged): it is checked to hold what is read.

    uv run processing/spikes/benchstore.py [footage/bench/rec3 ...] [--quick]

With no recording, checks every one under footage/bench/. The needed
files are worked out from the recording itself: each must-see set's
clip.json (playbench.py) and each clip label set's items.json
(balleval.py), on each phone, from the file it has for each moment. --quick compares
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


LEAD_S = 22.0   # a clip is read from this long before it starts: pass A's background warm-up (follow.WARMUP_S) and the decoder's pre-roll
TAIL_S = 2.0    # ... to this long after it ends
LABEL_S = 5.0   # a labelled moment: this long either side of it
STEP_S = 0.25


def needed(rec: Path, manifest: dict, report: dict) -> dict[str, dict]:
    """objectKey -> {"from", "to" (s in that file), "sets"} for every file a
    must-see set (clip.json: playbench.py) or a clip label set (items.json:
    balleval.py) reads, on each phone, from the file that phone has for
    each moment (phase0.phone_at: Secondary's file k needn't cover Main's
    segment k, and where it has no footage nothing is read)."""
    devs = phase0.paired_devices(manifest, report)
    segs = devs["main"]["segments"]
    starts = [phase0.game_start(manifest, k) for k in range(len(segs) + 1)]
    spans: list[tuple[float, float, str]] = []
    for f in sorted(rec.glob("playeval/*/clip.json")):
        c = json.loads(f.read_text())
        g = starts[int(c["segment"])] + float(c["startS"])
        spans.append((g - LEAD_S, g + float(c["seconds"]) + TAIL_S, f.parent.name))
    for f in sorted(rec.glob("labels_*/items.json")):
        for it in json.loads(f.read_text()):
            g = starts[int(it["segment"])] + float(it["t"])
            spans.append((g - LABEL_S, g + LABEL_S, f.parent.name))
    out: dict[str, dict] = {}
    for a, b, name in spans:
        t = max(0.0, a)
        while t <= min(b, starts[-1] - 1e-6):
            k = max(i for i in range(len(segs)) if starts[i] <= t)
            for role in ("main", "secondary"):
                at = phase0.phone_at(devs, report, role, k, t - starts[k])
                if at is not None:
                    u = out.setdefault(at[0], {"from": at[1], "to": at[1], "sets": []})
                    u["from"], u["to"] = min(u["from"], at[1]), max(u["to"], at[1])
                    if name not in u["sets"]:
                        u["sets"].append(name)
            t += STEP_S
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
    by_key = {s["objectKey"]: (d["role"], s) for d in manifest["devices"] for s in d["segments"]}
    need = needed(rec, manifest, report)
    ok = True
    print(f"{rec}: {len(need)} files")
    for key in sorted(need, key=lambda k: (by_key[k][0], by_key[k][1]["index"])):
        role, s = by_key[key]
        path, u = rec / key, need[key]
        kept = s.get("trimmedToMs", s["durationMs"]) / 1000  # a file cut to what is read (docs/DATA.md)
        if not path.exists():
            status, ok = "MISSING", False
        elif path.stat().st_size != s["bytes"]:
            status, ok = f"SIZE {path.stat().st_size} != {s['bytes']}", False
        elif u["to"] > kept:
            status, ok = f"TRIMMED TOO SHORT: read to {u['to']:.0f} s, the file has {kept:.0f} s", False
        elif not quick and sha256(path) != s["sha256"]:
            status, ok = "SHA256 MISMATCH", False
        else:
            status = "ok"
        cut = f", cut to {kept:.0f} s" if "trimmedToMs" in s else ""
        print(f"  {role:9s} {path.name}  {status}  read {u['from']:.0f}-{u['to']:.0f} of {s['durationMs'] / 1000:.0f} s{cut}"
              f"  ({', '.join(u['sets'])})", flush=True)
    return ok


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    recs = [Path(a) for a in args] or sorted(p for p in STORE.iterdir() if (p / "manifest.json").exists())
    ok = all([check(rec, "--quick" in sys.argv) for rec in recs])
    print("all present and matching" if ok else "PROBLEMS above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
