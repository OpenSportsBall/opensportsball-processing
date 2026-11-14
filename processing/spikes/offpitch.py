#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Triage of the unmarked suggestions off the pitch in a recording's random
label set: is each one a ball or not?

    uv run processing/spikes/offpitch.py list footage/rec3 [set]      # crops, sheets and items.json
    python3 processing/spikes/offpitch_page.py footage/rec3 [port]    # the review page (no dependencies)
    uv run processing/spikes/offpitch.py premark footage/rec3         # take in draft verdicts (premark_*.json)
    uv run processing/spikes/offpitch.py count footage/rec3           # verdicts so far

balldata.py and `balljudge.py data` leave an unmarked suggestion outside
the pitch lines out of training, because it may be someone else's ball.
That also throws away the lights, cars and kit that taught the models what
is not a ball (docs/BACKLOG.md, 2026-11-13). A verdict here settles it:

  ball     a ball (a spare, a warm-up, the next pitch's): a positive, as any marked ball
  not      not a ball: a negative again
  unsure   can't tell: left out of training, as before

`premark` takes in drafts; only a verdict reviewed on the page ("by":
"reviewed") is used in training. [set] is the label set's folder (default `labels`, the random set; a clip
set such as `labels_seg6` works the same). Verdicts live in the label set, <recording>/<set>/offpitch.json:
{"<frame id>": [{"x", "y", "verdict", "by": "claude" | "reviewed"}]},
matched to a suggestion by its pixel. The crops and sheets are working
files in <recording>/offpitch_triage/ (offpitch_triage_<set>/ for another
set; not shared).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import balldata  # noqa: E402

NEAR_PX = 20        # a suggestion this close to a marked ball is that ball (as balldata.py)
WIDE, TIGHT, CELL = 320, 56, 160   # source px of the wide and tight views; each drawn at CELL px
VERDICTS = ("ball", "not", "unsure")


def triage_dir(rec: Path, lab: str = "labels") -> Path:
    """The working files of a label set's triage (crops, sheets, items.json)."""
    return rec / ("offpitch_triage" if lab == "labels" else f"offpitch_triage_{lab}")


def load_verdicts(rec: Path, lab: str = "labels") -> dict[str, list[dict]]:
    return balldata.load_verdicts(rec / lab)


def frames_of(rec: Path, lab: str = "labels"):
    lab = rec / lab
    items = json.loads((lab / "items.json").read_text())
    labels = json.loads((lab / "labels.json").read_text())
    for it in items:
        for f in (it["frames"] if it.get("kind") == "pair" else [it]):
            v = labels.get(f["id"])
            if v and v["status"] != "unsure":
                yield f, balldata.balls_of(v)


def view(img: np.ndarray, x: float, y: float, side: int, mark: bool) -> np.ndarray:
    pad = side
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    x0, y0 = int(round(x - side / 2)) + pad, int(round(y - side / 2)) + pad
    c = cv2.resize(big[y0:y0 + side, x0:x0 + side], (CELL, CELL),
                   interpolation=cv2.INTER_AREA if side > CELL else cv2.INTER_CUBIC)
    if mark:  # corner ticks around the spot, clear of it
        r, m = int(14 * CELL / side) + 6, CELL // 2
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.line(c, (m + sx * r, m + sy * r), (m + sx * (r - 5), m + sy * r), (0, 255, 255), 1)
            cv2.line(c, (m + sx * r, m + sy * r), (m + sx * r, m + sy * (r - 5)), (0, 255, 255), 1)
    return c


def cmd_list(rec: Path, lab: str = "labels") -> None:
    import balljudge
    import follow
    import phase0
    import pitch as P

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    cm, cs = follow.setup_cameras(rec, report, devs, [report["upsideDown"]["main"], report["upsideDown"]["secondary"]])
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    pose = P.load_pose(rec)
    diam_at = balljudge.expected_at(rec)
    judge = balljudge.load_judge(rec)
    out = triage_dir(rec, lab)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    items = []
    for f, balls in frames_of(rec, lab):
        img = None
        for i, c in enumerate(f["candidates"]):
            if any(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) <= NEAR_PX for b in balls):
                continue
            gx, gz = balldata.ground_xz(cams[f["phone"]], pose, c["x"], c["y"])
            if not balldata.off_pitch(pose, gx, gz):
                continue
            if img is None:
                img = cv2.imread(str(rec / lab / f["image"]))
            d = float(diam_at(f["phone"], c["x"], c["y"]))
            j = float(judge(np.stack([balljudge.crop_at(img, c["x"], c["y"], d)]))[0])
            cid = f"{f['id']}.{i}"
            cv2.imwrite(str(out / "crops" / f"{cid}.jpg"),
                        np.hstack([view(img, c["x"], c["y"], WIDE, True), view(img, c["x"], c["y"], TIGHT, False)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            items.append({"id": cid, "frame": f["id"], "phone": f["phone"], "x": c["x"], "y": c["y"], "conf": c.get("conf"),
                          "judge": round(j, 3), "expectPx": round(d, 1),
                          "outsideM": round(float(max(abs(gx) - pose.L / 2, abs(gz) - pose.W / 2)), 1)})
    items.sort(key=lambda it: -it["judge"])
    (out / "items.json").write_text(json.dumps(items, indent=1))
    # Contact sheets, in items.json order: 5 x 8 crops, each with its number.
    cols, rows = 5, 8
    for s in range(0, len(items), cols * rows):
        sheet = np.zeros((rows * (CELL + 18), cols * (2 * CELL + 8), 3), np.uint8)
        for k, it in enumerate(items[s:s + cols * rows]):
            r, c = divmod(k, cols)
            y0, x0 = r * (CELL + 18), c * (2 * CELL + 8)
            sheet[y0 + 18:y0 + 18 + CELL, x0:x0 + 2 * CELL] = cv2.imread(str(out / "crops" / f"{it['id']}.jpg"))
            cv2.putText(sheet, f"#{s + k}  j{it['judge']:.2f} c{it['conf'] or 0:.2f} {it['expectPx']:.0f}px {it['outsideM']:.0f}m out",
                        (x0 + 2, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"sheet_{s // (cols * rows):02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{rec.name}: {len(items)} unmarked suggestions off the pitch in {len({it['frame'] for it in items})} frames"
          f" -> {out}")


def cmd_premark(rec: Path, lab: str = "labels") -> None:
    """Draft verdicts from <recording>/offpitch_triage/premark_*.json
    ({item number in items.json: verdict}) into offpitch.json as "by":
    "claude". A verdict someone reviewed is never overwritten."""
    import offpitch_page

    items = json.loads((triage_dir(rec, lab) / "items.json").read_text())
    reviewed = {it["id"] for it in offpitch_page.merged(rec, lab) if it["by"] == "reviewed"}
    changes = {}
    for p in sorted(triage_dir(rec, lab).glob("premark_*.json")):
        for k, verdict in json.loads(p.read_text()).items():
            if verdict not in VERDICTS:
                sys.exit(f"{p}: {k} has verdict {verdict!r}")
            if items[int(k)]["id"] not in reviewed:
                changes[items[int(k)]["id"]] = verdict
    offpitch_page.save(rec, changes, by="claude", lab=lab)
    print(f"{rec.name}: {len(changes)} draft verdicts taken in ({len(items) - len(changes) - len(reviewed)} items without one)")
    cmd_count(rec, lab)


def cmd_count(rec: Path, lab: str = "labels") -> None:
    v = load_verdicts(rec, lab)
    n = {k: sum(x["verdict"] == k for xs in v.values() for x in xs) for k in VERDICTS}
    by = {k: sum(x.get("by") == k for xs in v.values() for x in xs) for k in ("claude", "reviewed")}
    print(f"{rec.name}: {n} ({by})")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["list", rec, *lab]:
            cmd_list(Path(rec), *lab)
        case ["premark", rec, *lab]:
            cmd_premark(Path(rec), *lab)
        case ["count", rec, *lab]:
            cmd_count(Path(rec), *lab)
        case _:
            sys.exit(__doc__)
