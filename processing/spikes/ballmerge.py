#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Merges balldata.py datasets from several recordings into one for balltrain.py.

    python3 processing/spikes/ballmerge.py <out dir> footage/rec3/balldata footage/rec5-game/balldata*3 ...

Each split (train, valid, test) is the union of the inputs' splits: images
are linked (not copied) under a per-input prefix, and the COCO annotations
renumbered. Each recording keeps its own train/valid split, so no moment
is on both sides. "<dir>*N" repeats that input's train split N times, to
weight a small new set (a ball colour the rest hasn't seen) against a
large old one; valid and test are never repeated.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(out: Path, specs: list[str]) -> None:
    inputs = [(Path(s.split("*")[0]), int(s.split("*")[1]) if "*" in s else 1) for s in specs]
    for split in ("train", "valid", "test"):
        dest = out / split
        dest.mkdir(parents=True, exist_ok=True)
        merged = {"images": [], "annotations": [], "categories": None}
        for k, (src, repeat) in enumerate(inputs):
            for r in range(repeat if split == "train" else 1):
                coco = json.loads((src / split / "_annotations.coco.json").read_text())
                if merged["categories"] is None:
                    merged["categories"] = coco["categories"]
                    merged.update({key: v for key, v in coco.items() if key not in merged})
                elif [c["name"] for c in coco["categories"]] != [c["name"] for c in merged["categories"]]:
                    sys.exit(f"{src}: categories differ")
                prefix = f"{k}_{src.parent.name}_" + (f"r{r}_" if r else "")
                image_id = {}
                for im in coco["images"]:
                    new = {**im, "id": len(merged["images"]), "file_name": prefix + im["file_name"]}
                    image_id[im["id"]] = new["id"]
                    merged["images"].append(new)
                    link = dest / new["file_name"]
                    if not link.exists():
                        os.symlink((src / split / im["file_name"]).resolve(), link)
                for a in coco["annotations"]:
                    merged["annotations"].append({**a, "id": len(merged["annotations"]), "image_id": image_id[a["image_id"]]})
        (dest / "_annotations.coco.json").write_text(json.dumps(merged))
        print(f"{split}: {len(merged['images'])} images, {len(merged['annotations'])} boxes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
