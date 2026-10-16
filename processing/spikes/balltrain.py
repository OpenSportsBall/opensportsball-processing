#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["rfdetr[train]>=1.11", "torch>=2.4"]
# ///
"""Fine-tunes RF-DETR on the ball dataset from balldata.py.

    uv run processing/spikes/balltrain.py footage/rec3/balldata <out dir> [epochs] [device]

Starts from the COCO-trained RF-DETR Medium (it already knows "sports
ball"), one class "ball", at 640 px (the tile size, so small balls aren't
shrunk). Runs on Apple's GPU ("mps") locally, "cuda" in the cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(data: Path, out: Path, epochs: int, device: str) -> None:
    import rfdetr

    model = rfdetr.RFDETRMedium()
    # Batch 2 x 8 accumulation (effective 16): batch 4 pushed a 32 GB Mac into swap once other work ran.
    model.train(dataset_dir=str(data), output_dir=str(out), epochs=epochs, batch_size=2, grad_accum_steps=8,
                lr=1e-4, resolution=640, device=device, early_stopping=True, early_stopping_patience=8)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40,
         sys.argv[4] if len(sys.argv) > 4 else "mps")
