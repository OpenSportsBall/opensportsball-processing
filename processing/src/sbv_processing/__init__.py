# SPDX-FileCopyrightText: 2026 The SportsBallVideo Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""SportsBallVideo processing pipeline.

Stages, run in one AWS Batch GPU job:

1. sync       cross-correlate the two audio tracks to get the frame offset
2. calibrate  homography, seam and gain from sampled frames -> remap tables
3. pass A     decode both videos, stitch, run the detector, write detections
4. path       smooth virtual-camera path over the whole game (CPU)
5. pass B     decode again, stitch only the crop region, encode 1080p
"""

__version__ = "0.1.0"
