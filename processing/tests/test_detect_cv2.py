# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""detect.py's GPU stand-ins for two OpenCV calls give OpenCV's numbers to
the bit: roles.py prepares its frames with them, and teams.py's kit model
was fitted on OpenCV's colours."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parents[1] / "spikes"))
import detect  # noqa: E402


def nv12(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h * 3 // 2, w), dtype=np.uint8)
    a[:8] = np.repeat(np.arange(0, 256, 256 // w, dtype=np.uint8)[None, :w], 8, 0)  # every luma level, the ends included
    a[h:h + 2, :8] = [0, 255, 255, 0, 0, 0, 255, 255]  # the chroma's corners, where the result leaves 0..255
    return a


def test_nv12_to_bgr_is_opencvs():
    h, w = 64, 128
    for seed in range(3):
        a = nv12(h, w, seed)
        want = cv2.cvtColor(a, cv2.COLOR_YUV2BGR_NV12)
        t = torch.from_numpy(a)
        got = detect.nv12_to_bgr_cv2(t[:h], t[h:].view(h // 2, w // 2, 2)).numpy()
        assert (got == want).all()


def test_halving_is_opencvs_area_resize():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    want = cv2.resize(img, (80, 45), interpolation=cv2.INTER_AREA)
    assert (detect.halve_area(torch.from_numpy(img)).numpy() == want).all()
