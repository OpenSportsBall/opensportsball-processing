# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""gpu.MOG2 is OpenCV's BackgroundSubtractorMOG2 moved to the GPU: it must
give the mask OpenCV gives, or the follow cam sees different players in
the cloud than on a Mac."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import gpu  # noqa: E402


def frames(n, h=90, w=160, seed=0):
    """A noisy still background with two players crossing it (one stops), a
    shadow under the first, and the light changing on one side halfway."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(40, 200, (h, w, 3)).astype(np.float32), (0, 0), 6)
    for i in range(n):
        f = base + rng.normal(0, 2.0, base.shape)
        x = int(10 + 1.5 * i) % (w - 12)
        f[30:50, x:x + 8] = (220, 30, 40)
        f[50:56, x:x + 12] *= 0.7
        x2 = min(60 + i, 120)
        f[60:85, x2:x2 + 9] = (20, 200, 230)
        if i > n // 2:
            f[:, : w // 3] *= 1.08
        yield np.clip(f, 0, 255).round().astype(np.uint8)


def test_mog2_gives_opencvs_mask_and_background():
    ref = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=True)
    mine = gpu.MOG2(90, 160, torch.device("cpu"))
    kinds = set()
    for i, f in enumerate(frames(240)):
        # As follow.py uses it: OpenCV's own rate while warming up, a quick
        # relearn for a while, then about a minute.
        rate = -1 if i < 100 else 0.02 if 150 <= i < 170 else 1 / 1800
        want = ref.apply(f, learningRate=rate)
        got = mine.apply(torch.from_numpy(f), rate).numpy()
        assert (want == got).all(), f"frame {i}: {(want != got).sum()} pixels differ"
        kinds |= set(np.unique(want).tolist())
    assert kinds == {0, 127, 255}  # the frames did exercise background, shadow and foreground
    assert np.abs(ref.getBackgroundImage().astype(int) - mine.background_image().astype(int)).max() <= 1


def test_morphology_matches_opencv():
    rng = np.random.default_rng(1)
    m = (rng.random((40, 60)) > 0.7).astype(np.uint8)
    t = torch.from_numpy(m).float()
    k3, k7 = np.ones((3, 3), np.uint8), np.ones((7, 7), np.uint8)
    assert (gpu.dilate(t, 7).numpy() == cv2.dilate(m, k7)).all()
    assert (gpu.dilate(t, 3, iterations=2).numpy() == cv2.dilate(m, k3, iterations=2)).all()
    opened = gpu.dilate(gpu.erode(t, 3), 3).numpy()
    assert (opened == cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)).all()
