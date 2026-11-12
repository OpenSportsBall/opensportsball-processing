# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""The stitcher's estimates that the phones' lenses can't have made are
dropped (phase0.calibrate), with a stitcher stood in for OpenCV's: the
11 Nov 2026 night game's pitch step kept one of 20 and 10 px at 960 wide
(lenses of ~104 deg), and the pitch picture was a patch of turf.

    cd processing && uv run --extra dev pytest tests/test_calibration_plausible.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import phase0  # noqa: E402

W = 960
DEVS = ({"video": {"horizontalFovDeg": 106.2}}, {"video": {"horizontalFovDeg": 103.6}})  # secondary, main


class _Cam:
    def __init__(self, focal, yaw):
        self.focal = focal
        self.R = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0].astype(np.float32)


def _stitchers(monkeypatch, focals):
    """cv2.Stitcher_create gives, in turn, stitchers that estimate these focal pairs."""
    todo = iter(focals)

    class Stitcher:
        def estimateTransform(self, images):
            self.f = next(todo)
            return cv2.Stitcher_OK

        def cameras(self):
            return [_Cam(self.f[0], -0.7), _Cam(self.f[1], 0.7)]

    monkeypatch.setattr(cv2, "Stitcher_create", lambda *_: Stitcher())


def _pairs(n):
    frame = np.zeros((W * 9 // 16, W, 3), np.uint8)
    return [(frame, frame)] * n


def test_expected_focals_from_the_manifest():
    sec, main = phase0.expected_focals(*DEVS, W)
    assert sec == pytest.approx(360.4, abs=0.5) and main == pytest.approx(377.7, abs=0.5)
    assert phase0.expected_focals({}, {"video": {}}, W) == [None, None]


def test_a_nonsense_estimate_is_dropped(monkeypatch):
    # Two estimates, as that night: the median can't tell which is right.
    _stitchers(monkeypatch, [(9.6, 20.0), (375.9, 350.8)])
    st, cal = phase0.calibrate(_pairs(2), phase0.expected_focals(*DEVS, W))
    assert cal["ok"] and cal["pairsOk"] == 1 and cal["focalPx"] == [375.9, 350.8]


def test_only_nonsense_is_a_failed_calibration(monkeypatch):
    _stitchers(monkeypatch, [(9.6, 20.0), (5000.0, 360.0)])
    st, cal = phase0.calibrate(_pairs(2), phase0.expected_focals(*DEVS, W))
    assert st is None and cal == {"ok": False, "pairsTried": 2, "pairsOk": 0}


def test_without_a_reported_lens_any_real_lens_passes(monkeypatch):
    # Older recordings: no field of view in the manifest.
    _stitchers(monkeypatch, [(9.6, 20.0), (300.0, 700.0)])
    st, cal = phase0.calibrate(_pairs(2))
    assert cal["pairsOk"] == 1 and cal["focalPx"] == [300.0, 700.0]
