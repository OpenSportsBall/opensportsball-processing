# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Regression test for pitch and goal mapping (processing/spikes/pitch.py).

Reruns the whole mapping (line fit from the starting guess, goal detection,
refit with the goals) on saved background panoramas and compares it with
results that were checked by eye. Fixtures hold footage, so they live
outside git: footage/fixtures/pitch/<name>/ (or $OSB_PITCH_FIXTURES), each
with background.png and expected.json, made by

    uv run processing/spikes/pitch.py <recording> fixture footage/fixtures/pitch/<name>

Run:

    uv run --with pytest --with numpy --with scipy --with opencv-python-headless --with torch \
        pytest processing/tests/test_pitch_regression.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "processing" / "spikes"))

FIXTURES = Path(os.environ.get("OSB_PITCH_FIXTURES", ROOT / "footage" / "fixtures" / "pitch"))
CASES = sorted(p for p in FIXTURES.glob("*") if (p / "expected.json").exists()) if FIXTURES.exists() else []

KEY_POINT_DEG = 0.1      # every key point within this of where it was (~3 panorama px)
LENGTH_M, WIDTH_M = 1.0, 1.0
HEIGHT_M = 0.2           # camera height
GOAL_SIZE_M = 0.3
LINES_PX = 3.5           # an end's markings from the paint (fixture: 1.8 and 2.8)
FRAME_PX = 1.5           # a goal frame from white structure (fixture: 0.3 and 0.06)


@pytest.fixture(scope="module", params=CASES, ids=[p.name for p in CASES])
def mapped(request):
    import cv2
    import pitch as P

    case = request.param
    expected = json.loads((case / "expected.json").read_text())
    bg = cv2.imread(str(case / "background.png"))
    lines, goals, final = P.map_pitch(bg, expected["startPose"])
    return P, expected, lines, goals, final


if not CASES:
    pytest.skip(f"no pitch fixtures under {FIXTURES}", allow_module_level=True)


def test_both_goals_found(mapped):
    _, expected, _, goals, _ = mapped
    assert [g["valid"] for g in goals] == [True, True], goals
    for got, want in zip(goals, expected["goals"]):
        assert abs(got["width"] - want["width"]) <= GOAL_SIZE_M
        assert abs(got["height"] - want["height"]) <= GOAL_SIZE_M
        assert got["endLinesPx"] <= LINES_PX, f"{got['side']} end's markings {got['endLinesPx']} px from the paint"
        assert got["frameToWhitePx"] <= FRAME_PX, f"{got['side']} goal frame {got['frameToWhitePx']} px off"


def test_pitch_size_and_camera(mapped):
    P, expected, _, _, final = mapped
    want = P.pose_from_json(expected["final"])
    assert abs(final.L - want.L) <= LENGTH_M, (final.L, want.L)
    assert abs(final.W - want.W) <= WIDTH_M, (final.W, want.W)
    assert abs(final.h - want.h) <= HEIGHT_M, (final.h, want.h)


def test_key_points(mapped):
    P, expected, _, goals, final = mapped
    got = P.key_points(final, goals)
    worst = {k: float(np.hypot(*(np.subtract(got[k], v)))) for k, v in expected["keyPointsDeg"].items()}
    bad = {k: round(d, 3) for k, d in worst.items() if d > KEY_POINT_DEG}
    assert not bad, f"key points moved more than {KEY_POINT_DEG} deg: {bad}"


def test_lines_only_fit_close(mapped):
    """The line fit alone (before the goals) shouldn't wander either: the
    goal search starts from it."""
    P, expected, lines, _, _ = mapped
    want = P.pose_from_json(expected["linesOnly"])
    got = P.key_points(lines, [])
    ref = P.key_points(want, [])
    worst = max(float(np.hypot(*np.subtract(got[k], ref[k]))) for k in ref)
    assert worst <= 3 * KEY_POINT_DEG, f"lines-only key points moved up to {worst:.3f} deg"
