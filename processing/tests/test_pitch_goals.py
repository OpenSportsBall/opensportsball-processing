# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Goals stay with the pitch map they are drawn on (processing/spikes/pitch.py).

No footage: a made-up camera and pitch, with goal frames painted into an
empty panorama.

    cd processing && uv run --extra dev pytest tests/test_pitch_goals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))

cv2 = pytest.importorskip("cv2")
import pitch as P  # noqa: E402

# rec7's mount, roughly: 6.4 m up, 4 m behind the near touchline of a 102 x 60 m pitch.
POSE = [0.0, -34.0, 6.4, 0.05, 0.06, -0.01, 102.0, 60.0, 1.0, 0.0, 0.0, 0.0]


def panorama(pose: P.PitchPose, goals: dict[int, tuple[float, float]]):
    """Dark turf with a white goal frame (centre z, width) on each given end's goal line."""
    pano = P.Pano()
    bg = np.full((pano.h, pano.w, 3), (40, 90, 40), np.uint8)
    for side, (z0, w) in goals.items():
        a = pose.angles3(*P.goal_frame(side * pose.L / 2, z0, w, P.GOAL_H * min(1.0, w / P.GOAL_W), step=0.02))
        x, y = pano.to_px(a[:, 0], a[:, 1])
        for xi, yi in zip(x.round().astype(int), y.round().astype(int)):
            cv2.circle(bg, (int(xi), int(yi)), 1, (255, 255, 255), -1)
    return pano, bg, np.zeros((pano.h, pano.w), np.uint8)


def post_tap(pose: P.PitchPose, side: int, z: float) -> dict:
    yaw, pitch = pose.angles(np.array([[side * pose.L / 2, z]]))[0]
    return {"yaw": float(yaw), "pitch": float(pitch)}


def test_rejected_goals_follow_the_pitch_that_is_kept():
    """rec7 in the app: goals found under a 128 m trial pitch were written at +-64 m and drawn
    on the 107 m pitch that was kept, 10 m past its goal lines."""
    trial, kept = P.PitchPose([*POSE[:6], 128.0, 68.0, *POSE[8:]]), P.PitchPose(POSE)
    found = [{"side": s, "x": s * 64.0, "z": 0.0, "width": 5.2, "height": 1.73, "valid": False} for s in (-1, 1)]
    got = P.goals_in(found, trial, kept)
    for g0, g in zip(found, got):
        seen = trial.angles(np.array([[g0["x"], g0["z"] - 2.6], [g0["x"], g0["z"] + 2.6]]))
        drawn = kept.angles(np.array([[g["x"], g["z"] - g["width"] / 2], [g["x"], g["z"] + g["width"] / 2]]))
        assert np.abs(seen - drawn).max() < 0.05, (g0, g)  # degrees: where it was seen
        assert g["valid"] is False and g["side"] == g0["side"]
    assert P.goals_in(found, kept, kept)[0]["x"] == -64.0  # same pose: unchanged


@pytest.mark.parametrize("width", [7.32, 4.88])
def test_a_tapped_post_sets_its_goals_place_and_width(width):
    pose = P.PitchPose(POSE)
    pano, bg, mask = panorama(pose, {-1: (0.0, width), 1: (0.0, width)})
    # taps a little off the posts, as a finger is
    taps = {"left near post base": post_tap(pose, -1, -width / 2 + 0.1),
            "right near post base": post_tap(pose, 1, -width / 2 - 0.1)}
    goals = P.goals_on_lines(bg, mask, pano, pose, taps)
    for g in goals:
        assert g["valid"] and g["from"] == "tap+frame", g
        assert g["x"] == pytest.approx(g["side"] * pose.L / 2, abs=0.01)  # on the goal line
        assert g["width"] == pytest.approx(width, abs=0.15), g
        assert abs(g["z"]) <= 0.2, g


def test_goals_without_taps_are_found_on_the_goal_lines():
    pose = P.PitchPose(POSE)
    pano, bg, mask = panorama(pose, {-1: (0.4, 7.32), 1: (-0.3, 7.32)})
    left, right = P.goals_on_lines(bg, mask, pano, pose, {})
    assert left["valid"] and right["valid"] and left["from"] == right["from"] == "frame"
    assert (left["width"], right["width"]) == (7.32, 7.32)
    assert left["z"] == pytest.approx(0.4, abs=0.15) and right["z"] == pytest.approx(-0.3, abs=0.15)


def test_a_tapped_post_gives_the_goal_when_its_frame_cant_be_seen():
    pose = P.PitchPose(POSE)
    pano, bg, mask = panorama(pose, {})  # no goals to see
    goals = P.goals_on_lines(bg, mask, pano, pose, {"left near post base": post_tap(pose, -1, -3.66),
                                                    "right near post base": {"skipped": True}})
    left, right = goals
    assert left["valid"] and left["from"] == "tap"
    assert left["z"] == 0.0 and left["width"] == pytest.approx(7.32, abs=0.05)
    assert right["valid"] is False and "from" not in right
