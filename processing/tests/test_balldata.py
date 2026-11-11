# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""balldata.py: unmarked suggestions off the pitch stay out of training,
without footage:

    uv run --with pytest --with numpy --with scipy --with opencv-python-headless --with torch \
        pytest processing/tests/test_balldata.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import balldata  # noqa: E402

POSE = SimpleNamespace(L=100.0, W=60.0)


def test_off_pitch_is_outside_the_lines_and_on_the_ground():
    assert not balldata.off_pitch(POSE, 0.0, 0.0)
    assert not balldata.off_pitch(POSE, 49.9, -29.9)  # just inside the corner
    assert balldata.off_pitch(POSE, 50.5, 0.0)  # behind the goal line
    assert balldata.off_pitch(POSE, 0.0, -31.0)  # the bench side
    # Above the horizon (lights, stands): no ball on another pitch, so still a negative.
    assert not balldata.off_pitch(POSE, float("nan"), float("nan"))


def test_left_out_keeps_marked_balls_and_suggestions_on_the_pitch():
    # Pixels right of x = 1000 are off the pitch here.
    is_off = lambda x, y: x > 1000  # noqa: E731
    on, stray, marked = {"x": 500, "y": 900}, {"x": 1500, "y": 900}, {"x": 1800, "y": 905}
    balls = [{"x": 1803, "y": 900}]  # a marked ball off the pitch (a throw-in) is still a ball
    assert balldata.left_out([on, stray, marked], balls, is_off, 20) == [stray]
    assert balldata.left_out([on, stray], [], is_off, 20) == [stray]


def test_tile_clear_counts_a_margin_beyond_the_tile():
    stray = [{"x": 1000, "y": 1000}]
    assert not balldata.tile_clear(600, 600, stray)  # inside
    assert not balldata.tile_clear(1000 + balldata.TILE_PAD - 1, 600, stray)  # just left of the tile's edge
    assert not balldata.tile_clear(1000 - balldata.TILE - balldata.TILE_PAD + 1, 600, stray)  # just right of it
    assert balldata.tile_clear(1000 + balldata.TILE_PAD + 1, 600, stray)
    assert balldata.tile_clear(600, 600, [])
