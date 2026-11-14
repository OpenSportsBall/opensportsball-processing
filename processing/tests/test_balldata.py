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


def _plan(verdicts, frames):
    return balldata.plan_verdicts(verdicts, frames)


def test_triage_verdicts_settle_an_off_pitch_suggestion():
    is_off = lambda x, y: x > 1000  # noqa: E731
    on, spare, lamp, blob, untouched = ({"x": 500, "y": 900}, {"x": 1500, "y": 900}, {"x": 1700, "y": 800},
                                        {"x": 1900, "y": 850}, {"x": 2100, "y": 870})
    cands = [on, spare, lamp, blob, untouched]
    verdicts = {"f1": [{"x": 1500.4, "y": 899.8, "verdict": "ball", "by": "reviewed"},
                       {"x": 1700, "y": 800, "verdict": "not", "by": "reviewed"},
                       {"x": 2100, "y": 870, "verdict": "not", "by": "claude"},  # a draft nobody reviewed: no effect
                       {"x": 1900, "y": 850, "verdict": "unsure", "by": "reviewed"}]}
    plan = _plan(verdicts, [("f1", "main", "train")])
    match = {"x": 300, "y": 950, "inPlay": True}
    balls, cleared, unused, hide = balldata.with_verdicts(cands, [match], plan["f1"])
    # "ball": a spare ball like any marked one; "not": a negative again; "unsure" and no verdict: still left out.
    assert balls == [match, {"x": 1500.4, "y": 899.8, "inPlay": False}]
    assert cleared == [lamp] and unused == [] and hide == []
    assert balldata.left_out(cands, balls, is_off, 20, cleared) == [blob, untouched]
    # Without verdicts nothing changes.
    assert balldata.with_verdicts(cands, [match], _plan({}, [("f1", "main", "train")])["f1"]) == ([match], [], [], [])
    assert balldata.left_out(cands, [match], is_off, 20) == [spare, lamp, blob, untouched]


def test_a_spot_counts_ten_times_and_its_balls_stay_on_one_side_of_the_split():
    # One ball lying in the same place in 40 frames of Main (pixels jitter a little), 30 train and 10 valid;
    # a lamp in the same 40; the other phone sees neither.
    frames = [(f"f{i:02d}", "main", "valid" if i % 4 == 0 else "train") for i in range(40)]
    frames += [(f"g{i:02d}", "secondary", "train") for i in range(5)]
    verdicts = {fid: [{"x": 1500 + i % 3, "y": 900, "verdict": "ball", "by": "reviewed"},
                      {"x": 3000, "y": 400 + i % 2, "verdict": "not", "by": "reviewed"}]
                for i, (fid, phone, _) in enumerate(frames) if phone == "main"}
    plan = _plan(verdicts, frames)
    split = dict((fid, sp) for fid, _, sp in frames)
    pos = [fid for fid in plan if plan[fid]["ball"]]
    assert 0 < len(pos) <= balldata.SPOT_CAP
    assert len({split[fid] for fid in pos}) == 1  # train or valid, never both
    # In every other frame of that phone the ball is hidden, never background; the other phone is untouched.
    assert all(plan[fid]["ball"] or plan[fid]["hide"] for fid, phone, _ in frames if phone == "main")
    assert all(plan[fid] == {"ball": [], "not": [], "hard": [], "hide": []} for fid, phone, _ in frames if phone != "main")
    # The lamp is a negative in all 40 frames, a hard negative in ten of them.
    assert sum(bool(plan[fid]["not"]) for fid in plan) == 40
    assert sum(bool(plan[fid]["hard"]) for fid in plan) == balldata.SPOT_CAP


def test_a_ball_seen_often_is_hidden_where_it_has_no_verdict():
    frames = [(f"f{i:02d}", "main", "train") for i in range(12)]
    often = {fid: [{"x": 800, "y": 1000, "verdict": "ball", "by": "reviewed"}] for fid, _, _ in frames[:6]}
    plan = _plan(often, frames)
    assert all(plan[fid]["hide"] == [{"x": 800.0, "y": 1000.0}] for fid, _, _ in frames[6:])
    # A suggestion at that spot is left out anyway, so it isn't hidden twice.
    assert balldata.with_verdicts([{"x": 801, "y": 1000}], [], plan["f08"])[3] == []
    # A ball seen once (a throw-in fetched, a ball kicked past) is hidden nowhere else.
    once = _plan({"f00": [{"x": 800, "y": 1000, "verdict": "ball", "by": "reviewed"}]}, frames)
    assert all(not once[fid]["hide"] for fid, _, _ in frames)
