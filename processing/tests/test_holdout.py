# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""holdout.py: held-out ball labels stay out of training, without footage:

    uv run --with pytest pytest processing/tests/test_holdout.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import holdout  # noqa: E402

SEG_S = 300.0  # five-minute segments


def make_rec(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    segs = [{"index": i, "durationMs": int(SEG_S * 1000)} for i in range(20)]
    (rec / "manifest.json").write_text(json.dumps({"devices": [{"role": "main", "segments": segs}]}))
    return rec


def make_set(rec, name, moments, mark=None):
    d = rec / name
    d.mkdir(parents=True)
    (d / "items.json").write_text(json.dumps([{"id": f"m{i}", "segment": s, "t": t} for i, (s, t) in enumerate(moments)]))
    if mark is not None:
        (d / "heldout.json").write_text(json.dumps(mark))
    return d


def test_a_held_out_clip_holds_its_time_for_every_set(tmp_path):
    rec = make_rec(tmp_path)
    make_set(rec, "labels_seg9", [(9, 60 + 2 * i) for i in range(60)], {"all": True})
    h = holdout.Holdout(rec)
    assert h.windows == [[9 * SEG_S + 60 - 20, 9 * SEG_S + 178 + 20]]
    assert h.held(9, 118.78)          # rec3's random-set moment inside seg9's clip
    assert h.held(9, 45)              # within the margin before it
    assert not h.held(9, 230)         # past the margin
    assert not h.held(3, 100)


def test_segments_and_game_seconds(tmp_path):
    rec = make_rec(tmp_path)
    make_set(rec, "labels", [(2, 10), (5, 10)], {"segments": [5]})
    make_set(rec, "engine/labels/ball-1", [(0, 10)], {"gameS": [[450, 600]]})
    h = holdout.Holdout(rec)
    assert h.held(5, 0) and h.held(5, 299) and not h.held(6, 1)
    assert h.held(1, 160) and not h.held(1, 140)   # 450-600 game s is segment 1, 150-300 s


def test_a_held_out_set_is_refused_as_training(tmp_path):
    rec = make_rec(tmp_path)
    make_set(rec, "labels_seg9", [(9, 60)], {"all": True})
    make_set(rec, "labels_seg3", [(3, 225)])
    with pytest.raises(holdout.HeldOut):
        holdout.Holdout(rec).refuse_training_set("labels_seg9")
    holdout.Holdout(rec).refuse_training_set("labels_seg3")  # not held out: fine


def test_data_built_before_a_mark_is_refused(tmp_path):
    rec = make_rec(tmp_path)
    make_set(rec, "labels_seg3", [(3, 225)])
    stamp = holdout.Holdout(rec).stamp()
    holdout.check_stamp(stamp, "crops")                 # nothing changed: fine
    with pytest.raises(holdout.HeldOut):
        holdout.check_stamp(None, "crops")              # built before the check existed
    holdout.mark(rec, "labels_seg3", {"all": True})
    with pytest.raises(holdout.HeldOut):
        holdout.check_stamp(stamp, "crops")             # a set was held out since


def test_marks_only_grow(tmp_path):
    rec = make_rec(tmp_path)
    make_set(rec, "labels", [(2, 10), (5, 10)])
    holdout.mark(rec, "labels", {"segments": [5]})
    holdout.mark(rec, "labels", {"segments": [2]})
    assert json.loads((rec / "labels" / "heldout.json").read_text())["segments"] == [2, 5]
