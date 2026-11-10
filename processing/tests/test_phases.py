# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""phases.py's split into periods of play on made-up recordings, without
footage: motion at a play level or a quiet one, minute by minute, and the
referee's whistle at each change.

    uv run --with pytest pytest processing/tests/test_phases.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("torch")  # follow.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import phases as PH  # noqa: E402

LEVEL = {"p": 1.0, "n": 0.1, "w": 0.45}  # play, nobody on the pitch, a warm-up


def recording(tmp_path, parts, whistles=True, seed=1):
    """A recording's phases/ folder from [(minutes, kind), ...]: one motion sample a second in 18
    cells, a 0.9 s whistle at each change between play and not, a short one every 90 s of play.
    Returns (rec, the changes in seconds)."""
    rng = np.random.default_rng(seed)
    act, ws, bounds, t, prev = [], [], [], 0.0, None
    for mins, kind in parts:
        n = int(mins * 60)
        act.append(LEVEL[kind] * 0.02 * (1 + 0.25 * rng.standard_normal(n)).clip(0.1))
        if prev is not None and (kind == "p") != (prev == "p"):
            bounds.append(t)
            ws.append({"t": t - (0.9 if prev == "p" else 0.0), "dur": 0.9, "hz": 3600.0, "tonalDb": 45.0, "loudDb": 30.0})
        if kind == "p":
            ws += [{"t": float(s), "dur": float(rng.uniform(0.2, 0.6)), "hz": 3600.0, "tonalDb": 42.0, "loudDb": 28.0}
                   for s in np.arange(t + 60, t + n - 30, 90)]
        prev, t = kind, t + n
    a = np.repeat(np.concatenate(act)[:, None], 18, 1).astype(np.float32)
    out = tmp_path / "rec" / "phases"
    out.mkdir(parents=True)
    np.savez(out / "features.npz", t=np.arange(len(a), dtype=float), act=a, fg=a * 2, pix=np.ones((2, 18)) * 100)
    (out / "whistles.json").write_text(json.dumps(sorted(ws, key=lambda w: w["t"]) if whistles else []))
    return out.parent, bounds


def periods(tmp_path, parts, monkeypatch, **kw):
    monkeypatch.setattr(PH, "plot", lambda *a, **k: None)
    rec, bounds = recording(tmp_path, parts, **kw)
    res = PH.segment(rec, 60.0)
    assert json.loads((rec / "phases" / "phases.json").read_text()) == res  # strict JSON (no NaN), as returned
    return res, bounds


def minutes(res):
    return [(round(p["start"] / 60, 1), round(p["end"] / 60, 1)) for p in res["periods"]]


def close(found, want, tol=0.3):
    return len(found) == len(want) and all(abs(a - b) <= tol for f, w in zip(found, want) for a, b in zip(f, w))


GAME = [(10, "n"), (45, "p"), (15, "n"), (45, "p"), (10, "n")]


def test_two_halves(tmp_path, monkeypatch):
    res, _ = periods(tmp_path, GAME, monkeypatch)
    assert close(minutes(res), [(10, 55), (70, 115)])
    assert res["play"] == "some"
    assert [round(res["found"][k] / 60, 1) for k in ("k1", "e1", "k2", "e2")] == [10.0, 55.0, 70.0, 115.0]
    assert all(p["startWhistle"] and p["endWhistle"] for p in res["periods"])


def test_two_halves_on_motion_alone(tmp_path, monkeypatch):
    res, _ = periods(tmp_path, GAME, monkeypatch, whistles=False)
    assert close(minutes(res), [(10, 55), (70, 115)])
    assert res["refereeHz"] is None


def test_extra_time_is_kept(tmp_path, monkeypatch):
    # Five states with halves of 20-55 min put all 30 min of extra time after the game.
    parts = GAME[:4] + [(5, "n"), (15, "p"), (2, "n"), (15, "p"), (8, "n")]
    res, _ = periods(tmp_path, parts, monkeypatch)
    assert close(minutes(res), [(10, 55), (70, 115), (120, 135), (137, 152)])
    assert res["found"] == {}  # k1..e2 name two halves only


def test_two_games_on_one_recording(tmp_path, monkeypatch):
    parts = [(8, "n"), (30, "p"), (6, "n"), (30, "p"), (12, "n"), (30, "p"), (6, "n"), (30, "p"), (8, "n")]
    res, _ = periods(tmp_path, parts, monkeypatch)
    assert close(minutes(res), [(8, 38), (44, 74), (86, 116), (122, 152)])


def test_quarters(tmp_path, monkeypatch):
    parts = [(10, "n"), (12, "p"), (2, "n"), (12, "p"), (10, "n"), (12, "p"), (2, "n"), (12, "p"), (10, "n")]
    res, _ = periods(tmp_path, parts, monkeypatch)
    assert close(minutes(res), [(10, 22), (24, 36), (46, 58), (60, 72)])


def test_one_period_gets_no_half_time(tmp_path, monkeypatch):
    # A break was assumed: 3 min of a 60-min period came out as half-time.
    for whistles in (True, False):
        res, _ = periods(tmp_path / str(whistles), [(10, "n"), (60, "p"), (10, "n")], monkeypatch, whistles=whistles)
        assert close(minutes(res), [(10, 70)])


def test_a_long_stoppage_is_a_break_not_the_end_of_the_half(tmp_path, monkeypatch):
    parts = [(10, "n"), (25, "p"), (12, "n"), (20, "p"), (6, "n"), (45, "p"), (10, "n")]
    res, _ = periods(tmp_path, parts, monkeypatch)
    assert close(minutes(res), [(10, 35), (47, 67), (73, 118)])


@pytest.mark.parametrize("whistles", [True, False])
@pytest.mark.parametrize("mins", [14, 40, 90])
def test_all_play_has_no_break_made_up(tmp_path, monkeypatch, mins, whistles):
    # Scaled to its own quietest tenth, a 40-min recording of nothing but play lost 7.5 min to a "half-time".
    res, _ = periods(tmp_path, [(mins, "p")], monkeypatch, whistles=whistles)
    assert res["play"] == "all" and len(res["periods"]) == 1
    p = res["periods"][0]
    assert (p["start"], p["end"]) == (res["fromS"], res["toS"]) and not p["startSeen"] and not p["endSeen"]


def test_recording_cut_short_in_the_second_half(tmp_path, monkeypatch):
    res, _ = periods(tmp_path, GAME[:3] + [(30, "p")], monkeypatch)
    assert close(minutes(res), [(10, 55), (70, 100)])
    assert not res["periods"][1]["endSeen"] and "e2" not in res["found"] and "k2" in res["found"]


def test_recording_begun_in_play(tmp_path, monkeypatch):
    res, _ = periods(tmp_path, [(15, "p"), (15, "n"), (45, "p"), (10, "n")], monkeypatch)
    assert close(minutes(res), [(0, 15), (30, 75)])
    assert not res["periods"][0]["startSeen"]


def test_an_active_warm_up_is_not_a_period(tmp_path, monkeypatch):
    res, _ = periods(tmp_path, [(25, "w"), (3, "n")] + GAME[1:], monkeypatch)
    assert close(minutes(res), [(28, 73), (88, 133)])


def test_short_pre_and_post_game_are_kept(tmp_path, monkeypatch):
    res, _ = periods(tmp_path, [(3, "n"), (30, "p"), (10, "n"), (30, "p"), (4, "n")], monkeypatch)
    assert close(minutes(res), [(0, 30 + 3), (43, 77)])  # under MIN_CUT_S either end: to the recording's edges
    assert res["periods"][0]["start"] == res["fromS"] and res["periods"][1]["end"] == res["toS"]


def test_refine_only_ever_keeps_more():
    # Two halves the split found, the second cut short by the recording: the equal-halves prior must not
    # shorten the first to match (it may only move a kick-off earlier or an end later).
    n = 600
    score = np.full(n, 0.6)  # play barely above THR, as on real recordings
    score[:60], score[330:400] = 0.0, 0.0
    w = {"t": 0.0, "dur": 1.2, "hz": 3600.0, "tonalDb": 45.0, "loudDb": 30.0}
    kick, end = [None] * (n + 1), [None] * (n + 1)
    for i in (60, 100, 400, 420):
        kick[i] = w
    for i in (290, 330, 350):
        end[i] = w
    b = PH.refine([60, 330, 400, n], score, kick, end)
    assert b[0] <= 60 and b[1] >= 330 and b[2] <= 400 and b[3] == n


def test_split_edge_cases():
    assert PH.split(np.zeros(0), [None], [None]) == []
    none = [None] * 201
    assert PH.split(np.full(200, np.nan), none, none) == [(0, 200)]  # nothing decoded: nothing is cut
    assert PH.split(np.zeros(200), none, none) == []
    assert PH.split(np.ones(200), none, none) == [(0, 200)]


def test_play_score_edge_cases():
    s, handled = PH.play_score(np.zeros((0, 3)), np.zeros((0, 3)))
    assert len(s) == 0 and len(handled) == 0
    s, handled = PH.play_score(np.full((5, 3), np.nan), np.full((5, 3), np.nan))
    assert np.isnan(s).all() and not handled.any()
    s, _ = PH.play_score(np.zeros((50, 3)), np.zeros((50, 3)))
    assert np.isfinite(s).all()


def test_referee_pitch():
    hz, ws = PH.referee([{"t": 1.0, "dur": 0.5, "hz": 3600.0, "tonalDb": 30.0, "loudDb": 15.0}])
    assert hz is None and ws == []  # nothing tonal and loud enough to go by
    one = [{"t": 1.0, "dur": 0.9, "hz": 3594.0, "tonalDb": 45.0, "loudDb": 30.0},
           {"t": 9.0, "dur": 0.9, "hz": 3656.0, "tonalDb": 30.0, "loudDb": 25.0}]
    hz, ws = PH.referee(one)
    assert abs(hz - 3594) <= 100 and len(ws) == 2  # the lowest of five tied bins put it at 3350 and dropped the second


def test_whistle_runs_finds_a_whistle_and_not_a_burst_of_noise():
    rng = np.random.default_rng(0)
    fs = PH.AUDIO_FS
    x = (0.01 * rng.standard_normal(60 * fs)).astype(np.float32)
    t = np.arange(int(0.8 * fs)) / fs
    x[20 * fs: 20 * fs + len(t)] += (0.3 * np.sin(2 * np.pi * 3600 * t)).astype(np.float32)
    x[40 * fs: 41 * fs] += (0.5 * rng.standard_normal(fs)).astype(np.float32)
    runs = PH.whistle_runs(x, 100.0)
    assert len(runs) == 1 and abs(runs[0]["t"] - 120.0) < 0.05 and abs(runs[0]["dur"] - 0.8) < 0.05
    assert abs(runs[0]["hz"] - 3600) < 40
    assert PH.whistle_runs(np.zeros(0, np.float32), 0.0) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_file_with_no_audio_track_gives_no_whistles(tmp_path):
    silent, loud = tmp_path / "silent.mp4", tmp_path / "loud.mp4"
    base = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x36:d=1"]
    made = subprocess.run([*base, "-an", "-c:v", "mpeg4", str(silent)], capture_output=True)
    if made.returncode != 0:
        pytest.skip("this ffmpeg can't make the test video")
    assert len(PH.load_audio(silent)) == 0
    assert PH.whistle_runs(PH.load_audio(silent), 0.0) == []
    subprocess.run([*base, "-f", "lavfi", "-i", "sine=frequency=3600:duration=1", "-shortest", "-c:v", "mpeg4",
                    "-c:a", "aac", str(loud)], capture_output=True, check=True)
    assert len(PH.load_audio(loud)) > PH.AUDIO_FS // 2
    with pytest.raises(FileNotFoundError):
        PH.load_audio(tmp_path / "missing.mp4")
