# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""A Secondary that has only part of the game (it stopped early, started
late, or has a hole), on made-up recordings without footage: what the
sync's report says it covers, the blank frames the readers get where it
has nothing, the view staying on Main there, and where the engine fits the
pitch.

    cd processing && uv run --extra dev pytest tests/test_secondary_gaps.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("torch")  # follow.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import engine  # noqa: E402
import follow  # noqa: E402
import phase0  # noqa: E402

FPS = 10.0
W = 32  # frames are 32x18


def manifest(main_s, sec_s):
    def segs(name, durs):
        return [{"index": i, "objectKey": f"{name}/{name}_{i:03d}.mp4", "durationMs": int(d * 1000)}
                for i, d in enumerate(durs)]
    return {"devices": [{"role": "main", "segments": segs("main", main_s)},
                        {"role": "secondary", "segments": segs("sec", sec_s)}]}


def piece(m, index, offset):
    s = m["devices"][1]["segments"][index]
    return {"index": index, "objectKey": s["objectKey"], "durationMs": s["durationMs"], "offsetS": offset}


def stopped_early():
    """Main: four segments of 100 s. Secondary: two files of 120 s, so the
    first 240 s of the game: segments 0 and 1 whole, 40 s of segment 2,
    none of segment 3 (the shape of the game of 2026-11-11: an iPhone
    cutting at 300 s that stopped after 45 of a Pixel's 99 minutes)."""
    m = manifest([100] * 4, [120] * 2)
    report = {"segments": [
        {"index": 0, "secondaryIndex": 0, "offsetS": 0.0, "secondaryPieces": [piece(m, 0, 0.0)]},
        {"index": 1, "secondaryIndex": 1, "offsetS": -20.0,
         "secondaryPieces": [piece(m, 0, 100.0), piece(m, 1, -20.0)]},
        {"index": 2, "secondaryIndex": 1, "offsetS": 80.0, "secondaryPieces": [piece(m, 1, 80.0)]},
        {"index": 3, "secondaryIndex": 1, "offsetS": 180.0, "synced": False, "noSecondary": True,
         "secondaryPieces": []}]}
    return m, report


def test_coverage_of_a_secondary_that_stopped_early():
    m, report = stopped_early()
    assert phase0.coverage(m, report) == [[0.0, 240.0]]
    assert phase0.covered_segments(m, report) == [0, 1]
    missing = phase0.secondary_missing(m, report, 230.0, 20, 1.0)
    assert missing.tolist() == [False] * 10 + [True] * 10


def test_coverage_with_a_late_start_and_a_hole():
    # Secondary's files: 50-170 s and 230-350 s of a 400 s game.
    m = manifest([100] * 4, [120] * 2)
    report = {"segments": [
        {"index": 0, "secondaryIndex": 0, "offsetS": -50.0, "secondaryPieces": [piece(m, 0, -50.0)]},
        {"index": 1, "secondaryIndex": 0, "offsetS": 50.0, "secondaryPieces": [piece(m, 0, 50.0)]},
        {"index": 2, "secondaryIndex": 1, "offsetS": -30.0, "secondaryPieces": [piece(m, 1, -30.0)]},
        {"index": 3, "secondaryIndex": 1, "offsetS": 70.0, "secondaryPieces": [piece(m, 1, 70.0)]}]}
    assert phase0.coverage(m, report) == [[50.0, 170.0], [230.0, 350.0]]
    assert phase0.covered_segments(m, report) == []


def test_coverage_of_a_report_without_pieces_is_the_paired_files():
    # Same segment lengths, both phones throughout: reports from before this change say nothing more.
    m = manifest([100] * 3, [100] * 3)
    report = {"segments": [{"index": i, "secondaryIndex": i, "offsetS": 0.3} for i in range(3)]}
    assert phase0.coverage(m, report) == [[0.0, 300.0]]  # the 0.3 s at each switch and the end: a held frame
    assert phase0.covered_segments(m, report) == [0, 1, 2]
    assert not phase0.secondary_missing(m, report, 0.0, 3000, 10.0).any()


@pytest.fixture
def numbered_frames(monkeypatch):
    """follow.frame_stream without files: each frame's luma is its file's
    number (x100) plus its second in that file."""
    def fake(path, start, seconds, width, fps=FPS):
        file_no = int(Path(path).stem.split("_")[-1])
        dur = 120.0 if "sec" in str(path) else 100.0
        size = width * round(width * 9 / 16) * 3 // 2
        for i in range(int(round(seconds * fps))):
            t = start + i / fps
            if t >= dur:
                return
            yield np.full(size, min(255, 100 * file_no + int(t) % 100), np.uint8)
    monkeypatch.setattr(follow, "frame_stream", fake)


def test_game_stream_goes_blank_where_secondary_has_no_footage(numbered_frames):
    m, report = stopped_early()
    devs = phase0.paired_devices(m, report)
    # 30 s across the end of Secondary's footage (240 s into the game), then on into segment 3.
    frames = list(follow.game_stream(Path("."), devs, report, 1, 2, 30.0, 90.0, W, FPS))
    assert len(frames) == 900  # exactly as many as Main's, whatever Secondary has
    blank = follow.blank_frame(W)
    is_blank = np.array([np.array_equal(f, blank) for f in frames])
    hold = int(follow.HOLD_MAX_S * FPS)
    assert not is_blank[:100 + hold].any()             # its footage, then its last frame held a moment
    assert int(frames[99][0]) == 100 + 19              # file 1, 119 s in: the last second it has
    assert all(int(f[0]) == 100 + 19 for f in frames[100:100 + hold])
    assert is_blank[100 + hold:].all()                 # then nothing, for the rest of segment 2 and all of 3
    # Main is untouched.
    main = list(follow.game_stream(Path("."), devs, report, 0, 2, 30.0, 90.0, W, FPS))
    assert len(main) == 900 and int(main[0][0]) == 200 + 30


def test_game_stream_is_blank_from_the_start_of_a_clip_past_the_footage(numbered_frames):
    m, report = stopped_early()
    devs = phase0.paired_devices(m, report)
    frames = list(follow.game_stream(Path("."), devs, report, 1, 3, 10.0, 5.0, W, FPS))
    assert len(frames) == 50
    assert all(np.array_equal(f, follow.blank_frame(W)) for f in frames)


def test_the_view_stays_on_main_where_secondary_has_no_footage(tmp_path):
    # Canvas -80..80 deg, Secondary on the left, Main seeing -10..80. Players (blobs) far to the
    # left, where only Secondary looks: with both phones the view goes there; alone it stops at Main's edge.
    T = 300
    extent = tuple(np.radians([-80.0, 80.0, -30.0, 30.0]))
    dets = [[(np.radians(-60.0 + k), 0.0, 50.0, 0.05) for k in range(4)] for _ in range(T)]
    main_yaw = [float(np.radians(-10.0)), float(np.radians(80.0))]
    alone = np.zeros(T, bool)
    alone[150:] = True
    yaw, _, hfov = follow.plan_path(tmp_path, tmp_path, 0, 0.0, dets, extent, "players", (), alone, main_yaw)
    left = np.degrees(yaw - hfov / 2)
    assert left[:150].max() < -40                                    # both phones: over on Secondary's side
    assert left[150:].min() >= -10 + follow.EDGE_MARGIN_DEG - 1e-6    # Main alone: never past its edge
    assert np.degrees(hfov[150:]).min() >= follow.EDGE_MIN_HFOV - 1e-6
    # The view pitched down reaches farther left at its bottom corners: with Main's camera, those stay in too.
    R = np.array([[np.cos(a := np.radians(35.0)), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    cam = follow.Camera(follow.REF_W / 2 / np.tan(np.radians(45.0)), follow.REF_W / 2, follow.REF_W * 9 / 32, 0.0, 0.0, R)
    low = tuple(np.radians([-80.0, 80.0, 10.0, 60.0]))  # only looking down
    assert not all(follow.inside_main(cam, y, p, h, -1.0) for y, p, h in
                   zip(*(v[150:] for v in follow.plan_path(tmp_path, tmp_path, 0, 0.0, dets, low, "players", (), alone, main_yaw))))
    yaw, pitch, hfov = follow.plan_path(tmp_path, tmp_path, 0, 0.0, dets, low, "players", (), alone, main_yaw, cam)
    assert all(follow.inside_main(cam, y, p, h, -1.0) for y, p, h in zip(yaw[150:], pitch[150:], hfov[150:]))
    assert all(follow.inside_main(cam, y, p, h, -1.0, bottom=True) for y, p, h in zip(yaw[150:], pitch[150:], hfov[150:]))
    # No frames alone: the path is what it was before.
    y0, _, h0 = follow.plan_path(tmp_path, tmp_path, 0, 0.0, dets, extent, "players", ())
    y1, _, h1 = follow.plan_path(tmp_path, tmp_path, 0, 0.0, dets, extent, "players", (), np.zeros(T, bool), main_yaw)
    assert np.array_equal(y0, y1) and np.array_equal(h0, h1)


def test_the_engine_fits_the_pitch_where_both_phones_have_footage(tmp_path):
    m, report = stopped_early()
    m["devices"][0]["segments"] = [dict(s) for s in m["devices"][0]["segments"]] * 2  # 8 segments; a quarter in is 2
    report["segments"] = report["segments"] + [dict(report["segments"][3]) for _ in range(4)]
    job = type("J", (), {"rec": tmp_path, "sample": None})()
    assert engine.sample_clip(job, m)[0] == 2      # no report yet: a quarter of the way in
    (tmp_path / "phase0").mkdir()
    (tmp_path / "phase0" / "report.json").write_text(json.dumps(
        {**report, "secondaryCoverage": phase0.coverage(m, report)}))
    assert engine.sample_clip(job, m)[0] == 1      # segment 2 has only 40 s of Secondary: the nearest whole one
