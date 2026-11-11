# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""What sharedata.py would upload, on a made-up footage/ folder (no network):

    uv run --with pytest pytest processing/tests/test_sharedata.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import sharedata as S  # noqa: E402


@pytest.fixture
def footage(tmp_path, monkeypatch):
    f = tmp_path / "footage"
    for rel, text in {
        "bench/rec3/manifest.json": json.dumps({"gameId": "f68af87f", "devices": []}),
        "bench/rec3/meta-main/upload.json": json.dumps({"accountId": "0b1c2d3e-0000-4000-8000-123456789abc"}),
        "bench/rec3/meta-main/recording.json": "{}",
        "bench/rec3/balltrain/checkpoint.pth": "weights",
        "models/ball_detector.pth": "weights",
        "rec7/manifest.json": "{}",
        "rec7/pitch/pitch.json": "{}",
        "rec7/fullgame/game.mp4": "video",
    }.items():
        (f / rel).parent.mkdir(parents=True, exist_ok=True)
        (f / rel).write_text(text)
    monkeypatch.setattr(S, "FOOTAGE", f)
    return f


def names(files, footage):
    return {p.relative_to(footage).as_posix() for p in files}


def test_upload_state_is_never_shared(footage):
    sent = names(S.matching_files(), footage)
    assert "bench/rec3/meta-main/upload.json" not in sent
    assert {"bench/rec3/manifest.json", "bench/rec3/meta-main/recording.json", "models/ball_detector.pth",
            "rec7/manifest.json", "rec7/pitch/pitch.json"} <= sent
    assert "rec7/fullgame/game.mp4" not in sent
    assert S.private_details(S.matching_files()) == {}


def test_upload_state_already_in_the_dataset_is_deleted(footage):
    class Api:
        def list_repo_files(self, repo, repo_type):
            return ["bench/rec3/manifest.json", "bench/rec3/meta-main/upload.json", "bench/rec3/meta-sec/upload.json",
                    "rec7/pitch/pitch.json", "rec7/pitch/old_map.json", "bench/rec3/detect/old.json"]

    # bench/ only grows, except for what is never shared; a mirrored folder loses what is gone locally.
    assert S.remote_stale(Api()) == ["bench/rec3/meta-main/upload.json", "bench/rec3/meta-sec/upload.json",
                                     "rec7/pitch/old_map.json"]


@pytest.mark.parametrize("rel, text, what", [
    ("bench/rec3/balltrain/training_config.json", '{"dataset_dir": "/Users/someone/projects/x/footage/rec3"}',
     "a home-directory path"),
    ("bench/fixtures/pitch/night/provenance/scripts/lens.py", 'sys.path.insert(0, "/home/someone/repo/spikes")',
     "a home-directory path"),
    ("rec7/pitch/notes.json", '{"accountId": "0b1c2d3e-0000-4000-8000-123456789abc"}', "an account id"),
    ("bench/rec3/detect/run.log", "s3://bucket/accounts/0b1c2d3e-0000-4000-8000-123456789abc/games/x", "an accounts/<id> key"),
])
def test_private_details_stop_the_upload(footage, rel, text, what, capsys):
    (footage / rel).parent.mkdir(parents=True, exist_ok=True)
    (footage / rel).write_text(text)
    assert S.private_details(S.matching_files()) == {rel: [what]}
    assert S.cmd_list() is False and "NOT TO SHARE" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="nothing uploaded"):  # before anything reaches the network
        S.cmd_upload()


def test_relative_paths_and_ids_that_belong_are_fine(footage):
    (footage / "bench/rec3/balltrain/training_config.json").write_text(
        '{"dataset_dir": "footage/rec3/balldata", "pretrain_weights": "~/.roboflow/models/rf-detr-medium.pth"}')
    (footage / "rec7/manifest.json").write_text('{"gameId": "3ba4b9f2-0000-4000-8000-123456789abc", "sessionId": "x"}')
    assert S.private_details(S.matching_files()) == {}
    assert S.cmd_list() is True
