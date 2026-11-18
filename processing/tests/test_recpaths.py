# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""recpaths.py: a recording in the bench store keeps its caches, outputs and
training products outside the store; a working recording keeps them itself."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import holdout  # noqa: E402
import recpaths  # noqa: E402


def test_a_working_recording_keeps_everything(tmp_path):
    rec = tmp_path / "footage" / "rec7"
    rec.mkdir(parents=True)
    assert recpaths.cache(rec) == rec and recpaths.out(rec) == rec and recpaths.train(rec) == rec
    assert not recpaths.in_store(rec)


def test_a_store_recording_is_split(tmp_path):
    rec = tmp_path / "footage" / "bench" / "rec7"
    rec.mkdir(parents=True)
    assert recpaths.in_store(rec)
    assert recpaths.cache(rec) == tmp_path / "footage" / "bench-cache" / "rec7"
    assert recpaths.out(rec) == tmp_path / "footage" / "bench-out" / "rec7"
    assert recpaths.train(rec) == tmp_path / "footage" / "train" / "rec7"
    assert recpaths.cache(rec).is_dir()  # made, so callers only make their own subfolder
    assert sorted(p.name for p in rec.iterdir()) == []  # and nothing appears in the store


def test_only_footage_bench_is_the_store(tmp_path):
    rec = tmp_path / "bench" / "footage" / "rec7"  # a folder named bench above footage/ is not the store
    rec.mkdir(parents=True)
    assert recpaths.cache(rec) == rec


def test_roots_can_be_named(tmp_path, monkeypatch):
    rec = tmp_path / "footage" / "bench" / "rec7"
    rec.mkdir(parents=True)
    monkeypatch.setenv("OSB_BENCH_CACHE", str(tmp_path / "elsewhere"))
    assert recpaths.cache(rec) == tmp_path / "elsewhere" / "rec7"


def test_a_stamp_finds_a_recording_that_moved_into_the_store(tmp_path, monkeypatch):
    """A stamp names the path its data was built from; when the recording has
    moved, the check must still run on it, not pass because the path is gone."""
    footage = tmp_path / "footage"
    (footage / "bench" / "rec7").mkdir(parents=True)
    (footage / "bench" / "rec7" / "manifest.json").write_text(json.dumps({"devices": []}))
    monkeypatch.setattr(holdout, "__file__", str(tmp_path / "processing" / "spikes" / "holdout.py"))
    assert holdout.located(footage / "rec7") == footage / "bench" / "rec7"
    assert holdout.located(footage / "rec99") is None
