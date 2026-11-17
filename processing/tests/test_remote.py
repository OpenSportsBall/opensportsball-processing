# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Footage left in the cloud (remote.py): prepare's two readers get the
same sound and the same frames from a placeholder that holds only what
they asked for as from the whole file, over plain HTTP ranges and from S3
(moto); what is here is fetched once; and the engine neither hashes a
placeholder nor starts a game on one.

    cd processing && uv run --extra dev pytest tests/test_remote.py
"""

import http.server
import json
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import engine  # noqa: E402
import phase0  # noqa: E402
import remote  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which(engine.accel.ffmpeg()) is None, reason="needs ffmpeg")

KEY = "dev1/dev1_000.mp4"
SECONDS = 12


def make_video(path: Path, index_first: bool) -> None:
    """A phone-like MP4: video with a keyframe a second, sound interleaved
    with it, the index at the end (an iPhone) or at the start (a Pixel)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [engine.accel.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={SECONDS}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={SECONDS}",
         "-c:v", "libx264", "-preset", "ultrafast", "-b:v", "4M", "-g", "30", "-pix_fmt", "yuv420p", "-c:a", "aac",
         *(["-movflags", "+faststart"] if index_first else []), str(path)], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"this ffmpeg can't make the test video: {r.stderr[-200:]}")


class Ranges(http.server.BaseHTTPRequestHandler):
    """Serves the folder it was given, honouring Range, and counts what it sent."""
    root: Path
    sent = 0
    requests = 0

    def do_GET(self):
        data = (self.root / self.path.lstrip("/")).read_bytes()
        a, b = self.headers["Range"].removeprefix("bytes=").split("-")
        part = data[int(a):int(b) + 1]
        type(self).sent += len(part)
        type(self).requests += 1
        self.send_response(206)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)

    def log_message(self, *a):
        pass


@pytest.fixture(params=[False, True], ids=["index-last", "index-first"])
def footage(tmp_path, monkeypatch, request):
    """The segment whole in one folder and served from it; an empty recording
    folder whose note says where the segments are."""
    whole = tmp_path / "cloud" / KEY
    make_video(whole, request.param)
    handler = type("H", (Ranges,), {"root": tmp_path / "cloud", "sent": 0, "requests": 0})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rec = tmp_path / "rec"
    rec.mkdir()
    note = tmp_path / "remote.json"
    note.write_text(json.dumps({"recording": str(rec), "urls": {KEY: f"http://127.0.0.1:{server.server_port}/{KEY}"}}))
    monkeypatch.setenv(remote.ENV, str(note))
    remote.reset()
    yield whole, rec / KEY, handler
    server.shutdown()
    remote.reset()


def test_a_placeholder_holds_the_index_and_nothing_else(footage):
    whole, here, served = footage
    remote.placeholder(here, whole.stat().st_size)
    assert here.stat().st_size == whole.stat().st_size and remote.partial(here)
    assert served.sent < 0.02 * whole.stat().st_size
    # The index answers what it answers for the whole file.
    assert phase0.probe_start(here, "v") == phase0.probe_start(whole, "v")
    # Asked for again, nothing is fetched.
    before = served.requests
    remote.placeholder(here, whole.stat().st_size)
    assert served.requests == before


def test_the_sound_is_the_whole_files_sound(footage):
    whole, here, served = footage
    remote.placeholder(here, whole.stat().st_size)
    got = phase0.load_audio(here)
    assert len(got) > 0.9 * SECONDS * phase0.FS and np.array_equal(got, phase0.load_audio(whole))
    assert served.sent < 0.25 * whole.stat().st_size  # the sound and the index, not the video
    before = served.requests
    phase0.load_audio(here)
    assert served.requests == before


def test_frames_are_the_whole_files_frames(footage):
    whole, here, served = footage
    remote.placeholder(here, whole.stat().st_size)
    for t in (0.0, 4.3, SECONDS - 1.0):
        want = phase0.frames_at(whole, t, 5, 30, 320)
        got = phase0.frames_at(here, t, 5, 30, 320)
        assert len(want) == len(got) == 5
        assert all(np.array_equal(a, b) for a, b in zip(want, got)), t
    a = phase0.frame_at(here, 7.0, 320)
    assert a is not None and np.array_equal(a, phase0.frame_at(whole, 7.0, 320))


def test_one_frame_does_not_fetch_the_file_and_is_fetched_once(footage):
    whole, here, served = footage
    remote.placeholder(here, whole.stat().st_size)
    phase0.frames_at(here, 6.0, 1, 30, 320)
    assert served.sent < 0.6 * whole.stat().st_size  # 4.5 s of 12, with the margins
    before = served.requests
    phase0.frames_at(here, 6.0, 1, 30, 320)
    phase0.frames_at(here, 5.8, 3, 30, 320)  # inside what is here
    assert served.requests == before
    # Overlapping what is here: only the rest is asked for.
    sent = served.sent
    phase0.frames_at(here, 7.0, 1, 30, 320)
    assert 0 < served.sent - sent < 0.2 * whole.stat().st_size


def test_a_whole_file_is_left_alone(footage):
    whole, here, served = footage
    here.parent.mkdir(parents=True)
    shutil.copy(whole, here)
    remote.placeholder(here, whole.stat().st_size)
    phase0.load_audio(here)
    phase0.frames_at(here, 3.0, 1, 30, 320)
    assert served.requests == 0 and not remote.partial(here)


def test_nothing_happens_without_the_note(footage, monkeypatch):
    whole, here, served = footage
    monkeypatch.delenv(remote.ENV)
    remote.reset()
    remote.placeholder(here, whole.stat().st_size)
    assert not here.exists()
    assert np.array_equal(phase0.frames_at(whole, 3.0, 1, 30, 320)[0], phase0.frames_at(whole, 3.0, 1, 30, 320)[0])
    assert served.requests == 0


def test_spans():
    assert remote._merged([[5, 9], [0, 3], [3, 4], [8, 12]]) == [[0, 4], [5, 12]]
    assert remote._missing([[0, 4], [5, 12]], 2, 20) == [(4, 5), (12, 20)]
    assert remote._missing([[0, 4]], 1, 3) == []
    assert remote._missing([], 1, 3) == [(1, 3)]


def job_for(rec: Path, size: int):
    m = {"schemaVersion": 1, "devices": [
        {"role": "main", "segments": [{"index": 0, "objectKey": KEY, "bytes": size, "sha256": "0" * 64, "durationMs": 12000}]},
        {"role": "secondary", "segments": []}]}
    (rec / "manifest.json").write_text(json.dumps(m))
    (rec / "engine").mkdir(exist_ok=True)
    return types.SimpleNamespace(rec=rec, dir=rec / "engine", manifest=lambda: m)


def test_the_engine_makes_placeholders_and_does_not_hash_them(footage, capsys):
    whole, here, served = footage
    job = job_for(here.parents[1], whole.stat().st_size)
    engine.check_footage(job)  # the manifest's SHA-256 is not this file's: a hashed file would fail
    assert remote.partial(here)
    assert not (job.dir / "verified.json").exists()
    events = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [(e["step"], e["status"]) for e in events] == [("remote-footage", "started"), ("remote-footage", "done")]


def test_run_refuses_a_placeholder(footage):
    whole, here, served = footage
    job = job_for(here.parents[1], whole.stat().st_size)
    engine.check_footage(job)
    with pytest.raises(engine.Failed) as e:
        engine.check_footage(job, whole=True)
    assert e.value.code == "footage_missing"
    # Also with no note: a placeholder left behind is still not the footage.
    remote.reset()
    import os
    os.environ.pop(remote.ENV)
    with pytest.raises(engine.Failed) as e:
        engine.check_footage(job, whole=True)
    assert e.value.code == "footage_missing"


def test_footage_that_cannot_be_reached_is_said_plainly(footage):
    whole, here, served = footage
    note = Path(__import__("os").environ[remote.ENV])
    spec = json.loads(note.read_text())
    spec["urls"][KEY] = "http://127.0.0.1:9/nothing"
    note.write_text(json.dumps(spec))
    remote.reset()
    remote.ATTEMPTS, was = 1, remote.ATTEMPTS
    try:
        with pytest.raises(engine.Failed) as e:
            engine.check_footage(job_for(here.parents[1], whole.stat().st_size))
    finally:
        remote.ATTEMPTS = was
    assert e.value.code == "footage_missing"


def test_from_s3(tmp_path, monkeypatch):
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    whole = tmp_path / "cloud" / KEY
    make_video(whole, False)
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    for k in ("AWS_PROFILE", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(k, raising=False)
    with moto.mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="osb-test-remote")
        s3.put_object(Bucket="osb-test-remote", Key="accounts/a/rec/" + KEY, Body=whole.read_bytes())
        rec = tmp_path / "rec"
        rec.mkdir()
        stats = tmp_path / "stats.jsonl"
        note = tmp_path / "remote.json"
        note.write_text(json.dumps({"recording": str(rec), "s3": {"bucket": "osb-test-remote", "prefix": "accounts/a/rec/"},
                                    "stats": str(stats)}))
        monkeypatch.setenv(remote.ENV, str(note))
        remote.reset()
        try:
            here = rec / KEY
            remote.placeholder(here, whole.stat().st_size)
            assert np.array_equal(phase0.load_audio(here), phase0.load_audio(whole))
            assert np.array_equal(phase0.frames_at(here, 5.0, 2, 30, 320)[1], phase0.frames_at(whole, 5.0, 2, 30, 320)[1])
            remote._write_stats()
            line = json.loads(stats.read_text().splitlines()[0])
            assert 0 < line["bytes"] < whole.stat().st_size and line["gets"] > 3
        finally:
            remote.reset()
