# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""cloud.py's side of the cloud job's contract, against a local S3 and
DynamoDB (moto) and a stand-in for engine.py; no footage, no GPU:

    uv run --with pytest --with boto3 --with 'moto[s3,dynamodb]' pytest processing/tests/test_cloud.py
"""

import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import cloud  # noqa: E402

BUCKET, TABLE, JOB = "osb-test-footage", "osb-test", "11111111-1111-4111-8111-111111111111"
FOOTAGE = "accounts/a/games/g/recordings/r/"
OUT = f"outputs/a/games/g/recordings/r/jobs/{JOB}/"
WORK = f"work/a/games/g/recordings/r/jobs/{JOB}/"
SEGMENTS = {"dev1/dev1_000.mp4": b"main-video" * 10, "dev2/dev2_000.mp4": b"second-video" * 7}

# A stand-in for engine.py: does what the real one does to the folder, as
# far as cloud.py can tell. ENGINE_FAIL makes it fail with that code;
# ENGINE_NEEDS_FOOTAGE makes --no-footage fail, as a step wanting video would.
FAKE_ENGINE = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    command, job = sys.argv[1], Path(sys.argv[2])
    rec = job.parent / json.loads(job.read_text())["recording"]
    say = lambda **e: print(json.dumps(e), flush=True)
    (rec / "engine" / "logs").mkdir(parents=True, exist_ok=True)
    (rec / "engine" / "calls.jsonl").open("a").write(json.dumps(sys.argv[1:]) + "\\n")
    (rec / "engine" / "logs" / "sync.log").write_text("a log")
    have = sorted(str(p.relative_to(rec)) for p in rec.rglob("*.mp4"))
    (rec / "engine" / "seen.json").write_text(json.dumps(have))
    print("not an event")
    if os.environ.get("ENGINE_FAIL"):
        say(event="error", code=os.environ["ENGINE_FAIL"], message="segment 3 could not be read")
        sys.exit(2)
    if "--no-footage" in sys.argv and os.environ.get("ENGINE_NEEDS_FOOTAGE"):
        say(event="error", code="pitch-refit-1_failed", message="no video")
        sys.exit(4)
    if command == "run" and os.environ.get("ENGINE_REFUSES"):
        (rec / "engine" / "questions.json").write_text(json.dumps({
            "schemaVersion": 1, "recording": str(rec), "pending": ["kits"], "kits": {"round": 2, "problem": "too few"}}))
        say(event="needs_answers", pending=["kits"])
        sys.exit(3)
    if command == "prepare":
        say(event="step", step="sync", status="started")
        (rec / "pitch").mkdir(exist_ok=True)
        (rec / "pitch" / "overlay.jpg").write_bytes(b"jpeg")
        (rec / "pitch" / "review.json").write_text("{}")
        os.utime(rec / "pitch" / "review.json", ns=(1_700_000_000_123_456_789,) * 2)
        say(event="step", step="sync", status="done", seconds=1)
        rounds = 2 if "--answers" in sys.argv else 1
        (rec / "engine" / "questions.json").write_text(json.dumps({
            "schemaVersion": 1, "recording": str(rec), "pending": ["pitch", "kits"],
            "pitch": {"round": rounds, "overlay": "pitch/overlay.jpg", "background": "../../secret.png",
                      "points": {"left corner flag": {"closeup": {"image": "engine/closeups/1/a.jpg"}}}},
            "kits": {"groups": [{"sheet": "/etc/passwd", "crops": [{"image": "pitch/overlay.jpg"}]}]}}))
        say(event="questions", pending=["pitch", "kits"])
    else:
        say(event="progress", stage="game", fraction=0.5, etaS=3600)
        (rec / "fullgame").mkdir(exist_ok=True)
        (rec / "fullgame" / "piece_00.mp4").write_bytes(b"piece")
        (rec / "fullgame" / "video.mp4").write_bytes(b"joined")
        (rec / "fullgame" / "game.mp4").write_bytes(b"the game")
        say(event="done", output=str(rec / "fullgame" / "game.mp4"), seconds=12, ballAimShare=0.42, ballLow=False)
""")


@pytest.fixture
def aws(tmp_path, monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                 "OSB_BUCKET": BUCKET, "OSB_TABLE": TABLE, "OSB_JOB_ID": JOB, "OSB_FOOTAGE_PREFIX": FOOTAGE,
                 "OSB_OUTPUT_PREFIX": OUT, "OSB_WORK_PREFIX": WORK, "OSB_SCRATCH": str(tmp_path / "scratch"),
                 "OSB_KEEP_SCRATCH": "1", "OSB_ENGINE_VERSION": "osb-engine test"}.items():
        monkeypatch.setenv(k, v)
    for k in ("ENGINE_FAIL", "ENGINE_NEEDS_FOOTAGE", "ENGINE_REFUSES", "OSB_JOB_OPTIONS", "OSB_REQUIRE_GPU", "AWS_PROFILE", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(k, raising=False)
    fake = tmp_path / "fake_engine.py"
    fake.write_text(FAKE_ENGINE)
    monkeypatch.setenv("OSB_ENGINE_CMD", json.dumps([sys.executable, str(fake)]))
    monkeypatch.setattr(cloud, "fetch_models", lambda c: None)
    monkeypatch.setattr(cloud, "_stopping", cloud.threading.Event())
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=BUCKET)
        manifest = {"schemaVersion": 1, "devices": [
            {"role": role, "segments": [{"objectKey": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}]}
            for role, (k, v) in zip(("main", "secondary"), SEGMENTS.items())]}
        s3.put_object(Bucket=BUCKET, Key=FOOTAGE + "manifest.json", Body=json.dumps(manifest))
        for k, v in SEGMENTS.items():
            s3.put_object(Bucket=BUCKET, Key=FOOTAGE + k, Body=v)
        ddb = boto3.client("dynamodb")
        ddb.create_table(TableName=TABLE, BillingMode="PAY_PER_REQUEST",
                         AttributeDefinitions=[{"AttributeName": n, "AttributeType": "S"} for n in ("pk", "sk")],
                         KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}])
        ddb.put_item(TableName=TABLE, Item={"pk": {"S": f"JOB#{JOB}"}, "sk": {"S": "JOB"}, "status": {"S": "queued"}})
        yield s3, ddb, tmp_path / "scratch" / JOB / "rec"


def go(monkeypatch, command):
    monkeypatch.setattr(sys, "argv", ["cloud.py", command])
    return cloud.main()


def keys(s3, prefix):
    return sorted(o["Key"][len(prefix):] for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []))


def record(ddb):
    return ddb.get_item(TableName=TABLE, Key={"pk": {"S": f"JOB#{JOB}"}, "sk": {"S": "JOB"}}).get("Item")


def body(s3, key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def calls(rec):
    return [json.loads(line) for line in (rec / "engine" / "calls.jsonl").read_text().splitlines()]


def test_prepare_fetches_the_footage_and_publishes_the_questions(aws, monkeypatch):
    s3, ddb, rec = aws
    assert go(monkeypatch, "prepare") == 0
    # The engine saw the recording folder as on a Mac: manifest and segments at their objectKeys.
    assert json.loads((rec / "engine" / "seen.json").read_text()) == sorted(SEGMENTS)
    assert calls(rec) == [["prepare", str(rec.parent / "job.json")]]
    # Each segment carries the time S3 has for it, so a later machine needn't hash it again.
    for k in SEGMENTS:
        modified = s3.head_object(Bucket=BUCKET, Key=FOOTAGE + k)["LastModified"].timestamp()
        assert (rec / k).stat().st_mtime == int(modified)

    q = json.loads(body(s3, OUT + "questions/questions.json"))
    assert q["recording"] == "." and q["pending"] == ["pitch", "kits"]
    # The pictures it names, where it names them; nothing from outside the folder, nothing missing.
    assert keys(s3, OUT) == ["questions/pitch/overlay.jpg", "questions/questions.json"]
    assert s3.head_object(Bucket=BUCKET, Key=OUT + "questions/pitch/overlay.jpg")["ContentType"] == "image/jpeg"

    # The folder without the footage is in the work prefix for the next command.
    work = keys(s3, WORK)
    assert "engine/questions.json" in work and "pitch/review.json" in work
    assert not [k for k in work if k.endswith(".mp4") or k == "manifest.json"]

    item = record(ddb)
    assert item["status"]["S"] == "queued"  # status is the server's
    assert item["engineVersion"]["S"] == "osb-engine test"
    assert item["progress"]["M"]["stage"]["S"] in ("download", "sync")
    assert "error" not in item


def test_job_options_reach_the_engine(aws, monkeypatch):
    s3, ddb, rec = aws
    monkeypatch.setenv("OSB_JOB_OPTIONS", json.dumps({"options": {"maxChunks": 1}, "output": {"height": 1080}, "recording": "/etc"}))
    assert go(monkeypatch, "prepare") == 0
    assert json.loads((rec.parent / "job.json").read_text()) == {
        "schemaVersion": 1, "recording": "rec", "options": {"maxChunks": 1}, "output": {"height": 1080}}
    monkeypatch.setenv("OSB_JOB_OPTIONS", "{not json")
    assert go(monkeypatch, "prepare") == 1
    assert record(ddb)["error"]["M"]["code"]["S"] == "bad_job"


def test_refit_applies_answers_without_the_footage(aws, monkeypatch):
    s3, ddb, rec = aws
    assert go(monkeypatch, "prepare") == 0
    cloud.shutil.rmtree(rec.parent)  # another machine
    answers = {"schemaVersion": 1, "pitch": {"round": 1, "approve": True}}
    s3.put_object(Bucket=BUCKET, Key=OUT + "answers.json", Body=json.dumps(answers))

    assert go(monkeypatch, "refit") == 0
    last = calls(rec)[-1]
    assert last[0] == "prepare" and "--no-footage" in last
    assert json.loads(Path(last[last.index("--answers") + 1]).read_text()) == answers
    assert json.loads((rec / "engine" / "seen.json").read_text()) == []  # no video was fetched
    # The engine's files came back with the times they had (it compares some).
    assert (rec / "pitch" / "review.json").stat().st_mtime_ns == 1_700_000_000_123_456_789
    assert json.loads(body(s3, OUT + "questions/questions.json"))["pitch"]["round"] == 2


def test_refit_fetches_the_footage_when_a_step_wants_the_video(aws, monkeypatch):
    s3, ddb, rec = aws
    assert go(monkeypatch, "prepare") == 0
    s3.put_object(Bucket=BUCKET, Key=OUT + "answers.json", Body=json.dumps({"schemaVersion": 1}))
    monkeypatch.setenv("ENGINE_NEEDS_FOOTAGE", "1")
    assert go(monkeypatch, "refit") == 0
    assert ["--no-footage" in c for c in calls(rec)[-2:]] == [True, False]
    assert json.loads((rec / "engine" / "seen.json").read_text()) == sorted(SEGMENTS)
    assert "error" not in record(ddb)


def test_run_uploads_the_game_and_its_result(aws, monkeypatch):
    s3, ddb, rec = aws
    assert go(monkeypatch, "prepare") == 0
    assert go(monkeypatch, "run") == 0
    assert calls(rec)[-1] == ["run", str(rec.parent / "job.json")]  # no answers.json: none passed
    assert body(s3, OUT + "game.mp4") == b"the game"
    assert json.loads(body(s3, OUT + "result.json")) == {
        "event": "done", "output": str(rec / "fullgame" / "game.mp4"), "seconds": 12, "ballAimShare": 0.42, "ballLow": False}
    # Finished pieces are kept for a resumed run; the output and its join are not kept twice.
    work = keys(s3, WORK)
    assert "fullgame/piece_00.mp4" in work
    assert "fullgame/game.mp4" not in work and "fullgame/video.mp4" not in work
    p = record(ddb)["progress"]["M"]
    assert (p["stage"]["S"], float(p["fraction"]["N"]), p["etaSeconds"]["N"]) == ("upload", 1.0, "0")


def test_run_asks_again_when_the_engine_refuses_an_answer(aws, monkeypatch):
    s3, ddb, rec = aws
    assert go(monkeypatch, "prepare") == 0
    s3.put_object(Bucket=BUCKET, Key=OUT + "answers.json", Body=json.dumps({"kits": {"round": 1, "notKit": [0, 1]}}))
    monkeypatch.setenv("ENGINE_REFUSES", "1")
    assert go(monkeypatch, "run") == 0
    assert "--answers" in calls(rec)[-1]
    q = json.loads(body(s3, OUT + "questions/questions.json"))
    assert q["pending"] == ["kits"] and q["kits"]["problem"] == "too few"
    assert "game.mp4" not in keys(s3, OUT) and "error" not in record(ddb)


def test_an_engine_error_goes_on_the_record_with_its_logs(aws, monkeypatch):
    s3, ddb, rec = aws
    monkeypatch.setenv("ENGINE_FAIL", "footage_corrupt")
    assert go(monkeypatch, "prepare") == 1
    assert record(ddb)["error"]["M"] == {"code": {"S": "footage_corrupt"}, "message": {"S": "segment 3 could not be read"}}
    assert keys(s3, OUT) == ["logs/sync.log"]
    assert "engine/logs/sync.log" in keys(s3, WORK)


def test_missing_footage_is_said_plainly(aws, monkeypatch):
    s3, ddb, rec = aws
    s3.delete_object(Bucket=BUCKET, Key=FOOTAGE + "dev2/dev2_000.mp4")
    assert go(monkeypatch, "prepare") == 1
    assert record(ddb)["error"]["M"]["code"]["S"] == "footage_missing"
    s3.delete_object(Bucket=BUCKET, Key=FOOTAGE + "manifest.json")
    assert go(monkeypatch, "run") == 1
    assert record(ddb)["error"]["M"]["code"]["S"] == "footage_missing"


def test_a_deleted_job_is_not_written_back(aws, monkeypatch):
    s3, ddb, rec = aws
    ddb.delete_item(TableName=TABLE, Key={"pk": {"S": f"JOB#{JOB}"}, "sk": {"S": "JOB"}})
    assert go(monkeypatch, "prepare") == 0
    assert record(ddb) is None


def test_no_gpu_stops_before_any_work(aws, monkeypatch):
    s3, ddb, rec = aws
    monkeypatch.setenv("OSB_REQUIRE_GPU", "1")
    monkeypatch.setattr(cloud.accel, "platform", lambda: "cpu")
    assert go(monkeypatch, "run") == 1
    assert record(ddb)["error"]["M"]["code"]["S"] == "no_gpu"
    assert not rec.exists()


def test_models_must_be_the_ones_the_engine_was_built_for(aws, monkeypatch, tmp_path):
    s3, ddb, rec = aws
    monkeypatch.undo()  # the real fetch_models
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                 "OSB_BUCKET": BUCKET, "OSB_TABLE": TABLE, "OSB_JOB_ID": JOB, "OSB_FOOTAGE_PREFIX": FOOTAGE,
                 "OSB_OUTPUT_PREFIX": OUT, "OSB_WORK_PREFIX": WORK}.items():
        os.environ[k] = v
    try:
        repo = tmp_path / "repo"
        (repo / "packaging" / "cloud").mkdir(parents=True)
        good = b"weights"
        (repo / "packaging" / "cloud" / "models.json").write_text(json.dumps({"models": [
            {"key": "v1/ball_judge.pt", "path": "footage/models/ball_judge.pt", "sha256": hashlib.sha256(good).hexdigest()}]}))
        cloud.REPO, real = repo, cloud.REPO
        c = cloud.Cloud()
        s3.put_object(Bucket=BUCKET, Key="models/v1/ball_judge.pt", Body=b"someone else's weights")
        with pytest.raises(cloud.Failed) as e:
            cloud.fetch_models(c)
        assert e.value.code == "bad_models" and not (repo / "footage" / "models" / "ball_judge.pt").exists()
        s3.put_object(Bucket=BUCKET, Key="models/v1/ball_judge.pt", Body=good)
        cloud.fetch_models(c)
        assert (repo / "footage" / "models" / "ball_judge.pt").read_bytes() == good
    finally:
        cloud.REPO = real
        for k in ("OSB_BUCKET", "OSB_TABLE", "OSB_JOB_ID", "OSB_FOOTAGE_PREFIX", "OSB_OUTPUT_PREFIX", "OSB_WORK_PREFIX"):
            os.environ.pop(k, None)


def test_pictures_finds_every_picture_a_question_names():
    q = {"pitch": {"overlay": "pitch/overlay.jpg", "background": "pitch/background.png",
                   "points": {"a": {"closeup": {"image": "engine/closeups/1/a.jpg", "fit": [1, 2]}}}},
         "kits": {"groups": [{"sheet": "engine/crops/kits1/sheet0.jpg", "crops": [{"image": "engine/crops/kits1/c0_00.jpg"}]}]},
         "cameras": {"panoramas": {"mainLeft": "engine/cameras/main_left.jpg", "secondaryLeft": "engine/cameras/secondary_left.jpg"}}}
    assert len(cloud.pictures(q)) == 7
