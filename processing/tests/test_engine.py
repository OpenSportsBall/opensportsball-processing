# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""engine.py's file contract and the answers it turns into roles.py's
sides, without footage:

    uv run --with pytest --with numpy --with opencv-python-headless --with jsonschema pytest processing/tests/test_engine.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import engine  # noqa: E402

# rec7's kit groups (teams.py, k=8, 2026-11-05): white is components 0 and 7,
# yellow 1 and 2; 5 and 6 are keepers / the referee; 3 and 4 are not kits.
# teams.py's own split put 0 against 2 and left the rest out.
REC7_AUTO = {"A": [0], "B": [2], "keeperOrReferee": [], "kitComponents": [0, 2],
             "perMoment": [10.6, 8.13, 5.85, 1.8, 1.67, 1.23, 0.55, 0.47]}


def teams_dir(tmp_path, auto=REC7_AUTO):
    d = tmp_path / "teams" / "engine"
    d.mkdir(parents=True)
    (d / "sides.json").write_text(json.dumps(auto))
    people = [{"frame": f"m{i % 5}_main", "component": c, "onPitch": True, "box": [0, 0, 10, 60]}
              for i, c in enumerate([0, 1, 2, 3, 4, 5, 6, 7] * 3)]
    (d / "people.json").write_text(json.dumps(people))
    return d


def test_rec7_answer_gives_the_sides_set_by_hand(tmp_path):
    s = engine.sides_from_answer(teams_dir(tmp_path), {"round": 1, "merge": [[0, 7], [1, 2]],
                                                       "keeperOrReferee": [5, 6], "notKit": [3, 4]})
    assert sorted([s["A"], s["B"]]) == [[0, 7], [1, 2]]
    assert s["keeperOrReferee"] == [5, 6]


def test_no_changes_keeps_teams_py_split(tmp_path):
    s = engine.sides_from_answer(teams_dir(tmp_path), {"round": 1})
    assert sorted([s["A"], s["B"]]) == [[0], [2]]


def test_too_few_kit_groups_is_refused(tmp_path):
    assert engine.sides_from_answer(teams_dir(tmp_path), {"round": 1, "merge": [[0, 2]]}) is None


def test_kit_score(tmp_path):
    job = type("J", (), {"dir": tmp_path})()
    groups = [{"component": 0, "people": 300, "crops": [{"id": f"c0_{i}"} for i in range(24)]},
              {"component": 1, "people": 100, "crops": [{"id": f"c1_{i}"} for i in range(24)]}]
    (tmp_path / "questions.json").write_text(json.dumps({"kits": {"groups": groups}}))
    s = engine.kit_score(job, {}, {"round": 1, "tapped": {"0": ["c0_1", "c0_2", "c0_3"], "1": []},
                                   "merge": [[0, 1]]})
    assert s["groupPurity"] == {"0": 0.875, "1": 1.0}
    assert s["purity"] == pytest.approx((0.875 * 300 + 100) / 400, abs=1e-3)
    assert s["structure"] == "merged"


@pytest.mark.parametrize("answers,ok", [
    ({"pitch": {"round": 1, "approve": True, "share": True}}, True),
    ({"pitch": {"round": 2, "points": {"left corner flag": {"yaw": -86.9, "pitch": 7.6},
                                       "right corner flag": {"fromFit": True},
                                       "left near post base": {"skipped": True}}}}, True),
    ({"pitch": {"round": 1}}, False),                                  # neither approve nor points
    ({"pitch": {"round": 1, "approve": True, "points": {}}}, False),   # both
    ({"kits": {"round": 1, "merge": [[0, 7], [1, 2]], "keeperOrReferee": [5, 6], "tapped": {"0": ["c0_01"]}}}, True),
    ({"kits": {"round": 1, "merge": [[3]]}}, False),                   # a merge of one
])
def test_answers_schema(answers, ok):
    if ok:
        engine.validate(answers, "engine-answers.schema.json", "answers")
    else:
        with pytest.raises(engine.Failed):
            engine.validate(answers, "engine-answers.schema.json", "answers")


def test_job_schema(tmp_path):
    engine.validate({"schemaVersion": 1, "recording": "footage/rec7", "output": {"height": 1080},
                     "options": {"maxChunks": 1}}, "engine-job.schema.json", "job")
    with pytest.raises(engine.Failed):
        engine.validate({"schemaVersion": 1}, "engine-job.schema.json", "job")


def test_events_match_the_schema(capsys):
    engine.emit("progress", stage="game", fraction=0.25, chunks=10, ballDone=3, followDone=2, elapsedS=100, etaS=None)
    engine.emit("error", code="footage_missing", message="x")
    for line in capsys.readouterr().out.splitlines():
        engine.validate(json.loads(line), "engine-event.schema.json", "event")


# ------------------------------------------------------------ label, share


def make_job(tmp_path):
    rec = tmp_path / "rec"
    (rec / "engine").mkdir(parents=True)
    (rec / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "gameId": "g1", "sessionId": "s1",
                                                   "createdAt": "2026-11-05T06:07:37Z", "devices": []}))
    (tmp_path / "job.json").write_text(json.dumps({"schemaVersion": 1, "recording": "rec"}))
    return engine.Job(tmp_path / "job.json")


def make_set(job, name, kind, ids):
    sd = job.dir / "labels" / name
    sd.mkdir(parents=True)
    items = [{"id": i, "image": f"images/{i}.jpg", "videoT": 10.0 * n,
              "pick": "detected" if kind == "ball" else "random"} for n, i in enumerate(ids)]
    for it in items:
        (sd / it["image"]).parent.mkdir(exist_ok=True)
        (sd / it["image"]).write_bytes(b"jpg")
    engine.write_set(sd, {"schemaVersion": 1, "set": name, "kind": kind, "labelled": 0, "items": items})
    return sd


def test_ball_labels_in_the_label_page_format(tmp_path):
    job = make_job(tmp_path)
    sd = make_set(job, "ball-1", "ball", ["a", "b", "c"])
    engine.apply_labels(job, {"set": "ball-1", "ball": {"a": {"status": "ball", "x": 10, "y": 20},
                                                        "b": {"status": "none"}}})
    engine.apply_labels(job, {"set": "ball-1", "ball": {"b": {"status": "unsure"}}})  # a later answer replaces
    labels = json.loads((sd / "labels.json").read_text())
    assert labels["a"]["balls"] == [{"x": 10, "y": 20, "inPlay": True}]
    assert labels["b"]["status"] == "unsure" and labels["b"]["balls"] == []
    assert json.loads((sd / "set.json").read_text())["labelled"] == 2


def test_moment_boxes_in_the_playeval_format(tmp_path):
    job = make_job(tmp_path)
    sd = make_set(job, "moments-1", "moments", ["m1", "m2"])
    engine.apply_labels(job, {"set": "moments-1", "moments": {
        "m1": {"yaw0": 12.0, "yaw1": 4.0, "pitch0": 6.0, "pitch1": 2.0}, "m2": {"skipped": True}}})
    boxes = json.loads((sd / "boxes.json").read_text())
    assert list(boxes) == ["m1"]  # skipped moments aren't boxes
    assert (boxes["m1"]["yaw0"], boxes["m1"]["yaw1"], boxes["m1"]["pitch0"], boxes["m1"]["pitch1"]) == (4, 12, 2, 6)


def test_label_answers_must_match_the_set(tmp_path):
    job = make_job(tmp_path)
    make_set(job, "ball-1", "ball", ["a"])
    for bad in ({"set": "ball-1", "ball": {"zz": {"status": "none"}}},          # not in the set
                {"set": "ball-1", "moments": {"a": {"skipped": True}}},         # wrong kind
                {"set": "ball-9", "ball": {"a": {"status": "none"}}}):          # no such set
        with pytest.raises(engine.Failed):
            engine.apply_labels(job, bad)


@pytest.mark.parametrize("answers,ok", [
    ({"set": "ball-1", "ball": {"a": {"status": "ball", "x": 1, "y": 2}}}, True),
    ({"set": "ball-1", "ball": {"a": {"status": "ball"}}}, False),            # a ball needs a position
    ({"set": "moments-1", "moments": {"m": {"yaw0": 1, "yaw1": 2, "pitch0": 1, "pitch1": 2}}}, True),
    ({"set": "ball-1"}, False),                                               # nothing answered
])
def test_label_answers_schema(answers, ok):
    if ok:
        engine.validate(answers, "engine-label-answers.schema.json", "answers")
    else:
        with pytest.raises(engine.Failed):
            engine.validate(answers, "engine-label-answers.schema.json", "answers")


def test_an_unfinished_set_carries_on(tmp_path):
    job = make_job(tmp_path)
    make_set(job, "ball-1", "ball", ["a"])
    (job.dir / "labels" / "ball-2").mkdir()  # stopped before set.json
    assert engine.label_dir(job, "ball")[0] == "ball-2"
    assert engine.label_dir(job, "moments")[0] == "moments-1"


def test_share_follows_the_users_switches(tmp_path, capsys):
    job = make_job(tmp_path)
    (job.rec / "pitch").mkdir()
    (job.rec / "pitch" / "pitch.json").write_text(json.dumps({"lengthM": 100.0, "widthM": 60.0, "heightM": 6.0}))
    (job.rec / "pitch" / "background.png").write_bytes(b"png")
    st = job.state()
    st["pitch"].update(status="approved", approvedRound=1)
    st["kits"]["status"] = "answered"
    st["applied"] = [{"question": "pitch", "answer": {"round": 1, "approve": True}, "at": "x"},
                     {"question": "kits", "answer": {"round": 1, "share": False}, "at": "x"}]
    job.save_state(st)
    sd = make_set(job, "ball-1", "ball", ["a", "b"])
    engine.apply_labels(job, {"set": "ball-1", "ball": {"a": {"status": "none"}}})
    capsys.readouterr()
    engine.share(job, ["pitch", "kits", "ball", "moments"])
    events = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    for e in events:
        engine.validate(e, "engine-event.schema.json", "event")
    by = {e["name"]: e for e in events}
    assert by["kits"]["path"] is None                    # the user said no
    pitch = json.loads((job.dir / "share" / "pitch.json").read_text())
    assert pitch["summary"]["approvedFirstGuess"] is True
    assert {f["path"] for f in pitch["files"]} == {"pitch/background.png", "pitch/pitch.json"}
    ball = json.loads((job.dir / "share" / "ball-1.json").read_text())
    assert [f["path"] for f in ball["files"] if f["path"].endswith(".jpg")] == ["engine/labels/ball-1/images/a.jpg"]
    assert ball["recording"]["gameId"] == "g1"


def test_ball_sets_hold_out_the_last_quarter_of_each_chunk(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    segs = [{"index": i, "durationMs": 300000} for i in range(4)]
    (job.rec / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "gameId": "g1", "sessionId": "s1",
                                                       "createdAt": "x", "devices": [{"role": "main", "segments": segs}]}))
    (job.rec / "fullgame").mkdir()
    (job.rec / "fullgame" / "game.mp4").write_bytes(b"")
    (job.rec / "detect").mkdir()
    for t in (0, 600):
        (job.rec / "detect" / f"ball_0_{t}_600.win.json").write_text("{}")

    def fake_labelset(job_, name, args, **kw):  # labelset.py ball: frames at these game times
        sd = Path(args[args.index("--out") + 1])
        p = sd / "items.json"
        items = json.loads(p.read_text()) if p.exists() else []
        for g in ((100, 500) if "_0_0_" in args[3] else (700, 1100)):
            items.append({"id": f"b{g}", "image": f"images/b{g}.jpg", "segment": g // 300, "t": g % 300,
                          "width": 3840, "height": 2160, "candidates": []})
        p.write_text(json.dumps(items))
    monkeypatch.setattr(engine, "step", fake_labelset)
    monkeypatch.setattr(engine, "game_seconds", lambda job_: 1200.0)
    engine.label(job, "ball", 4, [], 0)
    sd = job.dir / "labels" / "ball-1"
    assert json.loads((sd / "heldout.json").read_text())["gameS"] == [[450, 600], [1050, 1200]]
    held = {it["id"]: it["heldOut"] for it in json.loads((sd / "set.json").read_text())["items"]}
    assert held == {"b100": False, "b500": True, "b700": False, "b1100": True}


# ------------------------------------------------- no stitch: cameras question


def test_no_stitch_asks_left_right_then_needs_taps(tmp_path, monkeypatch, capsys):
    """rec5's path through prepare with stand-in steps: the stitch fails, only the cameras
    question is asked; the answer builds rough cameras; approving the pitch is refused until
    tapped points have fitted the cameras (jointfit taps, search, install)."""
    job = make_job(tmp_path)
    segs = [{"index": i, "durationMs": 300000, "objectKey": f"m/{i}.mp4"} for i in range(4)]
    (job.rec / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "gameId": "g1", "sessionId": "s1",
        "createdAt": "x", "devices": [{"role": "main", "segments": segs},
                                      {"role": "secondary", "segments": [dict(s, objectKey=f"s/{i}.mp4") for i, s in enumerate(segs)]}]}))
    for d in ("m", "s"):
        (job.rec / d).mkdir()
        for i in range(4):
            (job.rec / d / f"{i}.mp4").write_bytes(b"")
    calls = []

    def fake_step(job_, name, args, *, done=None, uv_with=(), watch=None):
        calls.append(name)
        if done is not None and done.exists():
            return
        r = job_.rec
        if name == "sync":
            (r / "phase0").mkdir(exist_ok=True); (r / "phase0" / "report.json").write_text("{}")
        elif name == "pitch" and not (r / "phase0" / "cameras.json").exists():
            raise engine.Failed("calibration_failed", "the two phones' views couldn't be matched")
        elif name == "camera-choices":
            (job_.dir / "cameras").mkdir(parents=True, exist_ok=True)
            for ph in ("main", "secondary"):
                (job_.dir / "cameras" / f"{ph}_left.jpg").write_bytes(b"jpg")
        elif name == "rough-cameras":
            assert args[-2:] == ["--left", "secondary"]
            (r / "phase0" / "cameras.json").write_text(json.dumps({"rough": {"left": "secondary"}}))
        elif name in ("pitch", "cameras-install-1"):
            (r / "pitch").mkdir(exist_ok=True)
            (r / "pitch" / "pitch.json").write_text(json.dumps({"fittedWith": "lines" if name == "pitch" else "joint+taps"}))
            (r / "pitch" / "review.json").unlink(missing_ok=True)
        elif name.startswith("goals"):
            (r / "pitch" / "goals.json").write_text("{}")
        elif name == "kit-sample":
            (r / "labels_engine").mkdir(); (r / "labels_engine" / "items.json").write_text("[]")
        elif name == "kits":
            teams_dir(tmp_path / "rec")
        elif name == "pitch-review":
            (r / "pitch" / "review.json").write_text(json.dumps({"pitch": json.loads((r / "pitch" / "pitch.json").read_text()),
                                                                "lines": [], "goals": [], "points": {"left corner flag": {"yaw": 1, "pitch": 2}}}))
    monkeypatch.setattr(engine, "step", fake_step)
    monkeypatch.setattr(engine, "write_questions", lambda job_, st: engine.save_questions(job_, {
        "schemaVersion": 1, "recording": "r", "pending": engine.pending_questions(st),
        **({"cameras": engine.cameras_question(job_, st)} if st.get("cameras") else {}),
        **({} if engine.pending_questions(st) == ["cameras"] else {"pitch": {"round": st["pitch"]["round"],
            "status": st["pitch"]["status"], "background": "b", "overlay": "o", "fit": {}, "lines": [], "points": {},
            **({"approveAllowed": False} if st["pitch"].get("approveAllowed") is False else {}),
            **({"problem": st["pitch"]["problem"]} if st["pitch"].get("problem") else {})},
            "kits": {"round": 1, "status": "pending", "k": 8, "groups": [], "splitAllowed": True}})}))

    def questions():
        return json.loads((job.dir / "questions.json").read_text())

    assert engine.prepare(job, None) == 0
    q = questions()
    assert q["pending"] == ["cameras"] and "pitch" not in q
    assert q["cameras"]["panoramas"] == {"mainLeft": "engine/cameras/main_left.jpg",
                                         "secondaryLeft": "engine/cameras/secondary_left.jpg"}
    assert engine.prepare(job, None) == 0 and "pitch" not in calls[calls.index("camera-choices") + 1:]  # asked again, no retry

    engine.prepare(job, {"cameras": {"round": 1, "leftPhone": "secondary"}})
    q = questions()
    assert q["pending"] == ["pitch", "kits"] and q["pitch"]["approveAllowed"] is False
    assert q["cameras"]["status"] == "answered" and q["cameras"]["leftPhone"] == "secondary"

    engine.prepare(job, {"pitch": {"round": 1, "approve": True}})
    q = questions()
    assert q["pitch"]["status"] == "pending" and q["pitch"]["round"] == 1 and "problem" in q["pitch"]

    calls.clear()
    engine.prepare(job, {"pitch": {"round": 1, "points": {"left corner flag": {"yaw": 1.5, "pitch": 2.5}}}})
    assert [c for c in calls if c.split("-")[0] in ("pitch", "cameras", "goals")] == \
        ["pitch", "goals", "pitch-taps-1", "cameras-search-1", "cameras-install-1", "goals-1", "pitch-review"]
    q = questions()
    assert q["pitch"]["round"] == 2 and "approveAllowed" not in q["pitch"] and "problem" not in q["pitch"]

    engine.prepare(job, {"pitch": {"round": 2, "approve": True}})
    assert questions()["pending"] == ["kits"]
    for line in capsys.readouterr().out.splitlines():
        engine.validate(json.loads(line), "engine-event.schema.json", "event")


@pytest.mark.parametrize("answers,ok", [
    ({"cameras": {"round": 1, "leftPhone": "main"}}, True),
    ({"cameras": {"round": 1, "leftPhone": "left"}}, False),
    ({"cameras": {"round": 1}}, False),
])
def test_cameras_answer_schema(answers, ok):
    if ok:
        engine.validate(answers, "engine-answers.schema.json", "answers")
    else:
        with pytest.raises(engine.Failed):
            engine.validate(answers, "engine-answers.schema.json", "answers")


# ------------------------------------------------- footage hashes, disk, cleanup


def footage_job(tmp_path, segs=3, seg_ms=600000):
    import hashlib

    job = make_job(tmp_path)
    devs = []
    for role in ("main", "secondary"):
        ss = []
        for i in range(segs):
            key = f"{role}/{i}.mp4"
            data = f"{role} {i}".encode()
            (job.rec / role).mkdir(exist_ok=True)
            (job.rec / key).write_bytes(data)
            ss.append({"index": i, "objectKey": key, "bytes": len(data), "durationMs": seg_ms,
                       "sha256": hashlib.sha256(data).hexdigest()})
        devs.append({"role": role, "segments": ss})
    (job.rec / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "gameId": "g1", "sessionId": "s1",
                                                       "createdAt": "x", "devices": devs}))
    return job


def test_footage_is_hashed_once_and_a_bad_file_refused(tmp_path, monkeypatch, capsys):
    job = footage_job(tmp_path)
    hashed = []
    real = engine.sha256_file
    monkeypatch.setattr(engine, "sha256_file", lambda p: hashed.append(p.name) or real(p))
    engine.check_footage(job)
    assert len(hashed) == 6
    engine.check_footage(job)
    assert len(hashed) == 6  # remembered by size and modification time
    (job.rec / "secondary" / "1.mp4").write_bytes(b"secondaRy 1")  # same size, other bytes: a broken download
    with pytest.raises(engine.Failed) as e:
        engine.check_footage(job)
    assert e.value.code == "footage_corrupt" and e.value.exit_code == engine.EXIT_BAD_INPUT
    assert len(hashed) == 7  # only the changed file was hashed again
    for line in capsys.readouterr().out.splitlines():
        engine.validate(json.loads(line), "engine-event.schema.json", "event")


def test_disk_estimate_and_check(tmp_path, monkeypatch):
    job = footage_job(tmp_path, segs=10)        # 6000 s, the 1440p default
    monkeypatch.setattr(engine, "game_seconds", lambda j: 6000.0)
    need = engine.disk_needed(job)
    assert 46e9 < need < 52e9                    # pieces 12 + video.mp4 12 + game.mp4 25 GB (rec7 used 11 + 12 + 23)
    pieces = job.rec / "fullgame" / "pieces"
    pieces.mkdir(parents=True)
    for k in range(5):
        (pieces / f"piece_{k:02d}.mp4").write_bytes(b"")
    assert engine.disk_needed(job) == pytest.approx(need - 5 * 600 * engine.PIECE_BPS, rel=1e-3)
    import collections
    monkeypatch.setattr("shutil.disk_usage", lambda p: collections.namedtuple("U", "total used free")(0, 0, 10e9))
    with pytest.raises(engine.Failed) as e:
        engine.check_disk(job)
    assert e.value.code == "not_enough_disk"


@pytest.mark.parametrize("length,keep,ok", [(5999.0, False, True), (5999.0, True, True), (3000.0, False, False)])
def test_cleanup_after_a_checked_game(tmp_path, monkeypatch, capsys, length, keep, ok):
    job = footage_job(tmp_path, segs=10)
    job.keep_intermediates = keep
    monkeypatch.setattr(engine, "game_seconds", lambda j: 6000.0)
    monkeypatch.setattr(engine, "video_seconds", lambda p: length)
    out = job.rec / "fullgame"
    (out / "pieces").mkdir(parents=True)
    for k in range(10):
        (out / "pieces" / f"piece_{k:02d}.mp4").write_bytes(b"x" * 10)
    for n in ("video.mp4", "audio.m4a", "game.mp4"):
        (out / n).write_bytes(b"x" * 100)
    if not ok:
        with pytest.raises(engine.Failed) as e:
            engine.finish_output(job, out / "game.mp4", 1.0)
        assert e.value.code == "output_check_failed" and (out / "video.mp4").exists() and not (out / "done.json").exists()
        return
    engine.finish_output(job, out / "game.mp4", 1.0)
    assert (out / "done.json").exists() and (out / "game.mp4").exists()
    assert (out / "video.mp4").exists() == keep and (out / "pieces").exists() == keep
    ev = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert ev[-1]["step"] == "cleanup" and ev[-1]["freedBytes"] == (0 if keep else 300)


def test_the_youtube_copy_is_counted_in_the_disk_check(tmp_path, monkeypatch):
    job = footage_job(tmp_path, segs=10)
    monkeypatch.setattr(engine, "game_seconds", lambda j: 6000.0)
    job.height = 1080
    need = engine.disk_needed(job)               # pieces 12 + video.mp4 12 + game.mp4 12 GB
    job.youtube_copy = True
    copy = 5999 * engine.GAME_BPS[1440]          # ~25 GB (the last chunk stops 1 s short)
    assert engine.disk_needed(job) == pytest.approx(need + copy)
    out = job.rec / "fullgame"
    out.mkdir()
    (out / "game.mp4").write_bytes(b"x")
    (out / "done.json").write_text("{}")         # the game is done: only the copy is still to be written
    assert engine.disk_needed(job) == pytest.approx(copy)
    (out / "game-youtube.mp4").write_bytes(b"x")
    (out / "done.json").write_text(json.dumps({"youtube": "game-youtube.mp4"}))
    assert engine.disk_needed(job) == 0


def test_a_1440p_game_needs_no_youtube_copy(tmp_path):
    job = make_job(tmp_path)
    for height, wanted in ((1080, True), (1440, False)):
        (tmp_path / "job.json").write_text(json.dumps({"schemaVersion": 1, "recording": "rec",
                                                       "output": {"height": height, "youtubeCopy": True}}))
        assert engine.Job(tmp_path / "job.json").youtube_copy == wanted
    assert not job.youtube_copy and engine.youtube_copy(job, job.rec / "fullgame" / "game.mp4") == {}


def test_the_youtube_copy_of_a_finished_game(tmp_path, monkeypatch, capsys):
    """With ffmpeg itself: game.mp4 stays, game-youtube.mp4 is 1440p with the audio, and a rerun
    of a finished game makes only the copy, once."""
    import subprocess

    job = make_job(tmp_path)
    job.height, job.youtube_copy = 1080, True
    monkeypatch.setattr(engine, "game_seconds", lambda j: 2.0)
    out = job.rec / "fullgame"
    out.mkdir()
    game = out / "game.mp4"
    subprocess.run([engine.accel.ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(game)], check=True)
    before = game.read_bytes()
    (out / "done.json").write_text(json.dumps({"output": "game.mp4", "seconds": 5}))
    assert engine.youtube_made(job) is None and "YouTube copy" in engine.next_step(job, None)
    got = engine.youtube_copy(job, game)
    copy = out / "game-youtube.mp4"
    assert got == {"youtubeOutput": str(copy)} and engine.youtube_made(job) == copy
    probe = subprocess.run([engine.accel.ffprobe(), "-v", "error", "-show_entries", "stream=codec_type,width,height",
                            "-of", "json", str(copy)], capture_output=True, text=True, check=True)
    streams = {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}
    assert (streams["video"]["width"], streams["video"]["height"]) == (2560, 1440) and "audio" in streams
    assert game.read_bytes() == before and not (out / "game-youtube.part.mp4").exists()
    d = json.loads((out / "done.json").read_text())
    assert d["youtube"] == "game-youtube.mp4" and d["youtubeBytes"] == copy.stat().st_size and d["seconds"] == 5
    assert engine.next_step(job, None).startswith("done")
    capsys.readouterr()
    st = job.state()
    st["pitch"]["status"] = "approved"
    st["kits"]["status"] = "answered"
    job.save_state(st)
    (job.dir / "questions.json").write_text("{}")
    monkeypatch.setattr(engine, "check_footage", lambda j: {})
    monkeypatch.setattr(engine, "step", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rendered again")))
    assert engine.run(job, None) == engine.EXIT_OK
    ev = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    for e in ev:
        engine.validate(e, "engine-event.schema.json", "event")
    assert ev[-1]["event"] == "done" and ev[-1]["youtubeOutput"] == str(copy)
    assert {"step": "youtube", "status": "skipped"}.items() <= ev[-2].items()


def test_a_youtube_copy_that_fails_keeps_the_game(tmp_path, monkeypatch, capsys):
    job = make_job(tmp_path)
    job.height, job.youtube_copy = 1080, True
    monkeypatch.setattr(engine, "game_seconds", lambda j: 2.0)
    out = job.rec / "fullgame"
    out.mkdir()
    (out / "game.mp4").write_bytes(b"not a video")
    (out / "done.json").write_text(json.dumps({"output": "game.mp4"}))
    with pytest.raises(engine.Failed) as e:
        engine.youtube_copy(job, out / "game.mp4")
    assert e.value.code == "youtube_copy_failed" and (out / "game.mp4").exists()
    assert not (out / "game-youtube.mp4").exists() and "youtube" not in json.loads((out / "done.json").read_text())


def test_a_game_that_is_1440p_already_is_its_own_youtube_copy(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    job.height, job.youtube_copy = 1080, True    # asked for after game.mp4 was made at 1440p
    out = job.rec / "fullgame"
    out.mkdir()
    (out / "game.mp4").write_bytes(b"x")
    (out / "done.json").write_text(json.dumps({"output": "game.mp4"}))
    monkeypatch.setattr(engine, "video_height", lambda p: 1440)
    assert engine.youtube_copy(job, out / "game.mp4") == {"youtubeOutput": str(out / "game.mp4")}
    assert engine.youtube_made(job) == out / "game.mp4" and not (out / "game-youtube.mp4").exists()


def test_a_finished_game_isnt_rendered_again(tmp_path, monkeypatch, capsys):
    job = footage_job(tmp_path)
    st = job.state()
    st["pitch"]["status"] = "approved"
    st["kits"]["status"] = "answered"
    job.save_state(st)
    (job.dir / "questions.json").write_text("{}")
    out = job.rec / "fullgame"
    out.mkdir()
    (out / "game.mp4").write_bytes(b"x")
    (out / "done.json").write_text(json.dumps({"seconds": 30000}))

    def no_step(*a, **k):
        raise AssertionError("rendered again")
    monkeypatch.setattr(engine, "step", no_step)
    assert engine.run(job, None) == engine.EXIT_OK
    ev = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert ev[-1]["event"] == "done" and ev[-2] == {**ev[-2], "step": "game", "status": "skipped"}


# ------------------------------------- questions a script or a model answers


def asked_job(tmp_path):
    """A prepared recording, small: a 20 x 8 degree panorama with one line and
    two keypoints (one off the picture), and two kit groups of three people."""
    import cv2
    import numpy as np

    job = make_job(tmp_path)
    rec = job.rec
    (rec / "pitch").mkdir()
    pano = {"image": "background.png", "degPerPx": 0.03, "yawRange": [-10.0, 10.0], "pitchRange": [0.0, 8.0]}
    cv2.imwrite(str(rec / "pitch" / "background.png"), np.full((266, 666, 3), 90, np.uint8))
    (rec / "pitch" / "pitch.json").write_text("{}")
    (rec / "pitch" / "review.json").write_text(json.dumps({
        "pitch": {"lengthM": 100.0, "widthM": 60.0, "cameraHeightM": 6.0, "fittedWith": "lines"},
        "lines": [[[-9.0, 4.0], [9.0, 4.0]]], "goals": [], "panorama": pano,
        "points": {"left corner flag": {"yaw": -9.5, "pitch": 4.0}, "halfway far end": {"yaw": 0.0, "pitch": 2.0},
                   "right corner flag": {"yaw": 14.0, "pitch": 4.0}}}))
    frames = rec / "labels_engine"
    frames.mkdir()
    cv2.imwrite(str(frames / "f0.jpg"), np.full((400, 600, 3), 120, np.uint8))
    (frames / "items.json").write_text(json.dumps([{"frames": [{"id": "m0_main", "image": "f0.jpg"}]}]))
    d = rec / "teams" / "engine"
    d.mkdir(parents=True)
    (d / "people.json").write_text(json.dumps([
        {"frame": "m0_main", "component": c, "onPitch": True, "box": [20 + 60 * i, 100, 50 + 60 * i, 180]}
        for i, c in enumerate([0, 0, 0, 1, 1, 1])]))
    st = job.state()
    q = engine.write_questions(job, st)
    job.save_state(st)
    return job, q


def flags(**kw):
    import argparse

    base = dict(left=None, approve=False, closeup=[], pano=[], point=[], keep=[], skip=[], ok=False, merge=[],
                split=None, keeper=None, not_kit=None, tapped=[], share=False)
    return argparse.Namespace(**{**base, **kw})


def test_questions_say_what_is_asked_and_come_with_pictures(tmp_path):
    job, q = asked_job(tmp_path)
    for name in ("pitch", "kits"):
        assert q[name]["ask"] and any("osb-engine answer" in h for h in q[name]["howToAnswer"])
    pts = q["pitch"]["points"]
    assert "closeup" not in pts["right corner flag"]  # the fit puts it off the panorama
    for name in ("left corner flag", "halfway far end"):  # one against the picture's edge, one in the middle
        c = pts[name]["closeup"]
        assert (job.rec / c["image"]).exists()
        assert c["yaw0"] + c["fit"][0] * c["degPerPx"] == pytest.approx(pts[name]["yaw"], abs=c["degPerPx"])
        assert c["pitch0"] + c["fit"][1] * c["degPerPx"] == pytest.approx(pts[name]["pitch"], abs=c["degPerPx"])
        assert 0 <= c["fit"][0] < c["width"] and 0 <= c["fit"][1] < c["height"]
    for g in q["kits"]["groups"]:
        assert (job.rec / g["sheet"]).exists() and len(g["crops"]) == 3


def test_the_brief_is_one_document_with_whole_paths(tmp_path, capsys):
    job, _ = asked_job(tmp_path)
    capsys.readouterr()
    assert engine.questions_brief(job) == 0
    b = json.loads(capsys.readouterr().out)
    assert b["pending"] == ["pitch", "kits"] and [x["question"] for x in b["questions"]] == ["pitch", "kits"]
    pitch, kits = b["questions"]
    assert Path(pitch["overlay"]).is_absolute() and "lines" not in pitch
    assert Path(pitch["points"]["halfway far end"]["closeup"]["image"]).exists()
    assert pitch["points"]["right corner flag"]["closeup"] is None
    assert Path(kits["groups"][0]["sheet"]).exists() and kits["groups"][0]["crops"] == ["c0_00", "c0_01", "c0_02"]
    assert "answer" in b["next"]


def test_an_answer_from_flags_is_what_the_app_would_send(tmp_path, capsys):
    job, q = asked_job(tmp_path)
    c = q["pitch"]["points"]["halfway far end"]["closeup"]
    x, y = c["fit"][0] + 30, c["fit"][1] - 15  # 30 px right and 15 px up of the fit, in the close-up
    assert engine.answer(job, "pitch", flags(closeup=[f"halfway far end={x},{y}"], pano=["left corner flag=20,133"],
                                             skip=["right corner flag"])) == 0
    assert engine.answer(job, "kits", flags(merge=["0,1"], tapped=["c0_01,c1_02"], share=True)) == 0
    a = json.loads((job.dir / "answers.json").read_text())  # both answers, in the one file
    engine.validate(a, "engine-answers.schema.json", "answers")
    assert a["pitch"]["share"] is False and a["kits"]["share"] is True  # sharing only when asked for
    assert a["pitch"]["points"]["halfway far end"] == {"yaw": pytest.approx(0.3, abs=0.011), "pitch": pytest.approx(1.85, abs=0.011)}
    assert a["pitch"]["points"]["left corner flag"] == {"yaw": pytest.approx(-9.4), "pitch": pytest.approx(3.99)}
    assert a["pitch"]["points"]["right corner flag"] == {"skipped": True}
    assert a["kits"] == {"round": 1, "share": True, "merge": [[0, 1]], "tapped": {"0": ["c0_01"], "1": ["c1_02"]}}
    assert "prepare" in json.loads(capsys.readouterr().out.split("\n}\n")[-2] + "\n}")["next"]


@pytest.mark.parametrize("question, kw, says", [
    ("pitch", {}, "needs --approve"),
    ("pitch", {"approve": True, "keep": ["halfway far end"]}, "stands alone"),
    ("pitch", {"closeup": ["no such point=1,2"]}, "no such point"),
    ("pitch", {"closeup": ["right corner flag=1,2"]}, "no close-up"),
    ("pitch", {"closeup": ["halfway far end=5000,2"]}, "outside its close-up"),
    ("pitch", {"keep": ["halfway far end"], "skip": ["halfway far end"]}, "more than once"),
    ("kits", {}, "needs --ok"),
    ("kits", {"ok": True, "merge": ["0,1"]}, "stands alone"),
    ("kits", {"merge": ["0,9"]}, "no such group"),
    ("kits", {"merge": ["0"]}, "two or more"),
    ("kits", {"tapped": ["c9_00"]}, "no such crop"),
    ("cameras", {"left": "main"}, "isn't waiting"),
    ("goals", {}, "takes one of"),
])
def test_an_answer_that_doesnt_fit_the_question_is_refused(tmp_path, question, kw, says):
    job, _ = asked_job(tmp_path)
    with pytest.raises(engine.Failed) as e:
        engine.answer(job, question, flags(**kw))
    assert says in e.value.message and e.value.exit_code == engine.EXIT_BAD_INPUT
    assert not (job.dir / "answers.json").exists()


def test_approving_waits_for_marked_points_when_the_cameras_came_from_the_pitch(tmp_path):
    job, _ = asked_job(tmp_path)
    st = job.state()
    st["pitch"]["approveAllowed"] = False
    q = engine.write_questions(job, st)
    assert "isn't allowed yet" in q["pitch"]["howToAnswer"][0]
    with pytest.raises(engine.Failed, match="can't be approved yet"):
        engine.answer(job, "pitch", flags(approve=True))


def test_status_says_what_to_run_next(tmp_path, capsys):
    job = make_job(tmp_path)
    engine.status(job)
    assert "prepare" in json.loads(capsys.readouterr().out)["next"]
    job, _ = asked_job(tmp_path / "asked")
    capsys.readouterr()
    engine.status(job)
    line = capsys.readouterr().out
    engine.validate(json.loads(line), "engine-event.schema.json", "event")
    assert "questions" in json.loads(line)["next"]


# ------------------------------ the ball, chunk by chunk, while the game runs


def follow_log(job, k, aim, detected=(0, 100)):
    logs = job.rec / "fullgame" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"follow_{k:02d}.log").write_text(
        f"ball: 12 on-pitch candidates, 0 static spots, path detected in {detected[0]}/{detected[1]} frames;"
        f" aim at the ball in {aim}% of the clip\n")


@pytest.mark.parametrize("shares,final,low", [
    ([31], False, None),            # rec7 with the standard judge: chunk 0 alone looks fine
    ([31, 2], False, True),         # ... and the second chunk says it isn't
    ([31, 0, 2, 1, 0, 2, 1, 0, 2, 1], True, True),
    ([28, 64, 41], False, False),   # rec7 with its own judge
    ([4], True, True),              # a one-chunk game is judged on that chunk once it is done
    ([40, 3, 55, 60], False, False),  # one poor chunk (a long stoppage) doesn't say the ball is unknown
])
def test_a_ball_the_models_dont_know_shows_from_the_second_chunk(tmp_path, shares, final, low):
    job = make_job(tmp_path)
    assert engine.ball_share(job) == {}
    for k, s in enumerate(shares):
        follow_log(job, k, s)
    b = engine.ball_share(job, final=final)
    assert b["ballAimByChunk"] == shares and b.get("ballLow") is low


def ready_to_run(tmp_path):
    job = footage_job(tmp_path)
    st = job.state()
    st["pitch"]["status"] = "approved"
    st["kits"]["status"] = "answered"
    job.save_state(st)
    (job.dir / "questions.json").write_text("{}")
    return job


def fullgame_events(job, lines):
    with (job.dir / "events.jsonl").open("a") as f:
        for e in lines:
            f.write(json.dumps({"at": "2026-11-10T20:00:00-0800", **e}) + "\n")


def test_run_says_the_ball_is_low_after_the_second_chunk(tmp_path, monkeypatch, capsys):
    job = ready_to_run(tmp_path)
    monkeypatch.setattr(engine, "check_disk", lambda j: None)
    monkeypatch.setattr(engine, "finish_output", lambda *a: None)

    def fake_fullgame(job_, name, args, *, done=None, uv_with=(), watch=None):
        fullgame_events(job, [{"event": "plan", "chunks": 3, "chunkS": 600, "gameS": 1800, "done": []}])
        watch(None)
        for k, aim in enumerate([31, 2, 0]):
            fullgame_events(job, [{"event": "start", "stage": "ball", "chunk": k},
                                  {"event": "chunk", "stage": "ball", "chunk": k, "ok": True, "seconds": 5}])
            follow_log(job, k, aim)
            fullgame_events(job, [{"event": "start", "stage": "follow", "chunk": k},
                                  {"event": "chunk", "stage": "follow", "chunk": k, "ok": True, "seconds": 7}])
            watch(None)
    monkeypatch.setattr(engine, "step", fake_fullgame)
    assert engine.run(job, None) == engine.EXIT_OK
    ev = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    for e in ev:
        engine.validate(e, "engine-event.schema.json", "event")
    ball = [e for e in ev if e["event"] == "ball"]
    assert [(e["chunksDone"], e.get("ballLow")) for e in ball] == [(1, None), (2, True), (3, True)]
    assert ball[1]["chunks"] == 3 and ball[1]["ballAimByChunk"] == [31, 2]
    assert ev[-1]["event"] == "done" and ev[-1]["ballLow"] is True


def test_status_lists_each_chunk(tmp_path, capsys):
    job = ready_to_run(tmp_path)
    engine.status(job)
    assert json.loads(capsys.readouterr().out)["chunkDetails"] == []
    # An earlier run did chunk 0 and was stopped in chunk 1; this one is in chunk 2's ball stage.
    fullgame_events(job, [{"event": "plan", "chunks": 3, "chunkS": 600, "gameS": 1799, "done": []},
                          {"event": "start", "stage": "ball", "chunk": 0},
                          {"event": "chunk", "stage": "ball", "chunk": 0, "ok": True, "seconds": 1500.5, "log": "x"},
                          {"event": "start", "stage": "ball", "chunk": 1},
                          {"event": "start", "stage": "follow", "chunk": 0},
                          {"event": "chunk", "stage": "follow", "chunk": 0, "ok": True, "seconds": 1400},
                          {"event": "plan", "chunks": 3, "chunkS": 600, "gameS": 1799, "done": [0]},
                          {"event": "start", "stage": "ball", "chunk": 1},
                          {"event": "chunk", "stage": "ball", "chunk": 1, "ok": True, "seconds": 900},
                          {"event": "start", "stage": "follow", "chunk": 1},
                          {"event": "chunk", "stage": "follow", "chunk": 1, "ok": False, "seconds": 20, "log": "x"},
                          {"event": "start", "stage": "ball", "chunk": 2}])
    follow_log(job, 0, 31, detected=(50, 200))
    (job.rec / "fullgame" / "logs" / "ball_00.log").write_text("")
    engine.status(job)
    line = json.loads(capsys.readouterr().out)
    engine.validate(line, "engine-event.schema.json", "event")
    c = line["chunkDetails"]
    assert [(x["chunk"], x["startS"], x["seconds"]) for x in c] == [(0, 0, 600), (1, 600, 600), (2, 1200, 599)]
    assert c[0]["ball"] == {"status": "done", "seconds": 1500.5, "log": str(job.rec / "fullgame" / "logs" / "ball_00.log")}
    assert c[0]["follow"]["status"] == "done" and c[0]["ballAimPct"] == 31 and c[0]["ballDetectedPct"] == 25
    assert c[1]["ball"]["status"] == "done" and c[1]["follow"] == {"status": "failed", "seconds": 20}
    assert c[2]["ball"]["status"] == "running" and c[2]["follow"] == {"status": "waiting"}
    assert line["ballAimByChunk"] == [31] and "ballLow" not in line


def test_a_ball_set_can_be_made_while_the_game_runs(tmp_path, monkeypatch, capsys):
    job = make_job(tmp_path)
    segs = [{"index": i, "durationMs": 300000} for i in range(6)]
    (job.rec / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "gameId": "g1", "sessionId": "s1",
                                                       "createdAt": "x", "devices": [{"role": "main", "segments": segs}]}))
    monkeypatch.setattr(engine, "game_seconds", lambda job_: 1800.0)
    with pytest.raises(engine.Failed) as e:
        engine.label(job, "moments", 0, [10.0], 0)
    assert e.value.code == "no_game"
    (job.rec / "detect").mkdir()
    with pytest.raises(engine.Failed) as e:
        engine.label(job, "ball", 4, [], 0)
    assert e.value.code == "no_candidates"
    (job.rec / "detect" / "ball_0_0_600.win.json").write_text("{}")
    used = []

    def fake_labelset(job_, name, args, **kw):
        used.append(args[3])
        sd = Path(args[args.index("--out") + 1])
        p = sd / "items.json"
        items = json.loads(p.read_text()) if p.exists() else []
        n = len(items)
        items += [{"id": f"b{n + i}", "image": f"images/b{n + i}.jpg", "segment": 0, "t": 10.0 * (n + i),
                   "width": 3840, "height": 2160, "candidates": []} for i in range(int(args[4]))]
        p.write_text(json.dumps(items))
    monkeypatch.setattr(engine, "step", fake_labelset)
    # The first try stopped before the set was written; by the next, chunk 1 is through too.
    sd = job.dir / "labels" / "ball-1"
    assert json.loads((sd / "request.json").read_text())["partial"] is True   # kept from the refused try
    (sd / "request.json").write_text(json.dumps({"kind": "ball", "count": 4, "at": [], "random": 0, "seed": 1,
                                                 "partial": True, "chunks": [0]}))
    (job.rec / "detect" / "ball_0_600_600.win.json").write_text("{}")
    assert engine.label(job, "ball", 4, [], 0) == engine.EXIT_OK
    assert set(used) == {"ball_0_0_600.win.json"}          # the chunks it was first made from
    s = json.loads((sd / "set.json").read_text())
    assert s["partial"] is True and len(s["items"]) == 4
    engine.validate(json.loads(capsys.readouterr().out.splitlines()[-1]), "engine-event.schema.json", "event")
    # A new set, asked for now, takes both chunks.
    used.clear()
    assert engine.label(job, "ball", 4, [], 0) == engine.EXIT_OK
    assert set(used) == {"ball_0_0_600.win.json", "ball_0_600_600.win.json"}
    assert json.loads((job.dir / "labels" / "ball-2" / "request.json").read_text())["chunks"] == [0, 1]
