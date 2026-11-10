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
