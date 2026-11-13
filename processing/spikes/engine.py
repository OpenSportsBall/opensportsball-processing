#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.10", "jsonschema>=4.20"]
# ///
"""osb-engine: one game from footage to game.mp4, driven through files, for
the Mac app, the cloud job and a terminal alike.

    uv run processing/spikes/engine.py prepare job.json [--answers answers.json [--no-footage]]
    uv run processing/spikes/engine.py run     job.json --answers answers.json
    uv run processing/spikes/engine.py status  job.json
    uv run processing/spikes/engine.py questions job.json
    uv run processing/spikes/engine.py answer  job.json pitch --approve
    uv run processing/spikes/engine.py label   job.json ball [--count 50]
    uv run processing/spikes/engine.py label   job.json moments [--at 61.5,1834] [--random 3]
    uv run processing/spikes/engine.py label   job.json --answers labels.json
    uv run processing/spikes/engine.py share   job.json [--kinds pitch,ball,moments]

The job (schemas/engine-job.schema.json) names a recording folder: the
merged manifest.json and every segment file at its objectKey, as
footagestore.py fetch or the app's download leaves it.

prepare: syncs the phones (phase0.py), fits the pitch (pitch.py: lines,
goals, review) and groups a sample of people by kit (labelset.py clip,
teams.py: the two sides, the keepers and the referee, which roles.py reads;
nobody is asked about them), then writes <recording>/engine/questions.json
(schemas/engine-questions.schema.json): the pitch to approve or correct.
When the phones' views can't be stitched
(rec5: too little overlap, one view soft), it first asks only which phone
filmed the left side (the panorama drawn both ways, jointfit.py choices: the
right one shows the pitch in one piece), then builds
rough cameras from the manifest's gravity sensors and the lens table
(jointfit.py rough) and carries on; the pitch must then be marked (tapped
points), and the taps fit the cameras and pitch together over a range of
starting angles (jointfit.py taps, search, install) before the pitch can
be approved. Given answers
(schemas/engine-answers.schema.json) it applies them: corrected pitch
points are refitted (pitch.py taps) and asked again. Run it until
questions.json has nothing pending. Applying answers reads the pictures
the first prepare left, not the video, so --no-footage does it without
the segment files on disk (the cloud job, cloud.py).

run: the whole game (fullgame.py: ball, roles and follow in chunks), once
every question is answered. Writes <recording>/fullgame/game.mp4. It first
checks there is room for what is still to be written (the pieces, their
join video.mp4 and game.mp4 at the output's bitrate: ~49 GB for 100
minutes at 1440p, rec7 took 46) and stops with not_enough_disk if not. Once game.mp4 is
written and as long as the game, it deletes the pieces, video.mp4 and
audio.m4a (job option keepIntermediates keeps them) and writes
fullgame/done.json, so a rerun reports the game done instead of
rendering it again.

With the job's output.youtubeCopy (and height 1080), run then also writes
fullgame/game-youtube.mp4: game.mp4 scaled up to 1440p, which YouTube
serves with VP9 and a much higher bitrate than a 1080p upload (FINDINGS.md;
no detail is added). game.mp4 stays the 1080p render, half the size, to
keep and to play. The copy takes ~26 min and ~23 GB for 100 minutes, is
counted in the disk check, and is made from game.mp4 alone: asking for it
after the game is done makes just the copy.

prepare and run check every segment file against the manifest's size and
SHA-256 (footage_corrupt: download it again); a file is hashed once, then
remembered by its size and modification time (engine/verified.json).

status: what's done and what's pending, and the command to run next, as
one JSON line.

questions, answer: the same questions for a script or a model with no app
to draw them (docs/ENGINE-AGENTS.md). Every question in questions.json
says in words what is asked and how to answer (ask, howToAnswer), each
pitch keypoint has a close-up picture. `questions` prints what is pending as one JSON document made to be
read (absolute image paths, no polylines); `answer` writes one question's
answer into <recording>/engine/answers.json from flags, checked against
what was asked (pitch points as a pixel of a close-up),
and prints the prepare command that applies it. It records share as
false unless --share is given: sharing is the user's choice. These two
print a JSON document, not progress events.

run says how the ball is going as it goes: a `ball` event after each chunk
the camera has followed (ballAimShare, ballAimByChunk), with ballLow from
the second chunk when the models don't know this ball (BALL_LOW_SHARE),
hours before the game is done. status gives the same, and each chunk's two
stages with their times and logs (chunkDetails).

label (for users who agree to share): makes a label set in
<recording>/engine/labels/<set>/set.json (schemas/engine-labelset.schema.json).
"ball": frames of Main picked across the game as labelset.py's ball mode
picks them, half where the detector found the ball, half where the
tracker bridged a gap it missed, each with suggestions. A ball set can be
made while the game is still running, from the chunks through the ball
stage by then (`partial` in the set); moments need the finished game. "moments": the
stitched panorama at each time the user pressed Missed play (--at, seconds
into game.mp4) plus a few random ones, with where the camera pointed, for
must-see boxes. A stopped set carries on; a finished one makes the next
(ball-2, ...). --answers (schemas/engine-label-answers.schema.json) stores
labels in the formats the benches read: labels.json (label/server.py's)
and boxes.json (playeval.py's).

share: writes <recording>/engine/share/<name>.json
(schemas/engine-contribution.schema.json) for each result the user agreed
to share: the pitch (its answer's share switch, on by
default) and every label set with labels. It lists files; it sends
nothing. The app uploads them.

Both commands print progress as JSON lines on stdout
(schemas/engine-event.schema.json) and keep each step's output in
<recording>/engine/logs/. Every step is skipped when its output exists,
and fullgame.py skips finished chunks, so after a crash, a reboot or a
stop (SIGINT / SIGTERM stop the step's whole process group) the same
command carries on where it was.

Exit codes: 0 done (prepare: questions written, maybe pending); 2 bad job,
answers or footage (missing, the wrong size or a different SHA-256); 3 run
with questions unanswered; 4 a step failed (see the error event's log) or
not enough disk; 130 stopped.

Platform choices (GPU, decoder, encoder, Core ML) are in accel.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accel  # noqa: E402  (python_cmd, ffprobe: the packaged engine's programs)
SCHEMAS = HERE.parents[1] / "schemas"
VERSION = 1
TEAMS_SET = "engine"          # labels_engine/ and teams/engine/: the kit sample this engine makes
TEAMS_K = 8                   # teams.py components: kits, keepers, referee, and room for the rest
CLOSEUP_DEG = (9.0, 6.0)      # a keypoint's close-up: this much panorama (yaw, pitch) around it
CLOSEUP_SCALE = 3             # close-up pixels per panorama pixel
QUESTIONS = ("cameras", "pitch")
HEARTBEAT_S = 60.0
# Failures a user can understand, found by the line a step ends with.
KNOWN_FAILURES = [
    ("calibration failed", "calibration_failed",
     "the two phones' views couldn't be matched to stitch them (too little overlap, or one view soft)"),
    ("footage doesn't overlap in time", "no_overlap",
     "the two phones' footage doesn't overlap in time: the second phone has nothing from while the main one filmed"),
]
RANDOM_MOMENT_GAP_S = 20.0    # random must-see moments: at least this far from the others
BALL_SET_COUNT = 50           # frames in a ball label set
# The camera aimed at the ball in under this share of the chunks after the
# first: the models don't know this ball. On rec7 the standard judge gave
# 31% for chunk 0 and 0-2% for every other chunk; rec7's own judge 28-64%.
# So one chunk can't tell, and the first is left out once there are two.
BALL_LOW_SHARE = 0.10
PIECE_BPS = 16e6 / 8          # follow.py's pieces: 16 Mbit/s (measured ~14.7 on rec7: 11 GB for 100 min)
GAME_BPS = {1080: PIECE_BPS, 1440: 33e6 / 8}  # game.mp4: fullgame asks 45 Mbit/s, hardware gives ~2/3 (rec7: ~31)
YOUTUBE_H = 1440              # the YouTube copy: the least height YouTube serves as VP9 at its higher bitrate
YOUTUBE_NAME = "game-youtube.mp4"
YOUTUBE_BITRATE = "45M"       # as fullgame.UPLOAD_BITRATE: hardware H.264 lands at ~2/3 of it
DISK_MARGIN = 3e9             # free space kept beyond the estimate (logs, caches, the OS)
LENGTH_TOLERANCE = 0.02       # game.mp4 within 2% of the rendered chunks before the intermediates go
HELD_OUT_SHARE = 0.25         # ball sets: the last quarter of every chunk is held out for testing (holdout.py)

EXIT_OK, EXIT_BAD_INPUT, EXIT_NEEDS_ANSWERS, EXIT_FAILED, EXIT_STOPPED = 0, 2, 3, 4, 130


class Stop(Exception):
    pass


class Failed(Exception):
    def __init__(self, code: str, message: str, log: Path | None = None, exit_code: int = EXIT_FAILED):
        super().__init__(message)
        self.code, self.message, self.log, self.exit_code = code, message, log, exit_code


def emit(event: str, **fields) -> None:
    """One progress event on stdout (engine-event.schema.json)."""
    print(json.dumps({"event": event, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}), flush=True)


def validate(obj, schema_name: str, what: str) -> None:
    import jsonschema

    schema = json.loads((SCHEMAS / schema_name).read_text())
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "(top level)"
        raise Failed(f"bad_{what}", f"{what} {where}: {e.message}", exit_code=EXIT_BAD_INPUT)


# ----------------------------------------------------------------- the job


class Job:
    def __init__(self, path: Path):
        try:
            j = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise Failed("bad_job", f"can't read the job {path}: {e}", exit_code=EXIT_BAD_INPUT)
        validate(j, "engine-job.schema.json", "job")
        rec = Path(j["recording"])
        self.rec = rec if rec.is_absolute() else (path.parent / rec).resolve()
        o = j.get("options", {})
        self.height = j.get("output", {}).get("height", 1440)
        # A 1440p game.mp4 is the YouTube file already: no second one then.
        self.youtube_copy = bool(j.get("output", {}).get("youtubeCopy", False)) and self.height < YOUTUBE_H
        self.chunk_s = float(o.get("chunkS", 600))
        self.max_chunks = int(o.get("maxChunks", 0))
        self.sample = o.get("sample")       # {"segment", "startS", "seconds"}: where pitch and kits are fitted
        self.keep_intermediates = bool(o.get("keepIntermediates", False))
        self.dir = self.rec / "engine"
        self.logs = self.dir / "logs"
        self.state_path = self.dir / "state.json"

    def manifest(self) -> dict:
        try:
            return json.loads((self.rec / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise Failed("footage_missing", f"no readable manifest.json in {self.rec}: {e}", exit_code=EXIT_BAD_INPUT)

    def state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"schemaVersion": VERSION, "pitch": {"round": 1, "status": "pending"}, "applied": []}

    def save_state(self, st: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(self.state_path)


# ------------------------------------------------------------- subprocesses

_current: subprocess.Popen | None = None
_stopping = threading.Event()


def _on_signal(signum, frame) -> None:
    _stopping.set()
    p = _current
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def step(job: Job, name: str, args: list[str], *, done: Path | None = None, uv_with: tuple[str, ...] = (),
         watch=None) -> None:
    """Runs one pipeline script under uv in its own process group, logging to
    engine/logs/<name>.log. Skipped (with an event) when `done` exists.
    `watch(proc)` runs while it does (progress), every few seconds."""
    global _current
    if done is not None and done.exists():
        emit("step", step=name, status="skipped")
        return
    if _stopping.is_set():
        raise Stop()
    job.logs.mkdir(parents=True, exist_ok=True)
    log = job.logs / f"{name}.log"
    py = accel.python_cmd()
    extra = [a for w in uv_with for a in ("--with", w)] if py[0] == "uv" else []  # a packaged Python has them all
    cmd = [*py, *extra, str(HERE / args[0]), *args[1:]]
    emit("step", step=name, status="started", log=str(log))
    t0 = time.time()
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        _current = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=job.rec.parent,
                                    start_new_session=True)
        while _current.poll() is None:
            if watch:
                watch(_current)
            time.sleep(2)
        rc = _current.returncode
        _current = None
    if _stopping.is_set():
        raise Stop()
    if rc != 0 or (done is not None and not done.exists()):
        tail = log.read_text(errors="replace")[-2000:]
        for marker, code, message in KNOWN_FAILURES:
            if marker in tail:
                raise Failed(code, message, log)
        raise Failed(f"{name}_failed", f"{name} failed (exit {rc})", log)
    emit("step", step=name, status="done", seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------ prepare


def check_footage(job: Job) -> dict:
    m = job.manifest()
    if m.get("schemaVersion") != 1:
        raise Failed("bad_manifest", f"manifest schemaVersion {m.get('schemaVersion')} is not one this engine knows",
                     exit_code=EXIT_BAD_INPUT)
    roles = {d.get("role") for d in m.get("devices", [])}
    if roles != {"main", "secondary"}:
        raise Failed("bad_manifest", f"needs a main and a secondary phone, found {sorted(r for r in roles if r)}",
                     exit_code=EXIT_BAD_INPUT)
    missing = []
    for d in m["devices"]:
        for s in d.get("segments", []):
            f = job.rec / s["objectKey"]
            if not f.exists() or ("bytes" in s and f.stat().st_size != s["bytes"]):
                missing.append(s["objectKey"])
    if missing:
        raise Failed("footage_missing", f"{len(missing)} segment file(s) missing or the wrong size, e.g. {missing[0]}",
                     exit_code=EXIT_BAD_INPUT)
    verify_footage(job, m)
    return m


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify_footage(job: Job, m: dict) -> None:
    """Every segment against the manifest's SHA-256, hashed once: a file whose size and
    modification time haven't changed since it matched isn't hashed again (engine/verified.json)."""
    from concurrent.futures import ThreadPoolExecutor

    path = job.dir / "verified.json"
    seen = json.loads(path.read_text()) if path.exists() else {}
    todo = []
    for d in m["devices"]:
        for sg in d.get("segments", []):
            if "sha256" not in sg:
                continue
            f = job.rec / sg["objectKey"]
            st = f.stat()
            key = {"bytes": st.st_size, "mtimeNs": st.st_mtime_ns, "sha256": sg["sha256"]}
            if seen.get(sg["objectKey"]) != key:
                todo.append((sg["objectKey"], f, key))
    if not todo:
        return
    emit("step", step="verify-footage", status="started", files=len(todo))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:  # hashlib releases the GIL on large blocks
        digests = list(pool.map(lambda t: sha256_file(t[1]), todo))
    bad = [k for (k, _, key), dg in zip(todo, digests) if dg != key["sha256"]]
    for (k, _, key), dg in zip(todo, digests):
        if dg == key["sha256"]:
            seen[k] = key
        else:
            seen.pop(k, None)
    job.dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, indent=1))
    if bad:
        raise Failed("footage_corrupt", f"{len(bad)} segment file(s) don't match the manifest's SHA-256 "
                     f"(a broken download): download them again, e.g. {bad[0]}", exit_code=EXIT_BAD_INPUT)
    emit("step", step="verify-footage", status="done", seconds=round(time.time() - t0, 1), files=len(todo))


def covered_segments(job: Job, m: dict) -> list[int] | None:
    """Main's segments the second phone has all of, from the sync's report
    (phase0.py's secondaryCoverage; phase0.covered_segments). None when the
    report doesn't say: every segment then."""
    report = job.rec / "phase0" / "report.json"
    spans = json.loads(report.read_text()).get("secondaryCoverage") if report.exists() else None
    if spans is None:
        return None
    out, g = [], 0.0
    for k, sg in enumerate(next(d for d in m["devices"] if d["role"] == "main")["segments"]):
        dur = sg["durationMs"] / 1000
        if any(a <= g + 2.0 and b >= g + dur - 2.0 for a, b in spans):
            out.append(k)
        g += dur
    return out


def sample_clip(job: Job, m: dict) -> tuple[int, float, float]:
    """Where the pitch and the kits are fitted: the job's choice, else a
    quarter of the way into the recording (the warm-up is over, the light
    hasn't changed much), 2 minutes from 1 minute into that segment. When
    the second phone stopped early (or started late), the nearest segment
    to that which has both phones: the pitch is fitted on both views."""
    if job.sample:
        return int(job.sample["segment"]), float(job.sample["startS"]), float(job.sample["seconds"])
    main = next(d for d in m["devices"] if d["role"] == "main")
    n = len(main["segments"])
    seg = min(n - 1, max(0, n // 4))
    covered = covered_segments(job, m)
    if covered:
        seg = min(covered, key=lambda k: abs(k - seg))
    dur = main["segments"][seg]["durationMs"] / 1000
    start = min(60.0, max(0.0, dur - 130.0))
    return seg, start, 120.0


def prepare(job: Job, answers: dict | None, footage: bool = True) -> int:
    if footage:
        m = check_footage(job)
    else:
        # Applying answers reads the pictures an earlier prepare left (the
        # background, the kit sample), not the video: the cloud job does it
        # without fetching the footage again. A step that does want the
        # video fails as it would with a file missing.
        if not answers or not (job.dir / "questions.json").exists():
            raise Failed("bad_job", "--no-footage is for applying answers after a prepare", exit_code=EXIT_BAD_INPUT)
        m = job.manifest()
    st = job.state()
    rec = str(job.rec)
    if answers:
        validate(answers, "engine-answers.schema.json", "answers")
    step(job, "sync", ["phase0.py", rec], done=job.rec / "phase0" / "report.json")
    seg, start, secs = sample_clip(job, m)  # after the sync: it says where both phones have footage
    pitch_dir = job.rec / "pitch"
    if answers and answers.get("cameras"):
        apply_cameras(job, st, answers["cameras"])
        job.save_state(st)
    if (st.get("cameras") or {}).get("status") == "pending":
        return ask_cameras(job, st, seg)
    try:
        step(job, "pitch", ["pitch.py", rec, str(seg)], done=pitch_dir / "pitch.json")
    except Failed as f:
        if f.code != "calibration_failed" or st.get("cameras"):
            raise
        # No stitch: ask which phone is on the left, then build the cameras from the pitch.
        st["cameras"] = {"round": 1, "status": "pending", "reason": f.message}
        return ask_cameras(job, st, seg)
    step(job, "goals", ["pitch.py", rec, "goals"], done=pitch_dir / "goals.json")
    step(job, "kit-sample", ["labelset.py", rec, "clip", TEAMS_SET, str(seg), f"{start:g}", f"{secs:g}", "2"],
         done=job.rec / f"labels_{TEAMS_SET}" / "items.json")
    teams_dir = job.rec / "teams" / TEAMS_SET
    # sides.json is the last thing teams.py writes that roles.py reads. A folder set up by an engine
    # that still asked about the kit groups has people.json without it, and every ball chunk then fails.
    step(job, "kits", ["teams.py", rec, TEAMS_SET, str(TEAMS_K)], done=teams_dir / "sides.json")

    if answers:
        apply_answers(job, st, answers, seg)
        job.save_state(st)
    # The pitch question shows the current fit: review.json follows pitch.json.
    review = pitch_dir / "review.json"
    if not review.exists() or review.stat().st_mtime < (pitch_dir / "pitch.json").stat().st_mtime:
        if review.exists():
            review.unlink()
        step(job, "pitch-review", ["pitch.py", rec, "review"], done=review)
    q = write_questions(job, st)
    job.save_state(st)
    emit("questions", path=str(job.dir / "questions.json"), pending=q["pending"])
    return EXIT_OK


def ask_cameras(job: Job, st: dict, seg: int) -> int:
    """The stitch failed: the panorama with each phone on the left, and the question, alone."""
    step(job, "camera-choices", ["jointfit.py", str(job.rec), "choices", str(job.dir / "cameras"), "--seg", str(seg)],
         done=job.dir / "cameras" / "secondary_left.jpg")
    q = write_questions(job, st)
    job.save_state(st)
    emit("questions", path=str(job.dir / "questions.json"), pending=q["pending"])
    return EXIT_OK


def apply_cameras(job: Job, st: dict, a: dict) -> None:
    c = st.get("cameras")
    if not c or c["status"] != "pending" or a["round"] != c["round"]:
        return
    step(job, "rough-cameras", ["jointfit.py", str(job.rec), "rough", "--left", a["leftPhone"]])
    c.update(status="answered", leftPhone=a["leftPhone"], fitted=False)
    st["pitch"]["approveAllowed"] = False  # the first guess is drawn through rough cameras: taps first
    st["applied"].append({"question": "cameras", "answer": a, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    emit("answer", question="cameras", result="rough cameras", leftPhone=a["leftPhone"])


def current_answers(answers: dict) -> dict:
    """Answers without the kit-group check, which is no longer asked: an
    answers file written before that may still carry one."""
    return {k: v for k, v in answers.items() if k != "kits"}


def pending_questions(st: dict | None) -> list[str]:
    if not st:
        return ["pitch"]
    if (st.get("cameras") or {}).get("status") == "pending":
        return ["cameras"]
    return ["pitch"] if st["pitch"]["status"] == "pending" else []


def apply_answers(job: Job, st: dict, answers: dict, seg: int) -> None:
    validate(answers, "engine-answers.schema.json", "answers")
    rec = str(job.rec)
    a = answers.get("pitch")
    if a and a["round"] == st["pitch"]["round"] and st["pitch"]["status"] == "pending":
        st["pitch"].pop("problem", None)
        if a.get("approve") and st["pitch"].get("approveAllowed") is False:
            st["pitch"]["problem"] = ("mark the pitch first: the phones couldn't be stitched, so the lines are "
                                      "drawn through rough cameras until the marked points fit them")
            emit("answer", question="pitch", result="rejected", round=st["pitch"]["round"])
        elif a.get("approve"):
            st["pitch"]["status"] = "approved"
            st["pitch"]["approvedRound"] = a["round"]
            emit("answer", question="pitch", result="approved")
        elif a.get("points"):
            review = json.loads((job.rec / "pitch" / "review.json").read_text())
            taps = {}
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            for name, p in a["points"].items():
                if p.get("skipped"):
                    taps[name] = {"skipped": True, "at": now}
                elif p.get("fromFit"):
                    fit = review["points"].get(name)
                    if fit:
                        taps[name] = {**fit, "fromFit": True, "at": now}
                else:
                    taps[name] = {"yaw": p["yaw"], "pitch": p["pitch"], "at": now}
            (job.rec / "pitch" / "taps.json").write_text(json.dumps(taps, indent=1))
            cams = st.get("cameras")
            if cams and cams.get("status") == "answered":
                # Cameras from the pitch: the taps (in each phone's pixels) fit both lenses and the
                # pitch together; from rough cameras over a range of starting angles.
                r, jf = a["round"], "jointfit.py"
                step(job, f"pitch-taps-{r}", [jf, rec, "taps"])
                if cams.get("fitted"):
                    step(job, f"cameras-fit-{r}", [jf, rec, "fit", "--seg", str(seg)])
                else:
                    step(job, f"cameras-search-{r}", [jf, rec, "search", "--seg", str(seg)])
                step(job, f"cameras-install-{r}", [jf, rec, "install", str(job.rec / "joint"), "--seg", str(seg)])
                step(job, f"goals-{r}", ["pitch.py", rec, "goals"], done=job.rec / "pitch" / "goals.json")
                cams["fitted"] = True
                st["pitch"]["approveAllowed"] = True
            else:
                step(job, f"pitch-refit-{a['round']}", ["pitch.py", rec, "taps"])
                # The goals were found on the pitch before this refit, in its metres: find them again
                # on the new goal lines (a tapped post fixes its goal).
                (job.rec / "pitch" / "goals.json").unlink(missing_ok=True)
                step(job, f"goals-{a['round']}", ["pitch.py", rec, "goals"], done=job.rec / "pitch" / "goals.json")
            st["pitch"]["round"] += 1
            emit("answer", question="pitch", result="refitted", round=st["pitch"]["round"])
        st["applied"].append({"question": "pitch", "answer": a, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def cameras_question(job: Job, st: dict) -> dict | None:
    c = st.get("cameras")
    if not c:
        return None
    q = {"round": c["round"], "status": c["status"],
         "panoramas": {"mainLeft": "engine/cameras/main_left.jpg",
                       "secondaryLeft": "engine/cameras/secondary_left.jpg"}}
    if c.get("leftPhone"):
        q["leftPhone"] = c["leftPhone"]
    if c.get("reason"):
        q["reason"] = c["reason"]
    q.update(ASK["cameras"])
    return q


def save_questions(job: Job, q: dict) -> dict:
    validate(q, "engine-questions.schema.json", "questions")
    job.dir.mkdir(parents=True, exist_ok=True)
    tmp = job.dir / "questions.tmp"
    tmp.write_text(json.dumps(q, indent=1))
    tmp.replace(job.dir / "questions.json")
    return q


def write_questions(job: Job, st: dict) -> dict:
    rec = job.rec
    cams_q = cameras_question(job, st)
    if cams_q and cams_q["status"] == "pending":
        return save_questions(job, {"schemaVersion": VERSION, "recording": str(rec), "pending": ["cameras"],
                                    "cameras": cams_q})
    review = json.loads((rec / "pitch" / "review.json").read_text())
    pitch_q = {"round": st["pitch"]["round"], "status": st["pitch"]["status"],
               "background": "pitch/" + review.get("panorama", {}).get("image", "background.png"),
               "overlay": "pitch/overlay_taps.jpg" if review["pitch"].get("fittedWith", "").endswith("taps")
               else "pitch/overlay.jpg",
               "panorama": review.get("panorama"), "fit": review["pitch"], "lines": review["lines"],
               # only goals that were found: a rejected one is a guess the reader would take for a goal
               "goals": [g for g in review["goals"] if g.get("valid", True)], "points": review["points"]}
    paint = rec / "pitch" / "paint.json"
    if paint.exists():
        pitch_q["lineColour"] = json.loads(paint.read_text())
    if st["pitch"].get("approveAllowed") is False:
        pitch_q["approveAllowed"] = False
    if st["pitch"].get("problem"):
        pitch_q["problem"] = st["pitch"]["problem"]
    closeups = point_closeups(job, st["pitch"]["round"], review)
    pitch_q["points"] = {n: {**p, **({"closeup": closeups[n]} if n in closeups else {})}
                         for n, p in review["points"].items()}
    pitch_q.update(ask_pitch(pitch_q.get("approveAllowed", True)))
    guess = job.dir / "pitch_guess.json"
    if st["pitch"]["round"] == 1 and not guess.exists():
        # Our first guess, before any correction: shared next to the final fit.
        job.dir.mkdir(parents=True, exist_ok=True)
        guess.write_text((rec / "pitch" / "pitch.json").read_text())
    q = {"schemaVersion": VERSION, "recording": str(rec), "pending": pending_questions(st), "pitch": pitch_q}
    if cams_q:
        q["cameras"] = cams_q
    return save_questions(job, q)


# ------------------------------------------- questions a model can answer
#
# The app draws the questions itself. A script or a model driving the engine
# has only the files, so each question also says in words what is asked and
# how to answer it (ask, howToAnswer), and comes with pictures made to be
# looked at one at a time: a close-up of every pitch keypoint.

SHARE_NOTE = ("Sharing is the user's choice: `answer` records share as false unless --share is given. "
              "Pass --share only when the user has agreed to share this result.")
ASK = {"cameras": {
    "ask": ("The two phones' views couldn't be stitched. panoramas.mainLeft and panoramas.secondaryLeft show the "
            "pitch drawn with Main on the left and with Secondary on the left. Which one shows the pitch in one "
            "piece? The wrong one shows two broken halves: lines meeting in a V, the halfway line at both edges."),
    "howToAnswer": ["Main on the left: osb-engine answer <job> cameras --left main",
                    "Secondary on the left: osb-engine answer <job> cameras --left secondary",
                    "Then apply it: osb-engine prepare <job> --answers <recording>/engine/answers.json. "
                    "The pitch question follows, and its points must be marked before it can be approved."]}}


def ask_pitch(approve_allowed: bool) -> dict:
    how = []
    if approve_allowed:
        how.append("The lines sit on the paint everywhere: osb-engine answer <job> pitch --approve")
    else:
        how.append("Approving isn't allowed yet (approveAllowed is false): mark the points first.")
    how += [
        "They don't: say where keypoints really are, at least four spread over both ends of the pitch. "
        "osb-engine answer <job> pitch --closeup 'NAME=X,Y' ... with X,Y the pixel in that point's close-up image "
        "(points.NAME.closeup.image) where the keypoint really is; or --pano 'NAME=X,Y' in the background "
        "image's pixels; or --point 'NAME=YAW,PITCH' in panorama degrees.",
        "--keep NAME for a point the fit already has right (it then holds the fit there); "
        "--skip NAME for one that can't be seen. Points not named are left out of the refit.",
        "Then apply it: osb-engine prepare <job> --answers <recording>/engine/answers.json. The engine refits "
        "and asks again with the next round; approve once the lines sit on the paint.",
        SHARE_NOTE]
    return {"ask": ("overlay is the empty pitch (background) with the fitted pitch drawn on it in red. Do the red "
                    "lines sit on the painted lines everywhere: touchlines, goal lines, penalty and goal areas, "
                    "halfway line, centre circle? Each point's close-up shows one named keypoint enlarged, with "
                    "the fitted lines in red and a ring where the fit puts the keypoint (closeup.fit). "
                    "A wrong pitch spoils everything after it, so look at both ends and the far side."),
            "howToAnswer": how}


def point_closeups(job: Job, rnd: int, review: dict) -> dict:
    """A close-up of the background around each keypoint, the fitted lines in
    red and a ring where the fit puts the point: engine/pitch_points/round<n>/.
    Per point: the image, where the fit is in it, and its pixels in panorama
    degrees (yaw = yaw0 + x * degPerPx, pitch = pitch0 + y * degPerPx), so a
    correction can be given as a pixel of the close-up. A point the fit puts
    off the panorama has none."""
    import cv2
    import numpy as np

    pano = review.get("panorama")
    if not pano or not review.get("points"):
        return {}
    bg = cv2.imread(str(job.rec / "pitch" / pano["image"]))
    if bg is None:
        return {}
    dpp, yaw0, pitch0 = pano["degPerPx"], pano["yawRange"][0], pano["pitchRange"][0]
    H, W = bg.shape[:2]
    w, h = min(W, round(CLOSEUP_DEG[0] / dpp)), min(H, round(CLOSEUP_DEG[1] / dpp))
    s = CLOSEUP_SCALE
    lines = []
    for ln in review.get("lines", []):
        a = np.asarray(ln, float).reshape(-1, 2)
        a = a[np.isfinite(a).all(1)]
        if len(a) >= 2:
            lines.append(np.c_[(a[:, 0] - yaw0) / dpp, (a[:, 1] - pitch0) / dpp])
    out_dir = job.dir / "pitch_points" / f"round{rnd}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, p in review["points"].items():
        cx, cy = (p["yaw"] - yaw0) / dpp, (p["pitch"] - pitch0) / dpp
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        x1 = int(min(max(round(cx - w / 2), 0), W - w))
        y1 = int(min(max(round(cy - h / 2), 0), H - h))
        cut = cv2.resize(bg[y1:y1 + h, x1:x1 + w], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for ln in lines:
            near = ((ln[:, 0] > x1 - w) & (ln[:, 0] < x1 + 2 * w) & (ln[:, 1] > y1 - h) & (ln[:, 1] < y1 + 2 * h))
            if near.any():
                pts = np.round((ln - [x1, y1]) * s).astype(np.int32)
                cv2.polylines(cut, [pts], False, (0, 0, 255), 1, cv2.LINE_AA)
        fx, fy = round((cx - x1) * s), round((cy - y1) * s)
        cv2.circle(cut, (fx, fy), 16, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(cut, (fx, fy), 16, (0, 255, 255), 1, cv2.LINE_AA)
        path = out_dir / (name.replace(" ", "_").replace("/", "_") + ".jpg")
        cv2.imwrite(str(path), cut, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out[name] = {"image": str(path.relative_to(job.rec)), "width": cut.shape[1], "height": cut.shape[0],
                     "fit": [fx, fy], "degPerPx": round(dpp / s, 6), "yaw0": round(yaw0 + x1 * dpp, 4),
                     "pitch0": round(pitch0 + y1 * dpp, 4)}
    return out


def load_questions(job: Job) -> dict:
    try:
        return json.loads((job.dir / "questions.json").read_text())
    except (OSError, json.JSONDecodeError):
        raise Failed("not_prepared", "no questions yet: run prepare first", exit_code=EXIT_BAD_INPUT)


def next_step(job: Job, st: dict | None) -> str:
    """What to do next, for status and the briefs."""
    job_arg = "<job>"
    if (job.rec / "fullgame" / "game.mp4").exists() and (job.rec / "fullgame" / "done.json").exists():
        if job.youtube_copy and youtube_made(job) is None:
            return f"osb-engine run {job_arg} (the game video is written; this makes its YouTube copy, about half an hour)"
        return "done: the game video is written"
    if not (job.dir / "questions.json").exists():
        return f"osb-engine prepare {job_arg}"
    pending = pending_questions(st)
    if pending:
        return (f"osb-engine questions {job_arg}, then osb-engine answer {job_arg} {pending[0]} ..., then "
                f"osb-engine prepare {job_arg} --answers {job.dir / 'answers.json'}")
    return f"osb-engine run {job_arg} (hours: run it in the background and read its progress lines)"


def questions_brief(job: Job) -> int:
    """`questions`: what is waiting for an answer, as one JSON document on
    stdout made to be read: the words, the pictures (absolute paths) and the
    numbers an answer needs, without the polylines the app draws from."""
    q = load_questions(job)
    rec = job.rec
    full = lambda p: str(rec / p) if p else None  # noqa: E731
    out = {"recording": str(rec), "pending": q["pending"], "questions": []}
    for name in q["pending"]:
        d = q[name]
        b = {"question": name, "round": d["round"], "ask": d.get("ask"), "howToAnswer": d.get("howToAnswer")}
        if d.get("problem"):
            b["problem"] = d["problem"]
        if name == "cameras":
            b["panoramas"] = {k: full(v) for k, v in d["panoramas"].items()}
            if d.get("reason"):
                b["reason"] = d["reason"]
        elif name == "pitch":
            b.update(overlay=full(d["overlay"]), background=full(d["background"]), fit=d["fit"],
                     approveAllowed=d.get("approveAllowed", True), panorama=d.get("panorama"))
            if d.get("lineColour"):
                b["lineColour"] = d["lineColour"]
            b["points"] = {n: {"yaw": p["yaw"], "pitch": p["pitch"],
                               **({"closeup": {**p["closeup"], "image": full(p["closeup"]["image"])}}
                                  if p.get("closeup") else {"closeup": None})}
                           for n, p in d["points"].items()}
        out["questions"].append(b)
    out["next"] = next_step(job, job.state() if job.state_path.exists() else None)
    print(json.dumps(out, indent=1), flush=True)
    return EXIT_OK


def _pairs(values: list[str], flag: str) -> dict[str, tuple[float, float]]:
    """--closeup / --pano / --point values: 'NAME=A,B'."""
    out = {}
    for v in values:
        name, _, rest = v.rpartition("=")
        try:
            a, b = (float(x) for x in rest.split(","))
        except ValueError:
            raise Failed("bad_answers", f"{flag} takes 'NAME=A,B', not {v!r}", exit_code=EXIT_BAD_INPUT)
        if not name:
            raise Failed("bad_answers", f"{flag} takes 'NAME=A,B', not {v!r}", exit_code=EXIT_BAD_INPUT)
        out[name.strip()] = (a, b)
    return out


def build_answer(q: dict, question: str, a) -> dict:
    """One question's answer (engine-answers.schema.json) from `answer`'s
    flags, checked against what was asked."""
    def bad(message: str):
        return Failed("bad_answers", message, exit_code=EXIT_BAD_INPUT)

    if question not in QUESTIONS:
        raise bad(f"answer takes one of {', '.join(QUESTIONS)}, not {question!r}")
    if question not in q["pending"]:
        raise bad(f"{question} isn't waiting for an answer (pending: {', '.join(q['pending']) or 'nothing'})")
    d = q[question]
    ans: dict = {"round": d["round"]}
    if question == "cameras":
        if a.left not in ("main", "secondary"):
            raise bad("cameras needs --left main or --left secondary")
        ans["leftPhone"] = a.left
        return ans
    ans["share"] = bool(a.share)
    if question == "pitch":
        at, pano_px, deg = _pairs(a.closeup, "--closeup"), _pairs(a.pano, "--pano"), _pairs(a.point, "--point")
        named = [*at, *pano_px, *deg, *a.keep, *a.skip]
        if a.approve:
            if named:
                raise bad("--approve stands alone: approve the pitch as drawn, or give points to refit it")
            if d.get("approveAllowed") is False:
                raise bad("the pitch can't be approved yet: mark its points first (approveAllowed is false)")
            ans["approve"] = True
            return ans
        if not named:
            raise bad("pitch needs --approve, or points (--closeup, --pano, --point, --keep, --skip)")
        twice = sorted({n for n in named if named.count(n) > 1})
        unknown = sorted(set(named) - set(d["points"]))
        if twice:
            raise bad(f"given more than once: {', '.join(twice)}")
        if unknown:
            raise bad(f"no such point: {', '.join(unknown)} (the points are: {', '.join(d['points'])})")
        points = {}
        for n, (x, y) in at.items():
            c = d["points"][n].get("closeup")
            if not c:
                raise bad(f"{n} has no close-up (the fit puts it off the panorama): use --pano or --point")
            if not (0 <= x < c["width"] and 0 <= y < c["height"]):
                raise bad(f"{n}: {x:g},{y:g} is outside its close-up ({c['width']} x {c['height']})")
            points[n] = {"yaw": round(c["yaw0"] + x * c["degPerPx"], 3), "pitch": round(c["pitch0"] + y * c["degPerPx"], 3)}
        if pano_px:
            p = d.get("panorama")
            if not p:
                raise bad("this question has no panorama mapping: use --closeup or --point")
            for n, (x, y) in pano_px.items():
                points[n] = {"yaw": round(p["yawRange"][0] + x * p["degPerPx"], 3),
                             "pitch": round(p["pitchRange"][0] + y * p["degPerPx"], 3)}
        for n, (yaw, pitch) in deg.items():
            points[n] = {"yaw": yaw, "pitch": pitch}
        for n in a.keep:
            points[n] = {"fromFit": True}
        for n in a.skip:
            points[n] = {"skipped": True}
        ans["points"] = points
        return ans


def answer(job: Job, question: str | None, a) -> int:
    """`answer`: writes one question's answer into <recording>/engine/answers.json
    (keeping the other questions' answers there) and says what to run next.
    It changes nothing else: prepare --answers applies it."""
    q = load_questions(job)
    ans = build_answer(q, question or "", a)
    path = job.dir / "answers.json"
    try:
        answers = current_answers(json.loads(path.read_text())) if path.exists() else {}
    except json.JSONDecodeError:
        answers = {}
    answers[question] = ans
    validate(answers, "engine-answers.schema.json", "answers")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(answers, indent=1))
    tmp.replace(path)
    print(json.dumps({"written": str(path), "question": question, "answer": ans,
                      "next": f"osb-engine prepare <job> --answers {path}"}, indent=1), flush=True)
    return EXIT_OK


# --------------------------------------------------------------------- run


def run(job: Job, answers: dict | None) -> int:
    check_footage(job)
    st = job.state()
    if answers:
        # Answers given to run are applied as prepare would; anything that
        # needs asking again leaves the run waiting.
        prepare(job, answers)
        st = job.state()
    pending = pending_questions(st)
    if not (job.dir / "questions.json").exists() or pending:
        emit("needs_answers", pending=pending or ["pitch"], questions=str(job.dir / "questions.json"))
        return EXIT_NEEDS_ANSWERS
    out = job.rec / "fullgame"
    done_mark = out / "done.json"
    if done_mark.exists() and (out / "game.mp4").exists():
        d = json.loads(done_mark.read_text())
        emit("step", step="game", status="skipped")
        yt = youtube_copy(job, out / "game.mp4")
        emit("done", output=str(out / "game.mp4"), **yt, seconds=d.get("seconds", 0), **ball_share(job))
        return EXIT_OK
    check_disk(job)
    events = job.dir / "events.jsonl"
    seen = [0]
    prog = {"chunks": 0, "ball": set(), "follow": set(), "start": time.time(), "doneBefore": 0, "last": 0.0,
            "fail": None, "result": None}
    if events.exists():
        seen[0] = events.stat().st_size  # earlier runs' events are history

    def watch(proc):
        if not events.exists():
            return
        with events.open() as f:
            f.seek(seen[0])
            for line in f:
                if not line.endswith("\n"):
                    break
                seen[0] += len(line.encode())
                e = json.loads(line)
                if e["event"] == "plan":
                    prog["chunks"] = e["chunks"]
                    prog["ball"] = set(e["done"])
                    prog["follow"] = set(e["done"])
                    prog["doneBefore"] = 2 * len(e["done"])
                    report(force=True)
                    report_ball()
                elif e["event"] == "chunk":
                    if e["ok"]:
                        prog[e["stage"]].add(e["chunk"])
                    else:
                        prog["fail"] = prog["fail"] or e  # the first; the rest follow from it
                    report(force=True)
                    if e["ok"] and e["stage"] == "follow":
                        report_ball()
                elif e["event"] == "mux":
                    emit("step", step="mux", status="started")
                elif e["event"] in ("done", "failed"):
                    prog["result"] = e
        report()

    def report_ball():
        """After each chunk the camera has followed: how much of the game so
        far it aimed at the ball, and from the second chunk whether that is
        low, hours before the game is done."""
        b = ball_share(job, final=False)
        if b:
            emit("ball", chunksDone=len(b["ballAimByChunk"]), chunks=prog["chunks"], **b)

    def report(force=False):
        n = prog["chunks"]
        if not n or (not force and time.time() - prog["last"] < HEARTBEAT_S):
            return
        prog["last"] = time.time()
        units = len(prog["ball"]) + len(prog["follow"])
        frac = units / (2 * n)
        this_run = units - prog["doneBefore"]
        el = time.time() - prog["start"]
        eta = round(el / this_run * (2 * n - units)) if this_run > 0 else None
        emit("progress", stage="game", fraction=round(frac, 4), chunks=n, ballDone=len(prog["ball"]),
             followDone=len(prog["follow"]), elapsedS=round(el), etaS=eta)

    args = ["fullgame.py", str(job.rec), "--teams", TEAMS_SET, "--events", str(events),
            "--chunk", f"{job.chunk_s:g}", "--height", str(job.height)]
    if job.max_chunks:
        args += ["--max-chunks", str(job.max_chunks)]
    game = job.rec / "fullgame" / "game.mp4"
    try:
        step(job, "game", args, watch=watch)
    except Failed as f:
        e = prog["fail"]
        if e and e.get("log"):
            f.log = Path(e["log"])
            f.message = f"{e['stage']} stage failed on chunk {e['chunk']}"
        raise
    watch(None)
    secs = round(time.time() - prog["start"], 1)
    finish_output(job, game, secs)
    yt = youtube_copy(job, game)
    emit("done", output=str(game), **yt, seconds=round(time.time() - prog["start"], 1), **ball_share(job))
    return EXIT_OK


def planned_chunks(job: Job) -> list[float]:
    """Each chunk's length in seconds, as fullgame.py cuts the game (the last stops 1 s short)."""
    total = game_seconds(job)
    out, t = [], 0.0
    while t < total - 1:
        out.append(min(job.chunk_s, total - t - 1.0))
        t += job.chunk_s
    return out[:job.max_chunks] if job.max_chunks else out


def disk_needed(job: Job) -> float:
    """Bytes run still has to write: the pieces not made yet, their join (video.mp4) and game.mp4,
    all on disk together at the end, and the YouTube copy if the job asks for one. For a finished
    game, only a copy that is still to be made."""
    chunks = planned_chunks(job)
    copy = sum(chunks) * GAME_BPS[YOUTUBE_H] if job.youtube_copy and youtube_made(job) is None else 0.0
    if (job.rec / "fullgame" / "done.json").exists() and (job.rec / "fullgame" / "game.mp4").exists():
        return copy
    pieces = job.rec / "fullgame" / "pieces"
    todo = sum(sec for k, sec in enumerate(chunks) if not (pieces / f"piece_{k:02d}.mp4").exists())
    total = sum(chunks)
    game_bps = GAME_BPS.get(job.height, GAME_BPS[1440])
    return todo * PIECE_BPS + total * PIECE_BPS + total * game_bps + copy


def check_disk(job: Job) -> None:
    import shutil

    need = disk_needed(job) + DISK_MARGIN
    free = shutil.disk_usage(job.rec).free
    if free < need:
        raise Failed("not_enough_disk", f"this game needs about {need / 1e9:.0f} GB free on the disk holding the "
                     f"recording and {free / 1e9:.0f} GB is: free {(need - free) / 1e9:.0f} GB more, then run again")


def video_seconds(path: Path) -> float | None:
    try:
        r = subprocess.run([accel.ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def video_height(path: Path) -> int | None:
    try:
        r = subprocess.run([accel.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip().split(",")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def finish_output(job: Job, game: Path, secs: float) -> None:
    """game.mp4 checked against the chunks' length; then the intermediates go (unless kept) and
    done.json marks the game finished."""
    out = game.parent
    want = sum(planned_chunks(job))
    got = video_seconds(game)
    if got is None or abs(got - want) > LENGTH_TOLERANCE * want:
        raise Failed("output_check_failed", f"game.mp4 is {got if got is None else round(got)} s long, the chunks "
                     f"{want:.0f} s: kept the pieces and video.mp4 to look into it", log=job.logs / "game.log")
    freed = 0
    if not job.keep_intermediates:
        for p in [*sorted((out / "pieces").glob("piece_*.mp4")), out / "video.mp4", out / "audio.m4a"]:
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
        if (out / "pieces").exists() and not any((out / "pieces").iterdir()):
            (out / "pieces").rmdir()
    (out / "done.json").write_text(json.dumps({"output": "game.mp4", "bytes": game.stat().st_size,
                                               "videoS": round(got, 2), "seconds": secs,
                                               "intermediatesKept": job.keep_intermediates,
                                               "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    emit("step", step="cleanup", status="done", freedBytes=freed, kept=job.keep_intermediates)


def ball_by_chunk(job: Job) -> dict[int, dict]:
    """Per chunk the camera has followed, from follow.py's summary line
    (fullgame/logs/follow_NN.log): `aimPct`, the percent of the chunk the
    camera aimed at the ball, and `detectedPct`, the percent of frames with
    a ball the judge accepted."""
    import re

    out = {}
    for p in sorted((job.rec / "fullgame" / "logs").glob("follow_*.log")):
        text = p.read_text(errors="replace")
        m = re.findall(r"aim at the ball in (\d+)% of the clip", text)
        if not m:
            continue
        d = {"aimPct": int(m[-1])}
        n = re.findall(r"detected in (\d+)/(\d+) frames", text)
        if n and int(n[-1][1]):
            d["detectedPct"] = round(100 * int(n[-1][0]) / int(n[-1][1]))
        out[int(p.stem.split("_")[1])] = d
    return out


def ball_share(job: Job, final: bool = True) -> dict:
    """How much of the game the camera aimed at the ball (per chunk and
    over all of them), and `ballLow` when that says the models don't know
    this ball (BALL_LOW_SHARE): the app then offers ball labelling. Before
    the game is done (`final` false) one chunk isn't enough to say."""
    by = ball_by_chunk(job)
    if not by:
        return {}
    shares = [by[k]["aimPct"] for k in sorted(by)]
    out = {"ballAimShare": round(sum(shares) / len(shares) / 100, 3), "ballAimByChunk": shares}
    judged = shares[1:] if len(shares) >= 2 else (shares if final else [])
    if judged:
        out["ballLow"] = sum(judged) / len(judged) / 100 < BALL_LOW_SHARE
    return out


def chunk_details(job: Job) -> list[dict]:
    """Each chunk of the game for `status`: when it starts in the game, how
    long it is, where its two stages stand (ball: candidates and roles;
    follow: the camera and the render), how long each took, their logs, and
    what the camera made of the ball. From fullgame.py's events, so only
    for a game this engine has started."""
    events = job.dir / "events.jsonl"
    if not events.exists():
        return []
    plan, stages = None, {}
    for line in events.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue  # a line still being written
        if e.get("event") == "plan":
            plan = e
            # A new run: what an earlier one left running was stopped.
            stages = {k: v for k, v in stages.items() if v["status"] != "running"}
        elif e.get("event") == "start":
            stages[(e["chunk"], e["stage"])] = {"status": "running", "startedAt": e["at"]}
        elif e.get("event") == "chunk":
            s = {"status": "done" if e["ok"] else "failed", "seconds": e.get("seconds")}
            stages[(e["chunk"], e["stage"])] = s
    if plan is None:
        return []
    logs = job.rec / "fullgame" / "logs"
    ball = ball_by_chunk(job)
    out = []
    for k in range(plan["chunks"]):
        start = k * plan["chunkS"]
        c = {"chunk": k, "startS": round(start, 1), "seconds": round(min(plan["chunkS"], plan["gameS"] - start), 1)}
        rendered = (job.rec / "fullgame" / "pieces" / f"piece_{k:02d}.mp4").exists() or k in plan.get("done", [])
        for stage in ("ball", "follow"):
            s = dict(stages.get((k, stage)) or {"status": "done" if rendered else "waiting"})
            log = logs / f"{stage}_{k:02d}.log"
            if log.exists():
                s["log"] = str(log)
            c[stage] = {a: b for a, b in s.items() if b is not None}
        if k in ball:
            c["ballAimPct"] = ball[k]["aimPct"]
            if "detectedPct" in ball[k]:
                c["ballDetectedPct"] = ball[k]["detectedPct"]
        out.append(c)
    return out


# ------------------------------------------------------------------ status


# ------------------------------------------------------------------- label


def game_seconds(job: Job) -> float:
    """Length of the game video, from fullgame.py's plan (else the manifest)."""
    events = job.dir / "events.jsonl"
    if events.exists():
        for line in reversed(events.read_text().splitlines()):
            e = json.loads(line)
            if e.get("event") == "plan":
                return float(e["gameS"])
    main = next(d for d in job.manifest()["devices"] if d["role"] == "main")
    return sum(s["durationMs"] for s in main["segments"]) / 1000


def video_t(job: Job, seg: int, t: float) -> float:
    """Seconds into game.mp4 of `t` s into Main's segment `seg` (the engine
    renders from the recording's start)."""
    main = next(d for d in job.manifest()["devices"] if d["role"] == "main")
    return round(sum(s["durationMs"] for s in main["segments"][:seg]) / 1000 + t, 3)


def label_dir(job: Job, kind: str) -> tuple[str, Path]:
    """The set to work on: an unfinished one of this kind, else the next."""
    root = job.dir / "labels"
    n = 1
    while (root / f"{kind}-{n}").exists():
        if not (root / f"{kind}-{n}" / "set.json").exists():
            return f"{kind}-{n}", root / f"{kind}-{n}"
        n += 1
    return f"{kind}-{n}", root / f"{kind}-{n}"


def camera_at(job: Job, t: float) -> dict | None:
    """Where the follow cam pointed at `t` s into game.mp4, from the chunks'
    camera paths (fullgame/logs/path_NN.json)."""
    for p in sorted((job.rec / "fullgame" / "logs").glob("path_*.json")):
        j = json.loads(p.read_text())
        i = int(round((t - j["startS"]) * j["fps"]))
        if 0 <= i < len(j["yawDeg"]):
            return {"yawDeg": j["yawDeg"][i], "pitchDeg": j["pitchDeg"][i], "hfovDeg": j["hfovDeg"][i]}
    return None


def label(job: Job, kind: str | None, count: int, at: list[float], n_random: int) -> int:
    if kind not in ("ball", "moments"):
        raise Failed("bad_label", "label needs ball or moments (or --answers)", exit_code=EXIT_BAD_INPUT)
    finished = (job.rec / "fullgame" / "game.mp4").exists()
    if kind == "moments" and not finished:
        raise Failed("no_game", "must-see moments are marked on a finished game: run it first",
                     exit_code=EXIT_BAD_INPUT)
    name, sd = label_dir(job, kind)
    req_path = sd / "request.json"
    if req_path.exists():
        req = json.loads(req_path.read_text())  # carry on with what was asked first
    else:
        n_set = int(name.split("-")[1])
        req = {"kind": kind, "count": count, "at": at, "random": n_random, "seed": n_set}
        if kind == "ball" and not finished:
            req["partial"] = True  # asked for before the game was done: made from its first chunks
        sd.mkdir(parents=True, exist_ok=True)
        req_path.write_text(json.dumps(req))
    rec = str(job.rec)
    game_s = game_seconds(job)
    out: dict = {"schemaVersion": VERSION, "set": name, "kind": kind, "labelled": 0}
    if req.get("partial"):
        out["partial"] = True
    if kind == "ball":
        # The ball stage's candidates, chunk by chunk; frames shared out by length.
        # While the game is still running the set is made from the chunks
        # the ball stage has finished (their candidate files are written
        # last, whole): which ones is kept with the request, so carrying on
        # with the set later picks from the same chunks.
        chunks = []
        k, t = 0, 0.0
        while t < game_s - 1:
            f = sorted((job.rec / "detect").glob(f"ball_0_{t:g}_*.win.json"))
            if f and k in req.get("chunks", [k]):
                chunks.append((k, f[0].name, float(f[0].name.split("_")[3].removesuffix(".win.json"))))
            k, t = k + 1, t + job.chunk_s
        if not chunks:
            raise Failed("no_candidates", "no ball candidates in detect/ yet: the game's first chunk must be "
                         "through the ball stage (run it first)", exit_code=EXIT_BAD_INPUT)
        if "chunks" not in req:
            req["chunks"] = [c[0] for c in chunks]
            req_path.write_text(json.dumps(req))
        # Held out for testing before anything is labelled: the last quarter
        # of every chunk, the same stretches for every set of this game, so
        # no set trains on another's test time (docs/TESTING-PROCESSING.md).
        import holdout

        if not (sd / holdout.MARK).exists():
            wins = [[round(k * job.chunk_s + n * (1 - HELD_OUT_SHARE), 3), round(k * job.chunk_s + n, 3)]
                    for k, _, n in chunks]
            (sd / holdout.MARK).write_text(json.dumps({"gameS": wins}, indent=1) + "\n")
        total = sum(c[2] for c in chunks)
        share = [int(req["count"] * c[2] / total) for c in chunks]
        for i in range(req["count"] - sum(share)):
            share[i % len(share)] += 1
        for (k, cands, _), n in zip(chunks, share):
            mark = sd / f".chunk{k:02d}"
            if n == 0 or mark.exists():
                continue
            step(job, f"label-{name}-chunk{k:02d}", ["labelset.py", rec, "ball", cands, str(n), "main",
                                                      str(100 * req["seed"] + k), "--out", str(sd)])
            mark.touch()
        items = json.loads((sd / "items.json").read_text()) if (sd / "items.json").exists() else []
        if len(items) < req["count"] and not (sd / ".nojudge").exists():
            # Few frames on a ball path: the judge doesn't know this ball
            # (rec7's yellow one), so the path is thin, and these are the
            # games that most need labels. Make up the rest from the
            # detector's candidates without the judge.
            short = req["count"] - len(items)
            for i, (k, cands, _) in enumerate(chunks):
                n = short // len(chunks) + (1 if i < short % len(chunks) else 0)
                if n:
                    step(job, f"label-{name}-nojudge{k:02d}", ["labelset.py", rec, "ball", cands, str(n), "main",
                                                                str(100 * req["seed"] + 50 + k), "--out", str(sd),
                                                                "--nojudge"])
            (sd / ".nojudge").touch()
            items = json.loads((sd / "items.json").read_text()) if (sd / "items.json").exists() else []
        held = holdout.Holdout(job.rec)
        out["items"] = [{"id": it["id"], "image": it["image"], "videoT": video_t(job, it["segment"], it["t"]),
                         "width": it["width"], "height": it["height"], "pick": it.get("pick", "detected"),
                         "judged": it.get("judge", True), "heldOut": held.held(it["segment"], it["t"]),
                         "suggestions": it["candidates"]} for it in items]
    else:
        times = [round(float(x), 2) for x in req["at"]]
        bad = [x for x in times if not 0 <= x < game_s]
        if bad:
            raise Failed("bad_label", f"--at {bad[0]:g} s is outside the game (0-{game_s:.0f} s)",
                         exit_code=EXIT_BAD_INPUT)
        rng = random.Random(req["seed"])
        tries = 0
        picked = list(times)
        while len(picked) < len(times) + req["random"] and tries < 1000:
            tries += 1
            x = round(rng.uniform(30, max(31, game_s - 30)), 2)
            if all(abs(x - y) >= RANDOM_MOMENT_GAP_S for y in picked):
                picked.append(x)
        step(job, f"label-{name}", ["playeval.py", "make", rec, name, "0", "0", *[f"{x:g}" for x in picked],
                                    "--out", str(sd)], done=sd / "pano.json")
        meta = json.loads((sd / "pano.json").read_text())
        out["panorama"] = {"degPerPx": meta["panoDeg"], "yawRange": meta["yawDeg"], "pitchRange": meta["pitchDeg"],
                           "width": meta["width"], "height": meta["height"]}
        items = json.loads((sd / "items.json").read_text())
        out["items"] = [{"id": it["id"], "image": it["image"], "videoT": it["clipT"],
                         "pick": "missed" if any(abs(it["clipT"] - x) < 0.01 for x in times) else "random",
                         "camera": camera_at(job, it["clipT"])} for it in items]
    write_set(sd, out)
    emit("labelset", set=name, kind=kind, path=str(sd / "set.json"), items=len(out["items"]))
    return EXIT_OK


def write_set(sd: Path, s: dict) -> None:
    answers = json.loads((sd / "answers.json").read_text()) if (sd / "answers.json").exists() else {}
    for it in s["items"]:
        it["answered"] = it["id"] in answers
    s["labelled"] = sum(it["answered"] for it in s["items"])
    validate(s, "engine-labelset.schema.json", "labelset")
    tmp = sd / "set.tmp"
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(sd / "set.json")


def apply_labels(job: Job, answers: dict) -> int:
    sd = job.dir / "labels" / answers["set"]
    if not (sd / "set.json").exists():
        raise Failed("bad_answers", f"no label set {answers['set']} (engine/labels/)", exit_code=EXIT_BAD_INPUT)
    s = json.loads((sd / "set.json").read_text())
    given = answers.get(s["kind"])
    if given is None:
        raise Failed("bad_answers", f"{answers['set']} is a {s['kind']} set", exit_code=EXIT_BAD_INPUT)
    ids = {it["id"] for it in s["items"]}
    unknown = sorted(set(given) - ids)
    if unknown:
        raise Failed("bad_answers", f"{len(unknown)} item(s) not in {answers['set']}, e.g. {unknown[0]}",
                     exit_code=EXIT_BAD_INPUT)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    merged = json.loads((sd / "answers.json").read_text()) if (sd / "answers.json").exists() else {}
    merged.update(given)
    (sd / "answers.json").write_text(json.dumps(merged, indent=1))
    # The same labels in the formats the benches and training already read.
    if s["kind"] == "ball":
        labels = {i: {"status": a["status"],
                      "balls": [{"x": a["x"], "y": a["y"], "inPlay": True}] if a["status"] == "ball" else [],
                      "at": now} for i, a in merged.items()}
        (sd / "labels.json").write_text(json.dumps(labels, indent=1))
    else:
        boxes = {i: {"yaw0": min(a["yaw0"], a["yaw1"]), "yaw1": max(a["yaw0"], a["yaw1"]),
                     "pitch0": min(a["pitch0"], a["pitch1"]), "pitch1": max(a["pitch0"], a["pitch1"]),
                     "note": a.get("note", ""), "draftBy": "user", "edited": True, "reviewed": True, "at": now}
                 for i, a in merged.items() if not a.get("skipped")}
        (sd / "boxes.json").write_text(json.dumps(boxes, indent=1))
    write_set(sd, s)
    emit("labels", set=answers["set"], labelled=s["labelled"], items=len(s["items"]))
    return EXIT_OK


# ------------------------------------------------------------------- share


def engine_version() -> str:
    v = os.environ.get("OSB_ENGINE_VERSION")
    if v:
        return v
    try:
        r = subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def last_answer(st: dict, question: str) -> dict | None:
    return next((a["answer"] for a in reversed(st["applied"]) if a["question"] == question), None)


def share_one(job: Job, m: dict, name: str, kind: str, summary: dict, paths: list[Path]) -> None:
    import hashlib

    files = []
    for p in paths:
        if not p.exists():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        files.append({"path": str(p.relative_to(job.rec)), "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    c = {"schemaVersion": VERSION, "name": name, "kind": kind,
         "recording": {k: m[k] for k in ("gameId", "sessionId", "createdAt") if k in m},
         "engine": {"version": engine_version()}, "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "summary": summary, "files": files}
    validate(c, "engine-contribution.schema.json", "contribution")
    d = job.dir / "share"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(c, indent=1))
    emit("share", name=name, path=str(d / f"{name}.json"), files=len(files))


def share(job: Job, kinds: list[str]) -> int:
    if not job.state_path.exists():
        raise Failed("not_prepared", "nothing to share yet: prepare first", exit_code=EXIT_BAD_INPUT)
    st, m, rec = job.state(), job.manifest(), job.rec
    if "pitch" in kinds:
        a = last_answer(st, "pitch")
        if st["pitch"]["status"] != "approved":
            emit("share", name="pitch", path=None, reason="the pitch isn't approved yet")
        elif a is not None and a.get("share") is False:
            emit("share", name="pitch", path=None, reason="the user chose not to share it")
        else:
            guess = job.dir / "pitch_guess.json"
            if not guess.exists():
                guess = rec / "pitch" / "pitch_before_taps.json"
            fit = json.loads((rec / "pitch" / "pitch.json").read_text())
            points = [x["answer"] for x in st["applied"] if x["question"] == "pitch" and x["answer"].get("points")]
            summary = {"approvedRound": st["pitch"].get("approvedRound"),
                       "approvedFirstGuess": st["pitch"].get("approvedRound") == 1, "corrections": len(points),
                       "fit": {k: fit[k] for k in ("lengthM", "widthM", "heightM", "fittedWith") if k in fit},
                       "answers": [x["answer"] for x in st["applied"] if x["question"] == "pitch"]}
            share_one(job, m, "pitch", "pitch", summary,
                      [rec / "pitch" / "background.png", rec / "pitch" / "pitch.json", guess,
                       rec / "pitch" / "taps.json", rec / "pitch" / "goals.json", rec / "pitch" / "paint.json",
                       rec / "pitch" / "review.json", rec / "phase0" / "cameras.json"])
    for kind in ("ball", "moments"):
        if kind not in kinds:
            continue
        for sd in sorted((job.dir / "labels").glob(f"{kind}-*")):
            if not (sd / "set.json").exists():
                continue
            s = json.loads((sd / "set.json").read_text())
            answers = json.loads((sd / "answers.json").read_text()) if (sd / "answers.json").exists() else {}
            if not answers:
                emit("share", name=s["set"], path=None, reason="nothing labelled in it yet")
                continue
            by_status: dict[str, int] = {}
            for x in answers.values():
                key = x.get("status", "skipped" if x.get("skipped") else "box")
                by_status[key] = by_status.get(key, 0) + 1
            images = [sd / it["image"] for it in s["items"] if it["id"] in answers and not answers[it["id"]].get("skipped")]
            summary = {"items": len(s["items"]), "labelled": len(answers), "byAnswer": by_status,
                       "byPick": {p: sum(1 for it in s["items"] if it["id"] in answers and it.get("pick") == p)
                                  for p in {it.get("pick") for it in s["items"]} if p}}
            extra = [sd / "labels.json"] if kind == "ball" else [sd / "boxes.json", rec / "pitch" / "pitch.json"]
            share_one(job, m, s["set"], kind, summary,
                      [sd / "set.json", *extra, rec / "phase0" / "cameras.json", *images])
    return EXIT_OK


def youtube_made(job: Job) -> Path | None:
    """The finished YouTube copy, if there is one (done.json names it once it is checked)."""
    out = job.rec / "fullgame"
    try:
        named = json.loads((out / "done.json").read_text()).get("youtube")
    except (OSError, json.JSONDecodeError):
        return None
    return out / named if named and (out / named).exists() else None


def youtube_copy(job: Job, game: Path) -> dict:
    """The job's YouTube copy of a finished game.mp4, made now if it isn't there yet: scaled up to
    1440p, the audio as it is. {"youtubeOutput": its path} for the done event; {} when the job
    asks for none. Written under another name first, so a stopped copy is never taken for one.
    A game.mp4 that is 1440p already (made before the job asked for 1080p) is itself that file."""
    global _current
    if not job.youtube_copy:
        return {}
    made = youtube_made(job)
    if made is not None:
        emit("step", step="youtube", status="skipped")
        return {"youtubeOutput": str(made)}
    if _stopping.is_set():
        raise Stop()
    done_mark = game.with_name("done.json")
    if (video_height(game) or 0) >= YOUTUBE_H:
        done_mark.write_text(json.dumps({**json.loads(done_mark.read_text()), "youtube": game.name}, indent=1))
        emit("step", step="youtube", status="skipped")
        return {"youtubeOutput": str(game)}
    check_disk(job)
    copy, part = game.with_name(YOUTUBE_NAME), game.with_name("game-youtube.part.mp4")
    job.logs.mkdir(parents=True, exist_ok=True)
    log = job.logs / "youtube.log"
    cmd = [accel.ffmpeg(), "-v", "error", "-y", "-i", str(game),
           "-vf", f"scale={YOUTUBE_H * 16 // 9}:{YOUTUBE_H}:flags=lanczos",
           *accel.h264_args(YOUTUBE_BITRATE, YOUTUBE_BITRATE), "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", str(part)]
    emit("step", step="youtube", status="started", log=str(log))
    t0 = time.time()
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        try:
            _current = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:  # no ffmpeg to run
            f.write(f"{e}\n")
            rc = None
        else:
            while _current.poll() is None:
                time.sleep(2)
            rc = _current.returncode
            _current = None
    if _stopping.is_set():
        raise Stop()
    want, got = video_seconds(game), video_seconds(part) if rc == 0 else None
    if got is None or want is None or abs(got - want) > LENGTH_TOLERANCE * want:
        raise Failed("youtube_copy_failed", f"the YouTube copy wasn't made (ffmpeg exit {rc}, "
                     f"{got if got is None else round(got)} s of {want if want is None else round(want)} s); "
                     f"game.mp4 is finished and kept: run again for the copy alone", log)
    part.replace(copy)
    d = json.loads(done_mark.read_text())
    d.update(youtube=YOUTUBE_NAME, youtubeBytes=copy.stat().st_size, youtubeHeight=YOUTUBE_H,
             youtubeSeconds=round(time.time() - t0, 1))
    done_mark.write_text(json.dumps(d, indent=1))
    emit("step", step="youtube", status="done", seconds=round(time.time() - t0, 1))
    return {"youtubeOutput": str(copy)}


def status(job: Job) -> int:
    st = job.state() if job.state_path.exists() else None
    out = job.rec / "fullgame"
    pieces = sorted(p.name for p in (out / "pieces").glob("piece_*.mp4")) if (out / "pieces").exists() else []
    emit("status", prepared=(job.dir / "questions.json").exists(),
         pending=pending_questions(st),
         pieces=len(pieces), output=str(out / "game.mp4") if (out / "game.mp4").exists() else None,
         youtubeOutput=(lambda y: str(y) if y else None)(youtube_made(job)),
         **ball_share(job, final=(out / "done.json").exists()), chunkDetails=chunk_details(job),
         next=next_step(job, st))
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("prepare", "run", "status", "questions", "answer", "label", "share"))
    ap.add_argument("job", type=Path)
    ap.add_argument("what", nargs="?", help="label: ball or moments; answer: cameras or pitch")
    ap.add_argument("--answers", type=Path)
    ap.add_argument("--no-footage", dest="footage", action="store_false",
                    help="prepare --answers: apply the answers without the segment files on disk")
    ap.add_argument("--count", type=int, default=BALL_SET_COUNT, help="label ball: frames in the set")
    ap.add_argument("--at", default="", help="label moments: seconds into game.mp4 where the user pressed Missed play")
    ap.add_argument("--random", type=int, default=3, help="label moments: random moments added")
    ap.add_argument("--kinds", default="pitch,ball,moments", help="share: which results")
    g = ap.add_argument_group("answer (see the question's howToAnswer)")
    g.add_argument("--left", choices=("main", "secondary"), help="cameras: the phone on the left")
    g.add_argument("--approve", action="store_true", help="pitch: the lines sit on the paint")
    g.add_argument("--closeup", action="append", default=[], metavar="NAME=X,Y", help="pitch: a keypoint, in its close-up's pixels")
    g.add_argument("--pano", action="append", default=[], metavar="NAME=X,Y", help="pitch: a keypoint, in the background's pixels")
    g.add_argument("--point", action="append", default=[], metavar="NAME=YAW,PITCH", help="pitch: a keypoint, in panorama degrees")
    g.add_argument("--keep", action="append", default=[], metavar="NAME", help="pitch: the fit has this keypoint right")
    g.add_argument("--skip", action="append", default=[], metavar="NAME", help="pitch: this keypoint can't be seen")
    g.add_argument("--share", action="store_true", help="pitch: the user agreed to share this result")
    a = ap.parse_args()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        job = Job(a.job)
        if a.command == "questions":
            return questions_brief(job)
        if a.command == "answer":
            return answer(job, a.what, a)
        answers = None
        if a.answers:
            try:
                answers = json.loads(a.answers.read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise Failed("bad_answers", f"can't read the answers {a.answers}: {e}", exit_code=EXIT_BAD_INPUT)
            # Before any work starts.
            if a.command != "label":
                answers = current_answers(answers)
            if a.command == "label":
                validate(answers, "engine-label-answers.schema.json", "answers")
            else:
                validate(answers, "engine-answers.schema.json", "answers")
        if a.command == "label":
            if answers is not None:
                return apply_labels(job, answers)
            try:
                at = [float(x) for x in a.at.split(",") if x.strip()]
            except ValueError:
                raise Failed("bad_label", f"--at takes seconds separated by commas, not {a.at!r}",
                             exit_code=EXIT_BAD_INPUT)
            return label(job, a.what, a.count, at, a.random)
        if a.command == "share":
            return share(job, [k.strip() for k in a.kinds.split(",") if k.strip()])
        if a.command == "prepare":
            return prepare(job, answers, a.footage)
        if a.command == "run":
            return run(job, answers)
        return status(job)
    except Stop:
        emit("stopped", message="stopped; run the same command to carry on")
        return EXIT_STOPPED
    except Failed as f:
        emit("error", code=f.code, message=f.message, **({"log": str(f.log)} if f.log else {}))
        return f.exit_code


if __name__ == "__main__":
    sys.exit(main())
