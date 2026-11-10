#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.10", "jsonschema>=4.20"]
# ///
"""osb-engine: one game from footage to game.mp4, driven through files, for
the Mac app, the cloud job and a terminal alike.

    uv run processing/spikes/engine.py prepare job.json [--answers answers.json]
    uv run processing/spikes/engine.py run     job.json --answers answers.json
    uv run processing/spikes/engine.py status  job.json
    uv run processing/spikes/engine.py label   job.json ball [--count 50]
    uv run processing/spikes/engine.py label   job.json moments [--at 61.5,1834] [--random 3]
    uv run processing/spikes/engine.py label   job.json --answers labels.json
    uv run processing/spikes/engine.py share   job.json [--kinds pitch,kits,ball,moments]

The job (schemas/engine-job.schema.json) names a recording folder: the
merged manifest.json and every segment file at its objectKey, as
footagestore.py fetch or the app's download leaves it.

prepare: syncs the phones (phase0.py), fits the pitch (pitch.py: lines,
goals, review) and groups a sample of people by kit (labelset.py clip,
teams.py), then writes <recording>/engine/questions.json
(schemas/engine-questions.schema.json): the pitch to approve or correct,
and the kit groups to check. When the phones' views can't be stitched
(rec5: too little overlap, one view soft), it first asks only which phone
filmed the left side (the panorama drawn both ways, jointfit.py choices: the
right one shows the pitch in one piece), then builds
rough cameras from the manifest's gravity sensors and the lens table
(jointfit.py rough) and carries on; the pitch must then be marked (tapped
points), and the taps fit the cameras and pitch together over a range of
starting angles (jointfit.py taps, search, install) before the pitch can
be approved. Given answers
(schemas/engine-answers.schema.json) it applies them: corrected pitch
points are refitted (pitch.py taps) and asked again; a split kit group
regroups with more components and is asked again (once); merges and
keeper / not-a-kit marks rewrite the sides roles.py uses. Run it until
questions.json has nothing pending.

run: the whole game (fullgame.py: ball, roles and follow in chunks), once
every question is answered. Writes <recording>/fullgame/game.mp4. It first
checks there is room for what is still to be written (the pieces, their
join video.mp4 and game.mp4 at the output's bitrate: ~49 GB for 100
minutes at 1440p, rec7 took 46) and stops with not_enough_disk if not. Once game.mp4 is
written and as long as the game, it deletes the pieces, video.mp4 and
audio.m4a (job option keepIntermediates keeps them) and writes
fullgame/done.json, so a rerun reports the game done instead of
rendering it again.

prepare and run check every segment file against the manifest's size and
SHA-256 (footage_corrupt: download it again); a file is hashed once, then
remembered by its size and modification time (engine/verified.json).

status: what's done and what's pending, as one JSON line.

label (after run, for users who agree to share): makes a label set in
<recording>/engine/labels/<set>/set.json (schemas/engine-labelset.schema.json).
"ball": frames of Main picked across the game as labelset.py's ball mode
picks them, half where the detector found the ball, half where the
tracker bridged a gap it missed, each with suggestions. "moments": the
stitched panorama at each time the user pressed Missed play (--at, seconds
into game.mp4) plus a few random ones, with where the camera pointed, for
must-see boxes. A stopped set carries on; a finished one makes the next
(ball-2, ...). --answers (schemas/engine-label-answers.schema.json) stores
labels in the formats the benches read: labels.json (label/server.py's)
and boxes.json (playeval.py's).

share: writes <recording>/engine/share/<name>.json
(schemas/engine-contribution.schema.json) for each result the user agreed
to share: the pitch and the kit groups (each answer's share switch, on by
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
SCHEMAS = HERE.parents[1] / "schemas"
VERSION = 1
TEAMS_SET = "engine"          # labels_engine/ and teams/engine/: the kit sample this engine makes
TEAMS_K = 8                   # teams.py components: kits, keepers, referee, and room for the rest
SPLIT_K = TEAMS_K + 2         # after the user says a group holds two kits
CROPS_PER_GROUP = 24          # people shown per kit group
STAT_MIN_HEIGHT_PX = 45       # teams.py: people this tall (on the pitch) shape the groups; shown first
CROP_H = 160                  # crop height in the question, px
HEARTBEAT_S = 60.0
# Failures a user can understand, found by the line a step ends with.
KNOWN_FAILURES = [
    ("calibration failed", "calibration_failed",
     "the two phones' views couldn't be matched to stitch them (too little overlap, or one view soft)"),
]
RANDOM_MOMENT_GAP_S = 20.0    # random must-see moments: at least this far from the others
BALL_SET_COUNT = 50           # frames in a ball label set
PIECE_BPS = 16e6 / 8          # follow.py's pieces: 16 Mbit/s (measured ~14.7 on rec7: 11 GB for 100 min)
GAME_BPS = {1080: PIECE_BPS, 1440: 33e6 / 8}  # game.mp4: fullgame asks 45 Mbit/s, hardware gives ~2/3 (rec7: ~31)
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
        return {"schemaVersion": VERSION, "pitch": {"round": 1, "status": "pending"},
                "kits": {"round": 1, "status": "pending", "k": TEAMS_K}, "applied": []}

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
    cmd = ["uv", "run", "-q", *[a for w in uv_with for a in ("--with", w)], str(HERE / args[0]), *args[1:]]
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


def sample_clip(job: Job, m: dict) -> tuple[int, float, float]:
    """Where the pitch and the kits are fitted: the job's choice, else a
    quarter of the way into the recording (the warm-up is over, the light
    hasn't changed much), 2 minutes from 1 minute into that segment."""
    if job.sample:
        return int(job.sample["segment"]), float(job.sample["startS"]), float(job.sample["seconds"])
    main = next(d for d in m["devices"] if d["role"] == "main")
    n = len(main["segments"])
    seg = min(n - 1, max(0, n // 4))
    dur = main["segments"][seg]["durationMs"] / 1000
    start = min(60.0, max(0.0, dur - 130.0))
    return seg, start, 120.0


def prepare(job: Job, answers: dict | None) -> int:
    m = check_footage(job)
    st = job.state()
    seg, start, secs = sample_clip(job, m)
    rec = str(job.rec)
    if answers:
        validate(answers, "engine-answers.schema.json", "answers")
    step(job, "sync", ["phase0.py", rec], done=job.rec / "phase0" / "report.json")
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
    step(job, "kits", ["teams.py", rec, TEAMS_SET, str(st["kits"]["k"])], done=teams_dir / "people.json")

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


def pending_questions(st: dict | None) -> list[str]:
    if not st:
        return ["pitch", "kits"]
    if (st.get("cameras") or {}).get("status") == "pending":
        return ["cameras"]
    return [n for n in ("pitch", "kits") if st[n]["status"] == "pending"]


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
            st["pitch"]["round"] += 1
            emit("answer", question="pitch", result="refitted", round=st["pitch"]["round"])
        st["applied"].append({"question": "pitch", "answer": a, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    k = answers.get("kits")
    if k and k["round"] == st["kits"]["round"] and st["kits"]["status"] == "pending":
        st["kits"].setdefault("scores", []).append(kit_score(job, st, k))
        teams_dir = job.rec / "teams" / TEAMS_SET
        if k.get("split") is not None and st["kits"]["k"] == TEAMS_K:
            # Regroup with more components; the groups are new, so ask again.
            for f in ("people.json", "model.npz", "sides.json", "sides_auto.json", "sheet.jpg"):
                (teams_dir / f).unlink(missing_ok=True)
            st["kits"]["k"] = SPLIT_K
            step(job, f"kits-k{SPLIT_K}", ["teams.py", rec, TEAMS_SET, str(SPLIT_K)], done=teams_dir / "people.json")
            st["kits"]["round"] += 1
            emit("answer", question="kits", result="regrouped", round=st["kits"]["round"])
        else:
            sides = sides_from_answer(teams_dir, k)
            if sides is None:
                st["kits"]["round"] += 1
                st["kits"]["problem"] = "fewer than two kit groups left: merge fewer, or mark fewer as not a kit"
                emit("answer", question="kits", result="rejected", round=st["kits"]["round"])
            else:
                auto = teams_dir / "sides_auto.json"
                if (teams_dir / "sides.json").exists() and not auto.exists():
                    (teams_dir / "sides.json").replace(auto)
                (teams_dir / "sides.json").write_text(json.dumps(sides, indent=1))
                st["kits"]["status"] = "answered"
                st["kits"].pop("problem", None)
                emit("answer", question="kits", result="answered", sides={"A": sides["A"], "B": sides["B"]})
        st["applied"].append({"question": "kits", "answer": k, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def kit_score(job: Job, st: dict, k: dict) -> dict:
    """Group purity (the share of the shown people not tapped as in the wrong
    group, per group and weighted by group size) and the structure answer."""
    q = json.loads((job.dir / "questions.json").read_text())["kits"]
    shown = {str(g["component"]): (len(g["crops"]), g["people"]) for g in q["groups"]}
    per = {}
    for c, (n, _) in shown.items():
        tapped = len(set(k.get("tapped", {}).get(c, [])))
        per[c] = round(1 - tapped / n, 3) if n else None
    w = [(per[c], shown[c][1]) for c in per if per[c] is not None]
    purity = round(sum(p * n for p, n in w) / max(1, sum(n for _, n in w)), 3) if w else None
    structure = "split" if k.get("split") is not None else "merged" if k.get("merge") else "ok"
    return {"round": k["round"], "groupPurity": per, "purity": purity, "structure": structure}


def sides_from_answer(teams_dir: Path, k: dict) -> dict | None:
    """sides.json for roles.py from the user's groups: merged components are
    one kit; the two largest kit groups (people on the pitch per moment) are
    the sides; keeper/referee as marked, else teams.py's candidates."""
    auto = {}
    for name in ("sides_auto.json", "sides.json"):  # teams.py's own, kept aside once answers replace it
        if (teams_dir / name).exists():
            auto = json.loads((teams_dir / name).read_text())
            break
    people = json.loads((teams_dir / "people.json").read_text())
    ncomp = 1 + max(p["component"] for p in people)
    per_moment = auto.get("perMoment") or per_moment_from(people, ncomp)
    kor = set(k.get("keeperOrReferee") or [])
    not_kit = set(k.get("notKit") or [])
    parent = list(range(ncomp))

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    merged = set()
    for g in k.get("merge") or []:
        g = [c for c in g if c < ncomp]
        merged.update(g)
        for c in g[1:]:
            parent[find(c)] = find(g[0])
    kit_like = set(auto.get("kitComponents", [])) | merged
    groups: dict[int, list[int]] = {}
    for c in range(ncomp):
        if c in kor or c in not_kit:
            continue
        groups.setdefault(find(c), []).append(c)
    cands = [g for g in groups.values() if any(c in kit_like for c in g)]
    cands.sort(key=lambda g: -sum(per_moment[c] for c in g))
    if len(cands) < 2:
        return None
    a, b = sorted(cands[0]), sorted(cands[1])
    solos = sorted(kor) if k.get("keeperOrReferee") is not None else \
        [c for c in auto.get("keeperOrReferee", []) if c not in a + b and c not in not_kit]
    return {"A": a, "B": b, "keeperOrReferee": solos, "kitComponents": sorted(set(a + b)),
            "perMoment": per_moment, "source": "engine answers", "answer": k}


def per_moment_from(people: list, ncomp: int) -> list[float]:
    moments = {p["frame"].rsplit("_", 1)[0] for p in people}
    counts = [0] * ncomp
    for p in people:
        if p["onPitch"] and p["box"][3] - p["box"][1] >= STAT_MIN_HEIGHT_PX:
            counts[p["component"]] += 1
    return [round(c / max(1, len(moments)), 2) for c in counts]


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
    return q


def save_questions(job: Job, q: dict) -> dict:
    validate(q, "engine-questions.schema.json", "questions")
    job.dir.mkdir(parents=True, exist_ok=True)
    tmp = job.dir / "questions.tmp"
    tmp.write_text(json.dumps(q, indent=1))
    tmp.replace(job.dir / "questions.json")
    return q


def write_questions(job: Job, st: dict) -> dict:
    import cv2

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
               "goals": review["goals"], "points": review["points"]}
    paint = rec / "pitch" / "paint.json"
    if paint.exists():
        pitch_q["lineColour"] = json.loads(paint.read_text())
    if st["pitch"].get("approveAllowed") is False:
        pitch_q["approveAllowed"] = False
    if st["pitch"].get("problem"):
        pitch_q["problem"] = st["pitch"]["problem"]
    guess = job.dir / "pitch_guess.json"
    if st["pitch"]["round"] == 1 and not guess.exists():
        # Our first guess, before any correction: shared next to the final fit.
        job.dir.mkdir(parents=True, exist_ok=True)
        guess.write_text((rec / "pitch" / "pitch.json").read_text())

    teams_dir = rec / "teams" / TEAMS_SET
    people = json.loads((teams_dir / "people.json").read_text())
    items = json.loads((rec / f"labels_{TEAMS_SET}" / "items.json").read_text())
    image_of = {f["id"]: rec / f"labels_{TEAMS_SET}" / f["image"] for it in items for f in it["frames"]}
    sides = {}
    for name in ("sides.json", "sides_auto.json"):
        if (teams_dir / name).exists() and not sides:
            sides = json.loads((teams_dir / name).read_text())
    ncomp = 1 + max(p["component"] for p in people)
    per_moment = sides.get("perMoment") or per_moment_from(people, ncomp)
    rnd = st["kits"]["round"]
    crop_dir = job.dir / "crops" / f"kits{rnd}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rnd)
    cache: dict[Path, object] = {}
    groups = []
    for c in range(ncomp):
        mine = [p for p in people if p["component"] == c]
        tall = [p for p in mine if p["onPitch"] and p["box"][3] - p["box"][1] >= STAT_MIN_HEIGHT_PX]
        pool = tall if len(tall) >= CROPS_PER_GROUP else mine
        pick = rng.sample(pool, min(CROPS_PER_GROUP, len(pool)))
        crops = []
        for n, p in enumerate(pick):
            path = image_of.get(p["frame"])
            if path is None:
                continue
            if path not in cache:
                if len(cache) > 8:
                    cache.clear()
                cache[path] = cv2.imread(str(path))
            img = cache[path]
            if img is None:
                continue
            x1, y1, x2, y2 = p["box"]
            pad = int(0.15 * (y2 - y1))
            h, w = img.shape[:2]
            cut = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if cut.size == 0:
                continue
            scale = CROP_H / cut.shape[0]
            cut = cv2.resize(cut, (max(1, round(cut.shape[1] * scale)), CROP_H),
                             interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
            cid = f"c{c}_{n:02d}"
            out = crop_dir / f"{cid}.jpg"
            if not out.exists():
                cv2.imwrite(str(out), cut, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crops.append({"id": cid, "image": str(out.relative_to(rec)), "frame": p["frame"], "box": p["box"]})
        proposed = ("A" if c in sides.get("A", []) else "B" if c in sides.get("B", [])
                    else "keeperOrReferee" if c in sides.get("keeperOrReferee", []) else "other")
        groups.append({"component": c, "people": len(mine), "perMoment": per_moment[c], "proposed": proposed,
                       "crops": crops})
    kits_q = {"round": rnd, "status": st["kits"]["status"], "k": st["kits"]["k"], "groups": groups,
              "splitAllowed": st["kits"]["k"] == TEAMS_K}
    if st["kits"].get("problem"):
        kits_q["problem"] = st["kits"]["problem"]
    (job.dir / f"kits_round{rnd}.json").write_text(json.dumps(kits_q, indent=1))  # what was shown, for share
    q = {"schemaVersion": VERSION, "recording": str(rec), "pending": pending_questions(st), "pitch": pitch_q,
         "kits": kits_q}
    if cams_q:
        q["cameras"] = cams_q
    return save_questions(job, q)


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
        emit("needs_answers", pending=pending or ["pitch", "kits"], questions=str(job.dir / "questions.json"))
        return EXIT_NEEDS_ANSWERS
    out = job.rec / "fullgame"
    done_mark = out / "done.json"
    if done_mark.exists() and (out / "game.mp4").exists():
        d = json.loads(done_mark.read_text())
        emit("step", step="game", status="skipped")
        emit("done", output=str(out / "game.mp4"), seconds=d.get("seconds", 0), **ball_share(job))
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
                elif e["event"] == "chunk":
                    if e["ok"]:
                        prog[e["stage"]].add(e["chunk"])
                    else:
                        prog["fail"] = prog["fail"] or e  # the first; the rest follow from it
                    report(force=True)
                elif e["event"] == "mux":
                    emit("step", step="mux", status="started")
                elif e["event"] in ("done", "failed"):
                    prog["result"] = e
        report()

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
    emit("done", output=str(game), seconds=secs, **ball_share(job))
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
    all on disk together at the end."""
    chunks = planned_chunks(job)
    pieces = job.rec / "fullgame" / "pieces"
    todo = sum(sec for k, sec in enumerate(chunks) if not (pieces / f"piece_{k:02d}.mp4").exists())
    total = sum(chunks)
    game_bps = GAME_BPS.get(job.height, GAME_BPS[1440])
    return todo * PIECE_BPS + total * PIECE_BPS + total * game_bps


def check_disk(job: Job) -> None:
    import shutil

    need = disk_needed(job) + DISK_MARGIN
    free = shutil.disk_usage(job.rec).free
    if free < need:
        raise Failed("not_enough_disk", f"this game needs about {need / 1e9:.0f} GB free on the disk holding the "
                     f"recording and {free / 1e9:.0f} GB is: free {(need - free) / 1e9:.0f} GB more, then run again")


def video_seconds(path: Path) -> float | None:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip())
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


def ball_share(job: Job) -> dict:
    """How much of the game the camera aimed at the ball (follow.py's
    "aim at the ball in N% of the clip", per chunk). Low means the models
    don't know this ball: the app can then offer ball labelling."""
    import re

    shares = []
    for p in sorted((job.rec / "fullgame" / "logs").glob("follow_*.log")):
        m = re.findall(r"aim at the ball in (\d+)% of the clip", p.read_text())
        if m:
            shares.append(int(m[-1]))
    if not shares:
        return {}
    return {"ballAimShare": round(sum(shares) / len(shares) / 100, 3), "ballAimByChunk": shares}


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
    if not (job.rec / "fullgame" / "game.mp4").exists():
        raise Failed("no_game", "label sets are made from a finished game: run it first", exit_code=EXIT_BAD_INPUT)
    name, sd = label_dir(job, kind)
    req_path = sd / "request.json"
    if req_path.exists():
        req = json.loads(req_path.read_text())  # carry on with what was asked first
    else:
        n_set = int(name.split("-")[1])
        req = {"kind": kind, "count": count, "at": at, "random": n_random, "seed": n_set}
        sd.mkdir(parents=True, exist_ok=True)
        req_path.write_text(json.dumps(req))
    rec = str(job.rec)
    game_s = game_seconds(job)
    out: dict = {"schemaVersion": VERSION, "set": name, "kind": kind, "labelled": 0}
    if kind == "ball":
        # The ball stage's candidates, chunk by chunk; frames shared out by length.
        chunks = []
        k, t = 0, 0.0
        while t < game_s - 1:
            f = sorted((job.rec / "detect").glob(f"ball_0_{t:g}_*.win.json"))
            if f:
                chunks.append((k, f[0].name, float(f[0].name.split("_")[3].removesuffix(".win.json"))))
            k, t = k + 1, t + job.chunk_s
        if not chunks:
            raise Failed("no_candidates", "no ball candidates in detect/: was the game run by this engine?",
                          exit_code=EXIT_BAD_INPUT)
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
    if "kits" in kinds:
        a = last_answer(st, "kits")
        if st["kits"]["status"] != "answered":
            emit("share", name="kits", path=None, reason="the kit groups aren't answered yet")
        elif a is not None and a.get("share") is False:
            emit("share", name="kits", path=None, reason="the user chose not to share them")
        else:
            summary = {"scores": st["kits"].get("scores", []), "k": st["kits"]["k"],
                       "answers": [x["answer"] for x in st["applied"] if x["question"] == "kits"]}
            rounds = sorted(job.dir.glob("kits_round*.json"))
            crops = sorted((job.dir / "crops").glob("kits*/*.jpg"))
            share_one(job, m, "kits", "kits", summary,
                      [*rounds, *crops, rec / "teams" / TEAMS_SET / "sides.json",
                       rec / "teams" / TEAMS_SET / "sides_auto.json"])
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


def status(job: Job) -> int:
    st = job.state() if job.state_path.exists() else None
    out = job.rec / "fullgame"
    pieces = sorted(p.name for p in (out / "pieces").glob("piece_*.mp4")) if (out / "pieces").exists() else []
    emit("status", prepared=(job.dir / "questions.json").exists(),
         pending=pending_questions(st),
         pieces=len(pieces), output=str(out / "game.mp4") if (out / "game.mp4").exists() else None)
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("prepare", "run", "status", "label", "share"))
    ap.add_argument("job", type=Path)
    ap.add_argument("what", nargs="?", help="label: ball or moments")
    ap.add_argument("--answers", type=Path)
    ap.add_argument("--count", type=int, default=BALL_SET_COUNT, help="label ball: frames in the set")
    ap.add_argument("--at", default="", help="label moments: seconds into game.mp4 where the user pressed Missed play")
    ap.add_argument("--random", type=int, default=3, help="label moments: random moments added")
    ap.add_argument("--kinds", default="pitch,kits,ball,moments", help="share: which results")
    a = ap.parse_args()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        job = Job(a.job)
        answers = None
        if a.answers:
            try:
                answers = json.loads(a.answers.read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise Failed("bad_answers", f"can't read the answers {a.answers}: {e}", exit_code=EXIT_BAD_INPUT)
            # Before any work starts.
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
            return prepare(job, answers)
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
