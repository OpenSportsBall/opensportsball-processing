#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["boto3>=1.35"]
# ///
"""osb-engine in the cloud: one AWS Batch job of the main repo's cloud
processing (its docs/CLOUD-PROCESSING-PLAN.md and, for what this program
is given and must leave behind, docs/API.md, "The engine's container").

    cloud.py prepare     footage from S3, engine.py prepare, questions to S3
    cloud.py refit       the same, applying answers.json; without the footage when it can
    cloud.py run         engine.py run (answers applied first), game.mp4 to S3;
                         or the questions again, when the engine won't use an answer

It is the only part of the engine that knows about S3, DynamoDB or Batch.
engine.py does the work, on a recording folder on the scratch disk, exactly
as on a Mac:

    OSB_JOB_ID           the job's id
    OSB_BUCKET           the footage bucket
    OSB_TABLE            the table holding the job record (pk JOB#<id>, sk JOB)
    OSB_FOOTAGE_PREFIX   accounts/.../recordings/<id>/: manifest.json and the segments (read)
    OSB_OUTPUT_PREFIX    outputs/.../jobs/<id>/: what the user gets (read, write)
    OSB_WORK_PREFIX      work/.../jobs/<id>/: the recording folder's other files (read, write)
    OSB_SCRATCH          a disk with room for the game (default /scratch)
    OSB_MODELS_PREFIX    where the models in models.json are (default models/)
    OSB_REQUIRE_GPU=1    stop at once (no_gpu) unless accel.py finds an NVIDIA GPU
    OSB_JOB_OPTIONS      JSON for the engine job's `options` and `output`
                         (schemas/engine-job.schema.json), e.g.
                         {"options": {"maxChunks": 1}} to try one chunk of a game

In: the footage (skipped when a file is already here at the manifest's
size), everything under the work prefix (the engine's state from the
job's earlier commands, or from this command before a Spot instance was
taken away), and answers.json from the output prefix.

Out, under the output prefix:
    questions/questions.json   the engine's questions.json; every picture it names
                               is at questions/<its path>
    game.mp4, result.json      the video, and the engine's done event (ballAimShare, ballLow)
    logs/                      the engine's logs, when a command fails
    profile/<command>.json     what the command used of the machine and how long each
                               part took (cloudprofile.py), with its samples beside it

The recording folder without the footage goes back to the work prefix as
the engine finishes steps and chunks, and at the end, so the next command
(or the next attempt) carries on. File times are kept: the engine
compares some.

The job record gets `progress` (at most every 30 s), `engineVersion`, and
before a failing exit `error` ({code, message}): the engine's own code
(footage_corrupt, calibration_failed, ...). `status` is the server's.

Exit 0 when the command's file is in S3 (for run: game.mp4, or new
questions); 130 when stopped (the instance is
going: Batch runs the command again elsewhere); anything else is a
failure.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import accel  # noqa: E402
from cloudprofile import Profile  # noqa: E402

PROGRESS_EVERY_S = 30.0
SYNC_EVERY_S = 300.0
# prepare's steps, in order, for a fraction (engine.py prepare).
PREPARE_STEPS = ["verify-footage", "sync", "pitch", "goals", "kit-sample", "kits", "pitch-review"]
# Fields of questions.json that name a picture, relative to the recording.
PICTURE_KEYS = {"image", "overlay", "background", "sheet", "mainLeft", "secondaryLeft"}
# Never copied to the work prefix: the output itself and what it is joined from at the end.
WORK_SKIP = {"fullgame/game.mp4", "fullgame/video.mp4", "fullgame/audio.m4a"}
EXIT_STOPPED = 130


class Failed(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if not v:
        raise Failed("bad_job", f"{name} is not set")
    return v


def log(*a) -> None:
    print("[cloud]", *a, file=sys.stderr, flush=True)


class Cloud:
    """The job's S3 folders and its record."""

    def __init__(self):
        import boto3
        from boto3.s3.transfer import TransferConfig

        self.s3 = boto3.client("s3")
        self.ddb = boto3.client("dynamodb")
        self.bucket = env("OSB_BUCKET")
        self.table = env("OSB_TABLE")
        self.job_id = env("OSB_JOB_ID")
        self.footage = env("OSB_FOOTAGE_PREFIX")
        self.out = env("OSB_OUTPUT_PREFIX")
        self.work = env("OSB_WORK_PREFIX")
        self.transfer = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=8)
        self._sent: dict[str, tuple[int, int]] = {}  # work files already there: size, mtime
        self._last_progress = 0.0

    # -- S3

    def list(self, prefix: str) -> list[dict]:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            out += page.get("Contents", [])
        return out

    def fetch(self, key: str, dest: Path, mtime_ns: int | None = None) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        self.s3.download_file(self.bucket, key, str(tmp), Config=self.transfer)
        if mtime_ns is not None:
            os.utime(tmp, ns=(mtime_ns, mtime_ns))
        tmp.replace(dest)

    def put(self, path: Path, key: str, **extra) -> None:
        self.s3.upload_file(str(path), self.bucket, key, Config=self.transfer, ExtraArgs=extra or None)

    def get_json(self, key: str):
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None

    # -- the recording folder

    def fetch_footage(self, rec: Path) -> tuple[int, float]:
        """manifest.json and every segment, as engine.py expects them. A
        file already here at the manifest's size is kept. Each file gets the
        time S3 has for it, so the engine's record of files it has hashed
        (engine/verified.json, kept in the work prefix) still holds on
        another machine. Returns the bytes fetched and the seconds it took."""
        objs = {o["Key"][len(self.footage):]: o for o in self.list(self.footage)}
        if "manifest.json" not in objs:
            raise Failed("footage_missing", "the recording's footage is no longer there")
        self.fetch(self.footage + "manifest.json", rec / "manifest.json")
        m = json.loads((rec / "manifest.json").read_text())
        todo = []
        for d in m.get("devices", []):
            for s in d.get("segments", []):
                o = objs.get(s["objectKey"])
                if o is None:
                    raise Failed("footage_missing", f"segment {s['objectKey']} is not in the bucket")
                f = rec / s["objectKey"]
                if not (f.exists() and f.stat().st_size == o["Size"]):
                    todo.append((s["objectKey"], f, int(o["LastModified"].timestamp()) * 1_000_000_000))
        if not todo:
            return 0, 0.0
        size = sum(objs[k]["Size"] for k, _, _ in todo)
        log(f"fetching {len(todo)} segment files, {size / 1e9:.1f} GB")
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda t: self.fetch(self.footage + t[0], t[1], t[2]), todo))
        log(f"footage fetched in {time.time() - t0:.0f} s")
        return size, time.time() - t0

    def restore_work(self, rec: Path) -> None:
        """Everything under the work prefix, with the times it had."""
        objs = self.list(self.work)

        def one(o):
            rel = o["Key"][len(self.work):]
            dest = rec / rel
            head = self.s3.head_object(Bucket=self.bucket, Key=o["Key"])
            mtime = int(head.get("Metadata", {}).get("mtime-ns", 0)) or None
            if not (dest.exists() and dest.stat().st_size == o["Size"]):
                self.fetch(o["Key"], dest, mtime)
            st = dest.stat()
            self._sent[rel] = (st.st_size, st.st_mtime_ns)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one, objs))
        if objs:
            log(f"restored {len(objs)} files from the work prefix")

    def save_work(self, rec: Path, footage: set[str]) -> None:
        """New and changed files of the recording folder, except the footage."""
        todo = []
        for f in rec.rglob("*"):
            if not f.is_file() or f.name.endswith((".part", ".tmp")):
                continue
            rel = f.relative_to(rec).as_posix()
            if rel in footage or rel in WORK_SKIP or rel == "manifest.json":
                continue
            st = f.stat()
            if self._sent.get(rel) != (st.st_size, st.st_mtime_ns):
                todo.append((rel, f, st))

        def one(t):
            rel, f, st = t
            try:
                self.put(f, self.work + rel, Metadata={"mtime-ns": str(st.st_mtime_ns)})
                self._sent[rel] = (st.st_size, st.st_mtime_ns)
            except FileNotFoundError:
                pass  # the engine replaced or removed it meanwhile; the next pass has it

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one, todo))

    # -- the job record

    def _update(self, expression: str, names: dict, values: dict) -> None:
        try:
            self.ddb.update_item(
                TableName=self.table, Key={"pk": {"S": f"JOB#{self.job_id}"}, "sk": {"S": "JOB"}},
                UpdateExpression=expression, ConditionExpression="attribute_exists(pk)",
                ExpressionAttributeNames=names, ExpressionAttributeValues=values)
        except self.ddb.exceptions.ConditionalCheckFailedException:
            pass  # the job was deleted: nothing to tell
        except Exception as e:  # progress must never stop the work
            log(f"could not update the job record: {e}")

    def progress(self, stage: str, fraction: float, eta_s: float | None, force: bool = False) -> None:
        if not force and time.time() - self._last_progress < PROGRESS_EVERY_S:
            return
        self._last_progress = time.time()
        p = {"stage": {"S": stage}, "fraction": {"N": f"{min(1.0, max(0.0, fraction)):.4f}"},
             "etaSeconds": {"N": str(int(eta_s))} if eta_s is not None else {"NULL": True},
             "at": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
        self._update("SET #p = :p, #v = :v", {"#p": "progress", "#v": "engineVersion"},
                     {":p": {"M": p}, ":v": {"S": engine_version()}})

    def error(self, code: str, message: str) -> None:
        self._update("SET #e = :e", {"#e": "error"},
                     {":e": {"M": {"code": {"S": code}, "message": {"S": message[:1000]}}}})


def engine_version() -> str:
    """The image says which commit it was built from (packaging/cloud/Dockerfile)."""
    return os.environ.get("OSB_ENGINE_VERSION") or "osb-engine (cloud)"


def fetch_models(cloud: Cloud) -> None:
    """The models this engine was built for (packaging/cloud/models.json:
    where each goes, and its SHA-256), from the models prefix. A model
    already here and right is kept."""
    spec = json.loads((REPO / "packaging" / "cloud" / "models.json").read_text())
    prefix = os.environ.get("OSB_MODELS_PREFIX", "models/")
    for m in spec["models"]:
        dest = REPO / m["path"]
        if dest.exists() and sha256(dest) == m["sha256"]:
            continue
        cloud.fetch(prefix + m["key"], dest)
        if sha256(dest) != m["sha256"]:
            dest.unlink()
            raise Failed("bad_models", f"model {m['key']} in the bucket is not the one this engine was built for")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def footage_files(rec: Path) -> set[str]:
    try:
        m = json.loads((rec / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {s["objectKey"] for d in m.get("devices", []) for s in d.get("segments", [])}


def pictures(node) -> set[str]:
    """Every picture a questions.json names."""
    found = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k in PICTURE_KEYS and isinstance(v, str):
                found.add(v)
            else:
                found |= pictures(v)
    elif isinstance(node, list):
        for v in node:
            found |= pictures(v)
    return found


def publish_questions(cloud: Cloud, rec: Path) -> None:
    """questions.json and its pictures to the output prefix. The pictures
    first, so the questions are never there without them."""
    q = json.loads((rec / "engine" / "questions.json").read_text())
    q["recording"] = "."  # the machine's folder means nothing to anyone else
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    rels = sorted(p for p in pictures(q) if not p.startswith("/") and ".." not in Path(p).parts and (rec / p).is_file())

    def one(rel):
        cloud.put(rec / rel, f"{cloud.out}questions/{rel}", ContentType=types.get(Path(rel).suffix.lower(), "application/octet-stream"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, rels))
    cloud.s3.put_object(Bucket=cloud.bucket, Key=f"{cloud.out}questions/questions.json",
                        Body=json.dumps(q).encode(), ContentType="application/json")
    log(f"questions published: pending {q.get('pending')}, {len(rels)} pictures")


def publish_logs(cloud: Cloud, rec: Path) -> None:
    logs = rec / "engine" / "logs"
    for f in sorted(logs.glob("*.log")) if logs.is_dir() else []:
        try:
            cloud.put(f, f"{cloud.out}logs/{f.name}", ContentType="text/plain")
        except Exception as e:
            log(f"could not save {f.name}: {e}")


_stopping = threading.Event()
_engine: subprocess.Popen | None = None


def _on_signal(signum, frame) -> None:
    _stopping.set()
    p = _engine
    if p is not None and p.poll() is None:
        p.send_signal(signal.SIGTERM)  # engine.py stops its step and exits 130


def engine_cmd() -> list[str]:
    # A test names its own stand-in.
    if os.environ.get("OSB_ENGINE_CMD"):
        return json.loads(os.environ["OSB_ENGINE_CMD"])
    return [*accel.python_cmd(), str(HERE / "engine.py")]


def gpu_usable() -> bool:
    """Whether PyTorch, in the engine's own Python, can use the GPU: the
    driver may be too old for the CUDA the wheels were built with, and the
    pipeline would then quietly run on the CPU for days."""
    try:
        out = subprocess.run([*accel.python_cmd()[:1], "-c", "import torch; print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    log(f"torch.cuda.is_available(): {out.stdout.strip() or out.stderr.strip()[-300:]}")
    return out.stdout.strip() == "True"


def chunk_details(rec: Path) -> list | None:
    """Each chunk's stages and times, as engine.py status reports them."""
    try:
        out = subprocess.run([*engine_cmd(), "status", str(rec.parent / "job.json")], capture_output=True, text=True,
                             timeout=300, cwd=rec.parent).stdout
        for line in out.splitlines():
            if line.startswith("{") and json.loads(line).get("event") == "status":
                return json.loads(line).get("chunkDetails")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None


def publish_profile(cloud: Cloud, profile: Profile, exit_code: int | None) -> None:
    try:
        profile.stop()
        profile.mark("end")
        base = f"{cloud.out}profile/{profile.command}"
        cloud.s3.put_object(Bucket=cloud.bucket, Key=base + ".json",
                            Body=json.dumps(profile.summary(exit_code), indent=1).encode(), ContentType="application/json")
        cloud.s3.put_object(Bucket=cloud.bucket, Key=base + "-samples.jsonl",
                            Body=profile.samples_jsonl().encode(), ContentType="application/x-ndjson")
        log(f"profile: {json.dumps(profile.summary(exit_code)['usage'])}")
    except Exception as e:  # never the reason a job fails
        log(f"could not save the profile: {e}")


def run_engine(cloud: Cloud, rec: Path, args: list[str], footage: set[str], profile: Profile | None = None) -> tuple[int, dict]:
    """engine.py with `args`, its events read as they come: progress to the
    job record, the folder to the work prefix as steps and chunks finish.
    Returns its exit code and the last event of each kind."""
    global _engine
    last: dict[str, dict] = {}
    done_steps = 0
    last_sync = time.time()
    _engine = subprocess.Popen([*engine_cmd(), *args], stdout=subprocess.PIPE, text=True, cwd=rec.parent)
    for line in _engine.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        print(line, flush=True)  # the job's log keeps the engine's own words
        last[e.get("event", "")] = e
        if profile:
            profile.event(e)
        if e.get("event") == "step":
            if e.get("status") in ("done", "skipped"):
                done_steps += 1
                if e.get("status") == "done":
                    cloud.save_work(rec, footage)
                    last_sync = time.time()
            if e.get("step") in PREPARE_STEPS:
                cloud.progress(e["step"], min(0.99, done_steps / len(PREPARE_STEPS)), None, force=e.get("status") == "started")
        elif e.get("event") == "progress":
            cloud.progress(e.get("stage", "game"), float(e.get("fraction", 0)), e.get("etaS"))
            if time.time() - last_sync > SYNC_EVERY_S:
                cloud.save_work(rec, footage)
                last_sync = time.time()
    rc = _engine.wait()
    _engine = None
    return rc, last


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OSB_COMMAND", "")
    if command not in ("prepare", "refit", "run"):
        print(f"cloud.py: unknown command {command!r}: use prepare, refit or run", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    cloud = None
    rec = None
    profile = Profile(command, engine_version())
    exit_code: int | None = None
    try:
        cloud = Cloud()
        profile.start()
        if os.environ.get("OSB_REQUIRE_GPU") == "1":
            if accel.platform() != "nvidia":
                raise Failed("no_gpu", "this machine has no NVIDIA GPU the engine can use (accel.py)")
            if not gpu_usable():
                raise Failed("no_gpu", "PyTorch cannot use this machine's GPU (is the NVIDIA driver too old for its CUDA build?)")
        profile.mark("gpu_checked")
        log(f"{command} for job {cloud.job_id} on {accel.platform()}; {engine_version()}")
        rec = Path(env("OSB_SCRATCH", "/scratch")) / cloud.job_id / "rec"
        rec.mkdir(parents=True, exist_ok=True)
        job = rec.parent / "job.json"
        try:
            extra = json.loads(os.environ.get("OSB_JOB_OPTIONS") or "{}")
        except json.JSONDecodeError as e:
            raise Failed("bad_job", f"OSB_JOB_OPTIONS is not JSON: {e}")
        job.write_text(json.dumps({"schemaVersion": 1, "recording": "rec",
                                   **{k: extra[k] for k in ("options", "output") if k in extra}}))
        cloud.progress("download", 0.0, None, force=True)
        cloud.restore_work(rec)
        profile.mark("work_restored")
        answers = cloud.get_json(cloud.out + "answers.json")
        args = [command if command == "run" else "prepare", str(job)]
        if answers is not None:
            (rec / "engine").mkdir(exist_ok=True)
            (rec / "engine" / "answers.json").write_text(json.dumps(answers))
            # run applies them first too: an answer that needed no refit (the
            # pitch approved, the kit groups settled) reaches the engine here.
            args += ["--answers", str(rec / "engine" / "answers.json")]

        rc, last = None, {}
        if command == "refit" and (rec / "engine" / "questions.json").exists():
            # Applying an answer reads the pictures prepare left, not the
            # video: try without 40 GB of footage first.
            manifest = cloud.get_json(cloud.footage + "manifest.json")
            if manifest is None:
                raise Failed("footage_missing", "the recording's footage is no longer there")
            (rec / "manifest.json").write_text(json.dumps(manifest))
            profile.mark("engine_started")
            rc, last = run_engine(cloud, rec, [*args, "--no-footage"], set(), profile)
            if rc not in (0, EXIT_STOPPED):
                log(f"refit without the footage failed ({(last.get('error') or {}).get('code')}): trying with it")
                rc = None
        if rc is None:
            if command != "refit":
                fetch_models(cloud)
                profile.mark("models_fetched")
            nbytes, seconds = cloud.fetch_footage(rec)
            if nbytes:
                profile.transfer("footage", nbytes, seconds)
            profile.mark("footage_fetched")
            profile.mark("engine_started")
            rc, last = run_engine(cloud, rec, args, footage_files(rec), profile)
        profile.mark("engine_finished")
        if command == "run":
            profile.chunks = chunk_details(rec)

        cloud.save_work(rec, footage_files(rec))
        profile.mark("work_saved")
        if _stopping.is_set() or rc == EXIT_STOPPED:
            log("stopped; the work so far is saved")
            exit_code = EXIT_STOPPED
            return EXIT_STOPPED
        if command == "run" and rc == 3 and (rec / "engine" / "questions.json").exists():
            # The engine would not use an answer (engine.py: needs_answers) and
            # has written its questions again: the user is asked, not failed.
            publish_questions(cloud, rec)
            exit_code = 0
            return 0
        if rc != 0:
            e = last.get("error") or {}
            publish_logs(cloud, rec)
            raise Failed(e.get("code") or "engine_failed", e.get("message") or f"the engine stopped (exit {rc})")
        if command == "run":
            game = rec / "fullgame" / "game.mp4"
            if not game.exists():
                raise Failed("engine_no_result", "the engine finished without a game.mp4")
            cloud.progress("upload", 0.99, None, force=True)
            cloud.s3.put_object(Bucket=cloud.bucket, Key=cloud.out + "result.json",
                                Body=json.dumps(last.get("done") or {}).encode(), ContentType="application/json")
            t = time.time()
            cloud.put(game, cloud.out + "game.mp4", ContentType="video/mp4")
            profile.transfer("video", game.stat().st_size, time.time() - t)
            cloud.progress("upload", 1.0, 0, force=True)
        else:
            publish_questions(cloud, rec)
        profile.mark("published")
        exit_code = 0
        return 0
    except Failed as f:
        print(json.dumps({"event": "error", "code": f.code, "message": f.message}), flush=True)
        if cloud:
            cloud.error(f.code, f.message)
        exit_code = 1
        return 1
    except Exception as e:  # S3 or the disk, not the engine
        log(f"crashed: {e!r}")
        if cloud:
            cloud.error("cloud_failed", "processing could not read or write its files")
            if rec is not None:
                try:
                    cloud.save_work(rec, footage_files(rec))
                except Exception:
                    pass
        raise
    finally:
        if cloud:
            publish_profile(cloud, profile, exit_code)
        # The instance is usually thrown away; a reused one gets its disk back.
        if rec is not None and not _stopping.is_set() and os.environ.get("OSB_KEEP_SCRATCH") != "1":
            shutil.rmtree(rec.parent, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
