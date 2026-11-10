#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26"]
# ///
"""Renders a whole game with the follow cam, in chunks, overnight.

    uv run processing/spikes/fullgame.py footage/rec3 [--start 0 | --start-segment N] [--chunk 600] [--max-chunks N] [--out fullgame]
                                                      [--features goalkick,goalframe,unsure,ball,goalin,ballhold] [--height 1440 | 1080]

Each chunk of the game (on Main's clock from segment 0; the readers run
on across segment switches) goes through ballwin.py (ball candidates,
~3x real time on an M-series Mac) and follow.py (player detection, path,
render, ~2.5x real time). The two run as a pipeline: the ball stage works
one chunk ahead of the follow cam, so the whole game takes about as long
as the ball stage alone. A chunk whose piece exists is skipped, so the
run can be resumed. At the end the pieces are concatenated and Main's
audio is muxed in: <recording>/<out>/game.mp4.

game.mp4 is scaled up from the pieces' 1080p to 1440p by default (--height
1080 keeps it as rendered, with no re-encode). Upscaling adds no detail, but
YouTube serves an upload of 1440p or more with VP9 and a much higher
bitrate, and on rec7 that looked clearly better than the 1080p upload's
AV1 stream. The scale and encode add ~26 min for a 95 min game and roughly
double the file (~22 GB).

Logs: <recording>/<out>/logs/, with each chunk's camera log and path
(camlog_NN.txt, path_NN.json; score a path with playeval.py).

Keep the Mac awake for the run: it idle-sleeps otherwise (the first run
lost a night to that). `caffeinate -i -w <pid of this script's python>`,
or start it as `caffeinate -i uv run ...`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import accel  # noqa: E402
DETECTOR = HERE.parents[1] / "footage" / "models" / "ball_detector.pth"  # the standard ball detector (README there)
PIECE_H = 1080       # follow.py's render height (follow.OUT_H; 16:9)
UPLOAD_BITRATE = {1440: "45M"}  # hardware H.264 lands at ~2/3 of this (~31 Mbit/s)


def run(cmd: list[str], log: Path, done_file: Path | None = None, grace_s: float = 180.0) -> int:
    """Runs `cmd`, logging to `log`. If `done_file` appears and the process
    then lingers for `grace_s` (teardown hangs have happened), it is killed
    and counted as done."""
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        seen = None
        while proc.poll() is None:
            time.sleep(5)
            if done_file is not None and done_file.exists():
                seen = seen or time.time()
                if time.time() - seen > grace_s:
                    f.write(f"\n[fullgame] {done_file.name} written but the process lingered {grace_s:.0f} s: killed\n")
                    proc.kill()
                    proc.wait()
                    return 0
        return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", type=Path)
    ap.add_argument("--chunk", type=float, default=600.0)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0,
                    help="s on Main's clock to start from (skip a warm-up before kickoff); audio is cut to match")
    ap.add_argument("--start-segment", type=int, default=0,
                    help="start at Main's segment N instead (skip N segments of warm-up); added to --start")
    ap.add_argument("--out", default="fullgame")
    ap.add_argument("--detector", type=Path, default=DETECTOR, help="ball detector checkpoint (default: the standard one)")
    ap.add_argument("--features", default="goalkick,goalframe,unsure,ball,goalin,ballhold,roles,kickoff")
    ap.add_argument("--roles-fps", type=float, default=4.0, help="roles.py rate (people move slowly; 4 is plenty)")
    ap.add_argument("--teams", default="", help="teams.py set for roles.py (the people stage); empty: no roles")
    ap.add_argument("--events", type=Path, default=None,
                    help="also append progress as JSON lines here (engine.py reads them; see schemas/engine-events)")
    ap.add_argument("--height", type=int, default=1440, choices=(PIECE_H, *UPLOAD_BITRATE),
                    help=f"game.mp4's height: {PIECE_H} keeps the render as is; more scales it up for YouTube (default 1440)")
    a = ap.parse_args()
    rec = a.rec
    manifest = json.loads((rec / "manifest.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    total = sum(s["durationMs"] for s in devs["main"]["segments"]) / 1000
    a.start += sum(s["durationMs"] for s in devs["main"]["segments"][:a.start_segment]) / 1000
    chunks = []
    t = a.start
    while t < total - 1:
        # The last chunk stops 1 s short of the recording's end: the decoder
        # has hung at the very end of the final segment file.
        chunks.append((t, min(a.chunk, total - t - 1.0)))
        t += a.chunk
    if a.max_chunks:
        chunks = chunks[:a.max_chunks]
    out = rec / a.out
    (out / "pieces").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (rec / "detect").mkdir(exist_ok=True)
    tag = "play_" + "_".join(f for f in a.features.split(",") if "=" not in f)
    print(f"{len(chunks)} chunks of {a.chunk:g} s ({total / 60:.1f} min of game) -> {out}", flush=True)
    ev_lock = threading.Lock()

    def event(**e) -> None:
        """One progress event for engine.py: chunk-level, so cheap."""
        if a.events is None:
            return
        e = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **e}
        with ev_lock, a.events.open("a") as f:
            f.write(json.dumps(e) + "\n")

    event(event="plan", chunks=len(chunks), chunkS=a.chunk, gameS=round(total - a.start, 1),
          done=[k for k in range(len(chunks)) if (out / "pieces" / f"piece_{k:02d}.mp4").exists()])

    def piece(k):
        return out / "pieces" / f"piece_{k:02d}.mp4"

    def cands(start, seconds):
        return rec / "detect" / f"ball_0_{start:g}_{seconds:g}.win.json"

    def roles_file(start, seconds):
        return rec / "detect" / f"roles_0_{start:g}_{seconds:g}.json"

    if "roles" in a.features and not a.teams:
        sys.exit("--teams <set> is needed for the roles feature (teams.py), or drop roles from --features")

    failed = []
    # Stop at the first failed chunk: the game can't be muxed without it, a
    # rerun resumes from the pieces and candidates already made, and a
    # failure is usually the same for every chunk (a fresh recording once
    # had every ball chunk fail after 20 min of work each).
    stop = threading.Event()

    def ball_worker():
        for k, (start, seconds) in enumerate(chunks):
            if stop.is_set():
                return
            if piece(k).exists():
                continue
            t0, rc = time.time(), 0
            if not cands(start, seconds).exists():  # the candidates may be there from an earlier run
                rc = run([*accel.python_cmd(), str(HERE / "ballwin.py"), str(rec), "0", f"{start:g}", f"{seconds:g}",
                          str(a.detector), "10"], out / "logs" / f"ball_{k:02d}.log", done_file=cands(start, seconds))
                if rc and cands(start, seconds).exists():
                    rc = 0  # the candidates are there; whatever happened after doesn't matter
            if rc == 0 and a.teams and not roles_file(start, seconds).exists():
                rc = run([*accel.python_cmd(), str(HERE / "roles.py"), str(rec), "0", f"{start:g}", f"{seconds:g}",
                          a.teams, f"{a.roles_fps:g}"], out / "logs" / f"roles_{k:02d}.log", done_file=roles_file(start, seconds))
                if rc and roles_file(start, seconds).exists():
                    rc = 0
            print(f"ball chunk {k}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {(time.time() - t0) / 60:.1f} min", flush=True)
            event(event="chunk", stage="ball", chunk=k, ok=rc == 0, seconds=round(time.time() - t0, 1),
                  log=str(out / "logs" / f"ball_{k:02d}.log"))
            if rc:
                failed.append(("ball", k))
                stop.set()

    def follow_worker():
        for k, (start, seconds) in enumerate(chunks):
            if piece(k).exists():
                continue
            while not cands(start, seconds).exists() or (a.teams and not roles_file(start, seconds).exists()):
                if stop.is_set():
                    print(f"follow chunk {k}: stopped, an earlier chunk failed", flush=True)
                    return
                time.sleep(10)
            else:
                t0 = time.time()
                rc = run([*accel.python_cmd(), str(HERE / "follow.py"), str(rec), "0", f"{start:g}", f"{seconds:g}",
                          "play", a.features], out / "logs" / f"follow_{k:02d}.log")
                src = rec / "follow" / f"follow_{tag}.mp4"
                if rc == 0 and src.exists():
                    src.rename(piece(k))
                    cl = rec / "follow" / "camlog.txt"
                    if cl.exists():
                        cl.rename(out / "logs" / f"camlog_{k:02d}.txt")
                    # The camera path, so the chunk can be scored (playeval.py score) without a rerun.
                    pj = rec / "follow" / f"path_{tag}.json"
                    if pj.exists():
                        pj.rename(out / "logs" / f"path_{k:02d}.json")
                    print(f"follow chunk {k}: ok in {(time.time() - t0) / 60:.1f} min -> {piece(k).name}", flush=True)
                    event(event="chunk", stage="follow", chunk=k, ok=True, seconds=round(time.time() - t0, 1))
                else:
                    failed.append(("follow", k))
                    stop.set()
                    print(f"follow chunk {k}: FAILED rc={rc}", flush=True)
                    event(event="chunk", stage="follow", chunk=k, ok=False, seconds=round(time.time() - t0, 1),
                          log=str(out / "logs" / f"follow_{k:02d}.log"))

    tb = threading.Thread(target=ball_worker)
    tf = threading.Thread(target=follow_worker)
    t0 = time.time()
    tb.start(); tf.start(); tb.join(); tf.join()
    if not all(piece(k).exists() for k in range(len(chunks))):
        print(f"not all pieces made (failures: {failed}); fix and rerun to resume", flush=True)
        event(event="failed", failures=[{"stage": s, "chunk": k} for s, k in failed])
        sys.exit(1)
    event(event="mux")

    # The camera log for the whole game: the chunks' logs in order.
    logs = sorted((out / "logs").glob("camlog_*.txt"))
    if logs:
        (out / "camlog.txt").write_text("".join(l.read_text() for l in logs))
    # Concatenate the pieces, then mux Main's audio (its segments are back to back).
    lst = out / "pieces.txt"
    lst.write_text("".join(f"file '{piece(k).resolve()}'\n" for k in range(len(chunks))))
    log = out / "logs" / "concat.log"
    run([accel.ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         str(out / "video.mp4")], log)
    alst = out / "audio.txt"
    alst.write_text("".join(f"file '{(rec / s['objectKey']).resolve()}'\n" for s in devs["main"]["segments"]))
    run([accel.ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(alst), "-vn", "-c:a", "copy",
         str(out / "audio.m4a")], log)
    if a.height == PIECE_H:
        video = ["-c", "copy"]
    else:
        br = UPLOAD_BITRATE[a.height]
        video = ["-map", "0:v", "-map", "1:a", "-vf", f"scale={a.height * 16 // 9}:{a.height}:flags=lanczos",
                 *accel.h264_args(br, br), "-profile:v", "high", "-pix_fmt", "yuv420p",
                 "-c:a", "copy", "-movflags", "+faststart"]
    rc = run([accel.ffmpeg(), "-v", "error", "-y", "-i", str(out / "video.mp4"), "-ss", f"{a.start:g}", "-i", str(out / "audio.m4a"),
              *video, "-shortest", str(out / "game.mp4")], log)
    print(f"{'done' if rc == 0 else 'mux FAILED'}: {out / 'game.mp4'} ({a.height}p) in {(time.time() - t0) / 3600:.1f} h",
          flush=True)
    event(event="done" if rc == 0 else "failed", output=str(out / "game.mp4"), seconds=round(time.time() - t0, 1),
          **({} if rc == 0 else {"failures": [{"stage": "mux", "chunk": None}]}))
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
