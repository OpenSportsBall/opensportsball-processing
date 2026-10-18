#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26"]
# ///
"""Renders a whole game with the follow cam, in chunks, overnight.

    uv run processing/spikes/fullgame.py footage/rec3 [--chunk 600] [--max-chunks N] [--out fullgame]
                                                      [--features goalkick,goalframe,unsure,ball,goalin,ballhold]

Each chunk of the game (on Main's clock from segment 0; the readers run
on across segment switches) goes through ballwin.py (ball candidates,
~3x real time on an M-series Mac) and follow.py (player detection, path,
render, ~2.5x real time). The two run as a pipeline: the ball stage works
one chunk ahead of the follow cam, so the whole game takes about as long
as the ball stage alone. A chunk whose piece exists is skipped, so the
run can be resumed. At the end the pieces are concatenated and Main's
audio is muxed in: <recording>/<out>/game.mp4.

Logs: <recording>/<out>/logs/.

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
CKPT = "balltrain/checkpoint_best_total.pth"


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
    ap.add_argument("--out", default="fullgame")
    ap.add_argument("--features", default="goalkick,goalframe,unsure,ball,goalin,ballhold,roles,kickoff")
    ap.add_argument("--roles-fps", type=float, default=4.0, help="roles.py rate (people move slowly; 4 is plenty)")
    ap.add_argument("--teams", default="", help="teams.py set for roles.py (the people stage); empty: no roles")
    a = ap.parse_args()
    rec = a.rec
    manifest = json.loads((rec / "manifest.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    total = sum(s["durationMs"] for s in devs["main"]["segments"]) / 1000
    chunks = []
    t = 0.0
    while t < total - 1:
        chunks.append((t, min(a.chunk, total - t)))
        t += a.chunk
    if a.max_chunks:
        chunks = chunks[:a.max_chunks]
    out = rec / a.out
    (out / "pieces").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    tag = "play_" + "_".join(f for f in a.features.split(",") if "=" not in f)
    print(f"{len(chunks)} chunks of {a.chunk:g} s ({total / 60:.1f} min of game) -> {out}", flush=True)

    def piece(k):
        return out / "pieces" / f"piece_{k:02d}.mp4"

    def cands(start, seconds):
        return rec / "detect" / f"ball_0_{start:g}_{seconds:g}.win.json"

    def roles_file(start, seconds):
        return rec / "detect" / f"roles_0_{start:g}_{seconds:g}.json"

    if "roles" in a.features and not a.teams:
        sys.exit("--teams <set> is needed for the roles feature (teams.py), or drop roles from --features")

    failed = []

    def ball_worker():
        for k, (start, seconds) in enumerate(chunks):
            if piece(k).exists() or cands(start, seconds).exists():
                continue
            t0 = time.time()
            rc = run(["uv", "run", "-q", str(HERE / "ballwin.py"), str(rec), "0", f"{start:g}", f"{seconds:g}",
                      str(rec / CKPT), "10"], out / "logs" / f"ball_{k:02d}.log", done_file=cands(start, seconds))
            if rc and cands(start, seconds).exists():
                rc = 0  # the candidates are there; whatever happened after doesn't matter
            if rc == 0 and a.teams and not roles_file(start, seconds).exists():
                rc = run(["uv", "run", "-q", str(HERE / "roles.py"), str(rec), "0", f"{start:g}", f"{seconds:g}",
                          a.teams, f"{a.roles_fps:g}"], out / "logs" / f"roles_{k:02d}.log", done_file=roles_file(start, seconds))
                if rc and roles_file(start, seconds).exists():
                    rc = 0
            print(f"ball chunk {k}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {(time.time() - t0) / 60:.1f} min", flush=True)
            if rc:
                failed.append(("ball", k))

    def follow_worker():
        for k, (start, seconds) in enumerate(chunks):
            if piece(k).exists():
                continue
            while not cands(start, seconds).exists() or (a.teams and not roles_file(start, seconds).exists()):
                if ("ball", k) in failed:
                    print(f"follow chunk {k}: skipped, no candidates", flush=True)
                    break
                time.sleep(10)
            else:
                t0 = time.time()
                rc = run(["uv", "run", "-q", str(HERE / "follow.py"), str(rec), "0", f"{start:g}", f"{seconds:g}",
                          "play", a.features], out / "logs" / f"follow_{k:02d}.log")
                src = rec / "follow" / f"follow_{tag}.mp4"
                if rc == 0 and src.exists():
                    src.rename(piece(k))
                    print(f"follow chunk {k}: ok in {(time.time() - t0) / 60:.1f} min -> {piece(k).name}", flush=True)
                else:
                    failed.append(("follow", k))
                    print(f"follow chunk {k}: FAILED rc={rc}", flush=True)

    tb = threading.Thread(target=ball_worker)
    tf = threading.Thread(target=follow_worker)
    t0 = time.time()
    tb.start(); tf.start(); tb.join(); tf.join()
    if not all(piece(k).exists() for k in range(len(chunks))):
        print(f"not all pieces made (failures: {failed}); fix and rerun to resume", flush=True)
        sys.exit(1)

    # Concatenate the pieces, then mux Main's audio (its segments are back to back).
    lst = out / "pieces.txt"
    lst.write_text("".join(f"file '{piece(k).resolve()}'\n" for k in range(len(chunks))))
    log = out / "logs" / "concat.log"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         str(out / "video.mp4")], log)
    alst = out / "audio.txt"
    alst.write_text("".join(f"file '{(rec / s['objectKey']).resolve()}'\n" for s in devs["main"]["segments"]))
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(alst), "-vn", "-c:a", "copy",
         str(out / "audio.m4a")], log)
    rc = run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "video.mp4"), "-i", str(out / "audio.m4a"),
              "-c", "copy", "-shortest", str(out / "game.mp4")], log)
    print(f"{'done' if rc == 0 else 'mux FAILED'}: {out / 'game.mp4'} in {(time.time() - t0) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
