#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""Backs footage/ up to S3, checks the backup, and restores it.

    uv run processing/spikes/footagestore.py check
    uv run processing/spikes/footagestore.py backup [--dry-run]
    uv run processing/spikes/footagestore.py restore [--dry-run]

The layout (docs/DATA.md) in the footage bucket (infra/cdk-outputs.json):

    accounts/...  the phones' uploads: raw video and manifests (the app's)
    bench/        = footage/bench/, the bench store
    data/         = footage/ without bench/, video or symlinks

bench/ is copied whole, video included (the benches need it).

"backup" uploads what is new or changed and never deletes anything in S3.
Raw video (*.mp4 under a recording's <deviceId>/) is left out of data/
when every segment of its recording is in accounts/ at the manifest's
size; a recording that is not (deleted from the website, or never
uploaded) gets its video copied to data/<rec>/<deviceId>/. Other video
(renders, previews) and audio are left out: they are made from the raw
video. Symlinks (variants of a recording sharing its video or caches) are
listed in footage/symlinks.json, which is backed up and which "restore"
recreates them from.

"check" lists, per file in scope, what is missing from S3 or differs in
size, and for every recording where its raw video is: accounts/, data/,
or nowhere but this machine. It exits 1 if anything is not backed up.

AWS profile: OSB_AWS_PROFILE (default sbv).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOOTAGE = ROOT / "footage"
PROFILE = os.environ.get("OSB_AWS_PROFILE", "sbv")
DEVICE_DIR = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SKIP_SUFFIXES = {".mp4", ".m4a", ".mov"}  # video: raw video is handled per recording
SKIP_NAMES = {".DS_Store"}


def bucket() -> str:
    return json.loads((ROOT / "infra" / "cdk-outputs.json").read_text())["OpenSportsBall"]["FootageBucket"]


def aws(*args: str, capture: bool = False) -> str:
    cmd = ["aws", "--profile", PROFILE, *args]
    if capture:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    subprocess.run(cmd, check=True)
    return ""


def listing(prefix: str) -> dict[str, int]:
    """S3 key -> size under `prefix`."""
    out = {}
    r = subprocess.run(["aws", "--profile", PROFILE, "s3", "ls", "--recursive", f"s3://{bucket()}/{prefix}"],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1) or (r.returncode == 1 and r.stderr.strip()):  # 1 with no error: empty prefix
        sys.exit(f"aws s3 ls {prefix}: {r.stderr.strip()}")
    for line in r.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            out[parts[3]] = int(parts[2])
    return out


def in_scope(base: Path, skip_top: set[str], video: bool) -> dict[str, int]:
    """Relative path -> size of every file under `base` that the backup
    covers: no symlinks (files or directories), and unless `video` (the
    bench store keeps its video) no video or device dirs."""
    out = {}
    for root, dirs, files in os.walk(base):
        rel_root = Path(root).relative_to(base)
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))
                   and (video or not DEVICE_DIR.match(d)) and not (rel_root == Path(".") and d in skip_top)]
        for f in files:
            p = Path(root) / f
            if p.is_symlink() or f in SKIP_NAMES or (not video and p.suffix.lower() in SKIP_SUFFIXES):
                continue
            out[str(p.relative_to(base))] = p.stat().st_size
    return out


def symlinks() -> dict[str, str]:
    """Every symlink under footage/ -> its target, relative to the link's
    directory when the target is inside footage/."""
    out = {}
    for root, dirs, files in os.walk(FOOTAGE):
        for name in dirs + files:
            p = Path(root) / name
            if p.is_symlink():
                target = Path(os.readlink(p))
                full = (p.parent / target).resolve() if not target.is_absolute() else target
                if full.is_relative_to(FOOTAGE):
                    target = Path(os.path.relpath(full, p.parent))
                out[str(p.relative_to(FOOTAGE))] = str(target)
    return dict(sorted(out.items()))


def recordings() -> list[tuple[str, dict, list[str]]]:
    """(recording dir, manifest, device dirs held here as real files) for
    every recording under footage/ whose manifest is a real file."""
    out = []
    for m in sorted(FOOTAGE.glob("*/manifest.json")):
        if m.is_symlink() or m.parent.name == "bench":
            continue
        manifest = json.loads(m.read_text())
        devs = [d["deviceId"] for d in manifest["devices"]
                if (m.parent / d["deviceId"]).is_dir() and not (m.parent / d["deviceId"]).is_symlink()]
        out.append((m.parent.name, manifest, devs))
    return out


def video_status(accounts: dict[str, int], data: dict[str, int]) -> list[tuple[str, str, str, int]]:
    """(rec, short game id, where its raw video is, segment count) per
    recording: "accounts", "data", or "LOCAL ONLY" / "MISSING"."""
    by_session = {}
    for key, size in accounts.items():
        m = re.search(r"/recordings/([^/]+)/(.+)$", key)
        if m:
            by_session[(m.group(1), m.group(2))] = size
    out = []
    for rec, manifest, devs in recordings():
        segs = [(d["deviceId"], s) for d in manifest["devices"] for s in d["segments"]]
        session = manifest["sessionId"]
        if all(by_session.get((session, s["objectKey"])) == s["bytes"] for _, s in segs):
            where = "accounts"
        elif all(data.get(f"data/{rec}/{s['objectKey']}") == s["bytes"] for _, s in segs):
            where = "data"
        elif devs:
            where = "LOCAL ONLY"
        else:
            where = "MISSING"
        out.append((rec, manifest["gameId"][:8], where, len(segs)))
    return out


def check() -> int:
    data, bench, accounts = listing("data/"), listing("bench/"), listing("accounts/")
    bad = 0
    for label, local, remote, prefix in (
            ("data", in_scope(FOOTAGE, {"bench"}, False), data, "data/"),
            ("bench", in_scope(FOOTAGE / "bench", set(), True), bench, "bench/")):
        missing = [p for p, s in local.items() if remote.get(prefix + p) is None]
        differ = [p for p, s in local.items() if remote.get(prefix + p) not in (None, s)]
        size = sum(local.values()) / 1e9
        print(f"{label}/: {len(local)} files ({size:.1f} GB) here, {len(missing)} missing from S3, "
              f"{len(differ)} differ in size")
        for p in (missing + differ)[:10]:
            print(f"  {p}")
        bad += len(missing) + len(differ)
    print("raw video:")
    for rec, game, where, n in video_status(accounts, data):
        print(f"  {rec:<12} game {game}  {n:>3} segments  {where}")
        bad += where in ("LOCAL ONLY", "MISSING")
    links = symlinks()
    saved = json.loads((FOOTAGE / "symlinks.json").read_text()) if (FOOTAGE / "symlinks.json").exists() else {}
    if links != saved:
        print(f"symlinks: footage/symlinks.json is out of date ({len(links)} links here, {len(saved)} listed)")
        bad += 1
    print("all backed up" if not bad else f"{bad} things not backed up: run backup")
    return 1 if bad else 0


def backup(dry: list[str]) -> None:
    b = bucket()
    (FOOTAGE / "symlinks.json").write_text(json.dumps(symlinks(), indent=1) + "\n")
    skip = [a for s in SKIP_SUFFIXES for a in ("--exclude", f"*{s}")] + \
           [a for n in SKIP_NAMES for a in ("--exclude", f"*{n}")]
    aws("s3", "sync", "--no-follow-symlinks", "--only-show-errors", *dry, "--exclude", "bench/*", *skip,
        str(FOOTAGE), f"s3://{b}/data/")
    aws("s3", "sync", "--no-follow-symlinks", "--only-show-errors", *dry, "--exclude", "*.DS_Store",
        str(FOOTAGE / "bench"), f"s3://{b}/bench/")
    for rec, game, where, _ in video_status(listing("accounts/"), listing("data/")):
        if where == "LOCAL ONLY":
            for dev in next(d for r, _, d in recordings() if r == rec):
                print(f"{rec}: raw video not in accounts/, copying {dev} to data/", flush=True)
                aws("s3", "sync", "--no-follow-symlinks", "--only-show-errors", *dry,
                    str(FOOTAGE / rec / dev), f"s3://{b}/data/{rec}/{dev}/")


def restore(dry: list[str]) -> None:
    b = bucket()
    aws("s3", "sync", "--only-show-errors", *dry, f"s3://{b}/data/", str(FOOTAGE))
    aws("s3", "sync", "--only-show-errors", *dry, f"s3://{b}/bench/", str(FOOTAGE / "bench"))
    links = FOOTAGE / "symlinks.json"
    for path, target in (json.loads(links.read_text()).items() if links.exists() else []):
        p = FOOTAGE / path
        if not p.exists() and not p.is_symlink():
            print(f"link {path} -> {target}")
            if not dry:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.symlink_to(target)
    print("Raw video in accounts/ is not restored: download a recording's segments with its manifest "
          "(docs/DATA.md).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    dry = ["--dryrun"] if "--dry-run" in sys.argv else []
    if cmd == "check":
        sys.exit(check())
    elif cmd == "backup":
        backup(dry)
        if not dry:
            sys.exit(check())
    elif cmd == "restore":
        restore(dry)
    else:
        sys.exit(__doc__)
