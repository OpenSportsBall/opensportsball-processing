#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["huggingface_hub>=2.0"]
# ///
"""Shares the benchmark data with contributors through a gated Hugging Face
dataset: anyone can read its card, but the files need a maintainer's
approval of an access request.

    uv run processing/spikes/sharedata.py fetch [--no-cache] [--no-video]   # contributors
    uv run processing/spikes/sharedata.py list [--remote]                   # what upload would send (and delete)
    uv run processing/spikes/sharedata.py upload                            # maintainers

The dataset mirrors footage/ path for path, so `fetch` drops everything
where the tools already look:

  bench/         the bench store (docs/DATA.md): what the benchmarks read, video included, and
                 every ball label set (LABELS.md there says which may be trained on)
  bench-cache/   what is made from the store and only saves time (pass A caches, ball
                 candidates, roles): `fetch --no-cache` leaves it out
  models/        the standard ball detector and judge

Never shared: what runs write (bench-out/) and training products (train/).

Before `fetch`, request access on the dataset's page and log in with
`hf auth login` (a read token is enough). `upload` needs a write token
for the dataset's organisation. It creates the repository private and
gated (manual approval) if it does not exist; making it visible, so
people can request access, is a deliberate step on its settings page.

Never shared: the app's upload state (`upload.json`: the account id and
S3 upload ids), which `upload` also removes from the dataset if it is
there. And `upload` stops, naming the files, when a text file it would
send holds a home-directory path (/Users/<name>/...), an account id or an
accounts/<id> key: make the path relative or leave the file out, then run
it again (`list` shows the same). Model files (.pth, .pt) aren't looked
into: a checkpoint keeps the paths it was trained with.

OSB_HF_DATASET overrides the repository (default below). See
docs/DATA.md, "Sharing with contributors".
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = os.environ.get("OSB_HF_DATASET", "OpenSportsBall/bench")
FOOTAGE = Path("footage")
CARD = Path(__file__).resolve().parents[2] / "docs" / "DATASET_CARD.md"

# Recordings outside the bench store whose pitch maps and labels we share.
# None since 2026-11-17: every shared recording's label sets, pitch map and
# must-see sets are in the store (bench/<rec>/).
ANNOTATED: dict[str, list[str]] = {}
SHARED = ["bench/**", "bench-cache/**", "models/**"]
VIDEO = ["bench/**/*.mp4"]

# The app's upload state: the account id, S3 multipart upload ids and part ETags.
NEVER = ["upload.json", "**/upload.json"]
# A play set's paths/ are bench output (playbench.py rewrites them), not
# annotations (for a recording in the store they are in bench-out/, never shared).
IGNORE = [".DS_Store", "**/.DS_Store", "**/__pycache__/**", ".cache/**", "rec*/playeval/*/paths/**", *NEVER]

# What a shared text file must not hold (private_details).
PRIVATE = {"a home-directory path": re.compile(r"/(?:Users|home)/[^/\s\"']+/"),
           "an account id": re.compile(r"\"accountId\"\s*:\s*\"[^\"]+\""),
           "an accounts/<id> key": re.compile(r"accounts/[0-9a-f]{8}-[0-9a-f]{4}-")}
TEXT = {".json", ".py", ".md", ".txt", ".csv", ".log", ".yaml", ".yml", ".toml", ".sh", ".html", ".js", ""}


def annotation_patterns() -> list[str]:
    out = []
    for rec, dirs in ANNOTATED.items():
        out += [f"{rec}/manifest.json", f"{rec}/phase0/report.json", f"{rec}/phase0/cameras.json"]
        out += [f"{rec}/{d}/**" for d in dirs]
    return out


def mirrored_patterns() -> list[str]:
    """The shared annotation folders, kept as copies of the local ones: a file
    that is no longer there locally (a pitch map's old geometry, moved aside)
    is deleted from the dataset on upload. bench/ and models/ only grow."""
    return [f"{rec}/{d}/**" for rec, dirs in ANNOTATED.items() for d in dirs]


def remote_stale(api) -> list[str]:
    """Dataset files `upload` deletes: those in the mirrored folders that are not in footage/ any
    more, and any that are never shared (NEVER)."""
    from fnmatch import fnmatch

    local = {f.relative_to(FOOTAGE).as_posix() for f in matching_files()}
    return sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                  if any(fnmatch(f, x) for x in (*mirrored_patterns(), *NEVER)) and f not in local)


def patterns(annotations_only: bool = False) -> list[str]:
    return annotation_patterns() if annotations_only else [*SHARED, *annotation_patterns()]


def matching_files(annotations_only: bool = False) -> list[Path]:
    """The local files `upload` sends (for `list` and a sanity check)."""
    from fnmatch import fnmatch

    pats, files = patterns(annotations_only), []
    for p in sorted(FOOTAGE.rglob("*")):
        rel = p.relative_to(FOOTAGE).as_posix()
        if p.is_file() and any(fnmatch(rel, x) for x in pats) and not any(fnmatch(rel, x) for x in IGNORE):
            files.append(p)
    return files


def private_details(files: list[Path]) -> dict[str, list[str]]:
    """{file: what it holds} for the text files among `files` that hold something in PRIVATE."""
    out = {}
    for f in files:
        if f.suffix.lower() not in TEXT or f.stat().st_size > 50e6:
            continue
        text = f.read_text(errors="ignore")
        found = [what for what, rx in PRIVATE.items() if rx.search(text)]
        if found:
            out[f.relative_to(FOOTAGE).as_posix()] = found
    return out


def cmd_list(remote: bool = False, annotations_only: bool = False) -> bool:
    """Prints what upload would send; False when a file holds private details."""
    files = matching_files(annotations_only)
    by_top: dict[str, list[int]] = {}
    for f in files:
        parts = f.relative_to(FOOTAGE).parts
        top = parts[0] if parts[0] == "models" else "/".join(parts[:2])
        n = by_top.setdefault(top, [0, 0])
        n[0] += 1
        n[1] += f.stat().st_size
    for top, (n, size) in sorted(by_top.items()):
        print(f"  {top:28s} {n:6d} files  {size / 1e9:7.2f} GB")
    total = sum(f.stat().st_size for f in files)
    print(f"  {'total':28s} {len(files):6d} files  {total / 1e9:7.2f} GB")
    if remote:
        from huggingface_hub import HfApi

        stale = remote_stale(HfApi())
        print(f"  upload would delete {len(stale)} dataset files no longer here or never shared" + (":" if stale else ""))
        for f in stale:
            print(f"    {f}")
    private = private_details(files)
    for rel, found in private.items():
        print(f"  NOT TO SHARE as it is: {rel} holds {', '.join(found)}")
    return not private


def cmd_upload(annotations_only: bool = False) -> None:
    """--annotations-only leaves bench/ and models/ alone: the benches write
    their outputs (paths, candidate files) into the bench store, and those
    stay local until someone means to share them."""
    if private_details(matching_files(annotations_only)):
        cmd_list(annotations_only=annotations_only)
        sys.exit("nothing uploaded: fix the files marked NOT TO SHARE (see this script's notes) and run it again")
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()["name"]
    if not api.repo_exists(REPO, repo_type="dataset"):
        api.create_repo(REPO, repo_type="dataset", private=True)
        print(f"created {REPO} (private)")
    api.update_repo_settings(REPO, repo_type="dataset", gated="manual")
    print(f"{REPO}: gated, manual approval (as {who})")
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
                    commit_message="Dataset card")
    cmd_list(remote=True, annotations_only=annotations_only)
    # Committed in batches; rerun after an interruption and it skips what is
    # already there.
    api.upload_folder(repo_id=REPO, folder_path=str(FOOTAGE), repo_type="dataset",
                      allow_patterns=patterns(annotations_only), ignore_patterns=IGNORE,
                      delete_patterns=[*mirrored_patterns(), *NEVER],  # re-uploaded files are kept, not deleted
                      commit_message="Bench store, models, pitch maps and ball labels")
    info = api.dataset_info(REPO)
    print(f"done: https://huggingface.co/datasets/{REPO} "
          f"({'private' if info.private else 'visible'}, gated={info.gated})")


def cmd_fetch(annotations_only: bool, cache: bool = True, video: bool = True) -> None:
    """--no-cache leaves bench-cache/ out (the benchmarks then remake what
    they need from the video); --no-video leaves the store's video out (the
    benchmarks then run from the caches only)."""
    from huggingface_hub import snapshot_download

    allow = [p for p in patterns(annotations_only) if cache or not p.startswith("bench-cache/")]
    path = snapshot_download(REPO, repo_type="dataset", local_dir=FOOTAGE, allow_patterns=allow,
                             ignore_patterns=None if video else VIDEO)
    print(f"fetched into {path}")
    print("check it: uv run processing/spikes/bench.py check" + ("; the video: uv run processing/spikes/benchstore.py" if video else ""))


def main() -> None:
    if not FOOTAGE.is_dir() and sys.argv[1:2] != ["fetch"]:
        sys.exit("run from the repository root (footage/ not found)")
    match sys.argv[1:2]:
        case ["fetch"]:
            FOOTAGE.mkdir(exist_ok=True)
            cmd_fetch("--annotations-only" in sys.argv, "--no-cache" not in sys.argv, "--no-video" not in sys.argv)
        case ["list"]:
            cmd_list(remote="--remote" in sys.argv, annotations_only="--annotations-only" in sys.argv)
        case ["upload"]:
            cmd_upload("--annotations-only" in sys.argv)
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
