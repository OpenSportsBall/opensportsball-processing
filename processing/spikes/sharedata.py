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

    uv run processing/spikes/sharedata.py fetch [--annotations-only]   # contributors
    uv run processing/spikes/sharedata.py list [--remote]              # what upload would send (and delete)
    uv run processing/spikes/sharedata.py upload [--annotations-only]  # maintainers

The dataset mirrors footage/ path for path, so `fetch` drops everything
where the tools already look:

  bench/                         the bench store (TESTING-PROCESSING.md section 6), video included
  models/                        the standard ball detector and judge
  <rec>/pitch/, <rec>/labels*/   pitch maps and ball label sets of recordings outside the bench
  <rec>/manifest.json, <rec>/phase0/report.json   what those labels need to find their frames
  <rec>/phase0/cameras.json      the cameras those pitch maps were made with (jointfit.py)

Before `fetch`, request access on the dataset's page and log in with
`hf auth login` (a read token is enough). `upload` needs a write token
for the dataset's organisation. It creates the repository private and
gated (manual approval) if it does not exist; making it visible, so
people can request access, is a deliberate step on its settings page.

OSB_HF_DATASET overrides the repository (default below). See
docs/DATA.md, "Sharing with contributors".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("OSB_HF_DATASET", "OpenSportsBall/bench")
FOOTAGE = Path("footage")
CARD = Path(__file__).resolve().parents[2] / "docs" / "DATASET_CARD.md"

# Recordings outside the bench store whose pitch maps and labels we share.
# (rec3's and rec4cal's are already inside bench/.)
ANNOTATED = {
    "rec5-game": ["pitch", "labels", "labels_kits"],
    "rec6": ["pitch", "labels"],
    "rec7": ["pitch", "labels_seg6"],
}

IGNORE = [".DS_Store", "**/.DS_Store", "**/__pycache__/**", ".cache/**"]


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
    """Dataset files in the mirrored folders that are not in footage/ any more."""
    from fnmatch import fnmatch

    local = {f.relative_to(FOOTAGE).as_posix() for f in matching_files()}
    return sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                  if any(fnmatch(f, x) for x in mirrored_patterns()) and f not in local)


def patterns(annotations_only: bool = False) -> list[str]:
    return annotation_patterns() if annotations_only else ["bench/**", "models/**", *annotation_patterns()]


def matching_files(annotations_only: bool = False) -> list[Path]:
    """The local files `upload` sends (for `list` and a sanity check)."""
    from fnmatch import fnmatch

    pats, files = patterns(annotations_only), []
    for p in sorted(FOOTAGE.rglob("*")):
        rel = p.relative_to(FOOTAGE).as_posix()
        if p.is_file() and any(fnmatch(rel, x) for x in pats) and not any(fnmatch(rel, x) for x in IGNORE):
            files.append(p)
    return files


def cmd_list(remote: bool = False, annotations_only: bool = False) -> None:
    files = matching_files(annotations_only)
    by_top: dict[str, list[int]] = {}
    for f in files:
        parts = f.relative_to(FOOTAGE).parts
        top = parts[0] if parts[0] in ("bench", "models") else "/".join(parts[:2])
        n = by_top.setdefault(top, [0, 0])
        n[0] += 1
        n[1] += f.stat().st_size
    for top, (n, size) in sorted(by_top.items()):
        print(f"  {top:28s} {n:6d} files  {size / 1e9:7.2f} GB")
    total = sum(f.stat().st_size for f in files)
    print(f"  {'total':28s} {len(files):6d} files  {total / 1e9:7.2f} GB")
    if remote:
        stale = remote_stale(HfApi())
        print(f"  upload would delete {len(stale)} dataset files no longer here" + (":" if stale else ""))
        for f in stale:
            print(f"    {f}")


def cmd_upload(annotations_only: bool = False) -> None:
    """--annotations-only leaves bench/ and models/ alone: the benches write
    their outputs (paths, candidate files) into the bench store, and those
    stay local until someone means to share them."""
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
                      delete_patterns=mirrored_patterns(),  # re-uploaded files are kept, not deleted
                      commit_message="Bench store, models, pitch maps and ball labels")
    info = api.dataset_info(REPO)
    print(f"done: https://huggingface.co/datasets/{REPO} "
          f"({'private' if info.private else 'visible'}, gated={info.gated})")


def cmd_fetch(annotations_only: bool) -> None:
    path = snapshot_download(REPO, repo_type="dataset", local_dir=FOOTAGE,
                             allow_patterns=patterns(annotations_only))
    print(f"fetched into {path}")
    if not annotations_only:
        print("check the bench video: uv run processing/spikes/benchstore.py")


def main() -> None:
    if not FOOTAGE.is_dir() and sys.argv[1:2] != ["fetch"]:
        sys.exit("run from the repository root (footage/ not found)")
    match sys.argv[1:2]:
        case ["fetch"]:
            FOOTAGE.mkdir(exist_ok=True)
            cmd_fetch("--annotations-only" in sys.argv)
        case ["list"]:
            cmd_list(remote="--remote" in sys.argv, annotations_only="--annotations-only" in sys.argv)
        case ["upload"]:
            cmd_upload("--annotations-only" in sys.argv)
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
