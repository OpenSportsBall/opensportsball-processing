#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Where a recording's made files go.

A working recording (footage/rec7) keeps everything in its own folder, as
it always has. A recording in the bench store (footage/bench/rec7) holds
only what the benchmarks read and nothing can remake: the manifest, sync
and cameras, pitch map, team colours, must-see sets, ball label sets and
the video. What is made from those goes beside the store, under the same
relative path:

    footage/bench-cache/<rec>/   cache(rec): costly to make, remade from the
                                 inputs and the video (pass A caches, ball
                                 candidates, roles, net series, backgrounds)
    footage/bench-out/<rec>/     out(rec): what a run writes (paths, bench
                                 scores, camera logs, debug images, renders);
                                 safe to delete at any time
    footage/train/<rec>/         train(rec): what training on the recording's
                                 label sets makes (tiled datasets, judge
                                 crops, checkpoints, a recording's own models)

So the store can be shared and checked as it is, the caches are an
optional download that only saves time, and neither a bench run nor a
training run changes the store. OSB_BENCH_CACHE, OSB_BENCH_OUT and
OSB_TRAIN name other roots (each takes the place of footage/bench-cache,
footage/bench-out or footage/train).

docs/DATA.md has the whole layout.
"""

from __future__ import annotations

import os
from pathlib import Path

FOOTAGE, STORE = "footage", "bench"  # a recording is in the bench store when it is under footage/bench/


def _store_index(rec: Path) -> int | None:
    """Index of the bench store's folder in `rec`'s absolute path (the
    innermost footage/bench above it), or None for a working recording."""
    parts = Path(rec).absolute().parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == STORE and parts[i - 1] == FOOTAGE:
            return i
    return None


def in_store(rec: Path) -> bool:
    return _store_index(rec) is not None


def _beside_store(rec: Path, name: str, env: str) -> Path:
    i = _store_index(rec)
    if i is None:
        return Path(rec)
    parts = Path(rec).absolute().parts
    root = Path(os.environ[env]) if os.environ.get(env) else Path(*parts[:i]) / name
    made = root.joinpath(*parts[i + 1:])
    made.mkdir(parents=True, exist_ok=True)  # callers make only their own subfolder
    return made


def cache(rec: Path) -> Path:
    """The folder for `rec`'s caches: `rec` itself, or its place under
    footage/bench-cache/ when it is in the bench store."""
    return _beside_store(rec, f"{STORE}-cache", "OSB_BENCH_CACHE")


def out(rec: Path) -> Path:
    """The folder for what runs on `rec` write: `rec` itself, or its place
    under footage/bench-out/ when it is in the bench store."""
    return _beside_store(rec, f"{STORE}-out", "OSB_BENCH_OUT")


def train(rec: Path) -> Path:
    """The folder for what training on `rec`'s labels makes: `rec` itself,
    or its place under footage/train/ when it is in the bench store."""
    return _beside_store(rec, "train", "OSB_TRAIN")
