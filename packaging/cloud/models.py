#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.35"]
# ///
"""The cloud engine's models: which ones an image is built for, and getting
them into the bucket it fetches them from.

    uv run packaging/cloud/models.py check             the local files are the pinned ones (exit 1 if not)
    uv run packaging/cloud/models.py pin               pin the local files: rewrite models.json
    uv run packaging/cloud/models.py upload <bucket>   put any pinned model the bucket lacks under models/
    uv run packaging/cloud/models.py fetch             download the pinned models this machine lacks (a fresh clone)
    uv run packaging/cloud/models.py release           publish our own pinned models as GitHub release assets

models.json pins each model by its SHA-256, and its key in the bucket
carries the start of that hash (models/<12 hex>/<name>). So a retrained
model is a new key: `pin` after replacing the file, commit, build. An
older image still finds the models it was benched with, and nothing in
the bucket is ever overwritten. build_image.sh runs `check` before it
builds and `upload` when it pushes, so an image is never deployed for
models that aren't there.

Where the local files are: `source` in models.json, relative to the
repository, or under the home folder when it starts with ~.

Our own models (the ball detector and judge: "release": true in
models.json) are also published as assets of a GitHub release of this
repository, one release per model file, tagged with its name and the
start of its hash (model-<name>-<12 hex>), so a clone can get the exact
files the code and the baselines were benched with: `fetch` downloads
each one this machine lacks and checks it against the pinned SHA-256.
`release` makes the release for any pinned model that has none (it needs
`gh`, logged in with write access). A model with "url" instead (RF-DETR's
stock weights) is downloaded from there. Where each model came from and
how it benched is in docs/MODELS.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = HERE / "models.json"
GITHUB = "OpenSportsBall/opensportsball-processing"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def local(m: dict) -> Path:
    src = m["source"]
    return Path(src).expanduser() if src.startswith("~") else REPO / src


def key_for(m: dict, digest: str) -> str:
    return f"{digest[:12]}/{Path(m['path']).name}"


def load() -> dict:
    return json.loads(SPEC.read_text())


def check(spec: dict) -> list[str]:
    """What is wrong with the local files, if anything."""
    problems = []
    for m in spec["models"]:
        f = local(m)
        if not f.exists():
            problems.append(f"{m['source']} is not on this machine")
        elif sha256(f) != m["sha256"]:
            problems.append(f"{m['source']} has changed since it was pinned")
    return problems


def pin(spec: dict) -> dict:
    for m in spec["models"]:
        f = local(m)
        if not f.exists():
            sys.exit(f"{m['source']} is not on this machine: nothing pinned")
        digest = sha256(f)
        if digest != m["sha256"]:
            print(f"{m['source']}: {m['sha256'][:12]} -> {digest[:12]}")
        m["sha256"], m["key"] = digest, key_for(m, digest)
    return spec


def upload(spec: dict, bucket: str, prefix: str = "models/") -> None:
    import boto3

    s3 = boto3.client("s3")
    for m in spec["models"]:
        key = prefix + m["key"]
        f = local(m)
        try:
            have = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
            have = None
        if have is not None and (not f.exists() or have == f.stat().st_size):
            print(f"in the bucket already: {key}")
            continue
        if not f.exists() or sha256(f) != m["sha256"]:
            sys.exit(f"{m['source']} is missing here or is not the pinned file: can't upload {key}")
        print(f"uploading {key} ({f.stat().st_size / 1e6:.0f} MB)")
        s3.upload_file(str(f), bucket, key)


def release_tag(m: dict) -> str:
    return f"model-{Path(m['path']).stem}-{m['sha256'][:12]}"


def download_url(m: dict) -> str | None:
    if m.get("release"):
        return f"https://github.com/{GITHUB}/releases/download/{release_tag(m)}/{Path(m['path']).name}"
    return m.get("url")


def fetch(spec: dict) -> int:
    """Downloads each pinned model that is not here; 1 if one could not be had."""
    import urllib.request

    bad = 0
    for m in spec["models"]:
        f = local(m)
        if f.exists():
            ok = sha256(f) == m["sha256"]
            print(f"{m['source']}: {'here' if ok else 'here, but NOT the pinned file (left alone)'}")
            bad += not ok
            continue
        url = download_url(m)
        if not url:
            print(f"{m['source']}: not here and not published anywhere this script knows")
            bad += 1
            continue
        print(f"{m['source']}: downloading {url}", flush=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        part = f.with_name(f.name + ".part")
        try:
            urllib.request.urlretrieve(url, part)
        except OSError as e:
            print(f"  failed: {e}")
            part.unlink(missing_ok=True)
            bad += 1
            continue
        if sha256(part) != m["sha256"]:
            print("  the download is not the pinned file: removed")
            part.unlink()
            bad += 1
            continue
        part.rename(f)
    return 1 if bad else 0


def release(spec: dict) -> int:
    """Makes a GitHub release for each of our own pinned models that has none."""
    import subprocess

    for m in spec["models"]:
        if not m.get("release"):
            continue
        tag, f = release_tag(m), local(m)
        if subprocess.run(["gh", "release", "view", tag, "--repo", GITHUB], capture_output=True).returncode == 0:
            print(f"released already: {tag}")
            continue
        if not f.exists() or sha256(f) != m["sha256"]:
            sys.exit(f"{m['source']} is missing here or is not the pinned file: can't release {tag}")
        print(f"releasing {tag} ({f.stat().st_size / 1e6:.0f} MB)")
        subprocess.run(["gh", "release", "create", tag, str(f), "--repo", GITHUB, "--title", tag, "--latest=false",
                        "--notes", f"`{Path(m['path']).name}`, sha256 `{m['sha256']}`. Pinned in packaging/cloud/models.json; "
                                   f"what it was trained on and how it benched: docs/MODELS.md."], check=True)
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    spec = load()
    if cmd == "check":
        problems = check(spec)
        for p in problems:
            print(p, file=sys.stderr)
        if problems:
            print("Pin them (models.py pin) and commit models.json, or put the pinned files back.", file=sys.stderr)
        return 1 if problems else 0
    if cmd == "pin":
        SPEC.write_text(json.dumps(pin(spec), indent=2) + "\n")
        return 0
    if cmd == "upload" and len(sys.argv) > 2:
        upload(spec, sys.argv[2])
        return 0
    if cmd == "fetch":
        return fetch(spec)
    if cmd == "release":
        return release(spec)
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
