#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare schemas/manifest.schema.json with the main repo's copy.

Usage: sync_schema.py <path to main repo checkout> [--update]

Exits 1 when the copies differ (without --update). With --update, copies
the main repo's schema here and records its commit in schemas/README.md.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "schemas" / "manifest.schema.json"
README = ROOT / "schemas" / "README.md"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    main_repo = Path(args[0]).resolve()
    source = main_repo / "shared" / "schemas" / "manifest.schema.json"
    if not source.is_file():
        sys.exit(f"not found: {source}")
    same = source.read_bytes() == LOCAL.read_bytes()
    if "--update" in sys.argv:
        commit = subprocess.run(
            ["git", "-C", str(main_repo), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        shutil.copyfile(source, LOCAL)
        text = re.sub(r"(Pinned from main repo commit: )`[^`]*`",
                      rf"\1`{commit}`", README.read_text())
        README.write_text(text)
        print(f"schema {'unchanged' if same else 'updated'}; pinned to {commit}")
        return
    if same:
        print("schema matches the main repo")
    else:
        print("schema differs from the main repo; run with --update", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
