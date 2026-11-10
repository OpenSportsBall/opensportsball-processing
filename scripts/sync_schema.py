#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare schemas/*.schema.json with the main repo's copies.

Usage: sync_schema.py <path to main repo checkout> [--update]

The pinned schemas are the upload manifest and the processing engine's
files (job, questions, answers, events). Exits 1 when any copy differs
(without --update). With --update, copies the main repo's schemas here
and records its commit in schemas/README.md.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["manifest.schema.json", "engine-job.schema.json", "engine-questions.schema.json",
         "engine-answers.schema.json", "engine-event.schema.json", "engine-labelset.schema.json",
         "engine-label-answers.schema.json", "engine-contribution.schema.json"]
README = ROOT / "schemas" / "README.md"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    main_repo = Path(args[0]).resolve()
    differ = []
    for name in NAMES:
        source = main_repo / "shared" / "schemas" / name
        local = ROOT / "schemas" / name
        if not source.is_file():
            sys.exit(f"not found: {source}")
        if not local.is_file() or source.read_bytes() != local.read_bytes():
            differ.append(name)
    if "--update" in sys.argv:
        commit = subprocess.run(
            ["git", "-C", str(main_repo), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        for name in NAMES:
            shutil.copyfile(main_repo / "shared" / "schemas" / name, ROOT / "schemas" / name)
        text = re.sub(r"(Pinned from main repo commit: )`[^`]*`",
                      rf"\1`{commit}`", README.read_text())
        README.write_text(text)
        print(f"schemas {'updated: ' + ', '.join(differ) if differ else 'unchanged'}; pinned to {commit}")
        return
    if not differ:
        print("schemas match the main repo")
    else:
        print(f"differ from the main repo: {', '.join(differ)}; run with --update", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
