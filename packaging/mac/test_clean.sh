#!/bin/sh
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
#
# Runs the packaged engine as a Mac with nothing installed would: an empty
# environment, a PATH of system folders only (no Homebrew, no uv, no
# Python), a fresh home folder (no caches, no ~/.roboflow), the network
# switched off for model downloads.
#
#     packaging/mac/test_clean.sh <out dir>/osb-engine <command> <job.json> [args...]
set -eu
ENGINE=$1
shift
HOMEDIR=$(mktemp -d -t osb-clean-home)
echo "clean home: $HOMEDIR" >&2
exec env -i HOME="$HOMEDIR" PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
  "$ENGINE/osb-engine" "$@"
