#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
#
# Builds the cloud engine image (packaging/cloud/Dockerfile) for linux/amd64
# and, given a repository, pushes it.
#
#   packaging/cloud/build_image.sh <tag> [<repository uri>]
#
# The repository is the main repo's stack's EngineRepositoryUri output; sign
# Docker in to it first (aws ecr get-login-password | docker login ...).
set -euo pipefail

tag="${1:?usage: build_image.sh <tag> [<repository uri>]}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
commit="$(git -C "$repo" rev-parse --short HEAD)"
if [ -n "$(git -C "$repo" status --porcelain -- processing/spikes schemas packaging/cloud)" ]; then
  commit="$commit-dirty"   # the image says so: its source isn't a commit anyone can fetch
fi

docker build --platform linux/amd64 --build-arg COMMIT="$commit" \
  -f "$repo/packaging/cloud/Dockerfile" -t "osb-engine:$tag" "$repo"
echo "Built osb-engine:$tag from $commit"
if [ -n "${2:-}" ]; then
  docker tag "osb-engine:$tag" "$2:$tag"
  docker push "$2:$tag"
  echo "Pushed $2:$tag"
fi
