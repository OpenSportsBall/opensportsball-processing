#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
#
# Builds the cloud engine image (packaging/cloud/Dockerfile) for linux/amd64
# and, given a repository, pushes it.
#
#   packaging/cloud/build_image.sh <tag> [<repository uri> <footage bucket>]
#
# The repository and the bucket are the main repo's stack's
# EngineRepositoryUri and FootageBucket outputs; sign Docker in to the
# repository first (aws ecr get-login-password | docker login ...).
#
# The models are part of a deploy: the build stops if a model on this
# machine is not the one models.json pins (a retrained model: models.py pin,
# commit), and a push first puts any pinned model the bucket lacks under
# its models/ prefix. An image is never pushed for models that aren't there.
set -euo pipefail

tag="${1:?usage: build_image.sh <tag> [<repository uri> <footage bucket>]}"
if [ -n "${2:-}" ] && [ -z "${3:-}" ]; then
  echo "A push needs the footage bucket too, for the models: build_image.sh <tag> <repository uri> <footage bucket>" >&2
  exit 2
fi
repo="$(cd "$(dirname "$0")/../.." && pwd)"
commit="$(git -C "$repo" rev-parse --short HEAD)"
if [ -n "$(git -C "$repo" status --porcelain -- processing/spikes schemas packaging/cloud)" ]; then
  commit="$commit-dirty"   # the image says so: its source isn't a commit anyone can fetch
fi

uv run -q "$repo/packaging/cloud/models.py" check

docker build --platform linux/amd64 --build-arg COMMIT="$commit" \
  -f "$repo/packaging/cloud/Dockerfile" -t "osb-engine:$tag" "$repo"
echo "Built osb-engine:$tag from $commit"
if [ -n "${2:-}" ]; then
  uv run -q "$repo/packaging/cloud/models.py" upload "$3"
  docker tag "osb-engine:$tag" "$2:$tag"
  docker push "$2:$tag"
  echo "Pushed $2:$tag"
fi
