#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
#
# Builds the cloud engine image on AWS CodeBuild and pushes it: what
# build_image.sh does with a push, without Docker on this machine.
#
#   packaging/cloud/build_image_cloud.sh <tag> <repository uri> <footage bucket>
#
# The repository and the bucket are the main repo's stack's
# EngineRepositoryUri and FootageBucket outputs. Needs the aws CLI signed in
# to that account (AWS_PROFILE, AWS_REGION), zip and uv.
#
# The first use makes a CodeBuild project and its role, both named
# osb-engine-build (packaging/cloud/README.md, "Build on AWS"). A build sees
# only the files zipped below: what the Dockerfile copies.
set -euo pipefail

tag="${1:?usage: build_image_cloud.sh <tag> <repository uri> <footage bucket>}"
uri="${2:?usage: build_image_cloud.sh <tag> <repository uri> <footage bucket>}"
bucket="${3:?usage: build_image_cloud.sh <tag> <repository uri> <footage bucket>}"
name=osb-engine-build
registry="${uri%%/*}"
repository="${uri#*/}"
account="${registry%%.*}"
region="$(echo "$registry" | cut -d. -f4)"
export AWS_REGION="$region" AWS_DEFAULT_REGION="$region"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
commit="$(git -C "$repo" rev-parse --short HEAD)"
if [ -n "$(git -C "$repo" status --porcelain -- processing/spikes schemas packaging/cloud)" ]; then
  commit="$commit-dirty"   # the image says so: its source isn't a commit anyone can fetch
fi

# A tag is never pushed twice: a job definition that names it would quietly
# get another engine.
if aws ecr describe-images --repository-name "$repository" --image-ids "imageTag=$tag" >/dev/null 2>&1; then
  echo "The repository already has an image tagged $tag: choose another tag." >&2
  exit 2
fi

uv run -q "$repo/packaging/cloud/models.py" check
uv run -q "$repo/packaging/cloud/models.py" upload "$bucket"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if ! aws iam get-role --role-name "$name" >/dev/null 2>&1; then
  cat > "$tmp/trust.json" <<EOF
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
  "Principal": {"Service": "codebuild.amazonaws.com"}, "Action": "sts:AssumeRole",
  "Condition": {"StringEquals": {"aws:SourceAccount": "${account}"}}}]}
EOF
  aws iam create-role --role-name "$name" --assume-role-policy-document "file://$tmp/trust.json" \
    --description "CodeBuild: builds and pushes the osb-engine image" >/dev/null
  echo "Made the role $name"
  made_role=1
fi
cat > "$tmp/policy.json" <<EOF
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:GetObjectVersion"],
   "Resource": "arn:aws:s3:::${bucket}/build/*"},
  {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
  {"Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload",
     "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage",
     "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
   "Resource": "arn:aws:ecr:${region}:${account}:repository/${repository}"},
  {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
   "Resource": "arn:aws:logs:${region}:${account}:log-group:/aws/codebuild/${name}*"}]}
EOF
aws iam put-role-policy --role-name "$name" --policy-name build --policy-document "file://$tmp/policy.json"

# The source: what the Dockerfile copies, and the commands to run. The base
# image comes from Amazon's mirror of Docker Hub's, which doesn't limit
# pulls from a shared address.
cat > "$tmp/buildspec.yml" <<EOF
version: 0.2
phases:
  build:
    commands:
      - aws ecr get-login-password | docker login --username AWS --password-stdin ${registry}
      - sed -i 's#^FROM ubuntu:#FROM public.ecr.aws/docker/library/ubuntu:#' packaging/cloud/Dockerfile
      - docker build --build-arg COMMIT=\$COMMIT -f packaging/cloud/Dockerfile -t ${uri}:\$TAG .
      - docker push ${uri}:\$TAG
EOF
(cd "$repo" && zip -q -r "$tmp/src.zip" LICENSE schemas packaging/cloud/Dockerfile \
  packaging/cloud/requirements.txt packaging/cloud/models.json \
  processing/spikes/lenses.json processing/spikes/*.py)
(cd "$tmp" && zip -q src.zip buildspec.yml)
key="build/engine-$tag.zip"
aws s3 cp "$tmp/src.zip" "s3://$bucket/$key" --only-show-errors

if [ -z "$(aws codebuild batch-get-projects --names "$name" --query 'projects[0].name' --output text | grep -v None)" ]; then
  [ -n "${made_role:-}" ] && sleep 15   # a new role takes a moment to be usable
  aws codebuild create-project --name "$name" \
    --description "Builds and pushes the osb-engine image (opensportsball-processing, packaging/cloud)" \
    --source "type=S3,location=$bucket/$key" --artifacts type=NO_ARTIFACTS \
    --environment type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_MEDIUM,privilegedMode=true \
    --service-role "arn:aws:iam::${account}:role/$name" --timeout-in-minutes 45 >/dev/null
  echo "Made the CodeBuild project $name"
fi

id="$(aws codebuild start-build --project-name "$name" --source-location-override "$bucket/$key" \
  --environment-variables-override "name=TAG,value=$tag" "name=COMMIT,value=$commit" \
  --query build.id --output text)"
echo "Building $uri:$tag from $commit (build $id)"
until status="$(aws codebuild batch-get-builds --ids "$id" --query 'builds[0].buildStatus' --output text)"; [ "$status" != IN_PROGRESS ]; do
  sleep 20
done
aws s3 rm "s3://$bucket/$key" --only-show-errors
if [ "$status" != SUCCEEDED ]; then
  echo "The build ended $status. The end of its log:" >&2
  aws logs tail "/aws/codebuild/$name" --log-stream-name-prefix "${id#*:}" --since 2h --format short | tail -40 >&2
  exit 1
fi
echo "Pushed $uri:$tag"
