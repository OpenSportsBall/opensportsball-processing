# The engine for the cloud job

Step 4 of the main repo's `docs/CLOUD-PROCESSING-PLAN.md`: one container
image that AWS Batch runs for a game someone asked to have processed in
the cloud. It is the same engine as the Mac app's (`engine.py`), on an
NVIDIA GPU, with `processing/spikes/cloud.py` in front: the only part that
knows about S3, DynamoDB or Batch. What the container is given and must
leave behind is the main repo's `docs/API.md`, "The engine's container".

**Not yet run on a GPU.** `cloud.py` is tested against a local S3 and
DynamoDB with a stand-in engine (`processing/tests/test_cloud.py`), and the
image builds and starts. The CUDA paths in `accel.py` (NVDEC, NVENC,
PyTorch on CUDA) have never run on real footage; see "First run" below.

## What is in it

| Part | From | Notes |
|---|---|---|
| Base | `ubuntu:26.04` | No NVIDIA base image: PyTorch's wheels carry their CUDA libraries, and the instance's NVIDIA container runtime brings the driver's |
| ffmpeg | Ubuntu's package (8.0) | Built with `h264_nvenc` and the `cuda` hwaccel; the build checks. 24.04's 6.1 gave grey frames after a seek into a Pixel segment. GPL build, run as a separate program |
| Python packages | `requirements.txt`, pinned for Linux x86-64 from `requirements.in`. The pins are compiled for Python 3.12 and installed on Ubuntu's 3.14; the build imports them | The Mac package's list without Core ML, plus boto3 |
| Engine code | `processing/spikes/*.py`, `lenses.json`, `schemas/` | The repo's layout, as in the Mac package |
| Models | not in the image | Fetched at job start from the footage bucket's `models/` prefix and checked against `models.json` |

About 3.6 GB. The image carries its commit (`OSB_ENGINE_VERSION`, and the
`org.opencontainers.image.revision` label), which is what the job record
shows as the engine's version and what the hosted service's source link
has to point at (AGPL section 13). `build_image.sh` marks a build from
uncommitted files `-dirty`: don't deploy one.

## Build and push

    packaging/cloud/build_image.sh <tag> [<repository uri> <footage bucket>]

The repository and bucket are the main repo's stack's
`EngineRepositoryUri` and `FootageBucket` outputs. With them it also
uploads the models (below) and pushes.
Then deploy that stack with `-c engineImageTag=<tag> -c engineCompute=gpu`
(its `infra/README.md`, "Cloud processing").

### Build on AWS

    packaging/cloud/build_image_cloud.sh <tag> <repository uri> <footage bucket>

The same image, built and pushed by AWS CodeBuild instead of Docker on
this machine. It needs the aws CLI signed in to the stack's account, `zip`
and `uv`, and no Docker. A Mac builds this linux/amd64 image under
emulation; CodeBuild builds it natively and pushes within the region.

What the script does, in order:

1. Stops if the repository already has the tag. A tag is never pushed
   twice: a job definition that names it would quietly get another engine.
2. Checks the models and uploads any the bucket lacks, as
   `build_image.sh` does.
3. Zips what the Dockerfile copies (`LICENSE`, `schemas/`, the three files
   it names in `packaging/cloud/`, `processing/spikes/*.py` and
   `lenses.json`) with a `buildspec.yml`, and puts the zip at
   `build/engine-<tag>.zip` in the footage bucket. The build sees only
   that zip, so a new file the Dockerfile copies has to be added to the
   script's list too.
4. Starts a build of the CodeBuild project `osb-engine-build` and waits.
   The build signs Docker in to the repository, runs `docker build` with
   the commit as `build_image.sh` gives it (`-dirty` included), and
   pushes. The base image is pulled from Amazon's mirror
   (`public.ecr.aws/docker/library/ubuntu`), because Docker Hub limits
   pulls from CodeBuild's shared addresses.
5. Deletes the zip. If the build failed, prints the end of its log
   (CloudWatch log group `/aws/codebuild/osb-engine-build`).

The first use makes two things in the account, outside the main repo's
stack:

| Resource | Name | Notes |
|---|---|---|
| CodeBuild project | `osb-engine-build` | Linux, 4 vCPU and 7 GB (`BUILD_GENERAL1_MEDIUM`), privileged so it can run Docker, 45 minute limit |
| IAM role | `osb-engine-build` | Reads `build/*` in the footage bucket, pushes and pulls in the engine repository, writes its own log. Nothing else |

Neither costs anything while idle. A build is billed by the minute
(about a cent a minute at this size) and starts from an empty machine, so
every build downloads the Python packages again. To remove them:

    aws codebuild delete-project --name osb-engine-build
    aws iam delete-role-policy --role-name osb-engine-build --policy-name build
    aws iam delete-role --role-name osb-engine-build

## Models

`models.json` pins each model by its SHA-256: where it is on the build
machine (`source`), where the engine expects it (`path`), and its key
under the bucket's `models/` prefix, which carries the start of the hash.
A job fetches them before `prepare` and `run`, checks each against its
hash, and stops with `bad_models` if one isn't right.

They are part of a deploy, not a separate chore:

- `build_image.sh` stops if a model on this machine is not the pinned one.
  After retraining: `uv run packaging/cloud/models.py pin`, commit
  `models.json`, build.
- A push (`build_image.sh <tag> <repository> <bucket>`) first uploads any
  pinned model the bucket lacks. A changed model is a new key, so nothing
  is overwritten and an older image still finds the models it was benched
  with.

## How a job goes

`cloud.py prepare`: the footage from S3 to the scratch disk (a file gets
the time S3 has for it, so `engine/verified.json` still holds on another
machine and nothing is hashed twice), `engine.py prepare`, then
`questions.json` and every picture it names to the job's output folder.

`cloud.py refit`: `engine.py prepare --answers --no-footage` on what the
first prepare left (restored from the job's work folder). Applying an
answer reads the background and the kit sample, not the video, so this
runs on a small CPU task without fetching 40 GB. If a step does want the
video, it runs again with the footage.

`cloud.py run`: `engine.py run --answers`. The answers are applied first;
if the engine won't use one (exit 3) its questions are published again
and the job goes back to the user. Otherwise `game.mp4` and `result.json`
(the done event: `ballAimShare`, `ballLow`) go to the output folder.

Throughout: the recording folder without the footage is copied to the
job's work folder as steps and chunks finish, with file times kept. A
reclaimed Spot instance, or the next command on another machine, restores
it and the engine carries on from its own done-files. SIGTERM stops the
engine's step, saves, and exits 130.

## First run on a GPU (not done)

Things only a real instance can show, cheapest first:

1. `accel.platform()` says `nvidia` in the container: `nvidia-smi` is on
   the PATH (the `utility` capability) and ffmpeg lists `h264_nvenc`.
   `OSB_REQUIRE_GPU=1` (set by the stack on GPU jobs) stops a job at once
   if not.
2. PyTorch can use the GPU. The pinned wheels (torch 2.15, `cu132`) are
   built with CUDA 13.2. The ECS AL2023 GPU image has NVIDIA driver
   580.178.04 (release 20261031), and NVIDIA's CUDA notes give 580 as the
   lowest driver for any CUDA 13.x program, with 13.2's own driver branch
   being 595. So it should work, by the lowest allowed driver; a job checks
   `torch.cuda.is_available()` in its first seconds and stops with `no_gpu`
   if not. If it does, pin a CUDA 12 build of torch in `requirements.in`.
   The A10G (g5) and L4 (g6) are both supported by CUDA 13, which dropped
   only GPUs older than Turing.
3. `ffmpeg -hwaccel cuda` decodes the phones' HEVC, and `h264_nvenc`
   encodes, inside the container (the `video` capability).
4. One chunk of one game: set `OSB_JOB_OPTIONS={"options":{"maxChunks":1}}`
   on the job. About ten minutes of game; it shows the time per chunk
   before a whole game is paid for.

Every command leaves `profile/<command>.json` in the job's output folder
(`cloudprofile.py`): when each part began, how busy the GPU, its memory,
the CPUs and the RAM were per engine step, each chunk's stage times, and
the footage and video transfer rates.
5. The must-see bench on the GPU against the Mac's 34/46, and a whole
   game's time and cost.
