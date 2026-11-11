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
| Base | `ubuntu:24.04` | No NVIDIA base image: PyTorch's wheels carry their CUDA libraries, and the instance's NVIDIA container runtime brings the driver's |
| ffmpeg | Ubuntu's package (6.1) | Built with `h264_nvenc` and the `cuda` hwaccel; the build checks. GPL build, run as a separate program |
| Python 3.12 packages | `requirements.txt`, pinned for Linux x86-64 from `requirements.in` | The Mac package's list without Core ML, plus boto3 |
| Engine code | `processing/spikes/*.py`, `lenses.json`, `schemas/` | The repo's layout, as in the Mac package |
| Models | not in the image | Fetched at job start from the footage bucket's `models/` prefix and checked against `models.json` |

About 3.6 GB. The image carries its commit (`OSB_ENGINE_VERSION`, and the
`org.opencontainers.image.revision` label), which is what the job record
shows as the engine's version and what the hosted service's source link
has to point at (AGPL section 13). `build_image.sh` marks a build from
uncommitted files `-dirty`: don't deploy one.

## Build and push

    packaging/cloud/build_image.sh <tag> [<repository uri>]

The repository is the main repo's stack's `EngineRepositoryUri` output.
Then deploy that stack with `-c engineImageTag=<tag> -c engineCompute=gpu`
(its `infra/README.md`, "Cloud processing").

## Models

`models.json` lists each model: its key under the bucket's `models/`
prefix, where the engine expects it, and its SHA-256. A job fetches them
before `prepare` and `run` and stops with `bad_models` if one in the
bucket is not the one this image was built for. Upload them once, from a
machine that has them (`footage/models/`, and RF-DETR's stock weights in
`~/.roboflow/models/`):

    aws s3 cp footage/models/ball_detector.pth s3://<footage bucket>/models/2026-11-01/ball_detector.pth
    aws s3 cp footage/models/ball_judge.pt     s3://<footage bucket>/models/2026-11-01/ball_judge.pt
    aws s3 cp ~/.roboflow/models/rf-detr-medium.pth s3://<footage bucket>/models/rfdetr/rf-detr-medium.pth

New models go under a new folder with a new `models.json`, so an older
image still finds the ones it was benched with.

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
2. PyTorch 2.15's wheels are built for CUDA 13: the instance's NVIDIA
   driver has to be new enough for it. If the ECS GPU image's driver is
   older, pin a CUDA 12 build of torch in `requirements.in`.
3. `ffmpeg -hwaccel cuda` decodes the phones' HEVC, and `h264_nvenc`
   encodes, inside the container (the `video` capability).
4. One chunk of one game: set `OSB_JOB_OPTIONS={"options":{"maxChunks":1}}`
   on the job. About ten minutes of game; it shows the time per chunk
   before a whole game is paid for.
5. The must-see bench on the GPU against the Mac's 34/46, and a whole
   game's time and cost.
