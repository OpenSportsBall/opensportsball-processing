# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only

# osb-engine for the cloud job: the pipeline, its Python and ffmpeg, and
# cloud.py in front (packaging/cloud/README.md). Built for linux/amd64 from
# the repository root:
#
#     packaging/cloud/build_image.sh <tag>
#
# No NVIDIA base image: PyTorch's wheels carry their own CUDA libraries, and
# the instance's NVIDIA container runtime brings the driver's (CUDA,
# NVDEC, NVENC) for the capabilities named below. Ubuntu's ffmpeg is built
# with NVENC and the cuda hwaccel; the build checks.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc \
 && ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -q cuda

COPY packaging/cloud/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --no-deps -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt \
 && /opt/venv/bin/python -c "import torch, rfdetr, cv2, scipy, sklearn, jsonschema, yaml, boto3; print('torch', torch.__version__)"

# The repository's layout, so the scripts find each other, the schemas and
# the models (fetched at job start: packaging/cloud/models.json) as in a checkout.
WORKDIR /engine
COPY LICENSE /engine/LICENSE
COPY schemas /engine/schemas
COPY packaging/cloud/models.json /engine/packaging/cloud/models.json
COPY processing/spikes/*.py processing/spikes/lenses.json /engine/processing/spikes/
RUN /opt/venv/bin/python -m compileall -q /engine/processing/spikes

ARG COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/OpenSportsBall/opensportsball-processing" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.licenses="AGPL-3.0-only"
ENV OSB_ENGINE_VERSION="osb-engine ${COMMIT}" \
    OSB_PYTHON=/opt/venv/bin/python \
    RF_HOME=/engine/rf_home \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

ENTRYPOINT ["/opt/venv/bin/python", "/engine/processing/spikes/cloud.py"]
