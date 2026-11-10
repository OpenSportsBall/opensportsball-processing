# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Every platform choice in one place: the GPU PyTorch runs on, the
video decoder and encoder ffmpeg uses, and whether the detectors run in
Core ML. The rest of the pipeline asks here instead of naming Apple's
(or NVIDIA's) APIs itself, so a new platform is a change to this file.

    Mac (Apple Silicon):  MPS, VideoToolbox decode and encode, Core ML detectors
    NVIDIA (cloud, Windows): CUDA, NVDEC (-hwaccel cuda) and NVENC
    anything else:        CPU, software decode, libx264

OSB_ACCEL=cpu forces the last row (a check that the pipeline runs
anywhere; slow). Imports nothing heavy: fullgame.py uses it without torch.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys

FORCE_CPU = os.environ.get("OSB_ACCEL", "") == "cpu"


@functools.cache
def platform() -> str:
    """"mac", "nvidia" or "cpu"."""
    if FORCE_CPU:
        return "cpu"
    if sys.platform == "darwin":
        return "mac"
    if shutil.which("nvidia-smi") and "h264_nvenc" in _ffmpeg_encoders():
        return "nvidia"
    return "cpu"


@functools.cache
def _ffmpeg_encoders() -> str:
    try:
        return subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                              timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def device():
    """The torch device for the GPU work (follow.py's warps, the judge,
    PyTorch detectors)."""
    import torch

    if not FORCE_CPU and torch.backends.mps.is_available():
        return torch.device("mps")
    if not FORCE_CPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_args() -> list[str]:
    """ffmpeg input options for hardware decoding (before -i)."""
    return {"mac": ["-hwaccel", "videotoolbox"], "nvidia": ["-hwaccel", "cuda"]}.get(platform(), [])


def h264_args(bitrate: str, maxrate: str | None = None) -> list[str]:
    """ffmpeg output options for H.264 at about `bitrate`, in hardware where
    there is an encoder for it. With `maxrate`, the rate is capped too (the
    upload copy)."""
    cap = ["-maxrate", maxrate, "-bufsize", maxrate] if maxrate else []
    enc = {"mac": ["-c:v", "h264_videotoolbox"],
           "nvidia": ["-c:v", "h264_nvenc", "-preset", "p5"]}.get(platform(), ["-c:v", "libx264", "-preset", "veryfast"])
    return [*enc, "-b:v", bitrate, *cap]


def coreml() -> bool:
    """Whether the detectors default to Core ML (detect.coreml_detector)."""
    return platform() == "mac"
