# processing

GPU pipeline that turns two phone recordings into one followed video.
License: **AGPL-3.0-only** (see `LICENSE` and `docs/LICENSING.md`).

Runs as a container on AWS Batch (Spot g5/g6, scales to zero): PyAV /
FFmpeg with NVDEC and NVENC for video, PyTorch for remapping and detection.
Stages are listed in `src/osb_processing/__init__.py`.

Input is the upload manifest (`shared/schemas/manifest.schema.json`) plus
the segment files in S3. Output is a 1080p MP4, a detections file, and
optionally a low-resolution full-field panorama.

Detector backends sit behind `osb_processing.detectors.base.Detector`.
Install one with an extra: `pip install -e '.[rfdetr]'` or
`pip install -e '.[ultralytics]'`.

Phase 0 spikes in `spikes/` are the working pipeline today (follow cam,
pitch map, ball tracking); how they are measured is in
`docs/TESTING-PROCESSING.md`, and what was tried in `docs/BACKLOG.md`.
