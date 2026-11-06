# processing

GPU pipeline that turns two phone recordings into one followed video.
License: **AGPL-3.0-only** (see `LICENSE` and `../CONTRIBUTING.md`).

## Today: `spikes/`

The Phase 0 scripts in `spikes/` are the working pipeline: sync and
stitch (`phase0.py`), pitch map (`pitch.py`), ball detection, judging and
path (`ballwin.py`, `balljudge.py`, `ballpath.py`), follow cam
(`follow.py`), and `fullgame.py`, which runs a whole game in chunks. It
runs locally on a Mac (MPS) or an NVIDIA GPU, started by hand on footage
downloaded from S3; a game takes about 6-11 hours on a Mac. The labelling
tools and model training live here too.

What was tried and what happened is summarised in `docs/FINDINGS.md`; how
it is measured is in `docs/TESTING-PROCESSING.md`; the dated detail is in
`docs/BACKLOG.md`; where recordings, labels and models live is in
`docs/DATA.md`.

## Planned: `src/osb_processing/`

The package the spikes move into once they are settled. Its stages are
listed in `src/osb_processing/__init__.py`. The plan is a container on
AWS Batch (Spot g5/g6, scaling to zero), started by the main repo's
`RecordingReady` event: PyAV / FFmpeg with NVDEC and NVENC for video,
PyTorch for remapping and detection. Not built yet.

Input will be the upload manifest (`schemas/manifest.schema.json`, pinned
from the main repo) plus the segment files in S3. Output: a 1080p MP4, a
detections file, and optionally a low-resolution full-field panorama.

Detector backends sit behind `osb_processing.detectors.base.Detector`, so
a backend can be swapped without touching tracking or rendering
(`CONTRIBUTING.md` item 3). Install one with an extra:
`pip install -e '.[rfdetr]'` or `pip install -e '.[ultralytics]'`. The
spikes call RF-DETR directly today; they adopt this interface when they
move into the package.
