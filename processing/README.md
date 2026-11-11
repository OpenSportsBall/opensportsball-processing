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

## In the cloud

`spikes/cloud.py` runs `engine.py` as one AWS Batch job of the main repo's
cloud processing: footage from S3, questions and the game video back to
S3, the folder's state kept between commands. The image is in
`../packaging/cloud/` (its README says what has and hasn't been run on a
GPU).

## Planned: `src/osb_processing/`

The package the spikes move into once they are settled. Its stages are
listed in `src/osb_processing/__init__.py`. The cloud job
runs the spikes through `engine.py` today (above); the package would
replace them there too. Not built yet.

Input will be the upload manifest (`schemas/manifest.schema.json`, pinned
from the main repo) plus the segment files in S3. Output: a 1080p MP4, a
detections file, and optionally a low-resolution full-field panorama.

Detector backends sit behind `osb_processing.detectors.base.Detector`, so
a backend can be swapped without touching tracking or rendering
(`CONTRIBUTING.md` item 3). Install one with an extra:
`pip install -e '.[rfdetr]'` or `pip install -e '.[ultralytics]'`. The
spikes call RF-DETR directly today; they adopt this interface when they
move into the package.
