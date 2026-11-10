# The packaged engine for the Mac app

Step D of the Mac app plan (the main repo's `docs/MAC-APP-PLAN.md`): one
folder, `osb-engine/`, that processes a game on an Apple Silicon Mac with
nothing installed. The Mac app (Apache-2.0) carries it inside its bundle
and runs `osb-engine` as a separate program, talking to it only through
files and progress lines (the main repo's `docs/LICENSING.md`).

## Build

    packaging/mac/build_engine.sh <out dir>

| Part | From | Notes |
|---|---|---|
| Python 3.12 | python-build-standalone (pinned release, SHA-256 checked) | |
| Python packages | `requirements.txt`, pinned for macOS arm64 from `requirements.in` | torch, rfdetr, coremltools, opencv, scipy, scikit-learn, jsonschema, pyyaml and what they need |
| ffmpeg, ffprobe | `build_ffmpeg.sh`: built from the release tarball | LGPL (no x264: the Mac encodes with VideoToolbox), static, no Homebrew libraries |
| Engine code | `processing/spikes/*.py`, `lenses.json`, `schemas/` | the repo's layout, so the scripts find everything as in a checkout |
| Models | `footage/models/` (ball detector, judge), RF-DETR's `rf-detr-medium.pth` | not in git; the build machine needs them |

The launcher sets `OSB_PYTHON`, `OSB_FFMPEG_DIR` (accel.py: the scripts run
on the bundled Python and ffmpeg instead of `uv run` and the PATH),
`RF_HOME` (RF-DETR's weights) and `HF_HUB_OFFLINE=1`, so nothing is
downloaded while a game is processed. The Core ML versions of the
detectors are made on first use (about a minute) in
`~/Library/Caches/opensportsball/coreml`.

## Test as a clean Mac

    packaging/mac/test_clean.sh <out dir>/osb-engine prepare job.json

Runs it with an empty environment, system folders only on the PATH (no
Homebrew, no uv, no Python), a new empty home folder and model downloads
switched off. A real second Mac (or a macOS VM) after signing.

## Sign and notarize (not done yet)

Needs a **Developer ID Application** certificate (Apple Developer account:
Certificates; this Mac has only "Apple Development") and notarytool
credentials (the App Store Connect API key the TestFlight upload uses
works). Every Mach-O in the folder (the Python, its `.so` and `.dylib`
files, ffmpeg, ffprobe) is signed with the hardened runtime, the bundled
Python with the entitlements it needs (likely `allow-unsigned-executable-memory`
for PyTorch and `disable-library-validation` only if a library can't be
signed), then the app bundle is notarized and stapled.
