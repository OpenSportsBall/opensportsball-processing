#!/bin/sh
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
#
# The packaged osb-engine for the Mac app: one folder a Mac with nothing
# installed can run (no Homebrew, no uv, no Python, nothing downloaded at
# run time).
#
#     packaging/mac/build_engine.sh <out dir> [<work dir>]
#
# <out dir>/osb-engine/ then holds:
#   osb-engine           the launcher: osb-engine prepare|run|status|label|share job.json ...
#   python/              CPython 3.12 (python-build-standalone) with requirements.txt installed
#   bin/                 ffmpeg, ffprobe (build_ffmpeg.sh: LGPL, VideoToolbox, static)
#   engine/              processing/spikes/*.py, schemas/, footage/models/ (the repo's layout,
#                        so the scripts find them as they do in a checkout)
#   rf_home/             RF-DETR's stock COCO weights (RF_HOME: the people detector, suggestions)
#   LICENSES/, NOTICE.md the engine's AGPL-3.0, ffmpeg's LGPL, where to get the source
#
# Needs on the build machine: Xcode's command line tools, uv, curl, the
# repo's footage/models/ (ball_detector.pth, ball_judge.pt; not in git) and
# RF-DETR's rf-detr-medium.pth (in ~/.roboflow/models after any run, or set
# RFDETR_MEDIUM). Signing and notarizing come after (packaging/mac/README.md).
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)/osb-engine
WORK=$(mkdir -p "${2:-$OUT.work}" && cd "${2:-$OUT.work}" && pwd)
PY_RELEASE=20261026
PY_VERSION=3.12.16
PY_SHA256=fbf4afef53c2a6b72bb741bc06d5671a9dfaf87bd84e5aa73ac9b3d0d589f14e
PY_TGZ="cpython-$PY_VERSION+$PY_RELEASE-aarch64-apple-darwin-install_only.tar.gz"
RFDETR_MEDIUM=${RFDETR_MEDIUM:-$HOME/.roboflow/models/rf-detr-medium.pth}
MODELS="$REPO/footage/models"

for f in "$MODELS/ball_detector.pth" "$MODELS/ball_judge.pt" "$RFDETR_MEDIUM"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/engine/processing" "$OUT/engine/footage/models" "$OUT/rf_home" "$OUT/LICENSES"

echo "== Python $PY_VERSION"
cd "$WORK"
[ -f "$PY_TGZ" ] || curl -sSfL -o "$PY_TGZ" \
  "https://github.com/astral-sh/python-build-standalone/releases/download/$PY_RELEASE/$(echo "$PY_TGZ" | sed 's/+/%2B/')"
echo "$PY_SHA256  $PY_TGZ" | shasum -a 256 -c -
tar xzf "$PY_TGZ" -C "$OUT"            # -> $OUT/python
PY="$OUT/python/bin/python3"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

echo "== Python packages"
uv pip install -q --python "$PY" --no-deps -r "$REPO/packaging/mac/requirements.txt"
"$PY" -c 'import torch, rfdetr, cv2, scipy, sklearn, coremltools, jsonschema, yaml; print("torch", torch.__version__, "MPS", torch.backends.mps.is_available())'

echo "== ffmpeg"
[ -x "$WORK/bin/ffmpeg" ] || "$REPO/packaging/mac/build_ffmpeg.sh" "$WORK/ffmpeg" "$WORK/bin"
cp "$WORK/bin/ffmpeg" "$WORK/bin/ffprobe" "$OUT/bin/"
cp "$WORK/bin/ffmpeg-LICENSE.txt" "$OUT/LICENSES/ffmpeg-LGPL-2.1.txt"

echo "== engine code, schemas, models"
mkdir -p "$OUT/engine/processing/spikes"
cp "$REPO"/processing/spikes/*.py "$REPO"/processing/spikes/lenses.json "$OUT/engine/processing/spikes/"
cp -R "$REPO/schemas" "$OUT/engine/schemas"
cp "$MODELS/ball_detector.pth" "$MODELS/ball_judge.pt" "$MODELS/README.md" "$OUT/engine/footage/models/"
cp "$RFDETR_MEDIUM" "$OUT/rf_home/rf-detr-medium.pth"
cp "$REPO/LICENSE" "$OUT/LICENSES/osb-engine-AGPL-3.0.txt"
# Compiled ahead, with hashes rather than times (still valid after a copy
# into the app): inside a signed app the engine must not write its own
# folder, and with these it never needs to.
"$PY" -m compileall -q -j 0 --invalidation-mode unchecked-hash "$OUT/engine/processing/spikes" "$OUT/python/lib" \
  || echo "(some files did not compile, e.g. packages' Python 2 test data; they are never imported)"

echo "== launcher"
cat > "$OUT/osb-engine" <<'LAUNCH'
#!/bin/sh
# osb-engine, packaged: the engine's own Python, ffmpeg and models; nothing
# from the PATH, the user's Python or the network.
HERE=$(cd "$(dirname "$0")" && pwd)
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1   # never write into the engine's folder (inside the signed app)
export OSB_PYTHON="$HERE/python/bin/python3"
export OSB_FFMPEG_DIR="$HERE/bin"
export RF_HOME="$HERE/rf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # a model that tries to download fails loudly instead
exec "$OSB_PYTHON" "$HERE/engine/processing/spikes/engine.py" "$@"
LAUNCH
chmod +x "$OUT/osb-engine"

COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
cat > "$OUT/NOTICE.md" <<NOTE
# osb-engine (packaged)

OpenSportsBall's processing engine, built from opensportsball-processing
$COMMIT. Source: https://github.com/OpenSportsBall/opensportsball-processing
(AGPL-3.0-only, LICENSES/osb-engine-AGPL-3.0.txt).

Bundled: CPython $PY_VERSION (python-build-standalone $PY_RELEASE, PSF
license), the Python packages in packaging/mac/requirements.txt (each
under its own license, in python/lib/python3.12/site-packages/*/), ffmpeg
(LGPL-2.1, LICENSES/ffmpeg-LGPL-2.1.txt; source at https://ffmpeg.org,
built by packaging/mac/build_ffmpeg.sh) and RF-DETR's COCO weights
(Apache-2.0).
NOTE
echo "$COMMIT" > "$OUT/VERSION"
du -sh "$OUT" "$OUT"/* | sort -h
echo "built $OUT"
