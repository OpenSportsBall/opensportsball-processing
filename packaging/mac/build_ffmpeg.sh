#!/bin/sh
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
#
# ffmpeg and ffprobe for the packaged engine: LGPL (no --enable-gpl, no
# x264: the Mac encodes with VideoToolbox), statically linked, nothing
# auto-detected (so no Homebrew library can sneak in), only system
# frameworks. The engine needs: mp4/mov in and out, HEVC and H.264 decode
# (VideoToolbox), h264_videotoolbox, AAC decode and copy, raw video and PCM
# pipes, the concat demuxer, scale.
#
#     packaging/mac/build_ffmpeg.sh <work dir> <out dir>
#
# Downloads the release tarball from ffmpeg.org (checked against
# FFMPEG_SHA256 below) and builds it with Xcode's clang (~5 min).
set -eu
VERSION=8.0.1
FFMPEG_SHA256=05ee0b03119b45c0bdb4df654b96802e909e0a752f72e4fe3794f487229e5a41
WORK=$1
OUT=$2
mkdir -p "$WORK" "$OUT"
cd "$WORK"
[ -f "ffmpeg-$VERSION.tar.xz" ] || curl -sSfLO "https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz"
GOT=$(shasum -a 256 "ffmpeg-$VERSION.tar.xz" | cut -d' ' -f1)
if [ -n "$FFMPEG_SHA256" ] && [ "$GOT" != "$FFMPEG_SHA256" ]; then
  echo "ffmpeg-$VERSION.tar.xz: SHA-256 $GOT, expected $FFMPEG_SHA256" >&2
  exit 1
fi
echo "ffmpeg-$VERSION.tar.xz SHA-256 $GOT"
rm -rf "ffmpeg-$VERSION"
tar xf "ffmpeg-$VERSION.tar.xz"
cd "ffmpeg-$VERSION"
PKG_CONFIG_PATH='' ./configure --prefix="$WORK/install" --cc=clang --arch=arm64 \
  --disable-autodetect --enable-videotoolbox --enable-audiotoolbox --enable-zlib \
  --disable-network --disable-doc --disable-ffplay --disable-debug \
  --enable-static --disable-shared \
  --extra-cflags=-mmacosx-version-min=13.0 --extra-ldflags=-mmacosx-version-min=13.0 \
  > "$WORK/configure.log"
make -j"$(sysctl -n hw.ncpu)" > "$WORK/make.log" 2>&1
cp ffmpeg ffprobe "$OUT/"
cp COPYING.LGPLv2.1 "$OUT/ffmpeg-LICENSE.txt"
echo "ffmpeg $VERSION -> $OUT"
"$OUT/ffmpeg" -hide_banner -buildconf | tr -d ' ' | grep -E '^--' | tr '\n' ' '
echo
# Only system libraries and frameworks: anything under /opt/homebrew or /usr/local is a mistake.
if otool -L "$OUT/ffmpeg" "$OUT/ffprobe" | grep -E '/opt/homebrew|/usr/local'; then
  echo "links a non-system library" >&2
  exit 1
fi
