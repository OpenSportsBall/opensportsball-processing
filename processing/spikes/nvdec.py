# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Video decoded on an NVIDIA GPU straight into PyTorch tensors, so a 4K
frame never crosses to the CPU: the ffmpeg pipe (follow.frame_stream)
decodes there too, but copies every 12 MB frame to main memory, writes it
down a pipe and has Python upload it again, which with two phones at 30
frames a second was about a quarter of a cloud instance's CPU time.

    frames(path, start, seconds, fps, width, height)
        -> gpu.GpuNV12 frames at constant fps (the frames, to the sample,
           and the timing `ffmpeg -ss start -t seconds -vf fps=fps` gives),
           or None where this can't be used

Used on an NVIDIA GPU (accel.platform() "nvidia") where torchcodec is
installed, for 8-bit 4:2:0 video asked for at its own size. OSB_NVDEC=0
turns it off; OSB_NVDEC=1 turns it on anywhere torchcodec runs (decoding on
the CPU: a check of this file on a machine without the GPU).
follow.frame_stream falls back to the pipe when frames() returns None.

The decoder is torchcodec's packet-level one (Demuxer, VideoPacketDecoder):
it gives the codec's own NV12 planes and each frame's time, where its
VideoDecoder gives RGB.
"""

from __future__ import annotations

import functools
from fractions import Fraction
import json
import math
import os
import subprocess
from pathlib import Path

import accel

LEAD_IN_S = (2.5, 8.0, 30.0)  # how far before the first frame wanted to start decoding; the next if that wasn't enough
_told: set[str] = set()


def _tell(why: str) -> None:
    if why not in _told:
        _told.add(why)
        print(f"nvdec: {why}: reading frames through the ffmpeg pipe", flush=True)


_gave_up = False


def give_up(why: str) -> None:
    """Stop using this reader in this process (after a failure part-way through a file)."""
    global _gave_up
    _gave_up = True
    _tell(why)


def wanted() -> bool:
    if _gave_up:
        return False
    flag = os.environ.get("OSB_NVDEC", "")
    return flag == "1" or (flag != "0" and accel.platform() == "nvidia")


@functools.cache
def _file_timing(path: str) -> tuple[int, Fraction]:
    """The file's start time in microseconds (ffmpeg counts -ss from it:
    the earliest stream's first timestamp; frame times here are the
    file's own) and the video's time base, s."""
    out = subprocess.run([accel.ffprobe(), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "format=start_time:stream=time_base", "-of", "json", path],
                         capture_output=True, text=True, timeout=60).stdout
    try:
        d = json.loads(out)
        return int(round(float(d["format"]["start_time"]) * 1e6)), Fraction(d["streams"][0]["time_base"])
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError):
        return 0, Fraction(1, 90000)


def _half_up(x: Fraction) -> int:
    return math.floor(x + Fraction(1, 2))


def _nv12(frame, dev):
    """A decoded frame's planes as gpu.GpuNV12 on `dev`."""
    import torch

    import gpu

    planes = frame.planes
    y = planes[0]
    if len(planes) == 2:  # NV12 as one interleaved chroma plane
        uv = planes[1].reshape(y.shape[0] // 2, y.shape[1] // 2, 2)
    else:  # U and V apart (the CPU decoder), or as two views of the GPU decoder's interleaved plane
        uv = torch.stack([planes[1], planes[2]], -1)
    # Copies: the decoder may reuse its own memory, and a caller holds frames (prefetch, repeats at a gap).
    return gpu.GpuNV12(y.to(dev, copy=True), uv.contiguous().to(dev))


def _decoded(demuxer, decoder, dev):
    for packet in demuxer:
        for f in decoder.decode(packet):
            yield f.pts_seconds, f
    for f in decoder.drain():
        yield f.pts_seconds, f


@functools.cache
def _first_good(path: str, at: float) -> float | None:
    """The time of the first frame that decodes properly after a seek to
    `at`, or None if there is none: the first one the software decoder
    gives. It decodes only up to that frame."""
    from torchcodec.decoders import Demuxer

    demuxer = Demuxer(path)
    decoder = demuxer.streams[0].make_decoder(device="cpu")
    demuxer.seek(at)
    decoder.reset()
    for packet in demuxer:
        for f in decoder.decode(packet):
            return f.pts_seconds
    for f in decoder.drain():
        return f.pts_seconds
    return None


def frames(path: Path, start: float, seconds: float, fps: float, width: int, height: int):
    """See the top of the file. Decoding starts LEAD_IN_S before `start`:
    a seek lands on what the file calls a keyframe, and Pixel phones call
    every frame one, so nothing decodes properly until the next real one
    (up to a second later). The software decoder and ffmpeg drop those
    frames; the GPU decoder hands them back damaged, so they are dropped
    here by time (_first_good). If the first good frame is still after
    `start`, it starts again from further back."""
    if not wanted():
        return None
    try:
        from torchcodec.decoders import Demuxer
    except Exception as e:  # not installed, or its ffmpeg libraries aren't found
        _tell(f"torchcodec didn't load ({type(e).__name__}: {e})")
        return None
    dev = accel.device()
    on = "cuda" if dev.type == "cuda" else "cpu"
    try:
        demuxer = Demuxer(str(path))
        stream = demuxer.streams[0]
        md = stream.metadata
        # The header's size is as displayed: turned for a file with a rotation flag (the frames aren't).
        if (md.width, md.height) not in ((width, height), (height, width)):
            return None  # asked for at another size: the pipe resizes
        if md.pixel_format not in ("yuv420p", "yuvj420p", "nv12"):
            _tell(f"{Path(path).name} is {md.pixel_format}, not 8-bit 4:2:0")
            return None
        decoder = stream.make_decoder(device=on)
        # The pipe's own arithmetic, so a frame half-way between two ticks goes the same way: ffmpeg is given
        # the start to four places, and shifts each frame's time by it in whole units of the time base.
        file_us, tb = _file_timing(str(path))
        shift = _half_up(Fraction(int(round(round(start, 4) * 1e6)) + file_us, 1_000_000) / tb)
        t0 = float(shift * tb)
        tick = 1 / fps
        for lead in LEAD_IN_S:
            at = max(0.0, t0 - lead)
            good = _first_good(str(path), at)
            if good is None:
                return iter(())  # a clip that starts past the file's end
            if good <= t0 + tick / 2 or at == 0.0:
                break
        else:
            _tell(f"{Path(path).name} gives no frame within {LEAD_IN_S[-1]:.0f} s before {start:.2f} s")
            return None
        demuxer.seek(at)
        decoder.reset()
        source = ((t, f) for t, f in _decoded(demuxer, decoder, dev) if t >= good - 1e-6)
        source = (((_half_up(Fraction(t) / tb) - shift) * tb, f) for t, f in source)  # exact, from the clip's start
        first = next(source, None)
        if first is None:
            return iter(())
        if first[1].bit_depth != 8 or (first[1].width, first[1].height) != (width, height):
            _tell(f"{Path(path).name} decodes as {first[1].width}x{first[1].height} {first[1].bit_depth}-bit")
            return None
    except Exception as e:
        _tell(f"couldn't open {Path(path).name} ({type(e).__name__}: {e})")
        return None

    def run():
        import itertools

        src = out = None
        for f in constant_rate(itertools.chain([first], source), 0, seconds, fps):
            if f is not src:  # a frame sent on several ticks (a gap in the file) is copied once
                src, out = f, _nv12(f, dev)
            yield out

    return run()


def constant_rate(frames, start: float, seconds: float, fps: float):
    """(time s, frame) in display order -> the frames the pipe's
    `ffmpeg -ss start -t seconds ... -vf fps=fps -f rawvideo` gives for a
    phone's file: frames before `start` are dropped, each later one lands
    on the output tick nearest its time (ticks are 1 / fps apart, counted
    from `start`; halves round up), a tick two frames land on keeps the
    later, a tick none lands on repeats the one before, and the ticks
    before the first frame's own get what that first tick ends up with.

    The clip has every tick before its end. The pipe sometimes stops one
    tick sooner (it rounds for an iPhone's file where it rounds up for a
    Pixel's), so this can give one frame more at the end, never fewer:
    follow.game_stream takes the number it asked for."""
    total = math.ceil(round(seconds, 4) * fps - 1e-6)  # every tick before the clip ends
    rate = Fraction(fps).limit_denominator(1001)
    held, tick, at = None, 0, 0  # the frame waiting to be sent, the tick it landed on, the next tick to send
    for t, frame in frames:
        x = (Fraction(t) - Fraction(start)) * rate
        if x < 0:
            continue
        k = _half_up(x)
        if held is not None and k > tick:  # nothing later lands on that tick: send it, for the empty ticks around it too
            for _ in range(at, min(k, total)):
                yield held
            at = k
        if k >= total:
            if held is None:
                return
            break
        held, tick = frame, k
    else:
        if held is not None:  # the file ended
            for _ in range(at, min(tick + 1, total)):
                yield held
        return
    for _ in range(at, total):
        yield held


def main(argv: list[str]) -> int:
    """`nvdec.py <video> [start s ...]`: this reader's frames against the
    ffmpeg pipe's, at a few places in the file. Says how many frames are
    the same to the sample, and how long each reader took."""
    import time

    import numpy as np

    import follow

    path = Path(argv[1])
    starts = [float(x) for x in argv[2:]] or [0.0, 37.21, 100.0]
    bad = 0
    for start, fps in [(s, f) for s in starts for f in (30, 10)]:
        got = {}
        for mode in ("0", "1"):
            os.environ["OSB_NVDEC"] = mode
            t = time.perf_counter()
            fr = list(follow.frame_stream(path, start, 3.0, follow.SRC_W, fps))
            got[mode] = ([np.asarray(f) for f in fr], time.perf_counter() - t, type(fr[0]).__name__ if fr else "-")
        a, b = got["0"][0], got["1"][0]
        same = sum(bool((x == y).all()) for x, y in zip(a, b))
        ok = len(b) in (len(a), len(a) + 1) and same == len(a) and got["1"][2] == "GpuNV12"  # constant_rate: one more at most
        bad += not ok
        print(f"{path.name} from {start:.2f} s at {fps}/s: pipe {len(a)} frames in {got['0'][1]:.2f} s, "
              f"direct {len(b)} {got['1'][2]} in {got['1'][1]:.2f} s, {same} the same: {'ok' if ok else 'DIFFERENT'}",
              flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
