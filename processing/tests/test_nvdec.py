# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""nvdec.constant_rate: which source frame goes on each output tick. The
rule is the ffmpeg pipe's on phone footage; `nvdec.py <video>` compares the
two readers' frames on a real file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import nvdec  # noqa: E402


def run(times, start, seconds, fps):
    return list(nvdec.constant_rate(((t, i) for i, t in enumerate(times)), start, seconds, fps))


def test_steady_frames_pass_through():
    assert run([i / 30 for i in range(90)], 0.0, 2.0, 30) == list(range(60))


def test_frames_before_the_start_are_dropped():
    assert run([i / 30 for i in range(90)], 1.0, 1.0, 30) == list(range(30, 60))


def test_a_gap_repeats_the_frame_before_it():
    times = [0.0, 1 / 30, 5 / 30, 6 / 30]
    assert run(times, 0.0, 7 / 30, 30) == [0, 1, 1, 1, 1, 2, 3]


def test_two_frames_on_one_tick_keep_the_later():
    times = [0.0, 0.030, 0.036, 0.0667]
    assert run(times, 0.0, 0.1, 30) == [0, 2, 3]


def test_a_lower_rate_drops_frames():
    assert run([i / 30 for i in range(60)], 0.0, 2.0, 10) == list(range(1, 60, 3))  # of the frames nearest a tick, the last


def test_a_late_first_frame_fills_the_ticks_before_it():
    # As a Pixel segment does: its first frame is 0.2434 s in (tick 7).
    times = [0.2434 + i / 30 for i in range(30)]
    assert run(times, 0.0, 0.5, 30) == [0] * 8 + list(range(1, 8))


def test_the_ticks_before_the_first_get_the_frame_that_tick_keeps():
    # A Pixel segment at 10 a second: three frames land on tick 3, and the pipe sends the last of them from tick 0.
    times = [0.2551 + i / 30 for i in range(30)]
    assert run(times, 0.0, 0.6, 10) == [2, 2, 2, 2, 5, 8]


def test_the_clip_is_exactly_as_long_as_asked_when_the_file_goes_on():
    assert len(run([i / 30 + 0.01 for i in range(300)], 2.0, 3.0, 30)) == 90


def test_a_file_that_ends_early_gives_what_it_has():
    assert run([i / 30 for i in range(10)], 0.0, 1.0, 30) == list(range(10))


def test_a_frame_already_on_the_gpu_uploads_as_the_flat_one_does():
    import numpy as np
    import torch

    import gpu

    w, h = 64, 36
    flat = np.random.default_rng(3).integers(0, 256, w * h * 3 // 2, dtype=np.uint8)
    t = torch.from_numpy(flat)
    held = gpu.GpuNV12(t[: w * h].view(h, w), t[w * h:].view(h // 2, w // 2, 2))
    dev = torch.device("cpu")
    for a, b in zip(gpu.upload_nv12(flat, w, h, dev), gpu.upload_nv12(held, w, h, dev)):
        assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    assert (np.asarray(held) == flat).all()
