# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""follow.NV12Encoder writes frames from a thread of its own: the same
bytes in the same order as writing them in the caller, whether a frame
comes as an array or as a tensor, and a failed write reaches the caller."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import follow  # noqa: E402


class Pipe:
    def __init__(self, fail_after=None, delay=0.0):
        self.got, self.fail_after, self.delay, self.closed = [], fail_after, delay, False

    def write(self, b):
        if self.fail_after is not None and len(self.got) >= self.fail_after:
            raise BrokenPipeError("the encoder went away")
        time.sleep(self.delay)
        self.got.append(bytes(b))

    def close(self):
        self.closed = True


class Proc:
    def __init__(self, pipe, rc=0):
        self.stdin, self.rc = pipe, rc

    def wait(self):
        return self.rc


def encoder(monkeypatch, pipe, rc=0):
    monkeypatch.setattr(follow.subprocess, "Popen", lambda *a, **k: Proc(pipe, rc))
    return follow.NV12Encoder(Path("out.mp4"), 16, 8)


def frames(n):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, 16 * 8 * 3 // 2, dtype=np.uint8) for _ in range(n)]


def test_frames_arrive_whole_and_in_order(monkeypatch):
    pipe = Pipe(delay=0.002)  # slower than the caller: the queue fills and the caller waits
    enc = encoder(monkeypatch, pipe)
    sent = frames(40)
    for k, f in enumerate(sent):
        enc.write(torch.from_numpy(f) if k % 2 else f)  # a tensor (pass B's) or an array
    enc.close()
    assert pipe.got == [f.tobytes() for f in sent] and pipe.closed
    assert enc.busy_s > 0


def test_a_failed_write_reaches_the_caller(monkeypatch):
    enc = encoder(monkeypatch, Pipe(fail_after=3))
    with pytest.raises(RuntimeError, match="encoder stopped"):
        for f in frames(200):  # far more than the queue holds: the caller must not be left waiting
            enc.write(f)
            time.sleep(0.001)
        enc.close()
    enc2 = encoder(monkeypatch, Pipe(), rc=1)
    enc2.write(frames(1)[0])
    with pytest.raises(RuntimeError, match="exited with 1"):
        enc2.close()
