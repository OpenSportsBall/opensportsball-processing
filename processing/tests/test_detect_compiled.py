# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""CompiledDetector: every call's crops reach a predictor of a fixed batch
size, padded, and each crop gets its own result back in order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "spikes"))
import detect  # noqa: E402


class Fixed:
    def __init__(self, b, seen):
        self.b, self.seen = b, seen

    def predict(self, crops, threshold):
        assert len(crops) == self.b
        self.seen.append(self.b)
        return [f"r{c}" for c in crops] if self.b > 1 else f"r{crops[0]}"


class Source:
    def __init__(self):
        self.seen = []

    def inference(self, compile, batch_size):
        assert compile
        return Fixed(batch_size, self.seen)


def test_each_crop_gets_its_result_whatever_the_count():
    src = Source()
    det = detect.CompiledDetector(src, [8, 1, 2, 4])
    for n, batches in [(1, [1]), (2, [2]), (3, [4]), (4, [4]), (5, [8]), (8, [8]), (9, [8, 1]), (19, [8, 8, 4])]:
        src.seen.clear()
        assert det.predict(list(range(n)), 0.1) == [f"r{i}" for i in range(n)]
        assert src.seen == batches
    assert det.predict([], 0.1) == []


def test_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("OSB_DETECTOR_COMPILED", raising=False)
    monkeypatch.delenv("OSB_DETECTOR_HALF", raising=False)
    m = object()
    assert detect.compiled(m, "ball") is m and detect.half_precision(m, "ball") is m
    monkeypatch.setenv("OSB_DETECTOR_HALF", "people")
    assert detect.half_precision(m, "ball") is m


def test_tensorrt_off_unless_asked_for_and_only_with_a_gpu(monkeypatch):
    m = object()
    monkeypatch.delenv("OSB_DETECTOR_TRT", raising=False)
    assert detect.tensorrt(m, "ball", None) is m
    monkeypatch.setenv("OSB_DETECTOR_TRT", "people=fp16:64")
    assert detect.tensorrt(m, "ball", None) is m
    monkeypatch.setattr(detect.torch.cuda, "is_available", lambda: False)
    assert detect.tensorrt(m, "people", None) is m


def test_tensorrt_falls_back_when_an_engine_cannot_be_made(monkeypatch, tmp_path, capsys):
    m = object()
    monkeypatch.setattr(detect.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("OSB_TRT_CACHE", str(tmp_path))
    monkeypatch.setenv("OSB_DETECTOR_TRT", "ball=fp32:1,2;people=fp16:64")

    def no_engine(self, make, sizes, half, cache, ckpt=None):
        assert (sizes, half, cache, ckpt) == ([1, 2], False, tmp_path, None)
        raise RuntimeError("no TensorRT here")
    monkeypatch.setattr(detect.TrtDetector, "__init__", no_engine)
    assert detect.tensorrt(m, "ball", None) is m
    assert "TensorRT unavailable for the ball detector" in capsys.readouterr().out


def test_tensorrt_engines_are_routed_like_compiled_ones(monkeypatch, tmp_path):
    monkeypatch.setattr(detect.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("OSB_TRT_CACHE", str(tmp_path))
    monkeypatch.setenv("OSB_DETECTOR_TRT", "people=fp16:4")
    seen = []

    def engines(self, make, sizes, half, cache, ckpt=None):
        assert half
        self.parts = {b: Fixed(b, seen) for b in sizes}
    monkeypatch.setattr(detect.TrtDetector, "__init__", engines)
    det = detect.tensorrt(object(), "people", None)
    assert det.predict(list(range(6)), 0.1) == [f"r{i}" for i in range(6)] and seen == [4, 4]
    # neither of the other two switches touches it
    monkeypatch.setenv("OSB_DETECTOR_HALF", "1")
    monkeypatch.setenv("OSB_DETECTOR_COMPILED", "people=1")
    assert detect.half_precision(detect.compiled(det, "people"), "people") is det


def test_a_bad_tensorrt_setting_stops_the_stage(monkeypatch):
    import pytest
    monkeypatch.setattr(detect.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("OSB_DETECTOR_TRT", "ball=half:1")
    with pytest.raises(SystemExit):
        detect.tensorrt(object(), "ball", None)
