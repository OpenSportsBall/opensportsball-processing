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
