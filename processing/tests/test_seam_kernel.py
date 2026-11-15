# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""seam.CELL is the GPU seam search's arithmetic. Built for the CPU it must
give seam_dp's seam, frame after frame, each pulled towards the last."""
import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import seam  # noqa: E402


def test_the_kernels_c_finds_seam_dps_seam(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("no C compiler")
    (tmp_path / "s.c").write_text(seam.CPU)
    subprocess.run([cc, "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(tmp_path / "s.so"),
                    str(tmp_path / "s.c")], check=True)
    fn = ctypes.CDLL(str(tmp_path / "s.so")).seam_frame
    p = ctypes.c_void_p
    fn.argtypes = [p, p, p, p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
    rng = np.random.default_rng(3)
    H, W, lam = 120, 90, 0.006
    back = np.zeros(H * W, np.int8)
    want = mine = None
    for i in range(60):
        cost = rng.random((H, W), dtype=np.float32)
        if i % 3 == 0:  # ties: whole stretches of equal cost, as outside the overlap
            cost[:, : W // 2] = 50.0
        if i % 7 == 0:
            cost[rng.integers(0, H, 40), rng.integers(0, W, 40)] += 3.0  # players
        want = seam.seam_dp(cost, want, lam)
        path = np.zeros(H, np.int32)
        prev = mine if mine is not None else path
        fn(cost.ctypes.data, prev.ctypes.data, path.ctypes.data, back.ctypes.data, H, W, lam, int(mine is not None))
        mine = path
        assert np.array_equal(want, mine), f"frame {i}"
