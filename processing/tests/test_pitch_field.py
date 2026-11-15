# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""follow.pitch_field: pass A keeps blobs where the fitted pitch is, not
where the background happens to look green (it doesn't, under floodlights)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import follow  # noqa: E402
import pitch as P  # noqa: E402


def test_the_mask_is_the_pitch_and_a_margin(tmp_path):
    (tmp_path / "pitch").mkdir()
    (tmp_path / "pitch" / "pitch.json").write_text(json.dumps({
        "cameraXZ": [0.0, -40.0], "heightM": 4.0, "rotRad": 0.0, "tiltRad": [0.0, 0.0],
        "lengthM": 100.0, "widthM": 60.0, "markingScale": 1.0}))
    pose = P.load_pose(tmp_path)
    spots = np.array([[0.0, 0.0], [49.0, 29.0], [-52.0, 0.0],   # on the pitch; in a corner; 2 m behind a goal line
                      [-56.0, 0.0], [0.0, 36.0]])               # 6 m behind it; 6 m past the far touchline
    ang = np.radians(pose.angles(spots))
    YY, PP = ang[:, 0][None], ang[:, 1][None]
    assert follow.pitch_field(tmp_path, YY, PP).tolist() == [[1, 1, 1, 0, 0]]
    sky = follow.pitch_field(tmp_path, np.zeros((1, 1)), np.full((1, 1), -0.3 if ang[0, 1] > 0 else 0.3))
    assert sky.tolist() == [[0]]


def test_no_fitted_pitch_no_mask(tmp_path):
    assert follow.pitch_field(tmp_path, np.zeros((1, 1)), np.zeros((1, 1))) is None
