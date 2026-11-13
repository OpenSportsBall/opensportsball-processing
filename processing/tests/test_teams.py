# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""teams.py's sides, which nobody is asked to check: roles.py must always
find a sides.json.

    cd processing && uv run --extra dev pytest tests/test_teams.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import teams  # noqa: E402


def game(groups, moments=20):
    """People per moment for each (component, headcount, [Lab a, Lab b, saturation, value])."""
    rng = np.random.default_rng(0)
    people, X, comp = [], [], []
    for m in range(moments):
        for c, n, feat in groups:
            for _ in range(n):
                people.append({"frame": f"m{m}_main", "onPitch": True, "box": [0, 0, 20, 80]})
                X.append(np.r_[np.array(feat) + rng.normal(0, 1, 4), np.zeros(teams.HUE_BINS)])
                comp.append(c)
    return people, np.array(X), comp


BLUE, RED, WHITE, REFEREE = [10, -45, 60, 50], [50, 30, 70, 50], [0, 0, 8, 90], [-30, 40, 70, 60]


def test_two_kits_are_the_two_sides(tmp_path):
    people, X, comp = game([(0, 10, BLUE), (1, 10, RED), (2, 1, REFEREE)])
    teams.sides(tmp_path, people, X, comp, 20, 3)
    s = json.loads((tmp_path / "sides.json").read_text())
    assert sorted([s["A"], s["B"]]) == [[0], [1]] and s["keeperOrReferee"] == [2]


def test_one_kit_found_still_leaves_sides_for_roles(tmp_path):
    """rec5: white is under the saturation floor, so only one kit component. The kit found is
    side A, side B is empty, and the referee's group is still marked."""
    people, X, comp = game([(0, 10, BLUE), (1, 10, WHITE), (2, 1, REFEREE)])
    teams.sides(tmp_path, people, X, comp, 20, 3)
    s = json.loads((tmp_path / "sides.json").read_text())
    assert (s["A"], s["B"], s["keeperOrReferee"]) == ([0], [], [2])
