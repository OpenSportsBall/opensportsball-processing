# SPDX-FileCopyrightText: 2026 The SportsBallVideo Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Detector interface.

Every detector backend implements :class:`Detector`. The rest of the
pipeline depends only on this module, never on a backend's package, so a
backend can be swapped without touching tracking or rendering. Keeping an
Apache-2.0 backend working preserves the option to offer this package under
a commercial license (docs/LICENSING.md rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import numpy as np


class Label(str, Enum):
    BALL = "ball"
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"


@dataclass(frozen=True)
class Detection:
    label: Label
    score: float
    # Box in panorama pixel coordinates.
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detector(Protocol):
    name: str

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Detect objects in a batch of RGB frames (H x W x 3, uint8).

        Frames are tiles of the stitched panorama; the caller handles tiling
        and maps boxes back to panorama coordinates. Returns one list of
        detections per input frame.
        """
        ...
