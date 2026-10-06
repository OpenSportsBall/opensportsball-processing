# SPDX-FileCopyrightText: 2026 The SportsBallVideo Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""Detector registry. Backends are imported lazily so an install only needs
the optional dependency for the backend it actually uses."""

from __future__ import annotations

from .base import Detection, Detector, Label

BACKENDS = {
    "ultralytics": "AGPL-3.0, extra: ultralytics",
    "rfdetr": "Apache-2.0, extra: rfdetr",
}


def load_detector(backend: str, weights: str) -> Detector:
    if backend not in BACKENDS:
        raise ValueError(f"unknown detector backend {backend!r}; choose from {sorted(BACKENDS)}")
    # Backends land in Phase 3 (players) and Phase 4 (ball).
    raise NotImplementedError(f"detector backend {backend!r} is not implemented yet")


__all__ = ["BACKENDS", "Detection", "Detector", "Label", "load_detector"]
