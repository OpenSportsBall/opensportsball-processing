#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Phase 0 spike: the match ball as one minimum-cost path through a clip's
candidates, solved offline (Viterbi) instead of ball.py's greedy online
chain.

    uv run processing/spikes/ballpath.py footage/rec3 footage/rec3/detect/ball_9_60_120.win.json [out.json]

Reads a ballclip.py / ballwin.py candidate file, filters it exactly as
ball.py does (usable, static_spots) and writes ball.json in ball.py's
per-frame format (default <recording>/ball/ball.json). balleval.py and
follow.py use this tracker in place of ball.py's link + choose (the
default since 2026-10-19; OSB_BALL_CHAIN=1 gives the chain back), so the
two are compared on the same candidates with the same filters: only the
linking differs. It finds the ball at
more labelled moments than the chain but cost two of 13 play boxes
(docs/TESTING-PROCESSING.md section 2), so the chain stays the default.

The graph: per frame, one node per usable candidate plus a "no ball"
node. A candidate costs -log(conf x judge) (NEAR_W less when a player is
within NEAR_PLAYER_M); "no ball" costs NONE_COST per frame. An edge
between candidates k frames apart (k <= SKIP_MAX) costs the motion plus
SKIP_RATIO x NONE_COST per skipped frame. Motion between candidates seen
by the same phone is angular, in that phone's view (yaw and pitch per
frame over MOTION_DEG, STEEP_DEG per degree beyond MAX_DEG): a ball in
the air is smooth there, where its ground-plane projection runs out and
back along the camera's ray at 5-14 m per frame. Between phones the
motion is on the pitch plane (per-frame displacement over MOTION_M,
STEEP per metre beyond MAX_SPEED x dt) plus CROSS_COST. Entering or
leaving "no ball" costs GAP_PENALTY, so the path bridges a gap of a few
seconds rather than restarting on a stray candidate, and restarts
anywhere after a long one. Skipped frames inside a skip edge are
interpolated up to BRIDGE_S, as ball.py bridges; SMOOTH_S > 0 smooths
the positions along each run of the path, HOLD_S > 0 holds a run's end
positions into the frames around it. STILL_COST > 0 charges a step
under STILL_DEG (a parked spare ball; ball.py's "moving" score).

~1200 frames x a few candidates: a plain dynamic programme, under a
second per clip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ball as B  # noqa: E402

NONE_COST = 2.0       # per frame with no ball on the path; must exceed a real run's node + motion cost per frame
                      # (~0.3 + ~0.5) by a margin or short confident runs are dropped (1.0 lost the clip's opening track)
GAP_PENALTY = 2.0     # to enter and again to leave "no ball" (1-3 are alike; 6+ loses found)
MOTION_DEG = 2.0      # same phone: per-frame angular step that costs 1 (linear; a pass at 50 m is 1-2 deg)
MAX_DEG = 8.0         # ... gentle up to here (a shot across the near view), STEEP_DEG per degree beyond
STEEP_DEG = 2.0
MOTION_M = 2.0        # between phones: per-frame pitch-plane step that costs 1 (linear)
STEEP = 4.0           # ... per metre of per-frame step beyond MAX_SPEED x dt
CROSS_COST = 1.0      # switching phone
SKIP_MAX = 10         # a candidate may follow one up to this many frames earlier directly (1: consecutive only)
SKIP_RATIO = 1.0      # each frame skipped by such an edge costs this x NONE_COST
NEAR_W = 0.0          # a candidate within NEAR_PLAYER_M of a player costs this much less
STILL_DEG = 0.1       # a same-phone step under this is "still" ...
STILL_COST = 0.0      # ... and costs this (0: off)
BRIDGE_S = 1.0        # frames skipped by an edge are interpolated when the skip is at most this long
HOLD_S = 0.0          # each run's end positions are held this long into the frames before and after it (0: off)
SMOOTH_S = 0.0        # half-width of a triangular smoothing of the positions along a run (0: none)
MAX_SPEED = B.MAX_SPEED


def node_costs(cands, people):
    out = []
    for i, fr in enumerate(cands):
        c = np.array([-np.log(max(1e-6, d["conf"] * d.get("judge", 1.0))) for d in fr], float)
        if NEAR_W and len(fr) and len(people[i]):
            P_ = np.array([d["xz"] for d in fr])
            near = np.array([np.hypot(*(people[i] - p).T).min() for p in P_]) < B.NEAR_PLAYER_M
            c = c - NEAR_W * near
        out.append(c)
    return out


def motion_cost(pa, pi, aa, ai, pha, phi, k, fps):
    """Candidates k frames apart: pitch-plane points, view angles (yaw,
    pitch in degrees) and phones of the earlier frame (rows) and the
    later (columns)."""
    same = pha[:, None] == phi[None, :]
    ang = np.hypot(*(aa[:, None, :] - ai[None, :, :]).transpose(2, 0, 1)) / k
    c_same = ang / MOTION_DEG + STEEP_DEG * np.maximum(0.0, ang - MAX_DEG)
    if STILL_COST:
        c_same = c_same + STILL_COST * (ang < STILL_DEG)
    step = np.hypot(*(pa[:, None, :] - pi[None, :, :]).transpose(2, 0, 1)) / k
    c_cross = step / MOTION_M + STEEP * np.maximum(0.0, step - MAX_SPEED / fps) + CROSS_COST
    return np.where(same, c_same, c_cross) + SKIP_RATIO * NONE_COST * (k - 1)


def path(cands, people, n, fps):
    """The minimum-cost path: per frame the index of the chosen candidate
    (-1: no ball) and, for a candidate, the frame of the candidate it
    follows (for bridging)."""
    node = node_costs(cands, people)
    pts = [np.array([d["xz"] for d in fr], float).reshape(-1, 2) for fr in cands]
    ang = [np.array([(d["yaw"], d["pitch"]) for d in fr], float).reshape(-1, 2) for fr in cands]
    phone = [np.array([d["phone"] for d in fr], int) for fr in cands]
    best = [np.full(len(fr), np.inf) for fr in cands]   # cheapest path ending on candidate j of frame i
    back = [np.full((len(fr), 2), -1, int) for fr in cands]  # (frame, index) it came from; index -1 = "no ball"
    best_none = np.full(n, np.inf)
    back_none = np.full(n, -1, int)   # candidate index at the previous frame, or -1 (stayed in "no ball")
    for i in range(n):
        m = len(cands[i])
        if i == 0:
            best_none[0] = NONE_COST
            if m:
                best[0] = node[0].copy()
                back[0][:, 0] = -1
            continue
        # "no ball" at i: stayed, or left a candidate at i - 1.
        best_none[i] = best_none[i - 1] + NONE_COST
        if len(cands[i - 1]):
            j = int(np.argmin(best[i - 1]))
            v = best[i - 1][j] + GAP_PENALTY + NONE_COST
            if v < best_none[i]:
                best_none[i], back_none[i] = v, j
        if not m:
            continue
        # A candidate at i: from "no ball" at i - 1 ...
        b = np.full(m, best_none[i - 1] + GAP_PENALTY)
        bk = np.full((m, 2), -1, int)
        bk[:, 0] = i - 1
        # ... or from a candidate up to SKIP_MAX frames back.
        for k in range(1, SKIP_MAX + 1):
            a = i - k
            if a < 0:
                break
            if not len(cands[a]):
                continue
            c = best[a][:, None] + motion_cost(pts[a], pts[i], ang[a], ang[i], phone[a], phone[i], k, fps)
            ja = c.argmin(0)
            v = c[ja, np.arange(m)]
            better = v < b
            b[better] = v[better]
            bk[better, 0] = a
            bk[better, 1] = ja[better]
        best[i] = b + node[i]
        back[i] = bk
    # Backtrack from the cheapest end.
    choice = np.full(n, -1, int)
    prev_frame = np.full(n, -1, int)
    i = n - 1
    j = int(np.argmin(best[i])) if len(cands[i]) and best[i].min() < best_none[i] else -1
    while i >= 0:
        if j < 0:
            j = back_none[i]
            i -= 1
            continue
        choice[i] = j
        a, ja = back[i][j]
        prev_frame[i] = a
        i, j = a, ja
    return choice, prev_frame


def track(cands, people, n, fps):
    """ball.py's link + choose in one: per frame an estimate (or None) and
    its source (run id, detection) where it rests on a detection."""
    choice, prev_frame = path(cands, people, n, fps)
    est = [None] * n
    src = [None] * n
    run = 0
    for i in range(n):
        if choice[i] < 0:
            continue
        a = prev_frame[i]
        if a < 0 or choice[a] < 0:  # entered from "no ball" or the clip's start
            run += 1
        d = cands[i][choice[i]]
        est[i] = np.array(d["xz"], float)
        src[i] = (run, d)
    # Bridge the frames a skip edge jumped over, up to BRIDGE_S.
    for i in range(n):
        a = prev_frame[i]
        if choice[i] >= 0 and a >= 0 and choice[a] >= 0 and 1 < i - a <= BRIDGE_S * fps:
            for m in range(a + 1, i):
                w = (m - a) / (i - a)
                est[m] = (1 - w) * est[a] + w * est[i]
    if SMOOTH_S > 0:
        est = smooth(est, fps)
    if HOLD_S > 0:
        h = int(round(HOLD_S * fps))
        held = list(est)
        for i in range(n):
            if est[i] is None:
                continue
            for m in range(max(0, i - h), min(n, i + h + 1)):
                if est[m] is None and held[m] is None:
                    held[m] = est[i]
        est = held
    return est, src


def smooth(est, fps):
    """A triangular window over each run of estimates (a run: consecutive
    frames with an estimate)."""
    h = max(1, int(round(SMOOTH_S * fps)))
    w = np.array([h + 1 - abs(k) for k in range(-h, h + 1)], float)
    out = list(est)
    i, n = 0, len(est)
    while i < n:
        if est[i] is None:
            i += 1
            continue
        j = i
        while j < n and est[j] is not None:
            j += 1
        if j - i >= 3:
            P_ = np.array(est[i:j])
            for m in range(j - i):
                lo, hi = max(0, m - h), min(j - i, m + h + 1)
                ww = w[lo - m + h:hi - m + h]
                out[i + m] = (P_[lo:hi] * ww[:, None]).sum(0) / ww.sum()
        i = j
    return out


def main(rec: Path, cands_path: Path, out: Path | None) -> None:
    import pitch as P

    pose = P.load_pose(rec)
    data = json.loads(cands_path.read_text())
    fps, n = data["fps"], len(data["frames"])
    cands = [[d for d in f["detections"] if B.usable(d, pose)] for f in data["frames"]]
    spots = B.static_spots(cands, fps, pose=pose)
    keep = [[d for d in fr if all(np.hypot(*(np.array(d["xz"]) - s)) > B.STATIC_RADIUS for s in spots)]
            for fr in cands]
    people = [np.zeros((0, 2))] * n
    est, src = track(keep, people, n, fps)
    direct = sum(s is not None for s in src)
    covered = sum(e is not None for e in est)
    runs = len({s[0] for s in src if s is not None})
    print(f"{sum(map(len, cands))} usable candidates in {n} frames; {len(spots)} static spots removed"
          f" ({sum(map(len, cands)) - sum(map(len, keep))} candidates)")
    print(f"path: {runs} runs; ball estimate in {covered}/{n} frames ({100 * covered / n:.0f}%), {direct} from a detection")
    out = out or rec / "ball" / "ball.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"fps": fps, "frames": [
        {"t": f["t"], "xz": None if e is None else [round(float(v), 2) for v in e],
         "detected": s is not None} for f, e, s in zip(data["frames"], est, src)]}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
