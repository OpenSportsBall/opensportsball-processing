#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Phase 0 spike: follow the match ball through the detector's candidates.

Run after detect.py and pitch.py (it reads detect/detections_pitch.json):

    uv run processing/spikes/ball.py footage/rec3

The detector finds the match ball only in bursts, at low confidence, among
spare balls and noise. This keeps candidates on the pitch, drops places
where candidates keep appearing without moving (spare balls, cones, paint),
links the rest into tracks with a constant-velocity prediction and a
physical speed limit, scores the tracks (long, moving, near players), and
picks the best one at each moment, bridging short gaps.

Positions are where the ray meets the ground, so a ball in the air
projects too far away; tracks tolerate that through the speed limit only.

Writes <recording>/ball/: ball.json (per-frame estimate), minimap.png and
crops.jpg (the detector's picture at chosen positions, to check by eye).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pitch as P  # noqa: E402

MAX_SPEED = 35.0          # m/s, a hard shot
GATE_M = 2.0              # association slack on top of speed x dt, metres
MAX_MISSED_S = 0.6        # a track survives this long without a detection (balleval.py on the clip sets)
STATIC_RADIUS = 1.5       # metres
STATIC_MIN_S = 30.0       # a spot seen over this long without moving is static (a ball placed for a goal
                          # kick sits still for 10 s or more; 4 s removed it) ...
STATIC_MIN_HITS = 10      # ... seen at least this often (a bench ball is only seen when a scan passes)
STATIC_LOITER_S = 8.0     # in a loiter cell (bench, spectators) or outside the lines, a ball still this long
STATIC_LOITER_HITS = 3    # with this many sightings is a spare ball (a ball that went out comes back in seconds)
NEAR_PLAYER_M = 6.0       # the match ball is usually this close to someone
BRIDGE_S = 1.0            # interpolate gaps up to this long
PITCH_MARGIN_M = 10.0     # candidates this far outside the lines still count (balls at the corners and touchlines,
                          # and far balls whose depth is metres out: 3 lost 6 labelled match balls, 2026-10-28)
MIN_HITS = 2              # a track shorter than this scores nothing
MIN_JUDGE = 0.3           # candidates scored below this by balljudge.py are ignored (0: judge not used)
BODY_FRAC = 0.8           # candidates in the top 80% of a person's box are kit, hands, knees


def on_body(ball: dict, people: list[dict]) -> bool:
    """A ball candidate inside the upper part of a person seen by the same
    phone: the detector mistakes shorts, hands and knees for balls. The
    feet, where the match ball usually is, stay allowed."""
    bx = (ball["box"][0] + ball["box"][2]) / 2
    by = (ball["box"][1] + ball["box"][3]) / 2
    for p in people:
        if p["phone"] != ball["phone"]:
            continue
        x1, y1, x2, y2 = p["box"]
        if x1 <= bx <= x2 and y1 <= by <= y1 + BODY_FRAC * (y2 - y1):
            return True
    return False


def usable(d: dict, pose) -> bool:
    """A candidate placed on the pitch, within PITCH_MARGIN_M of the lines,
    and not judged low (balljudge.py)."""
    if d["xz"] is None or d.get("judge", 1.0) < MIN_JUDGE:
        return False
    return abs(d["xz"][0]) <= pose.L / 2 + PITCH_MARGIN_M and abs(d["xz"][1]) <= pose.W / 2 + PITCH_MARGIN_M


def load(rec: Path):
    data = json.loads((rec / "detect" / "detections_pitch.json").read_text())
    fps = data["fps"]
    frames = data["frames"]
    balls, people = [], []
    body = 0
    for i, f in enumerate(frames):
        persons = [d for d in f["detections"] if d["cls"] == "person"]
        cands = [d for d in f["detections"] if d["cls"] == "ball" and d["onPitch"]]
        kept = [d for d in cands if not on_body(d, persons)]
        body += len(cands) - len(kept)
        balls.append(kept)
        people.append(np.array([d["xz"] for d in f["detections"] if d["cls"] == "person" and d["onPitch"]])
                      .reshape(-1, 2))
    print(f"{body} candidates on players' bodies dropped")
    return fps, frames, balls, people


def static_spots(balls, fps, loiter=None, pose=None):
    """Places with candidates spread over STATIC_MIN_S seconds that never
    left STATIC_RADIUS: grid the pitch, keep cells whose first and last
    sighting are far apart in time and whose sightings fill the span."""
    pts = [(i, *d["xz"]) for i, fr in enumerate(balls) for d in fr]
    spots = []
    if not pts:
        return spots
    a = np.array(pts)
    used = np.zeros(len(a), bool)
    for k in np.argsort(a[:, 0]):
        if used[k]:
            continue
        near = np.hypot(a[:, 1] - a[k, 1], a[:, 2] - a[k, 2]) < STATIC_RADIUS
        t = a[near, 0]
        span = (t.max() - t.min()) / fps
        in_loiter = loiter is not None and loiter(a[k, 1], a[k, 2])  # follow.py's loiter_cells
        if pose is not None and (abs(a[k, 1]) > pose.L / 2 or abs(a[k, 2]) > pose.W / 2):
            in_loiter = True  # outside the lines
        min_s, min_hits = (STATIC_LOITER_S, STATIC_LOITER_HITS) if in_loiter else (STATIC_MIN_S, STATIC_MIN_HITS)
        if span >= min_s and len(np.unique(t)) >= min_hits:
            spots.append(a[near, 1:].mean(0))
            used |= near
    return spots


class Track:
    def __init__(self, tid, i, d):
        self.id = tid
        self.hits = [(i, np.array(d["xz"]), d["conf"], d)]
        self.v = np.zeros(2)
        self.last = i

    def predict(self, i, fps):
        _, p, _, _ = self.hits[-1]
        return p + self.v * (i - self.last) / fps

    def add(self, i, d, fps):
        _, p, _, _ = self.hits[-1]
        q = np.array(d["xz"])
        v = (q - p) * fps / max(1, i - self.last)
        self.v = 0.5 * self.v + 0.5 * v if len(self.hits) > 1 else v
        self.hits.append((i, q, d["conf"], d))
        self.last = i


def link(balls, fps):
    """Greedy nearest-neighbour tracking with constant-velocity prediction."""
    tracks, alive, nid = [], [], 0
    for i, cands in enumerate(balls):
        alive = [t for t in alive if (i - t.last) / fps <= MAX_MISSED_S]
        free = list(range(len(cands)))
        pairs = []
        for t in alive:
            pred = t.predict(i, fps)
            gate = GATE_M + MAX_SPEED * (i - t.last) / fps
            for k in free:
                dist = np.hypot(*(np.array(cands[k]["xz"]) - pred))
                if dist <= gate:
                    pairs.append((dist, t, k))
        taken_t, taken_k = set(), set()
        for dist, t, k in sorted(pairs, key=lambda x: x[0]):
            if t.id in taken_t or k in taken_k:
                continue
            t.add(i, cands[k], fps)
            taken_t.add(t.id)
            taken_k.add(k)
        for k in free:
            if k not in taken_k:
                t = Track(nid, i, cands[k])
                nid += 1
                tracks.append(t)
                alive.append(t)
    return tracks


def score(t: Track, people) -> float:
    """Long, moving, confident and among players scores high."""
    n = len(t.hits)
    if n < MIN_HITS:
        return 0.0
    pos = np.array([h[1] for h in t.hits])
    travel = np.hypot(*(pos.max(0) - pos.min(0)))
    conf = float(np.mean([h[2] for h in t.hits]))
    near = []
    for i, p, _, _ in t.hits:
        pp = people[i]
        near.append(np.hypot(*(pp - p).T).min() if len(pp) else 99.0)
    near_frac = float(np.mean(np.array(near) < NEAR_PLAYER_M))
    moving = min(1.0, travel / 3.0)
    return n * conf * moving * (0.2 + near_frac)


GAP_COST = 0.5            # score lost per second the chain has no track
CHAIN_SPEED = 25.0        # m/s the ball may cover between one track's end and the next's start


def choose(tracks, people, n_frames, fps):
    """One physically consistent chain of tracks over the clip: dynamic
    programming over tracks ordered by start, where a track may follow
    another only if it starts after it ends and the ball could have covered
    the distance in between. Short gaps along the chain are interpolated."""
    scored = [(score(t, people), t) for t in tracks]
    cand = sorted([(s, t) for s, t in scored if s > 0], key=lambda x: x[1].hits[0][0])
    best = [s for s, _ in cand]
    prev = [-1] * len(cand)
    for j, (sj, tj) in enumerate(cand):
        start, p0 = tj.hits[0][0], tj.hits[0][1]
        for i in range(j):
            si, ti = cand[i]
            end, p1 = ti.hits[-1][0], ti.hits[-1][1]
            gap = (start - end) / fps
            if gap <= 0:
                continue
            if np.hypot(*(p0 - p1)) > GATE_M + CHAIN_SPEED * gap:
                continue
            v = best[i] + sj - GAP_COST * gap
            if v > best[j]:
                best[j], prev[j] = v, i
    est = [None] * n_frames
    src = [None] * n_frames
    if not cand:
        return est, src, sorted(scored, key=lambda x: -x[0])
    j = int(np.argmax(best))
    chain = []
    while j >= 0:
        chain.append(cand[j][1])
        j = prev[j]
    for t in chain:
        for i, p, c, d in t.hits:
            est[i] = p
            src[i] = (t.id, d)
    known = [i for i in range(n_frames) if est[i] is not None]
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= BRIDGE_S * fps:
            for i in range(a + 1, b):
                w = (i - a) / (b - a)
                est[i] = (1 - w) * est[a] + w * est[b]
    print(f"chain of {len(chain)} tracks, frames {min(known)}-{max(known)}")
    return est, src, sorted(scored, key=lambda x: -x[0])


def minimap(rec, pose, people, est, spots, path):
    s, pad = 8, 40
    W_, H_ = int(pose.L * s) + 2 * pad, int(pose.W * s) + 2 * pad

    def px(x, z):
        return int(pad + (x + pose.L / 2) * s), int(pad + (pose.W / 2 - z) * s)

    img = np.full((H_, W_, 3), (60, 120, 60), np.uint8)
    for x, z in P.pitch_lines(pose.L, pose.W, pose.k, step=0.1):
        cv2.circle(img, px(x, z), 1, (255, 255, 255), -1)
    for pp in people:
        for x, z in pp:
            cv2.circle(img, px(x, z), 1, (170, 200, 170), -1)
    for x, z in spots:
        cv2.circle(img, px(x, z), int(STATIC_RADIUS * s), (0, 0, 0), 2)
    pts = [px(*p) for p in est if p is not None]
    n = len(est)
    for i in range(1, n):
        if est[i] is not None and est[i - 1] is not None:
            hue = int(150 * i / max(1, n - 1))
            col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.line(img, px(*est[i - 1]), px(*est[i]), col, 3)
    cv2.imwrite(str(path), img)
    return len(pts)


def crops(rec: Path, frames, src, path, n=12):
    """The detector's picture around chosen ball detections, to check."""
    sys.path.insert(0, str(Path(__file__).parent))
    import phase0

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    seg = json.loads((rec / "detect" / "detections.json").read_text())["segment"]
    picks = [i for i in range(len(src)) if src[i] is not None]
    picks = [picks[k] for k in np.linspace(0, len(picks) - 1, min(n, len(picks))).astype(int)]
    tiles = []
    for i in picks:
        _, d = src[i]
        role = "main" if d["phone"] == 0 else "secondary"
        off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
        img = phase0.frames_at(rec / devs[role]["segments"][seg]["objectKey"], frames[i]["t"] + off, 1, 30,
                               3840, flip[d["phone"]])[0]
        x1, y1, x2, y2 = d["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        c = img[max(0, cy - 60):cy + 60, max(0, cx - 80):cx + 80].copy()
        c = cv2.resize(c, (320, 240), interpolation=cv2.INTER_NEAREST)
        cv2.circle(c, (160, 120), 18, (0, 0, 255), 1)
        cv2.putText(c, f"t={frames[i]['t']:.1f} c={d['conf']:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        tiles.append(c)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def main(rec: Path) -> None:
    out = rec / "ball"
    out.mkdir(exist_ok=True)
    pose = P.load_pose(rec)
    fps, frames, balls, people = load(rec)
    spots = static_spots(balls, fps)
    keep = [[d for d in fr if all(np.hypot(*(np.array(d["xz"]) - s)) > STATIC_RADIUS for s in spots)]
            for fr in balls]
    tracks = link(keep, fps)
    est, src, scored = choose(tracks, people, len(frames), fps)
    n = len(frames)
    direct = sum(s is not None for s in src)
    covered = sum(e is not None for e in est)
    print(f"{sum(map(len, balls))} on-pitch candidates in {n} frames; {len(spots)} static spots removed"
          f" ({sum(map(len, balls)) - sum(map(len, keep))} candidates)")
    print(f"{len(tracks)} tracks; best scores {[round(s, 1) for s, _ in scored[:6]]}")
    print(f"ball estimate in {covered}/{n} frames ({100 * covered / n:.0f}%), {direct} from a detection")
    (out / "ball.json").write_text(json.dumps({"fps": fps, "frames": [
        {"t": f["t"], "xz": None if e is None else [round(float(v), 2) for v in e],
         "detected": s is not None} for f, e, s in zip(frames, est, src)]}))
    minimap(rec, pose, people, est, spots, out / "minimap.png")
    crops(rec, frames, src, out / "crops.jpg")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
