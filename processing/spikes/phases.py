#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Game phases from cheap signals: pre-game, the two halves, half-time,
post-game, so the expensive stages can skip what isn't play.

    uv run processing/spikes/phases.py features footage/rec3 [--segs 0-6] [--fps 1] [--width 640]
    uv run processing/spikes/phases.py segment footage/rec3 [--pad 60]

`features` makes one pass over the whole recording, all on the CPU and the
hardware decoder (no network):

- Whistles, from Main's audio: narrow tones in the referee-whistle band
  (WHISTLE_BAND) that stand well above the rest of the band, as runs of
  STFT frames (phases/whistles.json).
- Motion on the pitch, from both phones at `fps` and `width`: the pitch
  (plus ON_PITCH_MARGIN) is cut into NX x NZ cells, each pixel labelled
  with the cell it shows (pitch.json and the cameras). Per sample and cell:
  the share of pixels that changed since the previous sample ("act") and
  that differ from a rolling background of the last few minutes ("fg").
  The mount is fixed, so both are people on the pitch (phases/features.npz).
- A thumbnail of both phones every THUMB_EVERY s, as contact sheets
  (phases/sheet_NN.jpg) to read the true phase times off by eye.

`segment` reads those and splits the recording into pre-game / first half /
half-time / second half / post-game: dynamic programming over a per-bin
play score with duration limits and a bonus for boundaries on the
referee's whistles, then (for a whole game) a search among nearby whistles
that also wants the halves equal. Pre- or post-game under MIN_CUT_S is
kept. Writes phases/phases.json and phases/plot.png. With phases/truth.json
({"k1", "e1", "k2", "e2"}: kick-offs and ends of the halves, seconds on
Main's clock) it prints the boundary errors, the play lost and the time
saved with `pad` seconds kept either side of each half.

`--segs a-b` limits the pass to Main's segments a..b (default: the first
run of segments both phones have on disk, as older recordings keep only
some of their video locally).

Times are on Main's clock from the start of segment 0, Main's segments
back to back (as follow.game_stream and fullgame.py --start).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import phase0  # noqa: E402
import pitch as P  # noqa: E402

# Whistles
AUDIO_FS = 16_000
NFFT, HOP = 512, 160           # 32 ms window, 10 ms hop
WHISTLE_BAND = (1800.0, 5000.0)
TONAL_DB = 18.0                # band peak over the band's median power
LOUD_DB = 12.0                 # band peak over its own 30 s running median (not a constant hum)
MAX_JUMP_HZ = 350.0            # pea whistles warble; a run may wander this much frame to frame
GAP_FRAMES = 3                 # tolerated dropouts inside a run
MIN_WHISTLE_S = 0.12
REF_TONAL_DB = 35.0            # runs this tonal set the referee's pitch
REF_HZ_TOL = 300.0

# Motion
NX, NZ = 6, 3                  # cells along the pitch and across
GRID_M = 0.1                   # pitch sampling for the cell masks
DIFF = 15                      # luma change counted as movement (after a 3x3 blur)
FG_DIFF = 25                   # luma difference from the background counted as foreground
BG_EVERY_S, BG_N = 6.0, 31     # background: median of a sample every 6 s over the last ~3 min
THUMB_EVERY = 15.0
THUMB_W = 320
SHEET_COLS, SHEET_ROWS = 4, 12  # one sheet = 48 thumbnails = 12 min at 15 s

# Segmentation
BIN_S = 10.0
HALF_S = (20 * 60.0, 55 * 60.0)
BREAK_S = (2 * 60.0, 25 * 60.0)
STATES = ("pre", "h1", "ht", "h2", "post")
THR = 0.5                      # play score splitting play from not
SMOOTH_BINS = 3
HANDLE_X = 5.0
WHISTLE_W = 8.0                # boundary bonus per whistle second, in bins of full play score
WHISTLE_CAP_S = 1.2
EQUAL_W = 0.5                  # penalty per bin of difference between the halves' lengths
MIN_CUT_S = 5 * 60.0           # pre- or post-game shorter than this is kept: little to save, play to lose
REFINE_S = 8 * 60.0            # the refinement tries whistles this far from each DP boundary


def setup(rec: Path):
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    return report, devs, flip


def main_starts(devs: dict) -> list[float]:
    """Start of each of Main's segments on Main's clock (back to back)."""
    d = [s["durationMs"] / 1000 for s in devs["main"]["segments"]]
    return list(np.concatenate([[0.0], np.cumsum(d)]))


# ---------------------------------------------------------------- whistles

def load_audio(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_FS),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def whistle_runs(x: np.ndarray, t0: float) -> list[dict]:
    """Whistle-like tones in one audio file; times on Main's clock from t0."""
    if len(x) < NFFT:
        return []
    n = 1 + (len(x) - NFFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, NFFT), (x.strides[0] * HOP, x.strides[0]))
    spec = np.abs(np.fft.rfft(frames * np.hanning(NFFT).astype(np.float32), axis=1)) ** 2
    freqs = np.fft.rfftfreq(NFFT, 1 / AUDIO_FS)
    band = (freqs >= WHISTLE_BAND[0]) & (freqs <= WHISTLE_BAND[1])
    b = spec[:, band] + 1e-12
    k = b.argmax(1)
    peak = b[np.arange(n), k]
    # the peak with its two neighbours, so a tone between bins isn't split
    lo, hi = np.maximum(k - 1, 0), np.minimum(k + 1, b.shape[1] - 1)
    peak3 = peak + b[np.arange(n), lo] + b[np.arange(n), hi]
    tonal = 10 * np.log10(peak3 / np.median(b, 1))
    level = 10 * np.log10(peak3)
    w = int(30 / (HOP / AUDIO_FS))
    from scipy.ndimage import median_filter
    loud = level - median_filter(level, size=min(w, n) | 1, mode="nearest")
    f = freqs[band][k]
    hit = (tonal > TONAL_DB) & (loud > LOUD_DB)

    runs, cur, gap = [], None, 0
    for i in range(n):
        if hit[i] and (cur is None or abs(f[i] - f[cur[-1]]) < MAX_JUMP_HZ):
            if cur is None:
                cur = [i]
            else:
                cur.append(i)
            gap = 0
        elif cur is not None:
            gap += 1
            if gap > GAP_FRAMES:
                runs.append(cur)
                cur, gap = None, 0
    if cur is not None:
        runs.append(cur)

    dt = HOP / AUDIO_FS
    out = []
    for r in runs:
        dur = (r[-1] - r[0] + 1) * dt
        if dur < MIN_WHISTLE_S:
            continue
        r = np.array(r)
        out.append({"t": round(t0 + r[0] * dt + NFFT / 2 / AUDIO_FS, 3), "dur": round(dur, 3),
                    "hz": round(float(np.median(f[r])), 1), "tonalDb": round(float(np.median(tonal[r])), 1),
                    "loudDb": round(float(np.median(loud[r])), 1)})
    return out


def whistles(rec: Path, devs: dict, segs: tuple[int, int]) -> list[dict]:
    starts = main_starts(devs)
    out = []
    for k in range(segs[0], segs[1] + 1):
        out += whistle_runs(load_audio(rec / devs["main"]["segments"][k]["objectKey"]), starts[k])
    return out


def local_segs(rec: Path, devs: dict, report: dict) -> tuple[int, int]:
    """The first run of Main segments whose video both phones have on disk."""
    def have(k):
        if not (rec / devs["main"]["segments"][k]["objectKey"]).exists():
            return False
        pieces = report["segments"][k].get("secondaryPieces")
        keys = [p["objectKey"] for p in pieces] if pieces else [devs["secondary"]["segments"][k]["objectKey"]]
        return all((rec / q).exists() for q in keys)
    ok = [have(k) for k in range(len(devs["main"]["segments"]))]
    if not any(ok):
        raise SystemExit("no segment has both phones' video on disk")
    a = ok.index(True)
    b = a
    while b + 1 < len(ok) and ok[b + 1]:
        b += 1
    return a, b


# ---------------------------------------------------------------- motion

def cell_labels(cam, pose: P.PitchPose, flip: bool) -> np.ndarray:
    """Per pixel of a raw (as recorded) frame: the pitch cell it shows, -1
    off the pitch. Ground first; pixels left over are filled from points 1 m
    up, so a person standing on a far cell still counts there."""
    hl, hw = pose.L / 2 + P.ON_PITCH_MARGIN, pose.W / 2 + P.ON_PITCH_MARGIN
    xs, zs = np.arange(-hl, hl, GRID_M), np.arange(-hw, hw, GRID_M)
    X, Z = np.meshgrid(xs, zs)
    xz = np.c_[X.ravel(), Z.ravel()]
    cell = (np.clip(((xz[:, 0] + hl) / (2 * hl) * NX).astype(int), 0, NX - 1) * NZ
            + np.clip(((xz[:, 1] + hw) / (2 * hw) * NZ).astype(int), 0, NZ - 1))
    lab = np.full((cam.h, cam.w), -1, np.int16)
    for height in (0.0, 1.0):
        ang = pose.angles3(xz, np.full(len(xz), height)) if height else pose.angles(xz)
        d = follow.direction(np.radians(ang[:, 0]), np.radians(ang[:, 1]))
        px, py, ok = cam.project(d)[:3]
        ok = ok & (px >= 0) & (py >= 0) & (px < cam.w) & (py < cam.h)
        hits = np.full((cam.h, cam.w), -1, np.int16)
        hits[py[ok].astype(int), px[ok].astype(int)] = cell[ok]
        # close the holes between samples, cell by cell
        filled = np.full_like(hits, -1)
        for c in range(NX * NZ):
            m = cv2.morphologyEx((hits == c).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            filled[(m > 0) & (filled < 0)] = c
        lab[(lab < 0) & (filled >= 0)] = filled[(lab < 0) & (filled >= 0)]
    return lab[::-1, ::-1].copy() if flip else lab


def features(rec: Path, fps: float, width: int, segs: tuple[int, int] | None) -> None:
    report, devs, flip = setup(rec)
    segs = segs or local_segs(rec, devs, report)
    starts = main_starts(devs)
    print(f"segments {segs[0]}-{segs[1]} of {len(starts) - 1}: {starts[segs[0]] / 60:.1f}-{starts[segs[1] + 1] / 60:.1f} min",
          flush=True)
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    pose = P.load_pose(rec)
    out = rec / "phases"
    out.mkdir(exist_ok=True)

    t0 = time.time()
    ws = whistles(rec, devs, segs)
    (out / "whistles.json").write_text(json.dumps(ws))
    print(f"whistles: {len(ws)} runs ({sum(w['dur'] >= 0.8 for w in ws)} of 0.8 s or more) in {time.time() - t0:.0f} s",
          flush=True)

    cams = [cm.at(width), cs.at(width)]
    h = round(width * 9 / 16)
    labs = [cell_labels(c, pose, f) for c, f in zip(cams, flip)]
    for c in cams:
        assert (c.w, c.h) == (width, h), (c.w, c.h, width, h)
    idx = [np.flatnonzero(lab.ravel() >= 0) for lab in labs]
    cl = [lab.ravel()[i] for lab, i in zip(labs, idx)]
    ncell = NX * NZ
    pix = np.stack([np.bincount(c, minlength=ncell) for c in cl])  # (phone, cell)
    print(f"cells: pixels per phone {pix.sum(1).tolist()}; cells unseen {(pix.sum(0) == 0).sum()}", flush=True)
    overlay = []
    for lab in labs:
        v = np.where(lab >= 0, (lab * 37 % 255).astype(np.uint8), 0)
        overlay.append(cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_JET) * (lab >= 0)[..., None])
    cv2.imwrite(str(out / "cells.jpg"), np.hstack(overlay))

    t_start = starts[segs[0]]
    total = starts[segs[1] + 1] - t_start - 1.0  # stop short of the end (decoder hang at the last file's end)
    gens = [follow.prefetch(follow.game_stream(rec, devs, report, ph, segs[0], 0.0, total, width, fps)) for ph in (0, 1)]
    n = int(round(total * fps))
    act = np.zeros((n, ncell), np.float32)
    fg = np.zeros((n, ncell), np.float32)
    luma = np.zeros((n, 2), np.float32)
    prev = [None, None]
    ring = [[], []]
    bg = [None, None]
    bg_every = max(1, int(round(BG_EVERY_S * fps)))
    thumb_every = max(1, int(round(THUMB_EVERY * fps)))
    thumbs = []
    t0 = time.time()
    for i, bufs in enumerate(zip(*gens)):
        if i >= n:
            break
        mov = np.zeros(ncell)
        fgc = np.zeros(ncell)
        tiles = []
        for ph, buf in enumerate(bufs):
            Y = buf[: width * h].reshape(h, width)
            if i % thumb_every == 0:
                g = cv2.resize(cv2.cvtColor(buf.reshape(h * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12),
                               (THUMB_W, round(THUMB_W * 9 / 16)), interpolation=cv2.INTER_AREA)
                tiles.append(g[::-1, ::-1] if flip[ph] else g)
            y = cv2.GaussianBlur(Y, (3, 3), 0).ravel()[idx[ph]].astype(np.int16)
            luma[i, ph] = y.mean()
            if prev[ph] is not None:
                mov += np.bincount(cl[ph], weights=np.abs(y - prev[ph]) > DIFF, minlength=ncell)
            prev[ph] = y
            if i % bg_every == 0:
                ring[ph] = (ring[ph] + [y])[-BG_N:]
                bg[ph] = np.median(np.stack(ring[ph]), 0).astype(np.int16)
            fgc += np.bincount(cl[ph], weights=np.abs(y - bg[ph]) > FG_DIFF, minlength=ncell)
        if tiles:
            row = np.hstack(tiles)
            label = f"{(t_start + i / fps) / 60:6.2f} min"
            cv2.putText(row, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(row, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            thumbs.append(row)
        with np.errstate(invalid="ignore", divide="ignore"):
            act[i] = mov / pix.sum(0)
            fg[i] = fgc / pix.sum(0)
        if i % int(600 * fps) == 0 and i:
            el = time.time() - t0
            print(f"  {i / fps / 60:.0f} min of {total / 60:.0f} in {el / 60:.1f} min", flush=True)
    n_done = i + 1 if n else 0
    print(f"motion: {n_done} samples in {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(out / "features.npz", t=t_start + np.arange(n_done) / fps, act=act[:n_done], fg=fg[:n_done],
                        luma=luma[:n_done], pix=pix, segs=np.array(segs), grid=np.array([NX, NZ]), lw=np.array([pose.L, pose.W]))
    per = SHEET_COLS * SHEET_ROWS
    for s in range(0, len(thumbs), per):
        tiles = thumbs[s: s + per]
        tiles += [np.zeros_like(thumbs[0])] * (per - len(tiles))
        rows = [np.hstack(tiles[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
        cv2.imwrite(str(out / f"sheet_{s // per:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"wrote {out}", flush=True)


# ---------------------------------------------------------------- segmentation

def binned(rec: Path):
    z = np.load(rec / "phases" / "features.npz")
    t, act, fg = z["t"], z["act"], z["fg"]
    seen = z["pix"].sum(0) > 0
    t0 = t[0] // BIN_S * BIN_S
    nb = int((t[-1] - t0) // BIN_S) + 1
    b = ((t - t0) // BIN_S).astype(int)
    cnt = np.bincount(b, minlength=nb)
    A = np.stack([np.bincount(b, weights=act[:, c], minlength=nb) for c in range(act.shape[1])], 1) / cnt[:, None]
    F = np.stack([np.bincount(b, weights=fg[:, c], minlength=nb) for c in range(fg.shape[1])], 1) / cnt[:, None]
    return t0 + np.arange(nb) * BIN_S, A[:, seen], F[:, seen], z


def play_score(A: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per bin, roughly 0 (nobody on the pitch) .. 1 (play): movement on
    the pitch, scaled between the game's quiet and busy ends and smoothed
    over SMOOTH_BINS. Bins where the mount is being handled (the whole frame
    changes: foreground HANDLE_X times the game's median, and a bin either
    side) are NaN, as is anything not decoded."""
    m, f = np.nanmean(A, 1), np.nanmean(F, 1)
    handled = np.nan_to_num(f) > HANDLE_X * np.nanmedian(f)
    handled = np.convolve(handled, np.ones(3), "same") > 0
    ok = ~handled & np.isfinite(m)
    lo, hi = np.percentile(m[ok], 10), np.percentile(m[ok], 90)
    s = np.where(ok, np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1.5), 0.0)
    k = np.ones(SMOOTH_BINS)
    with np.errstate(invalid="ignore"):
        s = np.convolve(s, k, "same") / np.convolve(ok.astype(float), k, "same")
    s[~ok] = np.nan
    return s, handled


def boundary_whistles(tb: np.ndarray, ws: list[dict]) -> tuple[list, list]:
    """For each bin boundary (n + 1 of them): the referee whistle a kick-off
    there would follow (the strongest starting within a bin of it) and the
    one a half ending there would end with (the longest ending within a bin)."""
    T = np.append(tb, tb[-1] + BIN_S)
    kick, end = [None] * len(T), [None] * len(T)
    for i, t in enumerate(T):
        near = [w for w in ws if abs(w["t"] - t) <= BIN_S]
        if near:
            kick[i] = max(near, key=lambda w: w["dur"] * w["tonalDb"])
        near = [w for w in ws if abs(w["t"] + w["dur"] - t) <= BIN_S]
        if near:
            end[i] = max(near, key=lambda w: w["dur"])
    return kick, end


def split(score: np.ndarray, kick: list, end: list, thr: float = THR, w_whistle: float = WHISTLE_W) -> list[int | None]:
    """Bin boundaries [k1, e1, k2, e2] (the first half is bins k1..e1-1, ...)
    maximising: play score above `thr` inside the halves and below it
    outside, plus w_whistle per second of referee whistle (capped) at each
    boundary that has one. Five ordered states with duration limits; the
    recording may end in any state (an older recording kept only in part),
    and then the boundaries not reached are None. DP over segment ends."""
    n = len(score)
    x = np.nan_to_num(score - thr)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    bonus = {"kick": np.array([w_whistle * min(w["dur"], WHISTLE_CAP_S) if w else 0.0 for w in kick]),
             "end": np.array([w_whistle * min(w["dur"], WHISTLE_CAP_S) if w else 0.0 for w in end])}
    which = [None, "kick", "end", "kick", "end"]
    lim = [(0, n), (int(HALF_S[0] / BIN_S), int(HALF_S[1] / BIN_S)),
           (int(BREAK_S[0] / BIN_S), int(BREAK_S[1] / BIN_S)),
           (int(HALF_S[0] / BIN_S), int(HALF_S[1] / BIN_S)), (0, n)]
    NEG = -1e18
    best = np.full((5, n + 1), NEG)
    back = np.zeros((5, n + 1), int)
    best[0] = -cs  # pre-game: bins 0..e-1

    def step(k, e, lo, hi):
        sign = 1 if STATES[k] in ("h1", "h2") else -1
        ss = np.arange(max(0, e - hi), e - lo + 1)
        if len(ss) == 0:
            return NEG, 0
        v = best[k - 1, ss] + sign * (cs[e] - cs[ss]) + bonus[which[k]][ss]
        j = int(np.argmax(v))
        return v[j], ss[j]

    for k in range(1, 5):
        for e in range(n + 1):
            best[k, e], back[k, e] = step(k, e, *lim[k])
    # Where the recording ends: in post-game as usual, or cut short in any state.
    finals = [(best[0, n], 0, None), (best[4, n], 4, None)]
    for k in (1, 2, 3):
        v, s0 = step(k, n, 1, lim[k][1])
        finals.append((v, k, s0))
    _, k, s0 = max(finals, key=lambda f: f[0])
    b: list[int | None] = [None] * 4
    e = n
    if s0 is not None:  # the last state, cut short, began at s0
        b[k - 1], e, k = int(s0), int(s0), k - 1
    while k > 0:
        s_ = int(back[k, e])
        b[k - 1] = s_
        e, k = s_, k - 1
    return b


def referee(ws: list[dict]) -> tuple[float, list[dict]]:
    """The referee's whistle pitch and the runs near it. Most tonal runs are
    something else (rec7: ~1,000 runs, mostly 1.7-2.5 kHz; the referee's
    ~70 at ~3.5 kHz, longer, louder and more tonal): the pitch is the
    mode of the runs weighted by all three."""
    wt = [max(0.0, w["tonalDb"] - REF_TONAL_DB) * max(0.0, w["loudDb"] - 20) * w["dur"] for w in ws]
    if not any(wt):
        return float("nan"), []
    edges = np.arange(WHISTLE_BAND[0], WHISTLE_BAND[1] + 100, 100)
    h, _ = np.histogram([w["hz"] for w in ws], edges, weights=wt)
    h = np.convolve(h, np.ones(5), "same")  # +-200 Hz
    hz = float(edges[int(h.argmax())] + 50)
    return hz, [w for w in ws if abs(w["hz"] - hz) <= REF_HZ_TOL and w["tonalDb"] >= REF_TONAL_DB - 10]


def objective(b: list[int], score: np.ndarray, kick: list, end: list, thr: float = THR,
              w_whistle: float = WHISTLE_W) -> float:
    """split()'s objective for whole-game boundaries, plus the halves-are-
    equal prior (leagues play fixed halves; stoppage time differs by a
    minute or two)."""
    x = np.nan_to_num(score - thr)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    k1, e1, k2, e2 = b
    v = 2 * ((cs[e1] - cs[k1]) + (cs[e2] - cs[k2])) - cs[-1]
    for i, ws_ in ((k1, kick), (e1, end), (k2, kick), (e2, end)):
        if ws_[i]:
            v += w_whistle * min(ws_[i]["dur"], WHISTLE_CAP_S)
    return v - EQUAL_W * abs((e1 - k1) - (e2 - k2))


def refine(b: list[int], score: np.ndarray, kick: list, end: list) -> list[int]:
    """Whole-game boundaries re-chosen among the bins with a whistle (and the
    DP's own) within REFINE_S of each, under objective()."""
    r = int(REFINE_S / BIN_S)
    cands = []
    for i, ws_ in zip(b, (kick, end, kick, end)):
        c = {i} | {j for j in range(max(0, i - r), min(len(ws_), i + r + 1)) if ws_[j]}
        cands.append(sorted(c))
    best, best_v = b, objective(b, score, kick, end)
    lh, lb = (int(HALF_S[0] / BIN_S), int(HALF_S[1] / BIN_S)), (int(BREAK_S[0] / BIN_S), int(BREAK_S[1] / BIN_S))
    for k1 in cands[0]:
        for e1 in cands[1]:
            if not lh[0] <= e1 - k1 <= lh[1]:
                continue
            for k2 in cands[2]:
                if not lb[0] <= k2 - e1 <= lb[1]:
                    continue
                for e2 in cands[3]:
                    if not lh[0] <= e2 - k2 <= lh[1]:
                        continue
                    v = objective([k1, e1, k2, e2], score, kick, end)
                    if v > best_v:
                        best, best_v = [k1, e1, k2, e2], v
    return best


def segment(rec: Path, pad: float) -> None:
    out = rec / "phases"
    tb, A, F, z = binned(rec)
    score, handled = play_score(A, F)
    hz, ws = referee(json.loads((out / "whistles.json").read_text()))
    kick, end = boundary_whistles(tb, ws)
    b = split(score, kick, end)
    if all(i is not None for i in b):
        b = refine(b, score, kick, end)
    names = ("k1", "e1", "k2", "e2")
    found, used = {}, {}
    for key, i in zip(names, b):
        if i is None:
            continue
        w = (kick if key[0] == "k" else end)[i]
        used[key] = w
        found[key] = float(tb[0] + i * BIN_S) if w is None else (w["t"] if key[0] == "k" else w["t"] + w["dur"])
    t_first, t_last = float(z["t"][0]), float(z["t"][-1])
    total = t_last - t_first
    # A short pre-game (rec3: 1 min) or a recording begun in play (rec5, rec6)
    # isn't worth the risk of cutting play: keep from the start.
    if "k1" in found and found["k1"] - t_first < MIN_CUT_S:
        found["k1"], used["k1"] = t_first, None
    if "e2" in found and t_last - found["e2"] < MIN_CUT_S:
        found["e2"], used["e2"] = t_last, None
    res = {"found": found, "whistle": used, "fromS": t_first, "toS": t_last, "padS": pad, "refereeHz": hz,
           "handledS": float(handled.sum() * BIN_S)}
    (out / "phases.json").write_text(json.dumps(res, indent=1))
    fmt = lambda s: f"{int(s // 60):3d}:{int(s % 60):02d}"  # noqa: E731
    print(f"{rec.name}: {fmt(t_first)}-{fmt(t_last)}; referee's whistle ~{hz:.0f} Hz, {len(ws)} runs; "
          f"mount handled {handled.sum() * BIN_S:.0f} s")
    truth = json.loads((out / "truth.json").read_text()) if (out / "truth.json").exists() else {}
    for key in names:
        f = found.get(key)
        tr = truth.get(key)
        err = f"  error {f - tr:+5.0f} s" if f is not None and tr is not None else ""
        wh = f"  whistle {used[key]['dur']:.1f} s" if used.get(key) else ("  (no whistle)" if f is not None else "")
        print(f"  {key}  {fmt(f) if f is not None else '    -'}  truth {fmt(tr) if tr is not None else '    -'}{err}{wh}")
    if all(k in found for k in names):
        kept = (found["e1"] - found["k1"]) + (found["e2"] - found["k2"]) + 4 * pad
        print(f"  keep {kept / 60:.1f} of {total / 60:.1f} min ({100 * (1 - kept / total):.0f}% saved, pad {pad:.0f} s)")
        if all(k in truth for k in names):
            keep = [(found["k1"] - pad, found["e1"] + pad), (found["k2"] - pad, found["e2"] + pad)]
            lost = 0.0
            for a_, b_ in ((truth["k1"], truth["e1"]), (truth["k2"], truth["e2"])):
                lost += (b_ - a_) - sum(max(0.0, min(b_, y) - max(a_, x)) for x, y in keep)
            ideal = (truth["e1"] - truth["k1"]) + (truth["e2"] - truth["k2"])
            print(f"  play lost {lost:.0f} s; perfect cut would keep {ideal / 60:.1f} min")
    plot(out / "plot.png", tb, A, F, score, ws, found, truth, t_first, t_last)


def plot(path: Path, tb, A, F, score, ws, found, truth, t_first, t_last) -> None:
    px_per_min = 12
    total = t_last - t_first
    W = int(total / 60 * px_per_min) + 80
    x = lambda s: int(60 + (s - t_first) / 60 * px_per_min)  # noqa: E731
    rows_h = [120, 120, A.shape[1] * 6, A.shape[1] * 6, 60]
    H = sum(rows_h) + 30
    img = np.full((H, W, 3), 255, np.uint8)
    y0 = 10

    def series(v, h, colour, label):
        nonlocal y0
        v = np.nan_to_num(v)
        top = max(np.percentile(v, 99), 1e-9)
        pts = np.array([[x(t + BIN_S / 2), y0 + h - int(min(a / top, 1) * (h - 4))] for t, a in zip(tb, v)])
        cv2.polylines(img, [pts.astype(np.int32)], False, colour, 1, cv2.LINE_AA)
        cv2.putText(img, label, (2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y0 += h

    def heat(M, label):
        nonlocal y0
        M = np.nan_to_num(M)
        top = max(np.percentile(M, 99), 1e-9)
        for c in range(M.shape[1]):
            for t, a in zip(tb, M[:, c]):
                g = 255 - int(min(a / top, 1) * 255)
                cv2.rectangle(img, (x(t), y0 + c * 6), (x(t + BIN_S), y0 + c * 6 + 5), (g, g, g), -1)
        cv2.putText(img, label, (2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        y0 += M.shape[1] * 6

    series(score, rows_h[0], (200, 0, 0), "play")
    series(np.nanmean(F, 1), rows_h[1], (0, 120, 0), "fg")
    heat(A, "act")
    heat(F, "fg")
    for w in ws:
        hh = int(min(w["dur"], 3) / 3 * (rows_h[4] - 4)) + 2
        cv2.line(img, (x(w["t"]), y0 + rows_h[4]), (x(w["t"]), y0 + rows_h[4] - hh), (0, 0, 0), 1)
    cv2.putText(img, "whistles", (2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for m in range(int(np.ceil(t_first / 300)) * 5, int(t_last / 60) + 1, 5):
        cv2.putText(img, str(m), (x(m * 60) - 6, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.line(img, (x(m * 60), H - 20), (x(m * 60), H - 16), (0, 0, 0), 1)
    for k, v in found.items():
        cv2.line(img, (x(v), 0), (x(v), H - 20), (0, 0, 255), 1)
    for k in ("k1", "e1", "k2", "e2"):
        if truth.get(k) is not None:
            cv2.line(img, (x(truth[k]), 0), (x(truth[k]), H - 20), (0, 180, 0), 2)
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "segment"])
    ap.add_argument("rec", type=Path)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--pad", type=float, default=60.0)
    ap.add_argument("--segs", default="", help="a-b: Main's segments a..b (default: the first run on disk)")
    a = ap.parse_args()
    if a.cmd == "features":
        segs = tuple(int(v) for v in a.segs.split("-")) if a.segs else None
        features(a.rec, a.fps, a.width, segs)
    else:
        segment(a.rec, a.pad)


if __name__ == "__main__":
    main()
