#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10"]
# ///
"""Phase 0 spike: a players-only follow cam rendered at 1080p.

Run after phase0.py (it reads phase0/report.json for the audio offsets):

    uv run processing/spikes/follow.py footage/rec3 <segment> <start s> <seconds>

1. Geometry: the panorama stitcher's cameras (focal length, principal
   point, rotation), scaled to the source resolution, map any direction
   to a pixel in each phone's frame. Estimated once; assumes a rigid mount.
2. Pass A: a low-resolution panorama of the field, background subtraction
   (the mount is fixed, so moving blobs on the turf are players), blob
   positions as yaw/pitch angles.
3. Path: per frame, aim at the players' centre and zoom to their spread;
   then smooth over seconds (offline, so it can use future frames) and cap
   the pan speed.
4. Pass B: a 1920x1080 virtual camera along the path, sampled straight from
   both phones' full-resolution frames. Each pixel comes from one phone:
   the two never capture the same instant (free-running 30 fps clocks,
   opposite rolling shutters), so blending them doubles moving players.
   The switch follows a per-frame seam through the overlap that avoids
   players and places the phones disagree, and sticks between frames.

The stitcher's cameras are refined at 1920 px on matches from frames
across the game (focal lengths, two radial terms, Secondary's rotation),
which roughly halves the alignment error at 4K.

Writes <recording>/follow/: follow.mp4, path.json, background.jpg and a
detections overlay. Stays inside one segment for now.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import phase0  # noqa: E402

FPS = 30
CAL_W = 960            # width the cameras are estimated at (phase0.PREVIEW_W)
REF_W = 1920           # width the camera model is refined and held at
SRC_W, SRC_H = 3840, 2160
OUT_W, OUT_H = 1920, 1080
PANO_DEG_PER_PX = 0.08  # pass A panorama resolution
WARMUP_S = 20           # background model warm-up before the excerpt
MIN_HFOV, MAX_HFOV = 30.0, 70.0
MARGIN_DEG = 4.0        # framing margin on each side of the players
SMOOTH_S = 1.5          # Gaussian sigma for the path
MAX_PAN_DEG_S = 20.0    # pan speed cap
SEAM_FEATHER_DEG = 0.3  # detail blend width either side of the seam
TONE_FEATHER_DEG = 6.0  # brightness/colour blend width across the seam
SEAM_LAMBDA = 0.003     # seam stickiness between frames, per column per row
PLAYER_COST = 3.0       # seam cost on a detected player, per pixel


# --------------------------------------------------------------- geometry

class Camera:
    """One phone: world direction (y down) <-> pixel, at a given width.

    Pinhole plus two radial distortion terms. Parameters are held at
    REF_W and scaled to the width asked for.
    """

    def __init__(self, f, cx, cy, k1, k2, R, width=REF_W):
        s = width / REF_W
        self.f, self.cx, self.cy = f * s, cx * s, cy * s
        self.k1, self.k2, self.R = k1, k2, R
        self.w, self.h = width, round(width * 9 / 16)
        self.ref = (f, cx, cy, k1, k2, R)
        # The radial model folds back beyond the image: directions far
        # outside would land inside it. Only trust radii up to the corners.
        corners = np.array([[0, 0], [self.w - 1, 0], [0, self.h - 1], [self.w - 1, self.h - 1]], float)
        c = self.unproject(corners) @ self.R
        self.r2_max = 1.05 * float(((c[:, 0] / c[:, 2]) ** 2 + (c[:, 1] / c[:, 2]) ** 2).max())

    @classmethod
    def from_stitcher(cls, cam):
        """From an OpenCV CameraParams estimated at CAL_W."""
        K = cam.K().astype(np.float64) * (REF_W / CAL_W)
        return cls(K[0, 0], K[0, 2], K[1, 2], 0.0, 0.0, cam.R.astype(np.float64))

    def at(self, width):
        return Camera(*self.ref, width=width)

    def project(self, d: np.ndarray):
        """World directions (..., 3) -> pixel maps and validity."""
        c = d @ self.R  # R^T d, row-vector form
        z = c[..., 2]
        ok = z > 1e-6
        zz = np.where(ok, z, 1)
        x, y = c[..., 0] / zz, c[..., 1] / zz
        r2 = x * x + y * y
        ok &= r2 < self.r2_max
        g = 1 + self.k1 * r2 + self.k2 * r2 * r2
        px = np.where(ok, self.f * x * g + self.cx, -1)
        py = np.where(ok, self.f * y * g + self.cy, -1)
        ok &= (px >= 0) & (px <= self.w - 1) & (py >= 0) & (py <= self.h - 1)
        return px.astype(np.float32), py.astype(np.float32), ok

    def unproject(self, px: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> world directions (N, 3)."""
        xd = (px[:, 0] - self.cx) / self.f
        yd = (px[:, 1] - self.cy) / self.f
        x, y = xd, yd
        for _ in range(8):  # invert the radial model by fixed-point iteration
            r2 = x * x + y * y
            g = 1 + self.k1 * r2 + self.k2 * r2 * r2
            x, y = xd / g, yd / g
        return np.stack([x, y, np.ones_like(x)], -1) @ self.R.T

    def edge_weight(self, x, y, ok):
        """Feather weight for the detection panorama: distance to the
        nearest image edge."""
        d = np.minimum(np.minimum(x, self.w - 1 - x), np.minimum(y, self.h - 1 - y))
        return np.where(ok, np.clip(d / (0.05 * self.w), 0, 1) ** 2, 0).astype(np.float32)


def match_pairs(pairs):
    """SIFT matches (secondary px, main px) pooled over frame pairs."""
    sift = cv2.SIFT_create(nfeatures=12000)
    out = []
    for m, s in pairs:
        km, dm = sift.detectAndCompute(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), None)
        ks, ds = sift.detectAndCompute(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), None)
        good = [a for a, b in (x for x in cv2.BFMatcher().knnMatch(ds, dm, k=2) if len(x) == 2)
                if a.distance < 0.7 * b.distance]
        out.append((np.float64([ks[g.queryIdx].pt for g in good]).reshape(-1, 2),
                    np.float64([km[g.trainIdx].pt for g in good]).reshape(-1, 2)))
    return out


def transfer_error(cm: Camera, cs: Camera, ps, pm):
    """Pixel distance in Main between each match and Secondary's point
    carried through the model."""
    x, y, ok = cm.project(cs.unproject(ps))
    e = np.hypot(x - pm[:, 0], y - pm[:, 1])
    return np.where(ok, e, np.inf)


def refine(cm: Camera, cs: Camera, matches, gate=40.0):
    """Least-squares refinement of both focal lengths, radial distortion
    and Secondary's rotation (Main's stays as the world reference), on
    matches the starting model already roughly agrees with."""
    ps = np.concatenate([m[0] for m in matches])
    pm = np.concatenate([m[1] for m in matches])
    keep = transfer_error(cm, cs, ps, pm) < gate
    ps, pm = ps[keep], pm[keep]
    fm, cxm, cym = cm.ref[:3]
    fs, cxs, cys = cs.ref[:3]
    R0 = cs.R

    def cams(v):
        rs = cv2.Rodrigues(v[6:9])[0] @ R0
        return (Camera(v[0], cxm, cym, v[2], v[4], cm.R),
                Camera(v[1], cxs, cys, v[3], v[5], rs))

    def resid(v):
        a, b = cams(v)
        xm, ym, _ = a.project(b.unproject(ps))
        xs, ys, _ = b.project(a.unproject(pm))
        prior = 20 * np.array([v[2], v[3], v[4], v[5]])  # keep distortion modest
        return np.concatenate([xm - pm[:, 0], ym - pm[:, 1], xs - ps[:, 0], ys - ps[:, 1], prior])

    v0 = np.array([fm, fs, 0, 0, 0, 0, 0, 0, 0], float)
    sol = least_squares(resid, v0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    return cams(sol.x), int(keep.sum())


def direction(yaw, pitch):
    """World direction for yaw (right +) and pitch (down +), radians."""
    return np.stack([np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                     np.cos(yaw) * np.cos(pitch)], -1)


def view_directions(yaw, pitch, hfov, w, h):
    """Directions through each pixel of a pinhole virtual camera."""
    f = (w / 2) / np.tan(hfov / 2)
    u, v = np.meshgrid((np.arange(w) - w / 2 + 0.5) / f, (np.arange(h) - h / 2 + 0.5) / f)
    z = direction(yaw, pitch)
    x = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    y = np.cross(z, x)
    return u[..., None] * x + v[..., None] * y + z


def blend(frames, maps, weights, gains):
    acc = np.zeros(maps[0][0].shape + (3,), np.float32)
    wsum = np.zeros(maps[0][0].shape, np.float32)
    for img, (mx, my), w, g in zip(frames, maps, weights, gains):
        acc += cv2.remap(img, mx, my, cv2.INTER_LINEAR).astype(np.float32) * (w * g)[..., None]
        wsum += w
    return np.clip(acc / np.maximum(wsum, 1e-6)[..., None], 0, 255).astype(np.uint8)


def seam_dp(cost: np.ndarray, prev: np.ndarray | None, lam: float) -> np.ndarray:
    """Lowest-cost top-to-bottom path, one column per row, moving at most
    two columns per row; `lam` per column per row pulls it towards the
    previous frame's path so the seam doesn't jump."""
    H, W = cost.shape
    if prev is not None:
        cost = cost + lam * np.abs(np.arange(W)[None, :] - prev[:, None])
    acc = cost[0].copy()
    back = np.zeros((H, W), np.int8)
    big = np.full(2, np.inf)
    for r in range(1, H):
        padded = np.concatenate([big, acc, big])
        cand = np.stack([padded[2 + k:2 + k + W] for k in (-2, -1, 0, 1, 2)])
        k = np.argmin(cand, 0)
        back[r] = k - 2
        acc = cost[r] + cand[k, np.arange(W)]
    path = np.empty(H, np.int32)
    path[-1] = int(np.argmin(acc))
    for r in range(H - 1, 0, -1):
        path[r - 1] = path[r] + back[r, path[r]]
    return path


def blend_two_band(frames, maps, sharp, wide, gains):
    """Fine detail switches at the seam (`sharp` weights); overall
    brightness and colour blend over a wide band (`wide` weights), so an
    exposure difference between the phones fades instead of stepping."""
    h, w = maps[0][0].shape
    small = (w // 8, h // 8)
    out = np.zeros((h, w, 3), np.float32)
    ssum = np.maximum(sharp[0] + sharp[1], 1e-6)[..., None]
    wsum = np.maximum(wide[0] + wide[1], 1e-6)[..., None]
    for img, (mx, my), ws_, ww, g in zip(frames, maps, sharp, wide, gains):
        r = cv2.remap(img, mx, my, cv2.INTER_LINEAR).astype(np.float32) * g
        low = cv2.resize(cv2.GaussianBlur(cv2.resize(r, small, interpolation=cv2.INTER_AREA), (0, 0), 2), (w, h))
        out += (r - low) * (ws_[..., None] / ssum) + low * (ww[..., None] / wsum)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- media io

def frame_stream(path: Path, start: float, seconds: float, width: int, rotate180: bool):
    """Yields BGR frames at constant FPS, decoded in hardware where possible."""
    height = round(width * 9 / 16)
    vf = ("hflip,vflip," if rotate180 else "") + f"fps={FPS},scale={width}:{height}"
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-noautorotate", "-hwaccel", "videotoolbox", "-ss", f"{start:.4f}",
         "-i", str(path), "-t", f"{seconds:.3f}", "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    proc.wait()


# ------------------------------------------------------------------- main

def main(rec: Path, seg: int, start: float, seconds: float) -> None:
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    out = rec / "follow"
    out.mkdir(exist_ok=True)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    files = [rec / devs["main"]["segments"][seg]["objectKey"],
             rec / devs["secondary"]["segments"][seg]["objectKey"]]
    offsets = [0.0, report["segments"][seg]["offsetS"]]

    # Geometry from segment 0's calibration (phase0's choice of frame pair).
    s0 = report["segments"][0]
    m0 = devs["main"]["segments"][0]
    sec0 = devs["secondary"]["segments"][0]
    dur = min(m0["durationMs"], sec0["durationMs"]) / 1000 - abs(s0["offsetS"]) - 1
    t0 = max(0.0, -s0["offsetS"]) + 0.5
    pairs = [(phase0.frames_at(rec / m0["objectKey"], t, 1, FPS, CAL_W, flip[0])[0],
              phase0.frames_at(rec / sec0["objectKey"], t + s0["offsetS"], 1, FPS, CAL_W, flip[1])[0])
             for t in np.linspace(t0, t0 + dur - 0.5, phase0.CALIB_FRAMES)]
    st, cal = phase0.calibrate(pairs)
    if st is None:
        sys.exit("calibration failed")
    cams = st.cameras()  # order [secondary, main]
    print(f"stitcher: {cal['angleBetweenDeg']} deg apart, focal {cal['focalPx']} @ {CAL_W}")

    # Refine at REF_W on matches from frame pairs spread over the game.
    cm, cs = Camera.from_stitcher(cams[1]), Camera.from_stitcher(cams[0])
    nseg = len(report["segments"])
    fit_pairs = []
    for sg in sorted({round(x) for x in np.linspace(0, nseg - 1, 6)}):
        o = report["segments"][sg]["offsetS"]
        for t in (40.0, 140.0):
            fit_pairs.append((
                phase0.frames_at(rec / devs["main"]["segments"][sg]["objectKey"], t, 1, FPS, REF_W, flip[0])[0],
                phase0.frames_at(rec / devs["secondary"]["segments"][sg]["objectKey"], t + o, 1, FPS, REF_W, flip[1])[0]))
    matches = match_pairs(fit_pairs)
    ps = np.concatenate([x[0] for x in matches]); pm = np.concatenate([x[1] for x in matches])
    before = transfer_error(cm, cs, ps, pm)
    (cm, cs), n = refine(cm, cs, matches)
    after = transfer_error(cm, cs, ps, pm)
    near = before < 40
    print(f"refined on {n} matches: median error {2 * np.median(before[near]):.1f} ->"
          f" {2 * np.median(after[near]):.1f} px at 4K; k1 {cm.k1:.4f}/{cs.k1:.4f}")

    # Pass A panorama: equirectangular over everything either phone sees.
    lo = [cm.at(CAL_W), cs.at(CAL_W)]
    edge = []
    for c in lo:
        xs = np.linspace(0, c.w - 1, 40)
        ys = np.linspace(0, c.h - 1, 24)
        px = np.r_[np.c_[xs, 0 * xs], np.c_[xs, 0 * xs + c.h - 1], np.c_[0 * ys, ys], np.c_[0 * ys + c.w - 1, ys]]
        edge.append(c.unproject(px))
    rays = np.concatenate(edge)
    yaws = np.arctan2(rays[:, 0], rays[:, 2])
    pitches = np.arctan2(rays[:, 1], np.hypot(rays[:, 0], rays[:, 2]))
    step = np.radians(PANO_DEG_PER_PX)
    y0, y1, p0, p1 = yaws.min(), yaws.max(), pitches.min(), pitches.max()
    PW, PH = int((y1 - y0) / step), int((p1 - p0) / step)
    YY, PP = np.meshgrid(y0 + np.arange(PW) * step, p0 + np.arange(PH) * step)
    pano_dirs = direction(YY, PP)
    pmaps, pweights = [], []
    for c in lo:
        x, y, ok = c.project(pano_dirs)
        pmaps.append((x, y))
        pweights.append(c.edge_weight(x, y, ok))
    print(f"pass A panorama {PW}x{PH}, yaw {np.degrees(y0):.0f}..{np.degrees(y1):.0f} deg")
    # Columns where both phones see something; the seam runs through them.
    both = (pweights[0] > 0) & (pweights[1] > 0)
    ocols = np.flatnonzero(both.any(0))
    c0, c1 = int(ocols.min()), int(ocols.max()) + 1
    main_right = YY[0][pweights[0].any(0)].mean() > YY[0][pweights[1].any(0)].mean()
    print(f"overlap yaw {np.degrees(y0 + c0 * step):.1f}..{np.degrees(y0 + c1 * step):.1f} deg,"
          f" Main on the {'right' if main_right else 'left'}")

    # Exposure: match Secondary to Main in the overlap of the first pair.
    a, b = pairs[len(pairs) // 2]
    wa, wb = pweights
    ov = (wa > 0) & (wb > 0)
    ia = cv2.remap(a, *pmaps[0], cv2.INTER_LINEAR)[ov].astype(np.float32)
    ib = cv2.remap(b, *pmaps[1], cv2.INTER_LINEAR)[ov].astype(np.float32)
    gain_s = (ia.mean(0) + 1) / (ib.mean(0) + 1)
    gains = [np.float32(1.0), np.float32(gain_s.mean())]
    print(f"secondary gain {gains[1]:.3f}")

    # Pass A: detect moving blobs on the turf.
    warm = min(WARMUP_S, start)
    a_start = start - warm
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=True)
    streams = [frame_stream(f, a_start + o, warm + seconds, CAL_W, fl)
               for f, o, fl in zip(files, offsets, flip)]
    dets: list[list[tuple[float, float, float]]] = []
    field = None
    n_warm = int(warm * FPS)
    kernel = np.ones((3, 3), np.uint8)
    sample = None
    seams, seam = [], None
    for i, (fa, fb) in enumerate(zip(*streams)):
        pano = blend([fa, fb], pmaps, pweights, gains)
        fg = mog.apply(pano, learningRate=-1)
        # Seam: avoid players and places where the two phones disagree.
        wa_ = cv2.remap(fa, *pmaps[0], cv2.INTER_LINEAR)[:, c0:c1].astype(np.float32)
        wb_ = cv2.remap(fb, *pmaps[1], cv2.INTER_LINEAR)[:, c0:c1].astype(np.float32) * gains[1]
        cost = cv2.blur(np.abs(wa_ - wb_).mean(2) / 255, (3, 3))
        players = cv2.dilate((fg[:, c0:c1] == 255).astype(np.uint8), np.ones((7, 7), np.uint8))
        cost += PLAYER_COST * players
        cost[~both[:, c0:c1]] = 50.0
        seam = seam_dp(cost, seam, SEAM_LAMBDA)
        if i == n_warm:
            bg = mog.getBackgroundImage()
            cv2.imwrite(str(out / "background.jpg"), bg)
            hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
            turf = ((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40)).astype(np.uint8) * 255
            turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(turf)
            field = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        if i < n_warm:
            continue
        fg = cv2.morphologyEx((fg == 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
        frame_dets = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            feet_x, feet_y = x + w // 2, min(y + h, PH - 1)
            if area < 12 or h < 5 or w > 3 * h or not field[feet_y, feet_x]:
                continue
            frame_dets.append((float(YY[0, int(cent[k][0])]), float(PP[int(cent[k][1]), 0]), float(area)))
        dets.append(frame_dets)
        seams.append(y0 + (c0 + seam) * step)
        if len(dets) == int(seconds * FPS / 2):
            sample = pano.copy()
            for yw, pt, _ in frame_dets:
                cv2.circle(sample, (int((yw - y0) / step), int((pt - p0) / step)), 6, (0, 0, 255), 2)
            cv2.polylines(sample, [np.c_[c0 + seam, np.arange(PH)].astype(np.int32)], False, (255, 255, 0), 1)
    if sample is not None:
        cv2.imwrite(str(out / "detections.jpg"), sample)
    sy = np.degrees(np.array(seams))
    print(f"seam yaw at the horizon row: {np.percentile(sy[:, PH // 2], [5, 50, 95]).round(1)} deg")
    counts = [len(d) for d in dets]
    print(f"pass A: {len(dets)} frames, players detected per frame median {int(np.median(counts))}"
          f" (range {min(counts)}-{max(counts)})")

    # Path: centre of play and spread, then smooth.
    T = len(dets)
    tyaw, tpitch, thfov = np.zeros(T), np.zeros(T), np.zeros(T)
    last = (0.5 * (y0 + y1), 0.5 * (p0 + p1), np.radians(MAX_HFOV))
    for i, fd in enumerate(dets):
        if len(fd) >= 2:
            yw = np.array([d[0] for d in fd])
            pt = np.array([d[1] for d in fd])
            lo_, hi_ = np.percentile(yw, [15, 85])
            spread = hi_ - lo_ + 2 * np.radians(MARGIN_DEG)
            last = (np.median(yw), np.median(pt),
                    np.clip(spread, np.radians(MIN_HFOV), np.radians(MAX_HFOV)))
        tyaw[i], tpitch[i], thfov[i] = last
    win = int(FPS) | 1
    sig = SMOOTH_S * FPS
    yaw = gaussian_filter1d(median_filter(tyaw, win, mode="nearest"), sig, mode="nearest")
    pitch = gaussian_filter1d(median_filter(tpitch, win, mode="nearest"), 2 * sig, mode="nearest")
    hfov = gaussian_filter1d(median_filter(thfov, 3 * win, mode="nearest"), 2 * sig, mode="nearest")
    vmax = np.radians(MAX_PAN_DEG_S) / FPS
    for i in range(1, T):
        yaw[i] = yaw[i - 1] + np.clip(yaw[i] - yaw[i - 1], -vmax, vmax)
    # Keep the view inside what the phones saw vertically.
    vfov = 2 * np.arctan(np.tan(hfov / 2) * OUT_H / OUT_W)
    pitch = np.clip(pitch, p0 + vfov / 2, p1 - vfov / 2)
    (out / "path.json").write_text(json.dumps({
        "segment": seg, "startS": start, "fps": FPS,
        "yawDeg": np.round(np.degrees(yaw), 2).tolist(),
        "pitchDeg": np.round(np.degrees(pitch), 2).tolist(),
        "hfovDeg": np.round(np.degrees(hfov), 2).tolist()}))
    print(f"path: yaw {np.degrees(yaw.min()):.0f}..{np.degrees(yaw.max()):.0f} deg,"
          f" hfov {np.degrees(hfov.min()):.0f}..{np.degrees(hfov.max()):.0f} deg")

    # Pass B: render the virtual camera from full-resolution frames. Each
    # pixel comes from one phone, switching at that frame's seam.
    hi = [cm.at(SRC_W), cs.at(SRC_W)]
    gw, gh = OUT_W // 4, OUT_H // 4  # projection grid, upsampled to full size
    rows = p0 + np.arange(PH) * step
    feather = np.radians(SEAM_FEATHER_DEG)
    tone = np.radians(TONE_FEATHER_DEG)
    streams = [frame_stream(f, start + o, seconds, SRC_W, fl)
               for f, o, fl in zip(files, offsets, flip)]
    enc = phase0.Encoder(out / "follow.mp4", (OUT_W, OUT_H), FPS)
    for i, (fa, fb) in enumerate(zip(*streams)):
        if i >= T:
            break
        d = view_directions(yaw[i], pitch[i], hfov[i], gw, gh)
        vy = np.arctan2(d[..., 0], d[..., 2])
        vp = np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
        off = vy - np.interp(vp, rows, seams[i])
        if not main_right:
            off = -off
        maps, oks = [], []
        for c in hi:
            x, y, ok = c.project(d)
            maps.append((cv2.resize(x, (OUT_W, OUT_H)), cv2.resize(y, (OUT_W, OUT_H))))
            oks.append(ok)
        both_ok = oks[0] & oks[1]

        def weights(width):
            wm = np.where(both_ok, np.clip(off / width + 0.5, 0, 1), oks[0]).astype(np.float32)
            ws = np.where(oks[1], 1 - wm, 0).astype(np.float32)
            return [cv2.resize(wm, (OUT_W, OUT_H)), cv2.resize(ws, (OUT_W, OUT_H))]

        enc.write(blend_two_band([fa, fb], maps, weights(feather), weights(tone), gains))
        if i % 300 == 0:
            print(f"pass B: frame {i}/{T}", flush=True)
    enc.close()
    print(f"wrote {out / 'follow.mp4'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
