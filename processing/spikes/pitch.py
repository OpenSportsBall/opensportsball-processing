#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Phase 0 spike: where the pitch is, in metres.

Run after phase0.py and follow.py (cached cameras):

    uv run processing/spikes/pitch.py footage/rec3 <segment> [cx,cz,h,rot_deg,tilt_x,tilt_z,L,W,k]
    uv run processing/spikes/pitch.py footage/rec3 goals
    uv run processing/spikes/pitch.py footage/rec4 taps     (after tapping keypoints in pitchtap/server.py)
    uv run processing/spikes/pitch.py footage/rec3 fixture footage/fixtures/pitch/<name>

The phones sit at (nearly) one point and the pitch is flat, so one 3x3
homography maps pitch coordinates (x along the touchline, z across, in
metres, origin at the centre spot) to viewing directions. It is fitted so
a standard pitch's lines land on the white lines in a clean background
panorama; length and width are free (they vary between pitches).

Writes <recording>/pitch/: background.jpg, lines.png, overlay.jpg and
pitch.json (the pose and dimensions). If detect.py has run, also places its
detections on the pitch: detect/detections_pitch.json (each with pitch
coordinates and whether it is on the pitch) and minimap.png (top-down).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import phase0  # noqa: E402

DEG_PER_PX = 0.03            # background panorama resolution
YAW_RANGE = (-100.0, 100.0)  # degrees
PITCH_RANGE = (-2.0, 38.0)   # degrees below the horizon (down is +)
BG_FRAMES = 9                # median over this many frame pairs


class Pano:
    """Equirectangular panorama: pixel <-> (yaw, pitch) in degrees."""

    def __init__(self):
        self.w = int((YAW_RANGE[1] - YAW_RANGE[0]) / DEG_PER_PX)
        self.h = int((PITCH_RANGE[1] - PITCH_RANGE[0]) / DEG_PER_PX)

    def to_px(self, yaw, pitch):
        return (np.asarray(yaw) - YAW_RANGE[0]) / DEG_PER_PX, (np.asarray(pitch) - PITCH_RANGE[0]) / DEG_PER_PX

    def angles(self):
        yaw = np.radians(YAW_RANGE[0] + np.arange(self.w) * DEG_PER_PX)
        pitch = np.radians(PITCH_RANGE[0] + np.arange(self.h) * DEG_PER_PX)
        return np.meshgrid(yaw, pitch)


def background(rec: Path, seg: int, pano: Pano):
    """Median over frame pairs spread across the segment: players move,
    lines don't."""
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    YY, PP = pano.angles()
    dirs = follow.direction(YY, PP)
    maps = []
    for c in cams:
        x, y, ok = c.project(dirs)
        d = np.minimum(np.minimum(x, c.w - 1 - x), np.minimum(y, c.h - 1 - y))
        maps.append((x, y, np.where(ok, d, -1)))
    # Each pixel from the phone that sees it farther from its frame edge.
    use_main = maps[0][2] >= maps[1][2]
    seen = (maps[0][2] >= 0) | (maps[1][2] >= 0)
    o = report["segments"][seg]["offsetS"]
    dur = min(devs["main"]["segments"][seg]["durationMs"], devs["secondary"]["segments"][seg]["durationMs"]) / 1000
    stack = []
    for t in np.linspace(5, dur - 5, BG_FRAMES):
        a = phase0.frames_at(rec / devs["main"]["segments"][seg]["objectKey"], t, 1, 30, follow.SRC_W, flip[0])[0]
        b = phase0.frames_at(rec / devs["secondary"]["segments"][seg]["objectKey"], t + o, 1, 30,
                             follow.SRC_W, flip[1])[0]
        wa = cv2.remap(a, maps[0][0], maps[0][1], cv2.INTER_LINEAR)
        wb = cv2.remap(b, maps[1][0], maps[1][1], cv2.INTER_LINEAR)
        stack.append(np.where(use_main[..., None], wa, wb))
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    bg[~seen] = 0
    return bg


def turf(bg: np.ndarray, hue_min: int = 30) -> np.ndarray:
    """The playing surface: the green region touching the bottom of the
    panorama, holes (lines, players) filled."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] > hue_min) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green)
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    field = (lab == keep).astype(np.uint8)
    # Fill holes: anything not reachable from the top or sides through
    # non-field. Not the bottom: the near lines run off it.
    inv = (1 - field).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv)
    border = set(np.unique(np.r_[lab[0], lab[:, 0], lab[:, -1]]))
    holes = np.isin(lab, [k for k in range(1, n) if k not in border])
    return (field.astype(bool) | holes)


def line_mask(bg: np.ndarray) -> np.ndarray:
    """White paint on the turf: brighter than its surroundings at some
    scale (lines are a few pixels wide far away, tens nearby) and
    unsaturated (the blue, yellow and red markings belong to other sports).
    "Unsaturated" is S < 100, not lower: a line one or two pixels wide at
    the far end blends with the grass around it, so its pixels stay fairly
    saturated (the 2026-10-21 field's end markings were lost at S < 60). The
    turf's hue bound is lower than turf()'s default because one phone's
    colour cast can put distant grass at hue ~25-30 (that field's Main half)."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    tophat = np.zeros_like(v)
    for k in (15, 41, 101):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = np.maximum(tophat, cv2.morphologyEx(v, cv2.MORPH_TOPHAT, se))
    white = (tophat > 25) & (hsv[..., 1] < 100) & (v > 120)
    field = cv2.dilate(turf(bg, hue_min=20).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return ((white & field) * 255).astype(np.uint8)


CIRCLE_R = 9.15
PEN_DEPTH, PEN_HALF = 16.5, 20.16
GOAL_DEPTH, GOAL_HALF = 5.5, 9.16
PEN_SPOT = 11.0


NOMINAL = (100.0, 64.0)  # sets how many points each line gets


def _elements(L: float, W: float, k: float = 1.0):
    """A pitch's markings: ("seg", a, b) or ("arc", centre, r, t0, t1). `k`
    scales the fixed-size ones (centre circle, areas, penalty spot): small
    pitches often shrink them."""
    hl, hw = L / 2, W / 2
    CIRCLE_R_, PEN_DEPTH_, PEN_HALF_ = k * CIRCLE_R, k * PEN_DEPTH, k * PEN_HALF
    GOAL_DEPTH_, GOAL_HALF_, PEN_SPOT_ = k * GOAL_DEPTH, k * GOAL_HALF, k * PEN_SPOT
    els = [("seg", (-hl, -hw), (hl, -hw)), ("seg", (-hl, hw), (hl, hw)),     # touchlines
           ("seg", (-hl, -hw), (-hl, hw)), ("seg", (hl, -hw), (hl, hw)),     # goal lines
           ("seg", (0, -hw), (0, hw)),                                       # halfway line
           ("arc", (0, 0), CIRCLE_R_, 0.0, 2 * np.pi)]                       # centre circle
    # Penalty arc: the part of the circle round the spot outside the area.
    t = np.arccos((PEN_DEPTH - PEN_SPOT) / CIRCLE_R)
    for sgn in (-1, 1):
        x0 = sgn * hl
        for depth, half in ((PEN_DEPTH_, PEN_HALF_), (GOAL_DEPTH_, GOAL_HALF_)):
            xi = x0 - sgn * depth
            els += [("seg", (xi, -half), (xi, half)), ("seg", (x0, -half), (xi, -half)),
                    ("seg", (x0, half), (xi, half))]
        spot = (x0 - sgn * PEN_SPOT_, 0)
        els.append(("arc", spot, CIRCLE_R_, -t, t) if sgn < 0 else ("arc", spot, CIRCLE_R_, np.pi - t, np.pi + t))
    return els


NEAR_TOUCHLINE, HALFWAY, CENTRE_CIRCLE = 0, 4, 5  # indices into _elements


def pitch_lines(L: float, W: float, k: float = 1.0, step: float = 0.5, only=None) -> np.ndarray:
    """Points (x, z) in metres along a standard pitch's lines: x along the
    touchlines, z across (away from the camera), origin at the centre spot.
    Each marking gets the same number of points whatever L and W are (set
    from NOMINAL), so a fit always sees the same residuals."""
    pts = []
    for i, (e, en) in enumerate(zip(_elements(L, W, k), _elements(*NOMINAL))):
        if only is not None and i not in only:
            continue
        if e[0] == "seg":
            n = max(2, int(np.hypot(en[2][0] - en[1][0], en[2][1] - en[1][1]) / step))
            pts.append(np.linspace(e[1], e[2], n))
        else:
            _, (cx, cz), r, t0, t1 = e
            n = max(8, int(en[2] * (en[4] - en[3]) / step))
            t = np.linspace(t0, t1, n)
            pts.append(np.c_[cx + r * np.cos(t), cz + r * np.sin(t)])
    return np.concatenate(pts)


class PitchPose:
    """Pitch coordinates <-> viewing directions.

    The camera stands at (cx, cz) on the pitch, h metres up. The pitch is
    turned by `rot` about the vertical, and the ground tilted by `tilt_x`,
    `tilt_z` (radians) relative to the panorama's level.
    `k` scales the fixed-size markings (see _elements). `warp` (degrees) adds
    w1 * u + w2 * u**2 to every vertical angle, u = yaw / 60 degrees: a smooth
    correction for what one flat pitch and the camera model can't explain
    (calibration error toward a frame's edge, a crowned or tilted pitch).
    Parameters: [cx, cz, h, rot, tilt_x, tilt_z, L, W, k(, w1, w2)].
    """

    def __init__(self, v):
        self.cx, self.cz, self.h, self.rot, self.tx, self.tz, self.L, self.W, self.k = v[:9]
        self.w1, self.w2 = (v[9], v[10]) if len(v) > 9 else (0.0, 0.0)
        cx_, sx = np.cos(self.tx), np.sin(self.tx)
        cz_, sz = np.cos(self.tz), np.sin(self.tz)
        rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
        rz = np.array([[cz_, -sz, 0], [sz, cz_, 0], [0, 0, 1]])
        self.Rt = rx @ rz

    def directions(self, xz: np.ndarray) -> np.ndarray:
        X, Z = xz[:, 0] - self.cx, xz[:, 1] - self.cz
        c, s = np.cos(self.rot), np.sin(self.rot)
        q = np.stack([c * X - s * Z, np.full_like(X, self.h), s * X + c * Z], -1)
        return q @ self.Rt.T

    def directions3(self, xz: np.ndarray, height: np.ndarray) -> np.ndarray:
        """Like directions, for points `height` metres above the ground."""
        X, Z = xz[:, 0] - self.cx, xz[:, 1] - self.cz
        c, s = np.cos(self.rot), np.sin(self.rot)
        q = np.stack([c * X - s * Z, self.h - np.asarray(height, float), s * X + c * Z], -1)
        return q @ self.Rt.T

    def warp(self, yaw_deg):
        u = np.asarray(yaw_deg) / 60.0
        return self.w1 * u + self.w2 * u * u

    def _to_angles(self, d):
        yaw = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
        pitch = np.degrees(np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2])))
        return np.c_[yaw, pitch + self.warp(yaw)]

    def angles3(self, xz: np.ndarray, height: np.ndarray) -> np.ndarray:
        return self._to_angles(self.directions3(xz, height))

    def angles(self, xz: np.ndarray) -> np.ndarray:
        return self._to_angles(self.directions(xz))

    def to_pitch(self, yaw_deg, pitch_deg) -> np.ndarray:
        """Where rays (degrees) meet the ground, in pitch metres; NaN above
        the horizon."""
        y, p = np.radians(yaw_deg), np.radians(np.asarray(pitch_deg) - self.warp(yaw_deg))
        d = np.stack([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)], -1) @ self.Rt
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(d[..., 1] > 1e-6, self.h / d[..., 1], np.nan)
        X0, Z0 = d[..., 0] * k, d[..., 2] * k
        c, s = np.cos(self.rot), np.sin(self.rot)
        return np.stack([c * X0 + s * Z0 + self.cx, -s * X0 + c * Z0 + self.cz], -1)


# Fit stages: (markings used or None for all, distance truncation px,
# indices of PitchPose parameters left free). Scale is fixed by the
# markings' size (k): one view can't tell a big pitch seen from high up from
# a small one seen from low down.
STAGES = [
    ((NEAR_TOUCHLINE, HALFWAY, CENTRE_CIRCLE), 120, (0, 1, 2, 3, 4, 5, 7)),
    ((NEAR_TOUCHLINE, HALFWAY, CENTRE_CIRCLE), 40, (0, 1, 2, 3, 4, 5, 7)),
    (None, 40, (0, 1, 2, 3, 4, 5, 6, 7)),
    (None, 12, (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)),
    (None, 6, (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)),
]
X_SCALE = np.array([1, 1, 0.1, 0.02, 0.01, 0.01, 1, 1, 0.05, 0.05, 0.05])


def fit_pose(mask: np.ndarray, pano: Pano, v0, stages=STAGES):
    """Least squares on the distance from the model's line points to the
    nearest painted pixel, truncated and blurred, coarse to fine, starting
    from the clearest markings so the fit can't lock onto the wrong line."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares

    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    v = np.array(v0, float)
    for only, trunc, free in stages:
        free = list(free)
        dt = cv2.GaussianBlur(np.minimum(dist, trunc), (0, 0), max(1.0, trunc / 6))

        def resid(sub, v=v, free=free):
            p = v.copy()
            p[free] = sub
            pose = PitchPose(p)
            ang = pose.angles(pitch_lines(pose.L, pose.W, pose.k, only=only))
            x, y = pano.to_px(ang[:, 0], ang[:, 1])
            ok = (x >= 0) & (x < pano.w - 1) & (y >= 0) & (y < pano.h - 1)
            r = np.zeros(len(x))
            r[ok] = map_coordinates(dt, [y[ok], x[ok]], order=1)
            return r / trunc

        sol = least_squares(resid, v[free], loss="soft_l1", f_scale=0.3, x_scale=X_SCALE[free], diff_step=1e-3)
        v[free] = sol.x
        print(f"  {'all markings' if only is None else 'near/halfway/circle'}, trunc {trunc:3d}px: camera at "
              f"({v[0]:.1f}, {v[1]:.1f}) m, {v[2]:.2f} m up, rot {np.degrees(v[3]):.1f} deg, pitch "
              f"{v[6]:.1f} x {v[7]:.1f} m, mean dist {trunc * np.abs(sol.fun).mean():.1f}px")
    return PitchPose(v)


def draw(bg: np.ndarray, pano: Pano, pose: PitchPose) -> np.ndarray:
    img = bg.copy()
    ang = pose.angles(pitch_lines(pose.L, pose.W, pose.k, step=0.1))
    x, y = pano.to_px(ang[:, 0], ang[:, 1])
    ok = (x >= 0) & (x < pano.w) & (y >= 0) & (y < pano.h)
    for xi, yi in zip(x[ok].astype(int), y[ok].astype(int)):
        cv2.circle(img, (xi, yi), 2, (0, 0, 255), -1)
    return img


def load_pose(rec: Path) -> PitchPose:
    j = json.loads((rec / "pitch" / "pitch.json").read_text())
    return PitchPose([*j["cameraXZ"], j["heightM"], j["rotRad"], *j["tiltRad"], j["lengthM"], j["widthM"],
                      j["markingScale"], *j.get("warpDeg", [0.0, 0.0])])


ON_PITCH_MARGIN = 1.0  # metres outside the lines that still count (players on the line, referees)


def place_detections(rec: Path, pose: PitchPose) -> None:
    """Pitch coordinates for detect.py's detections, an on-pitch flag, and a
    top-down map: people on the pitch coloured by time, people off it in
    grey, ball candidates on it as red dots."""
    path = rec / "detect" / "detections.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    hl, hw = pose.L / 2 + ON_PITCH_MARGIN, pose.W / 2 + ON_PITCH_MARGIN
    for f in data["frames"]:
        for d in f["detections"]:
            x, z = pose.to_pitch(d["yaw"], d["pitch"])
            d["xz"] = None if np.isnan(x) else [round(float(x), 2), round(float(z), 2)]
            d["onPitch"] = bool(d["xz"] is not None and abs(x) <= hl and abs(z) <= hw)
    (rec / "detect" / "detections_pitch.json").write_text(json.dumps(data))

    s, pad = 8, 40  # px per metre, border
    W_, H_ = int(pose.L * s) + 2 * pad, int(pose.W * s) + 2 * pad

    def px(x, z):  # far touchline at the top
        return int(pad + (x + pose.L / 2) * s), int(pad + (pose.W / 2 - z) * s)

    img = np.full((H_, W_, 3), (60, 120, 60), np.uint8)
    ln = pitch_lines(pose.L, pose.W, pose.k, step=0.1)
    for x, z in ln:
        cv2.circle(img, px(x, z), 1, (255, 255, 255), -1)
    n = len(data["frames"])
    on_counts = []
    for i, f in enumerate(data["frames"]):
        hue = int(150 * i / max(1, n - 1))
        col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        on = 0
        for d in f["detections"]:
            if d["xz"] is None:
                continue
            x, z = d["xz"]
            if not (-pose.L / 2 - 15 < x < pose.L / 2 + 15 and -pose.W / 2 - 15 < z < pose.W / 2 + 15):
                continue
            if d["cls"] == "person":
                on += d["onPitch"]
                cv2.circle(img, px(x, z), 2, col if d["onPitch"] else (150, 150, 150), -1)
            elif d["onPitch"]:
                cv2.circle(img, px(x, z), 2, (0, 0, 255), -1)
        on_counts.append(on)
    cv2.imwrite(str(rec / "pitch" / "minimap.png"), img)
    print(f"people on the pitch per frame: median {int(np.median(on_counts))}"
          f" (range {min(on_counts)}-{max(on_counts)}); all people {int(np.median([sum(d['cls'] == 'person' for d in f['detections']) for f in data['frames']]))}")


GOAL_W, GOAL_H = 7.32, 2.44   # full-size goal
GOAL_WIDTHS = (5.5, 7.6)      # searched: small-sided to full-size
END_SHIFT_M = (15.0, 6.0)     # an end's markings may sit this far off the whole-pitch fit (x, z)
GOAL_LINE_PX = 8.0            # a goal's base is within this many panorama pixels of its goal line
GOAL_CENTRE_FRAC = 0.15        # ... and its centre within this fraction of the 6-yard box's half-width
                               # of the boxes' centre (both scale-free: at a grazing angle metres along
                               # the ground are only pixels, and one end alone can't pin down scale)
GOAL_MAX_DIST_PX = 3.0        # mean distance from the frame to white structure, at most


def white_structures(bg: np.ndarray) -> np.ndarray:
    """Thin bright unsaturated things anywhere (posts, crossbars, nets,
    lines): not limited to the turf, since goals stand above it."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    tophat = np.maximum(cv2.morphologyEx(v, cv2.MORPH_TOPHAT, np.ones((1, 15), np.uint8)),
                        cv2.morphologyEx(v, cv2.MORPH_TOPHAT, np.ones((15, 1), np.uint8)))
    return (((tophat > 30) & (hsv[..., 1] < 70) & (v > 130)) * 255).astype(np.uint8)


def end_markings(side: int, L: float, k: float, step: float = 0.25) -> np.ndarray:
    """One end's goal line (between the 18-yard box's sides), 6- and 18-yard
    boxes and penalty arc, as pitch points."""
    from scipy.ndimage import map_coordinates  # noqa: F401  (keeps imports local to the goal code)

    x0 = side * L / 2
    pts = []
    half = k * PEN_HALF
    pts.append(np.c_[np.full(80, x0), np.linspace(-half, half, 80)])
    for depth, h_ in ((k * PEN_DEPTH, k * PEN_HALF), (k * GOAL_DEPTH, k * GOAL_HALF)):
        xi = x0 - side * depth
        n = int(2 * h_ / step)
        pts.append(np.c_[np.full(n, xi), np.linspace(-h_, h_, n)])
        m = int(depth / step)
        for zz in (-h_, h_):
            pts.append(np.c_[np.linspace(x0, xi, m), np.full(m, zz)])
    t = np.arccos((PEN_DEPTH - PEN_SPOT) / CIRCLE_R)
    a = np.linspace(-t, t, 40) + (np.pi if side > 0 else 0.0)
    spot = x0 - side * k * PEN_SPOT
    pts.append(np.c_[spot + k * CIRCLE_R * np.cos(a), k * CIRCLE_R * np.sin(a)])
    return np.concatenate(pts)


def goal_frame(x0: float, z0: float, w: float, h: float, step: float = 0.1):
    """Posts and crossbar of a goal on the goal line x0 centred at z0:
    (pitch xz, height) samples."""
    ph = np.arange(0.2, h + 1e-6, step)  # posts, from just above the ground
    cz = np.arange(-w / 2, w / 2 + 1e-6, step)
    xz = np.r_[np.c_[np.full(len(ph), x0), np.full(len(ph), z0 - w / 2)],
               np.c_[np.full(len(ph), x0), np.full(len(ph), z0 + w / 2)],
               np.c_[np.full(len(cz), x0), z0 + cz]]
    height = np.r_[ph, ph, np.full(len(cz), h)]
    return xz, height


END_TOUCHLINE_M = 35.0  # an end's local fit also uses this much of each touchline


def end_points(side: int, L: float, W: float, k: float) -> np.ndarray:
    """An end's markings plus the nearby stretch of both touchlines."""
    x0 = side * L / 2
    xs = np.linspace(x0, x0 - side * END_TOUCHLINE_M, 120)
    return np.concatenate([end_markings(side, L, k), np.c_[xs, np.full(120, -W / 2)], np.c_[xs, np.full(120, W / 2)]])


def fit_end_pose(line_dt: np.ndarray, pano: Pano, pose: PitchPose, side: int) -> tuple[PitchPose, float]:
    """The pitch pose refitted on one end's markings only. The whole-pitch
    fit can be metres off at a far end (a fraction of a degree near the
    horizon is a long way on the ground); locally the paint pins it down."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares

    v = np.array([pose.cx, pose.cz, pose.h, pose.rot, pose.tx, pose.tz, pose.L, pose.W, pose.k, pose.w1, pose.w2],
                 float)
    free = [0, 1, 2, 3, 4, 5, 6]
    for trunc in (60, 25, 10):
        dt = cv2.GaussianBlur(np.minimum(line_dt, trunc), (0, 0), max(1.0, trunc / 6))

        def resid(sub, v=v):
            p = v.copy()
            p[free] = sub
            q = PitchPose(p)
            ang = q.angles(end_points(side, q.L, q.W, q.k))
            x, y = pano.to_px(ang[:, 0], ang[:, 1])
            ok = (x >= 0) & (x < pano.w - 1) & (y >= 0) & (y < pano.h - 1)
            r = np.ones(len(x))
            r[ok] = map_coordinates(dt, [y[ok], x[ok]], order=1) / trunc
            return r

        sol = least_squares(resid, v[free], loss="soft_l1", f_scale=0.3, x_scale=X_SCALE[free], diff_step=1e-3)
        v[free] = sol.x
    return PitchPose(v), float(np.median(resid(v[free])) * 10)


GOAL_SEARCH_M = ((-15.0, 20.0), 15.0)  # candidates: along the pitch (inwards, outwards) and across
GOAL_CANDIDATES = 3                     # best distinct candidates kept per end
END_LINES_MAX_PX = 3.0                  # after the refit, an end's markings sit this close to the paint


def goal_candidates(goal_dt: np.ndarray, pano: Pano, pose: PitchPose, side: int) -> list[tuple]:
    """Goal-shaped white structures near one end: (mean distance px, x, z,
    width, height) in `pose`'s coordinates, best first, at least 4 m apart.
    Includes stored goals and other pitches' goals: the markings decide."""
    from scipy.ndimage import map_coordinates

    def score(x0, z0, w):
        h = GOAL_H * min(1.0, w / GOAL_W)
        ang = pose.angles3(*goal_frame(x0, z0, w, h, step=0.2))
        x, y = pano.to_px(ang[:, 0], ang[:, 1])
        ok = (x >= 0) & (x < pano.w - 1) & (y >= 0) & (y < pano.h - 1)
        if ok.mean() < 0.9:
            return 99.0
        return float(np.minimum(map_coordinates(goal_dt, [y[ok], x[ok]], order=1), 8).mean())

    hl = pose.L / 2
    found = []
    for dx in np.arange(GOAL_SEARCH_M[0][0], GOAL_SEARCH_M[0][1] + 1e-6, 0.5):
        for z0 in np.arange(-GOAL_SEARCH_M[1], GOAL_SEARCH_M[1] + 1e-6, 0.5):
            for w in np.arange(*GOAL_WIDTHS, 0.4):
                found.append((score(side * (hl + dx), z0, w), side * (hl + dx), z0, w))
    found.sort()
    kept = []
    for m, x0, z0, w in found:
        if all(np.hypot(x0 - k[1], z0 - k[2]) >= 4 for k in kept):
            # Polish around it.
            best = (m, x0, z0, w)
            for ox in np.arange(-0.5, 0.51, 0.1):
                for oz in np.arange(-0.5, 0.51, 0.1):
                    for ow in np.arange(-0.3, 0.31, 0.1):
                        m2 = score(x0 + ox, z0 + oz, w + ow)
                        if m2 < best[0]:
                            best = (m2, x0 + ox, z0 + oz, w + ow)
            kept.append((*best, GOAL_H * min(1.0, best[3] / GOAL_W)))
        if len(kept) == GOAL_CANDIDATES:
            break
    return kept


def detect_goals(bg: np.ndarray, mask: np.ndarray, pano: Pano, pose: PitchPose):
    """The two on-pitch goals and a pitch pose corrected with them. Goal-
    shaped white structures near each end are candidates; every left/right
    pair is tried by refitting the pitch so the two goals stand mid goal
    line and the paint still fits. The right pair leaves each end's goal
    line, 6- and 18-yard boxes on the paint with the goals centred on them;
    a stored goal or another pitch's can't. Run once per game.
    Returns (goals, pose)."""
    from scipy.ndimage import map_coordinates

    line_dt = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    goal_dt = cv2.distanceTransform((white_structures(bg) == 0).astype(np.uint8), cv2.DIST_L2, 5)

    def mean_dist(dt, ang, trunc=12):
        x, y = pano.to_px(ang[:, 0], ang[:, 1])
        ok = (x >= 0) & (x < pano.w - 1) & (y >= 0) & (y < pano.h - 1)
        d = np.full(len(x), float(trunc))
        d[ok] = np.minimum(map_coordinates(dt, [y[ok], x[ok]], order=1), trunc)
        return float(d.mean())

    cands = {side: goal_candidates(goal_dt, pano, pose, side) for side in (-1, 1)}
    for side in (-1, 1):
        print(f"{'left' if side < 0 else 'right'} candidates:",
              [(round(c[0], 1), round(c[1], 1), round(c[2], 1), round(c[3], 2)) for c in cands[side]])
    trials = []
    for cl in cands[-1]:
        for cr in cands[1]:
            sizes = {-1: (cl[3], cl[4]), 1: (cr[3], cr[4])}
            q = refit_with_goals(mask, bg, pano, pose, sizes)
            ends = {side: mean_dist(line_dt, q.angles(end_markings(side, q.L, q.k))) for side in (-1, 1)}
            frames = {side: mean_dist(goal_dt, q.angles3(*goal_frame(side * q.L / 2, 0.0, *sizes[side])), 8)
                      for side in (-1, 1)}
            trials.append((sum(ends.values()) + sum(frames.values()), q, sizes, ends, frames))
            print(f"  pair {cl[1]:.0f}/{cl[2]:.0f} + {cr[1]:.0f}/{cr[2]:.0f}: ends' lines {ends[-1]:.1f}/{ends[1]:.1f} px,"
                  f" goals {frames[-1]:.1f}/{frames[1]:.1f} px; pitch {q.L:.1f} x {q.W:.1f}, camera {q.h:.2f} m")
    _, q, sizes, ends, frames = min(trials, key=lambda t: t[0])
    goals = []
    for side in (-1, 1):
        valid = ends[side] <= END_LINES_MAX_PX and frames[side] <= GOAL_MAX_DIST_PX
        goals.append({"side": side, "x": round(side * q.L / 2, 2), "z": 0.0, "width": round(sizes[side][0], 2),
                      "height": round(sizes[side][1], 2), "valid": bool(valid),
                      "endLinesPx": round(ends[side], 2), "frameToWhitePx": round(frames[side], 2)})
    return goals, q


GOAL_WEIGHT = 3.0  # goal-frame residuals count this much more than line points


def refit_with_goals(mask: np.ndarray, bg: np.ndarray, pano: Pano, pose: PitchPose, sizes: dict) -> PitchPose:
    """Refit the whole-pitch pose on the painted lines plus the detected goal
    frames: goals are 3D, of known size, at the middle of each goal line, so
    they fix the camera height and the far ends' distances where flat lines
    seen at a grazing angle can't."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares

    line_dt = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    goal_dt = cv2.distanceTransform((white_structures(bg) == 0).astype(np.uint8), cv2.DIST_L2, 5)
    v = np.array([pose.cx, pose.cz, pose.h, pose.rot, pose.tx, pose.tz, pose.L, pose.W, pose.k, pose.w1, pose.w2],
                 float)
    free = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    sizes = [(side, w, h) for side, (w, h) in sizes.items()]

    def sample(dt, ang):
        x, y = pano.to_px(ang[:, 0], ang[:, 1])
        ok = (x >= 0) & (x < pano.w - 1) & (y >= 0) & (y < pano.h - 1)
        r = np.ones(len(x))
        return ok, x, y, r

    for trunc in (40, 12, 6):
        ldt = cv2.GaussianBlur(np.minimum(line_dt, trunc), (0, 0), max(1.0, trunc / 6))
        gdt = cv2.GaussianBlur(np.minimum(goal_dt, trunc), (0, 0), max(1.0, trunc / 6))

        def resid(sub, v=v):
            p = v.copy()
            p[free] = sub
            q = PitchPose(p)
            parts = []
            for dt, ang, wgt in [(ldt, q.angles(pitch_lines(q.L, q.W, q.k)), 1.0)] + [
                    (gdt, q.angles3(*goal_frame(side * q.L / 2, 0.0, w, h)), GOAL_WEIGHT) for side, w, h in sizes]:
                x, y = pano.to_px(ang[:, 0], ang[:, 1])
                ok = (x >= 0) & (x < pano.w - 1) & (y >= 0) & (y < pano.h - 1)
                r = np.zeros(len(x)) if dt is ldt else np.ones(len(x))  # goal points must stay in view
                r[ok] = map_coordinates(dt, [y[ok], x[ok]], order=1) / trunc
                parts.append(wgt * r)
            return np.concatenate(parts)

        sol = least_squares(resid, v[free], loss="soft_l1", f_scale=0.3, x_scale=X_SCALE[free], diff_step=1e-3)
        v[free] = sol.x
    return PitchPose(v)


def draw_goals(bg, pano, pose, goals, path):
    crops = []
    for g in goals:
        img = bg.copy()
        ang = pose.angles(end_markings(g["side"], pose.L, pose.k))
        x, y = pano.to_px(ang[:, 0], ang[:, 1])
        for xi, yi in zip(x.astype(int), y.astype(int)):
            if 0 <= xi < pano.w and 0 <= yi < pano.h:
                cv2.circle(img, (xi, yi), 1, (255, 255, 0), -1)
        ang = pose.angles3(*goal_frame(g["x"], g["z"], g["width"], g["height"], step=0.05))
        x, y = pano.to_px(ang[:, 0], ang[:, 1])
        for xi, yi in zip(x.astype(int), y.astype(int)):
            cv2.circle(img, (xi, yi), 1, (0, 0, 255) if g["valid"] else (0, 165, 255), -1)
        c = pose.angles3(np.array([[g["x"], g["z"]]]), [1.0])[0]
        cx, cy = pano.to_px(c[0], c[1])
        crop = img[max(0, int(cy) - 110):int(cy) + 150, max(0, int(cx) - 420):int(cx) + 420]
        crop = cv2.resize(crop, (1260, int(1260 * crop.shape[0] / crop.shape[1])))
        cv2.putText(crop, f"{'left' if g['side'] < 0 else 'right'} goal: {'VALID' if g['valid'] else 'rejected'},"
                    f" {g['width']:.2f} x {g['height']:.2f} m; lines {g['endLinesPx']:.1f}px, frame {g['frameToWhitePx']:.1f}px",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        crops.append(crop)
    cv2.imwrite(str(path), np.vstack(crops))


def draw_map(bg: np.ndarray, pano: Pano, pose: PitchPose, goals: list[dict]) -> np.ndarray:
    """The background panorama with the fitted pitch lines (red) and the
    detected goals' frames (yellow posts and crossbar, white ground
    outline of the goal mouth)."""
    img = draw(bg, pano, pose)
    for g in goals:
        if not g["valid"]:
            continue
        ang = pose.angles3(*goal_frame(g["x"], g["z"], g["width"], g["height"], step=0.03))
        x, y = pano.to_px(ang[:, 0], ang[:, 1])
        for xi, yi in zip(x.astype(int), y.astype(int)):
            cv2.circle(img, (xi, yi), 2, (0, 230, 255), -1)
        c = pose.angles3(np.array([[g["x"], g["z"]]]), [g["height"] + 1.0])[0]
        cx, cy = pano.to_px(c[0], c[1])
        cv2.putText(img, f"{g['width']:.1f} x {g['height']:.2f} m", (int(cx) - 70, max(20, int(cy))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
    cv2.putText(img, f"pitch {pose.L:.1f} x {pose.W:.1f} m, camera {pose.h:.2f} m up", (20, pano.h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return img


def goals_main(rec: Path) -> None:
    """Detect the on-pitch goals once per game and correct the pitch map with
    them (needs a normal run first). Writes pitch/goals.json, goals.jpg and
    an updated pitch.json (the lines-only fit is kept as pitch_lines_only.json)."""
    out = rec / "pitch"
    pano = Pano()
    bg = cv2.imread(str(out / "background.png"))
    base = out / "pitch_lines_only.json"
    if not base.exists():
        base.write_text((out / "pitch.json").read_text())
    j = json.loads(base.read_text())
    pose = PitchPose([*j["cameraXZ"], j["heightM"], j["rotRad"], *j["tiltRad"], j["lengthM"], j["widthM"],
                      j["markingScale"], *j.get("warpDeg", [0.0, 0.0])])
    goals, q = detect_goals(bg, line_mask(bg), pano, pose)
    for g in goals:
        print(f"{'left ' if g['side'] < 0 else 'right'} goal: {'valid' if g['valid'] else 'REJECTED'}; "
              f"{g['width']} x {g['height']} m; end markings {g['endLinesPx']} px from paint,"
              f" frame {g['frameToWhitePx']} px from white")
    draw_goals(bg, pano, q, goals, out / "goals.jpg")
    (out / "goals.json").write_text(json.dumps({"goals": goals}, indent=1))
    if all(g["valid"] for g in goals):
        print(f"pitch map corrected: {pose.L:.1f} x {pose.W:.1f} -> {q.L:.1f} x {q.W:.1f} m,"
              f" camera {pose.h:.2f} -> {q.h:.2f} m up")
        (out / "pitch.json").write_text(json.dumps({
            "cameraXZ": [q.cx, q.cz], "heightM": q.h, "rotRad": q.rot, "tiltRad": [q.tx, q.tz],
            "lengthM": q.L, "widthM": q.W, "markingScale": q.k, "warpDeg": [q.w1, q.w2],
            "fittedWith": "lines+goals"}, indent=1))
        cv2.imwrite(str(out / "overlay_goals.jpg"), cv2.resize(draw(bg, pano, q), (pano.w // 3, pano.h // 3)))
    cv2.imwrite(str(out / "map.jpg"), cv2.resize(draw_map(bg, pano, q, goals), (pano.w // 2, pano.h // 2)),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


# Tapped keypoints (pitchtap/server.py): name -> pitch (x, z) for a pose, per end. Post bases are
# only known to stand on the goal line (goals vary in width), so they are fitted to it, not to a point.
TAP_PX = 3.0      # how well a tap is placed, panorama pixels
TAP_WEIGHT = 3.0  # a tap's residual counts this much more than a line point's


def tap_targets(pose: PitchPose) -> dict:
    hl, hw, k = pose.L / 2, pose.W / 2, pose.k
    out = {}
    for s, e in ((-1, "left"), (1, "right")):
        x0 = s * hl
        out[f"{e} corner flag"] = (x0, -hw)
        out[f"{e} 18-yard goal-line corner"] = (x0, -k * PEN_HALF)
        out[f"{e} 18-yard front corner"] = (x0 - s * k * PEN_DEPTH, -k * PEN_HALF)
        out[f"{e} 6-yard goal-line corner"] = (x0, -k * GOAL_HALF)
        out[f"{e} 6-yard front corner"] = (x0 - s * k * GOAL_DEPTH, -k * GOAL_HALF)
    return out


def taps_main(rec: Path) -> None:
    """Refit the pitch with the lines plus tapped keypoints (pitch/taps.json, from
    pitchtap/server.py), starting from the current pitch.json. Writes pitch.json
    ("fittedWith": "lines+taps"; the previous fit is kept as pitch_before_taps.json),
    overlay_taps.jpg, and prints how far each tap ends up from its keypoint."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares

    out = rec / "pitch"
    pano = Pano()
    bg = cv2.imread(str(out / "background.png"))
    taps = {n: t for n, t in json.loads((out / "taps.json").read_text()).items() if not t.get("skipped")}
    j = json.loads((out / "pitch.json").read_text())
    (out / "pitch_before_taps.json").write_text(json.dumps(j, indent=1))
    v = np.array([*j["cameraXZ"], j["heightM"], j["rotRad"], *j["tiltRad"], j["lengthM"], j["widthM"],
                  j["markingScale"], *j.get("warpDeg", [0.0, 0.0])], float)
    points = [n for n in taps if not n.endswith("post base")]
    posts = [n for n in taps if n.endswith("post base")]
    obs = {n: np.array(pano.to_px(np.array([taps[n]["yaw"]]), np.array([taps[n]["pitch"]]))).ravel() for n in taps}
    print(f"{len(points)} tapped keypoints, {len(posts)} post bases")
    mask = line_mask(bg)
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    free = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]

    def tap_resid(pose):
        tg = tap_targets(pose)
        r = []
        if points:
            a = pose.angles(np.array([tg[n] for n in points]))
            x, y = pano.to_px(a[:, 0], a[:, 1])
            r.append((np.c_[x, y] - np.array([obs[n] for n in points])).ravel())
        for n in posts:
            s = -1 if n.startswith("left") else 1
            zz = np.linspace(-pose.W / 2, pose.W / 2, 400)
            a = pose.angles(np.c_[np.full_like(zz, s * pose.L / 2), zz])
            x, y = pano.to_px(a[:, 0], a[:, 1])
            r.append([np.min(np.hypot(x - obs[n][0], y - obs[n][1]))])
        return np.concatenate(r) / TAP_PX if r else np.zeros(0)

    for trunc in (40, 12, 6):
        dt = cv2.GaussianBlur(np.minimum(dist, trunc), (0, 0), max(1.0, trunc / 6))

        def resid(sub, v=v):
            p = v.copy()
            p[free] = sub
            pose = PitchPose(p)
            a = pose.angles(pitch_lines(pose.L, pose.W, pose.k))
            x, y = pano.to_px(a[:, 0], a[:, 1])
            ok = (x >= 0) & (x < pano.w - 1) & (y >= 0) & (y < pano.h - 1)
            r = np.zeros(len(x))
            r[ok] = map_coordinates(dt, [y[ok], x[ok]], order=1)
            return np.concatenate([r / trunc, TAP_WEIGHT * tap_resid(pose)])

        v[free] = least_squares(resid, v[free], loss="soft_l1", f_scale=0.3, x_scale=X_SCALE[free], diff_step=1e-3).x
    pose = PitchPose(v)
    res = TAP_PX * tap_resid(pose)
    for n, e in zip(points + posts, np.r_[np.hypot(res[:2 * len(points):2], res[1:2 * len(points):2]),
                                         np.abs(res[2 * len(points):])]):
        print(f"  {n:30s} {e:5.1f} px from the fit")
    print(f"pitch {pose.L:.1f} x {pose.W:.1f} m, camera {pose.h:.2f} m up, "
          f"{-pose.cz - pose.W / 2:.2f} m behind the near touchline")
    (out / "pitch.json").write_text(json.dumps({**pose_json(pose), "fittedWith": "lines+taps"}, indent=1))
    img = draw(bg, pano, pose)
    for n in taps:
        cv2.drawMarker(img, tuple(int(c) for c in obs[n]), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.imwrite(str(out / "overlay_taps.jpg"), cv2.resize(img, (pano.w // 2, pano.h // 2)), [cv2.IMWRITE_JPEG_QUALITY, 90])


def load_goals(rec: Path):
    path = rec / "pitch" / "goals.json"
    return [g for g in json.loads(path.read_text())["goals"] if g["valid"]] if path.exists() else None


# Starting guess for the night game (hand-read; see main). Kept here so the
# regression fixture reproduces the same fit.
START_POSE = [0.0, -31.5, 1.9, float(np.radians(-4)), 0.0, 0.0, 100.0, 59.0, 1.0, 0.0, 0.0]


def pose_json(pose: PitchPose) -> dict:
    return {"cameraXZ": [pose.cx, pose.cz], "heightM": pose.h, "rotRad": pose.rot, "tiltRad": [pose.tx, pose.tz],
            "lengthM": pose.L, "widthM": pose.W, "markingScale": pose.k, "warpDeg": [pose.w1, pose.w2]}


def pose_from_json(j: dict) -> PitchPose:
    return PitchPose([*j["cameraXZ"], j["heightM"], j["rotRad"], *j["tiltRad"], j["lengthM"], j["widthM"],
                      j["markingScale"], *j.get("warpDeg", [0.0, 0.0])])


def map_pitch(bg: np.ndarray, start: list[float]):
    """The whole mapping from a background panorama: the lines-only fit,
    the goals, and the map corrected with them. Returns (lines pose, goals,
    final pose); final is the lines pose if the goals aren't both valid."""
    pano = Pano()
    mask = line_mask(bg)
    lines = fit_pose(mask, pano, start)
    goals, corrected = detect_goals(bg, mask, pano, lines)
    return lines, goals, corrected if all(g["valid"] for g in goals) else lines


def key_points(pose: PitchPose, goals: list[dict]) -> dict:
    """Viewing angles (degrees) of points that pin the map down: corners,
    centre spot, penalty spots, and each goal's post bases and crossbar ends."""
    hl, hw = pose.L / 2, pose.W / 2
    pts = {"corner_far_left": ([-hl, hw], 0.0), "corner_far_right": ([hl, hw], 0.0),
           "corner_near_left": ([-hl, -hw], 0.0), "corner_near_right": ([hl, -hw], 0.0),
           "centre_spot": ([0.0, 0.0], 0.0),
           "penalty_spot_left": ([-hl + pose.k * PEN_SPOT, 0.0], 0.0),
           "penalty_spot_right": ([hl - pose.k * PEN_SPOT, 0.0], 0.0)}
    for g in goals:
        side = "left" if g["side"] < 0 else "right"
        for zname, dz in (("a", -g["width"] / 2), ("b", g["width"] / 2)):
            pts[f"goal_{side}_post_{zname}_base"] = ([g["x"], g["z"] + dz], 0.0)
            pts[f"goal_{side}_post_{zname}_top"] = ([g["x"], g["z"] + dz], g["height"])
    return {k: [round(float(v), 4) for v in pose.angles3(np.array([xz]), [h])[0]] for k, (xz, h) in pts.items()}


def save_fixture(rec: Path, dest: Path) -> None:
    """Freezes the current, checked mapping as a regression fixture: the
    background panorama it starts from, the starting guess, and the results
    (lines-only fit, goals, final map, key points' angles). Footage shows
    people: keep fixtures out of git (footage/ is ignored)."""
    out = rec / "pitch"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "background.png").write_bytes((out / "background.png").read_bytes())
    final = load_pose(rec)
    goals = json.loads((out / "goals.json").read_text())["goals"]
    expected = {
        "startPose": START_POSE,
        "linesOnly": json.loads((out / "pitch_lines_only.json").read_text()),
        "final": pose_json(final),
        "goals": goals,
        "keyPointsDeg": key_points(final, goals),
        "source": str(rec), "checkedBy": "eye, 2026-10-15: lines and both goals on the paint/posts",
    }
    (dest / "expected.json").write_text(json.dumps(expected, indent=1))
    print(f"fixture saved to {dest}: background.png, expected.json ({len(expected['keyPointsDeg'])} key points)")


def main(rec: Path, seg: int, start: list[float] | None = None) -> None:
    out = rec / "pitch"
    out.mkdir(exist_ok=True)
    pano = Pano()
    bg_path = out / "background.png"
    if bg_path.exists():
        bg = cv2.imread(str(bg_path))
    else:
        bg = background(rec, seg, pano)
        cv2.imwrite(str(bg_path), bg)
    cv2.imwrite(str(out / "background.jpg"), cv2.resize(bg, (pano.w // 3, pano.h // 3)))
    mask = line_mask(bg)
    cv2.imwrite(str(out / "lines.png"), mask)
    cv2.imwrite(str(out / "lines_small.png"), cv2.resize(mask, (pano.w // 3, pano.h // 3), interpolation=cv2.INTER_AREA))
    print(f"panorama {pano.w}x{pano.h}, line pixels {int((mask > 0).sum())}")
    # Start: on the halfway line (it looks vertical at ~4 deg yaw); the
    # painted centre circle spans ~32 deg and dips 2.5-5.3 deg, which puts
    # its centre ~31 m away with the camera ~1.9 m up; the near touchline
    # passes under the panorama, so it is within ~2 m of the camera.
    v0 = start or START_POSE
    cv2.imwrite(str(out / "overlay_init.jpg"), cv2.resize(draw(bg, pano, PitchPose(v0)), (pano.w // 3, pano.h // 3)))
    pose = fit_pose(mask, pano, v0)
    cv2.imwrite(str(out / "overlay.jpg"), cv2.resize(draw(bg, pano, pose), (pano.w // 3, pano.h // 3)))
    (out / "pitch.json").write_text(json.dumps({
        "cameraXZ": [pose.cx, pose.cz], "heightM": pose.h, "rotRad": pose.rot,
        "tiltRad": [pose.tx, pose.tz], "lengthM": pose.L, "widthM": pose.W, "markingScale": pose.k,
        "warpDeg": [pose.w1, pose.w2]}, indent=1))
    place_detections(rec, pose)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "goals":
        goals_main(Path(sys.argv[1]))
    elif len(sys.argv) > 2 and sys.argv[2] == "taps":
        taps_main(Path(sys.argv[1]))
    elif len(sys.argv) > 3 and sys.argv[2] == "fixture":
        save_fixture(Path(sys.argv[1]), Path(sys.argv[3]))
    else:
        # Optional third argument: a starting guess for another field, as
        # comma-separated PitchPose parameters (cx,cz,h,rot_deg,tx,tz,L,W,k).
        start = None
        if len(sys.argv) > 3:
            v = [float(x) for x in sys.argv[3].split(",")]
            start = [*v[:3], float(np.radians(v[3])), *v[4:], *([0.0, 0.0] if len(v) < 11 else [])]
        main(Path(sys.argv[1]), int(sys.argv[2]), start)
