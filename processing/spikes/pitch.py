#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Phase 0 spike: where the pitch is, in metres.

Run after phase0.py and follow.py (cached cameras):

    uv run processing/spikes/pitch.py footage/rec3 <segment>

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
    devs = {d["role"]: d for d in manifest["devices"]}
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


def turf(bg: np.ndarray) -> np.ndarray:
    """The playing surface: the green region touching the bottom of the
    panorama, holes (lines, players) filled."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40)).astype(np.uint8)
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
    unsaturated (the blue, yellow and red markings belong to other sports)."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    tophat = np.zeros_like(v)
    for k in (15, 41, 101):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = np.maximum(tophat, cv2.morphologyEx(v, cv2.MORPH_TOPHAT, se))
    white = (tophat > 25) & (hsv[..., 1] < 60) & (v > 120)
    field = cv2.dilate(turf(bg).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
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
    `k` scales the fixed-size markings (see _elements).
    Parameters: [cx, cz, h, rot, tilt_x, tilt_z, L, W, k].
    """

    def __init__(self, v):
        self.cx, self.cz, self.h, self.rot, self.tx, self.tz, self.L, self.W, self.k = v
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

    def angles(self, xz: np.ndarray) -> np.ndarray:
        d = self.directions(xz)
        return np.degrees(np.c_[np.arctan2(d[:, 0], d[:, 2]), np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))])

    def to_pitch(self, yaw_deg, pitch_deg) -> np.ndarray:
        """Where rays (degrees) meet the ground, in pitch metres; NaN above
        the horizon."""
        y, p = np.radians(yaw_deg), np.radians(pitch_deg)
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
    (None, 12, (0, 1, 2, 3, 4, 5, 6, 7)),
]
X_SCALE = np.array([1, 1, 0.1, 0.02, 0.01, 0.01, 1, 1, 0.05])


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
                      j["markingScale"]])


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


def main(rec: Path, seg: int) -> None:
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
    v0 = [0.0, -31.5, 1.9, np.radians(-4), 0.0, 0.0, 100.0, 59.0, 1.0]
    cv2.imwrite(str(out / "overlay_init.jpg"), cv2.resize(draw(bg, pano, PitchPose(v0)), (pano.w // 3, pano.h // 3)))
    pose = fit_pose(mask, pano, v0)
    cv2.imwrite(str(out / "overlay.jpg"), cv2.resize(draw(bg, pano, pose), (pano.w // 3, pano.h // 3)))
    (out / "pitch.json").write_text(json.dumps({
        "cameraXZ": [pose.cx, pose.cz], "heightM": pose.h, "rotRad": pose.rot,
        "tiltRad": [pose.tx, pose.tz], "lengthM": pose.L, "widthM": pose.W, "markingScale": pose.k}, indent=1))
    place_detections(rec, pose)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
