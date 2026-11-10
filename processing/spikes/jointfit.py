#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""Both phones' lenses and the pitch, fitted together in the phones' own pixels.

    uv run processing/spikes/jointfit.py <rec> taps
    uv run processing/spikes/jointfit.py <rec> fit [--seg N] [--no-taps] [--lenses lenses.json [--exclude-self]]
                                                   [--dist-sd 0.01] [--offset-sd 0.03] [--start DIR] [--out DIR]
    uv run processing/spikes/jointfit.py <rec> holdout [--folds 5] [--start DIR] [--out DIR] [fit's options]
    uv run processing/spikes/jointfit.py <rec> score <dir> [<dir> ...] [--seg N]
    uv run processing/spikes/jointfit.py lenses <rec> [<rec> ...]
    uv run processing/spikes/jointfit.py bench footage/fixtures/pitch/<name> <rec> <dir> [<dir> ...]

Putting a fit into a recording that already has boxes, goals and detections:
    uv run processing/spikes/jointfit.py mirror <rec> <fit dir> <mirror> <segment>
    uv run processing/spikes/jointfit.py convert-detect <rec> <mirror> "*"
    uv run processing/spikes/jointfit.py convert-passa <rec> <mirror>     # or let playbench.py redo pass A
    uv run processing/spikes/jointfit.py adopt <rec> <mirror> [<keep>]    # replaced files go to <rec>/<keep>/
<keep> defaults to pre_joint; a recording adopted before needs another name
(rec7: pre_joint is the stitch, joint_2026-11-09_files the first joint fit).
mirror draws the panorama from the segment's video; when that isn't on disk,
from the two per-phone backgrounds the fit cached (<fit dir>/bg_*_seg<N>.png).

When the phones can't be stitched (follow.setup_cameras: "calibration
failed", rec5), osb-engine (engine.py) builds the cameras from the pitch:
    uv run processing/spikes/jointfit.py <rec> choices <out dir> [--seg N]
    uv run processing/spikes/jointfit.py <rec> rough --left main|secondary [--angle 80]
    uv run processing/spikes/jointfit.py <rec> search [--seg N] [--out DIR]
    uv run processing/spikes/jointfit.py <rec> install DIR [--seg N]

Why: the stitch (phase0/cameras.json) is fitted on the strip where the two
phones overlap, which can't separate each phone's focal length from the
angle between them (the same Pixel 11 came out 4% apart on rec6 and rec7).
pitch.py then fits the pitch in that stitch's panorama and absorbs what the
cameras got wrong with its warp terms (rec7: w3 = 6.25 deg). A pitch is a
known shape in metres, so fitted together with the lenses it pins what the
overlap can't, and no warp is needed.

taps: pitch/taps.json (panorama degrees, only meaningful with the
cameras.json and background.png they were tapped on) -> pitch/taps_src.json:
each tap as a pixel in the upright 4K frame of the phone the panorama took
that pixel from, with hashes of the taps, panorama and cameras it came from.
Exact: the panorama is a grid of directions through those cameras. A tap is
only as right in that phone's frame as the panorama was there (Main's side
always; Secondary's as well as the stitch put Secondary's rotation for the
segment the panorama was built from, which isn't recorded).

choices: the panorama drawn through rough cameras with each phone on the
left (main_left.jpg, secondary_left.jpg), for the question that asks which
shows the pitch in one piece: the right order joins up (rec5: one halfway
line, the arc and touchlines run through), the wrong one shows two broken
halves. Two stills side by side don't settle it: the phones' views cross,
so each phone's inner edge isn't where the other's picture continues.

rough: phase0/cameras.json with no stitcher: each phone's tilt and roll
from the manifest's gravity sensor, its focal from the lens table (else
the manifest's field of view), the phones ROUGH_ANGLE deg apart with the
one the user named on the left. Good enough to draw a panorama to tap the
pitch on (taps are exact in the phone they land in, whatever the cameras);
not to render. Marked "rough" so the engine knows the fit must follow.

search: the joint fit from rough cameras, which keeps the start's angle
between the phones (rec5: 70 stays 70), so it is run from SEARCH_ANGLES x
START_HEIGHTS (pitch.START_POSE at those heights) and the start with the
smallest tap error (median + worst / 4, non-finite fits dropped) is
written to DIR (default <rec>/joint/). Needs taps (taps_src.json).

install: puts DIR's cameras.json and pitch.json into the recording
(phase0/cameras.json, the rough one kept as cameras_rough.json; pitch.json
with no warp), redraws pitch/background.png from the new cameras, turns
taps_src.json into taps.json in the new panorama's degrees, and removes
what depended on the old cameras (goals.json, review.json, overlays) for
the engine to remake.

fit: per-phone median backgrounds of one segment (default: the middle one),
their paint (pitch.line_mask, the recording's paint colour), and one least
squares over Main's and Secondary's focal length and two radial terms,
Secondary's rotation, the offset between the lenses (metres), and the pitch
pose (camera position and height, rotation, ground tilt, length, width; the
marking scale stays at pitch.json's). Residuals: every pitch marking point's
distance to paint in each phone that sees it, the taps in their phone's
pixels, and priors: focal lengths from the lens table (--lenses) or the
manifest's field of view, small distortion, a small lens offset. Main's
rotation stays the world frame, as in follow.Camera. Starts from
phase0/cameras.json and pitch/pitch.json (warp dropped), or from --start
DIR: another fit's directory, or a recording's pre_joint/ (the cameras and
pitch from before a joint fit was adopted). The robust loss covers the
paint and the taps; the priors are plain Gaussians. The last stage repeats
until it stops moving (report.json "fit": passes, cost, converged; a fit
that didn't converge says so on stderr). Writes <out> (default
<rec>/joint/): cameras.json (follow.setup_cameras' format plus "jointFit"),
pitch.json (warpDeg all 0), overlay_main.jpg, overlay_secondary.jpg and
report.json.

holdout: how well the fit places taps it wasn't given. The taps are dealt
into --folds groups; each group in turn is left out of a fit and scored by
that fit (px at 4K), against the same taps' error in the fit made with all
of them and in the start. The fair figure for "how close are the taps": a
fit with 20 free parameters and ~30 tap coordinates at weight 3 fits its
own taps closely whatever its geometry. Writes <out>/holdout.json (default
<rec>/joint/). Start from --start <rec>/pre_joint on an adopted recording
(its own cameras and pitch were fitted to every tap).

score: for each directory holding a cameras.json and pitch.json (a joint fit,
or "current" for phase0/cameras.json with pitch/pitch.json), the taps'
error in source px and on the ground (metres), and the mean distance from
each phone's paint to the fitted markings. Taps_src.json is the truth.

lenses: collects joint/cameras.json across recordings by phone model into
processing/spikes/lenses.json, the prior for fits without taps.

bench: pitchbench.py's marked-field benchmark scored in the phones' 4K pixels
(the fixture's traced markings carried through its cameras.json), so fits
made with different cameras compare.

Results so far (taps left out of the fit land about as close as with the
old stitch + warped pitch fit, with no warp; untapped fits not dependable
yet): docs/FINDINGS.md and docs/BACKLOG.md, 2026-11-09 and 2026-11-10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import phase0  # noqa: E402
import pitch as P  # noqa: E402

W, H = follow.SRC_W, follow.SRC_H
LENS_TABLE = Path(__file__).parent / "lenses.json"
ROUGH_ANGLE = 80.0     # rough cameras: degrees between the phones (the mount: rec5 81, rec7 79)
SEARCH_ANGLES = (74.0, 77.0, 80.0, 83.0, 86.0, 89.0)  # search: starts across the mount's range
START_HEIGHTS = (1.9, 3.0, 8.0)  # search: pitch.START_POSE at these camera heights (5 m diverged on rec5)
TAP_PX = 3.0          # how well a tap is placed, px at 4K (pitch.TAP_PX is 3 panorama px, ~2.5 here)
TAP_WEIGHT = 3.0      # as pitch.TAP_WEIGHT
TRUNCS = (32, 10, 5)  # px at 4K: coarse to fine (pitch.py's 40/12/6 panorama px)
FOCAL_SD_MANIFEST = 0.08  # the reported field of view can be ~5 deg off (Pixel 11 Pro): weak
FOCAL_SD_TAPS = 0.10      # with taps, the pitch decides
DIST_SD = 0.01        # k1, k2 prior sd: looser lets distortion absorb what the pitch should decide
OFFSET_SD_M = 0.03    # lens offset prior sd, each axis, about zero: the lenses are ~7 cm apart but which way
                      # depends on the mount, so the prior only holds the offset near that size (7 cm along
                      # one axis is 2.3 sd; looser ran to 20-36 cm without taps)
LOSS_SCALE = 0.3      # soft-L1 scale for the paint and tap residuals (not the priors: see soften)
MAX_PASSES = 8        # the last stage is repeated until it stops moving, at most this many times
CONV_COST = 1e-3      # ... a pass that lowers the cost by under this share
CONV_STEP = 0.05      # ... and moves no parameter by more than this in X_SCALE units (L, W 5 cm; h 5 mm; f 0.5)

# Parameter vector.
FM, K1M, K2M, FS, K1S, K2S = range(6)
ROT = slice(6, 9)     # Secondary's rotation, Rodrigues, on top of cameras.json's
OFF = slice(9, 12)    # Secondary's lens relative to Main's, metres, world frame
POSE = slice(12, 21)  # pitch.PitchPose's first nine: cx, cz, h, rot, tilt_x, tilt_z, L, W, k
NPAR = 21
K_IDX = 12 + 8
X_SCALE = np.r_[10, 0.01, 0.01, 10, 0.01, 0.01, 0.002, 0.002, 0.002, 0.02, 0.02, 0.02, P.X_SCALE[:9]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cameras(path: Path) -> tuple[follow.Camera, follow.Camera, dict]:
    j = json.loads(path.read_text())
    t = (j.get("lineFit") or {}).get("offsetHeights")
    return (follow.Camera(*j["main"][:5], np.array(j["main"][5])),
            follow.Camera(*j["secondary"][:5], np.array(j["secondary"][5]), t=t), j)


def load_pose(path: Path) -> P.PitchPose:
    return P.pose_from_json(json.loads(path.read_text()))


def rec_info(rec: Path):
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    return manifest, report, devs, flip


# ------------------------------------------------------------------ taps

def taps_main(rec: Path) -> None:
    out = rec / "pitch"
    cams_path = rec / "phase0" / "cameras.json"
    cm, cs, _ = load_cameras(cams_path)
    taps = json.loads((out / "taps.json").read_text())
    a, b = cm.at(W), cs.at(W)
    res = {}
    for name, t in taps.items():
        if t.get("skipped"):
            res[name] = {"skipped": True}
            continue
        d = follow.direction(np.radians([t["yaw"]]), np.radians([t["pitch"]]))
        seen = {}
        for role, c in (("main", a), ("secondary", b)):
            x, y, ok = c.project(d)
            if ok[0]:
                seen[role] = (float(x[0]), float(y[0]), min(x[0], W - 1 - x[0], y[0], H - 1 - y[0]))
        if not seen:
            res[name] = {"skipped": True, "reason": "outside both frames"}
            continue
        # The panorama takes each pixel from the phone that sees it farther from its frame edge (pitch.background).
        role = max(seen, key=lambda r: seen[r][2])
        e = {"phone": role, "x": round(seen[role][0], 2), "y": round(seen[role][1], 2)}
        if t.get("fromFit"):
            e["fromFit"] = True
        other = [r for r in seen if r != role]
        if other:
            e["alsoIn"] = {other[0]: {"x": round(seen[other[0]][0], 2), "y": round(seen[other[0]][1], 2)}}
        res[name] = e
    doc = {
        "about": "Pitch keypoint taps as pixels in each phone's upright frame (jointfit.py taps).",
        "imageSize": [W, H],
        "source": {
            "taps": "pitch/taps.json", "tapsSha256": sha256(out / "taps.json"),
            "background": "pitch/background.png", "backgroundSha256": sha256(out / "background.png"),
            "cameras": json.loads(cams_path.read_text()), "camerasSha256": sha256(cams_path),
            "panorama": {"degPerPx": P.DEG_PER_PX, "yawRangeDeg": list(P.YAW_RANGE),
                         "pitchRangeDeg": list(P.PITCH_RANGE)},
        },
        "taps": res,
    }
    (out / "taps_src.json").write_text(json.dumps(doc, indent=1))
    n = sum(not v.get("skipped") for v in res.values())
    by = {r: sum(v.get("phone") == r for v in res.values()) for r in ("main", "secondary")}
    print(f"{rec.name}: {n} taps -> pitch/taps_src.json ({by['main']} in Main, {by['secondary']} in Secondary)")


def load_src_taps(rec: Path) -> dict:
    p = rec / "pitch" / "taps_src.json"
    if not p.exists():
        return {}
    return {n: t for n, t in json.loads(p.read_text())["taps"].items() if not t.get("skipped")}


# ------------------------------------------------------------ the model

class Model:
    """Parameter vector <-> both cameras (at 4K) and the pitch pose."""

    def __init__(self, cm: follow.Camera, cs: follow.Camera):
        self.cxm, self.cym, self.Rm = cm.ref[1], cm.ref[2], cm.R
        self.cxs, self.cys, self.Rs0 = cs.ref[1], cs.ref[2], cs.R

    def cameras(self, v):
        rs = cv2.Rodrigues(np.asarray(v[ROT], float))[0] @ self.Rs0
        return (follow.Camera(v[FM], self.cxm, self.cym, v[K1M], v[K2M], self.Rm, width=W),
                follow.Camera(v[FS], self.cxs, self.cys, v[K1S], v[K2S], rs, width=W))

    @staticmethod
    def pose(v) -> P.PitchPose:
        return P.PitchPose(np.r_[v[POSE], 0.0, 0.0, 0.0])

    def project(self, v, pts_world: np.ndarray, cams=None):
        """World points (metres from Main's lens) -> [(x, y, ok) in Main, in Secondary]."""
        a, b = cams or self.cameras(v)
        return [a.project(pts_world), b.project(pts_world - np.asarray(v[OFF], float))]


def start_vector(cm: follow.Camera, cs: follow.Camera, pose: P.PitchPose) -> np.ndarray:
    v = np.zeros(NPAR)
    v[FM], v[K1M], v[K2M] = cm.ref[0], cm.k1, cm.k2
    v[FS], v[K1S], v[K2S] = cs.ref[0], cs.k1, cs.k2
    # cameras.json's lens offset is in camera heights: metres with the pitch fit's height.
    v[OFF] = np.asarray(cs.t, float) * pose.h
    v[POSE] = pose.vector()[:9]
    return v


def manifest_focal(dev: dict) -> float:
    """Focal length at follow.REF_W from the reported horizontal field of view."""
    return (follow.REF_W / 2) / np.tan(np.radians(dev["video"]["horizontalFovDeg"]) / 2)


def lens_key(dev: dict) -> str:
    return f"{dev['model']}|{dev['video'].get('lens', '?')}|{dev['video']['width']}x{dev['video']['height']}"


def half_fov_deg(f: float, k1: float, k2: float) -> float:
    """Half the horizontal field of view, degrees: the angle whose ray lands on the frame's side edge
    (follow.Camera's radial model, f at follow.REF_W). f, k1 and k2 trade off against each other in
    a fit; this is what they agree on."""
    r = follow.REF_W / 2 / f
    for _ in range(50):
        r = follow.REF_W / 2 / f / (1 + k1 * r * r + k2 * r ** 4)
    return float(np.degrees(np.arctan(r)))


def dist_spread(e: dict) -> float:
    """How far a lens's k1 and k2 scatter across its fits: the larger sample sd of the two (0 for one fit)."""
    rows = e.get("fits", [])
    if len(rows) < 2:
        return 0.0
    return float(max(np.std([r["k1"] for r in rows], ddof=1), np.std([r["k2"] for r in rows], ddof=1)))


def lens_priors(manifest: dict, table: dict | None, have_taps: bool) -> dict:
    """{role: (f, f_sd, k1, k2, source)} for both phones, and "distSdBy": {role: sd} where the
    table's k1, k2 for that lens scatter by more than DIST_SD (the Pixel 11 Pro ultrawide: k1
    -0.022 on rec4cal, -0.0004 on rec6): a prior can't be tighter than the fits it averages."""
    out = {"distSdBy": {}}
    for d in manifest["devices"]:
        e = (table or {}).get(lens_key(d))
        if e:
            sd = max(e["fSd"] / e["f"], 0.01)
            out[d["role"]] = (e["f"], FOCAL_SD_TAPS if have_taps else sd, e["k1"], e["k2"], f"table (n={e['n']})")
            if dist_spread(e) > DIST_SD:
                out["distSdBy"][d["role"]] = dist_spread(e)
        else:
            out[d["role"]] = (manifest_focal(d), FOCAL_SD_TAPS if have_taps else FOCAL_SD_MANIFEST, 0.0, 0.0,
                              "manifest FOV")
    return out


# ---------------------------------------------------------- observations

def backgrounds(rec: Path, seg: int, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    pm, ps = cache / f"bg_main_seg{seg}.png", cache / f"bg_secondary_seg{seg}.png"
    if pm.exists() and ps.exists():
        return cv2.imread(str(pm)), cv2.imread(str(ps))
    _, report, devs, flip = rec_info(rec)
    bgs = follow.phone_backgrounds(rec, devs, report, flip, seg)
    if bgs is None:
        sys.exit(f"segment {seg} is too short for a background")
    cache.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pm), bgs[0])
    cv2.imwrite(str(ps), bgs[1])
    return bgs


class Paint:
    """One phone's paint and its distance transform."""

    def __init__(self, bg: np.ndarray, colour: str):
        self.mask = P.line_mask(bg, colour)
        self.dist = cv2.distanceTransform((self.mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        self._dt = {}

    def dt(self, trunc: float) -> np.ndarray:
        if trunc not in self._dt:
            self._dt[trunc] = cv2.GaussianBlur(np.minimum(self.dist, trunc), (0, 0), max(1.0, trunc / 6))
        return self._dt[trunc]


def sample(img: np.ndarray, x, y, ok) -> np.ndarray:
    r = np.zeros(len(x))
    if ok.any():
        r[ok] = map_coordinates(img, [y[ok], x[ok]], order=1)
    return r


# ------------------------------------------------------------- residuals

def tap_world_targets(pose: P.PitchPose, names) -> dict:
    tg = P.tap_targets(pose)
    return {n: tg[n] for n in names if n in tg}


def tap_residuals(model: Model, v, taps: dict, cams=None) -> np.ndarray:
    """Per tap, (dx, dy) / TAP_PX in its phone; a post base: its distance to the goal line."""
    pose = model.pose(v)
    tg = tap_world_targets(pose, taps)
    r = []
    for n, t in taps.items():
        k = 0 if t["phone"] == "main" else 1
        if n.endswith("post base"):
            s = -1 if n.startswith("left") else 1
            zz = np.linspace(-pose.W / 2, pose.W / 2, 400)
            x, y, ok = model.project(v, pose.directions(np.c_[np.full_like(zz, s * pose.L / 2), zz]), cams)[k]
            d = np.hypot(x - t["x"], y - t["y"])
            r.append([np.min(np.where(ok, d, 1e4)) / TAP_PX])
        elif n in tg:
            x, y, ok = model.project(v, pose.directions(np.array([tg[n]])), cams)[k]
            r.append([(x[0] - t["x"]) / TAP_PX, (y[0] - t["y"]) / TAP_PX] if ok[0] else [30.0, 30.0])
    return np.concatenate(r) if r else np.zeros(0)


def line_residuals(model: Model, v, paints, trunc, cams=None) -> np.ndarray:
    pose = model.pose(v)
    pts = pose.directions(P.pitch_lines(pose.L, pose.W, pose.k))
    out = []
    for (x, y, ok), p in zip(model.project(v, pts, cams), paints):
        ok = ok & (x < W - 1) & (y < H - 1)
        out.append(sample(p.dt(trunc), x, y, ok) / trunc)
    return np.concatenate(out)


def prior_residuals(v, priors) -> np.ndarray:
    dist_sd, off_sd = priors.get("distSd", DIST_SD), priors.get("offsetSdM", OFFSET_SD_M)
    by = priors.get("distSdBy", {})
    dm, ds = max(dist_sd, by.get("main", 0.0)), max(dist_sd, by.get("secondary", 0.0))
    (fm, sdm, k1m, k2m, _), (fs, sds, k1s, k2s, _) = priors["main"], priors["secondary"]
    return np.r_[(v[FM] - fm) / (sdm * fm), (v[FS] - fs) / (sds * fs),
                 (v[K1M] - k1m) / dm, (v[K2M] - k2m) / dm,
                 (v[K1S] - k1s) / ds, (v[K2S] - k2s) / ds,
                 np.asarray(v[OFF]) / off_sd]


def soften(r: np.ndarray, scale: float = LOSS_SCALE) -> np.ndarray:
    """Soft-L1 as a residual: squared, it is scipy's loss="soft_l1" at f_scale=scale (quadratic near
    zero, linear beyond ~scale), so paint off the lines and a misplaced tap pull, but not by their
    squares. Applied to the paint and the taps only. The priors stay plain: through the loss a
    prior 10 sd out costs 2.9 where a Gaussian costs 50, about seven stray paint points, and the
    sds above would mean little."""
    z = (np.asarray(r, float) / scale) ** 2
    return np.sign(r) * scale * np.sqrt(2 * (np.sqrt(1 + z) - 1))


# ------------------------------------------------------------------- fit

def fit(model: Model, v0, paints, taps, priors, verbose=True, log: dict | None = None) -> np.ndarray:
    """Coarse to fine, the pitch first, then the lenses with it. One least_squares call can stop
    short (its step falls under xtol; with one-sided differences a synthetic rec7 ended 15 cm high
    and 0.9 m long with the taps at 0.9 px), so the last stage is run again from its own result
    until a pass neither lowers the cost by CONV_COST nor moves a parameter by CONV_STEP, at most
    MAX_PASSES times, and the result says whether it got there. `log`, if given, gets "passes", "cost", "converged",
    "status" (scipy's, last pass) and "costs" (after each pass of the last stage)."""
    v = v0.copy()
    pose_free = [12, 13, 14, 15, 16, 17, 18, 19]  # not k
    lens_free = [FM, K1M, K2M, FS, K1S, K2S, 6, 7, 8, 9, 10, 11]
    stages = [(TRUNCS[0], pose_free), (TRUNCS[1], pose_free),
              (TRUNCS[1], lens_free + pose_free), (TRUNCS[2], lens_free + pose_free)]
    nl = None
    costs, converged, sol = [], False, None
    for k, (trunc, free) in enumerate(stages):
        free = list(free)
        last = k == len(stages) - 1

        def resid(sub, free=free, trunc=trunc):
            p = v.copy()
            p[free] = sub
            cams = model.cameras(p)
            r = [soften(line_residuals(model, p, paints, trunc, cams))]
            if taps:
                r.append(soften(TAP_WEIGHT * tap_residuals(model, p, taps, cams)))
            r.append(prior_residuals(p, priors))
            return np.concatenate(r)

        for n_pass in range(MAX_PASSES if last else 1):
            before = v[free].copy()
            # Central differences: the paint term is sampled from an image, and one-sided steps gave a
            # Jacobian biased enough to stall the solver far out (synthetic rec7: cost 7.5 against 0.77).
            sol = least_squares(resid, before, x_scale=X_SCALE[free], jac="3-point", diff_step=1e-3)
            moved = float(np.max(np.abs(sol.x - before) / X_SCALE[free]))
            v[free] = sol.x
            if not last:
                break
            gain = (costs[-1] - sol.cost) if costs else np.inf
            costs.append(float(sol.cost))
            if n_pass and gain <= CONV_COST * sol.cost and moved <= CONV_STEP:
                converged = True
                break
        if verbose:
            nl = nl or len(line_residuals(model, v, paints, trunc))
            print(f"  trunc {trunc:2d}px, {'lenses+pitch' if FM in free else 'pitch only  '}: "
                  f"f {v[FM]:.1f}/{v[FS]:.1f} @ {follow.REF_W}, camera {v[14]:.2f} m up, "
                  f"pitch {v[18]:.1f} x {v[19]:.1f} m, paint {trunc * np.abs(line_residuals(model, v, paints, trunc)).mean():.2f}px"
                  + (f", {len(costs)} passes" + ("" if converged else ", NOT converged") if last else ""))
    if not converged:
        print(f"  warning: the fit was still moving after {len(costs)} passes (cost "
              + " -> ".join(f"{c:.3f}" for c in costs[-2:]) + "): don't trust its lengths to better than the last change",
              file=sys.stderr, flush=True)
    if log is not None:
        log.update(passes=len(costs), cost=costs[-1], costs=[round(c, 4) for c in costs], converged=converged,
                   status=int(sol.status))
    return v


# ---------------------------------------------------------------- output

def ground_xz(pose: P.PitchPose, origin: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Where rays (world frame, from `origin`, metres from Main's lens) meet the ground, pitch metres."""
    o, dd = np.asarray(origin, float) @ pose.Rt, d @ pose.Rt
    s = (pose.h - o[1]) / np.where(dd[:, 1] > 1e-9, dd[:, 1], np.nan)
    X0, Z0 = o[0] + s * dd[:, 0], o[2] + s * dd[:, 2]
    c, si = np.cos(pose.rot), np.sin(pose.rot)
    return np.stack([c * X0 + si * Z0 + pose.cx, -si * X0 + c * Z0 + pose.cz], -1)


def cameras_json(cams, v, info) -> dict:
    a, b = cams
    rel = (np.asarray(v[OFF]) / v[14]).tolist()  # follow.Camera.t is in camera heights
    # Camera.ref holds the parameters at follow.REF_W, as cameras.json does.
    return {"main": [v[FM], a.ref[1], a.ref[2], v[K1M], v[K2M], a.R.tolist()],
            "secondary": [v[FS], b.ref[1], b.ref[2], v[K1S], v[K2S], b.R.tolist()],
            "lineFit": {"offsetHeights": [round(x, 5) for x in rel]}, "jointFit": info}


def draw_overlay(bg, xs, ys, ok, taps, role, pred):
    img = bg.copy()
    for x, y in zip(xs[ok].astype(int), ys[ok].astype(int)):
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    for n, t in taps.items():
        if t["phone"] != role:
            continue
        cv2.drawMarker(img, (int(t["x"]), int(t["y"])), (255, 255, 0), cv2.MARKER_CROSS, 30, 3)
        if n in pred:
            cv2.line(img, (int(t["x"]), int(t["y"])), tuple(int(c) for c in pred[n]), (0, 255, 255), 2)
    return cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)


def start_paths(rec: Path, d: Path | None) -> tuple[Path, Path]:
    """(cameras.json, pitch.json) of the recording itself (d None), of a fit directory, or of a
    pre_joint-style one (phase0/cameras.json and pitch/pitch.json inside it)."""
    if d is None:
        return rec / "phase0" / "cameras.json", rec / "pitch" / "pitch.json"
    if (d / "phase0" / "cameras.json").exists():
        return d / "phase0" / "cameras.json", d / "pitch" / "pitch.json"
    return d / "cameras.json", d / "pitch.json"


def fit_main(rec: Path, args) -> None:
    out = Path(args.out) if args.out else rec / "joint"
    manifest, report, _, _ = rec_info(rec)
    seg = args.seg if args.seg is not None else len(report["segments"]) // 2
    cp, pp = start_paths(rec, Path(args.start) if args.start else None)
    cm, cs, _ = load_cameras(cp)
    pose0 = load_pose(pp)
    taps = {} if args.no_taps else load_src_taps(rec)
    table = json.loads(Path(args.lenses).read_text()) if args.lenses else None
    if table and args.exclude_self:
        table = drop_recording(table, rec.name)
    rep = fit_write(rec, cm, cs, pose0, seg, taps, table, out, args.dist_sd, args.offset_sd)
    print_report(rec.name, out.name, rep)


def fit_write(rec: Path, cm, cs, pose0: P.PitchPose, seg: int, taps: dict, table: dict | None, out: Path,
              dist_sd: float = DIST_SD, offset_sd: float = OFFSET_SD_M, verbose: bool = True) -> dict:
    """One joint fit from these cameras and pitch pose; writes cameras.json, pitch.json, report.json
    and the overlays to `out`, returns the report."""
    out.mkdir(parents=True, exist_ok=True)
    manifest, _, _, _ = rec_info(rec)
    taps = {n: t for n, t in taps.items() if not t.get("fromFit")}
    priors = lens_priors(manifest, table, bool(taps))
    priors.update(distSd=dist_sd, offsetSdM=offset_sd)
    colour = P.paint_colour(rec)
    if verbose:
        print(f"{rec.name}: segment {seg}, {len(taps)} taps, paint {colour}; priors "
              + ", ".join(f"{r} f {priors[r][0]:.1f} +-{100 * priors[r][1]:.0f}% ({priors[r][4]})" for r in ("main", "secondary"))
              + f", k1/k2 +-{dist_sd}, lens offset +-{offset_sd} m")
    bgm, bgs = backgrounds(rec, seg, rec / "joint")
    paints = [Paint(bgm, colour), Paint(bgs, colour)]
    model = Model(cm, cs)
    v0 = start_vector(cm, cs, pose0)
    log: dict = {}
    v = fit(model, v0, paints, taps, priors, verbose=verbose, log=log)
    cams = model.cameras(v)
    pose = model.pose(v)
    info = {"segment": seg, "taps": len(taps), "paint": colour, "priors": {k: (list(p) if isinstance(p, tuple) else p) for k, p in priors.items()},
            "fit": log,
            "lensOffsetM": [round(float(x), 4) for x in v[OFF]],
            "models": {r: lens_key(d) for r, d in ((d["role"], d) for d in manifest["devices"])}}
    (out / "cameras.json").write_text(json.dumps(cameras_json(cams, v, info)))
    (out / "pitch.json").write_text(json.dumps({**P.pose_json(pose), "warpDeg": [0.0, 0.0, 0.0],
                                                "fittedWith": "joint" + ("+taps" if taps else "")}, indent=1))
    rep = score_dir(rec, out, paints)
    rep["fit"] = log
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    pts = pose.directions(P.pitch_lines(pose.L, pose.W, pose.k, step=0.2))
    proj = model.project(v, pts, cams)
    all_taps = load_src_taps(rec)
    for k, (role, bg) in enumerate((("main", bgm), ("secondary", bgs))):
        pred = {n: e["pred"] for n, e in rep["taps"].items() if e["phone"] == role and "pred" in e}
        cv2.imwrite(str(out / f"overlay_{role}.jpg"), draw_overlay(bg, *proj[k], all_taps, role, pred),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return rep


# ---------------------------------------------------------------- holdout

def tap_px(model: Model, v, taps: dict) -> dict[str, float]:
    """Each tap's error under v, px at 4K (a post base: its distance to the goal line)."""
    return {n: float(np.linalg.norm(tap_residuals(model, v, {n: t})) * TAP_PX) for n, t in taps.items()
            if n.endswith("post base") or n in P.tap_targets(model.pose(v))}


def holdout_folds(names: list[str], folds: int) -> list[list[str]]:
    """The taps dealt into `folds` groups, in name order: the same split every run."""
    names = sorted(names)
    folds = max(2, min(folds, len(names)))
    return [names[i::folds] for i in range(folds)]


def holdout_main(rec: Path, args) -> None:
    out = Path(args.out) if args.out else rec / "joint"
    out.mkdir(parents=True, exist_ok=True)
    manifest, report, _, _ = rec_info(rec)
    seg = args.seg if args.seg is not None else len(report["segments"]) // 2
    start = Path(args.start) if args.start else None
    cp, pp = start_paths(rec, start)
    cm, cs, _ = load_cameras(cp)
    pose0 = load_pose(pp)
    taps = {n: t for n, t in load_src_taps(rec).items() if not t.get("fromFit")}
    if len(taps) < 6:
        sys.exit(f"holdout needs taps to leave out: {len(taps)} in taps_src.json")
    table = json.loads(Path(args.lenses).read_text()) if args.lenses else None
    if table and args.exclude_self:
        table = drop_recording(table, rec.name)
    priors = lens_priors(manifest, table, True)
    priors.update(distSd=args.dist_sd, offsetSdM=args.offset_sd)
    bgm, bgs = backgrounds(rec, seg, rec / "joint")
    colour = P.paint_colour(rec)
    paints = [Paint(bgm, colour), Paint(bgs, colour)]
    model = Model(cm, cs)
    v0 = start_vector(cm, cs, pose0)
    before = {n: e["px"] for n, e in score_dir(rec, start or Path("current"), paints)["taps"].items() if "px" in e}

    def run(use: dict) -> tuple[np.ndarray, dict]:
        log: dict = {}
        return fit(model, v0, paints, use, priors, verbose=False, log=log), log

    v_all, log_all = run(taps)
    fitted = tap_px(model, v_all, taps)
    rows, held = [], {}
    for k, names in enumerate(holdout_folds(list(taps), args.folds)):
        v, log = run({n: t for n, t in taps.items() if n not in names})
        px = tap_px(model, v, {n: taps[n] for n in names})
        held.update(px)
        rows.append({"left out": names, "px": px, "cameraHeightM": float(v[14]), "lengthM": float(v[18]),
                     "widthM": float(v[19]), "fit": log})
        print(f"  fold {k + 1}: left out {', '.join(names)}: " + ", ".join(f"{px[n]:.1f}" for n in names if n in px)
              + f" px; pitch {v[18]:.2f} x {v[19]:.2f} m, camera {v[14]:.2f} m up"
              + ("" if log["converged"] else " (NOT converged)"), flush=True)
    names = sorted(held)

    def stat(d: dict) -> list[float]:
        a = np.array([d[n] for n in names if n in d and np.isfinite(d[n])])
        return [float(np.median(a)), float(a.max())] if len(a) else [float("nan")] * 2

    L, Wd, hh = (np.array([r[k] for r in rows]) for k in ("lengthM", "widthM", "cameraHeightM"))
    res = {"about": "jointfit.py holdout: each tap scored by a fit that wasn't given it (px at 4K)",
           "start": str(cp.parent), "segment": seg, "folds": rows,
           "medianWorstPx": {"start": stat(before), "fittedWithAll": stat(fitted), "leftOut": stat(held)},
           "perTap": {n: {"start": before.get(n), "fittedWithAll": fitted.get(n), "leftOut": held[n]} for n in names},
           "allTaps": {"cameraHeightM": float(v_all[14]), "lengthM": float(v_all[18]), "widthM": float(v_all[19]), "fit": log_all},
           "acrossFolds": {"lengthM": [float(L.min()), float(L.max())], "widthM": [float(Wd.min()), float(Wd.max())],
                           "cameraHeightM": [float(hh.min()), float(hh.max())]}}
    (out / "holdout.json").write_text(json.dumps(res, indent=1))
    m = res["medianWorstPx"]
    print(f"{rec.name}: {len(names)} taps, median / worst px at 4K: the start {m['start'][0]:.1f} / {m['start'][1]:.1f}, "
          f"fitted with all {m['fittedWithAll'][0]:.1f} / {m['fittedWithAll'][1]:.1f}, "
          f"left out {m['leftOut'][0]:.1f} / {m['leftOut'][1]:.1f}")
    print(f"  across folds: pitch {L.min():.2f}-{L.max():.2f} x {Wd.min():.2f}-{Wd.max():.2f} m, camera {hh.min():.2f}-{hh.max():.2f} m up "
          f"(all taps: {v_all[18]:.2f} x {v_all[19]:.2f} m, {v_all[14]:.2f} m)")


# ------------------------------------------------------- no stitch: rough

def rough_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera-to-world rotation (columns: camera x right, y down, z forward; world y down, as
    follow.Camera) from a yaw and the manifest's orientation (pitchDeg < 0 looks down; its rollDeg
    turns the other way to follow.Camera's)."""
    y, p, r = np.radians([yaw_deg, -pitch_deg, -roll_deg])
    z = np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])
    x0 = np.array([np.cos(y), 0.0, -np.sin(y)])
    y0 = np.cross(z, x0)
    return np.stack([np.cos(r) * x0 + np.sin(r) * y0, -np.sin(r) * x0 + np.cos(r) * y0, z], axis=1)


def rough_cameras(manifest: dict, left: str, angle: float, table: dict | None) -> dict:
    """cameras.json with no stitcher (see rough above); `left` is the phone on the left."""
    out = {}
    for d in manifest["devices"]:
        e = (table or {}).get(lens_key(d))
        f = e["f"] if e else manifest_focal(d)
        k1, k2 = (e["k1"], e["k2"]) if e else (0.0, 0.0)
        yaw = (-angle / 2) if d["role"] == left else (angle / 2)
        o = d.get("orientation") or {}
        R = rough_rotation(yaw, o.get("pitchDeg", -8.0), o.get("rollDeg", 0.0))
        out[d["role"]] = [float(f), follow.REF_W / 2, follow.REF_W * 9 / 32, float(k1), float(k2), R.tolist()]
    out["secondaryLensOffsetCameraHeights"] = [0.0, 0.0, 0.0]
    out["rough"] = {"left": left, "angleDeg": angle, "focal": "lens table" if table else "manifest FOV",
                    "about": "no stitcher: gravity sensors, focal prior, the user's left/right; jointfit.py search follows"}
    return out


def rough_main(rec: Path, left: str, angle: float) -> None:
    cams = rec / "phase0" / "cameras.json"
    if cams.exists() and not json.loads(cams.read_text()).get("rough"):
        sys.exit(f"{cams} exists and isn't rough: the stitch worked, nothing to do")
    manifest = json.loads((rec / "manifest.json").read_text())
    table = json.loads(LENS_TABLE.read_text()) if LENS_TABLE.exists() else None
    j = rough_cameras(manifest, left, angle, table)
    cams.write_text(json.dumps(j))
    print(f"rough cameras: {left} on the left, {angle:g} deg apart, focal from {j['rough']['focal']}: "
          + ", ".join(f"{r} f {j[r][0]:.1f}" for r in ("main", "secondary")))


def choices_main(rec: Path, out: Path, seg: int | None) -> None:
    """The panorama through rough cameras with each phone on the left, 2000 px wide."""
    _, report, _, _ = rec_info(rec)
    seg = seg if seg is not None else len(report["segments"]) // 4
    out.mkdir(parents=True, exist_ok=True)
    cams = rec / "phase0" / "cameras.json"
    if cams.exists() and not json.loads(cams.read_text()).get("rough"):
        sys.exit(f"{cams} exists and isn't rough: the stitch worked, nothing to ask")
    manifest = json.loads((rec / "manifest.json").read_text())
    table = json.loads(LENS_TABLE.read_text()) if LENS_TABLE.exists() else None
    try:
        for left in ("main", "secondary"):
            cams.write_text(json.dumps(rough_cameras(manifest, left, ROUGH_ANGLE, table)))
            pano = P.Pano()
            bg = P.background(rec, seg, pano)
            cv2.imwrite(str(out / f"{left}_left.jpg"), cv2.resize(bg, (2000, round(2000 * bg.shape[0] / bg.shape[1])),
                                                                  interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    finally:
        cams.unlink(missing_ok=True)  # no cameras until the user has picked
    print(f"panoramas of segment {seg} with each phone on the left in {out}")


def tap_score(rep: dict) -> float:
    px = [t["px"] for t in rep["taps"].values() if "px" in t and not t.get("fromFit")]
    if not px or not np.all(np.isfinite(px)):
        return float("inf")
    return float(np.median(px) + max(px) / 4)


def search_main(rec: Path, seg: int | None, out: Path | None) -> None:
    """The joint fit from rough cameras at every start in SEARCH_ANGLES x START_HEIGHTS; the best goes to out."""
    import shutil

    out = out or rec / "joint"
    manifest, report, _, _ = rec_info(rec)
    seg = seg if seg is not None else len(report["segments"]) // 2
    cur = json.loads((rec / "phase0" / "cameras.json").read_text())
    rough = cur.get("rough") or {}
    if "left" not in rough:
        sys.exit("search starts from rough cameras (jointfit.py rough): phase0/cameras.json isn't")
    taps = load_src_taps(rec)
    if sum(not t.get("fromFit") for t in taps.values()) < 4:
        sys.exit("search needs taps: at least 4 tapped pitch points (taps_src.json)")
    table = json.loads(LENS_TABLE.read_text()) if LENS_TABLE.exists() else None
    tmp = out.parent / (out.name + "_search")
    best, results = None, []
    for angle in SEARCH_ANGLES:
        j = rough_cameras(manifest, rough["left"], angle, table)
        cm = follow.Camera(*j["main"][:5], np.array(j["main"][5]))
        cs = follow.Camera(*j["secondary"][:5], np.array(j["secondary"][5]), t=[0.0, 0.0, 0.0])
        for h in START_HEIGHTS:
            start = list(P.START_POSE)
            start[2] = h
            d = tmp / f"a{angle:g}_h{h:g}"
            try:
                rep = fit_write(rec, cm, cs, P.PitchPose(start), seg, taps, None, d, verbose=False)
                sc = tap_score(rep)
            except Exception as e:  # a start that runs off (the fit's own numerics)
                print(f"  {angle:g} deg, {h:g} m: failed ({e})", flush=True)
                continue
            fitted = json.loads((d / "cameras.json").read_text())
            ang = np.degrees(np.arctan2(fitted["main"][5][0][2], fitted["main"][5][2][2])
                             - np.arctan2(fitted["secondary"][5][0][2], fitted["secondary"][5][2][2]))
            print(f"  start {angle:g} deg apart, {h:g} m up: tap score {sc:.1f} px, fitted {abs(ang):.1f} deg apart", flush=True)
            results.append({"angleDeg": angle, "heightM": h, "score": sc if np.isfinite(sc) else None,
                            "fittedAngleDeg": round(float(abs(ang)), 2)})
            if np.isfinite(sc) and (best is None or sc < best[0]):
                best = (sc, d)
    if best is None:
        sys.exit("no start converged: check the taps")
    out.mkdir(parents=True, exist_ok=True)  # joint/ also caches the backgrounds: copy over, don't replace
    for f in best[1].iterdir():
        shutil.copy2(f, out / f.name)
    rep = json.loads((out / "report.json").read_text())
    rep["search"] = {"best": best[1].name, "starts": results}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"best start {best[1].name}: tap score {best[0]:.1f} px -> {out}")
    print_report(rec.name, out.name, rep)


def taps_in_angles(taps_src: dict, cams) -> dict:
    """taps_src.json's taps (phone pixels) as panorama degrees through these cameras (at W)."""
    taps = {}
    for n, t in taps_src["taps"].items():
        if t.get("skipped"):
            taps[n] = {"skipped": True}
            continue
        d = cams[0 if t["phone"] == "main" else 1].unproject(np.array([[t["x"], t["y"]]], float))[0]
        taps[n] = {"yaw": round(float(np.degrees(np.arctan2(d[0], d[2]))), 3),
                   "pitch": round(float(np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))), 3)}
        if t.get("fromFit"):
            taps[n]["fromFit"] = True
    return taps


def install_main(rec: Path, src: Path, seg: int | None) -> None:
    """A joint fit's cameras and pitch into the recording, before anything else used the old ones."""
    import shutil

    _, report, _, _ = rec_info(rec)
    seg = seg if seg is not None else len(report["segments"]) // 2
    p0, pd = rec / "phase0", rec / "pitch"
    cur = json.loads((p0 / "cameras.json").read_text())
    if cur.get("rough"):
        shutil.copy2(p0 / "cameras.json", p0 / "cameras_rough.json")
    shutil.copy2(src / "cameras.json", p0 / "cameras.json")
    shutil.copy2(src / "pitch.json", pd / "pitch.json")
    new_cm, new_cs, _ = load_cameras(p0 / "cameras.json")
    if (pd / "taps_src.json").exists():
        taps = taps_in_angles(json.loads((pd / "taps_src.json").read_text()), (new_cm.at(W), new_cs.at(W)))
        (pd / "taps.json").write_text(json.dumps(taps, indent=1))
    pano = P.Pano()
    bg = P.background(rec, seg, pano)
    cv2.imwrite(str(pd / "background.png"), bg)
    cv2.imwrite(str(pd / "background.jpg"), cv2.resize(bg, (pano.w // 3, pano.h // 3)))
    over = cv2.resize(P.draw(bg, pano, load_pose(pd / "pitch.json")), (pano.w // 2, pano.h // 2))
    # overlay_taps.jpg is the overlay for a fit from taps (what engine.py's pitch question shows)
    for n in ("overlay.jpg", "overlay_taps.jpg"):
        cv2.imwrite(str(pd / n), over, [cv2.IMWRITE_JPEG_QUALITY, 88])
    for n in ("goals.json", "review.json"):
        (pd / n).unlink(missing_ok=True)
    # taps_src.json stays: its pixels don't depend on the cameras (only its "source" record is old).
    print(f"installed {src}: cameras and pitch; panorama from segment {seg}; goals and review to remake")


# ----------------------------------------------------------------- score

def model_from_dir(rec: Path, d: Path):
    """(project(points_world) -> per phone (x, y, ok), offset, pose) for a fit directory or "current"."""
    cp, pp = start_paths(rec, None if d.name == "current" else d)
    cm, cs, _ = load_cameras(cp)
    pose = load_pose(pp)
    a, b = cm.at(W), cs.at(W)
    warped = any(abs(w) > 0 for w in pose.warp_json())

    def project(pts):
        if not warped and not cs.t.any():
            return [a.project(pts), b.project(pts)]
        if not warped:  # joint fits: true 3D through each lens (the offset is in camera heights)
            off = np.asarray(cs.t) * pose.h
            bb = follow.Camera(*cs.ref, width=W)
            return [a.project(pts), bb.project(pts - off)]
        # A warped pose lives in its panorama: angles (with the warp) -> directions -> each camera.
        ang = np.radians(pose._to_angles(pts))
        d = follow.direction(ang[:, 0], ang[:, 1])
        return [a.project(d), b.project(d)]

    def rays(k, xy):
        """Pixels in phone k -> (origin, world direction) for the ground; undoes the warp if any."""
        c = a if k == 0 else b
        if not warped:
            if k == 1 and cs.t.any():
                bb = follow.Camera(*cs.ref, width=W)
                return np.asarray(cs.t) * pose.h, bb.unproject(xy)
            return np.zeros(3), c.unproject(xy)
        d = c.unproject(xy)
        yaw = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
        pit = np.degrees(np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2])))
        p = np.radians(pose.unwarp(yaw, pit))
        return np.zeros(3), follow.direction(np.radians(yaw), p)

    return project, rays, pose


def score_dir(rec: Path, d: Path, paints) -> dict:
    project, rays, pose = model_from_dir(rec, d)
    taps = load_src_taps(rec)
    tg = P.tap_targets(pose)
    res = {"taps": {}, "paint": {}}
    for n, t in taps.items():
        k = 0 if t["phone"] == "main" else 1
        e = {"phone": t["phone"], "fromFit": bool(t.get("fromFit"))}
        o, dirn = rays(k, np.array([[t["x"], t["y"]]], float))
        g = ground_xz(pose, o, dirn)[0]
        if n.endswith("post base"):
            s = -1 if n.startswith("left") else 1
            zz = np.linspace(-pose.W / 2, pose.W / 2, 2000)
            x, y, ok = project(pose.directions(np.c_[np.full_like(zz, s * pose.L / 2), zz]))[k]
            e["px"] = float(np.min(np.where(ok, np.hypot(x - t["x"], y - t["y"]), np.inf)))
            e["m"] = float(abs(g[0] - s * pose.L / 2))
        elif n in tg:
            x, y, ok = project(pose.directions(np.array([tg[n]])))[k]
            e["px"] = float(np.hypot(x[0] - t["x"], y[0] - t["y"])) if ok[0] else float("inf")
            e["pred"] = [float(x[0]), float(y[0])]
            e["m"] = float(np.hypot(*(g - np.array(tg[n]))))
        res["taps"][n] = e
    pts = pose.directions(P.pitch_lines(pose.L, pose.W, pose.k, step=0.2))
    for role, (x, y, ok), p in zip(("main", "secondary"), project(pts), paints):
        ok = ok & (x < W - 1) & (y < H - 1)
        dd = sample(np.minimum(p.dist, 40), x, y, ok)[ok]
        res["paint"][role] = {"medianPx": float(np.median(dd)), "within5px": float(np.mean(dd < 5)), "n": int(ok.sum())}
    return res


def print_report(rec_name: str, label: str, rep: dict) -> None:
    t = {n: e for n, e in rep["taps"].items() if "px" in e and not e["fromFit"]}
    if t:
        px = np.array([e["px"] for e in t.values()])
        m = np.array([e["m"] for e in t.values()])
        print(f"{rec_name} {label}: taps {np.median(px):.1f} / {px.max():.1f} px (median / worst), "
              f"{np.median(m):.2f} / {m.max():.2f} m on the ground")
    for role, e in rep["paint"].items():
        print(f"  {role:9s} paint: model points {e['medianPx']:.1f} px median from paint, {100 * e['within5px']:.0f}% within 5 px")


def score_main(rec: Path, dirs: list[str], seg: int | None) -> None:
    _, report, _, _ = rec_info(rec)
    seg = seg if seg is not None else len(report["segments"]) // 2
    bgm, bgs = backgrounds(rec, seg, rec / "joint")
    colour = P.paint_colour(rec)
    paints = [Paint(bgm, colour), Paint(bgs, colour)]
    reps = {}
    for d in dirs:
        dd = Path(d) if "/" in d else (rec / d if d != "current" else Path("current"))
        reps[d] = score_dir(rec, dd, paints)
        print_report(rec.name, d, reps[d])
    names = sorted({n for r in reps.values() for n in r["taps"]})
    print(f"\n  {'tap':32s}" + "".join(f"{d[-18:]:>20s}" for d in dirs) + "   (px at 4K / m)")
    for n in names:
        row = f"  {n:32s}"
        for d in dirs:
            e = reps[d]["taps"].get(n, {})
            row += f"{e.get('px', float('nan')):9.1f} /{e.get('m', float('nan')):6.2f}  " if "px" in e else f"{'-':>20s}"
        print(row + (" (from fit)" if any(r['taps'].get(n, {}).get('fromFit') for r in reps.values()) else ""))


# ---------------------------------------------------------------- lenses

def drop_recording(table: dict, rec_name: str) -> dict:
    out = {}
    for k, e in table.items():
        rows = [r for r in e["fits"] if r["recording"] != rec_name]
        if rows:
            out[k] = summarize(rows)
    return out


def summarize(rows: list[dict]) -> dict:
    """A lens's table entry from its fits. f, k1 and k2 are means of values that trade off within
    each fit; halfFovDeg (half_fov_deg of each fit: mean and spread) is the part the fits pin down,
    and distSd how far k1, k2 scatter (lens_priors widens its prior to it)."""
    f = np.array([r["f"] for r in rows])
    fov = np.array([half_fov_deg(r["f"], r["k1"], r["k2"]) for r in rows])
    return {"f": float(f.mean()), "fSd": float(f.std(ddof=1)) if len(f) > 1 else float(0.03 * f.mean()),
            "k1": float(np.mean([r["k1"] for r in rows])), "k2": float(np.mean([r["k2"] for r in rows])),
            "distSd": dist_spread({"fits": rows}), "halfFovDeg": float(fov.mean()),
            "halfFovSdDeg": float(fov.std(ddof=1)) if len(fov) > 1 else None,
            "n": len(rows), "fits": rows}


def lenses_main(recs: list[Path]) -> None:
    rows: dict[str, list] = {}
    for rec in recs:
        j = json.loads((rec / "joint" / "cameras.json").read_text())
        info = j["jointFit"]
        if not info.get("taps"):
            print(f"{rec.name}: fitted without taps, left out")
            continue
        for role in ("main", "secondary"):
            c = j[role]
            rows.setdefault(info["models"][role], []).append(
                {"recording": rec.name, "role": role, "f": c[0], "k1": c[3], "k2": c[4]})
    table = {k: summarize(v) for k, v in sorted(rows.items())}
    LENS_TABLE.write_text(json.dumps(table, indent=1) + "\n")
    for k, e in table.items():
        print(f"{k:45s} f {e['f']:7.1f} +- {e['fSd']:5.1f} @ {follow.REF_W}  k1 {e['k1']:+.4f}  k2 {e['k2']:+.4f} (+- {e['distSd']:.4f})  "
              f"half FOV {e['halfFovDeg']:.2f}" + (f" +- {e['halfFovSdDeg']:.2f}" if e["halfFovSdDeg"] is not None else "")
              + f" deg  n={e['n']}  " + " ".join(f"{r['recording']}:{r['f']:.1f}" for r in e["fits"]))
    print(f"wrote {LENS_TABLE}")


# ---------------------------------------------------------------- mirror

MIRROR_OWN = {"phase0", "pitch", "playeval", "follow", "detect"}


class Reangle:
    """Panorama angles under one camera pair -> under another, through the phone that saw them
    (the one farther from its frame edge, as the panoramas pick)."""

    def __init__(self, old: tuple, new: tuple):
        self.old = [c.at(W) for c in old]
        self.new = [c.at(W) for c in new]

    def __call__(self, yaw_deg, pitch_deg) -> tuple[np.ndarray, np.ndarray]:
        d = follow.direction(np.radians(np.atleast_1d(yaw_deg)), np.radians(np.atleast_1d(pitch_deg)))
        proj = [c.project(d) for c in self.old]
        edge = [np.where(ok, np.minimum.reduce([x, W - 1 - x, y, H - 1 - y]), -1) for x, y, ok in proj]
        k = np.where(edge[0] >= edge[1], 0, 1)
        out = np.full((len(d), 3), np.nan)
        for i in (0, 1):
            m = (k == i) & (edge[i] >= 0)
            if m.any():
                out[m] = self.new[i].unproject(np.c_[proj[i][0][m], proj[i][1][m]].astype(float))
        yaw = np.degrees(np.arctan2(out[:, 0], out[:, 2]))
        pit = np.degrees(np.arctan2(out[:, 1], np.hypot(out[:, 0], out[:, 2])))
        return yaw, pit


def convert_box(b: dict, re: Reangle) -> dict:
    t = np.linspace(0, 1, 21)
    y0, y1, p0, p1 = b["yaw0"], b["yaw1"], b["pitch0"], b["pitch1"]
    ys = np.r_[y0 + t * (y1 - y0), y0 + t * (y1 - y0), np.full(21, y0), np.full(21, y1)]
    ps = np.r_[np.full(21, p0), np.full(21, p1), p0 + t * (p1 - p0), p0 + t * (p1 - p0)]
    yy, pp = re(ys, ps)
    ok = np.isfinite(yy)
    return {**b, "yaw0": round(float(yy[ok].min()), 2), "yaw1": round(float(yy[ok].max()), 2),
            "pitch0": round(float(pp[ok].min()), 2), "pitch1": round(float(pp[ok].max()), 2),
            "convertedBy": "jointfit.py mirror"}


def convert_goals(goals: dict, old_pose: P.PitchPose, new_pose: P.PitchPose, re: Reangle) -> dict:
    """Goal frames (pitch metres under the old fit) -> the new fit, through both post bases."""
    out = []
    for g in goals["goals"]:
        th = np.radians(g.get("angleToGoalLineDeg", 0.0))
        d = np.array([np.sin(th), np.cos(th)])
        c = np.array([g["x"], g["z"]])
        bases = np.array([c - g["width"] / 2 * d, c + g["width"] / 2 * d])
        a = old_pose.angles(bases)
        yy, pp = re(a[:, 0], a[:, 1])
        nb = new_pose.to_pitch(yy, pp)
        v = nb[1] - nb[0]
        out.append({**g, "x": round(float(nb.mean(0)[0]), 3), "z": round(float(nb.mean(0)[1]), 3),
                    "width": round(float(np.hypot(*v)), 3),
                    "angleToGoalLineDeg": round(float(np.degrees(np.arctan2(v[0], v[1]))), 2),
                    "convertedBy": "jointfit.py mirror"})
    return {**goals, "goals": out}


def mirror_main(src: Path, joint: Path, dest: Path, bg_seg: int) -> None:
    """A recording that shares src's video but uses a joint fit's cameras and pitch: everything at
    the top level is linked except phase0/, pitch/, playeval/ (boxes and goals converted), follow/
    (empty: pass A is redone) and detect/ (left out: ball and role files are in the old angles)."""
    if dest.exists():
        sys.exit(f"{dest} exists")
    dest.mkdir(parents=True)
    for e in src.iterdir():
        if e.name not in MIRROR_OWN and not e.name.startswith("joint") and not e.name.startswith("."):
            (dest / e.name).symlink_to(e.resolve())
    (dest / "phase0").mkdir()
    for e in (src / "phase0").iterdir():
        if e.name != "cameras.json":
            (dest / "phase0" / e.name).symlink_to(e.resolve())
    (dest / "phase0" / "cameras.json").write_text((joint / "cameras.json").read_text())
    (dest / "follow").mkdir()
    old_cm, old_cs, _ = load_cameras(src / "phase0" / "cameras.json")
    new_cm, new_cs, _ = load_cameras(joint / "cameras.json")
    re = Reangle((old_cm, old_cs), (new_cm, new_cs))
    old_pose, new_pose = load_pose(src / "pitch" / "pitch.json"), load_pose(joint / "pitch.json")
    pd = dest / "pitch"
    pd.mkdir()
    (pd / "pitch.json").write_text((joint / "pitch.json").read_text())
    for n in ("paint.json", "taps_src.json"):
        if (src / "pitch" / n).exists():
            (pd / n).write_text((src / "pitch" / n).read_text())
    if (src / "pitch" / "goals.json").exists():
        g = convert_goals(json.loads((src / "pitch" / "goals.json").read_text()), old_pose, new_pose, re)
        (pd / "goals.json").write_text(json.dumps(g, indent=1))
        print("goals: " + ", ".join(f"side {x['side']} x {x['x']:.2f} z {x['z']:.2f} w {x['width']:.2f}"
                                    for x in g["goals"]))
    # taps.json in the new panorama's degrees, from the stitch-free taps
    ts = pd / "taps_src.json"
    if ts.exists():
        taps = taps_in_angles(json.loads(ts.read_text()), (new_cm.at(W), new_cs.at(W)))
        (pd / "taps.json").write_text(json.dumps(taps, indent=1))
    for sdir in sorted((src / "playeval").glob("*/")) if (src / "playeval").exists() else []:
        if not (sdir / "clip.json").exists():
            continue
        od = dest / "playeval" / sdir.name
        od.mkdir(parents=True)
        for n in ("clip.json", "items.json", "README.md"):
            if (sdir / n).exists():
                (od / n).write_text((sdir / n).read_text())
        boxes = json.loads((sdir / "boxes.json").read_text())
        nb = {k: convert_box(b, re) for k, b in boxes.items()}
        (od / "boxes.json").write_text(json.dumps(nb, indent=1))
        shift = np.median([abs(nb[k]["yaw0"] - b["yaw0"]) + abs(nb[k]["yaw1"] - b["yaw1"])
                           for k, b in boxes.items()]) / 2
        print(f"playeval/{sdir.name}: {len(nb)} boxes converted (median edge shift {shift:.2f} deg)")
    pano = P.Pano()
    how = f"panorama from segment {bg_seg}"
    try:
        bg = P.background(dest, bg_seg, pano)
    except subprocess.CalledProcessError:  # the segment's video isn't on disk
        cached = [joint / f"bg_{r}_seg{bg_seg}.png" for r in ("main", "secondary")]
        if not all(c.exists() for c in cached):
            sys.exit(f"segment {bg_seg}'s video isn't on disk and {joint} has no cached backgrounds of it")
        bg = pano_from_phones((new_cm, new_cs), [cv2.imread(str(c)) for c in cached], pano)
        how = f"panorama from the fit's cached backgrounds of segment {bg_seg} (its video isn't on disk)"
    cv2.imwrite(str(pd / "background.png"), bg)
    cv2.imwrite(str(pd / "background.jpg"), cv2.resize(bg, (pano.w // 3, pano.h // 3)))
    cv2.imwrite(str(pd / "overlay.jpg"), cv2.resize(P.draw(bg, pano, new_pose), (pano.w // 2, pano.h // 2)),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{dest}: cameras and pitch from {joint}, {how}")


def pano_from_phones(cams: tuple, bgs: list[np.ndarray], pano: P.Pano) -> np.ndarray:
    """pitch.background's panorama from one background image per phone (4K, upright) instead of
    the video: each pixel from the phone that sees it farther from its frame edge."""
    YY, PP = pano.angles()
    out = np.zeros((pano.h, pano.w, 3), np.uint8)
    cams = [c.at(W) for c in cams]
    for r0 in range(0, pano.h, 256):  # in bands: the whole panorama's directions at once is several GB
        sl = slice(r0, min(pano.h, r0 + 256))
        dirs = follow.direction(YY[sl], PP[sl])
        maps = []
        for c in cams:
            x, y, ok = c.project(dirs)
            d = np.minimum(np.minimum(x, c.w - 1 - x), np.minimum(y, c.h - 1 - y))
            maps.append((x, y, np.where(ok, d, -1)))
        wa = cv2.remap(bgs[0], maps[0][0], maps[0][1], cv2.INTER_LINEAR)
        wb = cv2.remap(bgs[1], maps[1][0], maps[1][1], cv2.INTER_LINEAR)
        band = np.where((maps[0][2] >= maps[1][2])[..., None], wa, wb)
        band[(maps[0][2] < 0) & (maps[1][2] < 0)] = 0
        out[sl] = band
    return out


def convert_passa_main(src: Path, dest: Path) -> None:
    """follow.py's pass A caches (player blobs per frame in panorama angles: yaw, pitch, area, the
    feet's pitch) from src's cameras into dest's, each through the phone that sees it farther from
    its frame edge. For a mirror whose video isn't on disk; with the video, playbench.py redoes
    pass A and that is the better copy (a blob at the seam may have come from the other phone)."""
    old_cm, old_cs, _ = load_cameras(src / "phase0" / "cameras.json")
    new_cm, new_cs, _ = load_cameras(dest / "phase0" / "cameras.json")
    re = Reangle((old_cm, old_cs), (new_cm, new_cs))
    (dest / "follow").mkdir(exist_ok=True)
    for p in sorted((src / "follow").glob("passA_*.json")):
        j = json.loads(p.read_text())
        flat = np.array([d for fr in j["dets"] for d in fr], float).reshape(-1, 4)
        if len(flat):
            yaw, pit = re(np.degrees(flat[:, 0]), np.degrees(flat[:, 1]))
            _, feet = re(np.degrees(flat[:, 0]), np.degrees(flat[:, 3]))
            ok = np.isfinite(yaw) & np.isfinite(feet)
            moved = float(np.median(np.abs(yaw[ok] - np.degrees(flat[ok, 0])))) if ok.any() else 0.0
            new = np.where(ok[:, None], np.c_[np.radians(yaw), np.radians(pit), flat[:, 2], np.radians(feet)], flat)
        else:
            new, ok, moved = flat, np.zeros(0, bool), 0.0
        out, i = [], 0
        for fr in j["dets"]:
            out.append([[float(v) for v in d] for d in new[i:i + len(fr)]])
            i += len(fr)
        (dest / "follow" / p.name).write_text(json.dumps({"dets": out, "extent": j["extent"]}))
        print(f"converted {p.name}: {int(ok.sum())} of {len(flat)} blobs, median yaw shift {moved:.3f} deg")


def convert_detect_main(src: Path, dest: Path, start: str, src_detect: Path | None = None,
                        dest_detect: Path | None = None) -> None:
    """A clip's ball candidates (ball_*_<start>_*.json) and roles (roles_*_<start>_*.json) from src's
    cameras and pitch into dest's (a mirror, or an adopting recording): each through the phone that
    saw it (its "phone"). start "*" takes every clip."""
    src_detect, dest_detect = src_detect or src / "detect", dest_detect or dest / "detect"
    old_cm, old_cs, _ = load_cameras(src / "phase0" / "cameras.json")
    new_cm, new_cs, _ = load_cameras(dest / "phase0" / "cameras.json")
    old = (old_cm.at(W), old_cs.at(W))
    new = (new_cm.at(W), new_cs.at(W))
    old_pose, new_pose = load_pose(src / "pitch" / "pitch.json"), load_pose(dest / "pitch" / "pitch.json")
    hl, hw = new_pose.L / 2 + P.ON_PITCH_MARGIN, new_pose.W / 2 + P.ON_PITCH_MARGIN

    def reangle(phone, yaw, pit):
        x, y, ok = old[phone].project(follow.direction(np.radians(yaw), np.radians(pit)))
        d = new[phone].unproject(np.c_[x, y].astype(float))
        return (np.degrees(np.arctan2(d[:, 0], d[:, 2])), np.degrees(np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))), ok)

    dest_detect.mkdir(exist_ok=True)
    for p in sorted(src_detect.glob(f"ball_*_{start}_*.json")):
        j = json.loads(p.read_text())
        for fr in j["frames"]:
            for d in fr["detections"]:
                y, q, ok = reangle(d["phone"], np.array([d["yaw"]]), np.array([d["pitch"]]))
                if not ok[0]:
                    continue
                d["yaw"], d["pitch"] = round(float(y[0]), 3), round(float(q[0]), 3)
                if d.get("xz") is not None:
                    x, z = new_pose.to_pitch(y, q)[0]
                    d["xz"] = None if not np.isfinite(x) else [round(float(x), 2), round(float(z), 2)]
                d["onPitch"] = bool(d.get("xz") is not None and abs(d["xz"][0]) <= hl and abs(d["xz"][1]) <= hw)
        (dest_detect / p.name).write_text(json.dumps(j))
        print(f"converted {p.name}")
    for p in sorted(src_detect.glob(f"roles_*_{start}_*.json")):
        j = json.loads(p.read_text())
        for fr in j["frames"]:
            for q in fr["people"]:
                a = old_pose.angles(np.array([q["xz"]], float))
                y, pp, ok = reangle(q["phone"], a[:, 0], a[:, 1])
                if ok[0]:
                    x, z = new_pose.to_pitch(y, pp)[0]
                    if np.isfinite(x):
                        q["xz"] = [round(float(x), 2), round(float(z), 2)]
        (dest_detect / p.name).write_text(json.dumps(j))
        print(f"converted {p.name}")


ADOPT_PITCH = ("pitch.json", "goals.json", "taps.json", "taps_src.json", "background.png", "background.jpg",
               "overlay.jpg")


def adopt_main(rec: Path, mirror: Path, keep: str = "pre_joint") -> None:
    """Puts a mirror's joint cameras and pitch into the recording itself. Everything replaced goes to
    <rec>/<keep>/ first (phase0/cameras.json, the pitch/ files, playeval boxes, follow/'s pass A
    caches, the whole detect/). Then: the mirror's pitch files, boxes and pass A caches are copied in,
    every ball candidate and role file is carried into the new angles (convert-detect), and what
    pitch/ derives from the panorama (lines.png, lines_<colour>.png, review.json: only those already
    there) is made again. `keep` must not exist: a recording adopted before keeps its earlier files
    where they are and needs another name."""
    import shutil

    pre = rec / keep
    if pre.exists():
        sys.exit(f"{pre} exists: already adopted (give another folder name to keep the replaced files in)")
    if not (mirror / "pitch" / "background.png").exists():
        sys.exit(f"{mirror} has no pitch/background.png: the mirror wasn't finished")
    (pre / "phase0").mkdir(parents=True)
    (pre / "pitch").mkdir()
    shutil.copy2(rec / "phase0" / "cameras.json", pre / "phase0" / "cameras.json")
    shutil.copy2(mirror / "phase0" / "cameras.json", rec / "phase0" / "cameras.json")
    derived = [f.name for f in (rec / "pitch").glob("lines*.png")] + ["review.json"]
    for n in (*ADOPT_PITCH, *derived):
        if (rec / "pitch" / n).exists():
            shutil.copy2(rec / "pitch" / n, pre / "pitch" / n)
        if (mirror / "pitch" / n).exists():
            shutil.copy2(mirror / "pitch" / n, rec / "pitch" / n)
    for b in sorted((mirror / "playeval").glob("*/boxes.json")) if (mirror / "playeval").exists() else []:
        dst = rec / "playeval" / b.parent.name / "boxes.json"
        (pre / "playeval" / b.parent.name).mkdir(parents=True)
        shutil.copy2(dst, pre / "playeval" / b.parent.name / "boxes.json")
        shutil.copy2(b, dst)
    (pre / "follow").mkdir()
    for f in (rec / "follow").glob("passA_*.json"):
        shutil.move(str(f), pre / "follow" / f.name)
    for f in (mirror / "follow").glob("passA_*.json"):
        shutil.copy2(f, rec / "follow" / f.name)
    shutil.move(str(rec / "detect"), pre / "detect")
    (rec / "detect").mkdir()
    for e in (pre / "detect").iterdir():
        if not (e.suffix == ".json" and e.name.split("_")[0] in ("ball", "roles")):
            (shutil.copytree if e.is_dir() else shutil.copy2)(e, rec / "detect" / e.name)
    convert_detect_main(pre, rec, "*", pre / "detect", rec / "detect")
    pd = rec / "pitch"
    bg = cv2.imread(str(pd / "background.png"))
    if (pd / "lines.png").exists():
        mask = P.line_mask(bg, P.paint_colour(rec))
        cv2.imwrite(str(pd / "lines.png"), mask)
        if (pd / "lines_small.png").exists():
            cv2.imwrite(str(pd / "lines_small.png"), cv2.resize(mask, (mask.shape[1] // 3, mask.shape[0] // 3),
                                                              interpolation=cv2.INTER_AREA))
    if any(n.startswith("lines_") and n != "lines_small.png" for n in derived):
        P.write_line_overlays(pd, bg)
    if (pd / "review.json").exists():
        P.review_main(rec)
    (pre / "README.md").write_text(
        f"# Before the joint fit adopted {time.strftime('%Y-%m-%d %H:%M')}\n\nAdopted from {mirror} "
        f"(jointfit.py adopt). These are the files it replaced: phase0/cameras.json, the pitch/ files "
        f"{', '.join((*ADOPT_PITCH, *derived))}, playeval boxes, follow/'s pass A caches and detect/ (ball "
        "candidates and roles were carried into the new angles, not re-detected). Outputs made before this "
        "(fullgame/, phases/, renders, paths) are in the old geometry. To undo: copy these back.\n")
    print(f"{rec}: adopted {mirror}; old files in {pre}")


def stitch_main(rec: Path, cams_paths: list[str], segs: list[int]) -> None:
    """How well the two phones' paint lines up in the overlap (follow.line_median, px at 4K) for each
    camera file, on segments' median backgrounds: the seam a render shows."""
    _, report, devs, flip = rec_info(rec)
    data = []
    for sg in segs:
        bgs = follow.phone_backgrounds(rec, devs, report, flip, sg)
        if bgs is not None:
            data.append(follow.LineData(sg, *bgs))
    for cp in cams_paths:
        cm, cs, _ = load_cameras(Path(cp))
        row = [follow.line_median(cm, cs, d) for d in data]
        print(f"{cp:45s} " + "  ".join(f"seg {d.seg}: {e:.2f} px ({n})" for d, (e, n) in zip(data, row)))


# ----------------------------------------------------------------- bench

def bench_main(fixture: Path, rec: Path, dirs: list[str]) -> None:
    """pitchbench.py's marked-field benchmark, scored in the phones' 4K pixels so fits with other cameras
    can be compared: each traced benchmark point goes through the fixture's cameras.json into the phone
    the panorama took it from, and is scored by its distance to the fit's marking in that phone."""
    import pitchbench as B

    bench = json.loads((fixture / "benchmark.json").read_text())
    fm, fs, _ = load_cameras(fixture / "cameras.json")
    cams = (fm.at(W), fs.at(W))
    truth = {}
    for n, m in bench["markings"].items():
        pts = np.array([q for p in m["pieces"] for q, t in zip(p["polyline"], p["traced"]) if t], float)
        if not len(pts):
            continue
        d = follow.direction(np.radians(pts[:, 0]), np.radians(pts[:, 1]))
        proj = [c.project(d) for c in cams]
        edge = [np.where(ok, np.minimum.reduce([x, W - 1 - x, y, H - 1 - y]), -1) for x, y, ok in proj]
        k = np.where(edge[0] >= edge[1], 0, 1)
        truth[n] = [np.c_[proj[i][0], proj[i][1]][(k == i) & (edge[i] >= 0)] for i in (0, 1)]
    res = {}
    for dname in dirs:
        project, _, pose = model_from_dir(rec, Path(dname) if dname == "current" else rec / dname)
        res[dname] = {}
        for i, n in enumerate(B.NAMES):
            if n not in truth:
                continue
            model = project(pose.directions(P.pitch_lines(pose.L, pose.W, pose.k, step=0.05, only=(i,))))
            ds = []
            for ph in (0, 1):
                x, y, ok = model[ph]
                if len(truth[n][ph]) and ok.any():
                    ds.append(B.nearest(truth[n][ph], np.c_[x[ok], y[ok]]))
            if ds:
                res[dname][n] = np.concatenate(ds)
    print(f"benchmark {fixture.name} in source px (4K), traced points: median / 90th percentile\n")
    print(f"{'':22s}" + "".join(f"{d:>18s}" for d in dirs))
    for g, members in B.GROUPS.items():
        row = f"{g:22s}"
        for d in dirs:
            v = [res[d][m] for m in members if m in res[d]]
            row += f"{f'{np.median(np.concatenate(v)):5.1f} / {np.percentile(np.concatenate(v), 90):5.1f}':>18s}" if v else f"{'-':>18s}"
        print(row)
    row = f"{'all markings':22s}"
    for d in dirs:
        v = np.concatenate(list(res[d].values()))
        row += f"{f'{np.median(v):5.1f} / {np.percentile(v, 90):5.1f}':>18s}"
    print(row)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "lenses":
        lenses_main([Path(r) for r in sys.argv[2:]])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench_main(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "mirror":
        mirror_main(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "convert-detect":
        convert_detect_main(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "convert-passa":
        convert_passa_main(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "adopt":
        adopt_main(Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:5])
        return
    if len(sys.argv) > 2 and sys.argv[2] in ("rough", "choices", "search", "install"):
        ap = argparse.ArgumentParser()
        ap.add_argument("rec", type=Path)
        ap.add_argument("cmd")
        ap.add_argument("dir", nargs="?")
        ap.add_argument("--seg", type=int)
        ap.add_argument("--left", choices=["main", "secondary"])
        ap.add_argument("--angle", type=float, default=ROUGH_ANGLE)
        ap.add_argument("--out")
        a = ap.parse_args()
        if a.cmd == "rough":
            if not a.left:
                sys.exit("rough needs --left main|secondary")
            rough_main(a.rec, a.left, a.angle)
        elif a.cmd == "choices":
            choices_main(a.rec, Path(a.dir), a.seg)
        elif a.cmd == "search":
            search_main(a.rec, a.seg, Path(a.out) if a.out else None)
        else:
            install_main(a.rec, Path(a.dir), a.seg)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "stitch":
        i = sys.argv.index("--segs")
        stitch_main(Path(sys.argv[2]), sys.argv[3:i], [int(s) for s in sys.argv[i + 1].split(",")])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", type=Path)
    ap.add_argument("cmd", choices=["taps", "fit", "score", "holdout"])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--seg", type=int)
    ap.add_argument("--no-taps", action="store_true")
    ap.add_argument("--lenses", help="lens table for the focal priors (lenses.json)")
    ap.add_argument("--exclude-self", action="store_true", help="leave this recording out of the lens table")
    ap.add_argument("--dist-sd", type=float, default=DIST_SD, help="prior sd of k1, k2")
    ap.add_argument("--offset-sd", type=float, default=OFFSET_SD_M, help="prior sd of the lens offset, metres")
    ap.add_argument("--start", help="start from this directory's cameras and pitch (a fit's, or <rec>/pre_joint)")
    ap.add_argument("--folds", type=int, default=5, help="holdout: groups the taps are dealt into")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "taps":
        taps_main(a.rec)
    elif a.cmd == "fit":
        fit_main(a.rec, a)
    elif a.cmd == "holdout":
        holdout_main(a.rec, a)
    else:
        score_main(a.rec, a.dirs, a.seg)


if __name__ == "__main__":
    main()
