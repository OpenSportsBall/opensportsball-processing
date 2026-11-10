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
                                                   [--dist-sd 0.01] [--offset-sd 0.03] [--out DIR]
    uv run processing/spikes/jointfit.py <rec> score <dir> [<dir> ...] [--seg N]
    uv run processing/spikes/jointfit.py lenses <rec> [<rec> ...]
    uv run processing/spikes/jointfit.py bench footage/fixtures/pitch/<name> <rec> <dir> [<dir> ...]

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
phase0/cameras.json and pitch/pitch.json (warp dropped). Writes <out>
(default <rec>/joint/): cameras.json (follow.setup_cameras' format plus
"jointFit"), pitch.json (warpDeg all 0), overlay_main.jpg,
overlay_secondary.jpg and report.json.

score: for each directory holding a cameras.json and pitch.json (a joint fit,
or "current" for phase0/cameras.json with pitch/pitch.json), the taps'
error in source px and on the ground (metres), and the mean distance from
each phone's paint to the fitted markings. Taps_src.json is the truth.

lenses: collects joint/cameras.json across recordings by phone model into
processing/spikes/lenses.json, the prior for fits without taps.

bench: pitchbench.py's marked-field benchmark scored in the phones' 4K pixels
(the fixture's traced markings carried through its cameras.json), so fits
made with different cameras compare.

Results so far (taps 2-3x closer than the current fits, no warp; untapped
fits not dependable yet): docs/FINDINGS.md and docs/BACKLOG.md, 2026-11-09.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
TAP_PX = 3.0          # how well a tap is placed, px at 4K (pitch.TAP_PX is 3 panorama px, ~2.5 here)
TAP_WEIGHT = 3.0      # as pitch.TAP_WEIGHT
TRUNCS = (32, 10, 5)  # px at 4K: coarse to fine (pitch.py's 40/12/6 panorama px)
FOCAL_SD_MANIFEST = 0.08  # the reported field of view can be ~5 deg off (Pixel 11 Pro): weak
FOCAL_SD_TAPS = 0.10      # with taps, the pitch decides
DIST_SD = 0.01        # k1, k2 prior sd: looser lets distortion absorb what the pitch should decide
OFFSET_SD_M = 0.03    # lens offset prior sd (the lenses are ~7 cm apart; looser ran to 20-36 cm without taps)

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


def lens_priors(manifest: dict, table: dict | None, have_taps: bool) -> dict:
    """{role: (f, f_sd, k1, k2, source)} for both phones."""
    out = {}
    for d in manifest["devices"]:
        e = (table or {}).get(lens_key(d))
        if e:
            sd = max(e["fSd"] / e["f"], 0.01)
            out[d["role"]] = (e["f"], FOCAL_SD_TAPS if have_taps else sd, e["k1"], e["k2"], f"table (n={e['n']})")
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
    (fm, sdm, k1m, k2m, _), (fs, sds, k1s, k2s, _) = priors["main"], priors["secondary"]
    return np.r_[(v[FM] - fm) / (sdm * fm), (v[FS] - fs) / (sds * fs),
                 (v[K1M] - k1m) / dist_sd, (v[K2M] - k2m) / dist_sd,
                 (v[K1S] - k1s) / dist_sd, (v[K2S] - k2s) / dist_sd,
                 np.asarray(v[OFF]) / off_sd]


# ------------------------------------------------------------------- fit

def fit(model: Model, v0, paints, taps, priors, verbose=True) -> np.ndarray:
    v = v0.copy()
    pose_free = [12, 13, 14, 15, 16, 17, 18, 19]  # not k
    lens_free = [FM, K1M, K2M, FS, K1S, K2S, 6, 7, 8, 9, 10, 11]
    stages = [(TRUNCS[0], pose_free), (TRUNCS[1], pose_free),
              (TRUNCS[1], lens_free + pose_free), (TRUNCS[2], lens_free + pose_free)]
    nl = None
    for trunc, free in stages:
        free = list(free)

        def resid(sub, free=free, trunc=trunc):
            p = v.copy()
            p[free] = sub
            cams = model.cameras(p)
            r = [line_residuals(model, p, paints, trunc, cams)]
            if taps:
                r.append(TAP_WEIGHT * tap_residuals(model, p, taps, cams))
            r.append(prior_residuals(p, priors))
            return np.concatenate(r)

        sol = least_squares(resid, v[free], loss="soft_l1", f_scale=0.3, x_scale=X_SCALE[free], diff_step=1e-3)
        v[free] = sol.x
        if verbose:
            nl = nl or len(line_residuals(model, v, paints, trunc))
            print(f"  trunc {trunc:2d}px, {'lenses+pitch' if FM in free else 'pitch only  '}: "
                  f"f {v[FM]:.1f}/{v[FS]:.1f} @ {follow.REF_W}, camera {v[14]:.2f} m up, "
                  f"pitch {v[18]:.1f} x {v[19]:.1f} m, paint {trunc * np.abs(sol.fun[:nl]).mean():.2f}px")
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


def fit_main(rec: Path, args) -> None:
    out = Path(args.out) if args.out else rec / "joint"
    out.mkdir(parents=True, exist_ok=True)
    manifest, report, _, _ = rec_info(rec)
    nseg = len(report["segments"])
    seg = args.seg if args.seg is not None else nseg // 2
    cm, cs, _ = load_cameras(rec / "phase0" / "cameras.json")
    pose0 = load_pose(rec / "pitch" / "pitch.json")
    taps = {} if args.no_taps else load_src_taps(rec)
    taps = {n: t for n, t in taps.items() if not t.get("fromFit")}
    table = json.loads(Path(args.lenses).read_text()) if args.lenses else None
    if table and args.exclude_self:
        table = drop_recording(table, rec.name)
    priors = lens_priors(manifest, table, bool(taps))
    priors.update(distSd=args.dist_sd, offsetSdM=args.offset_sd)
    colour = P.paint_colour(rec)
    print(f"{rec.name}: segment {seg}, {len(taps)} taps, paint {colour}; priors "
          + ", ".join(f"{r} f {priors[r][0]:.1f} +-{100 * priors[r][1]:.0f}% ({priors[r][4]})" for r in ("main", "secondary"))
          + f", k1/k2 +-{args.dist_sd}, lens offset +-{args.offset_sd} m")
    bgm, bgs = backgrounds(rec, seg, rec / "joint")
    paints = [Paint(bgm, colour), Paint(bgs, colour)]
    model = Model(cm, cs)
    v0 = start_vector(cm, cs, pose0)
    v = fit(model, v0, paints, taps, priors)
    cams = model.cameras(v)
    pose = model.pose(v)
    info = {"segment": seg, "taps": len(taps), "paint": colour, "priors": {k: (list(p) if isinstance(p, tuple) else p) for k, p in priors.items()},
            "lensOffsetM": [round(float(x), 4) for x in v[OFF]],
            "models": {r: lens_key(d) for r, d in ((d["role"], d) for d in manifest["devices"])}}
    (out / "cameras.json").write_text(json.dumps(cameras_json(cams, v, info)))
    (out / "pitch.json").write_text(json.dumps({**P.pose_json(pose), "warpDeg": [0.0, 0.0, 0.0],
                                                "fittedWith": "joint" + ("+taps" if taps else "")}, indent=1))
    rep = score_dir(rec, out, paints)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    pts = pose.directions(P.pitch_lines(pose.L, pose.W, pose.k, step=0.2))
    proj = model.project(v, pts, cams)
    all_taps = load_src_taps(rec)
    for k, (role, bg) in enumerate((("main", bgm), ("secondary", bgs))):
        pred = {n: e["pred"] for n, e in rep["taps"].items() if e["phone"] == role and "pred" in e}
        cv2.imwrite(str(out / f"overlay_{role}.jpg"), draw_overlay(bg, *proj[k], all_taps, role, pred),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print_report(rec.name, out.name, rep)


# ----------------------------------------------------------------- score

def model_from_dir(rec: Path, d: Path):
    """(project(points_world) -> per phone (x, y, ok), offset, pose) for a fit directory or "current"."""
    if d.name == "current":
        cp, pp = rec / "phase0" / "cameras.json", rec / "pitch" / "pitch.json"
    else:
        cp, pp = d / "cameras.json", d / "pitch.json"
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
    f = np.array([r["f"] for r in rows])
    return {"f": float(f.mean()), "fSd": float(f.std(ddof=1)) if len(f) > 1 else float(0.03 * f.mean()),
            "k1": float(np.mean([r["k1"] for r in rows])), "k2": float(np.mean([r["k2"] for r in rows])),
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
        print(f"{k:45s} f {e['f']:7.1f} +- {e['fSd']:5.1f} @ {follow.REF_W}  k1 {e['k1']:+.4f}  k2 {e['k2']:+.4f}  n={e['n']}  "
              + " ".join(f"{r['recording']}:{r['f']:.1f}" for r in e["fits"]))
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
        cams = (new_cm.at(W), new_cs.at(W))
        taps = {}
        for n, t in json.loads(ts.read_text())["taps"].items():
            if t.get("skipped"):
                taps[n] = {"skipped": True}
                continue
            d = cams[0 if t["phone"] == "main" else 1].unproject(np.array([[t["x"], t["y"]]], float))[0]
            taps[n] = {"yaw": round(float(np.degrees(np.arctan2(d[0], d[2]))), 3),
                       "pitch": round(float(np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))), 3)}
            if t.get("fromFit"):
                taps[n]["fromFit"] = True
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
    bg = P.background(dest, bg_seg, pano)
    cv2.imwrite(str(pd / "background.png"), bg)
    cv2.imwrite(str(pd / "background.jpg"), cv2.resize(bg, (pano.w // 3, pano.h // 3)))
    cv2.imwrite(str(pd / "overlay.jpg"), cv2.resize(P.draw(bg, pano, new_pose), (pano.w // 2, pano.h // 2)),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{dest}: cameras and pitch from {joint}, panorama from segment {bg_seg}")


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


def adopt_main(rec: Path, mirror: Path) -> None:
    """Puts a mirror's joint cameras and pitch into the recording itself. Everything replaced goes to
    <rec>/pre_joint/ first (phase0/cameras.json, the pitch/ files, playeval boxes, follow/'s pass A
    caches, the whole detect/). Then: the mirror's pitch files, boxes and pass A caches are copied in,
    and every ball candidate and role file is carried into the new angles (convert-detect)."""
    import shutil

    pre = rec / "pre_joint"
    if pre.exists():
        sys.exit(f"{pre} exists: already adopted")
    (pre / "phase0").mkdir(parents=True)
    (pre / "pitch").mkdir()
    shutil.copy2(rec / "phase0" / "cameras.json", pre / "phase0" / "cameras.json")
    shutil.copy2(mirror / "phase0" / "cameras.json", rec / "phase0" / "cameras.json")
    for n in ADOPT_PITCH:
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
    (pre / "README.md").write_text(
        f"# Before the joint fit\n\nAdopted {time.strftime('%Y-%m-%d %H:%M')} from {mirror} "
        f"(jointfit.py adopt). These are the files it replaced: phase0/cameras.json, the pitch/ files "
        f"{', '.join(ADOPT_PITCH)}, playeval boxes, follow/'s pass A caches and detect/ (ball candidates and "
        "roles were carried into the new angles, not re-detected). Outputs made before this (fullgame/, "
        "phases/, renders, paths) are in the old geometry. To undo: copy these back.\n")
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
    if len(sys.argv) > 1 and sys.argv[1] == "adopt":
        adopt_main(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "stitch":
        i = sys.argv.index("--segs")
        stitch_main(Path(sys.argv[2]), sys.argv[3:i], [int(s) for s in sys.argv[i + 1].split(",")])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", type=Path)
    ap.add_argument("cmd", choices=["taps", "fit", "score"])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--seg", type=int)
    ap.add_argument("--no-taps", action="store_true")
    ap.add_argument("--lenses", help="lens table for the focal priors (lenses.json)")
    ap.add_argument("--exclude-self", action="store_true", help="leave this recording out of the lens table")
    ap.add_argument("--dist-sd", type=float, default=DIST_SD, help="prior sd of k1, k2")
    ap.add_argument("--offset-sd", type=float, default=OFFSET_SD_M, help="prior sd of the lens offset, metres")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "taps":
        taps_main(a.rec)
    elif a.cmd == "fit":
        fit_main(a.rec, a)
    else:
        score_main(a.rec, a.dirs, a.seg)


if __name__ == "__main__":
    main()
