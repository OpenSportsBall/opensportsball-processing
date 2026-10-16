#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4", "torchvision>=0.19"]
# ///
"""Where the players are looking: a ball witness from body orientation.

    uv run processing/spikes/facing.py footage/rec3 <set>

At each labelled moment of a clip set, every tracked player (follow.py's
pass A detections) is cropped from the phone frame that sees them best
and run through a COCO keypoint model (torchvision Keypoint R-CNN). The
shoulders give the body's facing: the observed shoulder width over the
expected width gives how far the body is turned from the camera, the
shoulder order says towards or away, and the nose's offset says left or
right. On the pitch that is a ray from the player; the point nearest all
the rays (players facing away from it dropped, twice) is where the
players are looking, which is usually the ball.

Prints the distance from that point to the labelled ball, and the
found-rate under balleval.py's tolerance. Writes facing_<set>.json with
the per-moment points for witness.py.
"""

from __future__ import annotations

import io
import contextlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import balleval as E  # noqa: E402
import follow  # noqa: E402
import pitch as P  # noqa: E402
import witness as Wt  # noqa: E402

SHOULDER_M = 0.42       # shoulder width
BODY_M = 1.75
MIN_KP = 0.3            # keypoint score to count
MIN_BODY_PX = 24        # skip people smaller than this (keypoints fail)
CROP_UP = 3.0           # crop side as a multiple of the body height
SHOULDER_KP, HIP_KP = ((5, 6), (11, 12))


def facing_from_keypoints(kp, ks, body_px):
    """Facing angle from the camera axis (0 = towards the camera, pi = away),
    signed (+ = turned to the image's right), or None."""
    ls, rs = kp[5], kp[6]
    if ks[5] < MIN_KP or ks[6] < MIN_KP:
        return None
    dx = ls[0] - rs[0]                        # left shoulder right of right shoulder: facing the camera
    width = abs(dx)
    expect = SHOULDER_M / BODY_M * body_px
    turn = float(np.arccos(np.clip(width / max(expect, 1e-6), 0, 1)))  # 0: square on, pi/2: side on
    towards = dx > 0
    mid = (ls + rs) / 2
    nose_side = 0.0
    if ks[0] >= MIN_KP:
        nose_side = float(np.sign(kp[0][0] - mid[0]))
    elif ks[3] >= MIN_KP or ks[4] >= MIN_KP:  # ears
        e = kp[3] if ks[3] >= ks[4] else kp[4]
        nose_side = float(np.sign(e[0] - mid[0]))
    if nose_side == 0:
        nose_side = 1.0
    ang = turn if towards else np.pi - turn
    return nose_side * ang


def vote_point(origins, dirs, weights, pose, cell=2.0, sigma=6.0):
    """Density vote: each ray deposits weight along its path (within the
    pitch), falling off with distance from the ray; the densest cell."""
    xs = np.arange(-pose.L / 2, pose.L / 2, cell) + cell / 2
    zs = np.arange(-pose.W / 2, pose.W / 2, cell) + cell / 2
    X, Z = np.meshgrid(xs, zs)
    grid = np.stack([X, Z], -1)
    acc = np.zeros(X.shape)
    for o, u, w in zip(origins, dirs, weights):
        rel = grid - o
        along = rel @ u
        perp = np.abs(rel[..., 0] * u[1] - rel[..., 1] * u[0])
        acc += w * (along > 0) * np.exp(-0.5 * (perp / sigma) ** 2) * np.exp(-along / 60.0)
    iy, ix = np.unravel_index(np.argmax(acc), acc.shape)
    return grid[iy, ix]


def attention_point(origins, dirs, weights, rounds=2):
    """Point nearest the rays (least squares over lines), dropping rays that
    point away from it, `rounds` times."""
    o, u, w = np.asarray(origins), np.asarray(dirs), np.asarray(weights, float)
    keep = np.ones(len(o), bool)
    pt = None
    for _ in range(rounds + 1):
        if keep.sum() < 2:
            return pt, keep
        A = np.zeros((2, 2)); b = np.zeros(2)
        for oi, ui, wi in zip(o[keep], u[keep], w[keep]):
            M = wi * (np.eye(2) - np.outer(ui, ui))
            A += M; b += M @ oi
        if abs(np.linalg.det(A)) < 1e-6:
            return pt, keep
        pt = np.linalg.solve(A, b)
        keep = keep & (((pt - o) * u).sum(1) > 0)
    return pt, keep


def main(rec: Path, name: str) -> None:
    import torch
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

    with contextlib.redirect_stdout(io.StringIO()):
        data, moments, people, cam_xz, pose = E.setup(rec, name)
    lab = rec / f"labels_{name}"
    items = {it["clipT"]: it for it in json.loads((lab / "items.json").read_text())}
    seg0, start = (lambda it: (int(it["id"].split("_")[0][1:]), float(it["id"].split("_")[1])))(next(iter(items.values())))
    seconds = len(data["frames"]) / data["fps"]
    players = Wt.players_per_frame(rec, seg0, start, seconds, data["fps"], pose)
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(dev).eval()

    def view_of(p_xz):
        """Which phone sees this pitch point, its pixel and body height px."""
        yaw, pit = np.radians(pose.angles(np.asarray(p_xz)[None])[0])
        best = None
        for role, cam in cams.items():
            px, py, ok = cam.project(follow.direction(yaw, pit)[None])
            if not ok[0]:
                continue
            edge = min(px[0], cam.w - px[0], py[0], cam.h - py[0])
            if best is None or edge > best[0]:
                dist = np.hypot(*(np.asarray(p_xz) - cam_xz))
                best = (edge, role, float(px[0]), float(py[0]), BODY_M / dist * cam.f)
        return best

    results, rows, t0 = {}, [], time.time()
    n_people = n_faced = 0
    for t, k, status, xz in moments:
        it = items[t]
        imgs = {f["phone"]: cv2.imread(str(lab / f["image"])) for f in it["frames"]}
        origins, dirs, weights = [], [], []
        for p_xz in players[k]:
            v = view_of(p_xz)
            if v is None:
                continue
            _, role, px, py, body = v
            if body < MIN_BODY_PX:
                continue
            img = imgs[role]
            side = int(CROP_UP * body)
            x0, y0 = int(px - side / 2), int(py - 0.8 * side)
            x0, y0 = max(0, min(img.shape[1] - side, x0)), max(0, min(img.shape[0] - side, y0))
            crop = img[y0:y0 + side, x0:x0 + side]
            if crop.size == 0:
                continue
            scale = max(1.0, 400 / side)
            if scale > 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(crop[..., ::-1].copy()).permute(2, 0, 1).float().div(255).to(dev)
            with torch.no_grad():
                out = model([x])[0]
            n_people += 1
            if not len(out["keypoints"]):
                continue
            # The person nearest the crop centre.
            c = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
            boxes = out["boxes"].cpu().numpy()
            j = int(np.argmin([np.hypot(*((bx[:2] + bx[2:]) / 2 - c)) for bx in boxes]))
            kp = out["keypoints"][j].cpu().numpy()[:, :2] / scale
            ks = out["keypoints_scores"][j].cpu().numpy()
            ang = facing_from_keypoints(kp, ks, body)
            if ang is None:
                continue
            n_faced += 1
            # Ray on the pitch: 'towards the camera' is the direction player -> camera, rotated by ang.
            to_cam = cam_xz - np.asarray(p_xz)
            to_cam /= np.hypot(*to_cam)
            ca, sa = np.cos(ang), np.sin(ang)
            d = np.array([ca * to_cam[0] - sa * to_cam[1], sa * to_cam[0] + ca * to_cam[1]])
            origins.append(np.asarray(p_xz)); dirs.append(d); weights.append(float(out["scores"][j]))
        pt, keep = attention_point(origins, dirs, weights) if len(origins) >= 3 else (None, None)
        vp = vote_point(origins, dirs, weights, pose) if len(origins) >= 3 else None
        results[str(t)] = {"lsq": None if pt is None else [round(float(pt[0]), 2), round(float(pt[1]), 2)],
                           "vote": None if vp is None else [round(float(vp[0]), 2), round(float(vp[1]), 2)],
                           "rays": [[round(float(o[0]), 1), round(float(o[1]), 1), round(float(d[0]), 3), round(float(d[1]), 3), round(w, 2)]
                                    for o, d, w in zip(origins, dirs, weights)]}
        if pt is not None:
            pt = np.clip(pt, [-pose.L / 2, -pose.W / 2], [pose.L / 2, pose.W / 2])
        if status == "ball" and xz is not None:
            angs = [np.degrees(np.arccos(np.clip(np.dot(d, (xz - o) / max(np.hypot(*(xz - o)), 1e-6)), -1, 1)))
                    for o, d in zip(origins, dirs) if np.hypot(*(xz - o)) > 3]
            rows.append((t, None if pt is None else float(np.hypot(*(pt - xz))), len(origins),
                         int(keep.sum()) if keep is not None else 0,
                         None if vp is None else float(np.hypot(*(vp - xz))), angs))
    (rec / "detect" / f"facing_{name}.json").write_text(json.dumps(results))
    errs = [r[1] for r in rows if r[1] is not None]
    verrs = [r[4] for r in rows if r[4] is not None]
    all_angs = [a for r in rows for a in r[5]]
    print(f"{name}: {len(rows)} labelled moments; people cropped {n_people}, facing found for {n_faced}; {(time.time() - t0) / 60:.1f} min")
    print(f"  single rays: angle to the ball median {np.median(all_angs):.0f} deg, 25th {np.percentile(all_angs, 25):.0f},"
          f" within 30 deg {100 * np.mean(np.array(all_angs) < 30):.0f}% (n={len(all_angs)}; random would be 17%)")
    print(f"  least-squares point: distance to the ball median {np.median(errs):.1f} m, 25th {np.percentile(errs, 25):.1f};"
          f" within 8 m {sum(e <= 8 for e in errs)}/{len(errs)}")
    print(f"  vote point:          distance to the ball median {np.median(verrs):.1f} m, 25th {np.percentile(verrs, 25):.1f};"
          f" within 8 m {sum(e <= 8 for e in verrs)}/{len(verrs)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
