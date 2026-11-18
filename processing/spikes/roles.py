#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "scikit-learn>=1.4", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11", "coremltools>=8,<10; sys_platform == 'darwin'"]
# ///
"""People along a clip with a role: side A, side B, keeper, referee, other.

    uv run processing/spikes/roles.py footage/rec3 <segment> <start s> <seconds> <teams set> [fps]

The stock RF-DETR finds people on both phones' frames at half
resolution (people are big enough; the ball isn't the point here), every
1/fps s. Each person's torso colour is assigned to the kit mixture of
teams.py's <teams set> (model.npz, sides.json): a kit component gives
side A or B; a keeper/referee component gives "keeper" when the person
stands in a goal area (within KEEPER_X_M of a goal line, KEEPER_Z_M of
its centre line) and "referee" otherwise; everything else is "other".

Writes <recording>/detect/roles_<segment>_<start>_<seconds>.json: per
frame, people as {xz, role, conf}. follow.py's "roles" feature reads it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import detect  # noqa: E402
import follow  # noqa: E402
import gpu  # noqa: E402
import phase0  # noqa: E402
import pitch as P  # noqa: E402
import teams  # noqa: E402

HALF = 1920
KEEPER_X_M = 22.0
KEEPER_Z_M = 25.0
FAR_MIN_PX = 12            # near a goal, people this tall still get a colour (keepers are far and small) ...
FAR_SAT = 30.0             # ... and a saturated torso there (not a kit colour) is the keeper
BATCH = int(os.environ.get("OSB_ROLES_BATCH", "4"))  # moments (a frame from each phone) per detector call on an NVIDIA GPU


class OnGpu:
    """A picture held on the GPU that reads like the array (H x W x 3
    uint8): indexing brings just that part to the CPU, so teams.features
    takes its torso and turf patches without the whole 4K frame coming down."""

    def __init__(self, t: torch.Tensor):
        self.t, self.shape = t, tuple(t.shape)

    def __getitem__(self, idx) -> np.ndarray:
        return self.t[idx].cpu().numpy()
# OSB_PEOPLE_COREML: Core ML for the people detector on a Mac unless "0" (detect.coreml_detector)
# OSB_PEOPLE_PRECISION=float16: a half-precision Core ML export for the people detector. Off unless asked
# for: ~11% less detector time on a Mac, but the export's outputs change from call to call on the same
# input, so two runs disagree on ~9% of people (FINDINGS.md, 2026-11-14). fp32 is the same every time.
PRECISION = os.environ.get("OSB_PEOPLE_PRECISION", "float32")


def main(rec: Path, seg: int, start: float, seconds: float, tset: str, fps: float) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    pose = P.load_pose(rec)
    # No goals.json (none detected): the goal-line centres, as in follow.py.
    goals = {g["side"]: np.array([g["x"], g["z"]]) for g in P.load_goals(rec) or []}
    hl, hw = pose.L / 2 + 1, pose.W / 2 + 1
    tdir = rec / "teams" / tset
    m = np.load(tdir / "model.npz")
    sides = json.loads((tdir / "sides.json").read_text())
    means, covs, weights, mu, sd, rank = m["means"], m["covs"], m["weights"], m["mu"], m["sd"], m["rank"]
    role_of_comp = {}
    for c in range(len(rank)):
        r = int(rank[c])
        role_of_comp[c] = "A" if r in sides["A"] else "B" if r in sides["B"] else "solo" if r in sides["keeperOrReferee"] else "other"

    def assign(f):
        z = (f - mu) / sd
        ll = -0.5 * (((z - means) ** 2) / covs).sum(1) - 0.5 * np.log(covs).sum(1) + np.log(weights)
        return int(np.argmax(ll))

    model = rfdetr.RFDETRMedium()
    model = detect.coreml_detector(model, "OSB_PEOPLE_COREML", rfdetr.RFDETRMedium, batches=(8,), precision=PRECISION)
    model = detect.half_precision(detect.tensorrt(model, "people", rfdetr.RFDETRMedium), "people")
    net = detect.NetworkClock(model)
    dev = gpu.device()
    W, H = follow.SRC_W, follow.SRC_H
    streams = [follow.prefetch(follow.game_stream(rec, devs, report, ph, seg, start, seconds, W, fps=fps))
               for ph in (0, 1)]
    frames_out, t0, t_det, sw = [], time.time(), 0.0, follow.Stopwatch()
    # Where Secondary has no footage (it stopped early) its frames are blank: Main's people alone.
    missing = phase0.secondary_missing(manifest, report, phase0.game_start(manifest, seg) + start,
                                       int(round(seconds * fps)) + 1, fps)
    from collections import Counter
    tally = Counter()

    on_gpu = dev.type == "cuda"
    batch_of = BATCH if on_gpu else 1

    def prepare(buf, fl: bool):
        """A phone's NV12 frame -> its BGR picture and the half-size RGB
        picture the detector sees. On an NVIDIA GPU both are made and kept
        there, with OpenCV's own arithmetic (detect.nv12_to_bgr_cv2: the
        kit model's colours were fitted on OpenCV's), so the frame never
        comes to the CPU. Elsewhere OpenCV makes them, which is quicker
        than going through the GPU when the detector wants arrays (Core ML)."""
        if not on_gpu:
            bgr = cv2.cvtColor(np.asarray(buf).reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_NV12)
            if fl:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            small = cv2.resize(bgr, (HALF, HALF * 9 // 16), interpolation=cv2.INTER_AREA)
            return bgr, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if isinstance(buf, gpu.GpuNV12):
            y, uv = buf.y.to(dev), buf.uv.to(dev)
        else:
            flat = torch.from_numpy(np.asarray(buf)).to(dev)
            y, uv = flat[:W * H].view(H, W), flat[W * H:].view(H // 2, W // 2, 2)
        bgr = detect.nv12_to_bgr_cv2(y, uv)
        if fl:
            bgr = bgr.flip(0, 1)
        return bgr, detect.halve_area(bgr).flip(-1).permute(2, 0, 1).float() / 255

    def people_of(i: int, pictures: dict, boxes_of: dict) -> list:
        people = []
        for phone, bgr_gpu in pictures.items():
            bgr = OnGpu(bgr_gpu) if torch.is_tensor(bgr_gpu) else bgr_gpu
            band = bgr[760::16, ::16].reshape(-1, 3)
            hsv = cv2.cvtColor(band.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            g = band[(hsv[:, 0] > 30) & (hsv[:, 0] < 95) & (hsv[:, 1] > 40)]
            turf = g.mean(0) if len(g) else np.array([60, 120, 60], float)
            for b in boxes_of[phone]:
                box = b[:4] * 2  # back to full resolution
                d = cams[phone].unproject(np.array([[(box[0] + box[2]) / 2, box[3]]], float))[0]
                x, zz = pose.to_pitch(np.degrees(np.arctan2(d[0], d[2])), np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2]))))
                if np.isnan(x) or abs(x) > hl or abs(zz) > hw:
                    continue
                side = 1 if x > 0 else -1
                gx, gz = goals.get(side, (side * pose.L / 2, 0.0))
                in_area = side * (gx - x) <= KEEPER_X_M and abs(zz - gz) <= KEEPER_Z_M
                saved = teams.MIN_HEIGHT_PX
                if in_area:
                    teams.MIN_HEIGHT_PX = FAR_MIN_PX
                r = teams.features(bgr, box, turf)
                teams.MIN_HEIGHT_PX = saved
                role = "other"
                if r is not None:
                    comp = assign(r[0])
                    role = role_of_comp[comp]
                    if role == "solo":
                        role = "keeper" if in_area else "referee"
                    elif in_area and role == "other" and r[0][2] >= FAR_SAT:
                        role = "keeper"  # saturated but not a kit colour, in a goal area: the keeper
                people.append({"xz": [round(float(x), 2), round(float(zz), 2)], "role": role, "phone": phone,
                               "conf": round(float(b[4]), 3), "h": int(box[3] - box[1])})
                tally[role] += 1
        return people

    def flush(batch: list) -> None:
        """The detector once for every picture of `batch` ([(frame number, {phone: (bgr, rgb)})]), then each frame's people."""
        nonlocal t_det
        keys = [(i, phone) for i, pics in batch for phone in pics]
        td = time.time()
        found = teams.person_boxes_many(model, [pics[phone][1] for i, pics in batch for phone in pics],
                                        band_top=detect.BAND_TOP // 2)
        t_det += time.time() - td
        sw.lap("detector")
        boxes = dict(zip(keys, found))
        for i, pics in batch:
            people = people_of(i, {ph: bgr for ph, (bgr, _) in pics.items()}, {ph: boxes[(i, ph)] for ph in pics})
            # The same person seen by both phones: keep one (nearest pair within 1.5 m, prefer the taller box).
            merged = []
            for p in sorted(people, key=lambda p: -p["h"]):
                if not any(q["phone"] != p["phone"] and np.hypot(*(np.subtract(q["xz"], p["xz"]))) < 1.5 for q in merged):
                    merged.append(p)
            frames_out.append({"t": round(start + i / fps, 3), "people": merged})
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{int(seconds * fps)} frames, {(time.time() - t0) / 60:.1f} min, roles so far {dict(tally)}", flush=True)
        sw.lap("roles")

    batch = []
    waited = [detect.Waited(st) for st in streams]
    for i, (fa, fb) in enumerate(zip(*waited)):
        sw.lap("frames")
        pics = {}
        for phone, (buf, fl) in enumerate(((fa, flip[0]), (fb, flip[1]))):
            if phone == 1 and missing[min(i, len(missing) - 1)]:
                continue
            pics[phone] = prepare(buf, fl)
        batch.append((i, pics))
        sw.lap("prepare")
        if len(batch) >= batch_of:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    path = rec / "detect" / f"roles_{seg}_{start:g}_{seconds:g}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"segment": seg, "fps": fps, "teams": tset, "frames": frames_out}))
    print(f"{len(frames_out)} frames in {(time.time() - t0) / 60:.1f} min; roles {dict(tally)} -> {path.name}; "
          f"detector {t_det / 60:.1f} min ({f'Core ML {PRECISION}' if isinstance(model, detect.CoreMLDetector) else 'PyTorch'})", flush=True)
    print("time by stage (frames: waiting for the decoder; on a GPU, prepare is only queued there and finishes within detector):\n"
          + sw.report({}) + ("\n  " + net.report(t_det) if net.seconds is not None else ""), flush=True)
    os._exit(0)  # the output is written; teardown has hung (decoder threads)


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), int(a[2]), float(a[3]), float(a[4]), a[5], float(a[6]) if len(a) > 6 else 10.0)
