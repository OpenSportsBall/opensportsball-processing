#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11", "pyyaml>=6"]
# ///
"""Ball candidates for a clip with the detector run in search windows.

    uv run processing/spikes/ballwin.py footage/rec3 <segment> <start s> <seconds> <checkpoint.pth> [fps]

ballclip.py scans ~28 tiles of 640 px per 4K frame. This keeps a simple
online track of the ball on the pitch (last accepted detection plus its
velocity) and, in each phone's frame, runs the detector only on a window
around the predicted position: one native 640 px window, plus a 2x zoom
window (a 320 px crop upscaled) when the ball is expected to be small.
A full scan runs every FULL_EVERY_S, or once the ball has been lost for
LOST_S. Every candidate is scored by balljudge.py's crop judge inline.

OSB_WIN_WASB=1 adds WASB (ballheat.py's multi-frame heatmap model) as a
second source: the clip is decoded at 3x the rate so each frame has its
neighbours, and WASB runs in fp16 on 512x288 tiles of the frame downscaled
2x and 4x (WASB_SCALES; the far balls come from these) in every full scan,
and otherwise on one window around the prediction (2x when the ball is
expected small, else 4x). Heatmap peaks become candidates with the peak
as their confidence, unless an RF-DETR box already holds the same ball.

Each phone's full scan is planned from the pitch map at the start (and
printed): the fewest 640 px tiles that cover every on-pitch pixel where the
ball would be under SCAN_HALF_PX, then the fewest 1280 px crops shrunk to
640 that cover the rest of the pitch (the near side, where the ball is
large). Tiles that see no pitch are dropped. On the must-see bench this
cut the ball stage 22% and captured 34/46 moments against 33 for the plain
grid (docs/BACKLOG.md, 2026-11-07). OSB_SCAN_PLAN=grid tiles the whole band
below BAND_TOP at full resolution instead.

OSB_LOST_LOCAL=1 makes the rescans while the ball is lost (every
LOST_SCAN_S) local: only the scan crops that see pitch within
LOCAL_R0_M + LOCAL_V_MS x the time lost of the last sighting. The full
scan every FULL_EVERY_S stays whole, so a ball kicked far is still found
within that.

OSB_FAR_ZOOM=lost (or full) adds a 2x pass over the far band (the crops
where the ball would be under ZOOM_BELOW_PX) to the scans while the ball is
lost (or to every full scan). One more labelled ball in 90 for ~30% more
time (docs/BACKLOG.md, 2026-10-28), so off by default.

Writes <recording>/detect/ball_<segment>_<start>_<seconds>.win.json in
ballclip.py's format (plus "src": scan / win / zoom / wasb / wasbwin per
candidate) and prints the tiles per frame against a full scan.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import balljudge  # noqa: E402
import detect  # noqa: E402
import follow  # noqa: E402
import phase0  # noqa: E402
import gpu  # noqa: E402
import pitch as P  # noqa: E402

FLOOR = 0.1
MERGE_DEG = 0.4         # the same ball seen by both phones
FULL_EVERY_S = float(os.environ.get("OSB_FULL_EVERY_S", 2.0))   # a full scan this often regardless
LOST_S = 1.0            # ... and once the ball has been lost this long, every LOST_SCAN_S
LOST_SCAN_S = float(os.environ.get("OSB_LOST_SCAN_S", 0.5))
ZOOM_BELOW_PX = 10.0    # 2x zoom window when the expected ball diameter is under this
ACCEPT_CONF = 0.3       # a detection at least this sure, and judged at least ACCEPT_JUDGE, updates the track ...
ACCEPT_JUDGE = 0.3
ACCEPT_JUMP_CONF = 0.6
GATE_M = 3.0            # ... if within this + MAX_SPEED x dt of the prediction
MAX_SPEED = 35.0
V_MAX = 20.0            # the online track's velocity is capped at this (a false jump gave it 33 m/s once) ...
V_DECAY = 0.85          # ... and shrinks by this per frame while the ball is unseen
JUMP_MAX_M = 15.0       # a candidate further than this from the last sighting needs ACCEPT_JUMP_CONF
WIN = 640
ZOOM_CROP = 320
FAR_ZOOM = os.environ.get("OSB_FAR_ZOOM", "")  # "lost": scans while the ball is lost add 2x tiles of the far band
                                              # (where the ball would be under ZOOM_BELOW_PX); "full": every scan
SCAN_PLAN = os.environ.get("OSB_SCAN_PLAN", "geom")  # "geom": full scans planned from the pitch map; "grid": all tiles
SCAN_HALF_PX = float(os.environ.get("OSB_SCAN_HALF_PX", 20.0))  # a ball at least this wide is scanned at half size
SCAN_SAMPLE = 40        # the plan samples the frame this many px apart
LOST_LOCAL = bool(os.environ.get("OSB_LOST_LOCAL"))  # lost-ball rescans only near the last sighting (see above)
LOCAL_R0_M = float(os.environ.get("OSB_LOCAL_R0_M", 10.0))   # their radius on the pitch: this ...
LOCAL_V_MS = float(os.environ.get("OSB_LOCAL_V_MS", 10.0))   # ... plus this x the seconds lost
FAR_STEP = 280          # far-band crops (ZOOM_CROP) start this far apart (40 px overlap)
WASB = bool(os.environ.get("OSB_WIN_WASB"))
WASB_STEER = bool(os.environ.get("OSB_WASB_STEER"))  # WASB's candidates also update the online track (moves RF-DETR's windows)
WASB_EVERY = bool(os.environ.get("OSB_WASB_EVERY"))  # WASB's full scan on every frame, not only with RF-DETR's
WASB_CKPT = os.environ.get("OSB_WASB_CKPT")  # a heattrain.py model (with the background input) instead of the stock one
WASB_SCALES = (2, 4)    # WASB's full scans: 512x288 tiles of the frame downscaled this much
WASB_MIN = float(os.environ.get("OSB_WASB_MIN", 0.3 if WASB_CKPT else 0.05))  # heatmap peaks above this are
                        # candidates (stock: ballheat.py's PEAK_MIN; a fine-tuned model is surer) ...
WASB_TOP = 30           # ... at most this many per phone and frame, the strongest
WASB_SAME_DIAM = 1.5    # a peak within this many ball diameters (or 8 px) of an RF-DETR box is that ball


class Wasb:
    """WASB in fp16 on the GPU: three consecutive frames stacked as
    channels in, the middle frame's heatmap out, per 512x288 tile of a
    downscaled frame. With a heattrain.py checkpoint the model also takes
    the empty pitch (the segment's background) as a fourth frame."""

    def __init__(self, dev: torch.device, ckpt: Path | None = None):
        import ballheat

        self.bg = ckpt is not None
        model = ballheat.wasb_with_bg(ckpt) if self.bg else ballheat.Heat("wasb", dev).model
        self.model = model.to(dev).eval().half()
        self.tile_grid, self.tw, self.th = ballheat.tile_grid, ballheat.IN_W, ballheat.IN_H
        self.mean = torch.tensor(ballheat.MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(ballheat.STD, device=dev).view(1, 3, 1, 1)
        ys, xs = torch.meshgrid(torch.arange(self.th, device=dev, dtype=torch.float32),
                                torch.arange(self.tw, device=dev, dtype=torch.float32), indexing="ij")
        self.xs, self.ys = xs, ys
        self.tiles = 0

    def scan_corners(self, h: int, w: int) -> list[tuple[int, int, int]]:
        return [(x, y, s) for s in WASB_SCALES for x, y in self.tile_grid(h, w, s)]

    def window_corner(self, h: int, w: int, px: tuple[float, float], s: int) -> tuple[int, int, int]:
        x0 = int(np.clip(px[0] - self.tw * s / 2, 0, w - self.tw * s)) // s * s
        y0 = int(np.clip(px[1] - self.th * s / 2, 0, h - self.th * s)) // s * s
        return x0, y0, s

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, corners: list[tuple[int, int, int]], bg: torch.Tensor | None = None,
                 base: int = 1) -> list[list[float]]:
        """frames (3, 3, H, W) RGB in [0, 1] (previous, this, next), already
        downscaled `base` times from the 4K frame, and for a background
        model bg (3, H', W') at 2x downscale -> peaks [x, y, score, scale]
        in 4K frame pixels, merged, strongest first. Corners are (x, y,
        scale) in 4K pixels."""
        x = (frames - self.mean) / self.std
        scales = {c[2] for c in corners}
        pyr = {s: F.avg_pool2d(x, s // base) if s > base else x for s in scales}
        if self.bg:
            b = (bg[None] - self.mean) / self.std
            pyr = {s: torch.cat([F.avg_pool2d(b, s // 2) if s > 2 else b, pyr[s]]) for s in scales}
        n_in = 12 if self.bg else 9
        batch = torch.stack([pyr[s][:, :, y // s:y // s + self.th, x0 // s:x0 // s + self.tw].reshape(n_in, self.th, self.tw)
                             for x0, y, s in corners]).half()
        hm = torch.sigmoid(self.model(batch)[0][:, 1].float())
        self.tiles += len(corners)
        top = F.max_pool2d(hm[:, None], 5, 1, 2)[:, 0]
        w3 = F.avg_pool2d(hm[:, None], 3, 1, 1)[:, 0]
        cx = F.avg_pool2d((hm * self.xs)[:, None], 3, 1, 1)[:, 0] / w3.clamp_min(1e-6)
        cy = F.avg_pool2d((hm * self.ys)[:, None], 3, 1, 1)[:, 0] / w3.clamp_min(1e-6)
        k, iy, ix = ((hm == top) & (hm > WASB_MIN)).nonzero(as_tuple=True)
        got = torch.stack([k.float(), cx[k, iy, ix], cy[k, iy, ix], hm[k, iy, ix]], 1).cpu().numpy()
        cands = []
        for t, u, v, score in got:
            x0, y0, s = corners[int(t)]
            cands.append([x0 + u * s + (s - 1) / 2, y0 + v * s + (s - 1) / 2, float(score), s])
        kept = []
        for c in sorted(cands, key=lambda c: -c[2]):
            if not any(np.hypot(c[0] - q[0], c[1] - q[1]) < 8 * max(c[3], q[3]) for q in kept):
                kept.append(c)
            if len(kept) >= WASB_TOP:
                break
        return kept


def merge_phones(found: list[dict]) -> list[dict]:
    """One candidate per ball seen by both phones: the surer one."""
    merged = []
    for d in sorted(found, key=lambda d: -d["conf"] * d["judge"]):
        if not any(m["phone"] != d["phone"] and np.hypot(m["yaw"] - d["yaw"], m["pitch"] - d["pitch"]) < MERGE_DEG
                   for m in merged):
            merged.append(d)
    return merged


def triplets(stream):
    """A frame stream at 3x the clip's rate -> (previous, this, next) around
    every third frame (the clip's own frames; the first has no previous
    and repeats itself)."""
    prev = None
    while True:
        try:
            a, b, c = next(stream), next(stream), next(stream)
        except StopIteration:
            return
        yield (a if prev is None else prev, a, b)
        prev = c


def main(rec: Path, seg: int, start: float, seconds: float, ckpt: Path, fps: float) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    roles = ("main", "secondary")
    pose = P.load_pose(rec)
    hl, hw = pose.L / 2 + P.ON_PITCH_MARGIN, pose.W / 2 + P.ON_PITCH_MARGIN
    diam_at = balljudge.expected_at(rec)
    judge = balljudge.load_judge(rec, os.environ.get("OSB_JUDGE") or None)  # a <rec>/balljudge/<name>.pt

    dev = gpu.device()
    model = rfdetr.RFDETRMedium(pretrain_weights=str(ckpt))
    W, H = follow.SRC_W, follow.SRC_H
    wasb = Wasb(dev, Path(WASB_CKPT) if WASB_CKPT else None) if WASB else None
    bgs: dict = {}

    def background(phone: int, t: float) -> torch.Tensor | None:
        """The background of the segment Main is in at clip time t (heattrain.py's, on the GPU)."""
        if not (wasb and wasb.bg):
            return None
        k, local = seg, start + t
        while k + 1 < len(devs["main"]["segments"]) and local >= devs["main"]["segments"][k]["durationMs"] / 1000:
            local -= devs["main"]["segments"][k]["durationMs"] / 1000
            k += 1
        if (phone, k) not in bgs:
            import heattrain
            bgs[(phone, k)] = torch.from_numpy(heattrain.load_bg(rec, roles[phone], k)).to(dev).permute(2, 0, 1).float() / 255
        return bgs[(phone, k)]
    if wasb:
        streams = [triplets(follow.prefetch(follow.game_stream(rec, devs, report, ph, seg, start, seconds, W,
                                                               fps=3 * fps), depth=9)) for ph in (0, 1)]
    else:
        streams = [follow.prefetch(follow.game_stream(rec, devs, report, ph, seg, start, seconds, W, fps=fps))
                   for ph in (0, 1)]
    t_wasb = 0.0

    def to_pixel(phone, xz):
        """Predicted pitch position -> this phone's pixel, or None if out of frame."""
        yaw, pit = np.radians(pose.angles(np.asarray(xz)[None])[0])
        x, y, ok = cams[phone].project(follow.direction(yaw, pit)[None])
        return (float(x[0]), float(y[0])) if ok[0] and 0 <= x[0] < W and 0 <= y[0] < H else None

    def run(img, boxes_xy, scale=1.0):
        """Detector on crops of `img` (CHW float) at the given top-left
        corners; returns rows [x1, y1, x2, y2, conf] in frame pixels."""
        crops = []
        for x, y in boxes_xy:
            side = int(WIN / scale)
            c = img[:, y:y + side, x:x + side]
            if scale != 1.0:
                c = F.interpolate(c[None], size=(WIN, WIN), mode="bilinear", align_corners=False)[0]
            crops.append(c)
        res = model.predict(crops, threshold=FLOOR)
        rows = []
        for (x, y), det in zip(boxes_xy, res):
            for box, sc in zip(det.xyxy, det.confidence):
                b = np.array(box) / scale + [x, y, x, y]
                rows.append(np.r_[b, sc])
        return rows

    def far_band(phone):
        """Corners of ZOOM_CROP crops covering the on-pitch pixels where the
        ball would be under ZOOM_BELOW_PX (the far half of the pitch)."""
        ys = list(range(detect.BAND_TOP, H - ZOOM_CROP, FAR_STEP)) + [H - ZOOM_CROP]
        xs = list(range(0, W - ZOOM_CROP, FAR_STEP)) + [W - ZOOM_CROP]
        out = []
        for y0 in ys:
            for x0 in xs:
                for sx in range(x0 + 20, x0 + ZOOM_CROP, 70):
                    for sy in range(y0 + 20, y0 + ZOOM_CROP, 70):
                        yaw, pit = detect.to_angles(cams[phone], np.array([[sx, sy]], float))[0]
                        gx, gz = pose.to_pitch(yaw, pit)
                        if (not np.isnan(gx) and abs(gx) <= hl and abs(gz) <= hw
                                and diam_at(roles[phone], sx, sy) < ZOOM_BELOW_PX):
                            out.append((x0, y0))
                            break
                    else:
                        continue
                    break
        return out
    far = [far_band(ph) for ph in (0, 1)] if FAR_ZOOM else [[], []]

    def scan_plan(phone):
        """[(corners, scale)] for this phone's full scan. geom: full-size
        tiles covering the on-pitch pixels where the ball would be under
        SCAN_HALF_PX, half-size crops (1280 px shrunk to WIN) covering the
        rest of the pitch. grid: every detect.tiles tile at full size."""
        full_tiles = detect.tiles(H, W)
        if SCAN_PLAN == "grid":
            return [(full_tiles, 1.0)]
        small, large = [], []   # on-pitch sample pixels by expected ball size
        for sy in range(detect.BAND_TOP + SCAN_SAMPLE // 2, H, SCAN_SAMPLE):
            for sx in range(SCAN_SAMPLE // 2, W, SCAN_SAMPLE):
                yaw, pit = detect.to_angles(cams[phone], np.array([[sx, sy]], float))[0]
                gx, gz = pose.to_pitch(yaw, pit)
                if np.isnan(gx) or abs(gx) > hl or abs(gz) > hw:
                    continue
                (small if diam_at(roles[phone], sx, sy) < SCAN_HALF_PX else large).append((sx, sy))
        small, large = np.array(small).reshape(-1, 2), np.array(large).reshape(-1, 2)
        if not len(small) and not len(large):
            print(f"scan plan {roles[phone]}: no pitch in view, scanning the whole grid", flush=True)
            return [(full_tiles, 1.0)]
        n_small, n_large = len(small), len(large)

        def inside(pts, x, y, side):
            return (pts[:, 0] >= x) & (pts[:, 0] < x + side) & (pts[:, 1] >= y) & (pts[:, 1] < y + side)

        def cover(pts, corners, side):
            """Greedy: the fewest of `corners` (squares of `side`) holding every point."""
            left, out = np.ones(len(pts), bool), []
            while left.any():
                gain = [int((inside(pts, x, y, side) & left).sum()) for x, y in corners]
                k = int(np.argmax(gain))
                if gain[k] == 0:
                    break
                out.append(corners[k])
                left &= ~inside(pts, *corners[k], side)
            return out
        # a ball within ~half its size of a crop edge is cut; the tiles overlap 64 px (128 half size)
        keep = sorted(cover(small, full_tiles, WIN), key=lambda c: (c[1], c[0]))
        if len(large) and keep:
            covered = np.zeros(len(large), bool)
            for x, y in keep:
                covered |= inside(large, x + 32, y + 32, WIN - 64)
            large = large[~covered]
        half = []
        if len(large):
            side, step = 2 * WIN, 2 * WIN - 128
            ys = sorted(set(range(detect.BAND_TOP, H - side, step // 4)) | {H - side})
            xs = sorted(set(range(0, W - side, step // 4)) | {W - side})
            half = sorted(cover(large, [(x, y) for y in ys for x in xs], side), key=lambda c: (c[1], c[0]))
        if len(keep) + len(half) >= len(full_tiles):
            keep, half = full_tiles, []
        plan = [(keep, 1.0), (half, 0.5)]
        print(f"scan plan {roles[phone]}: {len(keep)} full-size tiles of {len(full_tiles)}, {len(half)} half-size"
              f" (ball >= {SCAN_HALF_PX:.0f} px); pitch samples {n_small} small-ball, {n_large} large-ball", flush=True)
        return plan
    plans = [scan_plan(ph) for ph in (0, 1)]

    def footprints(phone):
        """Per plan crop: (corners, scale, pitch xz of the on-pitch sample pixels inside it)."""
        pix, gxz = [], []
        for sy in range(detect.BAND_TOP + SCAN_SAMPLE // 2, H, SCAN_SAMPLE):
            for sx in range(SCAN_SAMPLE // 2, W, SCAN_SAMPLE):
                yaw, pit = detect.to_angles(cams[phone], np.array([[sx, sy]], float))[0]
                gx, gz = pose.to_pitch(yaw, pit)
                if not np.isnan(gx) and abs(gx) <= hl and abs(gz) <= hw:
                    pix.append((sx, sy)); gxz.append((gx, gz))
        pix, gxz = np.array(pix).reshape(-1, 2), np.array(gxz).reshape(-1, 2)
        out = []
        for corners, sc in plans[phone]:
            side = WIN / sc
            for x, y in corners:
                m = (pix[:, 0] >= x) & (pix[:, 0] < x + side) & (pix[:, 1] >= y) & (pix[:, 1] < y + side)
                out.append(((x, y), sc, gxz[m]))
        return out
    feet = [footprints(ph) for ph in (0, 1)] if LOST_LOCAL else None

    def local_plan(phone, centre, r):
        """The plan's crops that see pitch within r m of centre, as [(corners, scale)]."""
        by_scale: dict = {}
        for corner, sc, g in feet[phone]:
            if len(g) and np.min(np.hypot(g[:, 0] - centre[0], g[:, 1] - centre[1])) <= r:
                by_scale.setdefault(sc, []).append(corner)
        return [(c, sc) for sc, c in by_scale.items()]
    if FAR_ZOOM:
        print(f"far band: {len(far[0])} + {len(far[1])} crops at 2x ({FAR_ZOOM} scans)", flush=True)

    frames_out, t0 = [], time.time()
    ball_xz, ball_v, last_seen, last_full = None, np.zeros(2), -10 ** 9, -10 ** 9
    last_whole = -10 ** 9   # the last whole (not local) scan
    tiles_total, n_full, n_local = 0, 0, 0
    for i, (fa, fb) in enumerate(zip(*streams)):
        t = i / fps
        lost = (i - last_seen) / fps
        since_full = (i - last_full) / fps
        full = since_full >= FULL_EVERY_S or ((ball_xz is None or lost >= LOST_S) and since_full >= LOST_SCAN_S)
        local = (LOST_LOCAL and full and ball_xz is not None and (i - last_whole) / fps < FULL_EVERY_S)
        if ball_xz is not None:
            pred = ball_xz + ball_v * min((i - last_seen) / fps, LOST_S)  # beyond LOST_S: stay near the last sighting
            pred = np.clip(pred, [-pose.L / 2 - 3, -pose.W / 2 - 3], [pose.L / 2 + 3, pose.W / 2 + 3])
        else:
            pred = None
        found = []
        for phone, (buf, fl) in enumerate(((fa, flip[0]), (fb, flip[1]))):
            bufs = buf if wasb else (buf,)
            buf = bufs[1] if wasb else buf

            def rgb(b):
                im = detect.nv12_to_rgb(*gpu.upload_nv12(b, W, H, dev))
                return im.flip(1, 2) if fl else im

            img = rgb(buf)
            rows = []
            if full:
                for tl, sc in (local_plan(phone, ball_xz, LOCAL_R0_M + LOCAL_V_MS * lost) if local else plans[phone]):
                    if tl:
                        rows += [(r, "scan") for r in run(img, tl, scale=sc)]
                        tiles_total += len(tl)
                if FAR_ZOOM == "full" or (FAR_ZOOM == "lost" and (ball_xz is None or lost >= LOST_S)):
                    rows += [(r, "far") for r in run(img, far[phone], scale=2.0)]
                    tiles_total += len(far[phone])
            px = to_pixel(phone, pred) if pred is not None else None
            peaks = []
            if wasb and (full or WASB_EVERY or px is not None):
                tw0 = time.time()
                if full or WASB_EVERY:
                    corners, wsrc = wasb.scan_corners(H, W), "wasb"
                else:
                    corners = [wasb.window_corner(H, W, px, 2 if diam_at(roles[phone], *px) < ZOOM_BELOW_PX else 4)]
                    wsrc = "wasbwin"
                peaks = [(c, wsrc) for c in wasb(torch.stack([rgb(bufs[0]), img, rgb(bufs[2])]), corners,
                                                  background(phone, t))]
                t_wasb += time.time() - tw0
            if px is not None:
                x0 = int(np.clip(px[0] - WIN / 2, 0, W - WIN)); y0 = int(np.clip(px[1] - WIN / 2, 0, H - WIN))
                rows += [(r, "win") for r in run(img, [(x0, y0)])]
                tiles_total += 1
                if diam_at(roles[phone], *px) < ZOOM_BELOW_PX:
                    x0 = int(np.clip(px[0] - ZOOM_CROP / 2, 0, W - ZOOM_CROP))
                    y0 = int(np.clip(px[1] - ZOOM_CROP / 2, 0, H - ZOOM_CROP))
                    rows += [(r, "zoom") for r in run(img, [(x0, y0)], scale=2.0)]
                    tiles_total += 1
            if not rows and not peaks:
                continue
            kept = detect.nms(np.array([r for r, _ in rows])) if rows else np.zeros((0, 5))
            src_of = {tuple(np.round(r[:4])): s for r, s in rows}
            extra = []
            for (x, y, score, _), s in peaks:
                d = diam_at(roles[phone], x, y)
                if any(np.hypot(x - (x1 + x2) / 2, y - (y1 + y2) / 2) < max(8.0, WASB_SAME_DIAM * d)
                       for x1, y1, x2, y2, _ in kept):
                    continue
                box = np.array([x - d / 2, y - d / 2, x + d / 2, y + d / 2, score])
                extra.append(box)
                src_of[tuple(np.round(box[:4]))] = s
            if extra:
                kept = np.vstack([kept, extra])
            if not len(kept):
                continue
            bgr = (img.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
            crops = np.stack([balljudge.crop_at(bgr, (x1 + x2) / 2, (y1 + y2) / 2,
                                                diam_at(roles[phone], (x1 + x2) / 2, (y1 + y2) / 2))
                              for x1, y1, x2, y2, _ in kept])
            scores = judge(crops)
            for (x1, y1, x2, y2, sc), js in zip(kept, scores):
                yaw, pit = detect.to_angles(cams[phone], np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
                x, z = pose.to_pitch(yaw, pit)
                xz = None if np.isnan(x) else [round(float(x), 2), round(float(z), 2)]
                found.append({"cls": "ball", "phone": phone, "yaw": round(float(yaw), 3),
                              "pitch": round(float(pit), 3), "conf": round(float(sc), 3),
                              "judge": round(float(js), 3), "xz": xz,
                              "onPitch": bool(xz is not None and abs(x) <= hl and abs(z) <= hw),
                              "box": [round(float(v)) for v in (x1, y1, x2, y2)],
                              "src": src_of.get(tuple(np.round([x1, y1, x2, y2])), "scan")})
        if full:
            last_full, n_full, n_local = i, n_full + 1, n_local + local
            if not local:
                last_whole = i
        merged = merge_phones(found)
        # Online track: the surest on-pitch candidate near the prediction (or anywhere after a scan).
        # WASB's candidates don't steer it unless OSB_WASB_STEER: RF-DETR's windows go where they would without WASB.
        best = None
        for d in merged if WASB_STEER else merge_phones([d for d in found if not d["src"].startswith("wasb")]):
            if not d["onPitch"] or d["conf"] < ACCEPT_CONF or d["judge"] < ACCEPT_JUDGE:
                continue
            if pred is not None:
                gate = GATE_M + MAX_SPEED * (i - last_seen) / fps
                if np.hypot(*(np.array(d["xz"]) - pred)) > gate:
                    continue
                if np.hypot(*(np.array(d["xz"]) - ball_xz)) > JUMP_MAX_M and d["conf"] < ACCEPT_JUMP_CONF:
                    continue  # a big jump needs a sure detection
            if best is None or d["conf"] * d["judge"] > best["conf"] * best["judge"]:
                best = d
        if os.environ.get("OSB_WIN_DEBUG") and (best is not None or any(d["onPitch"] and d["conf"] >= ACCEPT_CONF for d in merged)):
            print(f"t={start + t:.1f} pred={None if pred is None else np.round(pred, 1)} lost={lost:.1f}"
                  f" gate={None if pred is None else round(GATE_M + MAX_SPEED * (i - last_seen) / fps, 1)}"
                  f" cands={[(round(d['xz'][0]), round(d['xz'][1]), d['conf'], d['judge']) for d in merged if d['onPitch'] and d['conf'] >= ACCEPT_CONF]}"
                  f" accepted={None if best is None else best['xz']}", flush=True)
        if best is not None:
            xz = np.array(best["xz"])
            if ball_xz is not None and i > last_seen:
                v = (xz - ball_xz) * fps / (i - last_seen)
                ball_v = 0.5 * ball_v + 0.5 * v
                n_v = np.hypot(*ball_v)
                if n_v > V_MAX:
                    ball_v *= V_MAX / n_v
            ball_xz, last_seen = xz, i
        else:
            ball_v = ball_v * V_DECAY
        frames_out.append({"t": round(start + t, 3), "detections": merged})
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{int(seconds * fps)} frames, {el / 60:.1f} min, {tiles_total / (i + 1):.1f} tiles/frame"
                  f" ({n_full} full scans)"
                  + (f"; WASB {wasb.tiles / (i + 1):.1f} tiles/frame, {t_wasb / 60:.1f} min" if wasb else ""), flush=True)
    n = len(frames_out)
    if not n:
        raise SystemExit("no frames decoded (are the clip's segment files on disk?)")
    tag = os.environ.get("OSB_WIN_TAG", "win")  # output name; e.g. "win4" for a cadence experiment
    path = rec / "detect" / f"ball_{seg}_{start:g}_{seconds:g}.{tag}.json"
    path.write_text(json.dumps({"segment": seg, "fps": fps, "model": str(ckpt), "mode": "window", "wasb": WASB,
                                "frames": frames_out}))
    print(f"{n} frame pairs in {(time.time() - t0) / 60:.1f} min; {tiles_total / n:.1f} tiles per frame pair"
          f" vs {2 * len(detect.tiles(H, W))} for full scans; {n_full} full scans"
          + (f" ({n_local} of them local)" if LOST_LOCAL else "")
          + (f"; WASB {wasb.tiles / n:.1f} tiles per frame pair, {t_wasb / 60:.1f} min" if wasb else "")
          + f" -> {path.name}", flush=True)
    os._exit(0)  # the output is written; interpreter teardown (MPS, decoder threads) has hung for hours


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), int(a[2]), float(a[3]), float(a[4]), Path(a[5]), float(a[6]) if len(a) > 6 else 10.0)
