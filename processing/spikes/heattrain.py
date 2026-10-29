#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4", "pyyaml>=6"]
# ///
"""WASB fine-tuned on this game's ball labels, with the empty pitch as a
fourth input frame.

    uv run processing/spikes/heattrain.py footage/rec3 bg
    uv run processing/spikes/heattrain.py footage/rec3 data labels labels_seg3 labels_seg9
    uv run processing/spikes/heattrain.py footage/rec3 train <name> [--hold <set>] [--steps N]
    uv run processing/spikes/heattrain.py footage/rec3 eval <name | stock> <set>

The stock WASB (ballheat.py, football weights from ISSIA-CNR) finds the
far balls only from 2x and 4x downscaled tiles, at ~10 peaks per frame
of clutter. The phones don't move, so each phone's view of the empty
pitch is known: `bg` writes a median of 25 frames spread over every
segment, per phone, at 2x downscale (<recording>/bg/<role>_<seg>.png,
upright as ballwin.py sees the frames). The model gets it as a fourth
frame ahead of the usual three (ballheat.wasb_with_bg: the stock model
exactly until trained), so a ball can show as what is new on the pitch.

`data` turns label sets (labelset.py, labelled on the page) into
training frames: each labelled phone image with status "ball" or "none"
is found in a decoded window (ballheat.decode_window), and it and its
neighbours at 30 fps are stored at 2x downscale under
<recording>/heattrain/<set>/ with the balls (in-play ones are targets,
spare balls are left out of the loss).

`train` fine-tunes from the football weights on 512x288 crops of those
frames at a random downscale of 1.6-4.5x from 4K, half of them around a
ball, with flips and brightness jitter; the target is a disk at the ball
(WASB's binary target, radius at least 2.5 px) under WASB's weighted
BCE, on the middle frame's output. --hold leaves a label set out (and
the random set's frames within 20 s of that clip) and scores it every
EVAL_EVERY steps. Checkpoints go to <recording>/heattrain/<name>.pt.

`eval` runs a model (or the stock one) over every frame of a set as
ballwin.py's full scans do (2x and 4x tiles) and prints, per peak
threshold, the in-play balls found (a peak within HIT_PX or 1.5 ball
diameters) and the peaks per frame, raw and after balljudge.py's crop
judge (>= 0.3, as the tracker filters).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ballheat  # noqa: E402
import balljudge  # noqa: E402
import gpu  # noqa: E402
import phase0  # noqa: E402

BG_FRAMES = 25          # frames in each segment's background median
SW, SH = 1920, 1080     # stored frames: 2x downscale of 4K
SCALE_MIN, SCALE_MAX = 1.6, 4.5   # training crops: downscale from 4K
POS_SHARE = 0.5         # share of crops placed around a ball
MIN_R = 2.5             # target disk radius in model pixels (WASB's binary_fixed_size)
HOLD_MARGIN_S = 20.0    # random-set frames this close to a held-out clip are left out too
STEPS, BATCH, LR = 3000, 8, 1e-4   # batch 8: ~11 GB of activations on MPS, 0.8 s a step
EVAL_EVERY = 500
HIT_PX = 12.0           # as ballheat.py: a hit is within this or 1.5 ball diameters
SCALES = (2, 4)
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
JUDGE_MIN = 0.3


def setup(rec: Path):
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = {"main": report["upsideDown"]["main"], "secondary": report["upsideDown"]["secondary"]}
    return devs, report, flip


def bg_path(rec: Path, role: str, seg: int) -> Path:
    return rec / "bg" / f"{role}_{seg}.png"


def load_bg(rec: Path, role: str, seg: int) -> np.ndarray:
    """The segment's background, RGB at 2x downscale."""
    im = cv2.imread(str(bg_path(rec, role, seg)))
    if im is None:
        raise SystemExit(f"no background {bg_path(rec, role, seg)}: run `heattrain.py {rec} bg` first")
    return im[..., ::-1].copy()


def make_bg(rec: Path) -> None:
    devs, _, flip = setup(rec)
    (rec / "bg").mkdir(exist_ok=True)
    for role in ("main", "secondary"):
        for s in devs[role]["segments"]:
            out, path = bg_path(rec, role, s["index"]), rec / s["objectKey"]
            if out.exists() or not path.exists():
                continue
            dur = s["durationMs"] / 1000
            frames = []
            for j in range(BG_FRAMES):
                got = phase0.frames_at(path, (j + 0.5) / BG_FRAMES * dur, 1, 1, 3840, flip[role])
                if got:
                    frames.append(cv2.resize(got[0], (SW, SH), interpolation=cv2.INTER_AREA))
            cv2.imwrite(str(out), np.median(np.stack(frames), 0).astype(np.uint8))
            print(f"{out.name}: median of {len(frames)} frames", flush=True)


def label_frames(rec: Path, name: str):
    """(frame dict, item) for every phone image of a label set."""
    items = json.loads((rec / name / "items.json").read_text())
    for it in items:
        for f in it.get("frames", [it]):
            yield f, it


def make_data(rec: Path, sets: list[str]) -> None:
    devs, report, flip = setup(rec)
    diam_at = balljudge.expected_at(rec)
    for name in sets:
        lab = rec / name
        labels = json.loads((lab / "labels.json").read_text())
        out = rec / "heattrain" / name
        out.mkdir(parents=True, exist_ok=True)
        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        skipped, t0 = {}, time.time()
        for f, it in label_frames(rec, name):
            fid, L = f["id"], labels.get(f["id"])
            if fid in meta or L is None or L["status"] not in ("ball", "none"):
                continue
            role, seg = f["phone"], it["segment"]
            path = rec / devs[role]["segments"][seg]["objectKey"]
            if not path.exists():
                skipped["no video"] = skipped.get("no video", 0) + 1
                continue
            off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
            frames, idx, real0 = ballheat.decode_window(path, it["t"] + off, flip[role],
                                                        cv2.imread(str(lab / f["image"])))
            if idx - 1 < real0 or idx + 1 >= len(frames):
                skipped["edge"] = skipped.get("edge", 0) + 1
                continue
            trip = np.stack([cv2.resize(fr, (SW, SH), interpolation=cv2.INTER_AREA)[..., ::-1]
                             for fr in frames[idx - 1:idx + 2]])
            np.save(out / f"{fid}.npy", trip)
            meta[fid] = {"role": role, "segment": seg, "t": it["t"], "status": L["status"],
                         "balls": [[b["x"], b["y"], bool(b.get("inPlay", True)), round(float(diam_at(role, b["x"], b["y"])), 2)]
                                   for b in L.get("balls", [])]}
            if len(meta) % 20 == 0:
                meta_path.write_text(json.dumps(meta))
                print(f"{name}: {len(meta)} frames, {(time.time() - t0) / 60:.1f} min", flush=True)
        meta_path.write_text(json.dumps(meta))
        print(f"{name}: {len(meta)} frames ({sum(any(b[2] for b in m['balls']) for m in meta.values())} with an"
              f" in-play ball); skipped {skipped or 'none'}", flush=True)


def load_sets(rec: Path, hold: str | None) -> tuple[list[dict], list[dict]]:
    """Training and held-out frames: dicts with the meta plus "set", "id"."""
    span = None
    if hold:
        c = json.loads((rec / hold / "items.json").read_text())
        ts = [it["t"] for it in c]
        span = (c[0]["segment"], min(ts) - HOLD_MARGIN_S, max(ts) + HOLD_MARGIN_S)
    train, held = [], []
    for d in sorted((rec / "heattrain").glob("*/meta.json")):
        name = d.parent.name
        for fid, m in json.loads(d.read_text()).items():
            row = dict(m, set=name, id=fid)
            if name == hold:
                held.append(row)
            elif span and m["segment"] == span[0] and span[1] <= m["t"] <= span[2]:
                continue
            else:
                train.append(row)
    return train, held


class Crops(torch.utils.data.Dataset):
    """Random 512x288 training crops: (12, 288, 512) float input (background,
    previous, this, next frame; ImageNet-normalised), target and loss mask."""

    def __init__(self, rec: Path, rows: list[dict], n: int, seed: int = 0):
        self.rec, self.rows, self.n, self.seed = rec, rows, n, seed
        self.pos = [i for i, r in enumerate(rows) if any(b[2] for b in r["balls"])]
        self.cache: dict = {}

    def __len__(self):
        return self.n

    def arrays(self, r):
        key = (r["set"], r["id"])
        if key not in self.cache:
            self.cache[key] = np.load(self.rec / "heattrain" / r["set"] / f"{r['id']}.npy", mmap_mode="r")
        bkey = (r["role"], r["segment"])
        if bkey not in self.cache:
            self.cache[bkey] = load_bg(self.rec, *bkey)
        return self.cache[bkey], self.cache[key]

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k))
        if rng.random() < POS_SHARE:
            r = self.rows[self.pos[rng.integers(len(self.pos))]]
            ball = [b for b in r["balls"] if b[2]][rng.integers(sum(b[2] for b in r["balls"]))]
        else:
            r, ball = self.rows[rng.integers(len(self.rows))], None
        bg, trip = self.arrays(r)
        s = rng.uniform(SCALE_MIN, SCALE_MAX) / 2          # stored pixels per model pixel
        cw, ch = 512 * s, 288 * s
        if ball is not None:
            bx, by = ball[0] / 2, ball[1] / 2
            x0 = rng.uniform(bx - cw + 8 * s, bx - 8 * s)
            y0 = rng.uniform(by - ch + 8 * s, by - 8 * s)
        else:
            x0 = rng.uniform(0, SW - cw)
            y0 = rng.uniform(ballheat.BAND_TOP / 2 - ch / 2, SH - ch)
        x0, y0 = float(np.clip(x0, 0, SW - cw)), float(np.clip(y0, 0, SH - ch))
        xi, yi, wi, hi = int(x0), int(y0), int(math.ceil(cw)), int(math.ceil(ch))
        interp = cv2.INTER_AREA if s > 1 else cv2.INTER_LINEAR
        ims = [cv2.resize(np.asarray(im[yi:yi + hi, xi:xi + wi]), (512, 288), interpolation=interp)
               for im in (bg, trip[0], trip[1], trip[2])]
        x = np.stack(ims).astype(np.float32)
        x = x * rng.uniform(0.8, 1.25) + rng.uniform(-20, 20)       # the same jitter on all four
        x = np.clip(x, 0, 255) / 255.0
        tgt, mask = np.zeros((288, 512), np.float32), np.ones((288, 512), np.float32)
        yy, xx = np.mgrid[0:288, 0:512]
        sx, sy = 512 / wi, 288 / hi
        for b in r["balls"]:
            u, v = (b[0] / 2 - xi) * sx, (b[1] / 2 - yi) * sy
            rad = max(MIN_R, b[3] / 2 / s / 2)
            d2 = (xx - u) ** 2 + (yy - v) ** 2
            if b[2]:
                tgt[d2 <= rad ** 2] = 1.0
            else:
                mask[d2 <= (2 * rad + 3) ** 2] = 0.0   # a spare ball: neither ball nor background
        if rng.random() < 0.5:
            x, tgt, mask = x[:, :, ::-1], tgt[:, ::-1], mask[:, ::-1]
        x = (x - ballheat.MEAN) / ballheat.STD
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(12, 288, 512))
        return torch.from_numpy(x), torch.from_numpy(tgt.copy()), torch.from_numpy(mask.copy())


def wbce(logits: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """WASB's weighted BCE (focal, gamma 2) on the middle frame's heatmap."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    loss = -((1 - p) ** 2 * tgt * torch.log(p) + p ** 2 * (1 - tgt) * torch.log(1 - p))
    return (loss * mask).sum() / mask.sum()


def score(rec: Path, rows: list[dict], wasb, judge=None) -> dict:
    """Full-scan peaks (2x and 4x tiles) per frame -> found / peaks per
    frame per threshold, raw and judged."""
    dev = gpu.device()
    corners = wasb.scan_corners(2160, 3840)
    n_ball = sum(any(b[2] for b in r["balls"]) for r in rows)
    found = {t: [0, 0] for t in THRESHOLDS}
    peaks = {t: [0, 0] for t in THRESHOLDS}
    for r in rows:
        bg = load_bg(rec, r["role"], r["segment"])
        trip = np.load(rec / "heattrain" / r["set"] / f"{r['id']}.npy")
        fr = torch.from_numpy(trip).to(dev).permute(0, 3, 1, 2).float() / 255
        b = torch.from_numpy(bg).to(dev).permute(2, 0, 1).float() / 255
        cands = wasb(fr, corners, b, base=2)
        js = [1.0] * len(cands)
        if judge is not None and cands:
            img = cv2.imread(str(rec / r["set"] / "images" / f"{r['id']}.jpg"))
            if img is not None:
                js = list(judge(np.stack([balljudge.crop_at(img, c[0], c[1], max(4.0, r_diam(r, c)))
                                          for c in cands])))
        balls = [bb for bb in r["balls"] if bb[2]]
        for t in THRESHOLDS:
            for k, use_j in ((0, False), (1, True)):
                kept = [c for c, j in zip(cands, js) if c[2] >= t and (not use_j or j >= JUDGE_MIN)]
                peaks[t][k] += len(kept)
                if balls and any(np.hypot(c[0] - bb[0], c[1] - bb[1]) <= max(HIT_PX, 1.5 * bb[3])
                                 for c in kept for bb in balls):
                    found[t][k] += 1
    return {"n": len(rows), "n_ball": n_ball, "found": found, "peaks": peaks}


_DIAM = {}


def r_diam(r: dict, c) -> float:
    """Expected ball diameter at a candidate (for the judge's crop)."""
    rec = _DIAM["rec"]
    if "fn" not in _DIAM:
        _DIAM["fn"] = balljudge.expected_at(rec)
    return float(_DIAM["fn"](r["role"], c[0], c[1]))


def print_score(label: str, s: dict) -> None:
    print(f"{label}: {s['n']} frames, {s['n_ball']} with an in-play ball")
    for t in THRESHOLDS:
        f, p = s["found"][t], s["peaks"][t]
        print(f"  >= {t:<4} raw found {f[0]:3d}/{s['n_ball']} ({100 * f[0] / max(1, s['n_ball']):3.0f}%)"
              f" {p[0] / s['n']:5.1f}/frame   judged found {f[1]:3d} ({100 * f[1] / max(1, s['n_ball']):3.0f}%)"
              f" {p[1] / s['n']:5.1f}/frame", flush=True)


def train(rec: Path, name: str, hold: str | None, steps: int) -> None:
    import ballwin

    dev = gpu.device()
    rows, held = load_sets(rec, hold)
    print(f"{len(rows)} training frames ({sum(any(b[2] for b in r['balls']) for r in rows)} with an in-play ball)"
          f"{f', {len(held)} held out ({hold})' if hold else ''}", flush=True)
    model = ballheat.wasb_with_bg().to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loader = torch.utils.data.DataLoader(Crops(rec, rows, steps * BATCH), batch_size=BATCH, num_workers=2,
                                         persistent_workers=True)
    out = rec / "heattrain" / f"{name}.pt"
    t0, run = time.time(), []
    for step, (x, tgt, mask) in enumerate(loader, 1):
        x, tgt, mask = x.to(dev), tgt.to(dev), mask.to(dev)
        loss = wbce(model(x)[0][:, 1], tgt, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(float(loss))
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {np.mean(run[-50:]):.5f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % EVAL_EVERY == 0 or step == steps:
            torch.save(model.state_dict(), out)
            if held:
                w = ballwin.Wasb(dev, out)
                print_score(f"step {step}, held out {hold}", score(rec, held, w))
                del w
            model.train()
    print(f"-> {out}")


def evaluate(rec: Path, name: str, hold: str) -> None:
    import ballwin

    dev = gpu.device()
    _DIAM["rec"] = rec
    _, held = load_sets(rec, hold)
    w = ballwin.Wasb(dev, None if name == "stock" else rec / "heattrain" / f"{name}.pt")
    judge = balljudge.load_judge(rec)
    print_score(f"{name} on {hold}", score(rec, held, w, judge))


if __name__ == "__main__":
    a = sys.argv
    rec, cmd = Path(a[1]), a[2]
    if cmd == "bg":
        make_bg(rec)
    elif cmd == "data":
        make_data(rec, a[3:])
    elif cmd == "train":
        opts = dict(zip(a[4::2], a[5::2]))
        train(rec, a[3], opts.get("--hold"), int(opts.get("--steps", STEPS)))
    elif cmd == "eval":
        evaluate(rec, a[3], a[4])
    else:
        raise SystemExit(__doc__)
