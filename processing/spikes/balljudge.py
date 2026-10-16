#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""A small classifier that judges ball candidates from a crop around each.

    uv run processing/spikes/balljudge.py data  footage/rec3      # crops from the labels + motion.py's candidates
    uv run processing/spikes/balljudge.py train footage/rec3 [epochs]
    uv run processing/spikes/balljudge.py eval  footage/rec3      # held-back moments: candidates ranked by the judge
    uv run processing/spikes/balljudge.py clip  footage/rec3 footage/rec3/detect/ball_9_60_120.json  # score a clip's candidates

Motion candidates (motion.py) are nearly free but ~120 per frame; the
fine-tuned detector is accurate but scans ~28 tiles of 640 px per frame.
This sits between: a crop of CROP x the expected ball size around each
candidate (context: grass, line, boot, body), resized to SIZE px, through
a few conv layers. Positives are crops at every labelled ball (jittered);
negatives are motion candidates and the stock detector's suggestions
that aren't a ball. Split by moment, as balldata.py does.

Writes <recording>/balljudge/: crops.npz, judge.pt, and eval output.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import balldata  # noqa: E402

SIZE = 48           # crop fed to the net
CROP = 4.0          # crop side as a multiple of the expected ball diameter ...
CROP_MIN = 32       # ... at least this many frame pixels
HIT_PX = 15
JITTER = 3          # positives: random offsets of up to this (frame px) for each copy
POS_COPIES = 6
NEG_PER_FRAME = 60  # negatives kept per frame (random, plus every detector miss)


def balls_of(v):
    return balldata.balls_of(v)


def expect_px(c: dict) -> float:
    return 2 * np.sqrt(c["expect"] / np.pi)


def crop_at(img: np.ndarray, x: float, y: float, diam: float) -> np.ndarray:
    side = int(max(CROP_MIN, CROP * diam))
    h, w = img.shape[:2]
    x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
    pad = side
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    c = big[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def frames_with_split(rec: Path):
    lab_dir = rec / "labels"
    items = json.loads((lab_dir / "items.json").read_text())
    labels = json.loads((lab_dir / "labels.json").read_text())
    motion = json.loads((lab_dir / "motion.json").read_text())
    out = []
    for it in items:
        moment = it["id"] if it.get("kind") == "pair" else it["id"].rsplit("_", 1)[0]
        for f in (it["frames"] if it.get("kind") == "pair" else [it]):
            v = labels.get(f["id"])
            if not v or v["status"] == "unsure" or f["id"] not in motion:
                continue
            out.append((balldata.split_of(moment), f, balls_of(v), motion[f["id"]]))
    return lab_dir, out


def expected_at(rec: Path):
    """Expected ball diameter (px) at a frame pixel, from the pitch map."""
    import follow
    import pitch as P

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = {"main": cm.at(3840), "secondary": cs.at(3840)}
    pose = P.load_pose(rec)

    def at(phone, x, y):
        cam = cams[phone]
        d = cam.unproject(np.array([[x, y]], float))[0]
        yaw = np.degrees(np.arctan2(d[0], d[2]))
        pit = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
        gx, gz = pose.to_pitch(yaw, pit)
        if np.isnan(gx):
            return 6.0
        dist = float(np.sqrt((gx - pose.cx) ** 2 + (gz - pose.cz) ** 2 + pose.h ** 2))
        return float(np.clip(0.22 / dist * cam.f, 4, 160))
    return at


def data(rec: Path) -> None:
    lab_dir, frames = frames_with_split(rec)
    out = rec / "balljudge"
    out.mkdir(exist_ok=True)
    diam_at = expected_at(rec)
    rng = np.random.default_rng(3)
    X, y, split, meta = [], [], [], []
    for k, (sp, f, balls, cands) in enumerate(frames):
        img = cv2.imread(str(lab_dir / f["image"]))
        for b in balls:
            d = diam_at(f["phone"], b["x"], b["y"])
            for _ in range(POS_COPIES):
                dx, dy = rng.integers(-JITTER, JITTER + 1, size=2)
                X.append(crop_at(img, b["x"] + dx, b["y"] + dy, d)); y.append(1); split.append(sp)
                meta.append((f["id"], "ball"))
        negs = [(c["x"], c["y"], expect_px(c)) for c in cands
                if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > HIT_PX for b in balls)]
        hard = [(c["x"], c["y"], diam_at(f["phone"], c["x"], c["y"])) for c in f["candidates"]
                if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > HIT_PX for b in balls)]
        keep = [negs[i] for i in rng.permutation(len(negs))[:NEG_PER_FRAME]] + hard
        for x, yy, d in keep:
            X.append(crop_at(img, x, yy, d)); y.append(0); split.append(sp); meta.append((f["id"], "neg"))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(frames)} frames, {len(X)} crops", flush=True)
    X = np.stack(X); y = np.array(y, np.uint8); split = np.array(split)
    np.savez_compressed(out / "crops.npz", X=X, y=y, split=split)
    for sp in ("train", "valid"):
        m = split == sp
        print(f"{sp}: {m.sum()} crops, {y[m].sum()} positives")


class Judge:
    """Tiny conv net; built lazily so numpy-only commands don't import torch."""

    @staticmethod
    def build():
        import torch.nn as nn

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 96),  # 48 -> 3
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, 1))


def to_tensor(X: np.ndarray, torch):
    return torch.from_numpy(X[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255


def train(rec: Path, epochs: int) -> None:
    import torch
    import torch.nn.functional as F

    out = rec / "balljudge"
    z = np.load(out / "crops.npz")
    X, y, split = z["X"], z["y"], z["split"]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tr, va = split == "train", split == "valid"
    Xt, yt = to_tensor(X[tr], torch), torch.from_numpy(y[tr]).float()
    Xv, yv = to_tensor(X[va], torch).to(dev), torch.from_numpy(y[va]).float().to(dev)
    net = Judge.build().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * 100)
    pos = np.where(y[tr] == 1)[0]; neg = np.where(y[tr] == 0)[0]
    rng = np.random.default_rng(0)
    best, t0 = 0.0, time.time()
    for ep in range(epochs):
        net.train()
        for _ in range(100):  # balanced batches: half positives
            idx = np.r_[rng.choice(pos, 64), rng.choice(neg, 64)]
            xb = Xt[idx].to(dev); yb = yt[idx].to(dev)
            if rng.random() < 0.5:
                xb = xb.flip(3)
            xb = xb * (0.7 + 0.6 * torch.rand(len(xb), 1, 1, 1, device=dev))  # brightness
            loss = F.binary_cross_entropy_with_logits(net(xb).squeeze(1), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            s = torch.sigmoid(torch.cat([net(Xv[i:i + 512]).squeeze(1) for i in range(0, len(Xv), 512)]))
        ap = average_precision(s.cpu().numpy(), yv.cpu().numpy())
        if ap > best:
            best = ap
            torch.save(net.state_dict(), out / "judge.pt")
        print(f"epoch {ep}: loss {loss.item():.3f}, valid AP {ap:.3f} (best {best:.3f}), {(time.time() - t0) / 60:.1f} min",
              flush=True)


def average_precision(scores, labels) -> float:
    o = np.argsort(-scores); l_ = labels[o]
    tp = np.cumsum(l_); prec = tp / np.arange(1, len(l_) + 1)
    return float((prec * l_).sum() / max(1, l_.sum()))


def load_judge(rec: Path):
    import torch

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    net = Judge.build().to(dev)
    net.load_state_dict(torch.load(rec / "balljudge" / "judge.pt", map_location=dev))
    net.eval()

    def score(crops: np.ndarray) -> np.ndarray:
        if not len(crops):
            return np.zeros(0)
        with torch.no_grad():
            x = to_tensor(crops, torch).to(dev)
            return torch.sigmoid(torch.cat([net(x[i:i + 1024]).squeeze(1) for i in range(0, len(x), 1024)])).cpu().numpy()
    return score


def evaluate(rec: Path) -> None:
    """Held-back moments: motion candidates (and the stock detector's
    suggestions) ranked by the judge; match-ball recall in the top k and
    wrong candidates per frame at thresholds."""
    lab_dir, frames = frames_with_split(rec)
    diam_at = expected_at(rec)
    score = load_judge(rec)
    rows = []
    t0, n_crops = time.time(), 0
    for sp, f, balls, cands in frames:
        if sp != "valid":
            continue
        img = cv2.imread(str(lab_dir / f["image"]))
        pts = [(c["x"], c["y"], expect_px(c), "motion") for c in cands]
        pts += [(c["x"], c["y"], diam_at(f["phone"], c["x"], c["y"]), "det") for c in f["candidates"]]
        crops = np.stack([crop_at(img, x, y, d) for x, y, d, _ in pts]) if pts else np.zeros((0, SIZE, SIZE, 3), np.uint8)
        s = score(crops); n_crops += len(crops)
        match = [b for b in balls if b["inPlay"]]
        rows.append((f, match[0] if match else None, balls, [(p, float(v)) for p, v in zip(pts, s)]))
    print(f"{len(rows)} held-back frames, {sum(r[1] is not None for r in rows)} with the match ball;"
          f" judge: {n_crops} crops in {time.time() - t0:.1f} s ({1000 * (time.time() - t0) / n_crops:.2f} ms each)")

    def hit(p, b):
        return np.hypot(p[0] - b["x"], p[1] - b["y"]) < HIT_PX

    n_match = sum(r[1] is not None for r in rows)
    for src in ("motion", "motion+det"):
        print(f"\n{src}:")
        for top in (1, 3, 5, 10):
            got = 0
            for f, mb, balls, sc in rows:
                if mb is None:
                    continue
                ranked = sorted([(v, p) for p, v in sc if src == "motion+det" or p[3] == "motion"], key=lambda q: -q[0])
                got += any(hit(p, mb) for _, p in ranked[:top])
            print(f"  top {top:2d}: match ball {100 * got / n_match:3.0f}%")
        for thr in (0.5, 0.8, 0.9, 0.95):
            got = wrong = 0
            for f, mb, balls, sc in rows:
                keep = [p for p, v in sc if v >= thr and (src == "motion+det" or p[3] == "motion")]
                if mb is not None:
                    got += any(hit(p, mb) for p in keep)
                wrong += sum(not any(hit(p, b) for b in balls) for p in keep)
            print(f"  >= {thr:.2f}: match ball {100 * got / n_match:3.0f}%, wrong {wrong / len(rows):5.2f} per frame")
    det_only = sum(any(hit((c["x"], c["y"]), mb) for c in f["candidates"] if c["conf"] >= 0.2)
                   for f, mb, _, _ in rows if mb is not None)
    print(f"\nstock detector alone (conf >= 0.2): match ball {100 * det_only / n_match:.0f}%")


def judge_clip(rec: Path, cands_path: Path) -> None:
    """Adds a "judge" score to every candidate in a ballclip.py file: the
    clip is streamed once per phone at the candidates' frame rate (a
    couple of minutes for 2 minutes of 4K), each candidate cropped from
    its frame and scored."""
    import follow

    data = json.loads(cands_path.read_text())
    seg, fps, frames = data["segment"], data["fps"], data["frames"]
    start = frames[0]["t"]
    seconds = len(frames) / fps
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    diam_at = expected_at(rec)
    score = load_judge(rec)
    W, H = follow.SRC_W, follow.SRC_H
    t0, n = time.time(), 0
    for phone in (0, 1):
        role = ("main", "secondary")[phone]
        stream = follow.game_stream(rec, devs, report, phone, seg, start, seconds, W, fps=fps)
        for f, buf in zip(frames, stream):
            dets = [d for d in f["detections"] if d["phone"] == phone]
            if not dets:
                continue
            img = cv2.cvtColor(buf.reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_NV12)
            if flip[phone]:
                img = cv2.rotate(img, cv2.ROTATE_180)
            crops = np.stack([crop_at(img, (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2,
                                      diam_at(role, (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2))
                              for d in dets])
            for d, v in zip(dets, score(crops)):
                d["judge"] = round(float(v), 3)
            n += len(dets)
        print(f"{role}: done, {n} candidates judged, {(time.time() - t0) / 60:.1f} min", flush=True)
    cands_path.write_text(json.dumps(data))
    js = np.array([d["judge"] for f in frames for d in f["detections"] if "judge" in d])
    print(f"judge >= 0.5: {100 * (js >= 0.5).mean():.0f}% of candidates; >= 0.9: {100 * (js >= 0.9).mean():.0f}%")


if __name__ == "__main__":
    cmd, rec = sys.argv[1], Path(sys.argv[2])
    if cmd == "clip":
        judge_clip(rec, Path(sys.argv[3]))
    elif cmd == "data":
        data(rec)
    elif cmd == "train":
        train(rec, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    elif cmd == "eval":
        evaluate(rec)
    else:
        raise SystemExit(__doc__)
