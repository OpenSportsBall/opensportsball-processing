#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""A small classifier that judges ball candidates from a crop around each.

    uv run processing/spikes/balljudge.py data  footage/rec3 [train clip sets] [test clip sets]   # e.g. seg9 seg3
    uv run processing/spikes/balljudge.py merge footage/rec5-game footage/rec3 footage/rec5-game*3   # crops of several
    uv run processing/spikes/balljudge.py train footage/rec3 [epochs] [model name]
    uv run processing/spikes/balljudge.py eval  footage/rec3      # held-back moments: candidates ranked by the judge
    uv run processing/spikes/balljudge.py clip  footage/rec3 <candidates.json> [model name] [output tag]

Motion candidates (motion.py) are nearly free but ~120 per frame; the
fine-tuned detector is accurate but scans ~28 tiles of 640 px per frame.
This sits between: a crop of CROP x the expected ball size around each
candidate (context: grass, line, boot, body), resized to SIZE px, through
a few conv layers. Positives are crops at every labelled ball (jittered);
negatives are motion candidates and the stock detector's suggestions
that aren't a ball. Split by moment, as balldata.py does.

Writes <recording>/balljudge/: crops.npz, judge.pt, and eval output.

Every game is judged by the standard model, footage/models/ball_judge.pt
(STANDARD; ballwin.py's inline scores, eval, and clip with no model name).
A model name picks <recording>/balljudge/<name>.pt instead, as trained
here; to make one the standard, copy it over STANDARD and note it in
footage/models/README.md.
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
import phase0  # noqa: E402

STANDARD = Path(__file__).resolve().parents[2] / "footage" / "models" / "ball_judge.pt"

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
    devs = phase0.paired_devices(manifest, report)
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


def clip_frames(rec: Path, name: str, split: str):
    """Frames of a clip label set (labelset.py clip) with their balls and,
    as candidates, every detection the clip's detector made at that
    moment (its false ones are the most useful negatives)."""
    lab = rec / f"labels_{name}"
    items = json.loads((lab / "items.json").read_text())
    labels = json.loads((lab / "labels.json").read_text())
    seg0, start = int(items[0]["id"].split("_")[0][1:]), float(items[0]["id"].split("_")[1])
    files = sorted((rec / "detect").glob(f"ball_{seg0}_{start:g}_*.json"))
    cands = json.loads(files[0].read_text()) if files else None
    out = []
    for it in items:
        k = int(round(it["clipT"] * cands["fps"])) if cands else -1
        for f in it["frames"]:
            v = labels.get(f["id"])
            if not v or v["status"] == "unsure":
                continue
            phone = 0 if f["phone"] == "main" else 1
            dets = [{"x": (d["box"][0] + d["box"][2]) / 2, "y": (d["box"][1] + d["box"][3]) / 2, "expect": 0}
                    for d in cands["frames"][k]["detections"] if d["phone"] == phone] if 0 <= k < len(cands["frames"]) else []
            out.append((split, dict(f, candidates=[]), v.get("balls", []), dets, lab))
    return out


def data(rec: Path, train_sets=(), test_sets=()) -> None:
    """Crops from the random label set (split by moment into train/valid)
    plus clip sets: `train_sets` join the training data, `test_sets` are
    held out whole (split "test")."""
    lab_dir, frames = frames_with_split(rec)
    frames = [(sp, f, balls, cands, lab_dir) for sp, f, balls, cands in frames]
    for name in train_sets:
        frames += clip_frames(rec, name, "train")
    for name in test_sets:
        frames += clip_frames(rec, name, "test")
    out = rec / "balljudge"
    out.mkdir(exist_ok=True)
    diam_at = expected_at(rec)
    rng = np.random.default_rng(3)
    X, y, split, meta = [], [], [], []
    for k, (sp, f, balls, cands, lab_dir) in enumerate(frames):
        img = cv2.imread(str(lab_dir / f["image"]))
        for b in balls:
            d = diam_at(f["phone"], b["x"], b["y"])
            for _ in range(POS_COPIES):
                dx, dy = rng.integers(-JITTER, JITTER + 1, size=2)
                X.append(crop_at(img, b["x"] + dx, b["y"] + dy, d)); y.append(1); split.append(sp)
                meta.append((f["id"], "ball"))
        negs = [(c["x"], c["y"], expect_px(c) if c.get("expect") else diam_at(f["phone"], c["x"], c["y"])) for c in cands
                if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > HIT_PX for b in balls)]
        if not f["candidates"] and not cands:  # a clip frame with no detections: random field crops
            h, w = img.shape[:2]
            negs = [(float(x), float(yy), diam_at(f["phone"], x, yy))
                    for x, yy in zip(rng.integers(0, w, 12), rng.integers(760, h, 12))]
        hard = [(c["x"], c["y"], diam_at(f["phone"], c["x"], c["y"])) for c in f["candidates"]
                if all(np.hypot(c["x"] - b["x"], c["y"] - b["y"]) > HIT_PX for b in balls)]
        keep = [negs[i] for i in rng.permutation(len(negs))[:NEG_PER_FRAME]] + hard
        for x, yy, d in keep:
            X.append(crop_at(img, x, yy, d)); y.append(0); split.append(sp); meta.append((f["id"], "neg"))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(frames)} frames, {len(X)} crops", flush=True)
    X = np.stack(X); y = np.array(y, np.uint8); split = np.array(split)
    np.savez_compressed(out / "crops.npz", X=X, y=y, split=split)
    for sp in ("train", "valid", "test"):
        m = split == sp
        print(f"{sp}: {m.sum()} crops, {y[m].sum()} positives")


def merge(out_rec: Path, specs: list[str]) -> None:
    """One crops.npz from several recordings' (each made by `data`; read
    from crops_<recording>.npz when there is one, as a merge leaves the
    output's own, else crops.npz): "<rec>*N"
    repeats that recording's train crops N times, to weight a new set (a
    ball colour the others haven't seen) against a large old one. Each
    keeps its own split."""
    out = out_rec / "balljudge"
    own = out / "crops.npz"
    if own.exists() and not (out / f"crops_{out_rec.name}.npz").exists():
        own.rename(out / f"crops_{out_rec.name}.npz")
    Xs, ys, splits = [], [], []
    for spec in specs:
        rec, n = Path(spec.split("*")[0]), int(spec.split("*")[1]) if "*" in spec else 1
        src = rec / "balljudge" / f"crops_{rec.name}.npz"
        if not src.exists():  # a recording's own crops; its crops.npz may be a merge made there
            src = src.with_name("crops.npz")
        z = np.load(src)
        tr = z["split"] == "train"
        for r in range(n):
            m = np.ones(len(tr), bool) if r == 0 else tr
            Xs.append(z["X"][m]); ys.append(z["y"][m]); splits.append(z["split"][m])
        print(f"{src}: {len(tr)} crops ({z['y'].sum()} positives), train x{n}")
    X, y, split = np.concatenate(Xs), np.concatenate(ys), np.concatenate(splits)
    np.savez_compressed(own, X=X, y=y, split=split)
    for sp in ("train", "valid", "test"):
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


def train(rec: Path, epochs: int, name: str = "judge") -> None:
    import torch
    import torch.nn.functional as F

    out = rec / "balljudge"
    z = np.load(out / "crops.npz")
    X, y, split = z["X"], z["y"], z["split"]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = split == "train", split == "valid", split == "test"
    if te.any():
        Xte, yte = to_tensor(X[te], torch).to(dev), y[te]
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
            torch.save(net.state_dict(), out / f"{name}.pt")
        extra = ""
        if te.any():
            with torch.no_grad():
                st = torch.sigmoid(torch.cat([net(Xte[i:i + 512]).squeeze(1) for i in range(0, len(Xte), 512)]))
            extra = f", held-out clip AP {average_precision(st.cpu().numpy(), yte):.3f}"
        print(f"epoch {ep}: loss {loss.item():.3f}, valid AP {ap:.3f} (best {best:.3f}){extra}, {(time.time() - t0) / 60:.1f} min",
              flush=True)


def average_precision(scores, labels) -> float:
    o = np.argsort(-scores); l_ = labels[o]
    tp = np.cumsum(l_); prec = tp / np.arange(1, len(l_) + 1)
    return float((prec * l_).sum() / max(1, l_.sum()))


def load_judge(rec: Path, name: str | None = None):
    """The standard judge, or <rec>/balljudge/<name>.pt when named."""
    import torch

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    net = Judge.build().to(dev)
    net.load_state_dict(torch.load(STANDARD if name is None else rec / "balljudge" / f"{name}.pt", map_location=dev))
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


def judge_clip(rec: Path, cands_path: Path, name: str | None = None, out_tag: str = "") -> None:
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
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    diam_at = expected_at(rec)
    score = load_judge(rec, name)
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
    if out_tag:
        cands_path = cands_path.with_name(cands_path.name[:-5] + f".{out_tag}.json")
    cands_path.write_text(json.dumps(data))
    js = np.array([d["judge"] for f in frames for d in f["detections"] if "judge" in d])
    print(f"judge >= 0.5: {100 * (js >= 0.5).mean():.0f}% of candidates; >= 0.9: {100 * (js >= 0.9).mean():.0f}%")


if __name__ == "__main__":
    cmd, rec = sys.argv[1], Path(sys.argv[2])
    if cmd == "clip":
        judge_clip(rec, Path(sys.argv[3]), (sys.argv[4] if len(sys.argv) > 4 else "") or None,
                   sys.argv[5] if len(sys.argv) > 5 else "")
    elif cmd == "data":
        train_sets = [x for x in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if x]
        test_sets = [x for x in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if x]
        data(rec, train_sets, test_sets)
    elif cmd == "merge":
        merge(rec, sys.argv[3:])
    elif cmd == "train":
        train(rec, int(sys.argv[3]) if len(sys.argv) > 3 else 30, sys.argv[4] if len(sys.argv) > 4 else "judge")
    elif cmd == "eval":
        evaluate(rec)
    else:
        raise SystemExit(__doc__)
