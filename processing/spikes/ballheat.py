#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4", "pyyaml>=6"]
# ///
"""Multi-frame heatmap ball detectors (WASB, TrackNetV2, TrackNetV3) tested
at a clip's ball labels, against the fine-tuned RF-DETR's candidates.

    uv run processing/spikes/ballheat.py footage/rec3 <set> [wasb] [tracknetv2] [tracknetv3]

These models take a few consecutive frames stacked as channels and return
a heatmap of the ball per frame, so they can use motion, not just looks.
All were trained at 512x288 on whole broadcast frames; here each phone's
4K frame is cut into 512x288 tiles at native resolution (far balls, 5-10
px, as the models saw them) and at 2x and 4x downscale (near balls), see
SCALES. Every heatmap blob above PEAK_MIN is a candidate with the blob's
peak as its score.

At each labelled frame (labelset.py clip, both phones) a detector is
"found" when a candidate lies within HIT_PX or 1.5 ball diameters of the
label, at a given score threshold; the price is candidates per frame.
The same measure is printed for ballclip.py's full scans (RF-DETR) and
ballwin.py's search windows, from their candidate files.

Models and weights (MIT) live under datasets/ (gitignored):
    git clone https://github.com/nttcom/WASB-SBDT datasets/wasb
    uvx gdown 1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l -O datasets/wasb/pretrained_weights/wasb_soccer_best.pth.tar
    uvx gdown 1yTCC_8cWnMCLOpErk6UpepzQD9UF8Wpi -O datasets/wasb/pretrained_weights/tracknetv2_soccer_best.pth.tar
    git clone https://github.com/qaz812345/TrackNetV3 datasets/tracknetv3
    uvx gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA -O TrackNetV3_ckpts.zip && unzip TrackNetV3_ckpts.zip -d datasets/tracknetv3
The WASB soccer weights were trained on ISSIA-CNR (six static 1080p
cameras on a pitch, resized to 512x288); TrackNetV3's on badminton.

Label sets made before 2026-10-18 grabbed their images with
phase0.frames_at, which returned a frame 0.2-0.9 s after the asked time
(the decoder's first frames after a seek are missing). Each label image
is therefore matched by pixel difference to a window decoded with a
preroll, and the matched frame (with its neighbours) is what the models
see; the RF-DETR baselines are read at the matched time as well as the
nominal one. The measured offset is written to items.json as "dtS" on
each frame (label time - nominal time; balleval.py reads it), also by

    uv run processing/spikes/ballheat.py footage/rec3 <set> dt

which measures or collects the offsets without running a model. Sets
made since carry dtS 0 from labelset.py (frames grabbed with a preroll).

Candidates per labelled frame are cached in <set>/heat_<model>.json
({frame id: {"dt": label time - nominal time, "cands": [[x, y, score, scale], ...]}}).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import balljudge  # noqa: E402
import gpu  # noqa: E402
import phase0  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WASB = ROOT / "datasets" / "wasb"
TNV3 = ROOT / "datasets" / "tracknetv3"

IN_W, IN_H = 512, 288
SCALES = [int(s) for s in os.environ.get("OSB_HEAT_SCALES", "1,2,4").split(",")]
BAND_TOP = 760          # as detect.py: above this row is sky and trees
OVERLAP = 32            # tile overlap in model pixels
PEAK_MIN = 0.05         # heatmap blobs above this are candidates
HIT_PX = 12.0           # a hit: within this or 1.5 ball diameters of the label
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
BATCH = 16
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class AttrDict(dict):
    """HRNet reads its config both as cfg['x'] and cfg.MODEL.EXTRA."""
    __getattr__ = dict.__getitem__


def attrdict(d):
    return AttrDict({k: attrdict(v) for k, v in d.items()}) if isinstance(d, dict) else d


class Heat:
    """A heatmap model: `frames_in` consecutive RGB frames in, one heatmap
    (for frame `pick` of the window) out, per 512x288 tile."""

    def __init__(self, name: str, dev: torch.device):
        self.name, self.dev = name, dev
        self.bg = False
        if name in ("wasb", "tracknetv2"):
            import yaml
            sys.path.insert(0, str(WASB / "src"))
            self.frames_in, self.pick, self.norm = 3, 1, "imagenet"
            if name == "wasb":
                from models.hrnet import HRNet
                cfg = yaml.safe_load((WASB / "src" / "configs" / "model" / "wasb.yaml").read_text())
                self.model = HRNet(attrdict(cfg))
            else:
                from models.unet2d import TrackNetV2
                self.model = TrackNetV2(9, 3, bilinear=True, halve_channel=False)
            ck = torch.load(WASB / "pretrained_weights" / f"{name}_soccer_best.pth.tar", map_location="cpu",
                            weights_only=False)
            self.model.load_state_dict(ck["model_state_dict"])
        elif name == "tracknetv3":
            sys.path.insert(0, str(TNV3))
            from model import TrackNet
            ck = torch.load(TNV3 / "ckpts" / "TrackNet_best.pt", map_location="cpu", weights_only=False)
            p = ck["param_dict"]
            assert p["bg_mode"] == "concat", p
            self.frames_in, self.pick, self.norm, self.bg = p["seq_len"], p["seq_len"] // 2, "unit", True
            self.model = TrackNet(in_dim=(self.frames_in + 1) * 3, out_dim=self.frames_in)
            self.model.load_state_dict(ck["model"])
        else:
            raise SystemExit(f"unknown model {name}")
        self.model.eval().to(dev)

    @torch.no_grad()
    def __call__(self, tiles: np.ndarray, bg: np.ndarray | None) -> np.ndarray:
        """tiles (N, frames_in, 288, 512, 3) RGB uint8, bg (N, 288, 512, 3) -> (N, 288, 512) heatmaps."""
        x = tiles.astype(np.float32) / 255.0
        if self.norm == "imagenet":
            x = (x - MEAN) / STD
        n = x.shape[0]
        x = x.transpose(0, 1, 4, 2, 3).reshape(n, -1, IN_H, IN_W)   # frames x channels stacked
        if self.bg:
            b = (bg.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
            x = np.concatenate([b, x], 1)
        out = []
        for i in range(0, n, BATCH):
            t = torch.from_numpy(x[i:i + BATCH]).to(self.dev)
            y = self.model(t)
            if isinstance(y, dict):          # WASB models return logits per output scale
                y = torch.sigmoid(y[0])
            out.append(y[:, self.pick].float().cpu().numpy())
        return np.concatenate(out)


def tile_grid(h: int, w: int, s: int):
    tw, th, ov = IN_W * s, IN_H * s, OVERLAP * s
    xs = list(range(0, w - tw, tw - ov)) + [w - tw]
    ys = list(range(BAND_TOP, h - th, th - ov)) + [h - th]
    return [(x, y) for y in ys for x in xs]


def peaks(hm: np.ndarray, x0: int, y0: int, s: int) -> list[list[float]]:
    """Blobs above PEAK_MIN -> [x, y, score, scale] in frame pixels."""
    mask = (hm > PEAK_MIN).astype(np.uint8)
    if not mask.any():
        return []
    n, lab = cv2.connectedComponents(mask)
    out = []
    for m in range(1, n):
        ys, xs = np.nonzero(lab == m)
        w = hm[ys, xs]
        out.append([x0 + float((xs * w).sum() / w.sum()) * s, y0 + float((ys * w).sum() / w.sum()) * s,
                    float(w.max()), s])
    return out


def merge(cands: list[list[float]]) -> list[list[float]]:
    kept = []
    for c in sorted(cands, key=lambda c: -c[2]):
        if not any(np.hypot(c[0] - k[0], c[1] - k[1]) < 8 * max(c[3], k[3]) for k in kept):
            kept.append(c)
    return kept


def detect(heat: Heat, frames: list[np.ndarray], bg: np.ndarray | None) -> tuple[list[list[float]], int]:
    """All candidates in a window of consecutive BGR frames; the number of tiles run."""
    h, w = frames[0].shape[:2]
    rgb = np.stack([f[..., ::-1] for f in frames])
    cands, n_tiles = [], 0
    for s in SCALES:
        grid = tile_grid(h, w, s)
        tiles, bgs = [], []
        for x, y in grid:
            crop = rgb[:, y:y + IN_H * s, x:x + IN_W * s]
            if s > 1:
                crop = np.stack([cv2.resize(c, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for c in crop])
            tiles.append(crop)
            if bg is not None:
                b = bg[y:y + IN_H * s, x:x + IN_W * s]
                bgs.append(cv2.resize(b, (IN_W, IN_H), interpolation=cv2.INTER_AREA) if s > 1 else b)
        hms = heat(np.stack(tiles), np.stack(bgs) if bgs else None)
        for (x, y), hm in zip(grid, hms):
            cands += peaks(hm, x, y, s)
        n_tiles += len(grid)
    return merge(cands), n_tiles


PRE_S = 1.0             # decode this long before the nominal time ...
WINDOW_N = int(os.environ.get("OSB_WINDOW_N", 60))  # ... and this many frames at 30 fps; the label frame is found among them
                        # (a run with a longer window fills in frames an earlier run skipped)


def decode_window(path: Path, t: float, flip: bool, ref: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    """Frames at 30 fps from t - PRE_S; the index of the one that is the
    label image; and the index where real frames begin (after a seek the
    decoder's first frames are one frame repeated)."""
    frames = phase0.frames_at(path, max(0.0, t - PRE_S), WINDOW_N, 30, 3840, flip)
    small = lambda im: cv2.resize(im[..., 1], (480, 270), interpolation=cv2.INTER_AREA).astype(np.int16)
    sm = [small(f) for f in frames]
    d = [float(np.abs(x - small(ref)).mean()) for x in sm]
    moving = [i for i in range(1, len(sm)) if np.abs(sm[i] - sm[i - 1]).mean() > 0.05]
    return frames, int(np.argmin(d)), (moving[0] - 1) if moving else len(frames)


def run_models(rec: Path, names: list[str], items: list[dict], devs: dict, report: dict, flip: dict,
               lab: Path) -> dict[str, dict]:
    """Candidates per model per labelled frame, from the cache where it has
    them; the frames a model's cache lacks are computed and added."""
    results = {}
    for name in names:
        cache = lab / f"heat_{name}.json"
        results[name] = json.loads(cache.read_text()) if cache.exists() and not os.environ.get("OSB_REDO") else {}
    todo = [name for name in names
            if any(f["id"] not in results[name] for it in items for f in it["frames"])]
    if not todo:
        return results
    heats = {name: Heat(name, gpu.device()) for name in todo}
    t0, tiles, n_frames, dts = time.time(), 0, 0, []
    for n, it in enumerate(items):
        for f in it["frames"]:
            need = [name for name in todo if f["id"] not in results[name]]
            if not need:
                continue
            role = f["phone"]
            seg = it["segment"]
            off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
            path = rec / devs[role]["segments"][seg]["objectKey"]
            t = it["t"] + off
            ref = cv2.imread(str(lab / f["image"]))
            frames, idx, real0 = decode_window(path, t, flip[role], ref)
            dt = idx / 30 - PRE_S
            if idx - 4 < real0 or idx > len(frames) - 5:
                print(f"{f['id']}: label frame at index {idx} of {len(frames)} (real frames from {real0}), skipped",
                      flush=True)
                continue
            dts.append(dt)
            bg = np.median(np.stack([fr[..., ::-1] for fr in frames[::6]]), 0).astype(np.uint8)
            for name in need:
                heat = heats[name]
                win = frames[idx - heat.pick: idx - heat.pick + heat.frames_in]
                cands, nt = detect(heat, win, bg if heat.bg else None)
                tiles += nt
                results[name][f["id"]] = {
                    "dt": round(dt, 3), "cands": [[round(c[0], 1), round(c[1], 1), round(c[2], 3), c[3]] for c in cands]}
            n_frames += 1
        if (n + 1) % 10 == 0 and n_frames:
            el = time.time() - t0
            print(f"{n + 1}/{len(items)} moments, {el / 60:.1f} min, {tiles / n_frames / len(heats):.0f}"
                  f" tiles/frame/model, label frames {np.mean(dts):+.2f} s from nominal", flush=True)
    for name in todo:
        (lab / f"heat_{name}.json").write_text(json.dumps(results[name]))
    print(f"{n_frames} frames x {len(todo)} models in {(time.time() - t0) / 60:.1f} min", flush=True)
    return results


def label_offsets(rec: Path, lab: Path, items: list[dict], devs: dict, report: dict, flip: dict) -> dict[str, float]:
    """Each label image's offset from its nominal time, as "dtS" on the frame
    in items.json: kept where the set has it, else taken from the
    heat_*.json caches, else measured by matching the image against a
    window decoded with a preroll. items.json is updated and the offsets
    returned per frame id."""
    dts = {f["id"]: f["dtS"] for it in items for f in it["frames"] if "dtS" in f}
    for cache in sorted(lab.glob("heat_*.json")):
        for fid, v in json.loads(cache.read_text()).items():
            if isinstance(v, dict) and "dt" in v:
                dts.setdefault(fid, v["dt"])
    for it in items:
        for f in it["frames"]:
            if f["id"] in dts:
                continue
            role, seg = f["phone"], it["segment"]
            off = 0.0 if role == "main" else report["segments"][seg]["offsetS"]
            frames, idx, real0 = decode_window(rec / devs[role]["segments"][seg]["objectKey"], it["t"] + off,
                                               flip[role], cv2.imread(str(lab / f["image"])))
            if idx <= real0 or idx >= len(frames) - 1:
                print(f"{f['id']}: label frame at the edge of the decoded window (index {idx} of {len(frames)},"
                      f" real frames from {real0}), offset not recorded", flush=True)
                continue
            dts[f["id"]] = round(idx / 30 - PRE_S, 3)
    changed = False
    for it in items:
        for f in it["frames"]:
            if f["id"] in dts and f.get("dtS") != dts[f["id"]]:
                f["dtS"], changed = dts[f["id"]], True
    if changed:
        (lab / "items.json").write_text(json.dumps(items, indent=1))
    n = sum(len(it["frames"]) for it in items)
    if dts:
        by = {r: [d for i, d in dts.items() if i.endswith("_" + r)] for r in ("main", "secondary")}
        print(f"label frames are {np.mean(list(dts.values())):+.2f} s from their nominal time"
              f" (main {np.mean(by['main']):+.2f}, secondary {np.mean(by['secondary']):+.2f};"
              f" {len(dts)}/{n} frames{', items.json updated' if changed else ''})")
    return dts


def cands_files(rec: Path, seg0: int, start: float, tag: str) -> list[Path]:
    """The clip's candidate files with exactly this tag ("" is ballclip.py's
    full scan, "win" ballwin.py's windows): ball_<seg>_<start>_<seconds>[.<tag>].json."""
    pat = re.compile(rf"ball_{seg0}_{start:g}_[\d.]+?{re.escape('.' + tag) if tag else ''}\.json$")
    return sorted(p for p in (rec / "detect").glob(f"ball_{seg0}_{start:g}_*.json") if pat.fullmatch(p.name))


def rfdetr_cands(rec: Path, items: list[dict], seg0: int, start: float, tag: str, dts: dict) -> dict | None:
    """ballclip.py / ballwin.py candidates at each labelled frame: [x, y, conf, judge].
    `dts` shifts a frame's time by the label's measured offset (else nominal)."""
    paths = cands_files(rec, seg0, start, tag)
    if not paths:
        return None
    data = json.loads(paths[0].read_text())
    out = {}
    for it in items:
        for f in it["frames"]:
            k = int(round((it["clipT"] + dts.get(f["id"], 0.0)) * data["fps"]))
            if k >= len(data["frames"]):
                continue
            phone = 0 if f["phone"] == "main" else 1
            out[f["id"]] = [[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2, d["conf"],
                             d.get("judge", 1.0)]
                            for d in data["frames"][k]["detections"] if d["phone"] == phone]
    return out


def score(name: str, cands: dict, targets: list[tuple], all_ids: list[str], rules: list[tuple[str, callable]]):
    ids = [i for i in all_ids if i in cands]
    print(f"\n{name}: candidates at {len(ids)}/{len(all_ids)} labelled frames")
    buckets = [("<8px", 0, 8), ("8-16", 8, 16), (">=16", 16, 1e9)]
    for label, keep in rules:
        found = {b[0]: [0, 0] for b in buckets}
        for fid, x, y, diam in targets:
            if fid not in cands:
                continue
            r = max(HIT_PX, 1.5 * diam)
            hit = any(np.hypot(c[0] - x, c[1] - y) <= r for c in cands[fid] if keep(c))
            for b, lo, hi in buckets:
                if lo <= diam < hi:
                    found[b][1] += 1
                    found[b][0] += hit
        n_t = sum(v[1] for v in found.values())
        n_f = sum(v[0] for v in found.values())
        per_frame = np.mean([sum(keep(c) for c in cands[i]) for i in ids]) if ids else 0
        by = "  ".join(f"{b} {v[0]}/{v[1]}" for b, v in found.items())
        print(f"  {label:22s} found {n_f:2d}/{n_t} ({100 * n_f / max(n_t, 1):3.0f}%)   {by}   "
              f"{per_frame:6.1f} cands/frame")


def main(rec: Path, name: str, models: list[str]) -> None:
    lab = rec / f"labels_{name}"
    items = json.loads((lab / "items.json").read_text())
    labels = json.loads((lab / "labels.json").read_text())
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    flip = {"main": report["upsideDown"]["main"], "secondary": report["upsideDown"]["secondary"]}
    seg0, start = int(items[0]["id"].split("_")[0][1:]), float(items[0]["id"].split("_")[1])
    diam_at = balljudge.expected_at(rec)

    # Targets: labelled frames with the match ball in play; all frames with any label (for the false-alarm rate).
    targets, all_ids = [], []
    for it in items:
        for f in it["frames"]:
            v = labels.get(f["id"])
            if not v:
                continue
            all_ids.append(f["id"])
            for b in v.get("balls", []):
                if b["inPlay"] and v["status"] == "ball":
                    targets.append((f["id"], b["x"], b["y"], float(diam_at(f["phone"], b["x"], b["y"]))))
                    break
    print(f"{name}: {len(all_ids)} labelled frames, {len(targets)} with the match ball in play;"
          f" expected diameter median {np.median([t[3] for t in targets]):.1f} px, "
          f"{sum(t[3] < 8 for t in targets)} under 8 px")

    results = run_models(rec, models, items, devs, report, flip, lab)
    dts = label_offsets(rec, lab, items, devs, report, flip)
    if not models:
        return
    rules = [("conf>=0.1", lambda c: c[2] >= 0.1), ("conf>=0.3", lambda c: c[2] >= 0.3),
             ("conf>=0.3 judge>=0.3", lambda c: c[2] >= 0.3 and c[3] >= 0.3)]
    for tag, label in (("", "rfdetr full scan (ballclip)"), (".win", "rfdetr windows (ballwin)")):
        for when, d in (("nominal t", {}), ("label-matched t", dts)):
            c = rfdetr_cands(rec, items, seg0, start, tag, d)
            if c:
                score(f"{label}, {when}", c, targets, all_ids, rules)
    for m, res in results.items():
        c = {fid: v["cands"] for fid, v in res.items()}
        score(m, c, targets, all_ids, [(f"score>={th}", (lambda th: lambda c: c[2] >= th)(th)) for th in THRESHOLDS])
        for s in SCALES:
            score(f"{m} scale {s} only", {k: [c for c in v if c[3] == s] for k, v in c.items()}, targets, all_ids,
                  [("score>=0.3", lambda c: c[2] >= 0.3)])


if __name__ == "__main__":
    args = sys.argv[3:] or ["wasb"]
    main(Path(sys.argv[1]), sys.argv[2], [] if args == ["dt"] else args)
