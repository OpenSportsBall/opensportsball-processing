#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4",
#                 "rfdetr>=1.11"]
# ///
"""Phase 0 spike: players and ball from an off-the-shelf detector.

Run after phase0.py (audio offsets) and follow.py (cached cameras):

    uv run processing/spikes/detect.py footage/rec3 <segment> <start s> <seconds> [fps]

RF-DETR (Apache-2.0, COCO-trained: "person" and "sports ball") on tiles
of each phone's full-resolution frame: the ball is ~5 px wide at the far
side, so the frame can't be shrunk first. Detections from overlapping
tiles are merged, then mapped to world directions (yaw, pitch) with the
phones' cameras, and the two phones' views of the same object merged.

Writes <recording>/detect/: detections.json (per frame), a timeline of
ball candidates by yaw over time (a ball in play should draw a continuous
trail; spare balls on the sideline draw straight lines) and a few frames
with detections drawn.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import accel  # noqa: E402
import follow  # noqa: E402
import phase0  # noqa: E402
import gpu  # noqa: E402

PERSON, BALL = 1, 37        # COCO class ids in RF-DETR's output
TILE, STRIDE = 640, 576     # 64 px overlap between tiles
BAND_TOP = 760              # first frame row tiled (4K); above it is sky and trees
THRESH = {PERSON: 0.35, BALL: 0.15}
MERGE_DEG = {PERSON: 0.6, BALL: 0.4}  # same object seen by both phones


class CoreMLDetector:
    """RF-DETR run by Core ML instead of PyTorch, with predict()'s
    preprocessing (resize without antialias, ImageNet normalisation) and
    rfdetr's export decoder, so it is a drop-in for `model.predict` on a
    list of crops: 3xHxW tensors in [0, 1] (ballwin.py) or HxWx3 uint8 RGB
    arrays (teams.py). `folder` holds `*b8*/*.mlpackage` and optionally
    `*b1*/*.mlpackage` (rfdetr's `export(format="coreml", batch_size=...,
    coreml_precision="float32")`; fp16 changes the detections). Class ids
    come out as predict()'s: 0-based for a fine-tuned checkpoint, COCO ids
    for the stock one. Needs coremltools (in ballwin.py's and roles.py's
    dependencies on macOS); `coreml_detector` picks or makes the exports.
    Measured 2026-11-08: footage/onnxbench, BACKLOG.md."""

    def __init__(self, folder: Path, rf):
        import coremltools as ct
        from rfdetr.export._runtime.decode import decode_detections
        self.decode = decode_detections
        self.res, self.means, self.stds = rf.model.resolution, rf.means, rf.stds
        self.units = os.environ.get("OSB_COREML_UNITS", "CPU_AND_GPU")  # coremltools ComputeUnit name
        self.models = {}
        for b in (1, 8):
            found = sorted(folder.glob(f"*b{b}*/*.mlpackage"))
            if not found:
                continue
            m = ct.models.MLModel(str(found[0]), compute_units=getattr(ct.ComputeUnit, self.units))
            d = m.get_spec().description
            self.models[b] = (m, d.input[0].name, [o.name for o in d.output])
            print(f"Core ML detector, batch {b}: {found[0]} on {self.units}", flush=True)
        if 8 not in self.models:
            raise FileNotFoundError(f"no *b8*/*.mlpackage in {folder}")

    def _run(self, b: int, x: np.ndarray):
        m, inp, outs = self.models[b]
        r = m.predict({inp: x})
        a = [np.asarray(r[o], dtype=np.float32) for o in outs]
        return (a[0], a[1]) if a[0].shape[-1] == 4 else (a[1], a[0])  # (boxes cxcywh, logits)

    def predict(self, crops, threshold):
        import torchvision.transforms.functional as TF
        crops = [torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).float() / 255
                 if isinstance(c, np.ndarray) else c for c in crops]
        x = torch.stack([TF.resize(c, [self.res, self.res], antialias=False) for c in crops])
        x = TF.normalize(x, self.means, self.stds).float().cpu().numpy()
        out, i = [], 0
        while i < len(x):
            left = len(x) - i
            # batch 8 costs ~8 x 21.8 ms even when padded; batch 1 ~25.8 ms a crop
            b, k = (8, min(8, left)) if left > 6 or 1 not in self.models else (1, 1)
            chunk = x[i:i + k]
            if k < b:
                chunk = np.concatenate([chunk, np.zeros((b - k, *chunk.shape[1:]), np.float32)])
            boxes, logits = self._run(b, chunk)
            for j in range(k):
                h, w = crops[i + j].shape[1:]
                out.append(self.decode(boxes[j], logits[j], (w, h), threshold, background_class_id=None))
            i += k
        return out


def coreml_detector(rf, env: str, make, ckpt: Path | None = None, batches=(1, 8)):
    """`rf` run by Core ML (CoreMLDetector) where that is the default, else
    `rf` itself. The environment variable `env` decides: "0" keeps PyTorch,
    a folder uses the exports there, unset (the default) uses Core ML on a
    Mac with fp32 exports cached in ~/Library/Caches/opensportsball/coreml,
    made on first use from `make()` (a fresh model of `ckpt`, None for stock;
    ~10 s a batch size) and keyed by the checkpoint and rfdetr's version.
    Anything failing falls back to PyTorch with a message: same detections
    either way (BACKLOG.md, 2026-11-08), Core ML ~30% faster per stage."""
    v = os.environ.get(env, "")
    if v == "0" or (not v and not accel.coreml()):
        return rf
    try:
        if v:
            folder = Path(v)
        else:
            from importlib.metadata import version
            ck = Path(ckpt) if ckpt else None  # None: the stock COCO model
            key = (f"{ck.stem}-{ck.stat().st_size}-{int(ck.stat().st_mtime)}" if ck else "stock-medium") \
                + f"-rfdetr{version('rfdetr')}-fp32"
            folder = Path.home() / "Library/Caches/opensportsball/coreml" / key
            for b in batches:
                if not list(folder.glob(f"b{b}/*.mlpackage")):
                    import shutil, tempfile
                    folder.mkdir(parents=True, exist_ok=True)
                    tmp = Path(tempfile.mkdtemp(dir=folder))
                    m, t = make(), time.time()
                    r = m.model.resolution
                    m.export(output_dir=str(tmp), format="coreml", batch_size=b, shape=(r, r),
                             coreml_precision="float32", verbose=False)
                    shutil.rmtree(folder / f"b{b}", ignore_errors=True)
                    tmp.rename(folder / f"b{b}")
                    print(f"Core ML export, batch {b}: {time.time() - t:.0f} s -> {folder / f'b{b}'}", flush=True)
                    del m
        return CoreMLDetector(folder, rf)
    except Exception as e:  # noqa: BLE001 -- PyTorch gives the same detections, only slower
        print(f"Core ML unavailable ({type(e).__name__}: {e}); using PyTorch", flush=True)
        return rf


def nv12_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """NV12 planes (1x1xHxW, 1x2xH/2xW/2) -> RGB 3xHxW in [0, 1]."""
    h, w = y.shape[-2:]
    uv = F.interpolate(uv.float(), size=(h, w), mode="bilinear", align_corners=False)[0]
    bgr = gpu.nv_to_bgr(y[0, 0].float(), uv)
    return bgr.flip(-1).permute(2, 0, 1).float() / 255


def nv12_crop_to_rgb(y: torch.Tensor, uv: torch.Tensor, x: int, y0: int, w: int, h: int) -> torch.Tensor:
    """nv12_to_rgb(y, uv)[:, y0:y0 + h, x:x + w] without converting the rest
    of the frame: the chroma is upsampled from a region one chroma pixel
    wider on each side (edge-clamped like the whole-frame upsample)."""
    fh, fw = y.shape[-2:]
    ux0, uy0 = x // 2 - 1, y0 // 2 - 1
    ux1, uy1 = (x + w + 1) // 2 + 1, (y0 + h + 1) // 2 + 1
    ix = torch.arange(ux0, ux1, device=uv.device).clamp(0, fw // 2 - 1)
    iy = torch.arange(uy0, uy1, device=uv.device).clamp(0, fh // 2 - 1)
    sub = uv.float()[:, :, iy][:, :, :, ix]
    up = F.interpolate(sub, scale_factor=2, mode="bilinear", align_corners=False)[0]
    ox, oy = x - 2 * ux0, y0 - 2 * uy0
    bgr = gpu.nv_to_bgr(y[0, 0, y0:y0 + h, x:x + w].float(), up[:, oy:oy + h, ox:ox + w])
    return bgr.flip(-1).permute(2, 0, 1).float() / 255


def tiles(h: int, w: int):
    ys = list(range(BAND_TOP, h - TILE, 530)) + [h - TILE]
    xs = list(range(0, w - TILE, STRIDE)) + [w - TILE]
    return [(x, y) for y in ys for x in xs]


def nms(boxes: np.ndarray, iou: float = 0.5, contain: float = 0.8) -> np.ndarray:
    """Greedy suppression on (x1, y1, x2, y2, score) rows: drops boxes that
    overlap a better one, or sit mostly inside it (a tile edge cut it)."""
    keep = []
    for i in np.argsort(-boxes[:, 4]):
        b = boxes[i]
        ok = True
        for k in keep:
            a = boxes[k]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = (a[2] - a[0]) * (a[3] - a[1]) + area_b - inter
            if inter / union > iou or inter / max(area_b, 1e-6) > contain:
                ok = False
                break
        if ok:
            keep.append(i)
    return boxes[keep]


def to_angles(cam: follow.Camera, pts: np.ndarray) -> np.ndarray:
    d = cam.unproject(pts)
    return np.degrees(np.c_[np.arctan2(d[:, 0], d[:, 2]), np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))])


def detect_frame(model, frames_rgb, cams):
    """Detections for one frame pair, merged across tiles and phones."""
    found = []
    for phone, (img, cam) in enumerate(zip(frames_rgb, cams)):
        h, w = img.shape[-2:]
        tl = tiles(h, w)
        res = model.predict([img[:, y:y + TILE, x:x + TILE] for x, y in tl],
                            threshold=min(THRESH.values()))
        for cls in (PERSON, BALL):
            rows = [np.r_[box + [x, y, x, y], sc]
                    for (x, y), det in zip(tl, res)
                    for box, c, sc in zip(det.xyxy, det.class_id, det.confidence)
                    if c == cls and sc >= THRESH[cls]]
            if not rows:
                continue
            for x1, y1, x2, y2, sc in nms(np.array(rows)):
                # People are placed by their feet; the ball by its centre.
                anchor = [(x1 + x2) / 2, y2 if cls == PERSON else (y1 + y2) / 2]
                yaw, pitch = to_angles(cam, np.array([anchor]))[0]
                size = float(np.degrees((y2 - y1) / cam.f))
                found.append({"cls": "person" if cls == PERSON else "ball", "phone": phone,
                              "yaw": round(float(yaw), 3), "pitch": round(float(pitch), 3),
                              "conf": round(float(sc), 3), "sizeDeg": round(size, 3),
                              "box": [round(float(v)) for v in (x1, y1, x2, y2)]})
    # The same object seen by both phones in the overlap: keep the surer one.
    merged = []
    for d in sorted(found, key=lambda d: -d["conf"]):
        if not any(m["cls"] == d["cls"] and m["phone"] != d["phone"]
                   and np.hypot(m["yaw"] - d["yaw"], m["pitch"] - d["pitch"]) < MERGE_DEG[
                       PERSON if d["cls"] == "person" else BALL]
                   for m in merged):
            merged.append(d)
    return merged


def main(rec: Path, seg: int, start: float, seconds: float, fps: float) -> None:
    import rfdetr

    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    files = [rec / devs["main"]["segments"][seg]["objectKey"],
             rec / devs["secondary"]["segments"][seg]["objectKey"]]
    offsets = [0.0, report["segments"][seg]["offsetS"]]
    out = rec / "detect"
    out.mkdir(exist_ok=True)

    dev = gpu.device()
    model = rfdetr.RFDETRMedium()
    W, H = follow.SRC_W, follow.SRC_H
    streams = [follow.prefetch(follow.frame_stream(f, start + o, seconds, W, fps=fps))
               for f, o in zip(files, offsets)]
    frames_out, t_det, n = [], 0.0, 0
    snapshots = {int(seconds * fps * k / 4) for k in (1, 2, 3)}
    for i, (fa, fb) in enumerate(zip(*streams)):
        rgb = []
        for buf, fl in ((fa, flip[0]), (fb, flip[1])):
            img = nv12_to_rgb(*gpu.upload_nv12(buf, W, H, dev))
            rgb.append(img.flip(1, 2) if fl else img)  # detectors expect people upright
        t0 = time.perf_counter()
        dets = detect_frame(model, rgb, cams)
        t_det += time.perf_counter() - t0
        n += 1
        frames_out.append({"t": round(start + i / fps, 3), "detections": dets})
        if i in snapshots:
            vis = [(im.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy() for im in rgb]
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                col = (0, 255, 0) if d["cls"] == "person" else (0, 0, 255)
                pad = 0 if d["cls"] == "person" else 10
                cv2.rectangle(vis[d["phone"]], (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, 3)
                if d["cls"] == "ball":
                    cv2.putText(vis[d["phone"]], f"{d['conf']:.2f}", (x1, y1 - 16), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, col, 3)
            cv2.imwrite(str(out / f"frame_{i:04d}.jpg"),
                        cv2.resize(np.hstack([vis[1], vis[0]]), (2 * 1280, 720)))
        if i % 50 == 0:
            print(f"frame {i}: {sum(d['cls'] == 'person' for d in dets)} people,"
                  f" {sum(d['cls'] == 'ball' for d in dets)} ball candidates", flush=True)
    (out / "detections.json").write_text(json.dumps({"segment": seg, "fps": fps, "frames": frames_out}))
    print(f"{n} frame pairs, detection {1000 * t_det / max(n, 1):.0f} ms per pair")
    timeline(frames_out, out / "timeline.png", fps)


def timeline(frames, path: Path, fps: float, px_per_deg: int = 6, px_per_frame: int = 3) -> None:
    """Yaw across, time down: people in grey, ball candidates by confidence
    (brighter red is surer)."""
    y0, y1 = -100, 100
    img = np.full((len(frames) * px_per_frame + 20, (y1 - y0) * px_per_deg, 3), 255, np.uint8)
    for i, f in enumerate(frames):
        row = 10 + i * px_per_frame
        for d in f["detections"]:
            x = int((d["yaw"] - y0) * px_per_deg)
            if d["cls"] == "person":
                cv2.circle(img, (x, row), 1, (190, 190, 190), -1)
        for d in f["detections"]:
            if d["cls"] == "ball":
                x = int((d["yaw"] - y0) * px_per_deg)
                c = int(255 * (1 - min(1.0, d["conf"] / 0.6)))
                cv2.circle(img, (x, row), 3, (c, c, 255), -1)
    for deg in range(y0, y1 + 1, 20):
        x = (deg - y0) * px_per_deg
        cv2.line(img, (x, 0), (x, 8), (0, 0, 0), 1)
        cv2.putText(img, f"{deg}", (x + 2, 8), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
         float(sys.argv[5]) if len(sys.argv) > 5 else 10.0)
