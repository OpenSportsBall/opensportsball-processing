#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.10"]
# ///
"""Phase 0 spike: audio sync and a two-phone stitch from one recording.

Run on a recording downloaded from S3 (manifest.json plus <deviceId>/ dirs):

    uv run processing/spikes/phase0.py footage/rec1 [preview start, s]

Syncs each segment pair by audio, calibrates a stitch from segment 0, and
writes report.json, a stitched still and a short stitched preview (from
the given Main time in segment 0) to <recording>/phase0/. Prototype only;
the pipeline proper goes in osb_processing once this works on real games.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FS = 48_000          # audio sample rate used for sync
PREVIEW_W = 960      # width of each phone's frame in the preview
PREVIEW_SECONDS = 12
CALIB_FRAMES = 12    # frame pairs sampled for the stitch


# ---------------------------------------------------------------- media io

def probe_start(path: Path, kind: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", f"{kind}:0",
         "-show_entries", "stream=start_time", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def load_audio(path: Path) -> np.ndarray:
    """Mono float32 at FS, sample 0 at file time 0 (audio start_time is 0)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(FS), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def frames_at(path: Path, start: float, count: int, fps: float,
              width: int) -> list[np.ndarray]:
    """`count` BGR frames resampled to constant `fps` from file time `start`."""
    height = round(width * 9 / 16 / 2) * 2
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-i", str(path),
         "-vf", f"fps={fps},scale={width}:{height}", "-frames:v", str(count),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True).stdout
    n = len(raw) // (width * height * 3)
    return list(np.frombuffer(raw[: n * width * height * 3], np.uint8)
                .reshape(n, height, width, 3))


# -------------------------------------------------------------------- sync

def gcc_phat(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Lag k (samples, sub-sample) with a[n] ~ b[n - k], and peak sharpness.

    Generalized cross-correlation with partial phase transform (PHAT-beta),
    limited to 150 Hz - 8 kHz. Whitening bins outside that band, where AAC
    leaves almost nothing, creates a false peak at zero lag. Sharpness is
    the peak over the next-best value outside +-5 ms; above ~3 is a clean
    lock.
    """
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    r = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    f = np.fft.rfftfreq(n, 1 / FS)
    r = np.where((f >= 150) & (f <= 8000), r / (np.abs(r) + 1e-20) ** 0.8, 0)
    cc = np.fft.irfft(r, n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    i = int(np.argmax(cc))
    frac = 0.0
    if 0 < i < len(cc) - 1:  # parabolic interpolation
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    guard = int(0.005 * FS)
    rest = np.concatenate([cc[: max(0, i - guard)], cc[i + guard:]])
    sharpness = float(cc[i] / (np.max(np.abs(rest)) + 1e-12))
    return float(i - max_lag + frac), sharpness


def sync(main_audio: np.ndarray, sec_audio: np.ndarray, guess_s: float) -> dict:
    """Offset d (s): an event at Main file time t is at Secondary time t + d."""
    a, b = main_audio, sec_audio
    # Whole clip, searching +-2 s around the BLE guess.
    shift = int(round(guess_s * FS))
    b_aligned = b[shift:] if shift >= 0 else np.concatenate([np.zeros(-shift, np.float32), b])
    k, sharp = gcc_phat(a, b_aligned[: len(a)], int(2 * FS))
    d = guess_s - k / FS

    # 4 s windows every 2 s to see drift between the two phones' clocks.
    win, hop = 4 * FS, 2 * FS
    windows = []
    for start in range(0, len(a) - win, hop):
        j = start + int(round(d * FS))
        if j < 0 or j + win > len(b):
            continue
        kw, sw = gcc_phat(a[start:start + win], b[j:j + win], int(0.05 * FS))
        windows.append({"mainT": round(start / FS + 2, 2),
                        "offsetS": round(d - kw / FS, 6),
                        "sharpness": round(sw, 2)})
    # Drop weak windows and any that locked onto a different peak.
    good = [w for w in windows
            if w["sharpness"] >= 3 and abs(w["offsetS"] - d) < 0.002]
    drift_ppm = None
    if len(good) >= 3:
        t = np.array([w["mainT"] for w in good])
        o = np.array([w["offsetS"] for w in good])
        drift_ppm = float(np.polyfit(t, o, 1)[0] * 1e6)
    return {"offsetS": round(d, 6), "sharpness": round(sharp, 2),
            "windows": windows, "windowsUsed": len(good),
            "driftPpm": None if drift_ppm is None else round(drift_ppm, 1)}


# ------------------------------------------------------------------ stitch

def to_cylinder(img: np.ndarray, fov_deg: float):
    """Project onto a cylinder whose radius is the focal length in pixels."""
    h, w = img.shape[:2]
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    cx, cy = w / 2, h / 2
    theta_max = np.arctan(cx / f)
    out_w = int(2 * f * theta_max)
    xs = (np.arange(out_w) - out_w / 2) / f              # theta
    ys = np.arange(h) - cy                                # height on cylinder
    theta, hh = np.meshgrid(xs, ys)
    map_x = (f * np.tan(theta) + cx).astype(np.float32)
    map_y = (hh / np.cos(theta) + cy).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y,
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, mask


def match(a: np.ndarray, b: np.ndarray, sift) -> tuple[np.ndarray, np.ndarray]:
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < 0.7 * n.distance]
    pb = np.float32([kb[m.queryIdx].pt for m in good])
    pa = np.float32([ka[m.trainIdx].pt for m in good])
    return pb, pa


def calibrate(main_frames, sec_frames, main_fov, sec_fov) -> dict:
    """Homography mapping Secondary's cylinder image into Main's.

    Pools matches from every sampled pair so moving players don't dominate,
    then one RANSAC fit.
    """
    sift = cv2.SIFT_create(nfeatures=4000)
    src, dst = [], []
    for m, s in zip(main_frames, sec_frames):
        cm, _ = to_cylinder(m, main_fov)
        cs, _ = to_cylinder(s, sec_fov)
        pb, pa = match(cm, cs, sift)
        src.append(pb)
        dst.append(pa)
    src, dst = np.concatenate(src), np.concatenate(dst)
    if len(src) < 8:
        return {"ok": False, "matches": int(len(src)), "inliers": 0}
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(inl.sum()) if inl is not None else 0
    ok = H is not None and inliers >= 40
    side = None
    if ok:
        # Where Secondary's centre lands relative to Main's centre.
        w = main_frames[0].shape[1]
        c = cv2.perspectiveTransform(np.float32([[[w / 2, w * 9 / 32]]]), H)[0, 0]
        side = "right" if c[0] > w / 2 else "left"
    return {"ok": bool(ok), "matches": int(len(src)), "inliers": inliers,
            "H": H.tolist() if H is not None else None,
            "secondarySide": side}


def compose(m, s, H, main_fov, sec_fov, gain=None):
    """Blend Main and the warped Secondary on a canvas covering both."""
    cm, mm = to_cylinder(m, main_fov)
    cs, ms = to_cylinder(s, sec_fov)
    H = np.asarray(H)
    h, w = cs.shape[:2]
    corners = cv2.perspectiveTransform(
        np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H).reshape(-1, 2)
    allpts = np.vstack([corners, [[0, 0], [cm.shape[1], cm.shape[0]]]])
    x0, y0 = np.floor(allpts.min(0)).astype(int)
    x1, y1 = np.ceil(allpts.max(0)).astype(int)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    size = (int(x1 - x0), int(y1 - y0))
    if size[0] > 6 * cm.shape[1] or size[1] > 4 * cm.shape[0]:
        raise ValueError(f"degenerate stitch, canvas {size}")
    wm = cv2.warpPerspective(cm, T, size)
    wmm = cv2.warpPerspective(mm, T, size)
    ws = cv2.warpPerspective(cs, T @ H, size)
    wms = cv2.warpPerspective(ms, T @ H, size)
    overlap = (wmm > 0) & (wms > 0)
    if gain is None:
        # Match Secondary's brightness to Main's in the overlap, per channel.
        gain = np.ones(3)
        if overlap.sum() > 1000:
            gain = (wm[overlap].mean(0) + 1) / (ws[overlap].mean(0) + 1)
    ws = np.clip(ws * gain, 0, 255).astype(np.uint8)
    # Feather: weight by distance to each image's edge.
    dm = cv2.distanceTransform(wmm, cv2.DIST_L2, 5)
    ds = cv2.distanceTransform(wms, cv2.DIST_L2, 5)
    alpha = (dm / (dm + ds + 1e-6))[..., None]
    out = (wm * alpha + ws * (1 - alpha)).astype(np.uint8)
    return out, gain, float(overlap.mean())


# -------------------------------------------------------------------- main

def ble_guess(manifest, main_seg, sec_seg, main_f, sec_f) -> float:
    """Offset d predicted by the BLE clock estimate, in seconds.

    startedAtDeviceMs is the first video frame, which sits at the video
    stream's start_time in the file; convert both to Main's clock.
    """
    off = (manifest.get("clockSync") or {}).get("secondaryOffsetMs", 0)
    main_t0 = main_seg["startedAtDeviceMs"] - 1000 * probe_start(main_f, "v")
    sec_t0 = sec_seg["startedAtDeviceMs"] - off - 1000 * probe_start(sec_f, "v")
    return (main_t0 - sec_t0) / 1000


class Encoder:
    """Pipes BGR frames to ffmpeg for an H.264 file any player opens."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int = 30):
        w, h = size[0] // 2 * 2, size[1] // 2 * 2
        self.size = (w, h)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        self.proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def main(rec: Path, preview_at: float | None) -> None:
    manifest = json.loads((rec / "manifest.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    main_d, sec_d = devs["main"], devs["secondary"]
    main_fov = main_d["video"]["horizontalFovDeg"]
    sec_fov = sec_d["video"]["horizontalFovDeg"]
    out = rec / "phase0"
    out.mkdir(exist_ok=True)
    report: dict = {"recording": manifest["sessionId"], "segments": []}

    # Sync each segment pair on its own: Android loses frames at every
    # segment switch, so a later pair's offset can differ from the first.
    pairs = list(zip(main_d["segments"], sec_d["segments"]))
    for ms, ss in pairs:
        mf_, sf_ = rec / ms["objectKey"], rec / ss["objectKey"]
        guess = ble_guess(manifest, ms, ss, mf_, sf_)
        s = sync(load_audio(mf_), load_audio(sf_), guess)
        seg = {"index": ms["index"], "bleOffsetS": round(guess, 4),
               "bleErrorMs": round(1000 * (guess - s["offsetS"]), 1), **s}
        report["segments"].append(seg)
        print(f"segment {ms['index']}: audio offset {s['offsetS'] * 1000:8.2f} ms"
              f" (sharpness {s['sharpness']}), BLE error {seg['bleErrorMs']} ms,"
              f" drift {s['driftPpm']} ppm, {s['windowsUsed']}/{len(s['windows'])} windows")

    # Stitch from segment 0: sample frame pairs across it, aligned by d.
    main_f = rec / pairs[0][0]["objectKey"]
    sec_f = rec / pairs[0][1]["objectKey"]
    d = report["segments"][0]["offsetS"]
    dur = min(pairs[0][0]["durationMs"], pairs[0][1]["durationMs"]) / 1000 - abs(d) - 1
    t_main0 = max(0.0, -d) + 0.5
    times = np.linspace(t_main0, t_main0 + dur - 0.5, CALIB_FRAMES)
    mf, sf = [], []
    for t in times:
        mf += frames_at(main_f, t, 1, 30, PREVIEW_W)
        sf += frames_at(sec_f, t + d, 1, 30, PREVIEW_W)
    cal = calibrate(mf, sf, main_fov, sec_fov)
    report["calibration"] = {k: v for k, v in cal.items() if k != "H"}
    print(f"stitch: {cal['inliers']} inliers of {cal['matches']} matches"
          f" -> {'ok' if cal['ok'] else 'FAILED'}"
          + (f", Secondary on the {cal['secondarySide']}" if cal["ok"] else ""))

    mid = len(mf) // 2
    cv2.imwrite(str(out / "side_by_side.jpg"), np.hstack([mf[mid], sf[mid]]))
    if cal["ok"]:
        try:
            pano, gain, ov = compose(mf[mid], sf[mid], cal["H"], main_fov, sec_fov)
            report["calibration"]["overlapFraction"] = round(ov, 3)
            cv2.imwrite(str(out / "stitched.jpg"), pano)
            n = int(min(PREVIEW_SECONDS, dur) * 30)
            t_prev = t_main0 if preview_at is None else min(max(preview_at, t_main0),
                                                            t_main0 + dur - n / 30)
            report["previewStartS"] = round(t_prev, 2)
            m_seq = frames_at(main_f, t_prev, n, 30, PREVIEW_W)
            s_seq = frames_at(sec_f, t_prev + d, n, 30, PREVIEW_W)
            enc = None
            for a, b in zip(m_seq, s_seq):
                p, _, _ = compose(a, b, cal["H"], main_fov, sec_fov, gain)
                if enc is None:
                    enc = Encoder(out / "stitched_preview.mp4", (p.shape[1], p.shape[0]))
                enc.write(p)
            if enc:
                enc.close()
        except ValueError as e:
            report["calibration"]["ok"] = False
            report["calibration"]["error"] = str(e)
            print(f"stitch: {e}")

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else None)
