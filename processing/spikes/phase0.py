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

Syncs each segment pair by audio, calibrates a stitch from segment 0,
re-estimates it once per segment to see whether the mount held, and writes
report.json, a stitched still and a short stitched preview to
<recording>/phase0/. The preview starts at the given game time (seconds on
Main's timeline, across segments). Prototype only; the pipeline proper
goes in osb_processing once this works on real games.

Stitching uses OpenCV's panorama stitcher, which estimates each camera's
focal length and rotation from the matches. The phones' reported field of
view isn't used: on the first field clip the Pixel 11 Pro's (108.8 deg)
was ~5 deg wider than the frame it recorded.

The cameras are estimated once and reused for every frame, which assumes
the phones stay fixed relative to each other on the mount.
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
CALIB_FRAMES = 8     # frame pairs tried for the stitch calibration


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
              width: int, rotate180: bool = False) -> list[np.ndarray]:
    """`count` BGR frames resampled to constant `fps` from file time `start`."""
    height = round(width * 9 / 16 / 2) * 2
    vf = ("hflip,vflip," if rotate180 else "") + f"fps={fps},scale={width}:{height}"
    raw = subprocess.run(
        # -noautorotate: the files' display rotation is handled from the
        # manifest's orientation, so don't let ffmpeg apply it as well.
        ["ffmpeg", "-v", "error", "-noautorotate", "-ss", f"{start:.4f}", "-i", str(path),
         "-vf", vf, "-frames:v", str(count),
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

def upside_down(frame: np.ndarray) -> bool:
    """True if the frame shows the field at the top: a phone mounted
    rotated 180 degrees. The file carries no rotation, so look for turf."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    turf = (hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 50)
    half = turf.shape[0] // 2
    return bool(turf[:half].mean() > turf[half:].mean())


def angle_between(st) -> float:
    """Rotation between the two estimated cameras, in degrees."""
    r0, r1 = (c.R.astype(np.float64) for c in st.cameras())
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(r1 @ r0.T)[0])))


def calibrate(pairs: list[tuple[np.ndarray, np.ndarray]]):
    """A stitcher whose cameras are estimated from one frame pair.

    Estimates on every pair, then keeps the pair whose focal lengths are
    closest to the median, since a single pair can be thrown off by
    players crowding the overlap.
    """
    fits = []
    for m, s in pairs:
        st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        if st.estimateTransform([s, m]) == cv2.Stitcher_OK:
            fits.append(([s, m], [c.focal for c in st.cameras()], st))
    if not fits:
        return None, {"ok": False, "pairsTried": len(pairs), "pairsOk": 0}
    med = np.median([f for _, f, _ in fits], axis=0)
    images, focals, st = min(fits, key=lambda x: np.abs(np.subtract(x[1], med)).sum())
    yaw = [float(np.degrees(cv2.Rodrigues(c.R.astype(np.float64))[0][1, 0]))
           for c in st.cameras()]
    return st, {"ok": True, "pairsTried": len(pairs), "pairsOk": len(fits),
                "focalPx": [round(f, 1) for f in focals],
                "focalSpreadPx": [round(float(x), 1) for x in np.ptp([f for _, f, _ in fits], axis=0)],
                "angleBetweenDeg": round(angle_between(st), 1),
                "secondarySide": "left" if yaw[0] < yaw[1] else "right"}


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
    # Each phone records which way up it was; older recordings don't, so
    # guess from the picture. Only 0 and 180 occur with the app's fixed
    # landscape orientation.
    flip, source = [], []
    for dev, f in ((main_d, main_f), (sec_d, sec_f)):
        rot = (dev.get("orientation") or {}).get("uprightRotationDeg")
        flip.append(rot == 180 if rot is not None else
                    upside_down(frames_at(f, times[len(times) // 2], 1, 30, PREVIEW_W)[0]))
        source.append("manifest" if rot is not None else "picture")
    report["upsideDown"] = {"main": flip[0], "secondary": flip[1], "from": source}
    cal_pairs = [(frames_at(main_f, t, 1, 30, PREVIEW_W, flip[0])[0],
                  frames_at(sec_f, t + d, 1, 30, PREVIEW_W, flip[1])[0]) for t in times]
    st, cal = calibrate(cal_pairs)
    report["calibration"] = cal
    print(f"upside down: main {flip[0]}, secondary {flip[1]} (from {'/'.join(source)})")
    print("stitch: " + (f"ok on {cal['pairsOk']}/{cal['pairsTried']} pairs, focal {cal['focalPx']} px"
                        f" (spread {cal['focalSpreadPx']}), {cal['angleBetweenDeg']} deg apart,"
                        f" Secondary on the {cal['secondarySide']}" if cal["ok"] else "FAILED"))

    # Did the mount hold? Re-estimate on a few pairs in every segment and
    # compare the angle between the phones with segment 0's.
    stability = []
    for (ms, ss), seg in zip(pairs, report["segments"]):
        seg_dur = min(ms["durationMs"], ss["durationMs"]) / 1000 - abs(seg["offsetS"]) - 1
        t0 = max(0.0, -seg["offsetS"]) + 0.5
        cal_i = calibrate([(frames_at(rec / ms["objectKey"], t, 1, 30, PREVIEW_W, flip[0])[0],
                            frames_at(rec / ss["objectKey"], t + seg["offsetS"], 1, 30,
                                      PREVIEW_W, flip[1])[0])
                           for t in np.linspace(t0, t0 + seg_dur - 0.5, 3)])[1]
        stability.append({"index": ms["index"], **({"angleBetweenDeg": cal_i["angleBetweenDeg"],
                                                    "focalPx": cal_i["focalPx"]} if cal_i["ok"] else {"ok": False})})
    angles = [x["angleBetweenDeg"] for x in stability if "angleBetweenDeg" in x]
    report["stability"] = {"perSegment": stability,
                           "angleRangeDeg": [min(angles), max(angles)] if angles else None}
    print(f"mount: angle between phones {min(angles) if angles else '?'}-{max(angles) if angles else '?'} deg"
          f" over {len(angles)}/{len(stability)} segments")

    m0, s0 = cal_pairs[len(cal_pairs) // 2]
    cv2.imwrite(str(out / "side_by_side.jpg"), np.hstack([m0, s0]))
    if st is not None:
        status, pano = st.composePanorama([s0, m0])
        if status == cv2.Stitcher_OK:
            cv2.imwrite(str(out / "stitched.jpg"), pano)
        # Find the segment pair holding the requested game time.
        i, local = 0, preview_at if preview_at is not None else t_main0
        while i < len(pairs) - 1 and local >= pairs[i][0]["durationMs"] / 1000:
            local -= pairs[i][0]["durationMs"] / 1000
            i += 1
        ms, ss = pairs[i]
        d_i = report["segments"][i]["offsetS"]
        seg_dur = min(ms["durationMs"], ss["durationMs"]) / 1000 - abs(d_i) - 1
        t0 = max(0.0, -d_i) + 0.5
        n = int(min(PREVIEW_SECONDS, seg_dur) * 30)
        local = min(max(local, t0), t0 + seg_dur - n / 30)
        report["preview"] = {"segment": ms["index"], "segmentTimeS": round(local, 2)}
        m_seq = frames_at(rec / ms["objectKey"], local, n, 30, PREVIEW_W, flip[0])
        s_seq = frames_at(rec / ss["objectKey"], local + d_i, n, 30, PREVIEW_W, flip[1])
        enc = None
        for a, b in zip(m_seq, s_seq):
            status, p = st.composePanorama([b, a])
            if status != cv2.Stitcher_OK:
                continue
            if enc is None:
                enc = Encoder(out / "stitched_preview.mp4", (p.shape[1], p.shape[0]))
            enc.write(p)
        if enc:
            enc.close()

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else None)
