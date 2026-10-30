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

Pairs each Main segment with the Secondary file that overlaps it most and
syncs the pair by audio, calibrates a stitch from segment 0,
re-estimates it once per segment to see whether the mount held, and writes
report.json, a stitched still and a short stitched preview to
<recording>/phase0/. The preview starts at the given game time (seconds on
Main's timeline, across segments). Prototype only; the pipeline proper
goes in osb_processing once this works on real games.

Phones that cut segments at different lengths (an iPhone at 300 s, a
Pixel at ~285 s) drift apart: Main's segment k and Secondary's k cover
different minutes, and one Secondary file covers only part of a Main
segment. Pairing by time handles the first; for the second, every
Secondary file overlapping each Main segment is synced too and listed as
the segment's "secondaryPieces", which follow.game_stream reads across.
The phones' start stamps disagree by a second or two, and the error grows
over a game, so each segment's sync is searched around the previous
segment's measured error rather than the stamps alone. Other scripts get
the pairing from paired_devices().

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
MIN_SETTLED_S = 10.0  # calibrate after the last settings lock only if this much of segment 0 is left
FIRST_SEARCH_S = 8.0  # the first sync has only the start stamps to go on: search wider
SYNC_OK = 3.0         # a sync this sharp updates the stamp error the next one searches around ...
SYNC_OK_WINDOWS = 0.5  # ... as does one where this share of its 4 s windows agree (whole-file sharpness
                       # runs low over long files even when they lock cleanly)
MIXED_MIN_S = 2.0     # a second Secondary file overlapping a Main segment this much: mixed segment lengths
PIECE_MIN_S = 0.05    # once mixed, a file overlapping a segment at least this much (a frame or so) is a piece
PIECE_MIN_SYNC_S = 30.0  # shorter pieces are placed from the stamps (and the segment's error), not synced


# ---------------------------------------------------------------- media io

def probe_start(path: Path, kind: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", f"{kind}:0",
         "-show_entries", "stream=start_time", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    # A stream with side data (the Pixel 11 Pro's display matrix) gets a
    # trailing empty CSV column: "0.039400,".
    return float(out.split(",")[0] or 0)


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


FRAME_PREROLL_S = 1.5  # decode this long before the asked time (see frame_at)


def frame_at(path: Path, t: float, width: int, rotate180: bool = False,
             preroll: float = FRAME_PREROLL_S) -> np.ndarray | None:
    """The BGR frame at file time `t`.

    Right after a seek the decoder's first frames are missing (up to ~0.9 s
    in these recordings, depending on where the seek falls between
    keyframes) and the fps filter fills the gap by repeating the first real
    frame, so frames_at(path, t, 1, ...)[0] is a frame up to 0.9 s *later*
    than asked, with no sign of it. Decoding from `preroll` s earlier puts
    the asked time past that run. None when the file ends before `t`, or
    when every frame of the window is the one repeated frame (the preroll
    was too short: try a longer one)."""
    if t < 0:
        return None  # (a negative frame count would have ffmpeg decode the whole file)
    pre = min(preroll, t)
    n = int(round(pre * 30)) + 1
    frames = frames_at(path, t - pre, n, 30, width, rotate180)
    if len(frames) < n:
        return None
    small = [cv2.resize(f[..., 1], (480, 270), interpolation=cv2.INTER_AREA).astype(np.int16) for f in frames]
    if not any(np.abs(small[i] - small[i - 1]).mean() > 0.05 for i in range(1, n)):
        return None
    return frames[-1]


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


def sync(main_audio: np.ndarray, sec_audio: np.ndarray, guess_s: float, search_s: float = 2.0) -> dict:
    """Offset d (s): an event at Main file time t is at Secondary time t + d."""
    a, b = main_audio, sec_audio
    # Whole clip, searching +-search_s around the guess.
    shift = int(round(guess_s * FS))
    b_aligned = b[shift:] if shift >= 0 else np.concatenate([np.zeros(-shift, np.float32), b])
    k, sharp = gcc_phat(a, b_aligned[: len(a)], int(search_s * FS))
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

def file_t0_ms(rec: Path, seg: dict) -> float:
    """Where a segment file's time 0 is on its phone's clock, ms.
    startedAtDeviceMs is the first video frame, which sits at the video
    stream's start_time in the file."""
    return seg["startedAtDeviceMs"] - 1000 * probe_start(rec / seg["objectKey"], "v")


def paired_devices(manifest: dict, report: dict) -> dict:
    """The manifest's devices by role, with Secondary's segments listed one
    per Main segment: the file phase0 paired with it (report segments'
    "secondaryIndex"), so devs["secondary"]["segments"][k] goes with Main's
    k and report["segments"][k]["offsetS"]. Reports from before pairing by
    time keep the manifest's order (index pairing)."""
    devs = {d["role"]: d for d in manifest["devices"]}
    idx = [s.get("secondaryIndex") for s in report.get("segments", [])]
    if "secondary" in devs and idx and None not in idx:
        by_index = {s["index"]: s for s in devs["secondary"]["segments"]}
        devs["secondary"] = {**devs["secondary"], "segments": [by_index[i] for i in idx]}
    return devs


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


def confident(s: dict) -> bool:
    """Whether a sync() result locked cleanly."""
    return s["sharpness"] >= SYNC_OK or s["windowsUsed"] >= SYNC_OK_WINDOWS * max(len(s["windows"]), 10)


def settled_at(dev: dict, seg: dict) -> float:
    """Seconds into `seg` (one of `dev`'s segments) when the phone last
    locked focus and white balance (the manifest's settingsLocks): before
    that it may not have been on the mount yet. 0 for recordings that
    don't say."""
    locks = dev.get("settingsLocks") or []
    if not locks:
        return 0.0
    return max(0.0, (locks[-1]["atDeviceMs"] - seg["startedAtDeviceMs"]) / 1000)


def main(rec: Path, preview_at: float | None) -> None:
    manifest = json.loads((rec / "manifest.json").read_text())
    devs = {d["role"]: d for d in manifest["devices"]}
    main_d, sec_d = devs["main"], devs["secondary"]
    out = rec / "phase0"
    out.mkdir(exist_ok=True)
    report: dict = {"recording": manifest["sessionId"], "segments": []}

    # Pair by time and sync each pair on its own: Android loses frames at
    # every segment switch, so a later pair's offset can differ from the
    # first. Offsets are Secondary file time minus Main file time for the
    # same moment; the stamps predict them, off by `err`.
    clock_ms = (manifest.get("clockSync") or {}).get("secondaryOffsetMs", 0)
    main_t0 = {s["index"]: file_t0_ms(rec, s) for s in main_d["segments"]}
    sec_t0 = {s["index"]: file_t0_ms(rec, s) - clock_ms for s in sec_d["segments"]}

    def stamped(ms, ss):
        return (main_t0[ms["index"]] - sec_t0[ss["index"]]) / 1000

    def overlap(ms, ss, d):
        return min(ms["durationMs"] / 1000, ss["durationMs"] / 1000 - d) - max(0.0, -d)

    audio: dict[str, np.ndarray] = {}

    def audio_of(seg):  # the last few files only: a game's worth doesn't fit
        key = seg["objectKey"]
        if key not in audio:
            if len(audio) >= 4:
                audio.pop(next(iter(audio)))
            audio[key] = load_audio(rec / key)
        return audio[key]

    err, have_err = 0.0, False
    pairs = []
    for ms in main_d["segments"]:
        ss = max(sec_d["segments"], key=lambda x: overlap(ms, x, stamped(ms, x) + err))
        pairs.append((ms, ss))
        guess = stamped(ms, ss) + err
        s = sync(audio_of(ms), audio_of(ss), guess, 2.0 if have_err else FIRST_SEARCH_S)
        if confident(s):
            err, have_err = s["offsetS"] - stamped(ms, ss), True
        seg = {"index": ms["index"], "secondaryIndex": ss["index"], "bleOffsetS": round(stamped(ms, ss), 4),
               "bleErrorMs": round(1000 * (stamped(ms, ss) - s["offsetS"]), 1), **s}
        report["segments"].append(seg)
        print(f"segment {ms['index']} + secondary {ss['index']}: audio offset {s['offsetS'] * 1000:8.2f} ms"
              f" (sharpness {s['sharpness']}), stamp error {-seg['bleErrorMs']} ms,"
              f" drift {s['driftPpm']} ppm, {s['windowsUsed']}/{len(s['windows'])} windows", flush=True)

    # Different segment lengths: a Main segment spans more than one
    # Secondary file. List each with its own offset ("secondaryPieces").
    def pieces_of(ms, seg_err, min_s=PIECE_MIN_S):
        return [x for x in sec_d["segments"] if overlap(ms, x, stamped(ms, x) + seg_err) >= min_s]
    errs = [seg["offsetS"] - stamped(ms, ss) for (ms, ss), seg in zip(pairs, report["segments"])]
    if any(len(pieces_of(ms, e, MIXED_MIN_S)) > 1 for (ms, _), e in zip(pairs, errs)):
        report["mixedSegmentLengths"] = True
        for (ms, ss), seg, seg_err in zip(pairs, report["segments"], errs):
            pieces = []
            for x in pieces_of(ms, seg_err):
                guess = stamped(ms, x) + seg_err
                ov = overlap(ms, x, guess)
                p = {"index": x["index"], "objectKey": x["objectKey"], "durationMs": x["durationMs"],
                     "offsetS": round(guess, 6), "overlapS": round(ov, 1), "synced": False}
                if x is ss:
                    p.update(offsetS=seg["offsetS"], synced=True, sharpness=seg["sharpness"])
                elif ov >= PIECE_MIN_SYNC_S:
                    r = sync(audio_of(ms), audio_of(x), guess)
                    if confident(r) and abs(r["offsetS"] - guess) < 0.5:
                        p.update(offsetS=r["offsetS"], synced=True, sharpness=r["sharpness"])
                pieces.append(p)
            seg["secondaryPieces"] = pieces
            print(f"segment {ms['index']}: " + ", ".join(
                f"secondary {p['index']} {p['overlapS']:.0f} s at {p['offsetS']:+.3f} s"
                f"{'' if p['synced'] else ' (from stamps)'}" for p in pieces), flush=True)

    # Stitch from segment 0: sample frame pairs across it, aligned by d.
    main_f = rec / pairs[0][0]["objectKey"]
    sec_f = rec / pairs[0][1]["objectKey"]
    d = report["segments"][0]["offsetS"]
    dur = min(pairs[0][0]["durationMs"], pairs[0][1]["durationMs"]) / 1000 - abs(d) - 1
    t_main0 = max(0.0, -d) + 0.5
    t_end = t_main0 + dur - 0.5
    # Only frames from after both phones last locked their settings: before,
    # one may still have been in someone's hand or on a lowered tripod.
    settled = max(settled_at(main_d, pairs[0][0]), settled_at(sec_d, pairs[0][1]) - d) + 1
    if settled > t_main0 and settled <= t_end - MIN_SETTLED_S:
        t_main0 = settled
    elif settled > t_main0:
        print(f"settings were last locked {settled:.0f} s into segment 0; calibrating from all of it")
    report["calibrationFromS"] = round(t_main0, 2)
    times = np.linspace(t_main0, t_end, CALIB_FRAMES)
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
