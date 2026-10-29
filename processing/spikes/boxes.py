#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4"]
# ///
"""What the pipeline saw at moments of a fullgame.py piece.

    uv run processing/spikes/boxes.py footage/rec4 <piece> <mm:ss> [mm:ss ...]

Per moment: the follow-cam frame on top; below, Secondary | Main at full
field of view with the ball candidates (ballwin, boxes labelled
detector conf / judge) and the people (roles.py, foot point and a box of
their detected height, coloured by role), plus the camera-log rule in force;
at the bottom, each phone at native resolution around its players.
Times are within the piece (as a player shows them), so a moment someone
points at in piece_01 is found without working out game time. Needs the
chunk's ballwin.py and roles.py outputs in detect/ and its camera log in
fullgame/logs/. Writes <recording>/fullgame/boxes/piece<NN>_<m>m<ss>s.jpg.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import follow  # noqa: E402
import phase0  # noqa: E402
import pitch as P  # noqa: E402

CHUNK = 600.0  # fullgame.py's default --chunk
ROLE_COL = {"A": (40, 40, 230), "B": (255, 255, 255), "referee": (0, 230, 255), "keeper": (255, 200, 0),
            "other": (150, 150, 150)}


def nearest(frames, t):
    return min(frames, key=lambda f: abs(f["t"] - t))


def main(rec: Path, piece: int, stamps: list[str]) -> None:
    manifest = json.loads((rec / "manifest.json").read_text())
    report = json.loads((rec / "phase0" / "report.json").read_text())
    devs = phase0.paired_devices(manifest, report)
    flip = [report["upsideDown"]["main"], report["upsideDown"]["secondary"]]
    cm, cs = follow.setup_cameras(rec, report, devs, flip)
    cams = [cm.at(follow.SRC_W), cs.at(follow.SRC_W)]
    pose = P.load_pose(rec)
    c0 = piece * CHUNK
    win = json.loads((rec / "detect" / f"ball_0_{c0:g}_{CHUNK:g}.win.json").read_text())["frames"]
    roles = json.loads((rec / "detect" / f"roles_0_{c0:g}_{CHUNK:g}.json").read_text())["frames"]
    camlog = [l for l in (rec / "fullgame" / "logs" / f"camlog_{piece:02d}.txt").read_text().splitlines()
              if l.strip() and not l.startswith("#")]
    out = rec / "fullgame" / "boxes"
    out.mkdir(exist_ok=True)
    W, H = follow.SRC_W, follow.SRC_H
    for st in stamps:
        m, s = st.split(":")
        local = int(m) * 60 + float(s)
        t = c0 + local
        # The follow-cam frame.
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{local:.3f}", "-i",
                              str(rec / "fullgame" / "pieces" / f"piece_{piece:02d}.mp4"), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], check=True, capture_output=True).stdout
        cam_img = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).copy()
        # Both phones, upright.
        phones = []
        for ph in (0, 1):
            buf = next(iter(follow.game_stream(rec, devs, report, ph, 0, t, 1 / follow.FPS, W)))
            bgr = cv2.cvtColor(np.asarray(buf).reshape(H * 3 // 2, W), cv2.COLOR_YUV2BGR_NV12)
            phones.append(cv2.rotate(bgr, cv2.ROTATE_180) if flip[ph] else bgr)
        fw = nearest(win, t)
        for d in fw["detections"]:
            if d["cls"] != "ball":
                continue
            x1, y1, x2, y2 = d["box"]
            good = d.get("judge", 0) >= 0.5
            col = (0, 140, 255) if good else (0, 0, 160)
            cv2.rectangle(phones[d["phone"]], (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), col, 6 if good else 3)
            cv2.putText(phones[d["phone"]], f"{d['conf']:.2f}/{d.get('judge', float('nan')):.2f}", (x1 - 12, y1 - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
        fr = nearest(roles, t)
        feet = {0: [], 1: []}
        for p in fr["people"]:
            yaw, pit = pose.angles(np.array([p["xz"]], float))[0]
            pit = pit - pose.warp(yaw)
            y, q = np.radians(yaw), np.radians(pit)
            d = np.array([np.sin(y) * np.cos(q), np.sin(q), np.cos(y) * np.cos(q)])
            px, py, ok = cams[p["phone"]].project(d)
            if not ok:
                continue
            fx, fy, h = int(px), int(py), p["h"]
            if h < 400:  # the zoom row centres on play, not on whoever stands by the camera
                feet[p["phone"]].append((fx, fy - h // 2))
            col = ROLE_COL.get(p["role"], (150, 150, 150))
            cv2.rectangle(phones[p["phone"]], (fx - int(0.22 * h), fy - h), (fx + int(0.22 * h), fy), col, 4)
            cv2.putText(phones[p["phone"]], p["role"], (fx - int(0.22 * h), fy - h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
        rule = [l for l in camlog if parse_t(l) <= t + 1e-6]
        rule = rule[-1].strip() if rule else ""
        sw = 1600
        bottom = np.hstack([cv2.resize(phones[1], (sw, 900)), cv2.resize(phones[0], (sw, 900))])
        top = np.zeros((900, 2 * sw, 3), np.uint8)
        top[:, sw // 2: sw // 2 + sw] = cv2.resize(cam_img, (sw, 900))
        # Zoom row: native-resolution crops of each phone around its players.
        zoom = []
        for ph in (1, 0):
            cw, ch = sw, 450
            if feet[ph]:
                cx_, cy_ = np.median(np.array(feet[ph]), axis=0)
            else:
                cx_, cy_ = W / 2, H / 2
            x0 = int(np.clip(cx_ - cw / 2, 0, W - cw))
            y0 = int(np.clip(cy_ - ch / 2, 0, H - ch))
            zoom.append(phones[ph][y0:y0 + ch, x0:x0 + cw])
        img = np.vstack([top, bottom, np.hstack(zoom)])
        lines = [f"piece_{piece:02d} {st}  (game {int(t // 60)}:{t % 60:04.1f}; balls @ {fw['t']:.1f}, people @ {fr['t']:.2f})",
                 f"camlog: {rule}",
                 "ball: orange = judge >= 0.5, dark red = rejected; label conf/judge.  people: red A, white B, "
                 "yellow referee, cyan keeper, grey other",
                 "middle: Secondary (left) | Main (right); bottom: the same at full resolution around the players. Each person is boxed once, on the phone with the taller view"]
        for k, txt in enumerate(lines):
            cv2.putText(img, txt, (20, 50 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
            cv2.putText(img, txt, (20, 50 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        name = out / f"piece{piece:02d}_{st.replace(':', 'm')}s.jpg"
        cv2.imwrite(str(name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{name}: {sum(d['cls'] == 'ball' for d in fw['detections'])} ball candidates, "
              f"{len(fr['people'])} people; {rule}", flush=True)


def parse_t(line: str) -> float:
    m, s = line.split()[0].split(":")
    return int(m) * 60 + float(s)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
