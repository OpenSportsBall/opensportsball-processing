#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy>=1.26", "scipy>=1.13", "opencv-python-headless>=4.10", "torch>=2.4", "rfdetr>=1.11"]
# ///
"""The ball detector and judges on SoccerTrack's wide view (docs/DATA.md,
External datasets): a camera, field and light none of our models has
seen, so a test of how they carry to a new game. Frame by frame, without
a camera model:

    uv run processing/spikes/stbench.py [step s] [max clips] [--expect-row]

Every `step` s of each 30 s clip: the standard detector (footage/models/
ball_detector.pth) on 640 px tiles of the 6500 x 1000 frame, then each
judge (the standard one and rec7's judge_v4) on every candidate. A
candidate is on the ball within max(8 px, 1.5 x the labelled width).
Judge crops are sized from the candidate's box, or with --expect-row
from the labelled ball width at that image row (what our pitch map gives
the judge). Prints how often the ball is among the candidates, kept by
each judge (>= 0.5) and the top pick; writes footage/soccertrack/
bench.json. Never train on this set: it is held out.
"""
import csv, json, sys, time
from pathlib import Path
import cv2, numpy as np

SP = Path(__file__).resolve().parent
ST = SP.parents[1] / "footage" / "soccertrack" / "wide_view"
OUT = ST.parent / "bench.json"
_a = [x for x in sys.argv[1:] if not x.startswith("--")]
STEP = float(_a[0]) if _a else 2.0
MAXC = int(_a[1]) if len(_a) > 1 else 999
sys.path.insert(0, str(SP))
import balljudge as J, detect  # noqa: E402
import rfdetr  # noqa: E402

FOOT = SP.parents[1] / "footage"
model = rfdetr.RFDETRMedium(pretrain_weights=str(FOOT / "models" / "ball_detector.pth"))
judges = {"v3": J.load_judge(FOOT, None), "v4": J.load_judge(FOOT / "rec7", "judge_v4")}
T = 640

def tiles(h, w):
    ys = list(range(0, h - T, 530)) + [h - T]; xs = list(range(0, w - T, 530)) + [w - T]
    return [(x, y) for y in ys for x in xs]

def balls(csvp):
    r = list(csv.reader(open(csvp))); h = r[:3]
    c = {h[2][i]: i for i, t in enumerate(h[0]) if t == "BALL"}
    out = {}
    for x in r[3:]:
        if x and x[0][:1].isdigit() and len(x) > c["bb_left"] and x[c["bb_left"]]:
            l, t_, w, hh = (float(x[c[k]]) for k in ("bb_left", "bb_top", "bb_width", "bb_height"))
            out[int(float(x[0]))] = (l + w / 2, t_ + hh / 2, w)
    return out

vids = sorted((ST).rglob("F_*.mp4"))[:MAXC]
anns = {p.stem: p for p in ST.rglob("F_*.csv")}
# Expected ball diameter by image row, from every label (what our pitch map gives
# the judge): the median labelled width in 50-px row bands, interpolated.
_all = [v for p in anns.values() for v in balls(p).values()]
_ys = np.array([v[1] for v in _all]); _ws = np.array([v[2] for v in _all])
_bins = np.arange(0, 1050, 50)
_med = [(c + 25, float(np.median(_ws[(_ys >= c) & (_ys < c + 50)]))) for c in _bins if ((_ys >= c) & (_ys < c + 50)).sum() >= 20]
EXPECT = "row" if "--expect-row" in sys.argv else "box"
def expect_at(y):
    return float(np.interp(y, [m[0] for m in _med], [m[1] for m in _med]))
print("ball width by row:", [(int(a), round(b, 1)) for a, b in _med], "| crop size from", EXPECT)
rows, t0 = [], time.time()
for vi, vp in enumerate(vids):
    if vp.stem not in anns:
        continue
    lab = balls(anns[vp.stem]); cap = cv2.VideoCapture(str(vp)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); every = max(1, int(round(STEP * fps)))
    first = min(lab) if lab else 0
    for k in range(0, n, every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, img = cap.read()
        if not ok:
            break
        b = lab.get(k + first)  # label frame numbering may start at 0 or 1
        if b is None:
            continue
        H, W = img.shape[:2]
        rgb = img[..., ::-1].copy()
        tl = tiles(H, W)
        res = model.predict([rgb[y:y + T, x:x + T] for x, y in tl], threshold=0.1)
        det = [np.r_[box + [x, y, x, y], sc] for (x, y), d in zip(tl, res) for box, sc in zip(d.xyxy, d.confidence)]
        det = detect.nms(np.array(det)) if det else np.zeros((0, 5))
        cands = []
        if len(det):
            cx, cy = (det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2
            dm = np.maximum(4, np.maximum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]))
            box = dm.copy()
            if EXPECT == "row":
                dm = np.array([expect_at(y) for y in cy])
            crops = np.stack([J.crop_at(img, x, y, d) for x, y, d in zip(cx, cy, dm)])
            sc = {k_: f(crops) for k_, f in judges.items()}
            rad = max(8.0, 1.5 * b[2])
            for i in range(len(det)):
                cands.append({"x": float(cx[i]), "y": float(cy[i]), "d": float(dm[i]), "box": float(box[i]),
                              "expect": expect_at(float(cy[i])), "conf": float(det[i, 4]),
                              **{k_: float(sc[k_][i]) for k_ in judges},
                              "hit": bool(np.hypot(cx[i] - b[0], cy[i] - b[1]) <= rad)})
        rows.append({"clip": vp.stem, "frame": k, "W": W, "H": H, "ball": b, "cands": cands})
    print(f"{vi + 1}/{len(vids)} clips, {len(rows)} frames, {(time.time() - t0) / 60:.1f} min", flush=True)
    OUT.write_text(json.dumps(rows))

def summary(rows):
    n = len(rows); f = lambda c, k: c["conf"] * c[k]
    print(f"{n} frames (ball labelled in every one)")
    print(f"detector: ball among candidates {sum(any(c['hit'] for c in r['cands']) for r in rows)}/{n},"
          f" candidates/frame {np.mean([len(r['cands']) for r in rows]):.1f}")
    for k in judges:
        keep = lambda r: [c for c in r["cands"] if c[k] >= 0.5]
        top = sum(bool(r["cands"]) and max(r["cands"], key=lambda c: f(c, k))["hit"] for r in rows)
        print(f"judge {k}: ball kept (>=0.5) {sum(any(c['hit'] for c in keep(r)) for r in rows)}/{n},"
              f" false kept/frame {np.mean([sum(not c['hit'] for c in keep(r)) for r in rows]):.2f},"
              f" top pick is the ball {top}/{n}")
summary(rows)
