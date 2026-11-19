# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""jointfit.py's fit on a made-up mount and pitch, without footage: paint
drawn through known cameras, taps at their true pixels, a start a few
percent off.

    uv run --with pytest pytest processing/tests/test_jointfit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("torch")  # follow.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
import follow  # noqa: E402
import jointfit as J  # noqa: E402
import pitch as P  # noqa: E402

# Two phones 80 deg apart looking 12 deg down from 6.5 m, 3.5 m behind the near touchline.
TRUE_POSE = [0.0, -33.5, 6.5, 0.0, 0.0, 0.0, 102.0, 60.0, 1.0]
PRIORS = {"main": (750.0, 0.1, 0.0, 0.0, "test"), "secondary": (730.0, 0.1, 0.0, 0.0, "test"),
          "distSd": 0.01, "offsetSdM": 0.03}


def cameras(fm=750.0, fs=730.0):
    return (follow.Camera(fm, 960.0, 540.0, 0.004, 0.003, J.rough_rotation(-40.0, -12.0, 0.0)),
            follow.Camera(fs, 960.0, 540.0, -0.008, 0.005, J.rough_rotation(40.0, -12.0, 0.0), t=[0.0, 0.0, 0.0]))


@pytest.fixture(scope="module")
def scene():
    cm, cs = cameras()
    pose = P.PitchPose(TRUE_POSE)
    model = J.Model(cm, cs)
    v_true = J.start_vector(cm, cs, pose)
    paints = []
    for x, y, ok in model.project(v_true, pose.directions(P.pitch_lines(pose.L, pose.W, pose.k, step=0.02))):
        mask = np.zeros((J.H, J.W), np.uint8)
        for xi, yi in zip(x[ok], y[ok]):
            cv2.circle(mask, (int(round(xi)), int(round(yi))), 2, 255, -1)
        p = J.Paint.__new__(J.Paint)
        p.mask, p._dt = mask, {}
        p.dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        paints.append(p)
    taps = {}
    for name, xz in P.tap_targets(pose).items():
        for k, (x, y, ok) in enumerate(model.project(v_true, pose.directions(np.array([xz], float)))):
            if ok[0] and name not in taps:
                taps[name] = {"phone": ("main", "secondary")[k], "x": float(x[0]), "y": float(y[0])}
    assert len(taps) >= 10, "the made-up mount should see most of the pitch's keypoints"
    # The start: both focal lengths, the camera's place and height and the pitch's size a few percent off.
    cm0, cs0 = cameras(fm=762.0, fs=748.0)
    start = list(TRUE_POSE)
    start[1], start[2], start[6], start[7] = -34.5, 6.25, 102.5, 60.6
    return model, v_true, J.start_vector(cm0, cs0, P.PitchPose(start)), paints, taps


def test_fit_recovers_the_pitch_and_says_it_converged(scene):
    model, v_true, v0, paints, taps = scene
    log = {}
    v = J.fit(model, v0, paints, taps, PRIORS, verbose=False, log=log)
    assert log["converged"] and 2 <= log["passes"] <= J.MAX_PASSES
    assert abs(v[14] - v_true[14]) < 0.08, "camera height, m"
    assert abs(v[18] - v_true[18]) < 0.30, "pitch length, m"
    assert abs(v[19] - v_true[19]) < 0.15, "pitch width, m"
    px = J.tap_px(model, v, taps)
    assert np.median(list(px.values())) < 1.5

    # A fit from its own result stays put (one-sided differences left it 0.9 m long and still moving).
    again = J.fit(model, v, paints, taps, PRIORS, verbose=False)
    assert abs(again[18] - v[18]) < 0.10 and abs(again[19] - v[19]) < 0.10 and abs(again[14] - v[14]) < 0.02


def test_a_tap_left_out_is_still_placed(scene):
    model, v_true, v0, paints, taps = scene
    left_out = J.holdout_folds(list(taps), 5)[0]
    v = J.fit(model, v0, paints, {n: t for n, t in taps.items() if n not in left_out}, PRIORS, verbose=False)
    px = J.tap_px(model, v, {n: taps[n] for n in left_out})
    assert set(px) == set(left_out) and max(px.values()) < 4.0


def test_holdout_folds_cover_every_tap_once():
    names = [f"tap {i}" for i in range(15)]
    folds = J.holdout_folds(names[::-1], 5)
    assert folds == J.holdout_folds(names, 5)  # the same split whatever order they come in
    assert sorted(n for f in folds for n in f) == sorted(names) and [len(f) for f in folds] == [3] * 5
    assert len(J.holdout_folds(names[:3], 5)) == 3


def test_soften_is_soft_l1_and_the_priors_are_not_softened():
    r = np.array([-3.0, -0.1, 0.0, 0.05, 0.3, 2.0, 40.0])
    z = (r / J.LOSS_SCALE) ** 2
    soft_l1 = 0.5 * J.LOSS_SCALE ** 2 * 2 * (np.sqrt(1 + z) - 1)  # scipy's cost for loss="soft_l1", f_scale
    assert np.allclose(0.5 * J.soften(r) ** 2, soft_l1)
    assert np.array_equal(np.sign(J.soften(r)), np.sign(r))

    # A lens offset 10 sd out costs what a Gaussian says, 50: not the 2.9 it would through the loss.
    v = np.zeros(J.NPAR)
    v[J.FM], v[J.FS] = PRIORS["main"][0], PRIORS["secondary"][0]
    v[9] = 10 * PRIORS["offsetSdM"]
    pr = J.prior_residuals(v, PRIORS)
    assert 0.5 * float((pr ** 2).sum()) == pytest.approx(50.0)
    assert 0.5 * float((J.soften(pr) ** 2).sum()) == pytest.approx(2.9, abs=0.1)


def test_lens_table_spread_widens_the_distortion_prior():
    dev = {"role": "secondary", "model": "Phone X", "video": {"lens": "ultrawide", "width": 3840, "height": 2160,
                                                              "horizontalFovDeg": 104.0}}
    main = {**dev, "role": "main", "model": "Phone Y"}
    fits = [{"recording": "a", "role": "secondary", "f": 744.6, "k1": -0.0223, "k2": 0.0156},
            {"recording": "b", "role": "secondary", "f": 737.3, "k1": -0.0004, "k2": -0.0021}]
    table = {J.lens_key(dev): J.summarize(fits)}
    e = table[J.lens_key(dev)]
    assert e["distSd"] == pytest.approx(np.std([-0.0223, -0.0004], ddof=1))
    # The two fits disagree on f by 1% and on k1 by 0.022, and on the field of view by under half a degree.
    assert e["halfFovSdDeg"] < 0.5
    priors = J.lens_priors({"devices": [main, dev]}, table, have_taps=True)
    assert priors["distSdBy"] == {"secondary": pytest.approx(e["distSd"])}
    priors.update(distSd=J.DIST_SD, offsetSdM=J.OFFSET_SD_M)
    v = np.zeros(J.NPAR)
    v[J.FM], v[J.FS] = priors["main"][0], priors["secondary"][0]
    v[J.K1M], v[J.K1S] = 0.02, e["k1"] + 0.02  # both 0.02 from their priors
    pr = J.prior_residuals(v, priors)
    assert pr[2] == pytest.approx(2.0) and pr[4] == pytest.approx(0.02 / e["distSd"]) and pr[4] < 1.5


def test_half_fov_matches_the_camera_model():
    for f, k1, k2 in ((750.0, 0.0, 0.0), (744.6, -0.0223, 0.0156), (751.7, 0.0034, 0.0034)):
        cam = follow.Camera(f, 960.0, 540.0, k1, k2, np.eye(3))
        a = np.radians(J.half_fov_deg(f, k1, k2))
        x, _, _ = cam.project(np.array([[np.tan(a), 0.0, 1.0]]))
        assert x[0] == pytest.approx(1920.0, abs=0.5)  # the ray at that angle lands on the frame's right edge


# 2026-11-18's game (auto.json of the fit that was taken), the shape auto_facts gives.
TAKEN = {"converged": True, "lengthM": [101.85, 101.42], "widthM": [59.35, 59.46], "heightM": [6.19, 6.25],
         "paintWithin5": {"main": [0.915, 0.951], "secondary": [0.897, 0.946]},
         "paintPoints": {"main": [1836, 1841], "secondary": [1994, 1984]},
         "seamPx4k": [0.38, 0.44], "lensOffsetM": 0.014,
         "lenses": {"main": {"f": 747.6, "k1": 0.0006, "k2": -0.001}, "secondary": {"f": 724.0, "k1": -0.004, "k2": 0.001}}}


@pytest.mark.parametrize("change, says", [
    ({}, None),
    ({"converged": False}, "still moving"),
    ({"lengthM": [108.0, 102.1]}, "length"),                  # rec4cal from its lines-only pitch
    ({"widthM": [63.6, 74.3]}, "width"),                      # rec8
    ({"heightM": [8.02, 10.28]}, "height"),
    ({"paintWithin5": {"main": [0.80, 0.82], "secondary": [0.9, 0.95]}}, "main: 82%"),
    ({"paintWithin5": {"main": [0.97, 0.95], "secondary": [0.9, 0.95]}}, "fewer markings"),
    ({"seamPx4k": [0.73, 6.04]}, "seam got worse"),           # rec4cal
    ({"seamPx4k": [float("inf"), float("inf")]}, "too little paint"),
    ({"lensOffsetM": 0.25}, "25 cm apart"),
    ({"lenses": {"main": {"f": 747.6, "k1": 0.08, "k2": 0.0}, "secondary": {"f": 724.0, "k1": 0.0, "k2": 0.0}}}, "distortion"),
    ({"tapsPx": [[3.5, 11.0], [2.1, 6.1]]}, None),           # with marked points: closer than pitch.py's refit put them
    ({"tapsPx": [[3.5, 11.0], [3.9, 11.5]]}, None),          # within a pixel of it
    ({"tapsPx": [[3.5, 11.0], [3.6, 19.0]]}, "marked points got further"),
    ({"tapsPx": [[3.5, 11.0], None]}, "can't be scored"),
])
def test_an_untapped_fit_is_taken_only_when_nothing_speaks_against_it(change, says):
    why = J.auto_reasons({**TAKEN, **change})
    assert (why == []) if says is None else (len(why) == 1 and says in why[0]), why


def test_a_fit_that_fails_leaves_the_stitch_and_says_so(tmp_path):
    rec = tmp_path / "rec"
    (rec / "phase0").mkdir(parents=True)
    (rec / "phase0" / "cameras.json").write_text("{}")  # no manifest, no report: the fit can't start
    J.auto_main(rec, None)
    res = json.loads((rec / "joint" / "auto.json").read_text())
    assert res["accepted"] is False and "the fit failed" in res["reasons"][0]
    assert (rec / "phase0" / "cameras.json").read_text() == "{}"


def test_cameras_that_arent_a_stitch_are_left_alone(tmp_path):
    rec = tmp_path / "rec"
    for d in ("phase0", "pitch"):
        (rec / d).mkdir(parents=True)
    (rec / "manifest.json").write_text(json.dumps({"devices": []}))
    (rec / "phase0" / "report.json").write_text(json.dumps({"segments": [{}, {}], "upsideDown": {"main": False, "secondary": False}}))
    (rec / "phase0" / "cameras.json").write_text(json.dumps({"rough": {"left": "main"}}))
    res = J.auto_fit(rec, None, rec / "joint")
    assert res["accepted"] is False and "rough" in res["reasons"][0]
    (rec / "phase0" / "cameras.json").write_text(json.dumps({"jointFit": {"segment": 1}}))
    res = J.auto_fit(rec, None, rec / "joint")  # a joint fit already: only marked points fit it again
    assert res["accepted"] is False and "joint fit already" in res["reasons"][0]


def test_too_few_marked_points_dont_fit_the_lenses(tmp_path):
    rec = tmp_path / "rec"
    for d in ("phase0", "pitch"):
        (rec / d).mkdir(parents=True)
    (rec / "manifest.json").write_text(json.dumps({"devices": []}))
    (rec / "phase0" / "report.json").write_text(json.dumps({"segments": [{}, {}], "upsideDown": {"main": False, "secondary": False}}))
    cm, cs = cameras()
    (rec / "phase0" / "cameras.json").write_text(json.dumps({"main": cm.to_json(), "secondary": cs.to_json()}))
    (rec / "pitch" / "background.png").write_bytes(b"png")
    # Looking 12 deg down between the phones: two points marked, one kept from the fit, one skipped.
    (rec / "pitch" / "taps.json").write_text(json.dumps({
        "left corner flag": {"yaw": -30.0, "pitch": 10.0}, "right corner flag": {"yaw": 30.0, "pitch": 10.0},
        "halfway far end": {"yaw": 0.0, "pitch": 8.0, "fromFit": True}, "circle far crossing": {"skipped": True}}))
    res = J.auto_fit(rec, None, rec / "joint", with_taps=True)
    assert res["accepted"] is False and res["taps"] == 2 and "2 marked points" in res["reasons"][0]
    assert json.loads((rec / "pitch" / "taps_src.json").read_text())["taps"]["left corner flag"]["phone"] == "main"
