# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""jointfit.py's fit on a made-up mount and pitch, without footage: paint
drawn through known cameras, taps at their true pixels, a start a few
percent off.

    uv run --with pytest pytest processing/tests/test_jointfit.py
"""

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
