# Standard ball models

Every game uses these unless a run says otherwise: `fullgame.py` passes `ball_detector.pth` to `ballwin.py`
(`--detector` to override), and `balljudge.load_judge` reads `ball_judge.pt` (a model name picks
`<recording>/balljudge/<name>.pt` instead). The files are not in git (`footage/`): they are pinned by hash in `packaging/cloud/models.json` and published as
GitHub release assets, and `uv run packaging/cloud/models.py fetch` downloads the pinned ones a machine lacks.
`footage/models/README.md` is a copy of this file, shipped with the models.

| File | sha256 | What |
|---|---|---|
| `ball_detector.pth` | `04632214…` | RF-DETR Medium fine-tuned on rec3 + rec5 + rec6 (`footage/balldata_rec3_rec5_rec6`, rec5's tiles x3, rec6's x2): 275 white-ball boxes, 54 orange (x3), 69 daylight (x2). Trained 2026-10-31 into `footage/rec6/balltrain_v3/` (best EMA epoch 15, effective mAP 0.351) |
| `ball_judge.pt` | `bd32076c…` | balljudge CNN, trained 2026-11-13 as `footage/rec6/balljudge/judge_v5.pt` (run "a" of two; valid AP 0.963, held-out clip AP 0.92). rec3's crops + rec5's (x3) + rec6's (x2) + rec7's `labels_seg6` (x4, its last 40 s held out), with the reviewed off-pitch verdicts (`offpitch.py`): a reviewed spare ball is a positive, a reviewed non-ball a negative, at most 10 frames per spot. Data: `footage/ab_offpitch/capped/` |

`ball_detector.pth` was rewritten on 2026-11-18 with the training settings' paths made relative (the dataset and output folders, the stock weights, the checkpoints' names: they had the training machine's home folder in them). Its 510 tensors are unchanged and it gives the same candidates as the file before (`c69adbc8…`, which images and apps built before then carry, and S3 keeps under `models/c69adbc8c417/`).

The judge is standard since 2026-11-13, replacing `judge_v3` (`footage/rec6/balljudge/judge_v3.pt`,
`1a94722b…`: trained 2026-10-31, effectively rec3 x4 + rec5 x9 + rec6 x2 through a merge bug). The detector
is unchanged: one retrained on the same triaged data benched worse (docs/BACKLOG.md). Benched with the
standard detector's candidates, `judge_v3` -> `judge_v5`:

- rec7 `labels_seg6` (yellow ball, night): 0 -> 24 of 52 found, 0 wrong (held-out moments 0 -> 7 of 17)
- rec3 seg9 (held out): 24 found / 2 wrong -> 24 / 3 of 37; a ball placed at 0 -> 4 of 12 "none" moments
- rec3 seg3: 44 / 2 -> 43 / 2 of 53
- ball path, share of frames: rec6 55 -> 59%, rec5 43 -> 42%
- not run: the must-see boxes

The detector is standard since 2026-11-01, when it and `judge_v3` replaced rec5's detector (`footage/rec5-game/balltrain_v2/`, `3943b492…`) and
judge (`footage/rec5-game/balljudge/judge_v2.pt`, `0a11d7bd…`). Benched against them as a pair
(docs/BACKLOG.md; logs in `footage/benchscan/*_v3.log`):

- rec6 0-10 min (daylight): ball path 30% -> 52% of frames; 30-31 of 32 sampled path frames on the
  real ball (`footage/rec6/benchcheck/`). Some of rec6's labelled moments are in these 10 minutes.
- rec5 0-10 min (orange ball): ball path 39% -> 39%
- rec4cal: play boxes 9 -> 10/15; aim at the ball 37/22% -> 36/19%
- rec3: play boxes 20 -> 20/26; balleval seg3 + seg9 67 -> 63/90 found (wrong 6 -> 3)

The new detector needs the new judge: with the old judge it gave rec6 29%, rec3 58/90 found.

Before that: rec3's detector (`footage/rec3/balltrain/`) and judge (`footage/rec3/balljudge/judge.pt`),
replaced 2026-10-30 by the rec5 pair.

To replace one: train it (balltrain.py / `balljudge.py train`), bench it, copy it over the file in `footage/models/`,
pin and publish it (`packaging/cloud/models.py pin`, then `release`; commit `models.json`),
and update this table.
