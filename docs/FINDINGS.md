# Findings: what we tried and what happened

A summary of the processing techniques tested on the night game
(2026-10-14, 97.5 minutes, two Pixels on a fixed mount) between
2026-10-12 and 2026-10-19, organised by problem. Each entry gives the
technique, its measured outcome, and its status. The dated detail,
with numbers per run, is in `docs/BACKLOG.md` (Phase 0 section); the
measurement methods are in `docs/TESTING-PROCESSING.md`.

Status key: **adopted** (in the default configuration), **available**
(implemented, off by default), **dropped** (tested, no gain), **open**
(promising, not finished).

Measurements referred to below:
- *play boxes*: 12 reviewed must-see moments on two 2-minute clips;
  a moment is captured when >= 98% of its box is in the 1080p view.
- *ball labels*: the ball's position every 2 s along the same clips
  (37 and 53 moments with the ball on the pitch); the tracker is scored
  as found / wrong / missed per moment.

## 1. Synchronising and stitching the two phones

| Technique | Outcome | Status |
|---|---|---|
| Audio sync by band-limited GCC-PHAT, per segment | Locks in every segment; clock drift a steady -9 ppm. The offset jumps by up to 390 ms at segment switches, and the phones' own timestamps don't show it, so every segment is re-synced. | adopted |
| OpenCV panorama stitcher for the camera geometry, refined at 1920 px (focal lengths, two radial terms, Secondary's rotation) | One calibration from segment 0 holds for the whole mounted game (overlap points within ~3 px); alignment error at 4K halved by the refinement. Handheld footage drifts and would need periodic re-estimation. | adopted |
| Homography fitted on field points | Worse than the stitcher: turf texture gives mostly false matches. | dropped |
| Rendering straight from both 4K sources through a virtual camera, each pixel from one phone along a per-frame seam that avoids players, two-band blend | Fixes the doubled players that blending produced (the phones never capture the same instant). | adopted |
| GPU pipeline (PyTorch on MPS/CUDA, hardware decode and encode, NV12 throughout) | 20 s of video in 64 s on an M-series Mac; a full game renders in ~2.5 h locally. | adopted |
| Clips that run across segment boundaries; a decoder pre-roll after seeks | The pre-roll removed a 0.25-0.5 s freeze at the start of every render; a ~0.2 s hold remains at each segment switch (a real gap in the recording). | adopted |

## 2. Mapping the pitch and the goals

| Technique | Outcome | Status |
|---|---|---|
| Pitch pose fitted to the painted lines (camera position and height, rotation, ground tilt, length and width, plus a smooth vertical-angle correction) from a median background panorama | Night game: 109.3 x 60.5 m, camera 2.7 m up. | adopted |
| Goals as goal-shaped white structures near each end, every left/right pair tried by refitting, validated against the markings | Corrects the map (the right end had been ~20 m off); the follow cam's goal rules only worked once the goals were detected. | adopted |
| Regression test on saved background panoramas (`processing/tests/test_pitch_regression.py`) | 15 key points within 0.1 deg, goals within 0.3 m; one fixture so far. | adopted |
| Automatic starting pose; youth and small-sided layouts; other venues and daylight | Not started. | open |

## 3. Where is play? The follow cam without the ball

| Technique | Outcome | Status |
|---|---|---|
| Median of all player blobs | Baseline; lags behind quick play. | dropped |
| Speed-weighted centroid of players (player "vectors") | Caught a quick counter the median missed, but jumps between groups. | superseded |
| Hotspot of running: players pooled over +-0.5 s, weighted linearly by speed, peak by mean shift, sticky (another peak must be 50% stronger for 0.7 s) | Followed an attack to goal; aim jumps > 40 deg/s fell from 435 to 73 frames. Needed linear (not squared) weights and 0.5 s position smoothing, because far-side jitter reads as running. | adopted |
| Only people inside the lines; cells by the touchline occupied >= 10% of the time (bench, subs) ignored | Removes the bench from the aim. | adopted |
| Lull hold (<= 1 runner for 1 s keeps the aim from just before) | Stops the camera drifting to where players wait. | adopted |
| Restart taker (a still player alone in a goal area, at a corner or on a touchline during a lull) | Frames the taker and the group. Flashes briefly in short lulls; needs the same gating as the goal kick. | adopted, open |
| goalframe: >= 4 players within 20 m of a goal while the hotspot is elsewhere, frame both | Catches goalmouth action the hotspot leaves. | adopted |
| goalkick: a lone near-still player in a penalty area soon after play was there, held as a state until play resumes | Catches goal kicks; false positives fixed later by three vetoes (see 5). | adopted |
| unsure: few runners near the hotspot keeps the view wide | Catches quick clearances at the price of width. | adopted |
| lead (aim 1 s ahead), nowalk (walkers weigh nothing), fastboost (steeper speed weight), weak (frame everyone when the signal is thin), allnear, flow (aim along the runners' velocity), lone (include a fast runner elsewhere), gk2 (goal-kick cooldowns), bigger hotspot bandwidth, no lull | None beat the baseline on the play boxes (4/13). The misses were aim, not width: the hotspot stays on the pack while one player has the ball. | dropped |
| Hotspot "weak signal" diagnosis | On the second clip only 10-18 of ~24 people are tracked per frame and few register as running; every rule degrades together. Better player detection (RF-DETR people instead of background blobs) is the real fix. | open |
| Zoom floor 50 deg (a 1080p view narrower than ~55 deg upsamples the 4K source; 65 deg is pixel-for-pixel at a phone's centre) | Sharper output, median width ~63 deg. | adopted |
| Never show past the canvas's left or right edge: stop there and zoom in (to 40 deg; the phones' edges have the most pixels per degree) | Removes the black edge; a further margin for lens fall-off is still to be measured. | adopted, open |
| Camera log (`camlog.txt`, one line per change of the driving rule) | Made the full-render reviews possible. | adopted |

With the ball (section 4) and roles (section 6): **12/12 play boxes** at HEAD (one box was removed and one redrawn on review of the offline ball path; before that 12/13).

## 4. Where is the ball?

Detection:

| Technique | Outcome | Status |
|---|---|---|
| Stock RF-DETR "sports ball" on 640 px tiles of the 4K frames | 58% of match balls at conf >= 0.2, 2.1 wrong suggestions per frame. | superseded |
| Fine-tuned RF-DETR Medium on 275 labelled balls (640 px tiles, hard negatives) | 76% of match balls on held-back frames at 1.1 wrong per frame (stock: 69% at 10.5); the top suggestion is the ball 69% of the time. Overfits this game (96% on training frames). | adopted |
| Frame differencing (motion candidates) | 55% of match balls at ~120 candidates per frame; ranking by contrast is poor. Half the misses are the ball merged into a player's blob, which the detector finds. | dropped |
| Crop judge (48 px crop classifier, 1.5 ms per crop) to rank candidates | Ranks about as well as possible (top-1 62% against a 77% ceiling) and cleans candidates (zero wrong tracks), but does not raise the tracker's found-rate; retraining on the clip labels (cross-clip) raised its crop AP to 0.92-0.96, found-rate unchanged. | adopted (filter), not a lever |
| Search-window detector (`ballwin.py`): windows around the predicted ball plus a 2x zoom window, full scans every 2 s or when lost | Same recall as full scans within a few points at 6-8x fewer tiles (6.5 min per 2-minute clip instead of 30). The online track needed a velocity cap, decay while unseen and a jump guard. | adopted |
| Multi-frame heatmap detectors (WASB, TrackNetV2/V3) on 512x288 tiles at native, 2x, 4x | WASB with the judge matches RF-DETR and is complementary: the union finds 44% / 78% of labelled balls against 27% / 60%; raw WASB finds nearly every ball of 8 px or more; the far balls come from the 2x and 4x tiles. | open: add as a second source in the search windows |
| Candidate ceiling | A usable candidate exists at ~62% of labelled moments; the misses are mostly balls under 8 px, 43-100+ m away. More pixels per degree at capture (a longer lens or 8K) is the hard limit. | open |

Tracking:

| Technique | Outcome | Status |
|---|---|---|
| Greedy online chain (`ball.py`: static-spot removal, linking under a speed limit, one physically consistent chain) | Tuned on the ball labels from 32% / 26% found to 54% / 62% (static filter 30 s, 2-hit tracks, judge >= 0.3, 0.6 s gap). | available (`OSB_BALL_CHAIN=1`) |
| Static-spot rules: 30 s for a still ball; 8 s in bench cells or outside the lines | Removed the spare balls by the bench that the camera followed for 10 s at a time. | adopted |
| Offline minimum-cost path (`ballpath.py`, Viterbi over candidates with an angular motion cost per phone) | Found 57% / 72-74% (chain 49-57% / 68-72%), coverage up to 44% / 72%; same play boxes; aims at the ball in 39% / 70% of the clips. | adopted (default) |
| Label-time correction (the decoder's late first frame put label images 0.2-0.9 s after their stored time) | Most "wrong" estimates were the clock; with it the tracker shows 1 wrong per clip and the rest are gaps. | adopted |

Ball witnesses from other motion (none moved the tracker's found-rate):

| Technique | Outcome | Status |
|---|---|---|
| Keeper bearings (the keeper stands on the ball-goal line) | The background-blob "keeper" is a defender 14-17 m out; bearings 11-52 deg off. Needs the real keeper (now available from roles). | open |
| Motion-consistent blobs over three frames, judged | 11-19% found alone; in union with the detector the tracker got worse. | dropped |
| Net ripple (change inside the goal mouth) | A specific shot/goal event for the near goal; fires only when the ball reaches the net; the far goal is too small. As a follow-cam feature (`netevent`) it cost a play box. | available |
| Facing direction (COCO keypoints, a ray per player) | Rays within 30 deg of the ball 37-41% of the time (random 17%); the best combination lands a median 28 m off. A weak prior at this player size. | dropped |

## 5. The follow cam with the ball

| Technique | Outcome | Status |
|---|---|---|
| ball: aim at the tracked ball where it exists for >= 0.5 s | 4/13 -> 8/13 play boxes; the misses were exactly the lone-runner and shot moments. | adopted |
| goalin: keep the goal mouth in frame when the ball is within 30 m of it | 9/13; catches the shot ahead of the pack. | adopted |
| ballhold: in a lull, stay on the ball seen in the last 2 s with the players around it | 11/13; fixed a throw-in by the camera where the lull hold reverted to the far side. | adopted |
| Goal-kick vetoes: a tracked ball far from the goal; a confident raw candidate far from the goal with none near it; no goal kick within 8 s of a kick-off; a keeper alone is not a goal kick | Removed false goal-kick framings during open play (13 in one 90-second window) and at kick-offs. A lull-based guard was tried and dropped (no help, cost a box). | adopted |
| netevent | See section 4. | available |

## 6. Teams, keepers, the referee, and kick-offs

| Technique | Outcome | Status |
|---|---|---|
| Kit colour clustering (torso patches white-balanced against the turf, Gaussian mixture) | Clean kits on both clips: blue, solid green, green/white stripes, pink (referee and keeper), yellow (keeper). | adopted |
| Sides by headcount (the two-way split of the kit components with the most equal numbers moment by moment) | Blue vs green + stripes on both clips (44% and 70% margins) without being told the stripes are the green team's second kit. Thresholds tuned on one game; the close-pairs rule (opponents mark each other) is the planned second vote. | adopted, open |
| Roles along a clip (`roles.py`): side, keeper (a solo colour in a goal area), referee (a solo colour elsewhere) | Keepers seen in 85-99% of frames at the goal line (after fixing a tiling bug that hid the far end). | adopted |
| Referee removed from the hotspot; keeper with the ball triggers the goal-kick framing | 11/13 -> 12/13 play boxes (the goal kick where the ball sat in the keeper's hands). | adopted |
| Kick-off from the sides alone (both teams in their own halves, nobody running) | Too brief a signature: stragglers still cross while the ball waits at the centre. | dropped |
| Kick-off from the ball parked at the centre spot for 3 s with hardly anyone running, sides nearly sorted; hold the centre circle until play moves | Fires at the first kick-off (held 0:46-1:10), at 8:54 and at the second-half kick-off (50:01); not at brief stops at the centre. | adopted |

## 7. Production shape and cost

| Technique | Outcome | Status |
|---|---|---|
| Chunked full-game runner (`fullgame.py`): ball stage one chunk ahead of the render, resumable, audio muxed at the end | A game in ~5-6 h on the Mac with the ball candidates reused, ~8-9 h from scratch. Lessons: keep the Mac awake, stop 1 s before the end of the last file (decoder hang), exit stages hard after writing. | adopted |
| Scan cadence 4 s / 1 s instead of 2 s / 0.5 s; ball stage at 5 fps | Each cuts the ball stage 30-50% for a small loss (found-rate -3 to -6 points, one play box). | available (env) |
| Cloud GPU (NVIDIA, fp16/TensorRT, NVDEC/NVENC) | Estimated 2-3 h per game; not built. | open |
| Pieces named by their configuration, so a rule change re-renders only the affected chunks; roles sharing the ball stage's decode | Not built. | open |

## 8. Hard limits found

- **Resolution of the far side.** At 2.7 m camera height, one panorama
  pixel at the far touchline is ~2 m of depth; a ball 60-100 m away is
  4-8 px. This bounds player tracking on the far side, ball detection
  (the ~62% candidate ceiling), facing direction, and the follow cam's
  zoom (nothing below ~55 deg adds detail). Capture is the lever: a
  higher mount, a longer lens, or 8K.
- **Keepers vanish from background subtraction.** A still keeper is
  absorbed into the background model; anything that needs the keeper
  (bearings, goal kicks) needs a person detector, which roles now gives.
- **One game of data.** Every threshold here was tuned on one night game
  and 12 + 90 labelled moments; the fine-tuned detector is overfit to
  it. A second game, ideally daylight at another venue, is the next
  thing to label.
