# Findings: what we tried and what happened

A summary of the processing techniques tested between 2026-10-12 and
2026-11-05, organised by problem. Most were developed on the first
night game (rec3, 2026-10-14, 97.5 minutes, two Pixels on a fixed
mount); the record now also covers rec4 (a second night game, 115
min), rec5 (Pixel + iPhone, orange ball, yellow paint), rec6 (the first
daylight game) and rec7 (Pixel + iPhone, a ~6.2 m mount, yellow ball).
Each entry gives the technique, its measured outcome, and its status.
The dated detail, with numbers per run, is in `docs/BACKLOG.md` (Phase
0 section); the measurement methods are in `docs/TESTING-PROCESSING.md`;
the recordings are listed in `docs/DATA.md`.

Status key: **adopted** (in the default configuration), **available**
(implemented, off by default), **dropped** (tested, no gain), **open**
(promising, not finished), **superseded** (replaced by a later row),
**caveat** (a limit on the measurements, not a technique).

Measurements referred to below:
- *play boxes*: reviewed must-see moments; a moment is captured when
  >= 98% of its box is in the 1080p view. rec3 has 26 (12 on two
  2-minute clips, 14 from the game's first 23 minutes), rec4 15 (game
  10:00-30:00), so the bench is /41; rec7 has 5 on one piece.
- *ball labels*: the ball's position every 2 s along rec3's two clips
  (37 and 53 moments with the ball on the pitch, 90 in all); the tracker
  is scored as found / wrong / missed per moment. Some of these labels
  are still Claude's unedited drafts (see section 4).
- *path share / aim share*: on games without tracker labels (rec5, rec6, rec7),
  the share of frames with a ball estimate or with the camera aimed at
  the ball. Coverage, not accuracy.

## 1. Synchronising and stitching the two phones

| Technique | Outcome | Status |
|---|---|---|
| Audio sync by band-limited GCC-PHAT (whitening only 150 Hz-8 kHz; full-band PHAT locked on a false zero-lag peak), per segment | Locks in every segment of every game so far. Clock drift not constant: rec3 -3.9 ppm in segment 0, settling at -8 to -9.4 from segment 5; rec4 -5.6; Pixel + iPhone (rec5) +14. The offset jumps at segment switches (rec3 up to 390 ms; rec4 ~120 ms per switch, 2.93 s by the end) and the phones' own timestamps don't show it, so every segment is re-synced. | adopted |
| Segments paired by time, with a second overlapping file read mid-segment (`secondaryPieces`) | iPhone (300 s) and Pixel (~285 s) segments drift 15 s apart per segment; index pairing froze the iPhone half for ~a quarter of rec5 and start stamps were up to 3.9 s off. Now all of rec5's pairs lock; raw rec5 matches a hand-built pairing to 0.02 ms. | adopted |
| OpenCV panorama stitcher for the camera geometry, refined at 1920 px (focal lengths, two radial terms, Secondary's rotation) | On rec3 one calibration from segment 0 holds for the whole mounted game (overlap points ~1-4 px median); alignment error at 4K halved by the refinement. Not on every game: failed on rec5 (0/8 pairs; narrow overlap, soft iPhone; cameras set by hand from paint and gravity), 3/8 on rec4 and rec7, 1/8 on rec6; the same Pixel 11's focal comes out 4% apart on rec6 and rec7. Handheld footage drifts. | adopted, open |
| Lens model per phone model (focal, k1, k2), focal pulled toward the manifest's FOV | Needed because the overlap can't separate focal from angle (rec7's near touchline bent ~1 deg). rec4cal used Main's factory focal and a fitted 726 px for the Pro (factory 670 gave a 21 px stitch error). Not built. | open |
| Line fit after the SIFT refine (`follow.refine_lines`): painted-line centrelines from each phone's median background in 6 segments, both lenses, the lens offset (parallax on a level ground) and Secondary's rotation per segment; the rotation most segments agree on is kept | rec7, on segments the fit didn't see: paint at the seam 1.80 -> 0.76 px median at 4K (p90 7.4 -> 2.9) on segment 6, 2.57 -> 0.82 on 15; lens offset ~7 cm (6.6-6.9 cm across in single-segment fits, the rec5 hand calibration's 7 cm). The SIFT matches had been mostly sky, trees and turf texture. Found rec7's mount turned ~0.4 deg between segments 1 and 4: one rotation can't fit both (segments 0-1 get worse, 3.6 -> 5.1 px); flagged as "mountMoved". New games only; rec6/rec7 cameras not rebuilt. | adopted, open |
| Calibrate from the last settings lock within segment 0 (not the whole game's last) | rec6 and rec7 had been calibrated from the game's last lock (rec7 re-locked at ~52 and ~75 min), in effect from all of segment 0, warm-up included. Fixed for new games; their cameras not rebuilt. | adopted |
| Homography fitted on field points | Worse than the stitcher: turf texture gives mostly false matches. | dropped |
| Rendering straight from both 4K sources through a virtual camera, each pixel from one phone along a per-frame seam that avoids players, two-band blend | Fixes the doubled players that blending produced (the phones never capture the same instant). | adopted |
| GPU pipeline (PyTorch on MPS/CUDA, hardware decode and encode, NV12 throughout) | 20 s of video in 64 s on an M-series Mac; a full game renders in ~2.5 h locally. | adopted |
| Clips that run across segment boundaries; a decoder pre-roll after seeks | The pre-roll removed a 0.25-0.5 s freeze at the start of every render; a ~0.2 s hold remains at each segment switch (a real gap in the recording). | adopted |
| Audio vs video alignment (motion matched across the overlap) | Agreed with the audio offset within ~1 frame where it correlated clearly (4 of 9 probes on rec3), noise elsewhere; the clap test is still to do. | open |
| iPhone focus: settle-then-sweep lock in the app (rec5's iPhone half was soft throughout) | On a tester's 40 s clip the locked stretch is as sharp as the best autofocus frames (unverified measurement); not yet seen on a game. | open |

## 2. Mapping the pitch and the goals

| Technique | Outcome | Status |
|---|---|---|
| Pitch pose fitted to the painted lines (camera position and height, rotation, ground tilt, length and width, plus a smooth vertical-angle correction) from a median background panorama | rec3: 109.3 x 60.5 m, camera 2.7 m up. On every other field the lines-only fit ran off or settled wrong (rec4, rec5, rec6, rec7); they were fitted with taps. The lines alone can't tell scaled-down markings from a smaller pitch: the 9.15 m circle sets the scale. | adopted |
| Line mask keeping faint far paint (saturation < 100, hue > 20) | Recovered rec4's far lines (lost to a colour cast); moved the rec3 fixture's fit (below). | adopted |
| Line paint colour: each colour fitted, the paint measured (Lab b* against the turf) under the best fit's near markings | Right on rec3, rec4, rec5 (white b* -10 / -12, yellow +3 against a +1 cut-off). Picking whichever finder fits best chose white on the yellow field. Far yellow paint under floodlights is not detectable. | adopted, open |
| Start pose by trying camera heights 1.9-10 m, best score on the paint | Picks rec7's high mount (6.5 m, 6.2 m after taps) and gives rec6 a better start than the 1.9 m default; fields still need taps for a good fit. | adopted |
| Pitch keypoint taps (`pitchtap/`, `pitch.py <rec> taps`): corner flags, box corners, post bases, centre | The only way to pin rec4's ends (automatic junctions found one box of four); used on rec4, rec5, rec6 and rec7. | adopted |
| Flat-pitch model on a field that drops away | rec4: no flat fit holds the near touchline (2.6 m below the camera) and the centre and ends (3.1 m); a crown term doesn't help. Height held at the measured 2.6 m is used. | open |
| Depression term w3 in the warp, fitted with taps only | rec7 near touchline 0.5 -> 0.27 m off. Absorbs a stitch focal error (6.2 deg, expected ~1); runs away in a lines-only fit. A lens table per phone model is the real fix. | adopted, open |
| Taps stored as pixels in each phone's frame (`jointfit.py <rec> taps` -> `pitch/taps_src.json`, with hashes of the taps, panorama and cameras) | Exact conversion through the cameras the taps were made on; survives rebuilding cameras.json and the panorama. Done for rec4cal, rec5-game, rec6, rec7. | adopted |
| Joint fit of both lenses (focal, k1, k2), Secondary's rotation, the lens offset and the pitch pose, in each phone's 4K pixels, no warp (`jointfit.py <rec> fit`), with taps, 2026-11-09 | Tap error at 4K, median / worst, against the current stitch + warped pitch: rec7 6.9 / 20.6 -> 1.8 / 7.6, rec4cal 2.5 / 35.4 -> 1.2 / 3.4, rec6 4.9 / 16.5 -> 2.7 / 7.0, rec5-game 6.5 / 49.6 -> 2.9 / 45.2 (two box taps ~45 px off in every fit). rec7's near touchline lies on the paint with no w3. rec4 traced-marking benchmark in 4K px: 4.5 -> 3.0 median, centre circle 22.8 -> 1.5, halfway 17.1 -> 8.8, near touchline 7.0 -> 9.2 (the flat-pitch problem above). Main (a Pixel 11 in all four) comes out at 743.7-751.7 px @ 1920 against the stitcher's 733.6-763.5; factory 752.8. In the follow cam (mirrors of rec4cal and rec7, ball candidates and roles carried across): play bench 13/20 -> 16/20 (rec4cal 9/15 -> 12/15, rec7 4/5 both); the seam is no better (refine_lines is for that). Adopted on rec4cal and rec7 (old files in <rec>/pre_joint/); with ballwin rerun on both geometries by today's code (rec7 with judge_v4), rec4cal 11/15 -> 13/15, rec7 4/5 both. | adopted on rec4cal, rec7 |
| Joint fit without taps, focal priors from the lens table (`lenses.json`, the recording itself left out) | rec7 3.1 / 9.5 px (better than its current tapped fit); rec6 6.4 / 16.0 (about equal); rec4cal 35.3 / 78.0 (pitch length runs to 117 m; 4.1 px on a run with a 1% different prior); rec5-game fails (yellow far paint not found in the phones' frames). The weak point without taps is finding the far ends' paint, not the lenses. Loose priors on the lens offset and distortion (10 cm, 0.05) let the offset run to 20-36 cm. Started from the tapped pitch.json, so better placed than a real untapped game. | open |
| Goals as goal-shaped white structures near each end, every left/right pair tried by refitting, validated against the markings | Corrects the map on rec3 (the right end had been ~20 m off); the follow cam's goal rules only worked once the goals were detected. Rejected on rec4 and rec7 (rec4's fitted by hand from the tapped posts); not run on rec5 (spare goals along the side). | adopted, open |
| Regression test on saved background panoramas (`processing/tests/test_pitch_regression.py`) | 15 key points within 0.1 deg, goals within 0.3 m; one fixture (rec3). Failing 4 checks since the line-mask change for rec4 (2026-10-22): right goal 7.8 m against 7.4, length +1.1 m, lines-only points up to 0.37 deg. | adopted, failing |
| Automatic starting pose from the centre circle; youth and small-sided layouts | Not started (the multi-height start above is a search, not a measurement). | open |

## 3. Where is play? The follow cam without the ball

| Technique | Outcome | Status |
|---|---|---|
| Median of all player blobs | Baseline; lags behind quick play. | dropped |
| Speed-weighted centroid of players (player "vectors") | Caught a quick counter the median missed, but jumps between groups. | superseded |
| Convergence point of running players (`play.py`) | Best at a 50 deg view on a 20 s test (78% vs 65%), noisier aim; scored against the early tracker only. Not wired into follow.py. | open |
| Hotspot of running: players pooled over +-0.5 s, weighted linearly by speed, peak by mean shift, sticky (another peak must be 50% stronger for 0.7 s) | Followed an attack to goal; aim jumps > 40 deg/s fell from 435 to 73 frames. Needed linear (not squared) weights and 0.5 s position smoothing, because far-side jitter reads as running. | adopted |
| Only people inside the lines; bench and subs ignored. First a 4 m grid of cells by the touchline occupied >= 10% of the time; since 2026-10-23 bands along the lines (outside a line or < 0.5 m inside, 2 m on the far touchline, in a 4 m stretch busy >= 10% of frames) | Removes the bench from the aim. The grid, computed per clip and dilated, reached ~7 m into rec4's pitch and dropped 32-49% of all detections; the bands drop none there. | adopted |
| Lull hold (<= 1 runner for 1 s keeps the aim from just before) | Stops the camera drifting to where players wait. | adopted |
| Restart taker (a still player alone in a goal area, at a corner or on a touchline during a lull) | Frames the taker and the group. Flashed briefly in short lulls and framed a lone keeper 40 m from play (rec4 25:20); since 2026-10-23 the taker must be where play just was (the ball in the last 5 s, else the held hotspot). | adopted |
| goalframe: >= 4 players within 20 m of a goal while the hotspot is elsewhere, frame both | Catches goalmouth action the hotspot leaves. | adopted |
| goalkick: a lone near-still player in a penalty area soon after play was there, held as a state until play resumes | Catches goal kicks; false positives fixed later by three vetoes (see 5). | adopted |
| unsure: few runners near the hotspot keeps the view wide | Catches quick clearances at the price of width. | adopted |
| lead (aim 1 s ahead), nowalk (walkers weigh nothing), fastboost (steeper speed weight), weak (frame everyone when the signal is thin), allnear, flow (aim along the runners' velocity), lone (include a fast runner elsewhere), gk2 (goal-kick cooldowns), bigger hotspot bandwidth, no lull | None beat the baseline on the play boxes (4/13). The misses were aim, not width: the hotspot stays on the pack while one player has the ball. | dropped |
| Hotspot "weak signal" diagnosis | On the second clip only 10-18 of ~24 people are tracked per frame and few register as running; every rule degrades together. On rec4 pass A tracks 7-12 of the 24-29 people roles.py detects, and roles.py at half resolution sees far-touchline players at ~20 px. Better player detection (RF-DETR people instead of background blobs) is the real fix; peoplesurge is a first use of it. | open |
| Zoom floor 50 deg (a 1080p view narrower than ~55 deg upsamples the 4K source; 65 deg is pixel-for-pixel at a phone's centre) | Sharper output, median width ~63 deg. | adopted |
| Never show past the canvas's left or right edge: stop there and zoom in (to 40 deg; the phones' edges have the most pixels per degree) | Removes the black edge; a further margin for lens fall-off is still to be measured. | adopted, open |
| Camera log (`camlog.txt`, one line per change of the driving rule) | Made the full-render reviews possible. | adopted |
| surge: >= 60% of >= 4 running players heading the same way along the pitch; at first aim 5 m ahead of the frontmost, then frame the 5-7 forward-most non-keepers plus 5 m ahead; since 2026-10-23 frame the runners heading the surge's way plus 5 m (fit within 70 deg) | Fires in 19% of frames but moves the aim only ~4.5 deg at the median: the hotspot's running pool already sits near the front. Front-five version on rec3: players only 13 -> 14/25, with the ball 19-20/25 (no gain). The front five took in subs on the touchline (rec4 28:47); framing the runners: rec3 21 vs 20/26, rec4 1/9 vs 0 and 4/6 vs 5. Can't catch a set piece (7:34): nobody runs. | available |
| peoplesurge: surge on roles.py's detected people (velocity from the nearest person +-0.5 s) instead of pass A's tracks | rec4 10:00-20:00 5/6 vs 3/6; the combined bench 29/41 with it, 28/41 without. Needs roles. | available |
| sideline: hotspot in a near-touchline corner, frame everyone in that corner with padding | Fires in 0.6% of frames: the hotspot drifts to the pack even when play is on the touchline (at the 10:02 throw-in it sat mid-pitch). Neutral; a wider zone costs coverage. The trigger would need the ball or the thrower. | available |
| Front-of-field position (the forward-most players parked far from the hotspot, a set piece) | Seen at 7:34: the five forward-most players at the left goal line 40 m beyond the hotspot, with nobody running. Not built. | open |

With the ball (section 4) and roles (section 6): **21/26 play boxes** (2026-10-20). On the first 12 moments 11/12; the 14 from the full game added three misses (a set piece with no ball candidate at 7:34, a throw-in with the ball in hands at 10:02, a single ball sighting at 14:05) and two 95% near-misses.

With rec4's 15 moments added: 25/41 at first, **29/41** (rec3 20/26, rec4 9/15) after four changes on 2026-10-23 (loiter bands, the two-ball jump filter, restart gating, surge framing its runners), each tested alone first; the jump filter and restart gating only paid off together. The linker fixes of 2026-10-28 kept 29/41; scanning every 0.2 s while lost gave 30/41 but was not adopted (17-38% more ball-stage time). Still missed on rec4: a false far-left ball (18:27), a second ball by the touchline (26:31), 26:49, 28:04, 29:56. A box counts as captured when 98% of it is anywhere in view, even at the edge; a centred test is not built.

## 4. Where is the ball?

Detection:

| Technique | Outcome | Status |
|---|---|---|
| Stock RF-DETR "sports ball" on 640 px tiles of the 4K frames | 58% of match balls at conf >= 0.2, 2.1 wrong suggestions per frame. | superseded |
| Fine-tuned RF-DETR Medium on 275 labelled balls (640 px tiles, hard negatives) | 76% of match balls on held-back frames at 1.1 wrong per frame (stock: 69% at 10.5); the top suggestion is the ball 69% of the time. Overfits this game (96% on training frames). | superseded |
| Detector and judge retrained with each new ball and light (rec3 + rec5 orange x3, 2026-10-30; + rec6 daylight x2, 2026-11-01, the standard pair in `footage/models/`) | Without labels for a new ball colour the judge rejects the ball: rec5 path 7%, rec7 aim 0-1%. rec5 7% -> 34%; rec6 path 30% -> 52%; rec4cal 9 -> 10/15; rec3 20/26 throughout but labelled found 68 -> 63/90 over the two retrains (wrong 5 -> 3). The new detector needs the new judge. The standard judge turned out to be trained on rec3 x4, rec5 x9, rec6 x2 (a merge bug, fixed); the fixed mix benched no better. | adopted |
| Yellow-ball judge (judge_v4: + rec7 x4, 119 labelled frames) | rec7 piece 2 (contains the labels): aim at the ball 1% -> 64%, play boxes 0/5 -> 4/5 (unverified); other pieces 28-63%. Against judge_v3 on the other games (2026-11-09): rec3 and rec4cal play benches even; rec3 balleval seg3 42 -> 41/53, seg9 21 -> 16/37 (wrong 1 -> 4); ball path share rec5 39 -> 40%, rec6 52 -> 45%; rec7 chunks 0-1% -> 17-47%. | available (`OSB_JUDGE=judge_v4`) |
| Detector and judges on an unseen camera (SoccerTrack wide view, 979 frames from 66 clips, 2 university games, daylight, white ball; `stbench.py`, 2026-11-09) | Detector: the ball among its candidates in 530/979 frames (54%), 1.9 candidates a frame; ranking candidates by its confidence alone puts the ball on top in 461/979 (AP 0.81). Judges: keep the ball (>= 0.5) in 36 (judge_v3) / 25 (judge_v4) of 979 and score clear white balls on grass near 0; ranking by confidence x judge picks the ball 377 / 342 times (AP 0.36 / 0.34), worse than the detector alone. The detector carries to a new camera; the judge has learnt our footage, not the ball. | open |
| The judge against the detector alone, and a size check in its place (2026-11-09; `balleval.py` with OSB_NOJUDGE=1, `ballpath.SIZE_W`) | rec3 seg9 (held out): judge_v3 found 21/37 with 1 wrong, the detector alone 19/37 with 9 wrong (and a ball at 5 of 12 'none' moments against 1); seg3 42 vs 43/53, 2 wrong both. rec7 seg6 (yellow ball, full scan; neither model trained on rec7): judge_v3 found 0/52, the detector alone 38/52 with 1 wrong. The detector-alone wrong picks are ball-sized white boots, feet, line crossings and a goal net, so a size check (box against the pitch map's expected ball size) barely helps: seg9 wrong 9 -> 7, rec7 and seg3 unchanged, SoccerTrack AP 0.806 -> 0.813 when set for that camera and worse with rec3's setting. The judge's value is rejecting same-size look-alikes on footage like its training; on a new ball or camera it removes the ball. | size check dropped (`SIZE_W` stays 0) |
| Judge given the expected ball diameter at the candidate | Held-out clip AP 0.915 -> 0.936, but no bench moved beyond noise (rec3 21 vs 20/26, rec6 path 48 vs 49%). | dropped |
| Frame differencing (motion candidates) | 55% of match balls at ~120 candidates per frame; ranking by contrast is poor. Half the misses are the ball merged into a player's blob, which the detector finds. | dropped |
| Crop judge (48 px crop classifier, 1.5 ms per crop) to rank candidates | Ranks about as well as possible (top-1 62% against a 77% ceiling) and cleans candidates (zero wrong tracks), but does not raise the tracker's found-rate; retraining on the clip labels (cross-clip) raised its crop AP to 0.92-0.96, found-rate unchanged. | adopted (filter), not a lever |
| Search-window detector (`ballwin.py`): windows around the predicted ball plus a 2x zoom window, full scans every 2 s or when lost | Same recall as full scans within a few points at 6-8x fewer tiles (6.5 min per 2-minute clip instead of 30). The online track needed a velocity cap, decay while unseen and a jump guard. | adopted |
| Multi-frame heatmap detectors (WASB, TrackNetV2/V3) on 512x288 tiles at native, 2x, 4x | WASB with the judge matches RF-DETR and is complementary: the union finds 44% / 78% of labelled balls against 27% / 60%; raw WASB finds nearly every ball of 8 px or more; the far balls come from the 2x and 4x tiles. | available |
| WASB as a second source in the search windows (`OSB_WIN_WASB=1`: fp16, 2x/4x tiles in full scans plus one window) | Tracker found 57 -> 62 of 90 labelled balls, wrong 4 -> 2; play boxes 27 -> 26/41 (rec3 18 -> 16, rec4 9 -> 10), within the bench's +-2 rerun noise. Ball stage ~1.5-1.7x. | available |
| WASB fine-tuned on this game's labels with the empty pitch as a fourth input frame (`heattrain.py`) | On held-out clips at ~1.6 peaks a frame: 19% -> 62% (seg 3) and 17% -> 38% (seg 9) of in-play balls. Through the tracker no gain: 61/90 vs stock WASB's 62/90. Between full scans WASB only looks near RF-DETR's track; scanning every frame floods the linker (52/90). The linker is now the limit. | available, not a lever yet |
| Candidate ceiling | A usable candidate exists at ~62% of labelled moments; the misses are mostly balls under 8 px, 43-100+ m away. More pixels per degree at capture (a longer lens or 8K) is the hard limit. | open |

Tracking:

| Technique | Outcome | Status |
|---|---|---|
| Greedy online chain (`ball.py`: static-spot removal, linking under a speed limit, one physically consistent chain) | Tuned on the ball labels from 32% / 26% found to 54% / 62% (static filter 30 s, 2-hit tracks, judge >= 0.3, 0.6 s gap); the same tuning cost play boxes then (10 -> 8/13: the judge rejects a ball in the thrower's hands), recovered by ballwin candidates and ballhold. | available (`OSB_BALL_CHAIN=1`) |
| Static-spot rules: 30 s for a still ball (at 4 s the ball placed for a goal kick was removed); 8 s in bench cells (since 2026-10-23 the loiter bands) or outside the lines; since 2026-10-29 only a run of sightings with no break over 10 s counts | Removed the spare balls by the bench that the camera followed for 10 s at a time. The run rule stopped the match ball crossing one spot twice from reading as static (rec3 0-10 min: 190 -> 24 spots). | adopted |
| Two-ball jump filter (`ballpath.py`: runs too far apart in time and space to join keep the one nearer the players) | Removed a 5 s detour to a second ball (rec4 20:25, 55 m in 1 s); cut jumps of 55-91 m/s and kept all up to 32 m/s on four rec4 chunks. Refined 2026-10-27 so the players decide only when one run is >= 10 m farther from them. | adopted |
| Offline minimum-cost path (`ballpath.py`, Viterbi over candidates with an angular motion cost per phone) | Found 57% / 72-74% (chain 49-57% / 68-72%), coverage up to 44% / 72%; same play boxes; aims at the ball in 39% / 70% of the clips. | adopted (default) |
| Path gap penalty 2 -> 1 (cheaper to enter and leave "no ball", so isolated confident sightings join the path) | Play boxes 19/25 -> 20/25 (19:36); ball labels the same found-rate with fewer wrong (seg 3: 4 -> 2). A cheaper no-ball cost (1.5) or no judge filter each lose a box. | adopted |
| Linker fixes: pitch margin 3 -> 10 m, a ball at rest bridged up to 2 s, two long runs both kept when distance from play cannot pick one (2026-10-28) | Found 57 -> 66 of 90 labelled balls (stock WASB 62 -> 70, tuned 61 -> 69), wrong 4 -> 5; play boxes 29/41 unchanged. A learned per-candidate P(ball) had too few negatives at the labelled moments to fit. | adopted |
| Longer ballhold (5-8 s) | Loses two boxes to stale holds. | dropped |
| Label-time correction (the decoder's late first frame put label images 0.2-0.9 s after their stored time) | Most "wrong" estimates were the clock; with it the tracker shows 1 wrong per clip and the rest are gaps. | adopted |
| Clip labels pre-marked by Claude, then reviewed | 20 of seg 9's 118 frame labels and 33 of seg 3's 102 are still unedited drafts (`draftBy: claude`), mostly tiny far balls, so the found-rates lean on the detector's own view there. | caveat |

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
| Kit colour clustering (torso patches white-balanced against the turf, Gaussian mixture) | Clean kits on both rec3 clips: blue, solid green, green/white stripes, pink (referee and keeper), yellow (keeper). White kits fall under the saturation floor (rec4, rec5); white and yellow merge under floodlights (rec7). | adopted, open |
| Sides by headcount (the two-way split of the kit components with the most equal numbers moment by moment) | Blue vs green + stripes on both clips (44% and 70% margins) without being told the stripes are the green team's second kit. Thresholds tuned on one game; the close-pairs rule (opponents mark each other) is the planned second vote. Failed on every game since (rec4 two shades of red as the sides, rec5 one kit component, rec7 wrong split): sides set by hand in `sides.json`. | adopted, open |
| Roles along a clip (`roles.py`): side, keeper (a solo colour in a goal area), referee (a solo colour elsewhere) | Keepers seen in 85-99% of frames at the goal line (after fixing a tiling bug that hid the far end). | adopted |
| Referee removed from the hotspot; keeper with the ball triggers the goal-kick framing | 11/13 -> 12/13 play boxes (the goal kick where the ball sat in the keeper's hands). | adopted |
| Kick-off from the sides alone (both teams in their own halves, nobody running) | Too brief a signature: stragglers still cross while the ball waits at the centre. | dropped |
| Kick-off from the ball parked at the centre spot for 3 s with hardly anyone running, sides nearly sorted; hold the centre circle until play moves | Fires at the first kick-off (held 0:46-1:10), at 8:54 and at the second-half kick-off (50:01); not at brief stops at the centre. | adopted |

## 7. Production shape and cost

| Technique | Outcome | Status |
|---|---|---|
| Chunked full-game runner (`fullgame.py`): ball stage one chunk ahead of the render, resumable, audio muxed at the end | A game in ~5-6 h on the Mac with the ball candidates reused, ~8-9 h from scratch. Lessons: keep the Mac awake, stop 1 s before the end of the last file (decoder hang), exit stages hard after writing. Other games run slower: rec5 11.0 h; rec7's ball stage 70-75 min per 10-minute chunk (~2x rec4cal's pace), 38-51 min once judge_v4 held the track. | adopted |
| Engine commands (`engine.py`: `prepare`, answers as files, `run`; JSON progress, exit codes, resume) on a fresh copy of rec7 (M2 Pro, Core ML detectors, 2026-11-08/09) | `prepare` reaches the questions in 14 min (sync 2.6, pitch 3.8, goals 0.4, kit sample 5.4, kit groups 1.9). The automatic pitch fit was wrong (as on every field but rec3); one round of tapped points fixed it (102.4 x 60.7 m, camera 6.35 m up; hand fit 102.3 x 60.5, 6.23). The 8 kit groups mixed yellow and white in one group (purity 0.855); after "split", 10 groups put yellow against white on their own and one merge round left 0.962. `run`: the whole game in ~8.5 h (ball stage with roles 44 min per 10-minute chunk, 33-50; render 24 min; join and 1440p scale 27 min), so the render waits on the ball stage about half the time. A stop (SIGTERM) after chunk 0 exited 130 with nothing left running; the same command resumed at chunk 1. Bad job, missing footage or answers exit 2; questions unanswered exit 3. The first try found two bugs: a fresh recording has no `detect/` (every ball chunk failed at its last write) and `fullgame.py` carried on for 4 h after the first failure; both fixed. | adopted |
| `game.mp4` scaled up to 1440p for YouTube (`fullgame.py --height`, default 1440; 1080 keeps the render) | The 1080p upload played as AV1 at YouTube's low 1080p bitrate and lost much of the detail. rec7's piece 2 uploaded at 1440p, 4K and 1440p with `hqdn3d` grain reduction all played as VP9 and looked substantially better; the three looked about the same, so plain 1440p. Upscaling adds no detail (a 54 deg view spans ~1,420 source px); the gain is YouTube's bitrate for that resolution. +26 min and ~22 GB for a 95 min game. | adopted |
| Scan cadence 4 s / 1 s instead of 2 s / 0.5 s; ball stage at 5 fps | Each cuts the ball stage 30-50% for a small loss (found-rate -3 to -8 points, the -8 for 4 s scans on seg 9 from transcript figures; one play box). | available (env) |
| Stage profile of one 10-minute chunk (rec4 30:00-40:00, M-series Mac, GPU-synchronised timers around the spikes, 2026-11-06) | RF-DETR is the cost: 81% of the ball stage (42.4 min; 52,106 crops at ~35 ms, 79% of them from full scans: 984 scans in 600 s) and 95% of roles (23.8 min at 4 fps, 8 crops per phone). Decode is not: ~1% busy on its own thread, the main loop waits on it < 0.1%. Also in the ball stage: whole-frame NV12 -> RGB on the GPU 11%, the 4K frame copied to the CPU for judge crops 6%. Render (rec7, 2026-11-05): pass B 58%, pass A warp+blend 20%, seam 10%. RF-DETR per crop on MPS: Medium 46 / 32.5 / 28.1 ms (batch 1 / 8 / 8 fp16), Small 38 / 25 / 22, Nano 25 / 14 / 13. Details in BACKLOG.md. | measured |
| Scan cadence on the must-see boxes: fresh ball candidates per cadence on every clip with reviewed boxes (rec3 5 sets, rec4 2, rec7 1; 46 moments), production features, 2026-11-07 | Full / lost scan every 2 / 0.5 s (today): 33/46, ball stage 235 min over the 8 clips, seg3 / seg9 labels 81% / 62%. 4 / 1 s: 32/46, 178 min (-24%), 75% / 62%. 2 / 1 s: 32/46, 195 min (-17%), 77% / 59%. 2 / 2 s: 32/46, 166 min (-29%), 72% / 57%. All three lose the same two rec3 moments (14:05 100 -> 0, 7:51 100 -> 95) and gain rec4 18:27 (0 -> 100), so the lost-scan interval decides it, not the full-scan one. At 14:05 a long kick moves the ball ~45 degrees; the 0.5 s rescan finds it, the 1 s rescans each land on a parked ball by the sideline (yaw 84, judge 1.0) until 14:09. Details in BACKLOG.md. | measured |
| Full scans planned from the pitch map (`ballwin.py`, now the default; `OSB_SCAN_PLAN=grid` for the old grid): full-size tiles only where the ball would be under 20 px, the near side in 1280 px crops shrunk to 640, tiles off the pitch dropped, 2026-11-07 | Scan crops per frame pair 42 -> 22 (rec3), 19 (rec4), 33 (rec7). On the must-see bench at today's cadence: 34/46 captured (grid 33), ball stage 183 min against 235 (-22%; rec3 -21%, rec4 -30%, rec7 -10%), seg3 / seg9 labels 83% / 65% (grid 81% / 62%). No moment worse; rec3 seg3 5:30 0 -> 100. With full scans every 4 s as well: 33/46, -25%, rec3 7:51 100 -> 90. Details in BACKLOG.md. | adopted |
| Lost-ball rescans limited to the pitch near the last sighting (`OSB_LOST_LOCAL=1`, branch `ballwin-local-rescan`), whole scan still every 2 s, 2026-11-07 | The ball is lost (>= 1 s unseen) in 38% of frames and 77% of full scans are lost-ball rescans, but making them local did not speed the ball stage: 10 m + 10 m/s radius +0.6% time (11,020 s vs 10,959 over 8 clips) with identical must-see coverage (34/46) and labels; 5 m + 5 m/s -2% on the first 4 clips before the run was stopped. Per-frame work, not scans, is now most of the ball stage. Details in BACKLOG.md. | tried, not adopted |
| Peak memory per stage (rec7 2-minute clip, `footprint` every 2 s, GPU included, 2026-11-07) | Ball 5.2 GB (Python 4.3, of it GPU 3.1; ~0.6 per ffmpeg decoder); roles 3.4 GB (Python 2.3, GPU 1.2); render 3.3 GB (Python 2.1, GPU 1.0). Ball stage and render overlapped, as `fullgame.py` runs them: 8.4 GB, with the render 2x slower (89 -> 171 s) and the ball stage 9% slower. Fits a 32 GB Mac easily; the ball detector's batches and 4K frames are the largest piece. | measured |
| NV12 -> RGB only for the crops read, and the judge's crops copied to the CPU as small patches instead of the 4K frame (`ballwin.py`, now the default; `OSB_CROP_RGB=0` for the old path), 2026-11-08 | Same ball output bit for bit (1,078/1,078 judge crops identical; two bench clips' candidates identical), so must-see unchanged at 34/46. Ball stage -24% on two 2-minute bench clips (524 s vs 687); the 2-minute profile 3.9 min vs 6.2 (-37%); the detector's network is now 77% of the stage. Cutting the judge crops on the GPU instead was -20% over 8 clips but 32/46: judge scores within 0.021 moved the ball in 0.3% of frames and the camera follower carried that for minutes. Details in BACKLOG.md. | adopted |
| Ball detector (RF-DETR Medium) exported to Core ML and ONNX, run outside PyTorch (M2 Pro, 2026-11-08; network only, plus parity on the 451 valid tiles) | Core ML fp32 on the GPU: 25.8 / 21.8 ms per crop at batch 1 / 8 against PyTorch MPS's 44.9 / 31.5, with the same 150 detections and confidences. Core ML fp16 is faster (23.0 / 18.1) but changes the detections (balls found at 0.5: 51 -> 47-50). ONNX Runtime's Core ML provider 31.0 / 26.5 with 27-44 s to load; the Neural Engine 78-86 ms; ONNX Runtime on the CPU 181 ms. In ballwin.py (opt-in `OSB_DETECTOR_COREML`) the must-see clips run 25-35% faster in the ball stage (rec4 10 min: 1355 -> 888 s) with the same play boxes (33/46) and ball labels (seg3 44/53, seg9 24/37). Roles (opt-in `OSB_PEOPLE_COREML`, the stock model) 29-32% faster on the same clips (rec4 10 min: 1346 -> 949 s) with the same people and play boxes. Now the default on a Mac (exported on first use, PyTorch elsewhere or with `=0`). A full game not yet tried. Details in BACKLOG.md. | adopted |
| Game phases from whistles and pitch motion (`phases.py`, 2026-11-08): the referee's whistle from Main's audio (each game's pitch found from its most tonal, loud, long runs: rec7 ~3.65 kHz, 68 runs of ~1,000 tonal ones), per-cell motion on the pitch at 1 fps / 640 px from both phones, a five-state split (pre, half, break, half, post) rewarded for boundaries on whistles and refined under halves-of-equal-length; pre- or post-game under 5 min kept | rec7 (the one whole game on disk): kick-offs within 1 s (9:01, 56:13), half-time end +107 s and full time kept to the end, no play lost, 9% of the recording cut (a perfect cut 17%). All four true boundaries sit on referee whistles (0.6-1.1 s). Partial recordings: rec3's kick-off found at 1:05 (kept, under 5 min); rec5 and rec6 begin in play and keep everything (without the 5-min rule the split invented kick-offs 1:40 and 3:55 in). The pass costs ~22 min per 100-min game (full-rate decode for 1 fps), ~13 s per game minute on the partials. Motion alone put rec7's warm-up in the first half; walking off and the second-half line-up read as play. Short recordings mis-pick the whistle pitch (2.2-2.7 kHz, crowd). Tuned on one game; needs whole games to judge. Details in BACKLOG.md. | open |
| Cloud GPU (NVIDIA, fp16/TensorRT, NVDEC/NVENC) | Estimated 2-3 h per game; not built. | open |
| Pieces named by their configuration, so a rule change re-renders only the affected chunks; roles sharing the ball stage's decode | Not built. The profile above puts decode at ~1% on the Mac, so the shared decode saves little there. | open |

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
  Slowing the background model to ~1 minute helped only partly.
- **Few games measured.** Every threshold was tuned on rec3 (one
  night game, 12 + 90 labelled moments, partly Claude's drafts) and
  tested on rec4's 15 play boxes. rec5, rec6 and rec7 (one daylight, two
  with an iPhone half, orange and yellow balls) have labels for
  training (rec7's a clip, used to train judge_v4) but no held-out
  tracker labels, and only rec7 has play boxes (5). On them, automatic
  sides, the lines-only pitch fit and (on rec5) the stitcher failed and
  were set by hand or with taps. Held-out clip labels on another game,
  ideally daylight, are the next thing to label.
- **Ball colour.** The judge only accepts ball colours it has been
  trained on; each new colour (orange, yellow) needed ~100 labelled
  frames before the camera followed the ball.
- **A flat pitch.** On rec4 no flat model fits both the near touchline
  (2.6 m below the camera) and the centre and ends (3.1 m). The
  follow cam has no curvature term.
