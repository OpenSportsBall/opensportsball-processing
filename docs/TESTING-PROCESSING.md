# Testing the processing pipeline

How the follow cam, the pitch map and the ball tracking are measured, what the
measurements say today, and how to extend them. Everything here runs locally
on the Phase 0 spikes in `processing/spikes/`. See `docs/BACKLOG.md` for the
history of what was tried and the main repo's `docs/TESTING-APPS.md` for the phone apps.

## Principles

- **Footage is private.** It shows identifiable adult players. Every label
  set, fixture and rendered clip lives under `footage/` (gitignored) and never
  enters the repository. Back them up privately; the tooling and the numbers
  are what the repo keeps.
- **Score against reviewed labels, not by eye.** Each algorithm change is run
  through the harness below before a render. Watching renders is for judging
  what to label next, not for deciding between variants.
- **Every ball labelling holds some back.** Whenever balls are labelled,
  part of the new labels is set aside as a test set before any model
  is trained on them, and no detector, judge or heatmap model trains on
  that part afterwards. Without it the only test of a model is the
  game it learnt from (see "Held-out ball labels" in section 3).
- **Small sets are noisy.** 13 play moments and 90 ball moments: a difference
  of one or two is noise. Treat them as a direction, and add moments before
  tuning further.

## 1. Pitch and goal mapping

`processing/tests/test_pitch_regression.py` reruns the whole mapping (line
fit from the starting guess, goal detection, refit with the goals) on saved
background panoramas and compares it with a result that was checked by eye.

    uv run --with pytest --with numpy --with scipy --with opencv-python-headless --with torch \
        pytest processing/tests/test_pitch_regression.py -v

The joint fit of both lenses and the pitch (`jointfit.py`) and the split
into periods of play (`phases.py`) have tests that need no footage: a
made-up mount and pitch whose paint and taps the fit has to recover, and
made-up recordings (two halves, extra time, two games, quarters, one
period, nothing but play, cut short) the split has to get right.

    cd processing && uv run --extra dev pytest tests/test_jointfit.py tests/test_phases.py

Goals staying with the pitch map they are drawn on (in the kept map's
metres when rejected, found again on a tapped map's goal lines, a tapped
post fixing its goal) is tested without footage too, on goal frames
painted into an empty panorama:

    cd processing && uv run --extra dev pytest tests/test_pitch_goals.py

A Secondary with only part of the game (it stopped early, started late, or
has a hole) is tested the same way, on made-up recordings
(`tests/test_secondary_gaps.py`): what the sync's report says Secondary
covers, the blank frames the readers get where it has nothing (and exactly
as many frames as Main's), the view staying inside Main's frame there,
corners included, and the engine fitting the pitch on a segment both
phones have. On footage: cut a recording's manifest so Secondary's later
files are gone and run `phase0.py`, then `fullgame.py --start` across the
point where it ends.

    cd processing && uv run --extra dev pytest tests/test_secondary_gaps.py

How close a joint fit puts the taps is measured on taps it wasn't given
(the fit's own taps flatter it); on a recording whose joint fit is adopted,
start from the files it replaced:

    uv run processing/spikes/jointfit.py footage/rec7 holdout --start footage/rec7/pre_joint --out /tmp/rec7-holdout

Fixtures: `footage/bench/fixtures/pitch/<name>/` (`background.png`, `expected.json`),
or `$OSB_PITCH_FIXTURES`. The test skips when none exist. Make one from a
recording whose `pitch.json` and `goals.json` you have verified:

    uv run processing/spikes/pitch.py footage/rec3 fixture footage/bench/fixtures/pitch/<name>

Tolerances: both goals found and within 0.3 m; goal frames within 1.5 px of
the white structure and end markings within 3.5 px of the paint; pitch
length and width within 1 m, camera height within 0.2 m; 15 key points
(corners, centre and penalty spots, post bases and tops) within 0.1 deg,
the lines-only fit within 0.3 deg. One fixture so far: the night game
(`night-2026-10-14`, rec3). It has failed 4 checks since 637b3ba
(2026-10-22), the line-mask change that recovered rec4's faint far paint:
rec3's right goal comes out 7.8 m wide against 7.4, the length +1.1 m, and
the lines-only key points up to 0.37 deg. The fixture's left goal was
updated to the new mask (7.6 -> 7.4 m); the right was kept until the right
goal is sorted out.

### Marked-field benchmark

`footage/bench/fixtures/pitch/night-2026-10-21/` (rec4, the 2026-10-21 night game) has every pitch marking
traced on the paint and completed, the goal frames fitted to their posts, and 15 keypoints tapped by eye
(`pitchtap/server.py`), in `benchmark.json`; `benchmark.png` shows it. Score a pitch fit against it:

| Step | Command |
|---|---|
| Score fits | `uv run processing/spikes/pitchbench.py footage/bench/fixtures/pitch/night-2026-10-21 <pitch.json> ... [--goals goals.json] [--all]` |
| Tap keypoints on a recording's panorama | preview `pitch-tap-rec4` (`processing/spikes/pitchtap/server.py <recording> [port]`), then `uv run processing/spikes/pitch.py <recording> taps` |

Fits must be made on the fixture's own `background.png` (its `cameras.json`). Baselines and how the
benchmark was made are in the fixture's README and `baseline_scores.txt`.

## 2. Follow cam: must-see boxes

A reviewer draws, on a full-resolution panorama of a moment, the part of the
field a follow cam has to show. A camera path is scored by how much of each
box it shows at that frame.

| Step | Command |
|---|---|
| Panoramas at chosen clip times | `uv run processing/spikes/playeval.py make footage/rec3 <set> <segment> <start s> <clip t> ...` |
| Draw and review boxes | preview `play-eval`, `play-eval-seg9`, `play-eval-seg0`, `play-eval-seg0-600`, `play-eval-seg0-1200` (`processing/spikes/evalpage/server.py <set dir> [port]`); sets that start at the game's start list game time, the others clip time |
| Score path files | `uv run processing/spikes/playeval.py score footage/rec3 <set> <path.json> ...` |
| Bench many variants | `uv run processing/spikes/playbench.py footage/rec3 footage/rec4cal [variant ...]` (any recordings; each set with a `clip.json`) |
| See what the pipeline saw at moments of a full-game piece (ball candidates, people by role, camera rule) | `uv run processing/spikes/boxes.py footage/rec4 <piece> <mm:ss> ...` (piece time; writes `fullgame/boxes/`) |

Sets: `footage/<recording>/playeval/<set>/` with `clip.json` (the follow.py clip its boxes are
timed in: `segment`, `startS`, `seconds`; the bench runs every set that has one), `pano/`, `items.json`,
`boxes.json` (panorama degrees, so boxes don't depend on any render) and
`paths/<variant>.json` plus `<variant>_debug.json` (per-frame mode, aim,
hotspot, ball). rec3 has five sets, 26 reviewed moments:

| Set | Clip | Moments |
|---|---|---|
| `seg9-60` | segment 9 from 60 s, 120 s (the first test clip) | 7 |
| `seg3-225` | segment 3 from 225 s, 120 s (crosses into segment 4) | 5 |
| `seg0-0` | game 0:00-10:00 | 6:30, 7:22, 7:34, 7:51, 9:45 |
| `seg0-600` | game 10:00-20:00 | 10:02, 12:09, 12:36, 13:37, 13:50, 14:05, 15:31, 19:36 |
| `seg0-1200` | game 20:00-30:00 | 22:36 |

rec4 (the 2026-10-21 night game) has two sets, `footage/rec4cal/playeval/seg0-600` (game
10:00-20:00: 16:37, 16:54, 16:57, 17:11, 17:25, 18:27; preview `play-eval-rec4cal-600`) and
`footage/rec4cal/playeval/seg0-1200` (game 20:00-30:00: 20:27, 25:10, 25:20, 25:59, 26:31, 26:49,
28:04, 28:47, 29:56; preview `play-eval-rec4cal-1200`). 20:27 has a second ball in play in the empty
half, kicked by a group in pink bibs. Their panoramas
and boxes are in the corrected camera model (factory focal length for Main, 726 px fitted for the Pixel 11
Pro, no distortion, levelled to Main's gravity sensor; `cameras.json`
in the set), the same one as the pitch benchmark `night-2026-10-21`, so score only paths made with those
cameras. Both have a `clip.json`, so `playbench.py footage/rec3 footage/rec4cal` benches all 41 moments
together (the pass-A caches for both clips are in `footage/rec4cal/follow/`). `footage/rec4/playeval/seg0-600`
holds the same boxes in the first camera model and is kept only to score the first full-game run (no
`clip.json`: not benched).

rec7 has one set, `footage/rec7/playeval/seg0-1484` (5 moments on the full-game run's piece 2), part of
the standard bench since 2026-11-15: `playbench.py footage/bench/rec3 footage/bench/rec4cal footage/bench/rec7`
benches 46 moments, baseline 36/46 with `peoplesurge` (rec3 21/26, rec4cal 11/15, rec7 4/5) on main `38d6771`
with pass A redone (2026-11-16); earlier figures of 33/46 (rec3 20, rec4cal 9) are probably from pass A caches
made before the fitted pitch. Figures quoted
as "/41" are rec3 and rec4cal alone.

rec8 (the 2026-11-11 night game, run by the Mac app) has four sets on the full-game run's pieces 0, 1, 5 and 6: `footage/rec8/playeval/seg0-0` (9:29, 9:44), `seg0-600` (11:31, 12:18, 13:11, 13:13, 13:28, 13:42, 14:51, 15:02, 17:33, 17:39), `seg0-3000` (54:37; 55:03, 56:39 and 56:54 have panoramas but no box) and `seg0-3600` (1:02:04, 1:03:02, 1:04:27; 1:05:16 without a box). All 16 are plays the follow cam missed while watching the game (the first 10 boxed on 2026-11-14, six more on 2026-11-15: 9:29, 9:44, 11:31, 12:18, 13:13, 13:42; 13:13 is the same play as 13:11, 2 s later, so that play counts twice), not random moments, so a score here is a count of misses fixed and says nothing about plays it already caught. From 54:37 on Secondary has no footage: those panoramas, and the camera, have Main's half of the pitch only. Each set has a `clip.json` (not part of the standard bench) and `paths/fullgame_2026-11-14.json`, the Mac app's run (`goalkick,goalframe,unsure,ball,goalin,ballhold,roles,kickoff`): 0/10 captured, in `lull` at six of them (the held aim, far from the ball; `fullgame/camlog.txt`). `paths/fullgame_2026-11-15.json` is a second run by the same app and engine build (0.1.8, engine `ed0d4d4`) after deleting `fullgame/`, `detect/` and `follow/`: 4/10 (15:02, 17:39, 1:02:04, 1:03:02; still missed: 13:11 at 57%, 13:28, 14:51, 17:33, 54:37, 1:04:27). Nothing but the run differs, so the 4 are run-to-run variation in the ball stage, not a fix: a change scored on these sets needs more than one run each way (BACKLOG.md, "Run-to-run variation"). The figures above are on the first 10 boxes. On the six added on 2026-11-15 the first run captures 2/6 (12:18 and 13:42, both on the `ball` rule there) and the second 0/6: in `lull` at 11:31, 12:18, 13:13 and 13:42 (a tracked ball, 2 s of `ballhold`, then the held aim until the ball is found again), on `hotspot` at 9:29 and on `goalkick` at 9:44 (73 deg wide, aimed away from the box). On all 16: 2/16 and 4/16. The sets were drawn from two engine label sets (`engine.py label job.json moments --at ...`, `moments-1` and `moments-2`) and split by piece.

Bench of both recordings on 2026-10-23 (after the loiter-band, ball-jump, restart and surge changes):
`goalkick,goalframe,unsure,ball,goalin,ballhold,roles,kickoff,peoplesurge` captures 29/41 (rec3 20/26,
rec4 9/15: 10:00-20:00 4/6, 20:00-30:00 5/9); without `peoplesurge` 28/41 (20/26, 8/15). Sets are
benched in name order, so a bench line's per-moment list runs seg0-0, seg0-1200, seg0-600, seg3-225,
seg9-60.

The first two sets had 13 moments until 2026-10-19: the seg 9 box at
1:57 (the keeper's goal-kick framing, 1.5 s before the kick) was removed
because a camera that follows the ball into the corner there is right;
figures quoted as "/13" were measured with it. The three `seg0` sets
(2026-10-19/20) reuse the full-game chunks' ball candidates and roles
(`detect/ball_0_<start>_600.win.json`, `roles_0_<start>_600.json`).

`playbench.py` caches follow.py's pass A (player detections) per clip in
`footage/<recording>/follow/passA_<seg>_<start>_<seconds>.json`, so a variant
plans in seconds (the first run of a 10-minute clip takes ~30 min). It writes
into the recording's `follow/`, as fullgame.py does: don't bench a recording
while a full-game run on it is going (bench a mirror of symlinks with its own
`follow/` instead). Variants are follow.py feature strings; a feature
`NAME=value` overrides a constant for that run (`MIN_HFOV=50`,
`ball.MIN_JUDGE=0`). `OSB_BALL_TAG` selects the ball candidate file
(`win` for ballwin.py's; unset prefers it when present).

A moment counts as captured when at least 98% of its box is in view. The
bench also reports mean coverage and the median view width (narrower is
better once captured, down to the resolution floor: a 1080p view narrower
than ~55 deg upsamples the 4K source).

Results on the 13 moments (2026-10-17):

| Configuration | Captured | Width (median) |
|---|---|---|
| A: `goalkick,goalframe,unsure` (players only) | 4/13 | 54 deg |
| A + 50 deg floor + edge rule, no ball | 5/13 | 59 deg |
| A + ball + goalin (full-scan candidates) | 9-10/13 | 54 deg |
| **A2: `goalkick,goalframe,unsure,ball,goalin,ballhold`**, ballwin candidates, 50 deg floor, edge rule | **11/13** | 63 deg |

Misses: the seg 9 goal kick at 1:10 (ball in the keeper's hands, 76%) and
seg 3's opening seconds (95%). Tried and not adopted: lead, nowalk,
fastboost, weak, allnear, flow, lone, gk2, netevent (each in the backlog).

With `roles` and `kickoff` (2026-10-18/19) the same configuration
captures 12/13 (the seg 9 goal kick found through the keeper's role),
missing only seg 3's opening seconds (95%).

Ball source, same run (2026-10-19, the path against the chain,
ballwin candidates; 13 moments as the set then was):

| Configuration | Ball source | Captured | Mean coverage | Width (median) |
|---|---|---|---|---|
| A2 (`goalkick,goalframe,unsure,ball,goalin,ballhold`) | chain | 11/13 | 98% | 61 deg |
| A2 | `ballpath.py` path | 9/13 | 96% | 61 deg |
| A2 + `roles,kickoff` | chain | 12/13 | 100% | 63 deg |
| A2 + `roles,kickoff` | `ballpath.py` path | 10/13 | 97% | 62 deg |

The path aims at the ball in 39% / 70% of the clips against the chain's
27% / 65%, but loses two seg 9 boxes: 1:57 (67%; the path tracks a slow
ball in the left corner at 1:54-1:56, 25 m from the goal, and the
`goalkick` rule drops the keeper's framing a second before the kick) and
0:13 (95%; the same targets, a slightly different smoothed camera).

With the 1:57 box removed and the 0:13 box redrawn to the clearance
itself (yaw -0.1 to 13.9, leaving out the two players at the old left
edge; 2026-10-19, 12 moments), same run: A2 + `roles,kickoff` captures
11/12 with either ball source (mean coverage 100%), the chain at a 63 deg
median width and 60 deg at the captures, the path at 62 and 55. Both miss
only seg 3's opening seconds (95%). On that the path became the default
ball source (`OSB_BALL_CHAIN=1` for the chain).

On all 26 moments (2026-10-20, ball path with gap penalty 1):

| Configuration | Captured | Mean coverage | Width (median) |
|---|---|---|---|
| A + `roles` (players only) | 13/25 | 70% | 53 deg |
| A + `roles,surge` | 14/25 | 75% | 54 deg |
| **A2 + `roles,kickoff`** (the default) | **21/26** | 89% | 54 deg |
| default + `surge` (N = 7) or `sideline` | 20/25 | 88% | 55 deg |

(The /25 rows were run before 7:22 was added; 7:22 is captured by the
default.) Misses of the default: 7:34 (19%; play parked in the left
penalty area, no ball candidate, almost nobody running), 10:02 (0%; a
throw-in, the ball in the thrower's hands), 14:05 (0%; a single ball
sighting), and two at 95%: 7:51 and the seg 3 clip's first seconds.

Add moments by appending clip times to `playeval.py make` for an existing
set (it keeps the others) or making a new set from another stretch of the
game; `playbench.py` lists the clips in `CLIPS`. A new set needs the
stretch's pass A cache, ball candidates and roles (made from the video,
see section 5); sets over the full-game chunks have them already.

## 3. Ball tracking: clip labels

The ball's position every 2 s along a clip, labelled on both phones, so the
tracker is measured directly (found / wrong / missed) rather than through
the play boxes.

| Step | Command |
|---|---|
| Build a clip set | `uv run processing/spikes/labelset.py footage/rec3 clip <name> <segment> <start s> <seconds> [step s]` |
| Label it | preview `ball-labels-seg9` / `ball-labels-seg3` (`processing/spikes/label/server.py footage/rec3/labels_<name> [port]`) |
| Score the tracker (ball.py's online chain, the default) | `uv run processing/spikes/balleval.py footage/rec3 <name> ["NAME=value,..." ...]` |
| Score ball.py's online chain instead | `OSB_BALL_CHAIN=1 uv run processing/spikes/balleval.py footage/rec3 <name> ["NAME=value,..." ...]` |
| Write `ball.json` with the path tracker | `uv run processing/spikes/ballpath.py footage/rec3 footage/rec3/detect/ball_<seg>_<start>_<seconds>[.win].json [out.json]` |
| Score detectors per frame | `uv run processing/spikes/ballheat.py footage/rec3 <name> [wasb] [tracknetv2] [tracknetv3]` |
| Fine-tune WASB with the background | `uv run processing/spikes/heattrain.py footage/rec3 bg`, then `data labels labels_seg3 labels_seg9`, `train <name> [--hold <set>]`, `eval <name \| stock> <set>`; in ballwin.py `OSB_WIN_WASB=1 OSB_WASB_CKPT=footage/rec3/heattrain/<name>.pt` (the recording needs `heattrain.py <rec> bg`) |
| Measure an old set's label-time offsets | `uv run processing/spikes/ballheat.py footage/rec3 <name> dt` (writes `dtS` per frame into `items.json`) |

Sets: `footage/rec3/labels_<name>/` (`items.json`, `images/`, `pano/`,
`labels.json` with every ball marked and the match ball flagged). Two sets
of 60 moments: `seg9` (37 with the match ball on the pitch) and `seg3` (53).
Both were pre-marked by Claude from the detector and judge and then
reviewed; 20 of seg 9's 118 frame labels and 33 of seg 3's 102 still carry
`draftBy: claude` (never edited, mostly tiny far balls).
The clip's candidates come from the files below; the set's suggestions are
drawn from them.

A hit is an estimate within 3 m + 5% of the ball's distance from the camera
(placing a far ball on the pitch is uncertain in depth; `OSB_HIT_M`,
`OSB_HIT_PER_M` to vary). Moments labelled "none" (ball not visible) and
"unsure" are not counted as wrong.

Label times: each frame in `items.json` carries `dtS`, its image's offset
from the moment's time (`t`, `clipT`). Sets made before 2026-10-18 were
grabbed with a plain seek, which returns a frame 0.2-0.9 s late (seg 9 main
+0.38 s, secondary +0.66; seg 3 main +0.86, secondary +0.17); `ballheat.py
<name> dt` measured it per frame by matching the image and wrote it.
`labelset.py` now decodes with a preroll (`phase0.frame_at`), so new sets
carry dtS 0. `balleval.py` reads the tracker at `clipT + dtS` of the phone
whose image was labelled; `OSB_NOMINAL=1` reproduces the old comparison.

Candidate sources, all in `footage/rec3/detect/ball_<seg>_<start>_<seconds>.<tag>.json`:

| Tag | Script | What |
|---|---|---|
| (none) | `ballclip.py` | fine-tuned RF-DETR on every 640 px tile, 10 fps (~30 min per 2 min clip) |
| `win` | `ballwin.py` | the same detector in search windows around the predicted ball, full scans every 2 s or when lost (~6.5 min per 2 min clip); the default source |
| `wasb` | `OSB_WIN_WASB=1 OSB_WIN_TAG=wasb ballwin.py` | the `win` source plus WASB (fp16, 3 frames, 2x/4x tiles) in the full scans and one window; ~1.5-1.7x the time. `base` is the same run without WASB, for comparisons |
| `flow` | `flowcands.py` | motion-only candidates (double difference, size, consistent step); also writes the goal-net change series `net_*.json` |

Every candidate carries a `judge` score from `balljudge.py` (a 48 px crop
classifier; `data`, `train`, `eval`, `clip`). `ballcompare.py` compares the
stock and fine-tuned detectors on the random label set's held-back frames.

Every game uses one standard detector and judge, `footage/models/ball_detector.pth`
and `ball_judge.pt` (provenance and bench scores in `footage/models/README.md`):
`fullgame.py` runs `ballwin.py` with the detector (`--detector` to override), and
`balljudge.load_judge` reads the judge unless a model name picks
`<recording>/balljudge/<name>.pt`. Since 2026-11-01 these are the rec3 + rec5
+ rec6 models (white, orange and daylight; detector balltrain_v3, judge
judge_v3); from 2026-10-30 the rec3 + rec5 pair; before, rec3's. Since
2026-11-13 the judge is `judge_v5` (the same recordings plus rec7's yellow
ball, with the reviewed off-pitch verdicts); the detector is unchanged. A run can pick
another judge with `OSB_JUDGE` (rec7's own `judge_v4`, not the standard). The benches pass a detector
explicitly; their candidate files keep the judge scores they were made with.

`ballpath.py` links the same filtered candidates as one minimum-cost path
over the whole clip (offline Viterbi; angular motion within a phone's
view, metric between phones, a "no ball" node per frame, edges up to 1 s
back) instead of ball.py's greedy online chain, and is the default ball
linker in `balleval.py` and `follow.py` since 2026-10-19 (`OSB_BALL_CHAIN=1`
gives the chain in both; the play boxes in section 2 decided it). Its
costs sweep like the chain's (`"ballpath.NONE_COST=3"`).

`ballheat.py` scores detectors per labelled frame instead of through the
tracker: a hit is a candidate within 12 px or 1.5 ball diameters of the
label, at a given score threshold, against candidates per frame. It runs
the multi-frame heatmap models (WASB, TrackNetV2, TrackNetV3; code and
weights under `datasets/`, see its docstring) on 512x288 tiles of each
phone's frame at native, 2x and 4x downscale, and prints the same measure
for the RF-DETR candidate files. Candidates are cached in
`labels_<name>/heat_<model>.json`.

Results at the labelled moments (tracker defaults as of 2026-10-17; seg 9
has 37 moments with the match ball on the pitch, seg 3 53). Rows marked
"nominal" compare at the stored time, as every run before 2026-10-18 did
(the 2026-10-16 figures were 54% / 62%, wrong 1 / 8 with full scans); the
others at the label image's time:

| Candidates | Label time | seg 9 found | seg 3 found | wrong | missed |
|---|---|---|---|---|---|
| tracker before tuning (4 s static filter, 3 hits, no judge) | nominal | 32% | 26% | 9 / 12 | |
| tuned defaults, full-scan candidates | nominal | 18/37 (49%) | 33/53 (62%) | 1 / 8 | 18 / 12 |
| tuned defaults, `win` candidates | nominal | 18/37 (49%) | 31/53 (58%) | 4 / 8 | 15 / 14 |
| tuned defaults, full-scan candidates | corrected | 17/37 (46%) | 37/53 (70%) | 0 / 2 | 20 / 14 |
| tuned defaults, `win` candidates | corrected | 17/37 (46%) | 35/53 (66%) | 1 / 1 | 19 / 17 |
| tuned defaults, full-scan candidates, 2026-10-19 (outside-the-lines static filter) | corrected | 16/37 (43%) | 36/53 (68%) | 0 / 2 | 21 / 15 |
| tuned defaults, `win` candidates, 2026-10-19 | corrected | 14/37 (38%) | 36/53 (68%) | 0 / 1 | 23 / 16 |
| `ballpath.py` offline path, full-scan candidates | corrected | 21/37 (57%) | 38/53 (72%) | 2 / 2 | 14 / 13 |
| `ballpath.py` offline path, `win` candidates | corrected | 18/37 (49%) | 39/53 (74%) | 2 / 4 | 17 / 10 |

The 2026-10-19 chain rows follow commit 2a5560d (a ball resting outside
the lines is a spare ball), which moved the chain by 1-3 moments; the
path rows use the same candidates and filters, so they differ from those
chain rows in the linking only. Coverage: chain 31 / 28 / 70 / 66%, path
44 / 40 / 70 / 72%.

Detectors per labelled frame (`ballheat.py`, 2026-10-18; balls in play on a
phone: 48 on seg 9, 58 on seg 3):

| Detector | seg 9 found | seg 3 found | candidates / frame |
|---|---|---|---|
| fine-tuned RF-DETR, conf >= 0.1 | 27% | 60% | 1.0 / 0.9 |
| WASB (football weights), score >= 0.05, judge >= 0.3 | 29% | 60% | 1.4 / 1.7 |
| union of the two | 44% | 78% | 2.4 / 2.6 |

The candidate ceiling (a usable candidate near the ball at all) is ~62% of
moments; the misses are mostly balls under 8 px, 43-100+ m away. On the
filtered candidates the tracker sees, a candidate within the hit radius
exists at the labelled frame at 15/37 and 31-32/53 moments, within 0.5 s
of it at 27-28/37 and 41-42/53; the path tracker's misses are 9/37 and
10/53 moments with no candidate within 0.5 s and the rest a lone
candidate 1-5 frames off, so the remaining gap is detection. Witnesses
from other motion (keeper bearings, motion blobs, net ripple, facing
direction: `witness.py`, `flowcands.py`, `facing.py`) were scored the same
way and none raises the found-rate; the backlog has their numbers.

### Held-out ball labels

Every time balls are labelled, on any recording and with any tool
(`labelset.py` random or clip sets, the engine's `label ball`), part of
the new labels is held back for testing:

- **Decide before training, and mark it.** Choose the held-out part
  when the set is made or labelled and mark it with `holdout.py` (below)
  before anything trains on the set; note it in the Label sets column
  of `docs/DATA.md`. Never pick it after seeing results.
- **Hold out time, not frames.** Frames a second or two apart show the
  same ball, players and light, so a held-out frame next to a trained
  one isn't independent. Hold out whole clips or whole segments: for a
  clip set, label a second clip as the test (rec3's `seg9`), or at
  least the last third of a single clip; for a random set, the moments
  in one or two whole segments. The engine's `label ball` sets hold out
  the last quarter of every chunk by themselves.
- **About a quarter, and at least ~30 ball moments.** Fewer than that,
  and a gain or loss of a few is noise (see "Small sets are noisy").
- **Train without it (enforced).** Pass a held-out clip as the test set
  (`balljudge.py data <rec> <train sets> <test sets>`, `heattrain.py
  train --hold <set>`). The valid split that `balldata.py` and
  `balljudge.py` make by hashing each moment chooses the epochs, so it
  doesn't count as held out.
- **Report on it.** A new detector or judge is scored on every game's
  held-out labels, not only on the game it was trained for (rec7's
  judge_v4 improved rec7 and lost 5 balls on rec3's held-out `seg9`).

How the code enforces it (`processing/spikes/holdout.py`):

| Step | Command |
|---|---|
| Make a test clip | `labelset.py <rec> clip <name> <segment> <start s> <seconds> [step s] --heldout` |
| Mark a set (or whole segments of one) held out | `holdout.py <rec> mark <set>` / `holdout.py <rec> mark <set> segments 9,15` |
| See what a recording holds out | `holdout.py <rec>` |

A mark is a `heldout.json` in the set's folder (`{"all": true}`,
`{"segments": [...]}` or, from the engine, `{"gameS": [[from, to]]}`).
The held-out time belongs to the recording: a labelled frame from any
set inside it, or within 20 s of a held-out set's frames, is test data.
`balljudge.py data`, `balldata.py` and `heattrain.py train` leave those
frames out and refuse a held-out set named as training data;
`balljudge.py data` and `balldata.py` stamp their output with the
held-out time they applied, and `balljudge.py merge` / `train`,
`ballmerge.py` and `balltrain.py` refuse data with no stamp or a stale
one (a set marked after it was built), so data built before 2026-11-09
must be rebuilt before training. Marks are only ever added; removing one
means deleting the file by hand, and only if nothing was scored on it.

**Unmarked suggestions off the pitch are not negatives.** A detector
suggestion or motion candidate that nobody marked and whose ground point
lies outside the pitch lines may be someone else's ball (a warm-up, the
next pitch, a spare by the bench). Since 2026-11-11 `balldata.py` makes no
hard-negative tile of it and keeps every tile clear of it (moved; failing
that a negative tile is dropped and a ball's tile has the spot painted
over), and `balljudge.py data` makes no negative crop of it, in every
split; both print how many they left out. Points above the horizon stay
negatives, and so does everything inside the lines: a picture with a second
ball on the pitch is still to be labelled "unsure". The tracker drops
candidates more than 10 m outside the lines anyway (`ball.usable`).
`OSB_OFF_PITCH_NEG=1` builds the data the old way, to compare. A
left-out suggestion can be settled instead: `offpitch.py list <rec>` cuts
a crop of each, `offpitch_page.py <rec>` shows them (grouped by spot, since
the phones don't move) to mark "ball", "not" or "unsure", and a reviewed
"ball" is then a positive and a reviewed "not" a negative again
(`<rec>/labels/offpitch.json`; drafts and "unsure" stay left out). The
phones don't move, so one object gives a verdict in every frame it shows
in: verdicts of one phone within 8 px are one spot, at most 10 frames of a
spot count, a ball spot stays on one side of train / valid, and a ball not
used is hidden (not background). A clip set works the same
(`offpitch.py list <rec> labels_seg6`, `balldata.py <rec> <out> labels
labels_seg6`). On the
random sets this leaves out 55% of rec3's unmarked suggestions (865 of
1,583: mostly lights, cars and kit beyond the fence, some balls on the next
pitch), 41% of rec5-game's (38 of 93) and 87% of rec6's (693 of 801: many
are balls behind the goals and by the benches).

An outside test set: SoccerTrack's wide view (`docs/DATA.md`, External
datasets), never trained on. `uv run processing/spikes/stbench.py 2.0 999
--expect-row` scores the detector and judges on it frame by frame (~20 min).

Held out so far: only rec3's `seg9` (marked 2026-11-09 in `rec3` and
`bench/rec3`; 9 moments of rec3's random set fall inside it and now stay
out of training). rec3's `seg3`, rec5's and rec6's
random sets and rec7's `seg6` are all in training, so the ball models
have no independent test on those games yet.

## 4. Where things are

The whole `footage/` layout, its S3 backup and how to restore it are in
[DATA.md](DATA.md).

| Path | What |
|---|---|
| `footage/bench/`, `bench-cache/`, `bench-out/`, `train/` | the bench store (inputs and video, mirrored to S3 and Hugging Face) and what is made from it (section 6) |
| `footage/rec3/pitch/` | pitch map, goals, background panorama, overview map |
| `footage/bench/fixtures/pitch/` | mapping regression fixtures; `night-2026-10-21/` is the marked-field benchmark |
| `footage/rec3/follow/` | pass A caches (`passA_*.json`), the last run's path files, `camlog.txt` and `play_debug.json` |
| `footage/rec3/playeval/` | play-box sets and benched paths, `bench.json` |
| `footage/rec3/labels/` | the random ball label set (detector training data) |
| `footage/rec3/labels_<name>/` | clip ball label sets |
| `footage/rec3/detect/` | ball candidate files by clip and tag, net series |
| `footage/models/` | the standard ball detector and judge every game uses, with a README of where each came from |
| `footage/rec3/balldata/`, `balltrain/`, `balljudge/` | detector dataset and checkpoints, judge crops and models |
| `footage/rec3/fullgame2/` | the second whole-game run's logs: per-chunk `camlog_*.txt` and the runner's logs (the video was removed) |

Preview servers are in `.claude/launch.json` (`ball-labels`, `ball-labels-seg9`,
`ball-labels-seg3`, `play-eval`, `play-eval-seg9`, `play-eval-seg0`,
`play-eval-seg0-600`, `play-eval-seg0-1200`); they listen on localhost only.

## 5. Without the game video

The phones' segments and all renders were removed locally on 2026-10-21
(the segments are in S3, checked byte-for-byte on 2026-10-19:
`accounts/<account>/games/f68af87f.../recordings/78618588.../<device>/`).
Everything in sections 1-4 still runs (checked after the removal: the
play-box bench 21/26, the tracker 49% / 74% found, the pitch test 4/4),
because the benches read only cached inputs.

What still runs: any change to the planner (follow.py's rules, framing,
features and constants), to the ball linker over the existing candidates
(ball.py filters, ballpath.py costs), and to role logic over the existing
roles files.

What needs the video back: rendering anything to watch; new per-frame
inputs (another player or ball detector, roles at another rate, new
search windows); and new label sets or play boxes outside the cached
stretches. Download only the segments covering the stretch (about 1.2 GB
each per phone) into `footage/rec3/<device id>/` with the manifest's
object keys, e.g.

    aws s3 cp --profile sbv s3://<footage bucket>/accounts/.../recordings/78618588-.../<device id>/<device id>_003.mp4 footage/rec3/<device id>/

## 6. The bench store

The short version, for running the benchmarks and recording a result, is
[BENCHMARKS.md](BENCHMARKS.md).

`footage/bench/` holds everything the benches in sections 1-3 read that
nothing can remake, for every recording with must-see boxes or ball
labels, so the benchmarks need only it, `footage/models/` and the code.
What is made from it is kept apart, under the same relative path
(`processing/spikes/recpaths.py`; the whole layout is in
[DATA.md](DATA.md), "The bench store"):

| Folder | What | |
|---|---|---|
| `footage/bench/<rec>/` | inputs: manifest, `phase0/`, `pitch/`, `teams/`, must-see sets, ball label sets, and the video the sets cover | S3 `bench/` (outside `accounts/`: deleting a game's raw footage won't touch it) and Hugging Face |
| `footage/bench-cache/<rec>/` | made from the inputs and the video, hours to remake: `follow/passA_*.json`, `detect/` (ball candidates, roles, net series), `bg/` | S3 `bench-cache/`; not on Hugging Face (a contributor makes their own) |
| `footage/bench-out/<rec>/` | what runs write: paths, `playeval/bench.json`, camera logs, debug images | nowhere; delete at will |
| `footage/train/<rec>/` | what training makes: `balldata/`, `balltrain*/`, `balljudge/`, `heattrain/` | nowhere |

So a bench run never changes the store, and a script given a store
recording (`footage/bench/rec3`) reads and writes its caches and outputs
in those folders by itself; given a working recording (`footage/rec3`) it
keeps everything in the recording's folder as before.

| Recording | In the store |
|---|---|
| `bench/rec3/` | the first game: 5 must-see sets (26 moments), `labels` (random set), `labels_seg3`, `labels_seg9` (held out). Video: Main segments 0-6 and 9 and their Secondary pairs (16 files, 17.6 GB): game 0:00-30:00, the seg 3 and seg 9 clips |
| `bench/rec4cal/` | the 2026-10-21 night game in the corrected camera model: 2 must-see sets (15 moments), `lens/`. Video: Main segments 2-6 and their Secondary pairs (10 files, 10.9 GB): game 10:00-30:00 |
| `bench/rec5-game/` | the orange-ball game: `labels` (the ball set, with its frames) and what it needs: the manifest, `phase0/`, `pitch/pitch.json`. No video, and no other labelled data (the kit labels, team colours and pitch taps were taken out 2026-11-18; they are in the working recording) |
| `bench/rec6/` | the first daylight game: `labels` (random set; no video in the store) |
| `bench/rec7/` | the 2026-11-04 night game (yellow ball): 1 must-see set (5 moments), `labels_seg6` (its last 40 s held out). Video: Main segments 5-7 and Secondary 4-6 (6 files, 6.9 GB); Secondary 4 holds the play clip's first seconds |
| `bench/rec8/` | the 2026-11-11 night game: 4 must-see sets (16 boxes, all plays the follow cam missed: a count of misses fixed, kept apart from the standard 46), `labels_seg11-136`, `engine/labels/ball-1`. Video: Main segments 0-4 and 10-14 and Secondary 0-4, where it stopped (15 files, 16.6 GB) |
| `bench/fixtures/pitch/` | the pitch regression fixtures and the marked-field benchmark |
| `bench/soccertrack/` | SoccerTrack's annotations (held out) |
| `bench/LABELS.md` | every ball label set: trained on, or held out |

The standard benchmarks, against the baselines in
`processing/bench_baselines.json` (in git: change a baseline in the same
commit as the change that moves it):

    uv run processing/spikes/bench.py check     # everything a run reads is here, and the caches match the inputs
    uv run processing/spikes/bench.py run       # the play bench on rec3, rec4cal, rec7 and rec8 (49/62), balleval on rec3's seg3 and seg9

Or one at a time, as in sections 1-3, on the store's recordings:

    uv run processing/spikes/playbench.py footage/bench/rec3 footage/bench/rec4cal footage/bench/rec7 [variant ...]
    OSB_BALL_TAG=win uv run processing/spikes/balleval.py footage/bench/rec3 seg3
    cd processing && uv run --extra dev pytest tests/test_pitch_regression.py

Checked when the layout was split (2026-11-17), in a copy holding only
`bench/`, `bench-cache/`, `models/` and the code: the play bench 36/46
(rec3 21/26, rec4cal 11/15, rec7 4/5) and balleval seg3 45/53 found
(2 wrong), seg9 23/37 (3 wrong), the same as on the working recordings;
nothing was written into `bench/`. rec8 gives 13/16 with the same
features (and without `peoplesurge`), as on the fresh mirrors of
2026-11-16; with the pass A caches of its first full-game run, made
before the fitted-pitch field (113a2c6), it gives 2/16. The pitch regression test fails 4/4
on its fixtures (since the line-mask change of 2026-10-22, 637b3ba;
section 1), not the store.

The baselines are from caches remade from the video on 2026-11-18 with
the standard models (`ball_detector.pth` c69adbc8 (since rewritten without local paths as 04632214: the same weights), `ball_judge.pt`
bd32076c, the stock player detector) and this code: every clip's ball
candidates (ballwin.py), roles (roles.py) and pass A. Play bench 49/62:
rec3 19/26, rec4cal 13/15, rec7 4/5, rec8 13/16; balleval seg3 41/53
found (1 wrong), seg9 24/37 (6 wrong). The figures above (rec3 21,
rec4cal 11, seg3 45, seg9 23) were on older candidates: rec3's from its
own first detector (2026-10-16 to 19) and rec4cal's from a detector
trained on rec4cal (2026-10-22), neither standard since October. The
caches from before are in `bench-out/<rec>/cache-before-regen/`.

The caches. `bench.py stamp` writes `bench-cache/<rec>/inputs.json`: the
cameras, sync report, pitch map, goals and standard models the caches
match; `bench.py check` reports them stale when any of those change, and
then they are remade from the video before a score means anything (the
pass A caches in the store were once left over from before the fitted
pitch, and gave 33/46). A missing pass A cache is remade by the play
bench itself (the video for the clip must be here); ball candidates by
`ballwin.py`, roles by `roles.py`, as `fullgame.py` runs them. Stamp
again once they are current.

| Step | Command |
|---|---|
| Check the video every set needs is there and matches the manifest | `uv run processing/spikes/benchstore.py [footage/bench/<rec>] [--quick]` |
| Get it as a contributor (gated; see DATA.md, "Sharing with contributors") | `uv run processing/spikes/sharedata.py fetch [--no-video]` |
| Restore it on a new machine | `uv run processing/spikes/footagestore.py restore` (the store, without waiting for the video: `aws s3 sync --profile sbv --exclude "*.mp4" s3://<footage bucket>/bench/ footage/bench/`) |
| Save changes to it | `uv run processing/spikes/footagestore.py backup`, then `sharedata.py upload` |

Adding a set: make it on the full recording (section 2 or 3), copy its
directory into the store and the caches it reads (candidates, pass A
cache, roles) into `bench-cache/<rec>/`, add a ball label set to
`bench/LABELS.md`, and copy any segment `benchstore.py` reports missing
into S3's `bench/<rec>/` from the recording's prefix (a copy within the
bucket; the manifest has the object key). A file the sets read only the
start of is cut there (2026-11-18, 62 -> 52 GB; DATA.md, "The bench
store"): the frames left are the upload's own, so detector runs
reproduce exactly. `benchstore.py` lists the part of each file that is
read.

## 7. The engine commands

`processing/spikes/engine.py` is how the Mac app (and later the cloud
job) runs a game: a job file in, questions out, answers in, progress as
JSON lines, `game.mp4` out. Its docstring has the commands and exit codes;
the file formats are `schemas/engine-*.schema.json`.

| Step | Command |
|---|---|
| Unit tests (answers to sides, kit score, the schemas, the no-stitch path with stand-in steps, footage hashes, the disk check and cleanup, the questions' words and pictures, answers from flags; no footage) | `uv run --with pytest --with numpy --with opencv-python-headless --with jsonschema pytest processing/tests/test_engine.py` |
| Sync, pitch fit, kit groups, then the questions | `uv run processing/spikes/engine.py prepare job.json` |
| Apply answers (asks again after a pitch correction) | `uv run processing/spikes/engine.py prepare job.json --answers answers.json` |
| The whole game | `uv run processing/spikes/engine.py run job.json --answers answers.json` |
| What's done and pending, and the command to run next | `uv run processing/spikes/engine.py status job.json` |
| The pending questions as one document to read: what is asked, how to answer, the pictures (a close-up per pitch keypoint) | `uv run processing/spikes/engine.py questions job.json` |
| Write one answer into `<recording>/engine/answers.json` from flags, checked against the question ([ENGINE-AGENTS.md](ENGINE-AGENTS.md)) | `uv run processing/spikes/engine.py answer job.json pitch --approve` (or `--closeup 'NAME=X,Y'`, `cameras --left main`) |
| A ball label set (50 frames of Main, half detected, half bridged); also while `run` is going, from the chunks through the ball stage | `uv run processing/spikes/engine.py label job.json ball [--count 50]` |
| A must-see set: panoramas where the user pressed Missed play, plus random moments | `uv run processing/spikes/engine.py label job.json moments --at <s,s,...> [--random 3]` |
| Store labels (also as `labels.json` / `boxes.json` for the benches) | `uv run processing/spikes/engine.py label job.json --answers labels.json` |
| List what the user agreed to share | `uv run processing/spikes/engine.py share job.json` |

A job is `{"schemaVersion": 1, "recording": "<folder>", "output": {"height": 1440}}`
(`"output": {"height": 1080, "youtubeCopy": true}` keeps `game.mp4` as the
1080p render and writes the 1440p upload beside it as `game-youtube.mp4`,
+~25 GB in the disk check; on a finished game `run` makes just the copy);
`"options": {"maxChunks": 1}` renders only the first chunk, for a quick
check. The questions are in `<recording>/engine/questions.json` with the
overlay and close-up images beside them; answers carry the question's
`round`, and an answer to an earlier round is ignored.

When the phones can't be stitched (rec5), prepare asks only `cameras`
first: the panorama drawn through rough cameras with each phone on the
left (`engine/cameras/main_left.jpg`, `secondary_left.jpg`) and which shows
the pitch in one piece (`{"cameras": {"round": 1, "leftPhone":
"secondary"}}`). Two stills side by side don't settle it: the phones'
views cross, so on rec5 Main's inner edge looked like it joined
Secondary's when it doesn't; the wrong panorama shows two broken halves. It then builds
rough cameras (`jointfit.py rough`: gravity sensors, lens table, 80 deg
apart) and asks the pitch as usual, except that the pitch has
`approveAllowed: false`: approving is refused (with a `problem`) until
tapped points have fitted the cameras and pitch together
(`jointfit.py taps`, `search` over starting angles 74-89 deg and three
camera heights, ~2 min; `install`, which redraws the panorama). The next
round can be approved or corrected again (`jointfit.py fit` from the
fitted cameras). To try the no-stitch path on a game that stitches,
delete `phase0/cameras.json` and make `pitch.py`'s stitch fail, or give
the copy rec5's footage.

Label sets are in `<recording>/engine/labels/<set>/` and hold the files
the existing tools read too: a ball set is a labelset.py set
(`items.json`, `images/`), so `label/server.py` can label it; a moments
set is a playeval.py set (`items.json`, `pano/`, `pano.json`). `share`
writes one `engine/share/<name>.json` per result: the files to upload,
with sizes and SHA-256, and the result in numbers (pitch: approved first
time or after how many corrections; label sets:
answers by kind). It sends nothing.

To test it end to end without disturbing a recording's earlier results,
make a fresh copy: a folder with the recording's `manifest.json` and
symlinks to its device folders. Everything the engine makes goes into the
copy. Check also: a stop (SIGTERM to engine.py) exits 130 and leaves no
`ballwin.py` / `roles.py` / `follow.py` running, and the same command then
resumes; a bad job, missing footage or bad answers exit 2; `run` with
questions pending exits 3. Both prepare and run hash every segment
against the manifest's SHA-256 the first time (a changed byte exits 2,
`footage_corrupt`); `run` refuses to start without room for what it
still writes (`not_enough_disk`: ~49 GB for 100 minutes at 1440p) and,
once `game.mp4` is checked, deletes the pieces, `video.mp4` (if one was made) and
`audio.m4a` unless the job has `"options": {"keepIntermediates": true}`.
A copy made for a test with fewer segments needs a manifest listing only
those.
