# Testing the processing pipeline

How the follow cam, the pitch map and the ball tracking are measured, what the
measurements say today, and how to extend them. Everything here runs locally
on the Phase 0 spikes in `processing/spikes/`. See `docs/BACKLOG.md` for the
history of what was tried and `docs/TESTING-APPS.md` for the phone apps.

## Principles

- **Footage is private.** It shows people, some possibly minors. Every label
  set, fixture and rendered clip lives under `footage/` (gitignored) and never
  enters the repository. Back them up privately; the tooling and the numbers
  are what the repo keeps.
- **Score against reviewed labels, not by eye.** Each algorithm change is run
  through the harness below before a render. Watching renders is for judging
  what to label next, not for deciding between variants.
- **Small sets are noisy.** 13 play moments and 90 ball moments: a difference
  of one or two is noise. Treat them as a direction, and add moments before
  tuning further.

## 1. Pitch and goal mapping

`processing/tests/test_pitch_regression.py` reruns the whole mapping (line
fit from the starting guess, goal detection, refit with the goals) on saved
background panoramas and compares it with a result that was checked by eye.

    uv run --with pytest --with numpy --with scipy --with opencv-python-headless --with torch \
        pytest processing/tests/test_pitch_regression.py -v

Fixtures: `footage/fixtures/pitch/<name>/` (`background.png`, `expected.json`),
or `$OSB_PITCH_FIXTURES`. The test skips when none exist. Make one from a
recording whose `pitch.json` and `goals.json` you have verified:

    uv run processing/spikes/pitch.py footage/rec3 fixture footage/fixtures/pitch/<name>

Tolerances: both goals found and within 0.3 m; goal frames within 1.5 px of
the white structure and end markings within 3.5 px of the paint; pitch
length and width within 1 m, camera height within 0.2 m; 15 key points
(corners, centre and penalty spots, post bases and tops) within 0.1 deg,
the lines-only fit within 0.3 deg. One fixture so far: the night game.

### Marked-field benchmark

`footage/fixtures/pitch/night-2026-10-21/` (rec4, the 2026-10-21 night game) has every pitch marking
traced on the paint and completed, the goal frames fitted to their posts, and 15 keypoints tapped by eye
(`pitchtap/server.py`), in `benchmark.json`; `benchmark.png` shows it. Score a pitch fit against it:

| Step | Command |
|---|---|
| Score fits | `uv run processing/spikes/pitchbench.py footage/fixtures/pitch/night-2026-10-21 <pitch.json> ... [--goals goals.json] [--all]` |
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
and boxes are in the corrected camera model (factory focal lengths, levelled to gravity; `cameras.json`
in the set), the same one as the pitch benchmark `night-2026-10-21`, so score only paths made with those
cameras. Both have a `clip.json`, so `playbench.py footage/rec3 footage/rec4cal` benches all 41 moments
together (the pass-A caches for both clips are in `footage/rec4cal/follow/`). `footage/rec4/playeval/seg0-600`
holds the same boxes in the first camera model and is kept only to score the first full-game run (no
`clip.json`: not benched).

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
`<recording>/balljudge/<name>.pt`. Since 2026-10-30 these are the rec3 + rec5
models (white and orange balls); before, rec3's. The benches pass a detector
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

## 4. Where things are

| Path | What |
|---|---|
| `footage/bench/` | the bench store: both games' bench inputs and video, mirrored to S3 (section 6) |
| `footage/rec3/pitch/` | pitch map, goals, background panorama, overview map |
| `footage/fixtures/pitch/` | mapping regression fixtures; `night-2026-10-21/` is the marked-field benchmark |
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

`footage/bench/` holds everything the benches in sections 1-3 read, for
both games, including the video they need, so any of them can be rerun
from scratch (new detectors, ballwin settings, roles) without going back
to the full recordings. It is mirrored to `s3://<footage bucket>/bench/`,
outside `accounts/`: the upload Lambda ignores it, and deleting a game's
raw footage after rendering won't touch it.

| Path | What |
|---|---|
| `bench/rec3/` | the first game: manifest, `phase0/`, `pitch/`, `teams/`, `bg/`, `playeval/` (5 sets, 26 moments), `follow/` (pass A caches), `detect/` (candidates by tag, roles, net series), `labels/` (random set), `labels_seg3/`, `labels_seg9/`, `balldata/`, `balltrain/` (the RF-DETR checkpoint), `balljudge/`, `heattrain/*.pt` |
| `bench/rec3/<device>/` | Main segments 0-6 and 9 and their Secondary pairs (16 files, 20 GB): game 0:00-30:00, the seg 3 and seg 9 clips |
| `bench/rec4cal/` | the 2026-10-21 night game in the corrected camera model: manifest, `phase0/`, `pitch/`, `lens/`, `teams/`, `bg/`, `playeval/` (2 sets, 15 moments), `follow/`, `detect/`, `balljudge/` (rec3's judge) |
| `bench/rec4cal/<device>/` | Main segments 2-6 and their Secondary pairs (10 files, 12.5 GB): game 10:00-30:00 |
| `bench/fixtures/pitch/` | the pitch regression fixtures and the marked-field benchmark |

Run the benches on it by giving its recordings:

    uv run processing/spikes/playbench.py footage/bench/rec3 footage/bench/rec4cal [variant ...]
    OSB_BALL_TAG=win uv run processing/spikes/balleval.py footage/bench/rec3 seg3
    OSB_PITCH_FIXTURES=footage/bench/fixtures/pitch uv run ... pytest processing/tests/test_pitch_regression.py

Checked when the store was made (2026-10-29): every segment matches its
manifest's sha256, the play bench gives 29/41 (rec3 20/26, rec4cal 9/15)
as on the working recordings, balleval matches them, and ballwin decodes
the rec4cal video. The pitch regression test fails 4/4 on both copies of
its fixtures alike (a regression since 2026-10-21, not the store).

The benches write their outputs (paths, candidate files, caches) into the
store as they do into any recording; don't push scratch variants to S3.

| Step | Command |
|---|---|
| Check the video every set needs is there and matches the manifest | `uv run processing/spikes/benchstore.py [footage/bench/<rec>] [--quick]` |
| Restore it on a new machine | `aws s3 sync --profile sbv s3://<footage bucket>/bench/ footage/bench/` |
| Save changes to it | `aws s3 sync --profile sbv footage/bench/ s3://<footage bucket>/bench/` |

Adding a set: make it on the full recording (section 2 or 3), copy its
directory and the files it needs (candidates, pass A cache, roles) into
the store, and download any segment `benchstore.py` reports missing from
the recording's S3 prefix with the manifest's object key. Video is kept as
whole, untouched segments so detector runs reproduce exactly; trimming them
to the bench stretches would save only ~15%.
