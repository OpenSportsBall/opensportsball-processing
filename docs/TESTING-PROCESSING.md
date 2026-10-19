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

## 2. Follow cam: must-see boxes

A reviewer draws, on a full-resolution panorama of a moment, the part of the
field a follow cam has to show. A camera path is scored by how much of each
box it shows at that frame.

| Step | Command |
|---|---|
| Panoramas at chosen clip times | `uv run processing/spikes/playeval.py make footage/rec3 <set> <segment> <start s> <clip t> ...` |
| Draw and review boxes | preview `play-eval` / `play-eval-seg9` (`processing/spikes/evalpage/server.py <set dir> [port]`) |
| Score path files | `uv run processing/spikes/playeval.py score footage/rec3 <set> <path.json> ...` |
| Bench many variants | `uv run processing/spikes/playbench.py footage/rec3 [variant ...]` |

Sets: `footage/rec3/playeval/<set>/` with `pano/`, `items.json`,
`boxes.json` (panorama degrees, so boxes don't depend on any render) and
`paths/<variant>.json` plus `<variant>_debug.json` (per-frame mode, aim,
hotspot, ball). Two sets, 13 reviewed moments: `seg9-60` (segment 9 from
60 s, the first test clip) and `seg3-225` (segment 3 from 225 s, crossing
into segment 4).

`playbench.py` caches follow.py's pass A (player detections) per clip in
`footage/rec3/follow/passA_<seg>_<start>_<seconds>.json`, so a variant
plans in seconds. Variants are follow.py feature strings; a feature
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

Add moments by appending clip times to `playeval.py make` for an existing
set (it keeps the others) or making a new set from another stretch of the
game; `playbench.py` lists the clips in `CLIPS`.

## 3. Ball tracking: clip labels

The ball's position every 2 s along a clip, labelled on both phones, so the
tracker is measured directly (found / wrong / missed) rather than through
the play boxes.

| Step | Command |
|---|---|
| Build a clip set | `uv run processing/spikes/labelset.py footage/rec3 clip <name> <segment> <start s> <seconds> [step s]` |
| Label it | preview `ball-labels-seg9` / `ball-labels-seg3` (`processing/spikes/label/server.py footage/rec3/labels_<name> [port]`) |
| Score the tracker | `uv run processing/spikes/balleval.py footage/rec3 <name> ["NAME=value,..." ...]` |
| Score detectors per frame | `uv run processing/spikes/ballheat.py footage/rec3 <name> [wasb] [tracknetv2] [tracknetv3]` |
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
| `flow` | `flowcands.py` | motion-only candidates (double difference, size, consistent step); also writes the goal-net change series `net_*.json` |

Every candidate carries a `judge` score from `balljudge.py` (a 48 px crop
classifier; `data`, `train`, `eval`, `clip`). `ballcompare.py` compares the
stock and fine-tuned detectors on the random label set's held-back frames.

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

Detectors per labelled frame (`ballheat.py`, 2026-10-18; balls in play on a
phone: 48 on seg 9, 58 on seg 3):

| Detector | seg 9 found | seg 3 found | candidates / frame |
|---|---|---|---|
| fine-tuned RF-DETR, conf >= 0.1 | 27% | 60% | 1.0 / 0.9 |
| WASB (football weights), score >= 0.05, judge >= 0.3 | 29% | 60% | 1.4 / 1.7 |
| union of the two | 44% | 78% | 2.4 / 2.6 |

The candidate ceiling (a usable candidate near the ball at all) is ~62% of
moments; the misses are mostly balls under 8 px, 43-100+ m away. Witnesses
from other motion (keeper bearings, motion blobs, net ripple, facing
direction: `witness.py`, `flowcands.py`, `facing.py`) were scored the same
way and none raises the found-rate; the backlog has their numbers.

## 4. Where things are

| Path | What |
|---|---|
| `footage/rec3/pitch/` | pitch map, goals, background panorama, overview map |
| `footage/fixtures/pitch/` | mapping regression fixtures |
| `footage/rec3/follow/` | renders, comparison videos, pass A caches, path files |
| `footage/rec3/playeval/` | play-box sets and benched paths, `bench.json` |
| `footage/rec3/labels/` | the random ball label set (detector training data) |
| `footage/rec3/labels_<name>/` | clip ball label sets |
| `footage/rec3/detect/` | ball candidate files by clip and tag, net series |
| `footage/rec3/balldata/`, `balltrain/`, `balljudge/` | detector dataset and checkpoints, judge crops and models |
| `footage/rec3/fullgame/` | whole-game render pieces and `game.mp4` (`fullgame.py`) |

Preview servers are in `.claude/launch.json` (`ball-labels`, `ball-labels-seg9`,
`ball-labels-seg3`, `play-eval`, `play-eval-seg9`); they listen on localhost only.
