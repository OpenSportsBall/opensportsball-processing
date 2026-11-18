# Running the benchmarks

How to run every benchmark, which files they need, which can be thrown
away, and where a result is written down. The detail behind each
benchmark (how its sets are made, what its numbers mean, its history) is
in [TESTING-PROCESSING.md](TESTING-PROCESSING.md); the whole `footage/`
layout is in [DATA.md](DATA.md).

## What there is

| Benchmark | Measures | Script | Reads | Baseline |
|---|---|---|---|---|
| Must-see moments (play bench) | whether the follow cam's path has each reviewed moment in frame: 62 moments in 12 clips of 4 games (104 minutes) | `playbench.py` | caches | `processing/bench_baselines.json` |
| Ball sets | whether the ball detector, judge and tracker find the labelled ball: rec3's `seg3` (trained on) and `seg9` (held out) | `balleval.py` | caches | `processing/bench_baselines.json` |
| Pitch regression | the pitch and goal fit against a result checked by eye | `tests/test_pitch_regression.py` | fixtures | passes or fails (fails 4 checks since 2026-10-22: TESTING-PROCESSING.md, section 1) |
| Marked-field pitch benchmark | a pitch fit against every marking traced on rec4's paint | `pitchbench.py` | fixtures | the fixture's `baseline_scores.txt` |
| New-camera ball test | the standard detector and judge on SoccerTrack's wide view, which no model of ours has seen | `stbench.py` | SoccerTrack's video (not in the store: DATA.md, "External datasets") | FINDINGS.md |
| Unit tests | the code without footage | `pytest` | nothing | all pass |

The first two are the standard benchmarks: run them before and after any
change to the follow cam's rules, the ball models or the detection
passes.

## Running them

Everything runs from the repository root.

    uv run processing/spikes/bench.py check     # everything a run reads is here, and the caches match their inputs
    uv run processing/spikes/bench.py run       # must-see and ball sets, compared with the baselines; exits 1 if one is below
    uv run processing/spikes/bench.py run play  # one of the two

`run` takes a few minutes and reads no video: it plans the camera path
again from the caches and scores it. One at a time, and the others:

    uv run processing/spikes/playbench.py footage/bench/rec3 footage/bench/rec4cal footage/bench/rec7 footage/bench/rec8 <features>
    OSB_BALL_TAG=win uv run processing/spikes/balleval.py footage/bench/rec3 seg3
    cd processing && uv run --extra dev pytest tests --deselect tests/test_pitch_regression.py
    cd processing && uv run --extra dev pytest tests/test_pitch_regression.py
    uv run processing/spikes/pitchbench.py footage/bench/fixtures/pitch/night-2026-10-21 <pitch.json>
    uv run processing/spikes/stbench.py 2.0 999

`<features>` is the standard list in `bench_baselines.json`.

### When the caches must be remade

The caches are the slow, video-reading half of the pipeline, kept so a
bench run doesn't repeat it. They are only valid for the code and models
that made them:

| You changed | Remake | How | Time (all 12 clips, an M-series Mac) |
|---|---|---|---|
| the follow cam's rules, framing, features or constants (`follow.py` after pass A, `play.py`) | nothing | | |
| the ball detector or judge (`footage/models/`), or `ballwin.py` | ball candidates | `ballwin.py footage/bench/<rec> <segment> <start> <seconds> footage/models/ball_detector.pth 10` per clip | 2.3 hours |
| the player detector or `roles.py` | roles | `roles.py footage/bench/<rec> <segment> <start> <seconds> <teams> 4` per clip | 2.5 hours |
| pass A (`follow.py` up to the blob detections, the pitch field), the cameras, the sync or the pitch map | pass A | delete `bench-cache/<rec>/follow/passA_*`; `bench.py run play` remakes them | 1.5 hours |

A clip's segment, start and length are in its
`footage/bench/<rec>/playeval/<set>/clip.json`. Remaking needs the clip's
video in the store (`benchstore.py` says whether it is). Then:

    uv run processing/spikes/bench.py stamp     # record the inputs the caches now match
    uv run processing/spikes/bench.py run

`bench.py check` reports caches stale when the cameras, sync report,
pitch map, goals or standard models have changed since the stamp. It
does not know about code: after a change to a detection pass, remake the
caches whether or not it says so.

## What must be kept

| Path | What | Size | Kept in |
|---|---|---|---|
| `footage/bench/<rec>/` without the video | what the benchmarks read and nothing can remake: manifest, sync report and cameras (`phase0/`), pitch map, team colours, the must-see sets (`playeval/`), the ball label sets, `LABELS.md`, the pitch fixtures | under 2 GB | S3 `bench/`, Hugging Face |
| `footage/bench/<rec>/<deviceId>/*.mp4` | the video the clips cover, cut to what is read (`benchstore.py` checks it) | 52 GB | S3 `bench/`, Hugging Face; the phones' uploads are in S3 `accounts/` |
| `footage/models/` | the standard ball detector and judge, and their README | 0.1 GB | S3 (the cloud engine's models); GitHub release assets once published (MODELS.md) |
| `processing/bench_baselines.json` | the scores to beat | | git |
| the code | | | git |

The video need not be on the machine: `bench.py run` works without it.
It is needed only to remake caches, and then only the clips being
remade. Without it nothing in the caches can be made again, so it is
never deleted from S3.

## What can be discarded

| Path | What | Cost of losing it |
|---|---|---|
| `footage/bench-cache/<rec>/` | pass A caches, ball candidates, roles, backgrounds, the stamp | hours to remake from the video (above); kept in S3 `bench-cache/` to save that. Not shared on Hugging Face: a contributor makes their own, with the code they are testing |
| `footage/bench-out/<rec>/` | what runs write: camera paths, `playeval/bench.json`, camera logs, debug images, `cache-before-regen/` | nothing; delete at will |
| `footage/train/<rec>/` | training datasets, judge crops, checkpoints | remade by training; a model that became standard is already in `footage/models/` |
| the video on this machine | | a download from S3 or Hugging Face (`sharedata.py fetch`, `footagestore.py restore`) |
| anything else under `footage/` | working recordings and finished experiments | not read by any benchmark; DATA.md says what each is |

## Recording a result

| What happened | Write it | Where |
|---|---|---|
| a change moved a standard score | the new baseline, in the same commit as the change, with the reason in the commit message | `processing/bench_baselines.json` |
| the caches were remade | which models and code made them, and the date | TESTING-PROCESSING.md, section 6 |
| a model was replaced | its hash, what it was trained on, and its bench against the one it replaced | `footage/models/README.md`, `packaging/cloud/models.json` |
| an experiment gave an answer worth keeping (adopted or not) | one row: what was tried, the scores, the verdict | FINDINGS.md |
| an experiment is unfinished, or an idea is not yet tried | an item | BACKLOG.md (delete it when done) |
| a must-see set or label set was added, or a held-out mark changed | the set | `footage/bench/LABELS.md`, the Recordings table in DATA.md, TESTING-PROCESSING.md section 6 |
| the store's files changed | upload them | `footagestore.py backup` (S3), then `sharedata.py upload` (Hugging Face) |

A score is quoted with what made it: the features, the recordings, and
whether the caches were current. Ball labels marked held out
(`bench/LABELS.md`) are never trained on, and every new labelling sets
part aside before training (TESTING-PROCESSING.md, "Held-out ball
labels").

## The baselines now

Made 2026-11-18 from caches remade from the video with the standard
models (`ball_detector.pth` c69adbc8, `ball_judge.pt` bd32076c, the stock
RF-DETR Medium for players):

| | Score |
|---|---|
| Must-see, rec3 | 19/26 |
| Must-see, rec4cal | 13/15 |
| Must-see, rec7 | 4/5 |
| Must-see, rec8 | 13/16 |
| Ball, rec3 seg3 | 41/53 found, 1 wrong |
| Ball, rec3 seg9 (held out) | 24/37 found, 6 wrong |
