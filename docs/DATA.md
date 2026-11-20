# Data: where recordings, labels and models live

Footage shows identifiable adult players: nothing here goes in git
(`footage/` is ignored) or anywhere public. What the benchmarks read is
on the working Mac under `footage/` and backed up to the footage bucket
in S3 (`FootageBucket` in the main repo's `infra/cdk-outputs.json`); the
phones' full uploads are in S3 only.

Trained weights (`*.pt`, `*.pth`, `*.onnx`) stay out of git too: the standard
models are pinned by hash in `packaging/cloud/models.json` and published
as GitHub release assets, so `uv run packaging/cloud/models.py fetch`
puts them in `footage/models/` on a fresh clone ([MODELS.md](MODELS.md)
says where each came from). Before training
on a public dataset (SoccerNet included), check its license, and record
which datasets every released model was trained on (for the standard
models, in [MODELS.md](MODELS.md)).

## Layout

On the Mac:

| Path | What |
|---|---|
| `footage/bench/` | the bench store: everything the benchmarks read that nothing can remake (below) |
| `footage/bench-cache/` | what is made from the store and only saves time |
| `footage/bench-out/` | what benchmark runs write; safe to delete |
| `footage/train/` | what training on the store's label sets makes |
| `footage/models/` | the standard ball detector and judge every game uses, with a README of where each came from |
| `footage/onnxbench/`, `ballmiss/`, `benchscan/` | finished experiments' scripts, logs and exports (BACKLOG.md and FINDINGS.md say what each measured); read by no benchmark |
| `footage/symlinks.json` | every symlink under `footage/`, written by the backup |

No working recording is on the Mac: the folders of rec2 to rec8 and their
variants, and the experiments that linked into them (`ab_offpitch`,
`mainbench`, `fieldfix`, `fieldfix-old`, `balldata_rec3_rec5`,
`balldata_rec3_rec5_rec6`), were deleted on 2026-11-18, once the games'
videos were published and the benchmark data was in the store. Older
entries in BACKLOG.md, FINDINGS.md, MODELS.md and TESTING-PROCESSING.md
name paths in them (`footage/rec3/...`, `footage/rec6/balljudge/...`):
those files are gone unless the same path exists under `bench/`,
`bench-cache/` or `train/`. What S3's `data/` still holds of them is
whatever the last backup before that day uploaded.

A new recording is worked on in a folder of its own until its inputs go
into the store ("A new recording", below):

| Path | What |
|---|---|
| `footage/<rec>/` | one recording: `manifest.json`, the phones' video in `<deviceId>/`, and everything made from it |
| `footage/<rec>/labels*/` | a ball label set: `items.json` (moments, frames, suggestions), `labels.json` (the labels), `images/`, `pano/`; `heldout.json` when it is (partly) held out for testing (`holdout.py`) |
| `footage/<rec>/playeval/` | play-box sets (TESTING-PROCESSING.md section 2) |
| `footage/<rec>/phase0/`, `pitch/`, `lens/`, `teams/`, `bg/` | sync, cameras, pitch map, team colours, background |
| `footage/<rec>/detect/`, `follow/`, `fullgame*/` | candidates, caches and renders (made from the video; hours to remake) |
| `footage/<rec>/balldata/`, `balltrain*/`, `balljudge/`, `heattrain/` | training sets and trained models from this recording |
| `footage/balldata_<recs>/` | a merged training set (`ballmerge.py`; links into the recordings' `balldata/`) |

### The bench store

The benchmarks are self-contained: `footage/bench/`, `footage/models/` and
the code are all a run needs (`bench.py check` says whether they are
there, `bench.py run` runs the standard benchmarks against the baselines
in `processing/bench_baselines.json`). A recording in the store holds
only inputs; `processing/spikes/recpaths.py` sends everything made from
them elsewhere, under the same relative path, so a run never changes the
store:

| Path | What | Shared |
|---|---|---|
| `bench/<rec>/` | `manifest.json`, `phase0/` (sync report, cameras), `pitch/`, `teams/`, `lens/`, must-see sets (`playeval/<set>/`: `clip.json`, `items.json`, `boxes.json`, `pano/`), ball label sets (`labels*/`, `engine/labels/ball-*/`), and in `<deviceId>/` the video segments the sets cover | S3 `bench/`, Hugging Face |
| `bench/LABELS.md` | every ball label set and whether it is trained on or held out | |
| `bench/fixtures/pitch/` | pitch-mapping fixtures and the marked-field benchmark | |
| `bench/soccertrack/` | SoccerTrack's annotations (External datasets, below) | |
| `bench-cache/<rec>/` | `follow/passA_*.json` (player detections per clip), `detect/` (ball candidates, roles, net series), `bg/` (backgrounds), `labels*/heat_*.json`; `inputs.json` records the cameras, pitch map and models they were made from (`bench.py stamp`), and `bench.py check` reports them stale when those change. Made from the video: hours to remake | S3 `bench-cache/` only: not shared on Hugging Face (a contributor makes their own from the video) |
| `bench-out/<rec>/` | `follow/` (paths, camera logs, debug images, renders), `playeval/<set>/paths/`, `playeval/bench.json` | never |
| `train/<rec>/` | `balldata/`, `balltrain*/`, `balljudge/` (crops and a recording's own judges), `heattrain/` | never (the models that become standard are published: `footage/models/`) |

The video is part of the store in S3 and on Hugging Face but need not be
on the machine: the standard benchmarks run from the caches, and
`benchstore.py` lists the files the sets read and which part of each. A
file the sets read only the start of is cut there (2026-11-18: a stream
copy, the frames that are left unchanged), ten seconds past the last
frame read; its manifest entry then has `trimmedToMs`, the cut file's
`bytes` and `sha256`, and the upload's own as `originalBytes` and
`originalSha256`. `durationMs` stays the upload's, so the game's clock is
the same; reading past the cut finds no frames. The uploads themselves
are under `accounts/` in S3. A file read only at its end is kept whole
(cutting its start would move every time in it). A working recording
(`footage/<rec>`, not in the store) keeps all of the above in its own
folder.

`<rec>` is `rec<N>` in recording order; a variant of a recording (another
calibration, a comparison) is `rec<N>-<what>` or `rec<N><what>` and links
to the original's video and caches instead of copying them.

In S3:

| Prefix | What | Written by |
|---|---|---|
| `accounts/<account>/games/<game>/recordings/<recording>/` | the phones' raw video and the manifest | the app (deleting a game on the website deletes it) |
| `bench/` | `footage/bench/`, video included | `footagestore.py backup`; the video by copies within the bucket from `accounts/` |
| `bench-cache/` | `footage/bench-cache/` | `footagestore.py backup` |
| `data/` | `footage/` without `bench/`, `bench-cache/`, `bench-out/`, video and symlinks; raw video only for a recording no longer in `accounts/` | `footagestore.py backup` |

The bucket is not versioned: an overwritten or deleted object is gone.
The backup only uploads; it never deletes from S3.

## Recordings

| Rec | Date | Game | In the store | What | Label sets |
|---|---|---|---|---|---|
| rec2 | 2026-10-12 | `a901c618` | no | first field test (2 segments); raw video only in S3 `data/rec2/` | |
| rec3 | 2026-10-14 | `f68af87f` | `bench/rec3` | first game, night, white ball (an evening game: 2026-10-15 UTC); bench | `labels` (random, 408 frames; trained), `labels_seg3` (clip; trained), `labels_seg9` (clip; **held out**) |
| rec4 | 2026-10-21 | `c92f9740` | `bench/rec4cal` | night game, white ball; the store has it with the corrected camera model, as `rec4cal` (its other variants, `rec4bench` and `rec4cal_cmp`, are deleted) | none (`labels_seg9` was never labelled, and is deleted) |
| rec5 | 2026-10-28 | `38a1d10f` | `bench/rec5-game` (the ball labels and what they need: manifest, sync report, cameras, pitch map; no video) | orange ball | `labels` (ball mode, 98 frames; trained, nothing held out) |
| rec6 | 2026-10-30 | `1f3a915a` | `bench/rec6` | first daylight game | `labels` (random, both phones, 196 frames; trained, nothing held out) |
| rec7 | 2026-11-04 | `3ba4b9f2` | `bench/rec7` (since 2026-11-15) | night game, Pixel 11 + iPhone 14 Pro, ~6.2 m mount, yellow ball; its own judge `train/rec7/balljudge/judge_v4.pt`; play boxes `playeval/seg0-1484` (5) | `labels_seg6` (clip, seg 6 100-220 s, both phones, 119 frames; seconds 180-220 held out since 2026-11-13, but `judge_v4` trained on all of it before) |
| rec8 | 2026-11-11 | `baed850f` | `bench/rec8` | night game, Pixel 11 (Main) + iPhone (Secondary, which stopped after 9 segments: Main alone from then on, the right half of the pitch); processed by the Mac app, whose folder in `~/Movies/OpenSportsBall/` holds the whole game's video; play boxes `playeval/seg0-0` (2), `seg0-600` (10), `seg0-3000` (1), `seg0-3600` (3): plays the follow cam missed | `labels_seg11-136`: 13 frames of Main, 1 s apart, game 54:31-54:43 around the missed play at 54:37 (Secondary has no footage there), the 2026-11-15 run's candidates as suggestions; labelled 2026-11-15 (7 ball, 3 unsure, 3 none: the ball crosses Main's frame right to left and is out of it from 54:41); the last 4 (54:40-54:43) held out. `engine/labels/ball-1`: the Mac app's `label ball` set, 50 frames of Main across the game (29 detected, 21 bridged), labelled 2026-11-14 (41 ball, 5 none, 4 unsure) and shared; downloaded 2026-11-17 from the contributions bucket (contribution `23eb2f20`, record in `engine/share/ball-1.cloud.json`), as the app's copy was gone. The upload has no `items.json` or `heldout.json`: both rebuilt from `set.json`, the manifest and `detect/`'s chunk lengths, and the windows give the same held-out flags the engine wrote. Held out: the last quarter of every 10-minute chunk (the engine's mark), which holds 6 of the 50 (all ball), well under the ~30 the rule asks for; the frame at 54:23.6 is also left out of training, within 20 s of `labels_seg11-136`'s held-out stretch. Not trained on yet |
| rec9 | 2026-11-18 | `e1eec75d` | `bench/rec9` | night game, Pixel 11 (Main) + iPhone18,1 (Secondary), mount ~6.2 m up at the halfway line; processed by the Mac app, whose folder in `~/Movies/OpenSportsBall/` holds the video (`bench/rec9`'s segments link to it). In the bench store for its pitch data (`bench/rec9/README.md`): the stitch whose focal lengths were ~2% short, its panorama, the 15 keypoints the user marked on it (`pitch/taps.json`, and as phone pixels `pitch/taps_src.json`), the pitch fits before and after the taps, and reference joint fits with their held-out scores (`joint/`) | none |

Add a row for every new recording. For every ball label set, say whether it
is trained on or held out for testing; every new labelling holds part back
(`docs/TESTING-PROCESSING.md`, "Held-out ball labels").

## External datasets

| Name | Where | What | Licence | Use |
|---|---|---|---|---|
| SoccerTrack (Scott et al., CVPRW 2022) | `footage/bench/soccertrack/wide_view/` (`videos/`, `annotations/`), `footage/bench/soccertrack/fisheye_keypoints.json` | 66 clips of 30 s, 25 fps, 6500 x 1000: one fixed fisheye camera beside the pitch, unwrapped; two university 11-v-11 games (2020-02-20, 2022-02-20), daylight, white ball. Boxes for every player and the ball (team and player `BALL`) in every frame; player boxes were made from the players' GNSS tracks. Downloaded 2026-11-09 from [Kaggle](https://www.kaggle.com/datasets/atomscott/soccertrack) (version 6; the drone and pre-drawn videos left out). | LGPL-3.0 (Kaggle); players gave signed consent (University of Tsukuba ethics approval) | **Held out: test only, never trained on.** A camera, field and light none of our models has seen |

## Backing up

    uv run processing/spikes/footagestore.py check     # what is not in S3; exit 1 if anything
    uv run processing/spikes/footagestore.py backup    # upload it, then check

Run `backup` after labelling, training or any run worth keeping. `check`
also reports, per recording, where its raw video is: `accounts` (the
app's upload), `data` (our copy) or `LOCAL ONLY`, which `backup` fixes by
copying it to `data/<rec>/`. The backup never deletes from S3, so
`data/` keeps what was uploaded from the working folders deleted on
2026-11-18.

## Restoring on a new machine

    uv run processing/spikes/footagestore.py restore

brings back `data/`, `bench/` and `bench-cache/` into `footage/` and recreates the
symlinks. `data/` includes the deleted working folders as last backed up
(without their video); the benchmarks need only `bench/` and
`bench-cache/`. Raw video still in `accounts/` is downloaded per recording when
needed, through the footage CloudFront distribution (its first TB a month
is free; S3's transfer out is not), checked against the manifest's sizes
and SHA-256:

    uv run processing/spikes/footagestore.py fetch <rec> [--segs a-b]

It needs `footage/<rec>/manifest.json` and AWS access that can read the
signing key in SSM (`/OpenSportsBall/footage-signing-key`). A variant
whose phone folders link to another recording (`rec5-game` -> `rec5`) gets
its video by fetching that one.

## Sharing with contributors

Contributors get the benchmark data from a gated Hugging Face dataset,
[`OpenSportsBall/bench`](https://huggingface.co/datasets/OpenSportsBall/bench):
its card is public, but each person's access request needs a maintainer's
approval, and they accept the terms in the card ([DATASET_CARD.md](DATASET_CARD.md))
when they ask. It mirrors `footage/` path for path and holds `bench/` (with
its video and every ball label set) and `models/`: about 54 GB, 52 GB of
it video. The caches (`bench-cache/`) are not shared: a contributor makes
their own from the video ([BENCHMARKS.md](BENCHMARKS.md)). Full recordings stay in S3 only.

    uv run processing/spikes/sharedata.py fetch [--no-video]              # contributors, after approval and `hf auth login`
    uv run processing/spikes/sharedata.py list                              # what upload would send
    uv run processing/spikes/sharedata.py upload                            # maintainers: after changing any of it
    uv run processing/spikes/bench.py check                                 # is what the benchmarks read all here?

`fetch --no-video` (about 3 GB) is enough to run the standard benchmarks;
the video is for remaking caches (a new detector, other settings) and
for looking at footage.

`upload` never sends the app's upload state (`upload.json`: the account id
and S3 upload ids) and removes it from the dataset if it is there. It stops,
naming the files, when a text file it would send holds a home-directory
path, an account id or an `accounts/<id>` key; `list` shows the same. Model
files aren't looked into: a checkpoint keeps the paths it was trained with.

Approve or reject requests on the dataset's **Settings → Gated user access**
page (Hugging Face emails the maintainers about new ones). Approve people you
know are working on the project; removing someone there stops further
downloads but can't recall a copy they already have. To share a new label set
or pitch map, put its recording in the bench store (TESTING-PROCESSING.md
section 6, "Adding a set"), add it to `bench/LABELS.md` and run `upload`.

## A new recording

1. Copy its manifest (a few KB, so straight from S3), then fetch its video
   as above:
   `aws s3 cp --profile sbv s3://<footage bucket>/accounts/<account>/games/<game>/recordings/<recording>/manifest.json footage/rec<N>/`
   and `footagestore.py fetch rec<N>`.
2. `phase0.py`, then whatever it is for (labels, pitch, a full game).
3. Add it to the table above and run `footagestore.py backup`.
4. Once it has labels, must-see boxes or a pitch map worth sharing, copy
   its inputs into the bench store (TESTING-PROCESSING.md section 6,
   "Adding a set") and run `sharedata.py upload`.
