# Data: where recordings, labels and models live

Footage shows identifiable adult players: nothing here goes in git
(`footage/` is ignored) or anywhere public. Everything is on the working
Mac under `footage/` and backed up to the footage bucket in S3
(`FootageBucket` in the main repo's `infra/cdk-outputs.json`).

Trained weights (`*.pt`, `*.onnx`) stay out of git too. Before training
on a public dataset (SoccerNet included), check its license, and record
which datasets every released model was trained on (for the standard
models, in `footage/models/`'s README).

## Layout

On the Mac:

| Path | What |
|---|---|
| `footage/<rec>/` | one recording: `manifest.json`, the phones' video in `<deviceId>/`, and everything made from it (below) |
| `footage/<rec>/labels*/` | a ball label set: `items.json` (moments, frames, suggestions), `labels.json` (the labels), `images/`, `pano/`; `heldout.json` when it is (partly) held out for testing (`holdout.py`) |
| `footage/<rec>/playeval/` | play-box sets (TESTING-PROCESSING.md section 2) |
| `footage/<rec>/phase0/`, `pitch/`, `lens/`, `teams/`, `bg/` | sync, cameras, pitch map, team colours, background |
| `footage/<rec>/detect/`, `follow/`, `fullgame*/` | candidates, caches and renders (made from the video; hours to remake) |
| `footage/<rec>/balldata/`, `balltrain*/`, `balljudge/`, `heattrain/` | training sets and trained models from this recording |
| `footage/balldata_<recs>/` | a merged training set (`ballmerge.py`; links into the recordings' `balldata/`) |
| `footage/models/` | the standard ball detector and judge every game uses, with a README of where each came from |
| `footage/bench/` | the bench store (TESTING-PROCESSING.md section 6) |
| `footage/fixtures/` | pitch-mapping fixtures |
| `footage/symlinks.json` | every symlink under `footage/`, written by the backup |

`<rec>` is `rec<N>` in recording order; a variant of a recording (another
calibration, a comparison) is `rec<N>-<what>` or `rec<N><what>` and links
to the original's video and caches instead of copying them.

In S3:

| Prefix | What | Written by |
|---|---|---|
| `accounts/<account>/games/<game>/recordings/<recording>/` | the phones' raw video and the manifest | the app (deleting a game on the website deletes it) |
| `bench/` | `footage/bench/`, video included | `footagestore.py backup` |
| `data/` | `footage/` without `bench/`, video and symlinks; raw video only for a recording no longer in `accounts/` | `footagestore.py backup` |

The bucket is not versioned: an overwritten or deleted object is gone.
The backup only uploads; it never deletes from S3.

## Recordings

| Rec | Date | Game | What | Label sets |
|---|---|---|---|---|
| rec2 | 2026-10-12 | `a901c618` | first field test (2 segments); raw video only in `data/rec2/` | |
| rec3 | 2026-10-14 | `f68af87f` | first game, night, white ball (an evening game: 2026-10-15 UTC); bench | `labels` (random, 408 frames; trained), `labels_seg3` (clip; trained), `labels_seg9` (clip; **held out**) |
| rec4 | 2026-10-21 | `c92f9740` | night game, white ball; `rec4cal` (corrected camera model, bench), `rec4bench`, `rec4cal_cmp` are its variants | `labels_seg9` (not labelled) |
| rec5 | 2026-10-28 | `38a1d10f` | orange ball; `rec5-game` (working copy), `rec5-cal` are its variants | `rec5-game/labels` (ball mode, 98 frames; trained, nothing held out) |
| rec6 | 2026-10-30 | `1f3a915a` | first daylight game | `labels` (random, both phones, 196 frames; trained, nothing held out) |
| rec7 | 2026-11-04 | `3ba4b9f2` | night game, Pixel 11 + iPhone 14 Pro, ~6.2 m mount, yellow ball; `balljudge/judge_v4.pt`; play boxes `playeval/seg0-1484` (5) | `labels_seg6` (clip, seg 6 100-220 s, both phones, 119 frames; trained, nothing held out) |

Add a row for every new recording. For every ball label set, say whether it
is trained on or held out for testing; every new labelling holds part back
(`docs/TESTING-PROCESSING.md`, "Held-out ball labels").

## External datasets

| Name | Where | What | Licence | Use |
|---|---|---|---|---|
| SoccerTrack (Scott et al., CVPRW 2022) | `footage/soccertrack/wide_view/` (`videos/`, `annotations/`), `footage/soccertrack/fisheye_keypoints.json` | 66 clips of 30 s, 25 fps, 6500 x 1000: one fixed fisheye camera beside the pitch, unwrapped; two university 11-v-11 games (2020-02-20, 2022-02-20), daylight, white ball. Boxes for every player and the ball (team and player `BALL`) in every frame; player boxes were made from the players' GNSS tracks. Downloaded 2026-11-09 from [Kaggle](https://www.kaggle.com/datasets/atomscott/soccertrack) (version 6; the drone and pre-drawn videos left out). | LGPL-3.0 (Kaggle); players gave signed consent (University of Tsukuba ethics approval) | **Held out: test only, never trained on.** A camera, field and light none of our models has seen |

## Backing up

    uv run processing/spikes/footagestore.py check     # what is not in S3; exit 1 if anything
    uv run processing/spikes/footagestore.py backup    # upload it, then check

Run `backup` after labelling, training or any run worth keeping. `check`
also reports, per recording, where its raw video is: `accounts` (the
app's upload), `data` (our copy) or `LOCAL ONLY`, which `backup` fixes by
copying it to `data/<rec>/`.

## Restoring on a new machine

    uv run processing/spikes/footagestore.py restore

brings back `data/` and `bench/` into `footage/` and recreates the
symlinks. Raw video still in `accounts/` is downloaded per recording when
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
when they ask. It mirrors `footage/` path for path and holds `bench/`,
`models/`, and the pitch maps and ball label sets of the recordings outside
the bench (`sharedata.py`'s `ANNOTATED`), about 37 GB. Full recordings stay in
S3 only.

    uv run processing/spikes/sharedata.py fetch [--annotations-only]   # contributors, after approval and `hf auth login`
    uv run processing/spikes/sharedata.py list                         # what upload would send
    uv run processing/spikes/sharedata.py upload                       # maintainers: after changing any of it

`upload` never sends the app's upload state (`upload.json`: the account id
and S3 upload ids) and removes it from the dataset if it is there. It stops,
naming the files, when a text file it would send holds a home-directory
path, an account id or an `accounts/<id>` key; `list` shows the same. Model
files aren't looked into: a checkpoint keeps the paths it was trained with.

Approve or reject requests on the dataset's **Settings → Gated user access**
page (Hugging Face emails the maintainers about new ones). Approve people you
know are working on the project; removing someone there stops further
downloads but can't recall a copy they already have. To share a new label set
or pitch map, add it to `ANNOTATED` (or to the bench store) and run `upload`.

## A new recording

1. Copy its manifest (a few KB, so straight from S3), then fetch its video
   as above:
   `aws s3 cp --profile sbv s3://<footage bucket>/accounts/<account>/games/<game>/recordings/<recording>/manifest.json footage/rec<N>/`
   and `footagestore.py fetch rec<N>`.
2. `phase0.py`, then whatever it is for (labels, pitch, a full game).
3. Add it to the table above and run `footagestore.py backup`.
4. Once it has labels or a pitch map worth sharing, add it to
   `sharedata.py`'s `ANNOTATED` and run `sharedata.py upload`.
