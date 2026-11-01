# Data: where recordings, labels and models live

Footage shows people, some possibly minors: nothing here goes in git
(`footage/` is ignored) or anywhere public. Everything is on the working
Mac under `footage/` and backed up to the footage bucket in S3
(`FootageBucket` in `infra/cdk-outputs.json`).

## Layout

On the Mac:

| Path | What |
|---|---|
| `footage/<rec>/` | one recording: `manifest.json`, the phones' video in `<deviceId>/`, and everything made from it (below) |
| `footage/<rec>/labels*/` | a ball label set: `items.json` (moments, frames, suggestions), `labels.json` (the labels), `images/`, `pano/` |
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
| rec3 | 2026-10-15 | `f68af87f` | first game, white ball; bench | `labels` (random, 408 frames), `labels_seg3`, `labels_seg9` (clips) |
| rec4 | 2026-10-21 | `c92f9740` | night game, white ball; `rec4cal` (corrected camera model, bench), `rec4bench`, `rec4cal_cmp` are its variants | `labels_seg9` (not labelled) |
| rec5 | 2026-10-28 | `38a1d10f` | orange ball; `rec5-game` (working copy), `rec5-cal` are its variants | `rec5-game/labels` (ball mode, 98 frames) |
| rec6 | 2026-10-30 | `1f3a915a` | first daylight game | `labels` (random, both phones, 196 frames) |

Add a row for every new recording.

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
needed, with the manifest's object keys:

    aws s3 cp --profile sbv --recursive s3://<footage bucket>/accounts/<account>/games/<game>/recordings/<recording>/ footage/<rec>/

## A new recording

1. Download it: `aws s3 cp` as above, into `footage/rec<N>/`.
2. `phase0.py`, then whatever it is for (labels, pitch, a full game).
3. Add it to the table above and run `footagestore.py backup`.
