# OpenSportsBall processing

The video pipeline for [OpenSportsBall](https://github.com/OpenSportsBall/opensportsball): two phones on a
dual-phone mount record the full field; this code syncs the two videos,
stitches them into one panorama, follows the play with a virtual camera, and
renders the game video.

> Status: pre-alpha. The Phase 0 spikes in `processing/spikes/` are the
> working pipeline today. See [docs/FINDINGS.md](docs/FINDINGS.md).

## How it fits

The phone app, website, API and AWS infrastructure live in the main
OpenSportsBall repository (Apache-2.0). This repository never shares code
with it, only data:

- **Input:** the upload manifest (pinned copy in
  [schemas/](schemas/README.md)) and the segment files in S3.
- **Trigger:** the main repo's stack emits a `RecordingReady` event when a
  recording's uploads are complete.
- **Output:** a 1080p MP4, a detections file, and optionally a low-resolution
  full-field panorama.
- **Engine:** `processing/spikes/engine.py` runs one game through files
  (`prepare`, then `run` once the pitch question is answered;
  JSON progress on stdout). The Mac app drives it this way; see
  [docs/TESTING-PROCESSING.md](docs/TESTING-PROCESSING.md), section 7.
  A script or a model can drive it the same way: the questions say in
  words what is asked and come with pictures to look at; see
  [docs/ENGINE-AGENTS.md](docs/ENGINE-AGENTS.md).

## Layout

| Path | What |
|---|---|
| `processing/` | GPU pipeline: sync, stitch, track, render; Phase 0 spikes, including the labelling tools and model training (`processing/spikes/`) |
| `schemas/` | Pinned copies of the main repo's schemas: the upload manifest and the engine's job, questions, answers and progress events |
| `docs/` | Findings, data layout, how processing is tested, backlog |
| `footage/`, `datasets/` | Local only, never committed; benchmark data comes from Hugging Face (below); see [docs/DATA.md](docs/DATA.md) |

## Benchmark data

The video, labels, pitch maps and models the benchmarks run on are in the
gated Hugging Face dataset
[OpenSportsBall/bench](https://huggingface.co/datasets/OpenSportsBall/bench).
The footage shows identifiable people, so request access there and a
maintainer will approve it. Then:

    hf auth login
    uv run processing/spikes/sharedata.py fetch

See [docs/DATA.md](docs/DATA.md#sharing-with-contributors) for what's in it
and [docs/TESTING-PROCESSING.md](docs/TESTING-PROCESSING.md) for the
benchmarks.

## License

**AGPL-3.0-only** (see [LICENSE](LICENSE)), except the pinned schema copies,
which are Apache-2.0 like their source. Anyone running a modified version of
this pipeline as a network service, the maintainers included, must offer
its users the modified source (AGPL section 13).

Contributions require the contributor license agreement; see
[CONTRIBUTING.md](CONTRIBUTING.md).
