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

## Layout

| Path | What |
|---|---|
| `processing/` | GPU pipeline: sync, stitch, track, render; Phase 0 spikes, including the labelling tools and model training (`processing/spikes/`) |
| `schemas/` | Pinned copy of the main repo's manifest schema |
| `docs/` | Findings, data layout, how processing is tested, backlog |
| `footage/`, `datasets/` | Local only, never committed; see [docs/DATA.md](docs/DATA.md) |

## License

**AGPL-3.0-only** (see [LICENSE](LICENSE)), except the pinned schema copy,
which is Apache-2.0 like its source. Anyone running a modified version of
this pipeline as a network service, the maintainers included, must offer
its users the modified source (AGPL section 13).

Contributions require the contributor license agreement; see
[CONTRIBUTING.md](CONTRIBUTING.md).
