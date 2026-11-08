---
license: other
license_name: opensportsball-bench-terms
pretty_name: OpenSportsBall benchmark data
tags:
  - video
  - sports
  - soccer
  - object-detection
  - camera-calibration
size_categories:
  - 1K<n<10K
extra_gated_heading: Request access to the OpenSportsBall benchmark data
extra_gated_description: >-
  This dataset contains video of real, identifiable adult players in soccer
  games. A maintainer reviews each request by hand, usually within a few days.
extra_gated_prompt: >-
  By requesting access you agree to these terms. Use the data only to develop
  or evaluate sports video processing, such as ball, player and pitch detection,
  camera calibration and tracking. Don't redistribute it or any frames from it,
  and don't publish frames in which a person can be recognised. Don't try to
  identify the people in it, and don't use it for face recognition, biometrics
  or surveillance. Keep your copy access-controlled and delete it when you
  stop contributing or when the maintainers ask. Access can be revoked at any
  time.
extra_gated_fields:
  Full name: text
  Affiliation or GitHub username: text
  What will you use the data for: text
  I agree to use this data only as described above and not to redistribute it: checkbox
extra_gated_button_content: Request access
---

# OpenSportsBall benchmark data

The data behind the benchmarks of
[OpenSportsBall](https://github.com/OpenSportsBall/opensportsball-processing),
an open-source system that films amateur field sports with two phones and
renders a broadcast-style video. It holds the recorded video and the labels
the processing pipeline is scored against, so contributors can rerun every
benchmark and compare their changes against the same numbers.

Access is gated: the footage shows real adult players who can be
identified, so it is shared only with contributors who accept the terms
above.

## Contents

The repository mirrors the `footage/` directory of
`opensportsball-processing`, so a download lands where the tools already look.

| Path | What |
|---|---|
| `bench/rec3/`, `bench/rec4cal/` | The bench store for two night games: the phones' video segments the benches read (unchanged, sha256-checked against each manifest), plus sync, pitch maps, team colours, background models, play-box sets, ball label sets and detector caches |
| `bench/fixtures/pitch/` | Pitch-mapping regression fixtures and the marked-field benchmark |
| `models/` | The standard ball detector and judge, with a README of what each was trained on |
| `rec5-game/`, `rec6/`, `rec7/` | Pitch maps (`pitch/`) and ball label sets (`labels*/`, with their frames) from three more games: an orange ball, the first daylight game, and a night game with a yellow ball. Their full videos are not included |

A ball label set holds `items.json` (the moments and frames picked for
labelling), `labels.json` (the ball positions, or none where it isn't
visible), `images/` (the labelled frames) and `pano/` (the stitched
frames).

## Getting it

After your request is approved, from a checkout of
`opensportsball-processing`:

    hf auth login                                      # a read token is enough
    uv run processing/spikes/sharedata.py fetch        # everything (~37 GB)
    uv run processing/spikes/sharedata.py fetch --annotations-only   # pitch maps and labels only (~1 GB)
    uv run processing/spikes/benchstore.py             # check the bench video

`docs/TESTING-PROCESSING.md` in that repository describes each benchmark and
how to run it.

## Licence

The data is not under an open licence. It is shared only under the terms
you accept when requesting access. The code that uses it is AGPL-3.0.
