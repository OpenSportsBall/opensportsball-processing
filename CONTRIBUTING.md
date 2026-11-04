# Contributing

Thanks for helping. Before you open a pull request:

1. **Sign the
   [contributor license agreement](https://github.com/OpenSportsBall/opensportsball/blob/main/CLA.md)
   (CLA).** CLA Assistant asks you to sign when you open your first pull
   request, and pull requests can't be merged until you do. One signature
   covers every repository in the OpenSportsBall organization. You keep
   the copyright to your work; the CLA lets the project offer this
   pipeline under a commercial license alongside the AGPL. (TODO before
   the repo goes public: connect CLA Assistant.)
2. **Add the SPDX header** to every new source file:

   ```
   # SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
   # SPDX-License-Identifier: AGPL-3.0-only
   ```

   Run `pipx run reuse lint` to check.
3. **Keep an Apache-2.0 detector working.** Detectors sit behind
   `processing/src/osb_processing/detectors/base.py`. Ultralytics YOLO is
   AGPL-3.0, which is fine here, but while it is a dependency the project
   can't offer a commercial license without also buying an Ultralytics
   enterprise license. Keeping RF-DETR (Apache-2.0) working preserves that
   option.
4. **Don't edit `schemas/` by hand.** It is a copy of the main repo's
   manifest schema; see [schemas/README.md](schemas/README.md).

## Footage and privacy

Game footage shows minors. Never commit video, frames, or labeled datasets
to this repository; `.gitignore` blocks common video formats. Share test
footage privately and only with the consent of the families involved.
