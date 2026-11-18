#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""A full-field panorama clip: everything either phone sees, as one video.

    uv run processing/spikes/pano.py footage/rec8 0 600 60          # segment 0, from 600 s, 60 s
    uv run processing/spikes/pano.py footage/rec8 0 600 60 --youtube  # and a copy YouTube plays as 360 video

follow.py with the virtual camera swapped for a fixed equirectangular
view: its cameras, exposure gain and pass A's per-frame seam are used as
they are, the planned path is not (so no ball, roles or pitch files are
needed), and pass B samples both phones' full-resolution frames. Like
follow.py it rewrites <recording>/follow/background.jpg and
detections.jpg on the way.

Writes <recording>/pano/pano_<segment>_<start>_<seconds>.mp4 (H.264, no
audio) and a .json beside it with the view's extent in degrees. The frame
is the box around both phones' views, black where neither sees.

PANO_PX_PER_DEG (default 7680 / 360, an 8K sphere's density: the phones
have ~26 px/deg at the centre of the frame) and PANO_BITRATE (default
60M) change the render.

--youtube also writes ..._360.mp4: the panorama at its own angles on a
whole 360 x 180 degree equirectangular frame, HEVC, tagged with Google's
Spherical Video V1 metadata (an XML `uuid` box in the video track), which
is what makes YouTube play it as 360 video. What that looked like is in
docs/FINDINGS.md, section 7.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import accel
import follow
import gpu
import recpaths

PX_PER_DEG = float(os.environ.get("PANO_PX_PER_DEG", 7680 / 360))
BITRATE = os.environ.get("PANO_BITRATE", "60M")
SPHERE_BITRATE = "100M"  # asked of the hardware HEVC encoder; a mostly black frame takes less
SPHERICAL_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")
SPHERICAL_XML = (b'<?xml version="1.0"?><rdf:SphericalVideo\n'
                 b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
                 b'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">'
                 b'<GSpherical:Spherical>true</GSpherical:Spherical>'
                 b'<GSpherical:Stitched>true</GSpherical:Stitched>'
                 b'<GSpherical:StitchingSoftware>OpenSportsBall</GSpherical:StitchingSoftware>'
                 b'<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>'
                 b'</rdf:SphericalVideo>')

view: dict = {}  # the panorama's extent and size, set once pass A knows what the phones see


def fixed_view(rec, out, seg, start, dets, extent, aim, features, alone, main_yaw, cm):
    """In place of follow.plan_path: sizes the render to the whole extent.
    The path it returns is not used (PanoRenderer ignores it)."""
    y0, y1, p0, p1 = (float(np.degrees(v)) for v in extent)
    w = int((y1 - y0) * PX_PER_DEG) // 8 * 8
    h = int((p1 - p0) * PX_PER_DEG) // 8 * 8
    # The view is centred on the extent; the few pixels lost to rounding come off both sides.
    yc, pc = (y0 + y1) / 2, (p0 + p1) / 2
    view.update(pxPerDeg=PX_PER_DEG, width=w, height=h,
                yawDeg=[yc - w / 2 / PX_PER_DEG, yc + w / 2 / PX_PER_DEG],
                pitchDeg=[pc - h / 2 / PX_PER_DEG, pc + h / 2 / PX_PER_DEG])
    follow.OUT_W, follow.OUT_H = w, h
    follow.PIECE_RATE = {h: (BITRATE, None)}
    os.environ.pop("OSB_RENDER_HEIGHT", None)
    print(f"panorama: yaw {y0:.1f}..{y1:.1f}, pitch {p0:.1f}..{p1:.1f} deg, {w} x {h} at {PX_PER_DEG:.2f} px/deg",
          flush=True)
    z = np.zeros(len(dets))
    return z, z, z


class PanoRenderer(gpu.ViewRenderer):
    """ViewRenderer whose output pixels are equal steps of yaw and pitch."""

    def directions(self, yaw, pitch, hfov):
        step = np.radians(1 / view["pxPerDeg"])
        ya = np.radians(sum(view["yawDeg"]) / 2) + self.u * step
        pa = np.radians(sum(view["pitchDeg"]) / 2) + self.v * step
        return torch.stack([torch.sin(ya) * torch.cos(pa), torch.sin(pa), torch.cos(ya) * torch.cos(pa)], -1)


def tag_spherical(src: Path, dst: Path) -> None:
    """Copies an MP4 with the spherical `uuid` box added to its video
    track. The moov box must come after mdat (ffmpeg's default), so no
    chunk offset moves."""
    with open(src, "rb") as f:
        total, at, moov_at = f.seek(0, 2), 0, None
        while at < total:
            f.seek(at)
            size, kind = struct.unpack(">I4s", f.read(8))
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            if size < 8:
                raise ValueError(f"{src}: a box of size {size} at {at}")
            if kind == b"moov":
                moov_at, moov_size = at, size
            at += size
        if moov_at is None or moov_at + moov_size != total:
            raise ValueError(f"{src}: the moov box must be the last one")
        f.seek(moov_at)
        moov = bytearray(f.read(moov_size))
    at = 8
    while at < len(moov):
        size, kind = struct.unpack(">I4s", moov[at:at + 8])
        if size < 8:
            raise ValueError(f"{src}: a box of size {size} inside moov")
        if kind == b"trak" and b"vide" in moov[at:at + size]:
            box = struct.pack(">I4s", 8 + 16 + len(SPHERICAL_XML), b"uuid") + SPHERICAL_UUID + SPHERICAL_XML
            moov[at + size:at + size] = box
            moov[at:at + 4] = struct.pack(">I", size + len(box))
            moov[0:4] = struct.pack(">I", len(moov))
            break
        at += size
    else:
        raise ValueError(f"{src}: no video track")
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(moov_at)
        f.write(moov)


def sphere(clip: Path, meta: dict) -> Path:
    """The clip on a whole equirectangular frame at the same density, each
    pixel at its own yaw and pitch, tagged as 360 video."""
    ppd = meta["pxPerDeg"]
    w = round(360 * ppd / 2) * 2
    h = w // 2
    x = round((meta["yawDeg"][0] + 180) * ppd / 2) * 2
    y = round((meta["pitchDeg"][0] + 90) * ppd / 2) * 2
    if x < 0 or y < 0 or x + meta["width"] > w or y + meta["height"] > h:
        sys.exit("the panorama does not fit a 360 x 180 degree frame at these angles")
    enc = (["-c:v", "hevc_videotoolbox"] if accel.platform() == "mac" else
           ["-c:v", "hevc_nvenc", "-preset", "p5"] if accel.platform() == "nvidia" else
           ["-c:v", "libx265", "-preset", "fast"])
    padded, out = clip.with_name(clip.stem + "_padded.mp4"), clip.with_name(clip.stem + "_360.mp4")
    subprocess.run([accel.ffmpeg(), "-v", "error", "-y", "-i", str(clip),
                    "-vf", f"pad={w}:{h}:{x}:{y}:black", *enc, "-b:v", SPHERE_BITRATE, "-tag:v", "hvc1",
                    "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", str(padded)], check=True)
    tag_spherical(padded, out)
    padded.unlink()
    return out


def main(rec: Path, seg: int, start: float, seconds: float, youtube: bool) -> None:
    follow.plan_path = fixed_view
    gpu.ViewRenderer = PanoRenderer
    follow.main(rec, seg, start, seconds, "pano")
    out = recpaths.out(rec) / "pano"
    out.mkdir(exist_ok=True)
    clip = out / f"pano_{seg}_{start:g}_{seconds:g}.mp4"
    shutil.move(recpaths.out(rec) / "follow" / "follow_pano.mp4", clip)
    meta = {"segment": seg, "startS": start, "seconds": seconds, **view}
    clip.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {clip}")
    if youtube:
        print(f"wrote {sphere(clip, meta)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--youtube"]
    if len(args) != 4:
        sys.exit(__doc__)
    main(Path(args[0]), int(args[1]), float(args[2]), float(args[3]), "--youtube" in sys.argv)
