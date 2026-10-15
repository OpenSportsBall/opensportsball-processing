# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""GPU half of the follow-cam spike: warping and blending in PyTorch.

Runs on Apple's GPU (MPS) locally and on CUDA in the cloud; falls back to
the CPU. Frames stay in NV12, the hardware decoder's native layout (a
luma plane plus a half-resolution interleaved chroma plane), from decode
to encode, so no frame is ever converted to RGB except the small pass A
panorama that background subtraction needs.

A phone mounted upside down is handled in the camera model (pixel
coordinates mirrored in both axes), so its frames are never flipped.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DT = torch.float16  # sampling and blending precision; plenty for 8-bit video


class TorchCamera:
    """`follow.Camera` on the GPU, reading raw (possibly upside-down)
    frames of the given width."""

    def __init__(self, cam, width: int, rot180: bool, dev: torch.device):
        c = cam.at(width)
        self.f, self.cx, self.cy, self.k1, self.k2 = c.f, c.cx, c.cy, c.k1, c.k2
        self.w, self.h, self.r2_max, self.rot180 = c.w, c.h, c.r2_max, rot180
        self.R = torch.tensor(c.R, dtype=torch.float32, device=dev)

    def project(self, d: torch.Tensor):
        """World directions (..., 3) -> raw pixel coordinates and validity."""
        c = d @ self.R
        z = c[..., 2]
        ok = z > 1e-6
        zz = torch.where(ok, z, torch.ones_like(z))
        x, y = c[..., 0] / zz, c[..., 1] / zz
        r2 = x * x + y * y
        ok &= r2 < self.r2_max
        g = 1 + self.k1 * r2 + self.k2 * r2 * r2
        px = self.f * x * g + self.cx
        py = self.f * y * g + self.cy
        ok &= (px >= 0) & (px <= self.w - 1) & (py >= 0) & (py <= self.h - 1)
        if self.rot180:
            px, py = self.w - 1 - px, self.h - 1 - py
        return px, py, ok

    def grid(self, px, py):
        """Pixel coordinates -> grid_sample's normalized coordinates. With
        align_corners=False the same grid addresses the half-size chroma
        plane at the same place in the picture. Directions far outside the
        frame give huge coordinates: clamp them first, or half precision
        turns them into infinities and the samples into NaN (which a zero
        blend weight doesn't remove)."""
        g = torch.stack([(px + 0.5) / self.w * 2 - 1, (py + 0.5) / self.h * 2 - 1], -1)
        return torch.nan_to_num(g, nan=2.0, posinf=2.0, neginf=-2.0).clamp(-2, 2)[None].to(DT)


def upload_nv12(buf: np.ndarray, w: int, h: int, dev: torch.device):
    """One NV12 frame (flat uint8) -> (Y 1x1xHxW, UV 1x2xH/2xW/2) on the GPU."""
    t = torch.from_numpy(buf).to(dev, non_blocking=True)
    y = t[: w * h].view(1, 1, h, w).to(DT)
    uv = t[w * h:].view(h // 2, w // 2, 2).permute(2, 0, 1)[None].to(DT)
    return y, uv


def sample(planes, grid):
    """Sample luma and chroma at the grid: (1xHxW, 2xHxW)."""
    y, uv = planes
    return (F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)[0],
            F.grid_sample(uv, grid, mode="bilinear", padding_mode="border", align_corners=False)[0])


def lowpass(x: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Cheap wide blur: average down by `factor`, smooth, scale back up."""
    h, w = x.shape[-2:]
    s = F.avg_pool2d(x[None], factor)
    s = F.avg_pool2d(s, 5, stride=1, padding=2, count_include_pad=False)
    return F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)[0]


def apply_gain(y, uv, g: float):
    return y * g, (uv - 128) * g + 128


class ViewRenderer:
    """Renders the 1080p virtual camera from both phones' 4K NV12 frames:
    one phone per pixel along the seam, two-band blended."""

    GRID = 4

    def __init__(self, cams, rot180, src_w, out_w, out_h, gains, main_right,
                 seam_rows0, seam_step, feather, tone, dev):
        self.dev = dev
        self.cams = [TorchCamera(c, src_w, r, dev) for c, r in zip(cams, rot180)]
        self.ow, self.oh = out_w, out_h
        self.gains = gains
        self.sign = 1.0 if main_right else -1.0
        self.r0, self.rstep = seam_rows0, seam_step
        self.feather, self.tone = feather, tone
        # The geometry is smooth, so it is computed on a grid GRID times
        # coarser and upsampled. Coarse cell k is centred on output pixel
        # GRID*k + (GRID-1)/2, which is where bilinear upsampling with
        # align_corners=False puts it.
        k = self.GRID
        cols = k * torch.arange(out_w // k, device=dev, dtype=torch.float32) + (k - 1) / 2 - out_w / 2 + 0.5
        rows = k * torch.arange(out_h // k, device=dev, dtype=torch.float32) + (k - 1) / 2 - out_h / 2 + 0.5
        self.v, self.u = torch.meshgrid(rows, cols, indexing="ij")

    def directions(self, yaw, pitch, hfov):
        f = (self.ow / 2) / np.tan(hfov / 2)
        z = torch.tensor([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)],
                         dtype=torch.float32, device=self.dev)
        x = torch.tensor([np.cos(yaw), 0.0, -np.sin(yaw)], dtype=torch.float32, device=self.dev)
        y = torch.linalg.cross(z, x)
        return (self.u / f)[..., None] * x + (self.v / f)[..., None] * y + z

    def render(self, frames, yaw, pitch, hfov, seam: torch.Tensor) -> np.ndarray:
        """`frames`: NV12 planes per phone; `seam`: seam yaw per panorama
        row. Returns one NV12 output frame (flat uint8)."""
        d = self.directions(yaw, pitch, hfov)
        vy = torch.atan2(d[..., 0], d[..., 2])
        vp = torch.atan2(d[..., 1], torch.hypot(d[..., 0], d[..., 2]))
        idx = ((vp - self.r0) / self.rstep).clamp(0, len(seam) - 1.001)
        i0 = idx.floor().long()
        t = idx - i0
        off = self.sign * (vy - (seam[i0] * (1 - t) + seam[i0 + 1] * t))

        size = (self.oh, self.ow)

        def up(x):  # coarse (h, w[, c]) -> full resolution, channels last
            x = x if x.dim() == 3 else x[..., None]
            return F.interpolate(x.permute(2, 0, 1)[None].float(), size=size, mode="bilinear",
                                 align_corners=False)[0].permute(1, 2, 0)

        off = up(off)[..., 0]
        samples, oks = [], []
        for cam, planes, g in zip(self.cams, frames, self.gains):
            px, py, ok = cam.project(d)
            full = up(torch.stack([px, py, ok.float()], -1))
            yy, uv = sample(planes, cam.grid(full[..., 0], full[..., 1]))
            samples.append(apply_gain(yy, uv, g))
            oks.append(full[..., 2] > 0.5)
        both = oks[0] & oks[1]

        def weights(width):
            wm = torch.where(both, (off / width + 0.5).clamp(0, 1), oks[0].float())
            ws = torch.where(oks[1], 1 - wm, torch.zeros_like(wm))
            return wm.to(DT), ws.to(DT)

        (sm, ss), (wm, ws) = weights(self.feather), weights(self.tone)
        ssum = (sm + ss).clamp_min(1e-3)
        wsum = (wm + ws).clamp_min(1e-3)
        seen = oks[0] | oks[1]
        out = []
        for k, fill in ((0, 16.0), (1, 128.0)):  # luma, then chroma
            a, b = samples[0][k], samples[1][k]
            la, lb = lowpass(a), lowpass(b)
            o = ((a - la) * sm + (b - lb) * ss) / ssum + (la * wm + lb * ws) / wsum
            out.append(torch.where(seen, o, torch.full_like(o, fill)))
        y = out[0][0]
        uv = F.avg_pool2d(out[1][None], 2)[0].permute(1, 2, 0)  # H/2 x W/2 x 2, interleaved
        frame = torch.cat([y.reshape(-1), uv.reshape(-1)]).round().clamp(0, 255).to(torch.uint8)
        return frame.cpu().numpy()


class PanoWarper:
    """Pass A: both phones' low-res NV12 frames onto the fixed
    equirectangular panorama, for background subtraction and the seam."""

    def __init__(self, maps, weights, rot180, src_w, src_h, gains, c0, c1, dev):
        self.dev = dev
        self.gains = gains
        self.c0, self.c1 = c0, c1
        self.grids, self.w = [], []
        for (x, y), wt, r in zip(maps, weights, rot180):
            if r:  # maps are for upright frames; read the raw one mirrored
                x, y = src_w - 1 - x, src_h - 1 - y
            g = np.stack([(x + 0.5) / src_w * 2 - 1, (y + 0.5) / src_h * 2 - 1], -1)
            self.grids.append(torch.tensor(g, device=dev)[None].to(DT))
            self.w.append(torch.tensor(wt, device=dev).to(DT))
        self.wsum = (self.w[0] + self.w[1]).clamp_min(1e-3)

    def warp(self, frames):
        """Returns the blended panorama as BGR uint8 (for OpenCV) and the
        seam cost over the overlap columns (float32, CPU)."""
        s = [apply_gain(*sample(p, g), gn) for p, g, gn in zip(frames, self.grids, self.gains)]
        y = (s[0][0] * self.w[0] + s[1][0] * self.w[1]) / self.wsum
        uv = (s[0][1] * self.w[0] + s[1][1] * self.w[1]) / self.wsum
        bgr = nv_to_bgr(y[0].float(), uv.float())
        c0, c1 = self.c0, self.c1
        diff = ((s[0][0][:, :, c0:c1] - s[1][0][:, :, c0:c1]).abs().sum(0)
                + (s[0][1][:, :, c0:c1] - s[1][1][:, :, c0:c1]).abs().sum(0)) / (3 * 255)
        diff = F.avg_pool2d(diff[None, None].float(), 3, stride=1, padding=1)[0, 0]
        return bgr.cpu().numpy(), diff.cpu().numpy()


def nv_to_bgr(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Limited-range BT.709 YCbCr -> BGR uint8 (H x W x 3)."""
    yy = (y - 16) * (255 / 219)
    cb = (uv[0] - 128) * (255 / 224)
    cr = (uv[1] - 128) * (255 / 224)
    r = yy + 1.5748 * cr
    g = yy - 0.1873 * cb - 0.4681 * cr
    b = yy + 1.8556 * cb
    return torch.stack([b, g, r], -1).round().clamp(0, 255).to(torch.uint8)
