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
    """Apple's GPU, NVIDIA's or the CPU: accel.py decides."""
    import accel

    return accel.device()


DT = torch.float16  # sampling and blending precision; plenty for 8-bit video


class TorchCamera:
    """`follow.Camera` on the GPU, reading raw (possibly upside-down)
    frames of the given width."""

    def __init__(self, cam, width: int, rot180: bool, dev: torch.device):
        c = cam.at(width)
        self.f, self.cx, self.cy, self.k1, self.k2 = c.f, c.cx, c.cy, c.k1, c.k2
        self.w, self.h, self.r2_max, self.rot180 = c.w, c.h, c.r2_max, rot180
        self.R = torch.tensor(c.R, dtype=torch.float32, device=dev)
        self.t = torch.tensor(c.t, dtype=torch.float32, device=dev) if c.t.any() else None
        self.far_h = c.far_h

    def project(self, d: torch.Tensor):
        """World directions (..., 3) from Main's lens -> raw pixel
        coordinates and validity (follow.Camera.project, lens offset included)."""
        if self.t is not None:
            u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            dy = u[..., 1:2]
            dist = torch.where(dy > 1 / self.far_h, 1 / dy.clamp_min(1e-12), torch.full_like(dy, self.far_h))
            d = u * dist - self.t
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


class GpuNV12:
    """One NV12 frame that never left the GPU (nvdec.py decodes into these):
    luma `y` (H x W) and chroma `uv` (H/2 x W/2 x 2), uint8 on the GPU.
    Stands in for the flat uint8 array the ffmpeg pipe gives: upload_nv12
    takes either, and np.asarray(frame) copies it to the CPU for code that
    still works there."""

    __slots__ = ("y", "uv")

    def __init__(self, y: torch.Tensor, uv: torch.Tensor):
        self.y, self.uv = y, uv

    def __array__(self, dtype=None, copy=None):
        a = torch.cat([self.y.reshape(-1), self.uv.reshape(-1)]).cpu().numpy()
        return a if dtype is None else a.astype(dtype, copy=False)


def upload_nv12(buf, w: int, h: int, dev: torch.device):
    """One NV12 frame (flat uint8, or a GpuNV12 already there) ->
    (Y 1x1xHxW, UV 1x2xH/2xW/2) on the GPU."""
    if isinstance(buf, GpuNV12):
        return buf.y.view(1, 1, h, w).to(DT), buf.uv.permute(2, 0, 1)[None].to(DT)
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

    def warp(self, frames, main_only: bool = False):
        """Returns the blended panorama as BGR uint8 (for OpenCV) and the
        seam cost over the overlap columns (float32, CPU). `main_only`:
        Secondary has no footage for this frame, so the panorama is what
        Main sees and black elsewhere, and there is no seam to cost."""
        bgr, diff = self.warp_gpu(frames, main_only)
        return bgr.cpu().numpy(), diff.cpu().numpy()

    def warp_gpu(self, frames, main_only: bool = False):
        """`warp`, with both results left on the GPU (BGR uint8 H x W x 3,
        cost float32)."""
        if main_only:
            yy, uv = apply_gain(*sample(frames[0], self.grids[0]), self.gains[0])
            seen = self.w[0] > 0
            bgr = nv_to_bgr(torch.where(seen, yy[0], torch.full_like(yy[0], 16)).float(),
                            torch.where(seen, uv, torch.full_like(uv, 128)).float())
            return bgr, torch.zeros((seen.shape[0], self.c1 - self.c0), dtype=torch.float32, device=self.dev)
        s = [apply_gain(*sample(p, g), gn) for p, g, gn in zip(frames, self.grids, self.gains)]
        y = (s[0][0] * self.w[0] + s[1][0] * self.w[1]) / self.wsum
        uv = (s[0][1] * self.w[0] + s[1][1] * self.w[1]) / self.wsum
        bgr = nv_to_bgr(y[0].float(), uv.float())
        c0, c1 = self.c0, self.c1
        diff = ((s[0][0][:, :, c0:c1] - s[1][0][:, :, c0:c1]).abs().sum(0)
                + (s[0][1][:, :, c0:c1] - s[1][1][:, :, c0:c1]).abs().sum(0)) / (3 * 255)
        diff = F.avg_pool2d(diff[None, None].float(), 3, stride=1, padding=1)[0, 0]
        return bgr, diff


def nv_to_bgr(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Limited-range BT.709 YCbCr -> BGR uint8 (H x W x 3)."""
    yy = (y - 16) * (255 / 219)
    cb = (uv[0] - 128) * (255 / 224)
    cr = (uv[1] - 128) * (255 / 224)
    r = yy + 1.5748 * cr
    g = yy - 0.1873 * cb - 0.4681 * cr
    b = yy + 1.8556 * cb
    return torch.stack([b, g, r], -1).round().clamp(0, 255).to(torch.uint8)


def dilate(x: torch.Tensor, k: int, iterations: int = 1) -> torch.Tensor:
    """cv2.dilate with a k x k square on a 0/1 mask (H x W, float)."""
    for _ in range(iterations):
        x = F.max_pool2d(x[None, None], k, stride=1, padding=k // 2)[0, 0]
    return x


def erode(x: torch.Tensor, k: int) -> torch.Tensor:
    """cv2.erode with a k x k square on a 0/1 mask: outside the picture
    counts as set, as OpenCV's default border does."""
    return -F.max_pool2d(-x[None, None], k, stride=1, padding=k // 2)[0, 0]


class MOG2:
    """OpenCV's BackgroundSubtractorMOG2 (Zivkovic's adaptive mixture of
    Gaussians; bgfg_gaussmix2.cpp) on the GPU, so pass A's panorama need
    not come back to the CPU for it: 29 ms of every frame on a cloud
    instance's four CPUs, the GPU idle meanwhile. The same model, update and
    shadow rule, with OpenCV's defaults; `apply` returns the same mask (0
    background, 127 shadow, 255 foreground), as a tensor.

    Each pixel keeps up to K modes (weight, mean colour, one variance),
    heaviest first. OpenCV walks each pixel's modes in a loop. Here the
    heaviest mode is done for every pixel at once, which settles nearly all
    of them (the pitch is the pitch), and the rest of the work (later
    modes, new modes, reordering, shadows) only for the pixels that need
    it: doing all of it everywhere is half a second a frame."""

    K = 5
    BACKGROUND_RATIO = 0.9   # TB: the heaviest modes up to this weight are the background
    VAR_THRESHOLD_GEN = 9.0  # Tg: a pixel within this of a mode belongs to it
    VAR_INIT, VAR_MIN, VAR_MAX = 15.0, 4.0, 75.0
    CT = 0.05                # complexity reduction: a mode must keep earning its weight
    TAU = 0.5                # a shadow is no darker than this times the background

    def __init__(self, h: int, w: int, dev: torch.device, history: int = 300, var_threshold: float = 24.0):
        self.h, self.w, self.history, self.tb = h, w, history, var_threshold
        self.frames = 0
        # Pixels flattened, so a subset of them is one index.
        self.weight = torch.zeros((self.K, h * w), dtype=torch.float32, device=dev)
        self.var = torch.zeros((self.K, h * w), dtype=torch.float32, device=dev)
        self.mean = torch.zeros((self.K, 3, h * w), dtype=torch.float32, device=dev)

    @torch.no_grad()
    def apply(self, bgr: torch.Tensor, learning_rate: float = -1) -> torch.Tensor:
        """`bgr`: H x W x 3 uint8. Returns the mask, H x W uint8."""
        K, TB, TG = self.K, self.BACKGROUND_RATIO, self.VAR_THRESHOLD_GEN
        x = bgr.reshape(-1, 3).t().float()  # 3 x P
        self.frames += 1
        alpha = learning_rate if learning_rate >= 0 and self.frames > 1 else 1.0 / min(2 * self.frames, self.history)
        alpha1, prune = 1.0 - alpha, -alpha * self.CT
        W, V, M = self.weight, self.var, self.mean
        # Every mode's weight decays; one that has decayed away is dropped
        # (and an unused one, weight 0, stays so). The mode a pixel matches
        # gains alpha instead, below.
        raw = alpha1 * W + prune
        after = torch.where(raw >= -prune, raw, torch.zeros_like(raw))

        # The heaviest mode, everywhere.
        used1 = W[1] > 0
        d = M[0] - x
        dist2 = (d * d).sum(0)
        can = W[0] > 0
        background = can & (dist2 < self.tb * V[0])
        fits = can & (dist2 < TG * V[0])
        w = raw[0] + alpha
        k = torch.where(fits, alpha / w, torch.zeros_like(w))
        M[0] -= k * d
        V[0] = torch.where(fits, (V[0] + k * (dist2 - V[0])).clamp(self.VAR_MIN, self.VAR_MAX), V[0])
        after[0] = torch.where(fits, w, after[0])

        # The other modes, where the first didn't match and there are others.
        later = (~fits & used1).nonzero()[:, 0]
        if later.numel():
            xs = x[:, later]
            fit_s = torch.zeros_like(later, dtype=torch.bool)
            bg_s = background[later]
            total = after[0, later]
            for m in range(1, K):
                can = (W[m, later] > 0) & ~fit_s
                mean, var = M[m][:, later], V[m, later]
                d = mean - xs
                dist2 = (d * d).sum(0)
                bg_s |= can & (total < TB) & (dist2 < self.tb * var)
                match = can & (dist2 < TG * var)
                w = raw[m, later] + alpha
                k = torch.where(match, alpha / w, torch.zeros_like(w))
                M[m][:, later] = mean - k * d
                V[m, later] = torch.where(match, (var + k * (dist2 - var)).clamp(self.VAR_MIN, self.VAR_MAX), var)
                wm = torch.where(match, w, after[m, later])
                after[m, later] = wm
                fit_s |= match
                total = total + wm
            fits[later] = fit_s
            background[later] = bg_s
        W.copy_(after / after.sum(0).clamp_min(1e-12))

        # A pixel no mode explains starts a new one, in the first free place
        # or over the lightest.
        changed = torch.zeros_like(fits)
        changed[later] = True
        if alpha > 0:
            new = (~fits).nonzero()[:, 0]
            if new.numel():
                wn = W[:, new]
                n = (wn > 0).sum(0)
                slot = n.clamp_max(K - 1)[None]
                wn = (wn * alpha1).scatter_(0, slot, torch.where(n == 0, 1.0, alpha)[None].to(wn.dtype))
                W[:, new] = wn
                V[:, new] = V[:, new].scatter_(0, slot, torch.full_like(slot, self.VAR_INIT, dtype=V.dtype))
                M[:, :, new] = M[:, :, new].scatter_(0, slot[:, None].expand(1, 3, -1), x[None, :, new])
                changed[new] = True
        # Heaviest first again, as OpenCV keeps them (it moves a mode up as
        # its weight grows): only where a later mode matched or a new one came.
        moved = changed.nonzero()[:, 0]
        if moved.numel():
            ws = W[:, moved]
            order = torch.sort(ws, dim=0, descending=True, stable=True).indices
            W[:, moved] = torch.gather(ws, 0, order)
            V[:, moved] = torch.gather(V[:, moved], 0, order)
            M[:, :, moved] = torch.gather(M[:, :, moved], 0, order[:, None].expand(-1, 3, -1))

        # What isn't background is foreground, or a shadow: darker than a
        # background mode, and the same colour.
        mask = torch.zeros(self.h * self.w, dtype=torch.uint8, device=x.device)
        fg = (~background).nonzero()[:, 0]
        if fg.numel():
            xs = x[:, fg]
            shadow = torch.zeros_like(fg, dtype=torch.bool)
            alive = torch.ones_like(shadow)
            seen = torch.zeros_like(fg, dtype=torch.float32)
            for m in range(K):
                mean, wm = M[m][:, fg], W[m, fg]
                num = (xs * mean).sum(0)
                den = (mean * mean).sum(0)
                alive &= (wm > 0) & (den > 0)
                a = num / den.clamp_min(1e-12)
                e = a * mean - xs
                hit = alive & (num <= den) & (num >= self.TAU * den) & ((e * e).sum(0) < self.tb * V[m, fg] * a * a)
                shadow |= hit
                seen += wm
                alive &= ~hit & (seen <= TB)
            mask[fg] = torch.where(shadow, 127, 255).to(torch.uint8)
        return mask.view(self.h, self.w)

    def background_image(self) -> np.ndarray:
        """cv2's getBackgroundImage: each pixel's background modes, averaged by weight (BGR uint8)."""
        acc = torch.zeros_like(self.mean[0])
        total = torch.zeros_like(self.weight[0])
        going = torch.ones_like(total, dtype=torch.bool)
        for m in range(self.K):
            w = torch.where(going, self.weight[m], torch.zeros_like(total))
            acc += w * self.mean[m]
            total += w
            going &= total <= self.BACKGROUND_RATIO
        img = acc / total.clamp_min(1e-12)
        return img.t().reshape(self.h, self.w, 3).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
