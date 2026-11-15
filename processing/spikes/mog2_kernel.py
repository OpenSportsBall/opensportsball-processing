# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""gpu.MOG2's update for one pixel, as C: on an NVIDIA GPU the whole frame
is then one kernel (a thread per pixel), where the PyTorch version is
about a hundred operations a frame and stops four times to count pixels:
13 ms a frame of pass A's 29.

PIXEL is the same arithmetic as gpu.MOG2.apply, step for step and in
single precision, so both give the same mask and leave the same model.
It is plain C: tests/test_gpu_mog2.py builds it for the CPU and holds it
to OpenCV's mask. `launch` compiles it for the GPU with CuPy (the only
thing CuPy is used for); the model's tensors stay PyTorch's.
"""

from __future__ import annotations

K = 5

PIXEL = r"""
#define K 5
#define TB 0.9f
#define TG 9.0f
#define VAR_INIT 15.0f
#define VAR_MIN 4.0f
#define VAR_MAX 75.0f
#define TAU 0.5f

/* One pixel. W, V: K modes (weight, variance); M: K x 3 means; each is
   `stride` apart per mode (per channel for M). x: the pixel's colour.
   a = alpha, a1 = 1 - alpha, pr = -alpha * CT, tb = the variance threshold. */
MOG2_FN unsigned char mog2_pixel(float *W, float *V, float *M, long long stride,
                                 float x0, float x1, float x2,
                                 float a, float a1, float pr, float tb, int learn)
{
    float w_[K], raw[K], after[K], v_[K], m_[K][3];
    float x[3];
    int m, c, j;
    x[0] = x0; x[1] = x1; x[2] = x2;
    for (m = 0; m < K; m++) {
        w_[m] = W[m * stride];
        v_[m] = V[m * stride];
        for (c = 0; c < 3; c++) m_[m][c] = M[(m * 3 + c) * stride];
        raw[m] = a1 * w_[m] + pr;
        after[m] = raw[m] >= -pr ? raw[m] : 0.0f;
    }

    /* The heaviest mode. */
    int used1 = w_[1] > 0.0f;
    float d[3], dist2, w, k;
    for (c = 0; c < 3; c++) d[c] = m_[0][c] - x[c];
    dist2 = (d[0] * d[0] + d[1] * d[1]) + d[2] * d[2];
    int can = w_[0] > 0.0f;
    int background = can && dist2 < tb * v_[0];
    int fits = can && dist2 < TG * v_[0];
    w = raw[0] + a;
    k = fits ? (1.0f / w) * a : 0.0f;  /* as PyTorch divides a constant by a tensor */
    for (c = 0; c < 3; c++) m_[0][c] = m_[0][c] - k * d[c];
    if (fits) {
        float nv = v_[0] + k * (dist2 - v_[0]);
        v_[0] = nv < VAR_MIN ? VAR_MIN : nv > VAR_MAX ? VAR_MAX : nv;
        after[0] = w;
    }

    /* The other modes, where the first didn't match and there are others. */
    int later = !fits && used1;
    if (later) {
        float total = after[0];
        for (m = 1; m < K; m++) {
            int cm = w_[m] > 0.0f && !fits;
            for (c = 0; c < 3; c++) d[c] = m_[m][c] - x[c];
            dist2 = (d[0] * d[0] + d[1] * d[1]) + d[2] * d[2];
            if (cm && total < TB && dist2 < tb * v_[m]) background = 1;
            int match = cm && dist2 < TG * v_[m];
            w = raw[m] + a;
            k = match ? (1.0f / w) * a : 0.0f;
            for (c = 0; c < 3; c++) m_[m][c] = m_[m][c] - k * d[c];
            if (match) {
                float nv = v_[m] + k * (dist2 - v_[m]);
                v_[m] = nv < VAR_MIN ? VAR_MIN : nv > VAR_MAX ? VAR_MAX : nv;
                after[m] = w;
                fits = 1;
            }
            total = total + after[m];
        }
    }
    float sum = after[0];
    for (m = 1; m < K; m++) sum = sum + after[m];
    if (sum < 1e-12f) sum = 1e-12f;
    for (m = 0; m < K; m++) w_[m] = after[m] / sum;

    /* A pixel no mode explains starts a new one, in the first free place
       or over the lightest. */
    int changed = later;
    if (learn && !fits) {
        int n = 0;
        for (m = 0; m < K; m++) n += w_[m] > 0.0f;
        int slot = n < K - 1 ? n : K - 1;
        for (m = 0; m < K; m++) w_[m] = w_[m] * a1;
        w_[slot] = n == 0 ? 1.0f : a;
        v_[slot] = VAR_INIT;
        for (c = 0; c < 3; c++) m_[slot][c] = x[c];
        changed = 1;
    }
    /* Heaviest first again; equal weights keep their order. */
    if (changed) {
        for (m = 1; m < K; m++) {
            float tw = w_[m], tv = v_[m], tm0 = m_[m][0], tm1 = m_[m][1], tm2 = m_[m][2];
            for (j = m; j > 0 && w_[j - 1] < tw; j--) {
                w_[j] = w_[j - 1]; v_[j] = v_[j - 1];
                m_[j][0] = m_[j - 1][0]; m_[j][1] = m_[j - 1][1]; m_[j][2] = m_[j - 1][2];
            }
            w_[j] = tw; v_[j] = tv; m_[j][0] = tm0; m_[j][1] = tm1; m_[j][2] = tm2;
        }
    }
    for (m = 0; m < K; m++) {
        W[m * stride] = w_[m];
        V[m * stride] = v_[m];
        for (c = 0; c < 3; c++) M[(m * 3 + c) * stride] = m_[m][c];
    }
    if (background) return 0;

    /* Not background: foreground, or a shadow (darker than a background
       mode, and the same colour). */
    int shadow = 0, alive = 1;
    float seen = 0.0f;
    for (m = 0; m < K; m++) {
        float num = (x[0] * m_[m][0] + x[1] * m_[m][1]) + x[2] * m_[m][2];
        float den = (m_[m][0] * m_[m][0] + m_[m][1] * m_[m][1]) + m_[m][2] * m_[m][2];
        alive = alive && w_[m] > 0.0f && den > 0.0f;
        float q = num / (den < 1e-12f ? 1e-12f : den);
        float e0 = q * m_[m][0] - x[0], e1 = q * m_[m][1] - x[1], e2 = q * m_[m][2] - x[2];
        int hit = alive && num <= den && num >= TAU * den
                  && (e0 * e0 + e1 * e1) + e2 * e2 < tb * v_[m] * q * q;
        shadow = shadow || hit;
        seen = seen + w_[m];
        alive = alive && !hit && seen <= TB;
    }
    return shadow ? 127 : 255;
}
"""

# The same for a whole frame on the CPU (tests): pixels one after another.
CPU = "#define MOG2_FN static\n" + PIXEL + r"""
void mog2_frame(const unsigned char *bgr, float *W, float *V, float *M, unsigned char *mask,
                long long n, float a, float a1, float pr, float tb, int learn)
{
    long long i;
    for (i = 0; i < n; i++)
        mask[i] = mog2_pixel(W + i, V + i, M + i, n, bgr[3 * i], bgr[3 * i + 1], bgr[3 * i + 2],
                             a, a1, pr, tb, learn);
}
"""

CUDA = "#define MOG2_FN __device__ __forceinline__\n" + PIXEL + r"""
extern "C" __global__
void mog2_frame(const unsigned char *bgr, float *W, float *V, float *M, unsigned char *mask,
                long long n, float a, float a1, float pr, float tb, int learn)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mask[i] = mog2_pixel(W + i, V + i, M + i, n, bgr[3 * i], bgr[3 * i + 1], bgr[3 * i + 2],
                             a, a1, pr, tb, learn);
}
"""

_kernel = None


def launch(bgr, weight, var, mean, mask, n: int, alpha: float, tb: float, ct: float) -> None:
    """One frame on the GPU. The arguments are PyTorch CUDA tensors (bgr
    uint8 n x 3 contiguous; weight, var K x n; mean K x 3 x n; mask uint8
    n), which CuPy reads in place."""
    global _kernel
    import cupy as cp
    import numpy as np

    if _kernel is None:
        # Multiply-adds are not fused: a fused one rounds once where C and
        # PyTorch round twice, and a pixel on a threshold would then differ.
        _kernel = cp.RawKernel(CUDA, "mog2_frame", options=("--fmad=false",))
    f = np.float32
    _kernel(((n + 255) // 256,), (256,),
            (cp.asarray(bgr), cp.asarray(weight), cp.asarray(var), cp.asarray(mean), cp.asarray(mask),
             np.int64(n), f(alpha), f(1.0 - alpha), f(-alpha * ct), f(tb), np.int32(alpha > 0)))
