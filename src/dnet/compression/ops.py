from __future__ import annotations

import math
from typing import Optional, Dict

import mlx.core as mx

from .kernels import (
    k_col_norm2_unroll,
    k_sumq_sumq2_int8,
    k_apply_mask_mul,
    k_apply_mask_select,
    k_col_norm2_int8_affine,
    k_dequant_mask_affine,
)
from .utils import _u32, infer_groups_and_gsize, reshape_rg


def _compute_norms_fp32(x32_2d: mx.array) -> mx.array:
    R, D = x32_2d.shape
    tgx = 512 if D >= (1 << 15) else 256
    norms = k_col_norm2_unroll(
        inputs=[x32_2d, _u32([R, D])],
        grid=(D, 1, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(D,)],
        output_dtypes=[mx.float32],
    )[0]
    return norms


def _compute_norms_quant_int8(
    q_2d: mx.array, scales: mx.array, biases: Optional[mx.array], group_size: int
) -> mx.array:
    R, D = q_2d.shape
    G, gsize = infer_groups_and_gsize(D, group_size, scales, biases)

    if (getattr(scales, "ndim", 1) == 1 and scales.shape[0] == G) and (
        biases is None or (getattr(biases, "ndim", 1) == 1 and biases.shape[0] == G)
    ):
        tgx = 512 if D >= (1 << 15) else 256
        sumq, sumq2 = k_sumq_sumq2_int8(
            inputs=[q_2d, mx.array([R, D], dtype=mx.uint32)],
            template=[("T", q_2d.dtype)],
            grid=(D, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(D,), (D,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        sg = mx.repeat(scales.astype(mx.float32), gsize)[:D]
        bg = mx.repeat(
            (
                biases.astype(mx.float32)
                if biases is not None
                else mx.zeros(G, mx.float32)
            ),
            gsize,
        )[:D]
        return sg * sg * sumq2 + 2.0 * sg * bg * sumq + float(R) * bg * bg

    s_rg = reshape_rg(scales, R, G)
    if biases is None:
        b_rg, has_b = mx.zeros((R, G), dtype=mx.float32), 0
    else:
        b_rg, has_b = reshape_rg(biases, R, G), 1

    tgx = 512 if D >= (1 << 15) else 256
    norms = k_col_norm2_int8_affine(
        inputs=[
            q_2d,
            s_rg.reshape(-1),
            b_rg.reshape(-1),
            mx.array([R, D, G, gsize, has_b], dtype=mx.uint32),
        ],
        template=[("T", q_2d.dtype)],
        grid=(D, 1, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(D,)],
        output_dtypes=[mx.float32],
    )[0]
    return norms


def _select_k_smallest_indices(norms: mx.array, k: int) -> mx.array:
    return mx.argpartition(norms, k - 1)[:k]


def _apply_mask(x2d: mx.array, mask01: mx.array, use_select: bool = False) -> mx.array:
    R, D = x2d.shape
    kernel = k_apply_mask_select if use_select else k_apply_mask_mul
    n = R * D
    tgx = 512 if n >= (1 << 20) else 256
    y = kernel(
        inputs=[x2d, mask01.astype(mx.float32), _u32([R, D])],
        template=[("T", x2d.dtype)],
        grid=(n, 1, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(R, D)],
        output_dtypes=[x2d.dtype],
    )[0]
    return y


def column_sparsify_tensor(
    tensor: mx.array,
    compression_percentage: float,
    *,
    quant: Optional[Dict[str, mx.array]] = None,
) -> mx.array:
    """Zero smallest-norm columns along last dim.

    quant dict (optional):
      {"scales": (G,), "biases": (G|None), "group_size": int, "bits": 4|8, "mode": "affine"|"symmetric"}
    If provided, tensor is int8/uint8 codes; norms are computed in quantized domain.
    """
    if not isinstance(tensor, mx.array):
        raise TypeError("Input must be an MLX array.")
    if not 0.0 <= compression_percentage <= 100.0:
        raise ValueError("compression_percentage must be 0..100.")
    if tensor.size == 0 or tensor.shape[-1] == 0 or compression_percentage == 0.0:
        return tensor
    if compression_percentage == 100.0:
        return mx.zeros_like(tensor)

    D = tensor.shape[-1]
    R = tensor.size // D
    k = min(math.ceil((compression_percentage / 100.0) * D), D)
    if k == 0:
        return tensor

    if quant is None:
        x32 = tensor.astype(mx.float32).reshape(R, D)
        norms = _compute_norms_fp32(x32)
    else:
        q = tensor.reshape(R, D)
        scales = quant["scales"]
        biases = quant.get("biases", None)
        gsz_in = int(quant.get("group_size", 64))
        norms = _compute_norms_quant_int8(q, scales, biases, gsz_in)

    idx = _select_k_smallest_indices(norms, k)

    mask = mx.ones((D,), dtype=mx.float32)
    mask[idx] = 0.0

    if quant is None:
        x2d = tensor.reshape(R, D)
        use_select = k >= D // 2
        y = _apply_mask(x2d, mask, use_select=use_select).reshape(tensor.shape)
        return y.astype(tensor.dtype)

    bits = int(quant.get("bits", 8))
    G, gsz_eff = infer_groups_and_gsize(
        D, int(quant.get("group_size", 64)), scales, biases
    )
    if bits == 8:
        s_rg = reshape_rg(scales, R, G)
        if biases is None:
            b_rg, has_b = mx.zeros((R, G), dtype=mx.float32), 0
        else:
            b_rg, has_b = reshape_rg(biases, R, G), 1
        n = R * D
        tgx = 512 if n >= (1 << 20) else 256
        y = k_dequant_mask_affine(
            inputs=[
                tensor.reshape(R, D),
                s_rg.reshape(-1),
                b_rg.reshape(-1),
                mask.astype(mx.float32),
                _u32([R, D, G, gsz_eff, has_b]),
            ],
            template=[("T", tensor.dtype)],
            grid=(n, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(R, D)],
            output_dtypes=[mx.float16],
        )[0]
        return y.reshape(tensor.shape)
    else:
        deq = mx.dequantize(
            tensor.reshape(R, D),
            quant["scales"],
            quant.get("biases", None),
            group_size=int(quant.get("group_size", 64)),
            bits=bits,
            mode=str(quant.get("mode", "affine")),
        ).astype(mx.float32)
        use_select = k >= D // 2
        y = _apply_mask(deq, mask, use_select=use_select).reshape(tensor.shape)
        return y
