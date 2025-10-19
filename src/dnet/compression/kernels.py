"""
Metal GPU kernels used by compression operations.
"""

from __future__ import annotations

import mlx.core as mx


_COL_NORM2_UNROLL_SRC = """
    uint c = thread_position_in_grid.x;
    uint rows = params[0];
    uint cols = params[1];
    if (c >= cols) return;
    float acc = 0.0f;
    uint r = 0u;
    for (; r + 3u < rows; r += 4u) {
        float v0 = float(inp[(r+0u) * cols + c]);
        float v1 = float(inp[(r+1u) * cols + c]);
        float v2 = float(inp[(r+2u) * cols + c]);
        float v3 = float(inp[(r+3u) * cols + c]);
        acc += v0*v0 + v1*v1 + v2*v2 + v3*v3;
    }
    for (; r < rows; ++r) {
        float v = float(inp[r * cols + c]);
        acc += v * v;
    }
    out[c] = acc;
"""

_SUMQ_SUMQ2_INT8_SRC = """
    uint c = thread_position_in_grid.x;
    uint rows = params[0];
    uint cols = params[1];
    if (c >= cols) return;
    float s1 = 0.0f, s2 = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        float v = float((T)q[r * cols + c]);
        s1 += v; s2 += v*v;
    }
    sumq[c]  = s1;
    sumq2[c] = s2;
"""

_APPLY_MASK_MUL_SRC = """
    uint gid = thread_position_in_grid.x;
    uint rows = params[0];
    uint cols = params[1];
    uint n = rows * cols;
    if (gid >= n) return;
    uint c = gid % cols;
    float m = mask[c];
    out[gid] = (T)(float(inp[gid]) * m);
"""

_APPLY_MASK_SELECT_SRC = """
    uint gid = thread_position_in_grid.x;
    uint rows = params[0];
    uint cols = params[1];
    uint n = rows * cols;
    if (gid >= n) return;
    uint c = gid % cols;
    float keep = mask[c];
    float v = float(inp[gid]);
    out[gid] = (T)((keep > 0.5f) ? v : 0.0f);
"""

_COL_NORM2_INT8_AFFINE_SRC = """
    uint c = thread_position_in_grid.x;
    uint rows   = params[0];
    uint cols   = params[1];
    uint groups = params[2];
    uint gsize  = params[3];
    uint has_b  = params[4];
    if (c >= cols) return;
    uint g = c / gsize;
    float acc = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        float s = scales[r * groups + g];
        float b = has_b != 0 ? biases[r * groups + g] : 0.0f;
        float qv = float((T)q[r * cols + c]);
        float t  = s * qv + b;
        acc += t * t;
    }
    out[c] = acc;
"""

_DEQ_MASK_AFFINE_INT8_SRC = """
    uint gid    = thread_position_in_grid.x;
    uint rows   = params[0];
    uint cols   = params[1];
    uint groups = params[2];
    uint gsize  = params[3];
    uint has_b  = params[4];
    uint n      = rows * cols;
    if (gid >= n) return;
    uint r = gid / cols;
    uint c = gid % cols;
    uint g = c / gsize;
    float s = scales[r * groups + g];
    float b = (has_b != 0u) ? biases[r * groups + g] : 0.0f;
    float qv = float((T)q[gid]);
    float v  = s * qv + b;
    float keep = mask[c];
    out[gid] = (keep > 0.5f) ? v : 0.0f;
"""

_GATHER_COLS_SRC = """
    uint gid = thread_position_in_grid.x;
    uint R = params[0];
    uint D = params[1];
    uint K = params[2];
    if (gid >= R * K) return;
    uint r = gid / K;
    uint j = gid % K;
    uint c = indices[j];
    out[gid] = (T)(inp[r * D + c]);
"""

_SCATTER_FROM_COMPACT_SRC = """
    uint gid = thread_position_in_grid.x;
    uint R = params[0];
    uint D = params[1];
    uint K = params[2];
    if (gid >= R * D) return;
    uint r = gid / D;
    uint c = gid % D;
    int j = int(pos[c]);
    float v = 0.0f;
    if (j >= 0) {
        v = float(vals[r * K + uint(j)]);
    }
    out[gid] = (T)(v);
"""

_DEQ_SCATTER_Q8_SRC = """
    uint gid = thread_position_in_grid.x;
    uint R = params[0];
    uint D = params[1];
    uint K = params[2];
    uint G = params[3];
    uint Gk = params[4];
    uint gsz = params[5];
    uint has_b = params[6];
    if (gid >= R * D) return;
    uint r = gid / D;
    uint c = gid % D;
    int j = int(pos[c]);
    float outv = 0.0f;
    if (j >= 0) {
        uint gorig = c / gsz;
        int gk = inv_map[gorig];
        if (gk >= 0 && uint(gk) < Gk) {
            float s = float(s_kept[r * Gk + uint(gk)]);
            float b = (has_b != 0u) ? float(b_kept[r * Gk + uint(gk)]) : 0.0f;
            float qv = float((T)codes[r * K + uint(j)]);
            outv = s * qv + b;
        }
    }
    out[gid] = outv;
"""


k_col_norm2_unroll = mx.fast.metal_kernel(
    name="col_norm2_unroll",
    input_names=["inp", "params"],
    output_names=["out"],
    source=_COL_NORM2_UNROLL_SRC,
)

k_sumq_sumq2_int8 = mx.fast.metal_kernel(
    name="sumq_sumq2_int8",
    input_names=["q", "params"],
    output_names=["sumq", "sumq2"],
    source=_SUMQ_SUMQ2_INT8_SRC,
)

k_apply_mask_mul = mx.fast.metal_kernel(
    name="apply_mask_mul",
    input_names=["inp", "mask", "params"],
    output_names=["out"],
    source=_APPLY_MASK_MUL_SRC,
)

k_apply_mask_select = mx.fast.metal_kernel(
    name="apply_mask_select",
    input_names=["inp", "mask", "params"],
    output_names=["out"],
    source=_APPLY_MASK_SELECT_SRC,
)

k_col_norm2_int8_affine = mx.fast.metal_kernel(
    name="col_norm2_int8_affine",
    input_names=["q", "scales", "biases", "params"],
    output_names=["out"],
    source=_COL_NORM2_INT8_AFFINE_SRC,
)

k_dequant_mask_affine = mx.fast.metal_kernel(
    name="dequant_mask_affine",
    input_names=["q", "scales", "biases", "mask", "params"],
    output_names=["out"],
    source=_DEQ_MASK_AFFINE_INT8_SRC,
)

k_gather_cols = mx.fast.metal_kernel(
    name="gather_cols",
    input_names=["inp", "indices", "params"],
    output_names=["out"],
    source=_GATHER_COLS_SRC,
)

k_scatter_from_compact = mx.fast.metal_kernel(
    name="scatter_from_compact",
    input_names=["vals", "pos", "params"],
    output_names=["out"],
    source=_SCATTER_FROM_COMPACT_SRC,
)

k_dequant_scatter_q8 = mx.fast.metal_kernel(
    name="dequant_scatter_q8",
    input_names=["codes", "s_kept", "b_kept", "pos", "inv_map", "params"],
    output_names=["out"],
    source=_DEQ_SCATTER_Q8_SRC,
)
