# kv_quant_sparse_flash_attention - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：Sparse FlashAttention + KV 量化（Per-Token-Head-Tile-128）
#   sparse_indices 驱动离散取数 + MLA-absorb（Q/K nope+rope 拼接 D=576/656）
#   + KV dequant + 在线 FlashSoftmax
# CANN 专有机制（MTE DataCopyPad / Cube-Vector 手动同步 / C_TEMPLATE vs V_TEMPLATE）不在此体现。
# 来源：attention/kv_quant_sparse_flash_attention/ (commit d28d8f9...), README.md.

import triton
import triton.language as tl


@triton.jit
def kv_quant_sparse_flash_attention_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, o_ptr,
    sparse_indices_ptr,             # [B, Q_S, KV_N, sparse_size]
    key_dequant_scale_ptr,          # per-tile dequant scale
    value_dequant_scale_ptr,
    block_table_ptr,                # PageAttention（可选）
    # strides
    stride_qb, stride_qn, stride_qs, stride_qd,
    stride_kb, stride_kn, stride_ks, stride_kd,
    stride_vb, stride_vn, stride_vs, stride_vd,
    # shape / scalars
    sm_scale,
    q_len, kv_len,                  # actual_seq_lengths
    sparse_size,                    # 每行 sparse block 数
    # feature flags (constexpr)
    FLASH_DECODE: tl.constexpr,     # FlashDecode 使能
    LAYOUT_T: tl.constexpr,         # 0:BSND / 1:TND
    KV_LAYOUT_T: tl.constexpr,      # 0:BSND / 1:TND / 2:PA_BSND
    USE_PA: tl.constexpr,           # PageAttention
    # block meta (constexpr) -- 值 source-reported
    BLOCK_M: tl.constexpr,          # S1 基本块
    BLOCK_N: tl.constexpr,          # S2 基本块
    D_NOPE: tl.constexpr,           # 512
    D_ROPE: tl.constexpr,           # 64
    D_KV: tl.constexpr,             # 656 = nope(512) + rope(128) + scale(16)
    SPARSE_BLOCK_SIZE: tl.constexpr,  # sparse 块大小
    TILE_SIZE: tl.constexpr,        # per-tile 量化块大小，默认 128
):
    pid = tl.program_id(0)  # 沿 B/N2/S1 轴切分

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_nope = tl.arange(0, D_NOPE)
    offs_d_rope = tl.arange(0, D_ROPE)

    # Q 块 [BLOCK_M, D_NOPE + D_ROPE]（MLA-absorb：nope+rope 拼接）
    q_nope = tl.load(q_ptr + offs_m[:, None] * stride_qs + offs_d_nope[None, :] * stride_qd).to(tl.float32)
    q_rope = tl.load(q_ptr + offs_m[:, None] * stride_qs + (D_NOPE + offs_d_rope)[None, :] * stride_qd).to(tl.float32)

    # 在线 softmax 状态
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_NOPE], dtype=tl.float32)  # V 仅 nope 部分

    # S2 分块循环（sparse_indices 驱动离散取数）
    for start_n in range(0, kv_len, BLOCK_N):
        # sparse_indices 查表 -> 离散 KV 物理偏移
        # sparse_indices[pid, start_n // sparse_size, :sparse_size] -> 取 sparse_size 个 block 索引
        idx = tl.load(sparse_indices_ptr + pid * sparse_size + tl.arange(0, sparse_size))  # [sparse_size]
        kv_offs = idx * SPARSE_BLOCK_SIZE + tl.arange(0, SPARSE_BLOCK_SIZE)[:, None]  # 展开到 block 内行

        # 离散 load KV（MLA-absorb：K D=656 = nope+rope+dequant_scale）
        k = tl.load(k_ptr + kv_offs[None, :, :] * stride_ks + offs_d_nope[None, None, :] * stride_kd,
                    mask=..., other=0.0)  # 伪过程：间接 load
        v = tl.load(v_ptr + kv_offs[None, :, :] * stride_vs + offs_d_nope[None, None, :] * stride_vd,
                    mask=..., other=0.0)

        # KV dequant（Per-Token-Head-Tile-128，scale 从 K D 维提取）
        k = k.to(tl.float32) * dequant_scale  # 伪过程：per-tile dequant
        v = v.to(tl.float32) * dequant_scale

        # Bmm1: Q·K^T（MLA-absorb：分段 dot）
        # 简化：nope 和 rope 分别 dot 后相加
        s = (tl.dot(q_nope, k.trans(1, 0)) + tl.dot(q_rope, k_rope.trans(1, 0))).to(tl.float32) * sm_scale

        # sparse_mode=3 rightDownCausal mask
        s = tl.where(_right_down_causal_mask(offs_m, offs_n), s, -float("inf"))

        # 在线 FlashSoftmax
        m_ij = tl.max(s, axis=1)
        s = s - m_ij[:, None]
        p = tl.exp(s).to(tl.float32)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = tl.maximum(m_i, m_ij)

    # rescale 写回
    acc = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * stride_os + offs_d_nope[None, :] * stride_od,
             acc.to(o_ptr.dtype.element_ty()))


# MLA-absorb 辅助：Q/K 从 K D=656 中提取 dequant_scale（最后 4*4=16 bytes）
# rightDownCausal mask 辅助
# grid = (B * N2 * ceil(S1 / BLOCK_M),)  # 沿 B/N2/S1 轴切分