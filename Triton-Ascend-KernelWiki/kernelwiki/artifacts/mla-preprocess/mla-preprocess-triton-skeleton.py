# mla_preprocess - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：MLA 前处理融合（RmsNormQuant + 3 matmul + RoPE + Cache scatter 写回）。
# CANN 专有机制（buffer iterator/ND↔NZ/ProcessCube-Vector 分离/swizzle）不在此体现。
# 来源：attention/mla_preprocess/ (commit d28d8f9...), README L12-54.
#
# 映射注记（见 artifact tiling_dispatch_tables.md §F）：
#   program_id <- CANN core 沿 tokenNum 切分
#   BLOCK_* <- mm 基本块 m0/k0/n0 (source-reported)
#   FP32 accumulator <- BF16 中间
#   slot_mapping scatter <- CANN Cache 写回

import triton
import triton.language as tl


@triton.jit
def mla_preprocess_kernel(
    # pointers
    x_ptr,            # [tokenNum, hiddenSize] FP16/BF16
    gamma0, beta0, quant_scale0, quant_offset0,   # 1st RmsNormQuant
    wdqkv_ptr, descale0, bias0,                    # mm1 降维权重 (INT8/BF16, NZ)
    gamma1, beta1, quant_scale1, quant_offset1,   # 2nd RmsNormQuant
    wuq_ptr, descale1, bias1,                      # mm2 上采样 Q
    gamma2,                                         # 3rd RmsNorm (K)
    cos_ptr, sin_ptr,                              # RoPE
    wuk_ptr,                                        # mm3 上采样 K
    kv_cache_ptr, kv_cache_rope_ptr, slot_mapping_ptr,  # Cache + 索引
    q_out_ptr, kv_cache_out_ptr, q_rope_out_ptr, kr_cache_out_ptr,
    # scalars
    token_num, hidden_size, eps,
    head_num,
    # meta (constexpr) -- 值 source-reported
    BLOCK_T: tl.constexpr,     # token 块
    HIDDEN: tl.constexpr,      # 2048~10240
    WDQ_DIM: tl.constexpr,     # 128（预留，实际 1536/2112）
    ROPE_DIM: tl.constexpr,    # 64
    HEAD_DIM: tl.constexpr,    # 512 (nope)
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = offs_t < token_num
    offs_h = tl.arange(0, HIDDEN)

    # ---- 1. RmsNormQuant₁ ----
    x = tl.load(x_ptr + offs_t[:, None] * HIDDEN + offs_h[None, :],
                mask=t_mask[:, None], other=0.0).to(tl.float32)
    g0 = tl.load(gamma0 + offs_h).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / HIDDEN + eps)   # [BLOCK_T]
    xn = x / rms[:, None] * g0[None, :]                   # RmsNorm
    xq = (xn + tl.load(bias0 + offs_h)) * tl.load(quant_scale0 + offs_h)   # +bias, ×deqScale (示意)

    # ---- 2. mm1: ×W^DQKV 降维 (Cube) -> 拆分 Q(通路1) / K(通路2) ----
    # wdqkv [WDQ_OUT, HIDDEN]; xq [BLOCK_T, HIDDEN]
    wdqkv = tl.load(wdqkv_ptr + tl.arange(0, WDQ_DIM)[:, None] * HIDDEN + offs_h[None, :])  # 简化
    y = tl.dot(xq, wdqkv.trans(1, 0).to(tl.float32))   # [BLOCK_T, WDQ_DIM]
    y_q, y_k = y_split(y)   # 通路1(Q) / 通路2(K)，按列拆分（伪）

    # ---- 3a. Q 通路: RmsNormQuant₂ -> mm2 ×W^UQ -> 拆 nope/rope ----
    # y_q -> RmsNormQuant₂ -> mm2 -> [BLOCK_T, headNum*192]
    yq_n = rmsnorm_quant(y_q, gamma1, beta1, quant_scale1, quant_offset1, eps)  # 伪过程
    wuq = tl.load(wuq_ptr + ...)   # [headNum*192, ...]
    q_up = tl.dot(yq_n, wuq.trans(1, 0).to(tl.float32))   # mm2
    q_nope, q_rope_in = q_split(q_up)   # 通路3 nope / 通路4 rope

    # 通路3: mm3 ×W^UK -> q^N
    wuk = tl.load(wuk_ptr + ...)
    q_n = tl.dot(q_nope, wuk.trans(1, 0).to(tl.float32))   # mm3 -> qOut [BLOCK_T, headNum, 512]
    # 通路4: RoPE -> q^R
    q_r = rope(q_rope_in, cos_ptr, sin_ptr, offs_t, ROPE_DIM)   # 伪过程 -> qRopeOut

    # ---- 3b. K 通路: RmsNorm₃ -> Cache k^N；RoPE -> Cache k^R ----
    k_n = rmsnorm(y_k, gamma2, eps)                         # 通路5 -> k^N
    k_r = rope(y_k, cos_ptr, sin_ptr, offs_t, ROPE_DIM)     # 通路6 -> k^R
    # Cache scatter 写回（slot_mapping 索引）
    slots = tl.load(slot_mapping_ptr + offs_t, mask=t_mask, other=0)
    tl.store(kv_cache_out_ptr + slots[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :],
             k_n, mask=t_mask[:, None])                     # k^N -> kvCacheOut
    tl.store(kr_cache_out_ptr + slots[:, None] * ROPE_DIM + tl.arange(0, ROPE_DIM)[None, :],
             k_r, mask=t_mask[:, None])                     # k^R -> krCacheOut

    # Q 输出
    tl.store(q_out_ptr + offs_t[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :],
             q_n, mask=t_mask[:, None])
    tl.store(q_rope_out_ptr + offs_t[:, None] * ROPE_DIM + tl.arange(0, ROPE_DIM)[None, :],
             q_r, mask=t_mask[:, None])


# 辅助（伪过程，实际为内联计算）：
#   rmsnorm_quant(x, γ, β, qs, qo, eps) = (rmsnorm(x,γ,eps) + β) * qs   # +qo 量化
#   rope(x, cos, sin, offs, dim) = x * cos + rotate_half(x) * sin
#   y_split / q_split 按 MLA 维度切分（nope/rope）
#
# cacheMode 分支：cacheMode=0 拼接 k^N+k^R 写同一 cache；1/3 拆分；2 额外 NZ+int8 量化。
# quantMode 分支：0=per_tensor 静态非对称（默认），1/2/3 未实现（README L315）。
# MLA 吸收：数学上 q^N=RmsNormQuant(x)·W^DQKV·W^UK（W^UQ·W^UK 折叠），
#   kernel 实现 3 个 matmul，是否运行时吸收 needs_verification。
#
# grid = (ceil(token_num / BLOCK_T),)   # 沿 tokenNum 多核
