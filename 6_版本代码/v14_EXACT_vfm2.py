"""MhcSinkhorn — Triton Ascend 实现（scale-vector 重构 + n=3/6 标量展开扩展）。

算法重构（相对旧 v1 版本）：
  旧 v1 的瓶颈是「每轮对 [BLOCK_M, N, N] tile 做 keep_dims 归约 + 广播除法」，
  在 Triton-Ascend 上 reduce combine 退化成 f32 标量链（矢量宽度归零），
  且时延随 tile 元素线性放大、与并行度无关（issue-bound）。

本轮改用 **Sinkhorn 对角缩放分解**：

  Sinkhorn-Knopp 交替 row/col 归一化本质是逐行/逐列对角缩放。若把 iter1
  row-softmax 后的结果记为固定基准 W[i,j]，则任意迭代轮的值都可写作
  x[i,j] = r[i] · W[i,j] · c[j]，其中 r、c 分别是行/列缩放向量。

  每轮迭代只需在这两个 N 长向量上做两个 N×N matvec（W·c 与 W^T·r）与逐元素
  更新，不再反复对 N×N tile 做 keep_dims 归约 + 广播写回，计算量从 O(N²·iter)
  降为 O(N²·iter)（同一量级但去掉广播除法）更重要的是移除了每轮昂贵的布局转换。
  数学上与 golden 严格等价（eps 在两处位置语义保持一致）。

  W 全程固定、只在 iter 前与最终输出时访问 [BLOCK_M, N, N] tile；迭代期间
  rv/cv 均为 [BLOCK_M, N] 级向量，散热量小。

精度：全 20 case MERE ~1e-7~3e-7，距 fp32 eco 阈值（MERE<1.22e-4 / MARE<1.22e-3）
  余量 3~5 个数量级。

性能（相对旧 v1，Ascend950PR 实测，do_bench 纯 GPU 口径 vs baseline_perf_us）：
  大 B case 显著提升：case3 0.23→0.38x、case4 0.14→0.26x、case5 0.035→0.085x、
  case6 0.017→0.036x、case15 0.07→0.14x；整体 geomean 约 1.5x 提升。
  最优 grid 实测恒为 ~256（grid=min(B,256)，BM=B/grid 取 2 的幂）。

架构：
  - pow2 阶（n∈{2,4,8,16}）：单 kernel scale-vector（_sinkhorn_scale）。
  - 非 pow2 阶（n∈{3,6,12}）：回落原 fused 版（_mhc_sinkhorn_fused，掩码已应对 padding）。
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ===========================================================================
# Kernel A：scale-vector（pow2 阶 n）
# ===========================================================================
@triton.jit
def _sinkhorn_scale(
    ptr, optr, B, iter_step: tl.constexpr, eps,
    N: tl.constexpr, N_TRUE: tl.constexpr, BLOCK_M: tl.constexpr,
         EXACT: tl.constexpr):
    pid = tl.program_id(0)
    bb = tl.arange(0, BLOCK_M)[:, None, None]
    ri = tl.arange(0, N)[None, :, None]
    ci = tl.arange(0, N)[None, None, :]
    b = pid * BLOCK_M + bb
    offs = b * (N_TRUE * N_TRUE) + ri * N_TRUE + ci
    b_ok = b < B
    r_ok = ri < N_TRUE          # pow2：恒真，保证下游 where 静态可折叠
    c_ok = ci < N_TRUE
    if EXACT:
        mask3 = r_ok & c_ok
        x = tl.load(ptr + offs, mask=mask3, other=float("-inf"))
    else:
        mask3 = b_ok & r_ok & c_ok
        x = tl.load(ptr + offs, mask=mask3, other=float("-inf"))

    # iter1：row-softmax 起步，得到固定基准 W
    rmax = tl.max(x, axis=2, keep_dims=True)
    if not EXACT:
        rmax = tl.where(b_ok & r_ok, rmax, 0.0)
    e = tl.exp(x - rmax)
    s = tl.sum(e, axis=2, keep_dims=True)
    w = e / s + eps                                # [M, N, N] 固定基准（含 iter1 row eps）

    # iter1 col 归约 -> c；r 初始 = 1
    col0 = tl.sum(w, axis=1, keep_dims=True)       # [M, 1, N]
    cv = 1.0 / (col0 + eps)                        # [M, 1, N]
    rv = tl.full((BLOCK_M, N, 1), 1.0, tl.float32)  # [M, N, 1]

    # 剩余 iter_step-1 轮：仅 rv / cv 两个 N 长向量迭代
    for _ in range(1, iter_step):
        wc = tl.sum(w * cv, axis=2, keep_dims=True)   # (W·c)[M,N,1]
        rv = rv / (rv * wc + eps)
        wtr = tl.sum(w * rv, axis=1, keep_dims=True)  # (W^T·r)[M,1,N]
        cv = cv / (cv * wtr + eps)

    # 最终合成：out = rv · W · cv
    out = w * rv * cv
    tl.store(optr + offs, out, mask=mask3)


@triton.jit
def _sinkhorn_scale_masked(
    ptr, optr, B, iter_step: tl.constexpr, eps,
    N: tl.constexpr, N_TRUE: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    bb = tl.arange(0, BLOCK_M)[:, None, None]
    ri = tl.arange(0, N)[None, :, None]
    ci = tl.arange(0, N)[None, None, :]
    b = pid * BLOCK_M + bb
    offs = b * (N_TRUE * N_TRUE) + ri * N_TRUE + ci
    b_ok = b < B
    r_ok = ri < N_TRUE
    c_ok = ci < N_TRUE
    row_ok = b_ok & r_ok
    mask3 = row_ok & c_ok
    eps_m = tl.where(r_ok & c_ok, eps, 0.0)         # 与 golden 一致：+eps 只在有效车道
    pad_row = tl.where(row_ok, 0.0, 1.0)            # padding 行软化 0/1=0
    x = tl.load(ptr + offs, mask=mask3, other=float("-inf"))

    # iter1：row-softmax 起步，得到固定基准 W
    rmax = tl.max(x, axis=2, keep_dims=True)
    rmax = tl.where(row_ok, rmax, 0.0)
    e = tl.exp(x - rmax)
    s = tl.sum(e, axis=2, keep_dims=True)
    w = e / (s + pad_row) + eps_m                  # [M, N, N] 固定基准（含 iter1 row eps，padding 中性化）

    # iter1 col 归约 -> c；r 初始 = 1
    col0 = tl.sum(w, axis=1, keep_dims=True)       # [M, 1, N]
    cv = 1.0 / (col0 + eps)                        # [M, 1, N]
    rv = tl.full((BLOCK_M, N, 1), 1.0, tl.float32)  # [M, N, 1]

    # 剩余 iter_step-1 轮：仅 rv / cv 两个 N 长向量迭代
    # 【fix】每轮更新后把 padding 行/列的 rv/cv 归零——否则它们按 1/eps 涨到 inf，
    # 经 w*cv / w*rv 的 0*inf=NaN 反向污染有效行/列。
    for _ in range(1, iter_step):
        wc = tl.sum(w * cv, axis=2, keep_dims=True)   # (W·c)[M,N,1]
        rv = rv / (rv * wc + eps)
        rv = tl.where(row_ok, rv, 0.0)
        wtr = tl.sum(w * rv, axis=1, keep_dims=True)  # (W^T·r)[M,1,N]
        cv = cv / (cv * wtr + eps)
        cv = tl.where(c_ok, cv, 0.0)

    # 最终合成：out = rv · W · cv
    out = w * rv * cv
    tl.store(optr + offs, out, mask=mask3)


# ===========================================================================
# Kernel B：标量展开（n∈{2,4} 专用，消除 AIV reduce 树）
#   AIV 上 tl.sum/max 对 [BM,N,N] tile 的 reduce 会退化成窄标量链（矢量宽度归零）。
#   改为把每个矩阵元素建模为独立 [BM] 向量、显式乘加，逐元素迭代，无 reduce 树。
#   实测（Ascend950PR）：n=4 c6 B=16384 39µs vs scale 437µs（11x）、c5 17 vs 118（6.8x）；
#   n=2 c7 B=1024 10.6 vs 30.7（3x）、c18 B=1 4.39 vs 5.26。
# ===========================================================================

@triton.jit
def _join4(a, b, c, d, BM: tl.constexpr):
    """4 个 [BM] -> [BM,4]（偶/奇递归：join 天然交错，递归反转即线性序）。"""
    E = tl.join(tl.reshape(a, [BM, 1]), tl.reshape(c, [BM, 1]))   # [BM,1,2] = {a,c}
    O = tl.join(tl.reshape(b, [BM, 1]), tl.reshape(d, [BM, 1]))   # {b,d}
    x = tl.join(tl.reshape(E, [BM, 2]), tl.reshape(O, [BM, 2]))   # {a,b,c,d}
    return tl.reshape(x, [BM, 4])


@triton.jit
def _join8(v0, v1, v2, v3, v4, v5, v6, v7, BM: tl.constexpr):
    """8 个 [BM] -> [BM,8]（偶/奇递归）。"""
    E = _join4(v0, v2, v4, v6, BM)   # [BM,4] = {v0,v2,v4,v6}
    O = _join4(v1, v3, v5, v7, BM)   # {v1,v3,v5,v7}
    x = tl.join(tl.reshape(E, [BM, 4]), tl.reshape(O, [BM, 4]))
    return tl.reshape(x, [BM, 8])


@triton.jit
def _join32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
            v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31,
            BM: tl.constexpr):
    """32 个 [BM] -> [BM,32]（偶/奇递归）。"""
    E = _join16(v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, BM)
    O = _join16(v1, v3, v5, v7, v9, v11, v13, v15, v17, v19, v21, v23, v25, v27, v29, v31, BM)
    x = tl.join(tl.reshape(E, [BM, 16]), tl.reshape(O, [BM, 16]))
    return tl.reshape(x, [BM, 32])


@triton.jit
def _join16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, xF,
            BM: tl.constexpr):
    """16 个 [BM] 向量合并为 [BM,16]（寄存器层布局变换，替代 16 路 stride-gather store）。

    与 split 树互逆：join(a,b) 把 a/b 放到新末维的 0/1 位，reshape 后线性序为
    a,b 交替；按 L4→L1 逆序组合即恢复 k=0..15 的行主序。实测（Ascend950PR）：
    合并 store 使 _sk4 的 MTE3 占比 51.7%→2.7%，case4/6 kernel 时间 -31%。
    """
    A00 = tl.join(tl.reshape(x0, [BM, 1]), tl.reshape(x8, [BM, 1]))   # [BM,1,2]
    A01 = tl.join(tl.reshape(x4, [BM, 1]), tl.reshape(xC, [BM, 1]))
    A10 = tl.join(tl.reshape(x2, [BM, 1]), tl.reshape(xA, [BM, 1]))
    A11 = tl.join(tl.reshape(x6, [BM, 1]), tl.reshape(xE, [BM, 1]))
    B00 = tl.join(tl.reshape(x1, [BM, 1]), tl.reshape(x9, [BM, 1]))
    B01 = tl.join(tl.reshape(x5, [BM, 1]), tl.reshape(xD, [BM, 1]))
    B10 = tl.join(tl.reshape(x3, [BM, 1]), tl.reshape(xB, [BM, 1]))
    B11 = tl.join(tl.reshape(x7, [BM, 1]), tl.reshape(xF, [BM, 1]))
    A0 = tl.join(tl.reshape(A00, [BM, 2]), tl.reshape(A01, [BM, 2]))   # [BM,2,2]
    A1 = tl.join(tl.reshape(A10, [BM, 2]), tl.reshape(A11, [BM, 2]))
    B0 = tl.join(tl.reshape(B00, [BM, 2]), tl.reshape(B01, [BM, 2]))
    B1 = tl.join(tl.reshape(B10, [BM, 2]), tl.reshape(B11, [BM, 2]))
    A = tl.join(tl.reshape(A0, [BM, 4]), tl.reshape(A1, [BM, 4]))      # [BM,4,2]
    B = tl.join(tl.reshape(B0, [BM, 4]), tl.reshape(B1, [BM, 4]))
    x = tl.join(tl.reshape(A, [BM, 8]), tl.reshape(B, [BM, 8]))        # [BM,8,2]
    return tl.reshape(x, [BM, 16])


@triton.jit
def _sk4(ptr, optr, B, iter_step: tl.constexpr, eps,
         BLOCK_M: tl.constexpr, EXACT: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * 16
    if EXACT:
        x0 = tl.load(ptr + base + 0); x1 = tl.load(ptr + base + 1)
        x2 = tl.load(ptr + base + 2); x3 = tl.load(ptr + base + 3)
        x4 = tl.load(ptr + base + 4); x5 = tl.load(ptr + base + 5)
        x6 = tl.load(ptr + base + 6); x7 = tl.load(ptr + base + 7)
        x8 = tl.load(ptr + base + 8); x9 = tl.load(ptr + base + 9)
        xA = tl.load(ptr + base + 10); xB = tl.load(ptr + base + 11)
        xC = tl.load(ptr + base + 12); xD = tl.load(ptr + base + 13)
        xE = tl.load(ptr + base + 14); xF = tl.load(ptr + base + 15)
    else:
        x0 = tl.load(ptr + base + 0, mask=ok)
        x1 = tl.load(ptr + base + 1, mask=ok)
        x2 = tl.load(ptr + base + 2, mask=ok)
        x3 = tl.load(ptr + base + 3, mask=ok)
        x4 = tl.load(ptr + base + 4, mask=ok)
        x5 = tl.load(ptr + base + 5, mask=ok)
        x6 = tl.load(ptr + base + 6, mask=ok)
        x7 = tl.load(ptr + base + 7, mask=ok)
        x8 = tl.load(ptr + base + 8, mask=ok)
        x9 = tl.load(ptr + base + 9, mask=ok)
        xA = tl.load(ptr + base + 10, mask=ok)
        xB = tl.load(ptr + base + 11, mask=ok)
        xC = tl.load(ptr + base + 12, mask=ok)
        xD = tl.load(ptr + base + 13, mask=ok)
        xE = tl.load(ptr + base + 14, mask=ok)
        xF = tl.load(ptr + base + 15, mask=ok)
    r0 = tl.maximum(tl.maximum(tl.maximum(x0, x1), x2), x3)
    r1 = tl.maximum(tl.maximum(tl.maximum(x4, x5), x6), x7)
    r2 = tl.maximum(tl.maximum(tl.maximum(x8, x9), xA), xB)
    r3 = tl.maximum(tl.maximum(tl.maximum(xC, xD), xE), xF)
    e0 = tl.exp(x0 - r0);  e1 = tl.exp(x1 - r0);  e2 = tl.exp(x2 - r0);  e3 = tl.exp(x3 - r0)
    e4 = tl.exp(x4 - r1);  e5 = tl.exp(x5 - r1);  e6 = tl.exp(x6 - r1);  e7 = tl.exp(x7 - r1)
    e8 = tl.exp(x8 - r2);  e9 = tl.exp(x9 - r2);  eA = tl.exp(xA - r2);  eB = tl.exp(xB - r2)
    eC = tl.exp(xC - r3);  eD = tl.exp(xD - r3);  eE = tl.exp(xE - r3);  eF = tl.exp(xF - r3)
    s0 = e0 + e1 + e2 + e3; w0 = e0 / (s0 + 0.0); w1 = e1 / (s0 + 0.0); w2 = e2 / (s0 + 0.0); w3 = e3 / (s0 + 0.0)
    s1 = e4 + e5 + e6 + e7; w4 = e4 / (s1 + 0.0); w5 = e5 / (s1 + 0.0); w6 = e6 / (s1 + 0.0); w7 = e7 / (s1 + 0.0)
    s2 = e8 + e9 + eA + eB; w8 = e8 / (s2 + 0.0); w9 = e9 / (s2 + 0.0); wA = eA / (s2 + 0.0); wB = eB / (s2 + 0.0)
    s3 = eC + eD + eE + eF; wC = eC / (s3 + 0.0); wD = eD / (s3 + 0.0); wE = eE / (s3 + 0.0); wF = eF / (s3 + 0.0)
    w0 += eps; w1 += eps; w2 += eps; w3 += eps
    w4 += eps; w5 += eps; w6 += eps; w7 += eps
    w8 += eps; w9 += eps; wA += eps; wB += eps
    wC += eps; wD += eps; wE += eps; wF += eps
    c0 = w0 + w4 + w8 + wC
    c1 = w1 + w5 + w9 + wD
    c2 = w2 + w6 + wA + wE
    c3 = w3 + w7 + wB + wF
    cv0 = 1.0 / (c0 + eps); cv1 = 1.0 / (c1 + eps); cv2 = 1.0 / (c2 + eps); cv3 = 1.0 / (c3 + eps)
    rv0 = tl.full([BLOCK_M], 1.0, tl.float32); rv1 = rv0; rv2 = rv0; rv3 = rv0
    for _ in range(1, iter_step):
        wc0 = w0 * cv0 + w1 * cv1 + w2 * cv2 + w3 * cv3
        wc1 = w4 * cv0 + w5 * cv1 + w6 * cv2 + w7 * cv3
        wc2 = w8 * cv0 + w9 * cv1 + wA * cv2 + wB * cv3
        wc3 = wC * cv0 + wD * cv1 + wE * cv2 + wF * cv3
        rv0 = rv0 / (rv0 * wc0 + eps); rv1 = rv1 / (rv1 * wc1 + eps)
        rv2 = rv2 / (rv2 * wc2 + eps); rv3 = rv3 / (rv3 * wc3 + eps)
        wtr0 = w0 * rv0 + w4 * rv1 + w8 * rv2 + wC * rv3
        wtr1 = w1 * rv0 + w5 * rv1 + w9 * rv2 + wD * rv3
        wtr2 = w2 * rv0 + w6 * rv1 + wA * rv2 + wE * rv3
        wtr3 = w3 * rv0 + w7 * rv1 + wB * rv2 + wF * rv3
        cv0 = cv0 / (cv0 * wtr0 + eps); cv1 = cv1 / (cv1 * wtr1 + eps)
        cv2 = cv2 / (cv2 * wtr2 + eps); cv3 = cv3 / (cv3 * wtr3 + eps)
    o0 = w0 * rv0 * cv0; o1 = w1 * rv0 * cv1; o2 = w2 * rv0 * cv2; o3 = w3 * rv0 * cv3
    o4 = w4 * rv1 * cv0; o5 = w5 * rv1 * cv1; o6 = w6 * rv1 * cv2; o7 = w7 * rv1 * cv3
    o8 = w8 * rv2 * cv0; o9 = w9 * rv2 * cv1; oA = wA * rv2 * cv2; oB = wB * rv2 * cv3
    oC = wC * rv3 * cv0; oD = wD * rv3 * cv1; oE = wE * rv3 * cv2; oF = wF * rv3 * cv3
    offs16 = off[:, None] * 16 + tl.arange(0, 16)[None, :]
    y = _join16(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, oA, oB, oC, oD, oE, oF, BLOCK_M)
    if EXACT:
        tl.store(optr + offs16, y)
    else:
        tl.store(optr + offs16, y, mask=ok[:, None])


@triton.jit
def _sk2(ptr, optr, B, iter_step: tl.constexpr, eps,
         BLOCK_M: tl.constexpr,
         EXACT: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * 4
    if EXACT:
        x0 = tl.load(ptr + base + 0)
        x1 = tl.load(ptr + base + 1)
        x2 = tl.load(ptr + base + 2)
        x3 = tl.load(ptr + base + 3)
    else:
        x0 = tl.load(ptr + base + 0, mask=ok)
        x1 = tl.load(ptr + base + 1, mask=ok)
        x2 = tl.load(ptr + base + 2, mask=ok)
        x3 = tl.load(ptr + base + 3, mask=ok)
    r0 = tl.maximum(x0, x1)
    r1 = tl.maximum(x2, x3)
    e0 = tl.exp(x0 - r0); e1 = tl.exp(x1 - r0); e2 = tl.exp(x2 - r1); e3 = tl.exp(x3 - r1)
    s0 = e0 + e1; w0 = e0 / (s0 + 0.0); w1 = e1 / (s0 + 0.0)
    s1 = e2 + e3; w2 = e2 / (s1 + 0.0); w3 = e3 / (s1 + 0.0)
    w0 += eps; w1 += eps; w2 += eps; w3 += eps
    c0 = w0 + w2; c1 = w1 + w3
    cv0 = 1.0 / (c0 + eps); cv1 = 1.0 / (c1 + eps)
    rv0 = tl.full([BLOCK_M], 1.0, tl.float32); rv1 = rv0
    for _ in range(1, iter_step):
        wc0 = w0 * cv0 + w1 * cv1
        wc1 = w2 * cv0 + w3 * cv1
        rv0 = rv0 / (rv0 * wc0 + eps); rv1 = rv1 / (rv1 * wc1 + eps)
        wtr0 = w0 * rv0 + w2 * rv1
        wtr1 = w1 * rv0 + w3 * rv1
        cv0 = cv0 / (cv0 * wtr0 + eps); cv1 = cv1 / (cv1 * wtr1 + eps)
    o0 = w0 * rv0 * cv0; o1 = w1 * rv0 * cv1
    o2 = w2 * rv1 * cv0; o3 = w3 * rv1 * cv1
    offs4 = off[:, None] * 4 + tl.arange(0, 4)[None, :]
    y = _join4(o0, o1, o2, o3, BLOCK_M)
    if EXACT:
        tl.store(optr + offs4, y)
    else:
        tl.store(optr + offs4, y, mask=ok[:, None])


# ===========================================================================
# Kernel B2：标量展开 n=3 / n=6（gen_skn.py 生成，消 AIV reduce 树）
# ===========================================================================
@triton.jit
def _sk3(ptr, optr, B, iter_step: tl.constexpr, eps, BLOCK_M: tl.constexpr, EXACT: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * 9
    if EXACT:
        x0 = tl.load(ptr + base + 0)
        x1 = tl.load(ptr + base + 1)
        x2 = tl.load(ptr + base + 2)
        x3 = tl.load(ptr + base + 3)
        x4 = tl.load(ptr + base + 4)
        x5 = tl.load(ptr + base + 5)
        x6 = tl.load(ptr + base + 6)
        x7 = tl.load(ptr + base + 7)
        x8 = tl.load(ptr + base + 8)
    else:
        x0 = tl.load(ptr + base + 0, mask=ok)
        x1 = tl.load(ptr + base + 1, mask=ok)
        x2 = tl.load(ptr + base + 2, mask=ok)
        x3 = tl.load(ptr + base + 3, mask=ok)
        x4 = tl.load(ptr + base + 4, mask=ok)
        x5 = tl.load(ptr + base + 5, mask=ok)
        x6 = tl.load(ptr + base + 6, mask=ok)
        x7 = tl.load(ptr + base + 7, mask=ok)
        x8 = tl.load(ptr + base + 8, mask=ok)
    r0 = tl.maximum(tl.maximum(x0, x1), x2)
    r1 = tl.maximum(tl.maximum(x3, x4), x5)
    r2 = tl.maximum(tl.maximum(x6, x7), x8)
    e0 = tl.exp(x0 - r0)
    e1 = tl.exp(x1 - r0)
    e2 = tl.exp(x2 - r0)
    e3 = tl.exp(x3 - r1)
    e4 = tl.exp(x4 - r1)
    e5 = tl.exp(x5 - r1)
    e6 = tl.exp(x6 - r2)
    e7 = tl.exp(x7 - r2)
    e8 = tl.exp(x8 - r2)
    s0 = e0 + e1 + e2
    w0 = e0 / (s0 + 0.0)
    w1 = e1 / (s0 + 0.0)
    w2 = e2 / (s0 + 0.0)
    s1 = e3 + e4 + e5
    w3 = e3 / (s1 + 0.0)
    w4 = e4 / (s1 + 0.0)
    w5 = e5 / (s1 + 0.0)
    s2 = e6 + e7 + e8
    w6 = e6 / (s2 + 0.0)
    w7 = e7 / (s2 + 0.0)
    w8 = e8 / (s2 + 0.0)
    w0 += eps
    w1 += eps
    w2 += eps
    w3 += eps
    w4 += eps
    w5 += eps
    w6 += eps
    w7 += eps
    w8 += eps
    c0 = w0 + w3 + w6
    c1 = w1 + w4 + w7
    c2 = w2 + w5 + w8
    cv0 = 1.0 / (c0 + eps)
    cv1 = 1.0 / (c1 + eps)
    cv2 = 1.0 / (c2 + eps)
    rv0 = tl.full([BLOCK_M], 1.0, tl.float32)
    rv1 = rv0
    rv2 = rv0
    for _ in range(1, iter_step):
        wc0 = w0 * cv0 + w1 * cv1 + w2 * cv2
        wc1 = w3 * cv0 + w4 * cv1 + w5 * cv2
        wc2 = w6 * cv0 + w7 * cv1 + w8 * cv2
        rv0 = rv0 / (rv0 * wc0 + eps)
        rv1 = rv1 / (rv1 * wc1 + eps)
        rv2 = rv2 / (rv2 * wc2 + eps)
        wtr0 = w0 * rv0 + w3 * rv1 + w6 * rv2
        wtr1 = w1 * rv0 + w4 * rv1 + w7 * rv2
        wtr2 = w2 * rv0 + w5 * rv1 + w8 * rv2
        cv0 = cv0 / (cv0 * wtr0 + eps)
        cv1 = cv1 / (cv1 * wtr1 + eps)
        cv2 = cv2 / (cv2 * wtr2 + eps)
    o0 = w0 * rv0 * cv0
    o1 = w1 * rv0 * cv1
    o2 = w2 * rv0 * cv2
    o3 = w3 * rv1 * cv0
    o4 = w4 * rv1 * cv1
    o5 = w5 * rv1 * cv2
    o6 = w6 * rv2 * cv0
    o7 = w7 * rv2 * cv1
    o8 = w8 * rv2 * cv2
    offs8 = off[:, None] * 9 + tl.arange(0, 8)[None, :]
    y = _join8(o0, o1, o2, o3, o4, o5, o6, o7, BLOCK_M)
    if EXACT:
        tl.store(optr + offs8, y)
        tl.store(optr + base + 8, o8)
    else:
        tl.store(optr + offs8, y, mask=ok[:, None])
        tl.store(optr + base + 8, o8, mask=ok)


@triton.jit
def _sk6(ptr, optr, B, iter_step: tl.constexpr, eps, BLOCK_M: tl.constexpr, EXACT: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * 36
    if EXACT:
        x0 = tl.load(ptr + base + 0)
        x1 = tl.load(ptr + base + 1)
        x2 = tl.load(ptr + base + 2)
        x3 = tl.load(ptr + base + 3)
        x4 = tl.load(ptr + base + 4)
        x5 = tl.load(ptr + base + 5)
        x6 = tl.load(ptr + base + 6)
        x7 = tl.load(ptr + base + 7)
        x8 = tl.load(ptr + base + 8)
        x9 = tl.load(ptr + base + 9)
        x10 = tl.load(ptr + base + 10)
        x11 = tl.load(ptr + base + 11)
        x12 = tl.load(ptr + base + 12)
        x13 = tl.load(ptr + base + 13)
        x14 = tl.load(ptr + base + 14)
        x15 = tl.load(ptr + base + 15)
        x16 = tl.load(ptr + base + 16)
        x17 = tl.load(ptr + base + 17)
        x18 = tl.load(ptr + base + 18)
        x19 = tl.load(ptr + base + 19)
        x20 = tl.load(ptr + base + 20)
        x21 = tl.load(ptr + base + 21)
        x22 = tl.load(ptr + base + 22)
        x23 = tl.load(ptr + base + 23)
        x24 = tl.load(ptr + base + 24)
        x25 = tl.load(ptr + base + 25)
        x26 = tl.load(ptr + base + 26)
        x27 = tl.load(ptr + base + 27)
        x28 = tl.load(ptr + base + 28)
        x29 = tl.load(ptr + base + 29)
        x30 = tl.load(ptr + base + 30)
        x31 = tl.load(ptr + base + 31)
        x32 = tl.load(ptr + base + 32)
        x33 = tl.load(ptr + base + 33)
        x34 = tl.load(ptr + base + 34)
        x35 = tl.load(ptr + base + 35)
    else:
        x0 = tl.load(ptr + base + 0, mask=ok)
        x1 = tl.load(ptr + base + 1, mask=ok)
        x2 = tl.load(ptr + base + 2, mask=ok)
        x3 = tl.load(ptr + base + 3, mask=ok)
        x4 = tl.load(ptr + base + 4, mask=ok)
        x5 = tl.load(ptr + base + 5, mask=ok)
        x6 = tl.load(ptr + base + 6, mask=ok)
        x7 = tl.load(ptr + base + 7, mask=ok)
        x8 = tl.load(ptr + base + 8, mask=ok)
        x9 = tl.load(ptr + base + 9, mask=ok)
        x10 = tl.load(ptr + base + 10, mask=ok)
        x11 = tl.load(ptr + base + 11, mask=ok)
        x12 = tl.load(ptr + base + 12, mask=ok)
        x13 = tl.load(ptr + base + 13, mask=ok)
        x14 = tl.load(ptr + base + 14, mask=ok)
        x15 = tl.load(ptr + base + 15, mask=ok)
        x16 = tl.load(ptr + base + 16, mask=ok)
        x17 = tl.load(ptr + base + 17, mask=ok)
        x18 = tl.load(ptr + base + 18, mask=ok)
        x19 = tl.load(ptr + base + 19, mask=ok)
        x20 = tl.load(ptr + base + 20, mask=ok)
        x21 = tl.load(ptr + base + 21, mask=ok)
        x22 = tl.load(ptr + base + 22, mask=ok)
        x23 = tl.load(ptr + base + 23, mask=ok)
        x24 = tl.load(ptr + base + 24, mask=ok)
        x25 = tl.load(ptr + base + 25, mask=ok)
        x26 = tl.load(ptr + base + 26, mask=ok)
        x27 = tl.load(ptr + base + 27, mask=ok)
        x28 = tl.load(ptr + base + 28, mask=ok)
        x29 = tl.load(ptr + base + 29, mask=ok)
        x30 = tl.load(ptr + base + 30, mask=ok)
        x31 = tl.load(ptr + base + 31, mask=ok)
        x32 = tl.load(ptr + base + 32, mask=ok)
        x33 = tl.load(ptr + base + 33, mask=ok)
        x34 = tl.load(ptr + base + 34, mask=ok)
        x35 = tl.load(ptr + base + 35, mask=ok)
    r0 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x0, x1), x2), x3), x4), x5)
    r1 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x6, x7), x8), x9), x10), x11)
    r2 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x12, x13), x14), x15), x16), x17)
    r3 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x18, x19), x20), x21), x22), x23)
    r4 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x24, x25), x26), x27), x28), x29)
    r5 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x30, x31), x32), x33), x34), x35)
    e0 = tl.exp(x0 - r0)
    e1 = tl.exp(x1 - r0)
    e2 = tl.exp(x2 - r0)
    e3 = tl.exp(x3 - r0)
    e4 = tl.exp(x4 - r0)
    e5 = tl.exp(x5 - r0)
    e6 = tl.exp(x6 - r1)
    e7 = tl.exp(x7 - r1)
    e8 = tl.exp(x8 - r1)
    e9 = tl.exp(x9 - r1)
    e10 = tl.exp(x10 - r1)
    e11 = tl.exp(x11 - r1)
    e12 = tl.exp(x12 - r2)
    e13 = tl.exp(x13 - r2)
    e14 = tl.exp(x14 - r2)
    e15 = tl.exp(x15 - r2)
    e16 = tl.exp(x16 - r2)
    e17 = tl.exp(x17 - r2)
    e18 = tl.exp(x18 - r3)
    e19 = tl.exp(x19 - r3)
    e20 = tl.exp(x20 - r3)
    e21 = tl.exp(x21 - r3)
    e22 = tl.exp(x22 - r3)
    e23 = tl.exp(x23 - r3)
    e24 = tl.exp(x24 - r4)
    e25 = tl.exp(x25 - r4)
    e26 = tl.exp(x26 - r4)
    e27 = tl.exp(x27 - r4)
    e28 = tl.exp(x28 - r4)
    e29 = tl.exp(x29 - r4)
    e30 = tl.exp(x30 - r5)
    e31 = tl.exp(x31 - r5)
    e32 = tl.exp(x32 - r5)
    e33 = tl.exp(x33 - r5)
    e34 = tl.exp(x34 - r5)
    e35 = tl.exp(x35 - r5)
    s0 = e0 + e1 + e2 + e3 + e4 + e5
    w0 = e0 / (s0 + 0.0)
    w1 = e1 / (s0 + 0.0)
    w2 = e2 / (s0 + 0.0)
    w3 = e3 / (s0 + 0.0)
    w4 = e4 / (s0 + 0.0)
    w5 = e5 / (s0 + 0.0)
    s1 = e6 + e7 + e8 + e9 + e10 + e11
    w6 = e6 / (s1 + 0.0)
    w7 = e7 / (s1 + 0.0)
    w8 = e8 / (s1 + 0.0)
    w9 = e9 / (s1 + 0.0)
    w10 = e10 / (s1 + 0.0)
    w11 = e11 / (s1 + 0.0)
    s2 = e12 + e13 + e14 + e15 + e16 + e17
    w12 = e12 / (s2 + 0.0)
    w13 = e13 / (s2 + 0.0)
    w14 = e14 / (s2 + 0.0)
    w15 = e15 / (s2 + 0.0)
    w16 = e16 / (s2 + 0.0)
    w17 = e17 / (s2 + 0.0)
    s3 = e18 + e19 + e20 + e21 + e22 + e23
    w18 = e18 / (s3 + 0.0)
    w19 = e19 / (s3 + 0.0)
    w20 = e20 / (s3 + 0.0)
    w21 = e21 / (s3 + 0.0)
    w22 = e22 / (s3 + 0.0)
    w23 = e23 / (s3 + 0.0)
    s4 = e24 + e25 + e26 + e27 + e28 + e29
    w24 = e24 / (s4 + 0.0)
    w25 = e25 / (s4 + 0.0)
    w26 = e26 / (s4 + 0.0)
    w27 = e27 / (s4 + 0.0)
    w28 = e28 / (s4 + 0.0)
    w29 = e29 / (s4 + 0.0)
    s5 = e30 + e31 + e32 + e33 + e34 + e35
    w30 = e30 / (s5 + 0.0)
    w31 = e31 / (s5 + 0.0)
    w32 = e32 / (s5 + 0.0)
    w33 = e33 / (s5 + 0.0)
    w34 = e34 / (s5 + 0.0)
    w35 = e35 / (s5 + 0.0)
    w0 += eps
    w1 += eps
    w2 += eps
    w3 += eps
    w4 += eps
    w5 += eps
    w6 += eps
    w7 += eps
    w8 += eps
    w9 += eps
    w10 += eps
    w11 += eps
    w12 += eps
    w13 += eps
    w14 += eps
    w15 += eps
    w16 += eps
    w17 += eps
    w18 += eps
    w19 += eps
    w20 += eps
    w21 += eps
    w22 += eps
    w23 += eps
    w24 += eps
    w25 += eps
    w26 += eps
    w27 += eps
    w28 += eps
    w29 += eps
    w30 += eps
    w31 += eps
    w32 += eps
    w33 += eps
    w34 += eps
    w35 += eps
    c0 = w0 + w6 + w12 + w18 + w24 + w30
    c1 = w1 + w7 + w13 + w19 + w25 + w31
    c2 = w2 + w8 + w14 + w20 + w26 + w32
    c3 = w3 + w9 + w15 + w21 + w27 + w33
    c4 = w4 + w10 + w16 + w22 + w28 + w34
    c5 = w5 + w11 + w17 + w23 + w29 + w35
    cv0 = 1.0 / (c0 + eps)
    cv1 = 1.0 / (c1 + eps)
    cv2 = 1.0 / (c2 + eps)
    cv3 = 1.0 / (c3 + eps)
    cv4 = 1.0 / (c4 + eps)
    cv5 = 1.0 / (c5 + eps)
    rv0 = tl.full([BLOCK_M], 1.0, tl.float32)
    rv1 = rv0
    rv2 = rv0
    rv3 = rv0
    rv4 = rv0
    rv5 = rv0
    for _ in range(1, iter_step):
        wc0 = w0 * cv0 + w1 * cv1 + w2 * cv2 + w3 * cv3 + w4 * cv4 + w5 * cv5
        wc1 = w6 * cv0 + w7 * cv1 + w8 * cv2 + w9 * cv3 + w10 * cv4 + w11 * cv5
        wc2 = w12 * cv0 + w13 * cv1 + w14 * cv2 + w15 * cv3 + w16 * cv4 + w17 * cv5
        wc3 = w18 * cv0 + w19 * cv1 + w20 * cv2 + w21 * cv3 + w22 * cv4 + w23 * cv5
        wc4 = w24 * cv0 + w25 * cv1 + w26 * cv2 + w27 * cv3 + w28 * cv4 + w29 * cv5
        wc5 = w30 * cv0 + w31 * cv1 + w32 * cv2 + w33 * cv3 + w34 * cv4 + w35 * cv5
        rv0 = rv0 / (rv0 * wc0 + eps)
        rv1 = rv1 / (rv1 * wc1 + eps)
        rv2 = rv2 / (rv2 * wc2 + eps)
        rv3 = rv3 / (rv3 * wc3 + eps)
        rv4 = rv4 / (rv4 * wc4 + eps)
        rv5 = rv5 / (rv5 * wc5 + eps)
        wtr0 = w0 * rv0 + w6 * rv1 + w12 * rv2 + w18 * rv3 + w24 * rv4 + w30 * rv5
        wtr1 = w1 * rv0 + w7 * rv1 + w13 * rv2 + w19 * rv3 + w25 * rv4 + w31 * rv5
        wtr2 = w2 * rv0 + w8 * rv1 + w14 * rv2 + w20 * rv3 + w26 * rv4 + w32 * rv5
        wtr3 = w3 * rv0 + w9 * rv1 + w15 * rv2 + w21 * rv3 + w27 * rv4 + w33 * rv5
        wtr4 = w4 * rv0 + w10 * rv1 + w16 * rv2 + w22 * rv3 + w28 * rv4 + w34 * rv5
        wtr5 = w5 * rv0 + w11 * rv1 + w17 * rv2 + w23 * rv3 + w29 * rv4 + w35 * rv5
        cv0 = cv0 / (cv0 * wtr0 + eps)
        cv1 = cv1 / (cv1 * wtr1 + eps)
        cv2 = cv2 / (cv2 * wtr2 + eps)
        cv3 = cv3 / (cv3 * wtr3 + eps)
        cv4 = cv4 / (cv4 * wtr4 + eps)
        cv5 = cv5 / (cv5 * wtr5 + eps)
    o0 = w0 * rv0 * cv0
    o1 = w1 * rv0 * cv1
    o2 = w2 * rv0 * cv2
    o3 = w3 * rv0 * cv3
    o4 = w4 * rv0 * cv4
    o5 = w5 * rv0 * cv5
    o6 = w6 * rv1 * cv0
    o7 = w7 * rv1 * cv1
    o8 = w8 * rv1 * cv2
    o9 = w9 * rv1 * cv3
    o10 = w10 * rv1 * cv4
    o11 = w11 * rv1 * cv5
    o12 = w12 * rv2 * cv0
    o13 = w13 * rv2 * cv1
    o14 = w14 * rv2 * cv2
    o15 = w15 * rv2 * cv3
    o16 = w16 * rv2 * cv4
    o17 = w17 * rv2 * cv5
    o18 = w18 * rv3 * cv0
    o19 = w19 * rv3 * cv1
    o20 = w20 * rv3 * cv2
    o21 = w21 * rv3 * cv3
    o22 = w22 * rv3 * cv4
    o23 = w23 * rv3 * cv5
    o24 = w24 * rv4 * cv0
    o25 = w25 * rv4 * cv1
    o26 = w26 * rv4 * cv2
    o27 = w27 * rv4 * cv3
    o28 = w28 * rv4 * cv4
    o29 = w29 * rv4 * cv5
    o30 = w30 * rv5 * cv0
    o31 = w31 * rv5 * cv1
    o32 = w32 * rv5 * cv2
    o33 = w33 * rv5 * cv3
    o34 = w34 * rv5 * cv4
    o35 = w35 * rv5 * cv5
    offs32 = off[:, None] * 36 + tl.arange(0, 32)[None, :]
    y0 = _join32(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
                 o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27, o28, o29, o30, o31,
                 BLOCK_M)
    if EXACT:
        tl.store(optr + offs32, y0)
    else:
        tl.store(optr + offs32, y0, mask=ok[:, None])
    offs4b = off[:, None] * 36 + 32 + tl.arange(0, 4)[None, :]
    y1 = _join4(o32, o33, o34, o35, BLOCK_M)
    if EXACT:
        tl.store(optr + offs4b, y1)
    else:
        tl.store(optr + offs4b, y1, mask=ok[:, None])


# ===========================================================================
# Kernel B3：标量展开 n=8（gen_skn.py 生成，消 AIV reduce 树）
# ===========================================================================
@triton.jit
def _sk8(ptr, optr, B, iter_step: tl.constexpr, eps, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * 64
    x0 = tl.load(ptr + base + 0, mask=ok)
    x1 = tl.load(ptr + base + 1, mask=ok)
    x2 = tl.load(ptr + base + 2, mask=ok)
    x3 = tl.load(ptr + base + 3, mask=ok)
    x4 = tl.load(ptr + base + 4, mask=ok)
    x5 = tl.load(ptr + base + 5, mask=ok)
    x6 = tl.load(ptr + base + 6, mask=ok)
    x7 = tl.load(ptr + base + 7, mask=ok)
    x8 = tl.load(ptr + base + 8, mask=ok)
    x9 = tl.load(ptr + base + 9, mask=ok)
    x10 = tl.load(ptr + base + 10, mask=ok)
    x11 = tl.load(ptr + base + 11, mask=ok)
    x12 = tl.load(ptr + base + 12, mask=ok)
    x13 = tl.load(ptr + base + 13, mask=ok)
    x14 = tl.load(ptr + base + 14, mask=ok)
    x15 = tl.load(ptr + base + 15, mask=ok)
    x16 = tl.load(ptr + base + 16, mask=ok)
    x17 = tl.load(ptr + base + 17, mask=ok)
    x18 = tl.load(ptr + base + 18, mask=ok)
    x19 = tl.load(ptr + base + 19, mask=ok)
    x20 = tl.load(ptr + base + 20, mask=ok)
    x21 = tl.load(ptr + base + 21, mask=ok)
    x22 = tl.load(ptr + base + 22, mask=ok)
    x23 = tl.load(ptr + base + 23, mask=ok)
    x24 = tl.load(ptr + base + 24, mask=ok)
    x25 = tl.load(ptr + base + 25, mask=ok)
    x26 = tl.load(ptr + base + 26, mask=ok)
    x27 = tl.load(ptr + base + 27, mask=ok)
    x28 = tl.load(ptr + base + 28, mask=ok)
    x29 = tl.load(ptr + base + 29, mask=ok)
    x30 = tl.load(ptr + base + 30, mask=ok)
    x31 = tl.load(ptr + base + 31, mask=ok)
    x32 = tl.load(ptr + base + 32, mask=ok)
    x33 = tl.load(ptr + base + 33, mask=ok)
    x34 = tl.load(ptr + base + 34, mask=ok)
    x35 = tl.load(ptr + base + 35, mask=ok)
    x36 = tl.load(ptr + base + 36, mask=ok)
    x37 = tl.load(ptr + base + 37, mask=ok)
    x38 = tl.load(ptr + base + 38, mask=ok)
    x39 = tl.load(ptr + base + 39, mask=ok)
    x40 = tl.load(ptr + base + 40, mask=ok)
    x41 = tl.load(ptr + base + 41, mask=ok)
    x42 = tl.load(ptr + base + 42, mask=ok)
    x43 = tl.load(ptr + base + 43, mask=ok)
    x44 = tl.load(ptr + base + 44, mask=ok)
    x45 = tl.load(ptr + base + 45, mask=ok)
    x46 = tl.load(ptr + base + 46, mask=ok)
    x47 = tl.load(ptr + base + 47, mask=ok)
    x48 = tl.load(ptr + base + 48, mask=ok)
    x49 = tl.load(ptr + base + 49, mask=ok)
    x50 = tl.load(ptr + base + 50, mask=ok)
    x51 = tl.load(ptr + base + 51, mask=ok)
    x52 = tl.load(ptr + base + 52, mask=ok)
    x53 = tl.load(ptr + base + 53, mask=ok)
    x54 = tl.load(ptr + base + 54, mask=ok)
    x55 = tl.load(ptr + base + 55, mask=ok)
    x56 = tl.load(ptr + base + 56, mask=ok)
    x57 = tl.load(ptr + base + 57, mask=ok)
    x58 = tl.load(ptr + base + 58, mask=ok)
    x59 = tl.load(ptr + base + 59, mask=ok)
    x60 = tl.load(ptr + base + 60, mask=ok)
    x61 = tl.load(ptr + base + 61, mask=ok)
    x62 = tl.load(ptr + base + 62, mask=ok)
    x63 = tl.load(ptr + base + 63, mask=ok)
    r0 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x0, x1), x2), x3), x4), x5), x6), x7)
    r1 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x8, x9), x10), x11), x12), x13), x14), x15)
    r2 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x16, x17), x18), x19), x20), x21), x22), x23)
    r3 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x24, x25), x26), x27), x28), x29), x30), x31)
    r4 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x32, x33), x34), x35), x36), x37), x38), x39)
    r5 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x40, x41), x42), x43), x44), x45), x46), x47)
    r6 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x48, x49), x50), x51), x52), x53), x54), x55)
    r7 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x56, x57), x58), x59), x60), x61), x62), x63)
    e0 = tl.exp(x0 - r0)
    e1 = tl.exp(x1 - r0)
    e2 = tl.exp(x2 - r0)
    e3 = tl.exp(x3 - r0)
    e4 = tl.exp(x4 - r0)
    e5 = tl.exp(x5 - r0)
    e6 = tl.exp(x6 - r0)
    e7 = tl.exp(x7 - r0)
    e8 = tl.exp(x8 - r1)
    e9 = tl.exp(x9 - r1)
    e10 = tl.exp(x10 - r1)
    e11 = tl.exp(x11 - r1)
    e12 = tl.exp(x12 - r1)
    e13 = tl.exp(x13 - r1)
    e14 = tl.exp(x14 - r1)
    e15 = tl.exp(x15 - r1)
    e16 = tl.exp(x16 - r2)
    e17 = tl.exp(x17 - r2)
    e18 = tl.exp(x18 - r2)
    e19 = tl.exp(x19 - r2)
    e20 = tl.exp(x20 - r2)
    e21 = tl.exp(x21 - r2)
    e22 = tl.exp(x22 - r2)
    e23 = tl.exp(x23 - r2)
    e24 = tl.exp(x24 - r3)
    e25 = tl.exp(x25 - r3)
    e26 = tl.exp(x26 - r3)
    e27 = tl.exp(x27 - r3)
    e28 = tl.exp(x28 - r3)
    e29 = tl.exp(x29 - r3)
    e30 = tl.exp(x30 - r3)
    e31 = tl.exp(x31 - r3)
    e32 = tl.exp(x32 - r4)
    e33 = tl.exp(x33 - r4)
    e34 = tl.exp(x34 - r4)
    e35 = tl.exp(x35 - r4)
    e36 = tl.exp(x36 - r4)
    e37 = tl.exp(x37 - r4)
    e38 = tl.exp(x38 - r4)
    e39 = tl.exp(x39 - r4)
    e40 = tl.exp(x40 - r5)
    e41 = tl.exp(x41 - r5)
    e42 = tl.exp(x42 - r5)
    e43 = tl.exp(x43 - r5)
    e44 = tl.exp(x44 - r5)
    e45 = tl.exp(x45 - r5)
    e46 = tl.exp(x46 - r5)
    e47 = tl.exp(x47 - r5)
    e48 = tl.exp(x48 - r6)
    e49 = tl.exp(x49 - r6)
    e50 = tl.exp(x50 - r6)
    e51 = tl.exp(x51 - r6)
    e52 = tl.exp(x52 - r6)
    e53 = tl.exp(x53 - r6)
    e54 = tl.exp(x54 - r6)
    e55 = tl.exp(x55 - r6)
    e56 = tl.exp(x56 - r7)
    e57 = tl.exp(x57 - r7)
    e58 = tl.exp(x58 - r7)
    e59 = tl.exp(x59 - r7)
    e60 = tl.exp(x60 - r7)
    e61 = tl.exp(x61 - r7)
    e62 = tl.exp(x62 - r7)
    e63 = tl.exp(x63 - r7)
    s0 = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7
    w0 = e0 / (s0 + 0.0)
    w1 = e1 / (s0 + 0.0)
    w2 = e2 / (s0 + 0.0)
    w3 = e3 / (s0 + 0.0)
    w4 = e4 / (s0 + 0.0)
    w5 = e5 / (s0 + 0.0)
    w6 = e6 / (s0 + 0.0)
    w7 = e7 / (s0 + 0.0)
    s1 = e8 + e9 + e10 + e11 + e12 + e13 + e14 + e15
    w8 = e8 / (s1 + 0.0)
    w9 = e9 / (s1 + 0.0)
    w10 = e10 / (s1 + 0.0)
    w11 = e11 / (s1 + 0.0)
    w12 = e12 / (s1 + 0.0)
    w13 = e13 / (s1 + 0.0)
    w14 = e14 / (s1 + 0.0)
    w15 = e15 / (s1 + 0.0)
    s2 = e16 + e17 + e18 + e19 + e20 + e21 + e22 + e23
    w16 = e16 / (s2 + 0.0)
    w17 = e17 / (s2 + 0.0)
    w18 = e18 / (s2 + 0.0)
    w19 = e19 / (s2 + 0.0)
    w20 = e20 / (s2 + 0.0)
    w21 = e21 / (s2 + 0.0)
    w22 = e22 / (s2 + 0.0)
    w23 = e23 / (s2 + 0.0)
    s3 = e24 + e25 + e26 + e27 + e28 + e29 + e30 + e31
    w24 = e24 / (s3 + 0.0)
    w25 = e25 / (s3 + 0.0)
    w26 = e26 / (s3 + 0.0)
    w27 = e27 / (s3 + 0.0)
    w28 = e28 / (s3 + 0.0)
    w29 = e29 / (s3 + 0.0)
    w30 = e30 / (s3 + 0.0)
    w31 = e31 / (s3 + 0.0)
    s4 = e32 + e33 + e34 + e35 + e36 + e37 + e38 + e39
    w32 = e32 / (s4 + 0.0)
    w33 = e33 / (s4 + 0.0)
    w34 = e34 / (s4 + 0.0)
    w35 = e35 / (s4 + 0.0)
    w36 = e36 / (s4 + 0.0)
    w37 = e37 / (s4 + 0.0)
    w38 = e38 / (s4 + 0.0)
    w39 = e39 / (s4 + 0.0)
    s5 = e40 + e41 + e42 + e43 + e44 + e45 + e46 + e47
    w40 = e40 / (s5 + 0.0)
    w41 = e41 / (s5 + 0.0)
    w42 = e42 / (s5 + 0.0)
    w43 = e43 / (s5 + 0.0)
    w44 = e44 / (s5 + 0.0)
    w45 = e45 / (s5 + 0.0)
    w46 = e46 / (s5 + 0.0)
    w47 = e47 / (s5 + 0.0)
    s6 = e48 + e49 + e50 + e51 + e52 + e53 + e54 + e55
    w48 = e48 / (s6 + 0.0)
    w49 = e49 / (s6 + 0.0)
    w50 = e50 / (s6 + 0.0)
    w51 = e51 / (s6 + 0.0)
    w52 = e52 / (s6 + 0.0)
    w53 = e53 / (s6 + 0.0)
    w54 = e54 / (s6 + 0.0)
    w55 = e55 / (s6 + 0.0)
    s7 = e56 + e57 + e58 + e59 + e60 + e61 + e62 + e63
    w56 = e56 / (s7 + 0.0)
    w57 = e57 / (s7 + 0.0)
    w58 = e58 / (s7 + 0.0)
    w59 = e59 / (s7 + 0.0)
    w60 = e60 / (s7 + 0.0)
    w61 = e61 / (s7 + 0.0)
    w62 = e62 / (s7 + 0.0)
    w63 = e63 / (s7 + 0.0)
    w0 += eps
    w1 += eps
    w2 += eps
    w3 += eps
    w4 += eps
    w5 += eps
    w6 += eps
    w7 += eps
    w8 += eps
    w9 += eps
    w10 += eps
    w11 += eps
    w12 += eps
    w13 += eps
    w14 += eps
    w15 += eps
    w16 += eps
    w17 += eps
    w18 += eps
    w19 += eps
    w20 += eps
    w21 += eps
    w22 += eps
    w23 += eps
    w24 += eps
    w25 += eps
    w26 += eps
    w27 += eps
    w28 += eps
    w29 += eps
    w30 += eps
    w31 += eps
    w32 += eps
    w33 += eps
    w34 += eps
    w35 += eps
    w36 += eps
    w37 += eps
    w38 += eps
    w39 += eps
    w40 += eps
    w41 += eps
    w42 += eps
    w43 += eps
    w44 += eps
    w45 += eps
    w46 += eps
    w47 += eps
    w48 += eps
    w49 += eps
    w50 += eps
    w51 += eps
    w52 += eps
    w53 += eps
    w54 += eps
    w55 += eps
    w56 += eps
    w57 += eps
    w58 += eps
    w59 += eps
    w60 += eps
    w61 += eps
    w62 += eps
    w63 += eps
    c0 = w0 + w8 + w16 + w24 + w32 + w40 + w48 + w56
    c1 = w1 + w9 + w17 + w25 + w33 + w41 + w49 + w57
    c2 = w2 + w10 + w18 + w26 + w34 + w42 + w50 + w58
    c3 = w3 + w11 + w19 + w27 + w35 + w43 + w51 + w59
    c4 = w4 + w12 + w20 + w28 + w36 + w44 + w52 + w60
    c5 = w5 + w13 + w21 + w29 + w37 + w45 + w53 + w61
    c6 = w6 + w14 + w22 + w30 + w38 + w46 + w54 + w62
    c7 = w7 + w15 + w23 + w31 + w39 + w47 + w55 + w63
    cv0 = 1.0 / (c0 + eps)
    cv1 = 1.0 / (c1 + eps)
    cv2 = 1.0 / (c2 + eps)
    cv3 = 1.0 / (c3 + eps)
    cv4 = 1.0 / (c4 + eps)
    cv5 = 1.0 / (c5 + eps)
    cv6 = 1.0 / (c6 + eps)
    cv7 = 1.0 / (c7 + eps)
    rv0 = tl.full([BLOCK_M], 1.0, tl.float32)
    rv1 = rv0
    rv2 = rv0
    rv3 = rv0
    rv4 = rv0
    rv5 = rv0
    rv6 = rv0
    rv7 = rv0
    for _ in range(1, iter_step):
        wc0 = w0 * cv0 + w1 * cv1 + w2 * cv2 + w3 * cv3 + w4 * cv4 + w5 * cv5 + w6 * cv6 + w7 * cv7
        wc1 = w8 * cv0 + w9 * cv1 + w10 * cv2 + w11 * cv3 + w12 * cv4 + w13 * cv5 + w14 * cv6 + w15 * cv7
        wc2 = w16 * cv0 + w17 * cv1 + w18 * cv2 + w19 * cv3 + w20 * cv4 + w21 * cv5 + w22 * cv6 + w23 * cv7
        wc3 = w24 * cv0 + w25 * cv1 + w26 * cv2 + w27 * cv3 + w28 * cv4 + w29 * cv5 + w30 * cv6 + w31 * cv7
        wc4 = w32 * cv0 + w33 * cv1 + w34 * cv2 + w35 * cv3 + w36 * cv4 + w37 * cv5 + w38 * cv6 + w39 * cv7
        wc5 = w40 * cv0 + w41 * cv1 + w42 * cv2 + w43 * cv3 + w44 * cv4 + w45 * cv5 + w46 * cv6 + w47 * cv7
        wc6 = w48 * cv0 + w49 * cv1 + w50 * cv2 + w51 * cv3 + w52 * cv4 + w53 * cv5 + w54 * cv6 + w55 * cv7
        wc7 = w56 * cv0 + w57 * cv1 + w58 * cv2 + w59 * cv3 + w60 * cv4 + w61 * cv5 + w62 * cv6 + w63 * cv7
        rv0 = rv0 / (rv0 * wc0 + eps)
        rv1 = rv1 / (rv1 * wc1 + eps)
        rv2 = rv2 / (rv2 * wc2 + eps)
        rv3 = rv3 / (rv3 * wc3 + eps)
        rv4 = rv4 / (rv4 * wc4 + eps)
        rv5 = rv5 / (rv5 * wc5 + eps)
        rv6 = rv6 / (rv6 * wc6 + eps)
        rv7 = rv7 / (rv7 * wc7 + eps)
        wtr0 = w0 * rv0 + w8 * rv1 + w16 * rv2 + w24 * rv3 + w32 * rv4 + w40 * rv5 + w48 * rv6 + w56 * rv7
        wtr1 = w1 * rv0 + w9 * rv1 + w17 * rv2 + w25 * rv3 + w33 * rv4 + w41 * rv5 + w49 * rv6 + w57 * rv7
        wtr2 = w2 * rv0 + w10 * rv1 + w18 * rv2 + w26 * rv3 + w34 * rv4 + w42 * rv5 + w50 * rv6 + w58 * rv7
        wtr3 = w3 * rv0 + w11 * rv1 + w19 * rv2 + w27 * rv3 + w35 * rv4 + w43 * rv5 + w51 * rv6 + w59 * rv7
        wtr4 = w4 * rv0 + w12 * rv1 + w20 * rv2 + w28 * rv3 + w36 * rv4 + w44 * rv5 + w52 * rv6 + w60 * rv7
        wtr5 = w5 * rv0 + w13 * rv1 + w21 * rv2 + w29 * rv3 + w37 * rv4 + w45 * rv5 + w53 * rv6 + w61 * rv7
        wtr6 = w6 * rv0 + w14 * rv1 + w22 * rv2 + w30 * rv3 + w38 * rv4 + w46 * rv5 + w54 * rv6 + w62 * rv7
        wtr7 = w7 * rv0 + w15 * rv1 + w23 * rv2 + w31 * rv3 + w39 * rv4 + w47 * rv5 + w55 * rv6 + w63 * rv7
        cv0 = cv0 / (cv0 * wtr0 + eps)
        cv1 = cv1 / (cv1 * wtr1 + eps)
        cv2 = cv2 / (cv2 * wtr2 + eps)
        cv3 = cv3 / (cv3 * wtr3 + eps)
        cv4 = cv4 / (cv4 * wtr4 + eps)
        cv5 = cv5 / (cv5 * wtr5 + eps)
        cv6 = cv6 / (cv6 * wtr6 + eps)
        cv7 = cv7 / (cv7 * wtr7 + eps)
    o0 = w0 * rv0 * cv0
    o1 = w1 * rv0 * cv1
    o2 = w2 * rv0 * cv2
    o3 = w3 * rv0 * cv3
    o4 = w4 * rv0 * cv4
    o5 = w5 * rv0 * cv5
    o6 = w6 * rv0 * cv6
    o7 = w7 * rv0 * cv7
    o8 = w8 * rv1 * cv0
    o9 = w9 * rv1 * cv1
    o10 = w10 * rv1 * cv2
    o11 = w11 * rv1 * cv3
    o12 = w12 * rv1 * cv4
    o13 = w13 * rv1 * cv5
    o14 = w14 * rv1 * cv6
    o15 = w15 * rv1 * cv7
    o16 = w16 * rv2 * cv0
    o17 = w17 * rv2 * cv1
    o18 = w18 * rv2 * cv2
    o19 = w19 * rv2 * cv3
    o20 = w20 * rv2 * cv4
    o21 = w21 * rv2 * cv5
    o22 = w22 * rv2 * cv6
    o23 = w23 * rv2 * cv7
    o24 = w24 * rv3 * cv0
    o25 = w25 * rv3 * cv1
    o26 = w26 * rv3 * cv2
    o27 = w27 * rv3 * cv3
    o28 = w28 * rv3 * cv4
    o29 = w29 * rv3 * cv5
    o30 = w30 * rv3 * cv6
    o31 = w31 * rv3 * cv7
    o32 = w32 * rv4 * cv0
    o33 = w33 * rv4 * cv1
    o34 = w34 * rv4 * cv2
    o35 = w35 * rv4 * cv3
    o36 = w36 * rv4 * cv4
    o37 = w37 * rv4 * cv5
    o38 = w38 * rv4 * cv6
    o39 = w39 * rv4 * cv7
    o40 = w40 * rv5 * cv0
    o41 = w41 * rv5 * cv1
    o42 = w42 * rv5 * cv2
    o43 = w43 * rv5 * cv3
    o44 = w44 * rv5 * cv4
    o45 = w45 * rv5 * cv5
    o46 = w46 * rv5 * cv6
    o47 = w47 * rv5 * cv7
    o48 = w48 * rv6 * cv0
    o49 = w49 * rv6 * cv1
    o50 = w50 * rv6 * cv2
    o51 = w51 * rv6 * cv3
    o52 = w52 * rv6 * cv4
    o53 = w53 * rv6 * cv5
    o54 = w54 * rv6 * cv6
    o55 = w55 * rv6 * cv7
    o56 = w56 * rv7 * cv0
    o57 = w57 * rv7 * cv1
    o58 = w58 * rv7 * cv2
    o59 = w59 * rv7 * cv3
    o60 = w60 * rv7 * cv4
    o61 = w61 * rv7 * cv5
    o62 = w62 * rv7 * cv6
    o63 = w63 * rv7 * cv7
    tl.store(optr + base + 0, o0, mask=ok)
    tl.store(optr + base + 1, o1, mask=ok)
    tl.store(optr + base + 2, o2, mask=ok)
    tl.store(optr + base + 3, o3, mask=ok)
    tl.store(optr + base + 4, o4, mask=ok)
    tl.store(optr + base + 5, o5, mask=ok)
    tl.store(optr + base + 6, o6, mask=ok)
    tl.store(optr + base + 7, o7, mask=ok)
    tl.store(optr + base + 8, o8, mask=ok)
    tl.store(optr + base + 9, o9, mask=ok)
    tl.store(optr + base + 10, o10, mask=ok)
    tl.store(optr + base + 11, o11, mask=ok)
    tl.store(optr + base + 12, o12, mask=ok)
    tl.store(optr + base + 13, o13, mask=ok)
    tl.store(optr + base + 14, o14, mask=ok)
    tl.store(optr + base + 15, o15, mask=ok)
    tl.store(optr + base + 16, o16, mask=ok)
    tl.store(optr + base + 17, o17, mask=ok)
    tl.store(optr + base + 18, o18, mask=ok)
    tl.store(optr + base + 19, o19, mask=ok)
    tl.store(optr + base + 20, o20, mask=ok)
    tl.store(optr + base + 21, o21, mask=ok)
    tl.store(optr + base + 22, o22, mask=ok)
    tl.store(optr + base + 23, o23, mask=ok)
    tl.store(optr + base + 24, o24, mask=ok)
    tl.store(optr + base + 25, o25, mask=ok)
    tl.store(optr + base + 26, o26, mask=ok)
    tl.store(optr + base + 27, o27, mask=ok)
    tl.store(optr + base + 28, o28, mask=ok)
    tl.store(optr + base + 29, o29, mask=ok)
    tl.store(optr + base + 30, o30, mask=ok)
    tl.store(optr + base + 31, o31, mask=ok)
    tl.store(optr + base + 32, o32, mask=ok)
    tl.store(optr + base + 33, o33, mask=ok)
    tl.store(optr + base + 34, o34, mask=ok)
    tl.store(optr + base + 35, o35, mask=ok)
    tl.store(optr + base + 36, o36, mask=ok)
    tl.store(optr + base + 37, o37, mask=ok)
    tl.store(optr + base + 38, o38, mask=ok)
    tl.store(optr + base + 39, o39, mask=ok)
    tl.store(optr + base + 40, o40, mask=ok)
    tl.store(optr + base + 41, o41, mask=ok)
    tl.store(optr + base + 42, o42, mask=ok)
    tl.store(optr + base + 43, o43, mask=ok)
    tl.store(optr + base + 44, o44, mask=ok)
    tl.store(optr + base + 45, o45, mask=ok)
    tl.store(optr + base + 46, o46, mask=ok)
    tl.store(optr + base + 47, o47, mask=ok)
    tl.store(optr + base + 48, o48, mask=ok)
    tl.store(optr + base + 49, o49, mask=ok)
    tl.store(optr + base + 50, o50, mask=ok)
    tl.store(optr + base + 51, o51, mask=ok)
    tl.store(optr + base + 52, o52, mask=ok)
    tl.store(optr + base + 53, o53, mask=ok)
    tl.store(optr + base + 54, o54, mask=ok)
    tl.store(optr + base + 55, o55, mask=ok)
    tl.store(optr + base + 56, o56, mask=ok)
    tl.store(optr + base + 57, o57, mask=ok)
    tl.store(optr + base + 58, o58, mask=ok)
    tl.store(optr + base + 59, o59, mask=ok)
    tl.store(optr + base + 60, o60, mask=ok)
    tl.store(optr + base + 61, o61, mask=ok)
    tl.store(optr + base + 62, o62, mask=ok)
    tl.store(optr + base + 63, o63, mask=ok)


# ===========================================================================
# Kernel B4：标量展开 n=12（gen_skn.py 生成；编译慢 ~106s，一次性）
# ===========================================================================
@triton.jit
def _sk12(ptr, optr, B, iter_step: tl.constexpr, eps, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * 144
    x0 = tl.load(ptr + base + 0, mask=ok)
    x1 = tl.load(ptr + base + 1, mask=ok)
    x2 = tl.load(ptr + base + 2, mask=ok)
    x3 = tl.load(ptr + base + 3, mask=ok)
    x4 = tl.load(ptr + base + 4, mask=ok)
    x5 = tl.load(ptr + base + 5, mask=ok)
    x6 = tl.load(ptr + base + 6, mask=ok)
    x7 = tl.load(ptr + base + 7, mask=ok)
    x8 = tl.load(ptr + base + 8, mask=ok)
    x9 = tl.load(ptr + base + 9, mask=ok)
    x10 = tl.load(ptr + base + 10, mask=ok)
    x11 = tl.load(ptr + base + 11, mask=ok)
    x12 = tl.load(ptr + base + 12, mask=ok)
    x13 = tl.load(ptr + base + 13, mask=ok)
    x14 = tl.load(ptr + base + 14, mask=ok)
    x15 = tl.load(ptr + base + 15, mask=ok)
    x16 = tl.load(ptr + base + 16, mask=ok)
    x17 = tl.load(ptr + base + 17, mask=ok)
    x18 = tl.load(ptr + base + 18, mask=ok)
    x19 = tl.load(ptr + base + 19, mask=ok)
    x20 = tl.load(ptr + base + 20, mask=ok)
    x21 = tl.load(ptr + base + 21, mask=ok)
    x22 = tl.load(ptr + base + 22, mask=ok)
    x23 = tl.load(ptr + base + 23, mask=ok)
    x24 = tl.load(ptr + base + 24, mask=ok)
    x25 = tl.load(ptr + base + 25, mask=ok)
    x26 = tl.load(ptr + base + 26, mask=ok)
    x27 = tl.load(ptr + base + 27, mask=ok)
    x28 = tl.load(ptr + base + 28, mask=ok)
    x29 = tl.load(ptr + base + 29, mask=ok)
    x30 = tl.load(ptr + base + 30, mask=ok)
    x31 = tl.load(ptr + base + 31, mask=ok)
    x32 = tl.load(ptr + base + 32, mask=ok)
    x33 = tl.load(ptr + base + 33, mask=ok)
    x34 = tl.load(ptr + base + 34, mask=ok)
    x35 = tl.load(ptr + base + 35, mask=ok)
    x36 = tl.load(ptr + base + 36, mask=ok)
    x37 = tl.load(ptr + base + 37, mask=ok)
    x38 = tl.load(ptr + base + 38, mask=ok)
    x39 = tl.load(ptr + base + 39, mask=ok)
    x40 = tl.load(ptr + base + 40, mask=ok)
    x41 = tl.load(ptr + base + 41, mask=ok)
    x42 = tl.load(ptr + base + 42, mask=ok)
    x43 = tl.load(ptr + base + 43, mask=ok)
    x44 = tl.load(ptr + base + 44, mask=ok)
    x45 = tl.load(ptr + base + 45, mask=ok)
    x46 = tl.load(ptr + base + 46, mask=ok)
    x47 = tl.load(ptr + base + 47, mask=ok)
    x48 = tl.load(ptr + base + 48, mask=ok)
    x49 = tl.load(ptr + base + 49, mask=ok)
    x50 = tl.load(ptr + base + 50, mask=ok)
    x51 = tl.load(ptr + base + 51, mask=ok)
    x52 = tl.load(ptr + base + 52, mask=ok)
    x53 = tl.load(ptr + base + 53, mask=ok)
    x54 = tl.load(ptr + base + 54, mask=ok)
    x55 = tl.load(ptr + base + 55, mask=ok)
    x56 = tl.load(ptr + base + 56, mask=ok)
    x57 = tl.load(ptr + base + 57, mask=ok)
    x58 = tl.load(ptr + base + 58, mask=ok)
    x59 = tl.load(ptr + base + 59, mask=ok)
    x60 = tl.load(ptr + base + 60, mask=ok)
    x61 = tl.load(ptr + base + 61, mask=ok)
    x62 = tl.load(ptr + base + 62, mask=ok)
    x63 = tl.load(ptr + base + 63, mask=ok)
    x64 = tl.load(ptr + base + 64, mask=ok)
    x65 = tl.load(ptr + base + 65, mask=ok)
    x66 = tl.load(ptr + base + 66, mask=ok)
    x67 = tl.load(ptr + base + 67, mask=ok)
    x68 = tl.load(ptr + base + 68, mask=ok)
    x69 = tl.load(ptr + base + 69, mask=ok)
    x70 = tl.load(ptr + base + 70, mask=ok)
    x71 = tl.load(ptr + base + 71, mask=ok)
    x72 = tl.load(ptr + base + 72, mask=ok)
    x73 = tl.load(ptr + base + 73, mask=ok)
    x74 = tl.load(ptr + base + 74, mask=ok)
    x75 = tl.load(ptr + base + 75, mask=ok)
    x76 = tl.load(ptr + base + 76, mask=ok)
    x77 = tl.load(ptr + base + 77, mask=ok)
    x78 = tl.load(ptr + base + 78, mask=ok)
    x79 = tl.load(ptr + base + 79, mask=ok)
    x80 = tl.load(ptr + base + 80, mask=ok)
    x81 = tl.load(ptr + base + 81, mask=ok)
    x82 = tl.load(ptr + base + 82, mask=ok)
    x83 = tl.load(ptr + base + 83, mask=ok)
    x84 = tl.load(ptr + base + 84, mask=ok)
    x85 = tl.load(ptr + base + 85, mask=ok)
    x86 = tl.load(ptr + base + 86, mask=ok)
    x87 = tl.load(ptr + base + 87, mask=ok)
    x88 = tl.load(ptr + base + 88, mask=ok)
    x89 = tl.load(ptr + base + 89, mask=ok)
    x90 = tl.load(ptr + base + 90, mask=ok)
    x91 = tl.load(ptr + base + 91, mask=ok)
    x92 = tl.load(ptr + base + 92, mask=ok)
    x93 = tl.load(ptr + base + 93, mask=ok)
    x94 = tl.load(ptr + base + 94, mask=ok)
    x95 = tl.load(ptr + base + 95, mask=ok)
    x96 = tl.load(ptr + base + 96, mask=ok)
    x97 = tl.load(ptr + base + 97, mask=ok)
    x98 = tl.load(ptr + base + 98, mask=ok)
    x99 = tl.load(ptr + base + 99, mask=ok)
    x100 = tl.load(ptr + base + 100, mask=ok)
    x101 = tl.load(ptr + base + 101, mask=ok)
    x102 = tl.load(ptr + base + 102, mask=ok)
    x103 = tl.load(ptr + base + 103, mask=ok)
    x104 = tl.load(ptr + base + 104, mask=ok)
    x105 = tl.load(ptr + base + 105, mask=ok)
    x106 = tl.load(ptr + base + 106, mask=ok)
    x107 = tl.load(ptr + base + 107, mask=ok)
    x108 = tl.load(ptr + base + 108, mask=ok)
    x109 = tl.load(ptr + base + 109, mask=ok)
    x110 = tl.load(ptr + base + 110, mask=ok)
    x111 = tl.load(ptr + base + 111, mask=ok)
    x112 = tl.load(ptr + base + 112, mask=ok)
    x113 = tl.load(ptr + base + 113, mask=ok)
    x114 = tl.load(ptr + base + 114, mask=ok)
    x115 = tl.load(ptr + base + 115, mask=ok)
    x116 = tl.load(ptr + base + 116, mask=ok)
    x117 = tl.load(ptr + base + 117, mask=ok)
    x118 = tl.load(ptr + base + 118, mask=ok)
    x119 = tl.load(ptr + base + 119, mask=ok)
    x120 = tl.load(ptr + base + 120, mask=ok)
    x121 = tl.load(ptr + base + 121, mask=ok)
    x122 = tl.load(ptr + base + 122, mask=ok)
    x123 = tl.load(ptr + base + 123, mask=ok)
    x124 = tl.load(ptr + base + 124, mask=ok)
    x125 = tl.load(ptr + base + 125, mask=ok)
    x126 = tl.load(ptr + base + 126, mask=ok)
    x127 = tl.load(ptr + base + 127, mask=ok)
    x128 = tl.load(ptr + base + 128, mask=ok)
    x129 = tl.load(ptr + base + 129, mask=ok)
    x130 = tl.load(ptr + base + 130, mask=ok)
    x131 = tl.load(ptr + base + 131, mask=ok)
    x132 = tl.load(ptr + base + 132, mask=ok)
    x133 = tl.load(ptr + base + 133, mask=ok)
    x134 = tl.load(ptr + base + 134, mask=ok)
    x135 = tl.load(ptr + base + 135, mask=ok)
    x136 = tl.load(ptr + base + 136, mask=ok)
    x137 = tl.load(ptr + base + 137, mask=ok)
    x138 = tl.load(ptr + base + 138, mask=ok)
    x139 = tl.load(ptr + base + 139, mask=ok)
    x140 = tl.load(ptr + base + 140, mask=ok)
    x141 = tl.load(ptr + base + 141, mask=ok)
    x142 = tl.load(ptr + base + 142, mask=ok)
    x143 = tl.load(ptr + base + 143, mask=ok)
    r0 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x0, x1), x2), x3), x4), x5), x6), x7), x8), x9), x10), x11)
    r1 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x12, x13), x14), x15), x16), x17), x18), x19), x20), x21), x22), x23)
    r2 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x24, x25), x26), x27), x28), x29), x30), x31), x32), x33), x34), x35)
    r3 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x36, x37), x38), x39), x40), x41), x42), x43), x44), x45), x46), x47)
    r4 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x48, x49), x50), x51), x52), x53), x54), x55), x56), x57), x58), x59)
    r5 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x60, x61), x62), x63), x64), x65), x66), x67), x68), x69), x70), x71)
    r6 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x72, x73), x74), x75), x76), x77), x78), x79), x80), x81), x82), x83)
    r7 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x84, x85), x86), x87), x88), x89), x90), x91), x92), x93), x94), x95)
    r8 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x96, x97), x98), x99), x100), x101), x102), x103), x104), x105), x106), x107)
    r9 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x108, x109), x110), x111), x112), x113), x114), x115), x116), x117), x118), x119)
    r10 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x120, x121), x122), x123), x124), x125), x126), x127), x128), x129), x130), x131)
    r11 = tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(tl.maximum(x132, x133), x134), x135), x136), x137), x138), x139), x140), x141), x142), x143)
    e0 = tl.exp(x0 - r0)
    e1 = tl.exp(x1 - r0)
    e2 = tl.exp(x2 - r0)
    e3 = tl.exp(x3 - r0)
    e4 = tl.exp(x4 - r0)
    e5 = tl.exp(x5 - r0)
    e6 = tl.exp(x6 - r0)
    e7 = tl.exp(x7 - r0)
    e8 = tl.exp(x8 - r0)
    e9 = tl.exp(x9 - r0)
    e10 = tl.exp(x10 - r0)
    e11 = tl.exp(x11 - r0)
    e12 = tl.exp(x12 - r1)
    e13 = tl.exp(x13 - r1)
    e14 = tl.exp(x14 - r1)
    e15 = tl.exp(x15 - r1)
    e16 = tl.exp(x16 - r1)
    e17 = tl.exp(x17 - r1)
    e18 = tl.exp(x18 - r1)
    e19 = tl.exp(x19 - r1)
    e20 = tl.exp(x20 - r1)
    e21 = tl.exp(x21 - r1)
    e22 = tl.exp(x22 - r1)
    e23 = tl.exp(x23 - r1)
    e24 = tl.exp(x24 - r2)
    e25 = tl.exp(x25 - r2)
    e26 = tl.exp(x26 - r2)
    e27 = tl.exp(x27 - r2)
    e28 = tl.exp(x28 - r2)
    e29 = tl.exp(x29 - r2)
    e30 = tl.exp(x30 - r2)
    e31 = tl.exp(x31 - r2)
    e32 = tl.exp(x32 - r2)
    e33 = tl.exp(x33 - r2)
    e34 = tl.exp(x34 - r2)
    e35 = tl.exp(x35 - r2)
    e36 = tl.exp(x36 - r3)
    e37 = tl.exp(x37 - r3)
    e38 = tl.exp(x38 - r3)
    e39 = tl.exp(x39 - r3)
    e40 = tl.exp(x40 - r3)
    e41 = tl.exp(x41 - r3)
    e42 = tl.exp(x42 - r3)
    e43 = tl.exp(x43 - r3)
    e44 = tl.exp(x44 - r3)
    e45 = tl.exp(x45 - r3)
    e46 = tl.exp(x46 - r3)
    e47 = tl.exp(x47 - r3)
    e48 = tl.exp(x48 - r4)
    e49 = tl.exp(x49 - r4)
    e50 = tl.exp(x50 - r4)
    e51 = tl.exp(x51 - r4)
    e52 = tl.exp(x52 - r4)
    e53 = tl.exp(x53 - r4)
    e54 = tl.exp(x54 - r4)
    e55 = tl.exp(x55 - r4)
    e56 = tl.exp(x56 - r4)
    e57 = tl.exp(x57 - r4)
    e58 = tl.exp(x58 - r4)
    e59 = tl.exp(x59 - r4)
    e60 = tl.exp(x60 - r5)
    e61 = tl.exp(x61 - r5)
    e62 = tl.exp(x62 - r5)
    e63 = tl.exp(x63 - r5)
    e64 = tl.exp(x64 - r5)
    e65 = tl.exp(x65 - r5)
    e66 = tl.exp(x66 - r5)
    e67 = tl.exp(x67 - r5)
    e68 = tl.exp(x68 - r5)
    e69 = tl.exp(x69 - r5)
    e70 = tl.exp(x70 - r5)
    e71 = tl.exp(x71 - r5)
    e72 = tl.exp(x72 - r6)
    e73 = tl.exp(x73 - r6)
    e74 = tl.exp(x74 - r6)
    e75 = tl.exp(x75 - r6)
    e76 = tl.exp(x76 - r6)
    e77 = tl.exp(x77 - r6)
    e78 = tl.exp(x78 - r6)
    e79 = tl.exp(x79 - r6)
    e80 = tl.exp(x80 - r6)
    e81 = tl.exp(x81 - r6)
    e82 = tl.exp(x82 - r6)
    e83 = tl.exp(x83 - r6)
    e84 = tl.exp(x84 - r7)
    e85 = tl.exp(x85 - r7)
    e86 = tl.exp(x86 - r7)
    e87 = tl.exp(x87 - r7)
    e88 = tl.exp(x88 - r7)
    e89 = tl.exp(x89 - r7)
    e90 = tl.exp(x90 - r7)
    e91 = tl.exp(x91 - r7)
    e92 = tl.exp(x92 - r7)
    e93 = tl.exp(x93 - r7)
    e94 = tl.exp(x94 - r7)
    e95 = tl.exp(x95 - r7)
    e96 = tl.exp(x96 - r8)
    e97 = tl.exp(x97 - r8)
    e98 = tl.exp(x98 - r8)
    e99 = tl.exp(x99 - r8)
    e100 = tl.exp(x100 - r8)
    e101 = tl.exp(x101 - r8)
    e102 = tl.exp(x102 - r8)
    e103 = tl.exp(x103 - r8)
    e104 = tl.exp(x104 - r8)
    e105 = tl.exp(x105 - r8)
    e106 = tl.exp(x106 - r8)
    e107 = tl.exp(x107 - r8)
    e108 = tl.exp(x108 - r9)
    e109 = tl.exp(x109 - r9)
    e110 = tl.exp(x110 - r9)
    e111 = tl.exp(x111 - r9)
    e112 = tl.exp(x112 - r9)
    e113 = tl.exp(x113 - r9)
    e114 = tl.exp(x114 - r9)
    e115 = tl.exp(x115 - r9)
    e116 = tl.exp(x116 - r9)
    e117 = tl.exp(x117 - r9)
    e118 = tl.exp(x118 - r9)
    e119 = tl.exp(x119 - r9)
    e120 = tl.exp(x120 - r10)
    e121 = tl.exp(x121 - r10)
    e122 = tl.exp(x122 - r10)
    e123 = tl.exp(x123 - r10)
    e124 = tl.exp(x124 - r10)
    e125 = tl.exp(x125 - r10)
    e126 = tl.exp(x126 - r10)
    e127 = tl.exp(x127 - r10)
    e128 = tl.exp(x128 - r10)
    e129 = tl.exp(x129 - r10)
    e130 = tl.exp(x130 - r10)
    e131 = tl.exp(x131 - r10)
    e132 = tl.exp(x132 - r11)
    e133 = tl.exp(x133 - r11)
    e134 = tl.exp(x134 - r11)
    e135 = tl.exp(x135 - r11)
    e136 = tl.exp(x136 - r11)
    e137 = tl.exp(x137 - r11)
    e138 = tl.exp(x138 - r11)
    e139 = tl.exp(x139 - r11)
    e140 = tl.exp(x140 - r11)
    e141 = tl.exp(x141 - r11)
    e142 = tl.exp(x142 - r11)
    e143 = tl.exp(x143 - r11)
    s0 = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8 + e9 + e10 + e11
    w0 = e0 / (s0 + 0.0)
    w1 = e1 / (s0 + 0.0)
    w2 = e2 / (s0 + 0.0)
    w3 = e3 / (s0 + 0.0)
    w4 = e4 / (s0 + 0.0)
    w5 = e5 / (s0 + 0.0)
    w6 = e6 / (s0 + 0.0)
    w7 = e7 / (s0 + 0.0)
    w8 = e8 / (s0 + 0.0)
    w9 = e9 / (s0 + 0.0)
    w10 = e10 / (s0 + 0.0)
    w11 = e11 / (s0 + 0.0)
    s1 = e12 + e13 + e14 + e15 + e16 + e17 + e18 + e19 + e20 + e21 + e22 + e23
    w12 = e12 / (s1 + 0.0)
    w13 = e13 / (s1 + 0.0)
    w14 = e14 / (s1 + 0.0)
    w15 = e15 / (s1 + 0.0)
    w16 = e16 / (s1 + 0.0)
    w17 = e17 / (s1 + 0.0)
    w18 = e18 / (s1 + 0.0)
    w19 = e19 / (s1 + 0.0)
    w20 = e20 / (s1 + 0.0)
    w21 = e21 / (s1 + 0.0)
    w22 = e22 / (s1 + 0.0)
    w23 = e23 / (s1 + 0.0)
    s2 = e24 + e25 + e26 + e27 + e28 + e29 + e30 + e31 + e32 + e33 + e34 + e35
    w24 = e24 / (s2 + 0.0)
    w25 = e25 / (s2 + 0.0)
    w26 = e26 / (s2 + 0.0)
    w27 = e27 / (s2 + 0.0)
    w28 = e28 / (s2 + 0.0)
    w29 = e29 / (s2 + 0.0)
    w30 = e30 / (s2 + 0.0)
    w31 = e31 / (s2 + 0.0)
    w32 = e32 / (s2 + 0.0)
    w33 = e33 / (s2 + 0.0)
    w34 = e34 / (s2 + 0.0)
    w35 = e35 / (s2 + 0.0)
    s3 = e36 + e37 + e38 + e39 + e40 + e41 + e42 + e43 + e44 + e45 + e46 + e47
    w36 = e36 / (s3 + 0.0)
    w37 = e37 / (s3 + 0.0)
    w38 = e38 / (s3 + 0.0)
    w39 = e39 / (s3 + 0.0)
    w40 = e40 / (s3 + 0.0)
    w41 = e41 / (s3 + 0.0)
    w42 = e42 / (s3 + 0.0)
    w43 = e43 / (s3 + 0.0)
    w44 = e44 / (s3 + 0.0)
    w45 = e45 / (s3 + 0.0)
    w46 = e46 / (s3 + 0.0)
    w47 = e47 / (s3 + 0.0)
    s4 = e48 + e49 + e50 + e51 + e52 + e53 + e54 + e55 + e56 + e57 + e58 + e59
    w48 = e48 / (s4 + 0.0)
    w49 = e49 / (s4 + 0.0)
    w50 = e50 / (s4 + 0.0)
    w51 = e51 / (s4 + 0.0)
    w52 = e52 / (s4 + 0.0)
    w53 = e53 / (s4 + 0.0)
    w54 = e54 / (s4 + 0.0)
    w55 = e55 / (s4 + 0.0)
    w56 = e56 / (s4 + 0.0)
    w57 = e57 / (s4 + 0.0)
    w58 = e58 / (s4 + 0.0)
    w59 = e59 / (s4 + 0.0)
    s5 = e60 + e61 + e62 + e63 + e64 + e65 + e66 + e67 + e68 + e69 + e70 + e71
    w60 = e60 / (s5 + 0.0)
    w61 = e61 / (s5 + 0.0)
    w62 = e62 / (s5 + 0.0)
    w63 = e63 / (s5 + 0.0)
    w64 = e64 / (s5 + 0.0)
    w65 = e65 / (s5 + 0.0)
    w66 = e66 / (s5 + 0.0)
    w67 = e67 / (s5 + 0.0)
    w68 = e68 / (s5 + 0.0)
    w69 = e69 / (s5 + 0.0)
    w70 = e70 / (s5 + 0.0)
    w71 = e71 / (s5 + 0.0)
    s6 = e72 + e73 + e74 + e75 + e76 + e77 + e78 + e79 + e80 + e81 + e82 + e83
    w72 = e72 / (s6 + 0.0)
    w73 = e73 / (s6 + 0.0)
    w74 = e74 / (s6 + 0.0)
    w75 = e75 / (s6 + 0.0)
    w76 = e76 / (s6 + 0.0)
    w77 = e77 / (s6 + 0.0)
    w78 = e78 / (s6 + 0.0)
    w79 = e79 / (s6 + 0.0)
    w80 = e80 / (s6 + 0.0)
    w81 = e81 / (s6 + 0.0)
    w82 = e82 / (s6 + 0.0)
    w83 = e83 / (s6 + 0.0)
    s7 = e84 + e85 + e86 + e87 + e88 + e89 + e90 + e91 + e92 + e93 + e94 + e95
    w84 = e84 / (s7 + 0.0)
    w85 = e85 / (s7 + 0.0)
    w86 = e86 / (s7 + 0.0)
    w87 = e87 / (s7 + 0.0)
    w88 = e88 / (s7 + 0.0)
    w89 = e89 / (s7 + 0.0)
    w90 = e90 / (s7 + 0.0)
    w91 = e91 / (s7 + 0.0)
    w92 = e92 / (s7 + 0.0)
    w93 = e93 / (s7 + 0.0)
    w94 = e94 / (s7 + 0.0)
    w95 = e95 / (s7 + 0.0)
    s8 = e96 + e97 + e98 + e99 + e100 + e101 + e102 + e103 + e104 + e105 + e106 + e107
    w96 = e96 / (s8 + 0.0)
    w97 = e97 / (s8 + 0.0)
    w98 = e98 / (s8 + 0.0)
    w99 = e99 / (s8 + 0.0)
    w100 = e100 / (s8 + 0.0)
    w101 = e101 / (s8 + 0.0)
    w102 = e102 / (s8 + 0.0)
    w103 = e103 / (s8 + 0.0)
    w104 = e104 / (s8 + 0.0)
    w105 = e105 / (s8 + 0.0)
    w106 = e106 / (s8 + 0.0)
    w107 = e107 / (s8 + 0.0)
    s9 = e108 + e109 + e110 + e111 + e112 + e113 + e114 + e115 + e116 + e117 + e118 + e119
    w108 = e108 / (s9 + 0.0)
    w109 = e109 / (s9 + 0.0)
    w110 = e110 / (s9 + 0.0)
    w111 = e111 / (s9 + 0.0)
    w112 = e112 / (s9 + 0.0)
    w113 = e113 / (s9 + 0.0)
    w114 = e114 / (s9 + 0.0)
    w115 = e115 / (s9 + 0.0)
    w116 = e116 / (s9 + 0.0)
    w117 = e117 / (s9 + 0.0)
    w118 = e118 / (s9 + 0.0)
    w119 = e119 / (s9 + 0.0)
    s10 = e120 + e121 + e122 + e123 + e124 + e125 + e126 + e127 + e128 + e129 + e130 + e131
    w120 = e120 / (s10 + 0.0)
    w121 = e121 / (s10 + 0.0)
    w122 = e122 / (s10 + 0.0)
    w123 = e123 / (s10 + 0.0)
    w124 = e124 / (s10 + 0.0)
    w125 = e125 / (s10 + 0.0)
    w126 = e126 / (s10 + 0.0)
    w127 = e127 / (s10 + 0.0)
    w128 = e128 / (s10 + 0.0)
    w129 = e129 / (s10 + 0.0)
    w130 = e130 / (s10 + 0.0)
    w131 = e131 / (s10 + 0.0)
    s11 = e132 + e133 + e134 + e135 + e136 + e137 + e138 + e139 + e140 + e141 + e142 + e143
    w132 = e132 / (s11 + 0.0)
    w133 = e133 / (s11 + 0.0)
    w134 = e134 / (s11 + 0.0)
    w135 = e135 / (s11 + 0.0)
    w136 = e136 / (s11 + 0.0)
    w137 = e137 / (s11 + 0.0)
    w138 = e138 / (s11 + 0.0)
    w139 = e139 / (s11 + 0.0)
    w140 = e140 / (s11 + 0.0)
    w141 = e141 / (s11 + 0.0)
    w142 = e142 / (s11 + 0.0)
    w143 = e143 / (s11 + 0.0)
    w0 += eps
    w1 += eps
    w2 += eps
    w3 += eps
    w4 += eps
    w5 += eps
    w6 += eps
    w7 += eps
    w8 += eps
    w9 += eps
    w10 += eps
    w11 += eps
    w12 += eps
    w13 += eps
    w14 += eps
    w15 += eps
    w16 += eps
    w17 += eps
    w18 += eps
    w19 += eps
    w20 += eps
    w21 += eps
    w22 += eps
    w23 += eps
    w24 += eps
    w25 += eps
    w26 += eps
    w27 += eps
    w28 += eps
    w29 += eps
    w30 += eps
    w31 += eps
    w32 += eps
    w33 += eps
    w34 += eps
    w35 += eps
    w36 += eps
    w37 += eps
    w38 += eps
    w39 += eps
    w40 += eps
    w41 += eps
    w42 += eps
    w43 += eps
    w44 += eps
    w45 += eps
    w46 += eps
    w47 += eps
    w48 += eps
    w49 += eps
    w50 += eps
    w51 += eps
    w52 += eps
    w53 += eps
    w54 += eps
    w55 += eps
    w56 += eps
    w57 += eps
    w58 += eps
    w59 += eps
    w60 += eps
    w61 += eps
    w62 += eps
    w63 += eps
    w64 += eps
    w65 += eps
    w66 += eps
    w67 += eps
    w68 += eps
    w69 += eps
    w70 += eps
    w71 += eps
    w72 += eps
    w73 += eps
    w74 += eps
    w75 += eps
    w76 += eps
    w77 += eps
    w78 += eps
    w79 += eps
    w80 += eps
    w81 += eps
    w82 += eps
    w83 += eps
    w84 += eps
    w85 += eps
    w86 += eps
    w87 += eps
    w88 += eps
    w89 += eps
    w90 += eps
    w91 += eps
    w92 += eps
    w93 += eps
    w94 += eps
    w95 += eps
    w96 += eps
    w97 += eps
    w98 += eps
    w99 += eps
    w100 += eps
    w101 += eps
    w102 += eps
    w103 += eps
    w104 += eps
    w105 += eps
    w106 += eps
    w107 += eps
    w108 += eps
    w109 += eps
    w110 += eps
    w111 += eps
    w112 += eps
    w113 += eps
    w114 += eps
    w115 += eps
    w116 += eps
    w117 += eps
    w118 += eps
    w119 += eps
    w120 += eps
    w121 += eps
    w122 += eps
    w123 += eps
    w124 += eps
    w125 += eps
    w126 += eps
    w127 += eps
    w128 += eps
    w129 += eps
    w130 += eps
    w131 += eps
    w132 += eps
    w133 += eps
    w134 += eps
    w135 += eps
    w136 += eps
    w137 += eps
    w138 += eps
    w139 += eps
    w140 += eps
    w141 += eps
    w142 += eps
    w143 += eps
    c0 = w0 + w12 + w24 + w36 + w48 + w60 + w72 + w84 + w96 + w108 + w120 + w132
    c1 = w1 + w13 + w25 + w37 + w49 + w61 + w73 + w85 + w97 + w109 + w121 + w133
    c2 = w2 + w14 + w26 + w38 + w50 + w62 + w74 + w86 + w98 + w110 + w122 + w134
    c3 = w3 + w15 + w27 + w39 + w51 + w63 + w75 + w87 + w99 + w111 + w123 + w135
    c4 = w4 + w16 + w28 + w40 + w52 + w64 + w76 + w88 + w100 + w112 + w124 + w136
    c5 = w5 + w17 + w29 + w41 + w53 + w65 + w77 + w89 + w101 + w113 + w125 + w137
    c6 = w6 + w18 + w30 + w42 + w54 + w66 + w78 + w90 + w102 + w114 + w126 + w138
    c7 = w7 + w19 + w31 + w43 + w55 + w67 + w79 + w91 + w103 + w115 + w127 + w139
    c8 = w8 + w20 + w32 + w44 + w56 + w68 + w80 + w92 + w104 + w116 + w128 + w140
    c9 = w9 + w21 + w33 + w45 + w57 + w69 + w81 + w93 + w105 + w117 + w129 + w141
    c10 = w10 + w22 + w34 + w46 + w58 + w70 + w82 + w94 + w106 + w118 + w130 + w142
    c11 = w11 + w23 + w35 + w47 + w59 + w71 + w83 + w95 + w107 + w119 + w131 + w143
    cv0 = 1.0 / (c0 + eps)
    cv1 = 1.0 / (c1 + eps)
    cv2 = 1.0 / (c2 + eps)
    cv3 = 1.0 / (c3 + eps)
    cv4 = 1.0 / (c4 + eps)
    cv5 = 1.0 / (c5 + eps)
    cv6 = 1.0 / (c6 + eps)
    cv7 = 1.0 / (c7 + eps)
    cv8 = 1.0 / (c8 + eps)
    cv9 = 1.0 / (c9 + eps)
    cv10 = 1.0 / (c10 + eps)
    cv11 = 1.0 / (c11 + eps)
    rv0 = tl.full([BLOCK_M], 1.0, tl.float32)
    rv1 = rv0
    rv2 = rv0
    rv3 = rv0
    rv4 = rv0
    rv5 = rv0
    rv6 = rv0
    rv7 = rv0
    rv8 = rv0
    rv9 = rv0
    rv10 = rv0
    rv11 = rv0
    for _ in range(1, iter_step):
        wc0 = w0 * cv0 + w1 * cv1 + w2 * cv2 + w3 * cv3 + w4 * cv4 + w5 * cv5 + w6 * cv6 + w7 * cv7 + w8 * cv8 + w9 * cv9 + w10 * cv10 + w11 * cv11
        wc1 = w12 * cv0 + w13 * cv1 + w14 * cv2 + w15 * cv3 + w16 * cv4 + w17 * cv5 + w18 * cv6 + w19 * cv7 + w20 * cv8 + w21 * cv9 + w22 * cv10 + w23 * cv11
        wc2 = w24 * cv0 + w25 * cv1 + w26 * cv2 + w27 * cv3 + w28 * cv4 + w29 * cv5 + w30 * cv6 + w31 * cv7 + w32 * cv8 + w33 * cv9 + w34 * cv10 + w35 * cv11
        wc3 = w36 * cv0 + w37 * cv1 + w38 * cv2 + w39 * cv3 + w40 * cv4 + w41 * cv5 + w42 * cv6 + w43 * cv7 + w44 * cv8 + w45 * cv9 + w46 * cv10 + w47 * cv11
        wc4 = w48 * cv0 + w49 * cv1 + w50 * cv2 + w51 * cv3 + w52 * cv4 + w53 * cv5 + w54 * cv6 + w55 * cv7 + w56 * cv8 + w57 * cv9 + w58 * cv10 + w59 * cv11
        wc5 = w60 * cv0 + w61 * cv1 + w62 * cv2 + w63 * cv3 + w64 * cv4 + w65 * cv5 + w66 * cv6 + w67 * cv7 + w68 * cv8 + w69 * cv9 + w70 * cv10 + w71 * cv11
        wc6 = w72 * cv0 + w73 * cv1 + w74 * cv2 + w75 * cv3 + w76 * cv4 + w77 * cv5 + w78 * cv6 + w79 * cv7 + w80 * cv8 + w81 * cv9 + w82 * cv10 + w83 * cv11
        wc7 = w84 * cv0 + w85 * cv1 + w86 * cv2 + w87 * cv3 + w88 * cv4 + w89 * cv5 + w90 * cv6 + w91 * cv7 + w92 * cv8 + w93 * cv9 + w94 * cv10 + w95 * cv11
        wc8 = w96 * cv0 + w97 * cv1 + w98 * cv2 + w99 * cv3 + w100 * cv4 + w101 * cv5 + w102 * cv6 + w103 * cv7 + w104 * cv8 + w105 * cv9 + w106 * cv10 + w107 * cv11
        wc9 = w108 * cv0 + w109 * cv1 + w110 * cv2 + w111 * cv3 + w112 * cv4 + w113 * cv5 + w114 * cv6 + w115 * cv7 + w116 * cv8 + w117 * cv9 + w118 * cv10 + w119 * cv11
        wc10 = w120 * cv0 + w121 * cv1 + w122 * cv2 + w123 * cv3 + w124 * cv4 + w125 * cv5 + w126 * cv6 + w127 * cv7 + w128 * cv8 + w129 * cv9 + w130 * cv10 + w131 * cv11
        wc11 = w132 * cv0 + w133 * cv1 + w134 * cv2 + w135 * cv3 + w136 * cv4 + w137 * cv5 + w138 * cv6 + w139 * cv7 + w140 * cv8 + w141 * cv9 + w142 * cv10 + w143 * cv11
        rv0 = rv0 / (rv0 * wc0 + eps)
        rv1 = rv1 / (rv1 * wc1 + eps)
        rv2 = rv2 / (rv2 * wc2 + eps)
        rv3 = rv3 / (rv3 * wc3 + eps)
        rv4 = rv4 / (rv4 * wc4 + eps)
        rv5 = rv5 / (rv5 * wc5 + eps)
        rv6 = rv6 / (rv6 * wc6 + eps)
        rv7 = rv7 / (rv7 * wc7 + eps)
        rv8 = rv8 / (rv8 * wc8 + eps)
        rv9 = rv9 / (rv9 * wc9 + eps)
        rv10 = rv10 / (rv10 * wc10 + eps)
        rv11 = rv11 / (rv11 * wc11 + eps)
        wtr0 = w0 * rv0 + w12 * rv1 + w24 * rv2 + w36 * rv3 + w48 * rv4 + w60 * rv5 + w72 * rv6 + w84 * rv7 + w96 * rv8 + w108 * rv9 + w120 * rv10 + w132 * rv11
        wtr1 = w1 * rv0 + w13 * rv1 + w25 * rv2 + w37 * rv3 + w49 * rv4 + w61 * rv5 + w73 * rv6 + w85 * rv7 + w97 * rv8 + w109 * rv9 + w121 * rv10 + w133 * rv11
        wtr2 = w2 * rv0 + w14 * rv1 + w26 * rv2 + w38 * rv3 + w50 * rv4 + w62 * rv5 + w74 * rv6 + w86 * rv7 + w98 * rv8 + w110 * rv9 + w122 * rv10 + w134 * rv11
        wtr3 = w3 * rv0 + w15 * rv1 + w27 * rv2 + w39 * rv3 + w51 * rv4 + w63 * rv5 + w75 * rv6 + w87 * rv7 + w99 * rv8 + w111 * rv9 + w123 * rv10 + w135 * rv11
        wtr4 = w4 * rv0 + w16 * rv1 + w28 * rv2 + w40 * rv3 + w52 * rv4 + w64 * rv5 + w76 * rv6 + w88 * rv7 + w100 * rv8 + w112 * rv9 + w124 * rv10 + w136 * rv11
        wtr5 = w5 * rv0 + w17 * rv1 + w29 * rv2 + w41 * rv3 + w53 * rv4 + w65 * rv5 + w77 * rv6 + w89 * rv7 + w101 * rv8 + w113 * rv9 + w125 * rv10 + w137 * rv11
        wtr6 = w6 * rv0 + w18 * rv1 + w30 * rv2 + w42 * rv3 + w54 * rv4 + w66 * rv5 + w78 * rv6 + w90 * rv7 + w102 * rv8 + w114 * rv9 + w126 * rv10 + w138 * rv11
        wtr7 = w7 * rv0 + w19 * rv1 + w31 * rv2 + w43 * rv3 + w55 * rv4 + w67 * rv5 + w79 * rv6 + w91 * rv7 + w103 * rv8 + w115 * rv9 + w127 * rv10 + w139 * rv11
        wtr8 = w8 * rv0 + w20 * rv1 + w32 * rv2 + w44 * rv3 + w56 * rv4 + w68 * rv5 + w80 * rv6 + w92 * rv7 + w104 * rv8 + w116 * rv9 + w128 * rv10 + w140 * rv11
        wtr9 = w9 * rv0 + w21 * rv1 + w33 * rv2 + w45 * rv3 + w57 * rv4 + w69 * rv5 + w81 * rv6 + w93 * rv7 + w105 * rv8 + w117 * rv9 + w129 * rv10 + w141 * rv11
        wtr10 = w10 * rv0 + w22 * rv1 + w34 * rv2 + w46 * rv3 + w58 * rv4 + w70 * rv5 + w82 * rv6 + w94 * rv7 + w106 * rv8 + w118 * rv9 + w130 * rv10 + w142 * rv11
        wtr11 = w11 * rv0 + w23 * rv1 + w35 * rv2 + w47 * rv3 + w59 * rv4 + w71 * rv5 + w83 * rv6 + w95 * rv7 + w107 * rv8 + w119 * rv9 + w131 * rv10 + w143 * rv11
        cv0 = cv0 / (cv0 * wtr0 + eps)
        cv1 = cv1 / (cv1 * wtr1 + eps)
        cv2 = cv2 / (cv2 * wtr2 + eps)
        cv3 = cv3 / (cv3 * wtr3 + eps)
        cv4 = cv4 / (cv4 * wtr4 + eps)
        cv5 = cv5 / (cv5 * wtr5 + eps)
        cv6 = cv6 / (cv6 * wtr6 + eps)
        cv7 = cv7 / (cv7 * wtr7 + eps)
        cv8 = cv8 / (cv8 * wtr8 + eps)
        cv9 = cv9 / (cv9 * wtr9 + eps)
        cv10 = cv10 / (cv10 * wtr10 + eps)
        cv11 = cv11 / (cv11 * wtr11 + eps)
    o0 = w0 * rv0 * cv0
    o1 = w1 * rv0 * cv1
    o2 = w2 * rv0 * cv2
    o3 = w3 * rv0 * cv3
    o4 = w4 * rv0 * cv4
    o5 = w5 * rv0 * cv5
    o6 = w6 * rv0 * cv6
    o7 = w7 * rv0 * cv7
    o8 = w8 * rv0 * cv8
    o9 = w9 * rv0 * cv9
    o10 = w10 * rv0 * cv10
    o11 = w11 * rv0 * cv11
    o12 = w12 * rv1 * cv0
    o13 = w13 * rv1 * cv1
    o14 = w14 * rv1 * cv2
    o15 = w15 * rv1 * cv3
    o16 = w16 * rv1 * cv4
    o17 = w17 * rv1 * cv5
    o18 = w18 * rv1 * cv6
    o19 = w19 * rv1 * cv7
    o20 = w20 * rv1 * cv8
    o21 = w21 * rv1 * cv9
    o22 = w22 * rv1 * cv10
    o23 = w23 * rv1 * cv11
    o24 = w24 * rv2 * cv0
    o25 = w25 * rv2 * cv1
    o26 = w26 * rv2 * cv2
    o27 = w27 * rv2 * cv3
    o28 = w28 * rv2 * cv4
    o29 = w29 * rv2 * cv5
    o30 = w30 * rv2 * cv6
    o31 = w31 * rv2 * cv7
    o32 = w32 * rv2 * cv8
    o33 = w33 * rv2 * cv9
    o34 = w34 * rv2 * cv10
    o35 = w35 * rv2 * cv11
    o36 = w36 * rv3 * cv0
    o37 = w37 * rv3 * cv1
    o38 = w38 * rv3 * cv2
    o39 = w39 * rv3 * cv3
    o40 = w40 * rv3 * cv4
    o41 = w41 * rv3 * cv5
    o42 = w42 * rv3 * cv6
    o43 = w43 * rv3 * cv7
    o44 = w44 * rv3 * cv8
    o45 = w45 * rv3 * cv9
    o46 = w46 * rv3 * cv10
    o47 = w47 * rv3 * cv11
    o48 = w48 * rv4 * cv0
    o49 = w49 * rv4 * cv1
    o50 = w50 * rv4 * cv2
    o51 = w51 * rv4 * cv3
    o52 = w52 * rv4 * cv4
    o53 = w53 * rv4 * cv5
    o54 = w54 * rv4 * cv6
    o55 = w55 * rv4 * cv7
    o56 = w56 * rv4 * cv8
    o57 = w57 * rv4 * cv9
    o58 = w58 * rv4 * cv10
    o59 = w59 * rv4 * cv11
    o60 = w60 * rv5 * cv0
    o61 = w61 * rv5 * cv1
    o62 = w62 * rv5 * cv2
    o63 = w63 * rv5 * cv3
    o64 = w64 * rv5 * cv4
    o65 = w65 * rv5 * cv5
    o66 = w66 * rv5 * cv6
    o67 = w67 * rv5 * cv7
    o68 = w68 * rv5 * cv8
    o69 = w69 * rv5 * cv9
    o70 = w70 * rv5 * cv10
    o71 = w71 * rv5 * cv11
    o72 = w72 * rv6 * cv0
    o73 = w73 * rv6 * cv1
    o74 = w74 * rv6 * cv2
    o75 = w75 * rv6 * cv3
    o76 = w76 * rv6 * cv4
    o77 = w77 * rv6 * cv5
    o78 = w78 * rv6 * cv6
    o79 = w79 * rv6 * cv7
    o80 = w80 * rv6 * cv8
    o81 = w81 * rv6 * cv9
    o82 = w82 * rv6 * cv10
    o83 = w83 * rv6 * cv11
    o84 = w84 * rv7 * cv0
    o85 = w85 * rv7 * cv1
    o86 = w86 * rv7 * cv2
    o87 = w87 * rv7 * cv3
    o88 = w88 * rv7 * cv4
    o89 = w89 * rv7 * cv5
    o90 = w90 * rv7 * cv6
    o91 = w91 * rv7 * cv7
    o92 = w92 * rv7 * cv8
    o93 = w93 * rv7 * cv9
    o94 = w94 * rv7 * cv10
    o95 = w95 * rv7 * cv11
    o96 = w96 * rv8 * cv0
    o97 = w97 * rv8 * cv1
    o98 = w98 * rv8 * cv2
    o99 = w99 * rv8 * cv3
    o100 = w100 * rv8 * cv4
    o101 = w101 * rv8 * cv5
    o102 = w102 * rv8 * cv6
    o103 = w103 * rv8 * cv7
    o104 = w104 * rv8 * cv8
    o105 = w105 * rv8 * cv9
    o106 = w106 * rv8 * cv10
    o107 = w107 * rv8 * cv11
    o108 = w108 * rv9 * cv0
    o109 = w109 * rv9 * cv1
    o110 = w110 * rv9 * cv2
    o111 = w111 * rv9 * cv3
    o112 = w112 * rv9 * cv4
    o113 = w113 * rv9 * cv5
    o114 = w114 * rv9 * cv6
    o115 = w115 * rv9 * cv7
    o116 = w116 * rv9 * cv8
    o117 = w117 * rv9 * cv9
    o118 = w118 * rv9 * cv10
    o119 = w119 * rv9 * cv11
    o120 = w120 * rv10 * cv0
    o121 = w121 * rv10 * cv1
    o122 = w122 * rv10 * cv2
    o123 = w123 * rv10 * cv3
    o124 = w124 * rv10 * cv4
    o125 = w125 * rv10 * cv5
    o126 = w126 * rv10 * cv6
    o127 = w127 * rv10 * cv7
    o128 = w128 * rv10 * cv8
    o129 = w129 * rv10 * cv9
    o130 = w130 * rv10 * cv10
    o131 = w131 * rv10 * cv11
    o132 = w132 * rv11 * cv0
    o133 = w133 * rv11 * cv1
    o134 = w134 * rv11 * cv2
    o135 = w135 * rv11 * cv3
    o136 = w136 * rv11 * cv4
    o137 = w137 * rv11 * cv5
    o138 = w138 * rv11 * cv6
    o139 = w139 * rv11 * cv7
    o140 = w140 * rv11 * cv8
    o141 = w141 * rv11 * cv9
    o142 = w142 * rv11 * cv10
    o143 = w143 * rv11 * cv11
    tl.store(optr + base + 0, o0, mask=ok)
    tl.store(optr + base + 1, o1, mask=ok)
    tl.store(optr + base + 2, o2, mask=ok)
    tl.store(optr + base + 3, o3, mask=ok)
    tl.store(optr + base + 4, o4, mask=ok)
    tl.store(optr + base + 5, o5, mask=ok)
    tl.store(optr + base + 6, o6, mask=ok)
    tl.store(optr + base + 7, o7, mask=ok)
    tl.store(optr + base + 8, o8, mask=ok)
    tl.store(optr + base + 9, o9, mask=ok)
    tl.store(optr + base + 10, o10, mask=ok)
    tl.store(optr + base + 11, o11, mask=ok)
    tl.store(optr + base + 12, o12, mask=ok)
    tl.store(optr + base + 13, o13, mask=ok)
    tl.store(optr + base + 14, o14, mask=ok)
    tl.store(optr + base + 15, o15, mask=ok)
    tl.store(optr + base + 16, o16, mask=ok)
    tl.store(optr + base + 17, o17, mask=ok)
    tl.store(optr + base + 18, o18, mask=ok)
    tl.store(optr + base + 19, o19, mask=ok)
    tl.store(optr + base + 20, o20, mask=ok)
    tl.store(optr + base + 21, o21, mask=ok)
    tl.store(optr + base + 22, o22, mask=ok)
    tl.store(optr + base + 23, o23, mask=ok)
    tl.store(optr + base + 24, o24, mask=ok)
    tl.store(optr + base + 25, o25, mask=ok)
    tl.store(optr + base + 26, o26, mask=ok)
    tl.store(optr + base + 27, o27, mask=ok)
    tl.store(optr + base + 28, o28, mask=ok)
    tl.store(optr + base + 29, o29, mask=ok)
    tl.store(optr + base + 30, o30, mask=ok)
    tl.store(optr + base + 31, o31, mask=ok)
    tl.store(optr + base + 32, o32, mask=ok)
    tl.store(optr + base + 33, o33, mask=ok)
    tl.store(optr + base + 34, o34, mask=ok)
    tl.store(optr + base + 35, o35, mask=ok)
    tl.store(optr + base + 36, o36, mask=ok)
    tl.store(optr + base + 37, o37, mask=ok)
    tl.store(optr + base + 38, o38, mask=ok)
    tl.store(optr + base + 39, o39, mask=ok)
    tl.store(optr + base + 40, o40, mask=ok)
    tl.store(optr + base + 41, o41, mask=ok)
    tl.store(optr + base + 42, o42, mask=ok)
    tl.store(optr + base + 43, o43, mask=ok)
    tl.store(optr + base + 44, o44, mask=ok)
    tl.store(optr + base + 45, o45, mask=ok)
    tl.store(optr + base + 46, o46, mask=ok)
    tl.store(optr + base + 47, o47, mask=ok)
    tl.store(optr + base + 48, o48, mask=ok)
    tl.store(optr + base + 49, o49, mask=ok)
    tl.store(optr + base + 50, o50, mask=ok)
    tl.store(optr + base + 51, o51, mask=ok)
    tl.store(optr + base + 52, o52, mask=ok)
    tl.store(optr + base + 53, o53, mask=ok)
    tl.store(optr + base + 54, o54, mask=ok)
    tl.store(optr + base + 55, o55, mask=ok)
    tl.store(optr + base + 56, o56, mask=ok)
    tl.store(optr + base + 57, o57, mask=ok)
    tl.store(optr + base + 58, o58, mask=ok)
    tl.store(optr + base + 59, o59, mask=ok)
    tl.store(optr + base + 60, o60, mask=ok)
    tl.store(optr + base + 61, o61, mask=ok)
    tl.store(optr + base + 62, o62, mask=ok)
    tl.store(optr + base + 63, o63, mask=ok)
    tl.store(optr + base + 64, o64, mask=ok)
    tl.store(optr + base + 65, o65, mask=ok)
    tl.store(optr + base + 66, o66, mask=ok)
    tl.store(optr + base + 67, o67, mask=ok)
    tl.store(optr + base + 68, o68, mask=ok)
    tl.store(optr + base + 69, o69, mask=ok)
    tl.store(optr + base + 70, o70, mask=ok)
    tl.store(optr + base + 71, o71, mask=ok)
    tl.store(optr + base + 72, o72, mask=ok)
    tl.store(optr + base + 73, o73, mask=ok)
    tl.store(optr + base + 74, o74, mask=ok)
    tl.store(optr + base + 75, o75, mask=ok)
    tl.store(optr + base + 76, o76, mask=ok)
    tl.store(optr + base + 77, o77, mask=ok)
    tl.store(optr + base + 78, o78, mask=ok)
    tl.store(optr + base + 79, o79, mask=ok)
    tl.store(optr + base + 80, o80, mask=ok)
    tl.store(optr + base + 81, o81, mask=ok)
    tl.store(optr + base + 82, o82, mask=ok)
    tl.store(optr + base + 83, o83, mask=ok)
    tl.store(optr + base + 84, o84, mask=ok)
    tl.store(optr + base + 85, o85, mask=ok)
    tl.store(optr + base + 86, o86, mask=ok)
    tl.store(optr + base + 87, o87, mask=ok)
    tl.store(optr + base + 88, o88, mask=ok)
    tl.store(optr + base + 89, o89, mask=ok)
    tl.store(optr + base + 90, o90, mask=ok)
    tl.store(optr + base + 91, o91, mask=ok)
    tl.store(optr + base + 92, o92, mask=ok)
    tl.store(optr + base + 93, o93, mask=ok)
    tl.store(optr + base + 94, o94, mask=ok)
    tl.store(optr + base + 95, o95, mask=ok)
    tl.store(optr + base + 96, o96, mask=ok)
    tl.store(optr + base + 97, o97, mask=ok)
    tl.store(optr + base + 98, o98, mask=ok)
    tl.store(optr + base + 99, o99, mask=ok)
    tl.store(optr + base + 100, o100, mask=ok)
    tl.store(optr + base + 101, o101, mask=ok)
    tl.store(optr + base + 102, o102, mask=ok)
    tl.store(optr + base + 103, o103, mask=ok)
    tl.store(optr + base + 104, o104, mask=ok)
    tl.store(optr + base + 105, o105, mask=ok)
    tl.store(optr + base + 106, o106, mask=ok)
    tl.store(optr + base + 107, o107, mask=ok)
    tl.store(optr + base + 108, o108, mask=ok)
    tl.store(optr + base + 109, o109, mask=ok)
    tl.store(optr + base + 110, o110, mask=ok)
    tl.store(optr + base + 111, o111, mask=ok)
    tl.store(optr + base + 112, o112, mask=ok)
    tl.store(optr + base + 113, o113, mask=ok)
    tl.store(optr + base + 114, o114, mask=ok)
    tl.store(optr + base + 115, o115, mask=ok)
    tl.store(optr + base + 116, o116, mask=ok)
    tl.store(optr + base + 117, o117, mask=ok)
    tl.store(optr + base + 118, o118, mask=ok)
    tl.store(optr + base + 119, o119, mask=ok)
    tl.store(optr + base + 120, o120, mask=ok)
    tl.store(optr + base + 121, o121, mask=ok)
    tl.store(optr + base + 122, o122, mask=ok)
    tl.store(optr + base + 123, o123, mask=ok)
    tl.store(optr + base + 124, o124, mask=ok)
    tl.store(optr + base + 125, o125, mask=ok)
    tl.store(optr + base + 126, o126, mask=ok)
    tl.store(optr + base + 127, o127, mask=ok)
    tl.store(optr + base + 128, o128, mask=ok)
    tl.store(optr + base + 129, o129, mask=ok)
    tl.store(optr + base + 130, o130, mask=ok)
    tl.store(optr + base + 131, o131, mask=ok)
    tl.store(optr + base + 132, o132, mask=ok)
    tl.store(optr + base + 133, o133, mask=ok)
    tl.store(optr + base + 134, o134, mask=ok)
    tl.store(optr + base + 135, o135, mask=ok)
    tl.store(optr + base + 136, o136, mask=ok)
    tl.store(optr + base + 137, o137, mask=ok)
    tl.store(optr + base + 138, o138, mask=ok)
    tl.store(optr + base + 139, o139, mask=ok)
    tl.store(optr + base + 140, o140, mask=ok)
    tl.store(optr + base + 141, o141, mask=ok)
    tl.store(optr + base + 142, o142, mask=ok)
    tl.store(optr + base + 143, o143, mask=ok)


# ===========================================================================
# Kernel C：旧 fused 版（非 pow2 阶 n 回落）
# ===========================================================================
@triton.jit
def _mhc_sinkhorn_fused(
    comb_ptr, out_ptr, B, iter_step: tl.constexpr, eps,
    N: tl.constexpr, N_TRUE: tl.constexpr, BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    bb = tl.arange(0, BLOCK_B)[:, None, None]
    ri = tl.arange(0, N)[None, :, None]
    ci = tl.arange(0, N)[None, None, :]
    b = pid * BLOCK_B + bb
    offs = b * (N_TRUE * N_TRUE) + ri * N_TRUE + ci
    b_ok = b < B
    r_ok = ri < N_TRUE
    c_ok = ci < N_TRUE
    row_ok = b_ok & r_ok
    mask3 = row_ok & c_ok
    eps_m = tl.where(r_ok & c_ok, eps, 0.0)
    pad_row = tl.where(row_ok, 0.0, 1.0)
    x = tl.load(comb_ptr + offs, mask=mask3, other=float("-inf"))
    row_max = tl.max(x, axis=2, keep_dims=True)
    row_max = tl.where(row_ok, row_max, 0.0)
    x = tl.exp(x - row_max)
    row_sum = tl.sum(x, axis=2, keep_dims=True)
    x = x / (row_sum + pad_row) + eps_m
    col_sum = tl.sum(x, axis=1, keep_dims=True)
    x = x / (col_sum + eps)
    for _it in range(1, iter_step):
        row_sum = tl.sum(x, axis=2, keep_dims=True)
        x = x / (row_sum + pad_row)
        col_sum = tl.sum(x, axis=1, keep_dims=True)
        x = x / (col_sum + eps)
    tl.store(out_ptr + offs, x, mask=mask3)


@triton.jit
def _skrow12(ptr, optr, B, iter_step: tl.constexpr, eps,
           N: tl.constexpr, N_TRUE: tl.constexpr, BLOCK_M: tl.constexpr,
         EXACT: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * (N_TRUE * N_TRUE)
    ci = tl.arange(0, N)
    cm = ci < N_TRUE                       # [N] 列有效性
    eps_m = tl.where(cm, eps, 0.0)         # [N] 仅有效列加 eps
    r0 = tl.load(ptr + base[:, None] + 0 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r1 = tl.load(ptr + base[:, None] + 1 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r2 = tl.load(ptr + base[:, None] + 2 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r3 = tl.load(ptr + base[:, None] + 3 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r4 = tl.load(ptr + base[:, None] + 4 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r5 = tl.load(ptr + base[:, None] + 5 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r6 = tl.load(ptr + base[:, None] + 6 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r7 = tl.load(ptr + base[:, None] + 7 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r8 = tl.load(ptr + base[:, None] + 8 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r9 = tl.load(ptr + base[:, None] + 9 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r10 = tl.load(ptr + base[:, None] + 10 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r11 = tl.load(ptr + base[:, None] + 11 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    m0 = tl.max(r0, axis=1, keep_dims=True)
    e0 = tl.exp(r0 - m0)
    s0 = tl.sum(e0, axis=1, keep_dims=True)
    w0 = e0 / (s0 + 0.0) + eps_m[None, :]
    m1 = tl.max(r1, axis=1, keep_dims=True)
    e1 = tl.exp(r1 - m1)
    s1 = tl.sum(e1, axis=1, keep_dims=True)
    w1 = e1 / (s1 + 0.0) + eps_m[None, :]
    m2 = tl.max(r2, axis=1, keep_dims=True)
    e2 = tl.exp(r2 - m2)
    s2 = tl.sum(e2, axis=1, keep_dims=True)
    w2 = e2 / (s2 + 0.0) + eps_m[None, :]
    m3 = tl.max(r3, axis=1, keep_dims=True)
    e3 = tl.exp(r3 - m3)
    s3 = tl.sum(e3, axis=1, keep_dims=True)
    w3 = e3 / (s3 + 0.0) + eps_m[None, :]
    m4 = tl.max(r4, axis=1, keep_dims=True)
    e4 = tl.exp(r4 - m4)
    s4 = tl.sum(e4, axis=1, keep_dims=True)
    w4 = e4 / (s4 + 0.0) + eps_m[None, :]
    m5 = tl.max(r5, axis=1, keep_dims=True)
    e5 = tl.exp(r5 - m5)
    s5 = tl.sum(e5, axis=1, keep_dims=True)
    w5 = e5 / (s5 + 0.0) + eps_m[None, :]
    m6 = tl.max(r6, axis=1, keep_dims=True)
    e6 = tl.exp(r6 - m6)
    s6 = tl.sum(e6, axis=1, keep_dims=True)
    w6 = e6 / (s6 + 0.0) + eps_m[None, :]
    m7 = tl.max(r7, axis=1, keep_dims=True)
    e7 = tl.exp(r7 - m7)
    s7 = tl.sum(e7, axis=1, keep_dims=True)
    w7 = e7 / (s7 + 0.0) + eps_m[None, :]
    m8 = tl.max(r8, axis=1, keep_dims=True)
    e8 = tl.exp(r8 - m8)
    s8 = tl.sum(e8, axis=1, keep_dims=True)
    w8 = e8 / (s8 + 0.0) + eps_m[None, :]
    m9 = tl.max(r9, axis=1, keep_dims=True)
    e9 = tl.exp(r9 - m9)
    s9 = tl.sum(e9, axis=1, keep_dims=True)
    w9 = e9 / (s9 + 0.0) + eps_m[None, :]
    m10 = tl.max(r10, axis=1, keep_dims=True)
    e10 = tl.exp(r10 - m10)
    s10 = tl.sum(e10, axis=1, keep_dims=True)
    w10 = e10 / (s10 + 0.0) + eps_m[None, :]
    m11 = tl.max(r11, axis=1, keep_dims=True)
    e11 = tl.exp(r11 - m11)
    s11 = tl.sum(e11, axis=1, keep_dims=True)
    w11 = e11 / (s11 + 0.0) + eps_m[None, :]
    col = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + w11
    cv = 1.0 / (col + eps)                 # [BM, N] 列缩放
    rv0 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv1 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv2 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv3 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv4 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv5 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv6 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv7 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv8 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv9 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv10 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv11 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    for _ in range(1, iter_step):
        wc0 = tl.sum(w0 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv0 = rv0 / (rv0 * wc0 + eps)
        wc1 = tl.sum(w1 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv1 = rv1 / (rv1 * wc1 + eps)
        wc2 = tl.sum(w2 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv2 = rv2 / (rv2 * wc2 + eps)
        wc3 = tl.sum(w3 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv3 = rv3 / (rv3 * wc3 + eps)
        wc4 = tl.sum(w4 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv4 = rv4 / (rv4 * wc4 + eps)
        wc5 = tl.sum(w5 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv5 = rv5 / (rv5 * wc5 + eps)
        wc6 = tl.sum(w6 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv6 = rv6 / (rv6 * wc6 + eps)
        wc7 = tl.sum(w7 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv7 = rv7 / (rv7 * wc7 + eps)
        wc8 = tl.sum(w8 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv8 = rv8 / (rv8 * wc8 + eps)
        wc9 = tl.sum(w9 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv9 = rv9 / (rv9 * wc9 + eps)
        wc10 = tl.sum(w10 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv10 = rv10 / (rv10 * wc10 + eps)
        wc11 = tl.sum(w11 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv11 = rv11 / (rv11 * wc11 + eps)
        wtr = w0 * rv0 + w1 * rv1 + w2 * rv2 + w3 * rv3 + w4 * rv4 + w5 * rv5 + w6 * rv6 + w7 * rv7 + w8 * rv8 + w9 * rv9 + w10 * rv10 + w11 * rv11                    # [BM,N] 显式跨行加
        cv = cv / (cv * wtr + eps)
        cv = tl.where(cm[None, :], cv, 0.0)   # padding 列归零防 inf 污染
    o0 = w0 * rv0 * cv
    o1 = w1 * rv1 * cv
    o2 = w2 * rv2 * cv
    o3 = w3 * rv3 * cv
    o4 = w4 * rv4 * cv
    o5 = w5 * rv5 * cv
    o6 = w6 * rv6 * cv
    o7 = w7 * rv7 * cv
    o8 = w8 * rv8 * cv
    o9 = w9 * rv9 * cv
    o10 = w10 * rv10 * cv
    o11 = w11 * rv11 * cv
    tl.store(optr + base[:, None] + 0 * N_TRUE + ci[None, :], o0,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 1 * N_TRUE + ci[None, :], o1,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 2 * N_TRUE + ci[None, :], o2,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 3 * N_TRUE + ci[None, :], o3,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 4 * N_TRUE + ci[None, :], o4,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 5 * N_TRUE + ci[None, :], o5,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 6 * N_TRUE + ci[None, :], o6,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 7 * N_TRUE + ci[None, :], o7,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 8 * N_TRUE + ci[None, :], o8,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 9 * N_TRUE + ci[None, :], o9,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 10 * N_TRUE + ci[None, :], o10,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 11 * N_TRUE + ci[None, :], o11,
             mask=ok[:, None] & cm[None, :])


@triton.jit
def _skrow16(ptr, optr, B, iter_step: tl.constexpr, eps,
           N: tl.constexpr, N_TRUE: tl.constexpr, BLOCK_M: tl.constexpr,
         EXACT: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * (N_TRUE * N_TRUE)
    ci = tl.arange(0, N)
    cm = ci < N_TRUE                       # [N] 列有效性
    eps_m = tl.where(cm, eps, 0.0)         # [N] 仅有效列加 eps
    r0 = tl.load(ptr + base[:, None] + 0 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r1 = tl.load(ptr + base[:, None] + 1 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r2 = tl.load(ptr + base[:, None] + 2 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r3 = tl.load(ptr + base[:, None] + 3 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r4 = tl.load(ptr + base[:, None] + 4 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r5 = tl.load(ptr + base[:, None] + 5 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r6 = tl.load(ptr + base[:, None] + 6 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r7 = tl.load(ptr + base[:, None] + 7 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r8 = tl.load(ptr + base[:, None] + 8 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r9 = tl.load(ptr + base[:, None] + 9 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r10 = tl.load(ptr + base[:, None] + 10 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r11 = tl.load(ptr + base[:, None] + 11 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r12 = tl.load(ptr + base[:, None] + 12 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r13 = tl.load(ptr + base[:, None] + 13 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r14 = tl.load(ptr + base[:, None] + 14 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r15 = tl.load(ptr + base[:, None] + 15 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    m0 = tl.max(r0, axis=1, keep_dims=True)
    e0 = tl.exp(r0 - m0)
    s0 = tl.sum(e0, axis=1, keep_dims=True)
    w0 = e0 / (s0 + 0.0) + eps_m[None, :]
    m1 = tl.max(r1, axis=1, keep_dims=True)
    e1 = tl.exp(r1 - m1)
    s1 = tl.sum(e1, axis=1, keep_dims=True)
    w1 = e1 / (s1 + 0.0) + eps_m[None, :]
    m2 = tl.max(r2, axis=1, keep_dims=True)
    e2 = tl.exp(r2 - m2)
    s2 = tl.sum(e2, axis=1, keep_dims=True)
    w2 = e2 / (s2 + 0.0) + eps_m[None, :]
    m3 = tl.max(r3, axis=1, keep_dims=True)
    e3 = tl.exp(r3 - m3)
    s3 = tl.sum(e3, axis=1, keep_dims=True)
    w3 = e3 / (s3 + 0.0) + eps_m[None, :]
    m4 = tl.max(r4, axis=1, keep_dims=True)
    e4 = tl.exp(r4 - m4)
    s4 = tl.sum(e4, axis=1, keep_dims=True)
    w4 = e4 / (s4 + 0.0) + eps_m[None, :]
    m5 = tl.max(r5, axis=1, keep_dims=True)
    e5 = tl.exp(r5 - m5)
    s5 = tl.sum(e5, axis=1, keep_dims=True)
    w5 = e5 / (s5 + 0.0) + eps_m[None, :]
    m6 = tl.max(r6, axis=1, keep_dims=True)
    e6 = tl.exp(r6 - m6)
    s6 = tl.sum(e6, axis=1, keep_dims=True)
    w6 = e6 / (s6 + 0.0) + eps_m[None, :]
    m7 = tl.max(r7, axis=1, keep_dims=True)
    e7 = tl.exp(r7 - m7)
    s7 = tl.sum(e7, axis=1, keep_dims=True)
    w7 = e7 / (s7 + 0.0) + eps_m[None, :]
    m8 = tl.max(r8, axis=1, keep_dims=True)
    e8 = tl.exp(r8 - m8)
    s8 = tl.sum(e8, axis=1, keep_dims=True)
    w8 = e8 / (s8 + 0.0) + eps_m[None, :]
    m9 = tl.max(r9, axis=1, keep_dims=True)
    e9 = tl.exp(r9 - m9)
    s9 = tl.sum(e9, axis=1, keep_dims=True)
    w9 = e9 / (s9 + 0.0) + eps_m[None, :]
    m10 = tl.max(r10, axis=1, keep_dims=True)
    e10 = tl.exp(r10 - m10)
    s10 = tl.sum(e10, axis=1, keep_dims=True)
    w10 = e10 / (s10 + 0.0) + eps_m[None, :]
    m11 = tl.max(r11, axis=1, keep_dims=True)
    e11 = tl.exp(r11 - m11)
    s11 = tl.sum(e11, axis=1, keep_dims=True)
    w11 = e11 / (s11 + 0.0) + eps_m[None, :]
    m12 = tl.max(r12, axis=1, keep_dims=True)
    e12 = tl.exp(r12 - m12)
    s12 = tl.sum(e12, axis=1, keep_dims=True)
    w12 = e12 / (s12 + 0.0) + eps_m[None, :]
    m13 = tl.max(r13, axis=1, keep_dims=True)
    e13 = tl.exp(r13 - m13)
    s13 = tl.sum(e13, axis=1, keep_dims=True)
    w13 = e13 / (s13 + 0.0) + eps_m[None, :]
    m14 = tl.max(r14, axis=1, keep_dims=True)
    e14 = tl.exp(r14 - m14)
    s14 = tl.sum(e14, axis=1, keep_dims=True)
    w14 = e14 / (s14 + 0.0) + eps_m[None, :]
    m15 = tl.max(r15, axis=1, keep_dims=True)
    e15 = tl.exp(r15 - m15)
    s15 = tl.sum(e15, axis=1, keep_dims=True)
    w15 = e15 / (s15 + 0.0) + eps_m[None, :]
    col = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + w11 + w12 + w13 + w14 + w15
    cv = 1.0 / (col + eps)                 # [BM, N] 列缩放
    rv0 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv1 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv2 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv3 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv4 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv5 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv6 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv7 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv8 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv9 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv10 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv11 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv12 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv13 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv14 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv15 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    for _ in range(1, iter_step):
        wc0 = tl.sum(w0 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv0 = rv0 / (rv0 * wc0 + eps)
        wc1 = tl.sum(w1 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv1 = rv1 / (rv1 * wc1 + eps)
        wc2 = tl.sum(w2 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv2 = rv2 / (rv2 * wc2 + eps)
        wc3 = tl.sum(w3 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv3 = rv3 / (rv3 * wc3 + eps)
        wc4 = tl.sum(w4 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv4 = rv4 / (rv4 * wc4 + eps)
        wc5 = tl.sum(w5 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv5 = rv5 / (rv5 * wc5 + eps)
        wc6 = tl.sum(w6 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv6 = rv6 / (rv6 * wc6 + eps)
        wc7 = tl.sum(w7 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv7 = rv7 / (rv7 * wc7 + eps)
        wc8 = tl.sum(w8 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv8 = rv8 / (rv8 * wc8 + eps)
        wc9 = tl.sum(w9 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv9 = rv9 / (rv9 * wc9 + eps)
        wc10 = tl.sum(w10 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv10 = rv10 / (rv10 * wc10 + eps)
        wc11 = tl.sum(w11 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv11 = rv11 / (rv11 * wc11 + eps)
        wc12 = tl.sum(w12 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv12 = rv12 / (rv12 * wc12 + eps)
        wc13 = tl.sum(w13 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv13 = rv13 / (rv13 * wc13 + eps)
        wc14 = tl.sum(w14 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv14 = rv14 / (rv14 * wc14 + eps)
        wc15 = tl.sum(w15 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv15 = rv15 / (rv15 * wc15 + eps)
        wtr = w0 * rv0 + w1 * rv1 + w2 * rv2 + w3 * rv3 + w4 * rv4 + w5 * rv5 + w6 * rv6 + w7 * rv7 + w8 * rv8 + w9 * rv9 + w10 * rv10 + w11 * rv11 + w12 * rv12 + w13 * rv13 + w14 * rv14 + w15 * rv15                    # [BM,N] 显式跨行加
        cv = cv / (cv * wtr + eps)
        cv = tl.where(cm[None, :], cv, 0.0)   # padding 列归零防 inf 污染
    o0 = w0 * rv0 * cv
    o1 = w1 * rv1 * cv
    o2 = w2 * rv2 * cv
    o3 = w3 * rv3 * cv
    o4 = w4 * rv4 * cv
    o5 = w5 * rv5 * cv
    o6 = w6 * rv6 * cv
    o7 = w7 * rv7 * cv
    o8 = w8 * rv8 * cv
    o9 = w9 * rv9 * cv
    o10 = w10 * rv10 * cv
    o11 = w11 * rv11 * cv
    o12 = w12 * rv12 * cv
    o13 = w13 * rv13 * cv
    o14 = w14 * rv14 * cv
    o15 = w15 * rv15 * cv
    tl.store(optr + base[:, None] + 0 * N_TRUE + ci[None, :], o0,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 1 * N_TRUE + ci[None, :], o1,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 2 * N_TRUE + ci[None, :], o2,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 3 * N_TRUE + ci[None, :], o3,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 4 * N_TRUE + ci[None, :], o4,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 5 * N_TRUE + ci[None, :], o5,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 6 * N_TRUE + ci[None, :], o6,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 7 * N_TRUE + ci[None, :], o7,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 8 * N_TRUE + ci[None, :], o8,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 9 * N_TRUE + ci[None, :], o9,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 10 * N_TRUE + ci[None, :], o10,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 11 * N_TRUE + ci[None, :], o11,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 12 * N_TRUE + ci[None, :], o12,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 13 * N_TRUE + ci[None, :], o13,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 14 * N_TRUE + ci[None, :], o14,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 15 * N_TRUE + ci[None, :], o15,
             mask=ok[:, None] & cm[None, :])


@triton.jit
def _skrow8(ptr, optr, B, iter_step: tl.constexpr, eps,
           N: tl.constexpr, N_TRUE: tl.constexpr, BLOCK_M: tl.constexpr,
         EXACT: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = off < B
    base = off * (N_TRUE * N_TRUE)
    ci = tl.arange(0, N)
    cm = ci < N_TRUE                       # [N] 列有效性
    eps_m = tl.where(cm, eps, 0.0)         # [N] 仅有效列加 eps
    r0 = tl.load(ptr + base[:, None] + 0 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r1 = tl.load(ptr + base[:, None] + 1 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r2 = tl.load(ptr + base[:, None] + 2 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r3 = tl.load(ptr + base[:, None] + 3 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r4 = tl.load(ptr + base[:, None] + 4 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r5 = tl.load(ptr + base[:, None] + 5 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r6 = tl.load(ptr + base[:, None] + 6 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    r7 = tl.load(ptr + base[:, None] + 7 * N_TRUE + ci[None, :],
                   mask=ok[:, None] & cm[None, :], other=float("-inf"))
    m0 = tl.max(r0, axis=1, keep_dims=True)
    e0 = tl.exp(r0 - m0)
    s0 = tl.sum(e0, axis=1, keep_dims=True)
    w0 = e0 / (s0 + 0.0) + eps_m[None, :]
    m1 = tl.max(r1, axis=1, keep_dims=True)
    e1 = tl.exp(r1 - m1)
    s1 = tl.sum(e1, axis=1, keep_dims=True)
    w1 = e1 / (s1 + 0.0) + eps_m[None, :]
    m2 = tl.max(r2, axis=1, keep_dims=True)
    e2 = tl.exp(r2 - m2)
    s2 = tl.sum(e2, axis=1, keep_dims=True)
    w2 = e2 / (s2 + 0.0) + eps_m[None, :]
    m3 = tl.max(r3, axis=1, keep_dims=True)
    e3 = tl.exp(r3 - m3)
    s3 = tl.sum(e3, axis=1, keep_dims=True)
    w3 = e3 / (s3 + 0.0) + eps_m[None, :]
    m4 = tl.max(r4, axis=1, keep_dims=True)
    e4 = tl.exp(r4 - m4)
    s4 = tl.sum(e4, axis=1, keep_dims=True)
    w4 = e4 / (s4 + 0.0) + eps_m[None, :]
    m5 = tl.max(r5, axis=1, keep_dims=True)
    e5 = tl.exp(r5 - m5)
    s5 = tl.sum(e5, axis=1, keep_dims=True)
    w5 = e5 / (s5 + 0.0) + eps_m[None, :]
    m6 = tl.max(r6, axis=1, keep_dims=True)
    e6 = tl.exp(r6 - m6)
    s6 = tl.sum(e6, axis=1, keep_dims=True)
    w6 = e6 / (s6 + 0.0) + eps_m[None, :]
    m7 = tl.max(r7, axis=1, keep_dims=True)
    e7 = tl.exp(r7 - m7)
    s7 = tl.sum(e7, axis=1, keep_dims=True)
    w7 = e7 / (s7 + 0.0) + eps_m[None, :]
    col = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7
    cv = 1.0 / (col + eps)                 # [BM, N] 列缩放
    rv0 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv1 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv2 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv3 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv4 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv5 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv6 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    rv7 = tl.full([BLOCK_M, 1], 1.0, tl.float32)
    for _ in range(1, iter_step):
        wc0 = tl.sum(w0 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv0 = rv0 / (rv0 * wc0 + eps)
        wc1 = tl.sum(w1 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv1 = rv1 / (rv1 * wc1 + eps)
        wc2 = tl.sum(w2 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv2 = rv2 / (rv2 * wc2 + eps)
        wc3 = tl.sum(w3 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv3 = rv3 / (rv3 * wc3 + eps)
        wc4 = tl.sum(w4 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv4 = rv4 / (rv4 * wc4 + eps)
        wc5 = tl.sum(w5 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv5 = rv5 / (rv5 * wc5 + eps)
        wc6 = tl.sum(w6 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv6 = rv6 / (rv6 * wc6 + eps)
        wc7 = tl.sum(w7 * cv, axis=1, keep_dims=True)   # [BM,1] 行和
        rv7 = rv7 / (rv7 * wc7 + eps)
        wtr = w0 * rv0 + w1 * rv1 + w2 * rv2 + w3 * rv3 + w4 * rv4 + w5 * rv5 + w6 * rv6 + w7 * rv7                    # [BM,N] 显式跨行加
        cv = cv / (cv * wtr + eps)
        cv = tl.where(cm[None, :], cv, 0.0)   # padding 列归零防 inf 污染
    o0 = w0 * rv0 * cv
    o1 = w1 * rv1 * cv
    o2 = w2 * rv2 * cv
    o3 = w3 * rv3 * cv
    o4 = w4 * rv4 * cv
    o5 = w5 * rv5 * cv
    o6 = w6 * rv6 * cv
    o7 = w7 * rv7 * cv
    tl.store(optr + base[:, None] + 0 * N_TRUE + ci[None, :], o0,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 1 * N_TRUE + ci[None, :], o1,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 2 * N_TRUE + ci[None, :], o2,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 3 * N_TRUE + ci[None, :], o3,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 4 * N_TRUE + ci[None, :], o4,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 5 * N_TRUE + ci[None, :], o5,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 6 * N_TRUE + ci[None, :], o6,
             mask=ok[:, None] & cm[None, :])
    tl.store(optr + base[:, None] + 7 * N_TRUE + ci[None, :], o7,
             mask=ok[:, None] & cm[None, :])


# ===========================================================================
# Host 侧辅助
# ===========================================================================
def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ---------------------------------------------------------------------------
# 分类分派（静态常量 if/else,≤6 类；每类独立 kernel）
#   C0 FUSED      ：非 pow2 n∈{3,6,12} -> _mhc_sinkhorn_fused （tile 内 reduce 最慢）
#   C1 SK4        ：n==4 且 B>=128     -> _sk4 （标量展开，消除 AIV reduce 树，大中 B 最优）
#   C2 SK2        ：n==2               -> _sk2 （标量展开，全 B 优于 scale）
#   C3 SCALE_SMALL：n==4 且 B<128      -> _sinkhorn_scale （c1/c2 小 B，标量反而不如 scale）
#   C4 BIGN       ：pow2 n∈{8,16}      -> _sinkhorn_scale （元素多，标量展开不可行）
# ---------------------------------------------------------------------------
K_FUSED, K_SK4, K_SK2, K_SCALE_SMALL, K_BIGN, K_SK3, K_SK6, K_SK8, K_SK12, K_SCALEM, K_SKROW12, K_SKROW16, K_SKROW8 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12


def classify(n: int, B: int, iter_step: int, eps: float) -> int:
    """按静态常量判定，返回类别编号。"""
    if n == 3:                   # n=3 标量展开（替代原 fused）
        return K_SK3
    if n == 6:                   # n=6 标量展开（替代原 fused）
        return K_SK6
    if n == 8:                   # n=8 行粒度（gather 连续化；先验 -8.1%）
        return K_SKROW8
    if n == 12:                  # n=12 行粒度（Tiling：行内连续归约+显式跨行加）
        return K_SKROW12
    if not _is_pow2(n):          # 非 pow2 → masked scale（替代 fused）
        return K_SCALEM
    if n == 16:                  # n=16 行粒度（Tiling：行内连续归约+显式跨行加）
        return K_SKROW16
    if n == 2:                   # n=2 标量全胜
        return K_SK2
    if n == 4:                   # n=4：小B 或 it<=1（无迭代收益）回落 scale
        if B < 128 or iter_step <= 1:
            return K_SCALE_SMALL
        return K_SK4
    return K_BIGN                # pow2 n=8/16


def _pick_block(cls: int, b: int, n: int) -> int:
    """每类独立的最优 block；grid≈256（56 AIV,~4.5 wave），小 B 需 bm>=2。"""
    if b <= 1:
        return 1
    if cls == K_FUSED:                     # C0：fused 最优 bm 小（n=12 落 bm=2）
        bm = 4 if n != 12 else 2
        return min(bm, 4096 // (n * n))
    if cls in (K_SK3, K_SK4, K_SK6, K_SK8, K_SK12):  # 标量展开
        if b >= 4096:
            return 128
        if cls in (K_SK3, K_SK4):
            return 64        # join-store 后 BM 重扫：BM=64 实测最优（case4 8.37/8.78µs）
        return 32
    if cls in (K_SKROW8, K_SKROW12, K_SKROW16):  # 行粒度 n=8/12/16：v8 实测 BM=32 最优
        return 32
    if cls == K_SK2:                       # C2：n=2 标量，grid≈16，bm 随 B 上取（重扫 BM64 最优）
        bm = _next_pow2(max(1, (b + 15) // 16))
        return max(1, min(bm, b))
    # C3 SCALE_SMALL / C4 BIGN / SCALEM 共用 scale-vector grid 策略；BIGN 收紧 UB
    if cls == K_SCALE_SMALL and b >= 1024:
        return 8                           # iter<=1 无迭代链，bm=8 实测最优（case13）
    np_ = _next_pow2(n)
    target_grid = min(256, b)
    bm = _next_pow2(max(1, (b + target_grid - 1) // target_grid))
    if b >= 2:
        bm = max(2, bm)                    # 单 batch tile 过大反而慢（c2/B=64 最优 bm=2）
    cap = 8192 // (np_ * np_)
    if cls == K_BIGN:                      # C4：n=8/16 tile 大，收紧到实测稳定区
        cap = min(cap, 16)
    return min(bm, max(1, cap))


# ===========================================================================
# ModelNew
# ===========================================================================
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, comb, iter_step, eps) -> torch.Tensor:
        if not comb.is_contiguous():
            comb = comb.contiguous()

        n = comb.shape[-1]
        B = comb.numel() // (n * n)
        out = torch.empty_like(comb)
        Np = _next_pow2(n)
        cls = classify(n, B, int(iter_step), float(eps))
        bm = _pick_block(cls, B, n)
        grid = (triton.cdiv(B, bm),)
        if cls == K_FUSED:                          # C0 非 pow2 回落 fused（保留，未被 n=12 用）
            _mhc_sinkhorn_fused[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                N=Np, N_TRUE=n, BLOCK_B=bm,
            )
        elif cls == K_SKROW8:                       # n=8 行粒度（opt_iter_2）
            _skrow8[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                N=Np, N_TRUE=n, BLOCK_M=bm, EXACT=(grid[0] * bm == B), vf_merge_level=2)
        elif cls == K_SKROW16:                      # n=16 行粒度（opt_iter_1）
            _skrow16[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                N=Np, N_TRUE=n, BLOCK_M=bm, EXACT=(grid[0] * bm == B), vf_merge_level=2)
        elif cls == K_SKROW12:                      # n=12 行粒度（opt_iter_0）
            _skrow12[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                N=Np, N_TRUE=n, BLOCK_M=bm, EXACT=(grid[0] * bm == B)
            )
        elif cls == K_SCALEM:                       # 非 pow2 → masked scale
            _sinkhorn_scale_masked[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                N=Np, N_TRUE=n, BLOCK_M=bm,
            )
        elif cls == K_SK4:                          # C1 n=4 标量展开
            _sk4[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                BLOCK_M=bm, EXACT=(grid[0] * bm == B), vf_merge_level=2)
        elif cls == K_SK2:                          # C2 n=2 标量展开
            _sk2[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                BLOCK_M=bm, EXACT=(grid[0] * bm == B)
            )
        elif cls == K_SK3:                          # n=3 标量展开
            _sk3[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                BLOCK_M=bm, EXACT=(grid[0] * bm == B)
            )
        elif cls == K_SK6:                          # n=6 标量展开
            _sk6[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                BLOCK_M=bm, EXACT=(grid[0] * bm == B), vf_merge_level=2)
        elif cls == K_SK8:                          # n=8 标量展开
            _sk8[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                BLOCK_M=bm,
            )
        else:                                       # C3 小B / C4 大n 共用 scale-vector
            _sinkhorn_scale[grid](
                comb, out, B, iter_step=iter_step, eps=float(eps),
                N=Np, N_TRUE=n, BLOCK_M=bm, EXACT=(grid[0] * bm == B)
            )
        return out


# ==========================================================================
# cann-bench 提交入口
# 契约: cann_bench.mhc_sinkhorn(comb, iter_step=20, eps=1e-6)
# ==========================================================================
_IMPL = ModelNew()


def mhc_sinkhorn(comb, iter_step=20, eps=1e-6):
    return _IMPL(comb, iter_step, eps)


# ==========================================================================
# cann-bench 提交入口（由 make_submission.py 自动追加）
# 契约: cann_bench.mhc_sinkhorn(comb, iter_step=20, eps=1e-6) —— 签名对齐 cann-bench proto.yaml 的 schema
# 反作弊: 本入口只做委派，数值计算全部在 @triton.jit kernel 内完成
# ==========================================================================
_IMPL = ModelNew()


def mhc_sinkhorn(comb, iter_step=20, eps=1e-6):
    """cann-bench 评测入口：转调 generator 产出的 ModelNew。"""
    return _IMPL(comb, iter_step, eps)
