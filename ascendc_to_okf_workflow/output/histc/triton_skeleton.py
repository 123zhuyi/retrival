# P2 骨架 — histc / histogram_fixed_width（CANN ops-math 转化）
#
# 语义：y[b] = #{i : bin(x_i)=b}，[min,max] 等宽 nbins 段，越界归边。
# 源锚点：src/ops-math/math/histogram_fixed_width/op_kernel/arch35/histogram_fixed_width_simt_full_load.h:67-99
# 双域 bin 式：浮点 idx=int32((clamp(v)−min)·invRange·bins)；整型 idx=int32((clamp(v)−min)·bins//range)（int64 中间必须保留）
# 特殊值（浮点域）：+inf/NaN 丢弃；−inf 当 max=+inf 或 min=−inf 时保留，否则丢弃；
#   min=−inf→全体末 bin；max=+inf→全体首 bin（−inf 随之入首 bin）。
#   ⚠️ 源内矛盾：kernel DoBin:72-74 在 max=+inf 时仍丢 −inf，golden(spec.py:65-79) 保留——
#   骨架按 golden（验收基准+torch 语义）对齐，差异登记 extraction §9。
# 路线（HIST_IMPL）——①编译门实测结论（910_9372 + CANN 9.1.0）：
#   0 = tl.atomic_add 向量 scatter —— ❌ 实测错值：向量内重复地址的 atomic 更新被静默丢弃
#       （全同地址 1024→128=1/8；异地址 63/63 正确）。与 C-018 gather 标量化同族，已回流契约。
#       wrapper 侧 impl=0 直接 raise，kernel 保留作证据。
#   1 = 向量化 bin 归属矩阵计数（主路径）：program 独占 BLOCK_BINS 个 bin，无原子，连续 store；
#       BLOCK_X×BLOCK_BINS 须 ≤128×64（128×128 实测 ub overflow 1606400>1572864 bits）。
#   2 = 标量 atomic_add（= 官方档 2 GM_ATOMIC 真同构）：逐元素标量原子加，正确性兜底；
#       bins 大（矩阵路线比较代价 numel×BLOCK_BINS 失控）时启用。
# 官方档 0/1 的 UB 私有直方图 ❌（无可寻址 UB scratch），SyncAll 清零 ❌→wrapper torch.zeros 流内序。
# 纪律：无 %；mask 整型；dtype 分支走 wrapper 传 IS_INT constexpr（C-023）；int64 仅 bin 计算中间域（4a 预警，非回归）。

import torch
import triton
import triton.language as tl


@triton.jit
def _histc_atomic_kernel(
    x_ptr, y_ptr, NUMEL,
    min_v, max_v, inv_range, bins, range_v,
    IS_INT: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(NUMEL, BLOCK_X)
    per_prog = tl.cdiv(NUM_BLOCKS, num_progs)
    start = pid * per_prog
    end = tl.minimum(start + per_prog, NUM_BLOCKS)

    min_ninf = min_v == float("-inf")
    max_pinf = max_v == float("inf")

    for blk in range(start, end):
        offs = blk * BLOCK_X + tl.arange(0, BLOCK_X)
        mask = offs < NUMEL
        if IS_INT:
            v = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int64)
            c = tl.minimum(tl.maximum(v, min_v), max_v)
            idx = ((c - min_v) * bins // range_v).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), bins - 1)
            valid = mask
        else:
            v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            is_pinf = v == float("inf")
            is_nan = v != v
            is_ninf = v == float("-inf")
            valid = mask & (~is_pinf) & (~is_nan) & (~(is_ninf & (min_ninf == 0) & (max_pinf == 0)))
            c = tl.minimum(tl.maximum(v, min_v), max_v)
            idx = ((c - min_v) * inv_range * bins).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), bins - 1)
            idx = tl.where(min_ninf != 0, bins - 1, idx)
            idx = tl.where(max_pinf != 0, 0, idx)
        tl.atomic_add(y_ptr + idx, 1, mask=valid)


@triton.jit
def _histc_atomic_scalar_kernel(
    x_ptr, y_ptr, NUMEL,
    min_v, max_v, inv_range, bins, range_v,
    IS_INT: tl.constexpr,
):
    # 官方档 2 GM_ATOMIC 的真同构：逐元素 atomic_add。
    # 注意：纯 0-d 标量张量算术在本后端编译不过（() → tensor<1xf32> materialization 失败），
    # 故用 1 宽向量承载标量语义；1 宽向量无向量内重复地址，规避 impl=0 的丢更新坑。
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    per_prog = tl.cdiv(NUMEL, num_progs)
    start = pid * per_prog
    end = tl.minimum(start + per_prog, NUMEL)

    min_ninf = min_v == float("-inf")
    max_pinf = max_v == float("inf")

    for off in range(start, end):
        offs = off + tl.arange(0, 1)
        if IS_INT:
            v = tl.load(x_ptr + offs).to(tl.int64)
            c = tl.minimum(tl.maximum(v, min_v), max_v)
            idx = ((c - min_v) * bins // range_v).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), bins - 1)
            valid = v == v  # 恒真 mask（整型域无特殊值）
        else:
            v = tl.load(x_ptr + offs).to(tl.float32)
            is_pinf = v == float("inf")
            is_nan = v != v
            is_ninf = v == float("-inf")
            valid = (~is_pinf) & (~is_nan) & (~(is_ninf & (min_ninf == 0) & (max_pinf == 0)))
            c = tl.minimum(tl.maximum(v, min_v), max_v)
            idx = ((c - min_v) * inv_range * bins).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), bins - 1)
            idx = tl.where(min_ninf != 0, bins - 1, idx)
            idx = tl.where(max_pinf != 0, 0, idx)
        tl.atomic_add(y_ptr + idx, 1, mask=valid)


@triton.jit
def _histc_matrix_kernel(
    x_ptr, y_ptr, NUMEL,
    min_v, max_v, inv_range, bins, range_v,
    IS_INT: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_BINS: tl.constexpr,
):
    pid = tl.program_id(0)
    bin_ids = pid * BLOCK_BINS + tl.arange(0, BLOCK_BINS)   # 本 program 独占的 bin
    bin_mask = bin_ids < bins
    cnt = tl.zeros([BLOCK_BINS], dtype=tl.int32)

    min_ninf = min_v == float("-inf")
    max_pinf = max_v == float("inf")

    for start in range(0, NUMEL, BLOCK_X):
        offs = start + tl.arange(0, BLOCK_X)
        mask = offs < NUMEL
        if IS_INT:
            v = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int64)
            c = tl.minimum(tl.maximum(v, min_v), max_v)
            idx = ((c - min_v) * bins // range_v).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), bins - 1)
            valid = mask
        else:
            v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            is_pinf = v == float("inf")
            is_nan = v != v
            is_ninf = v == float("-inf")
            valid = mask & (~is_pinf) & (~is_nan) & (~(is_ninf & (min_ninf == 0) & (max_pinf == 0)))
            c = tl.minimum(tl.maximum(v, min_v), max_v)
            idx = ((c - min_v) * inv_range * bins).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), bins - 1)
            idx = tl.where(min_ninf != 0, bins - 1, idx)
            idx = tl.where(max_pinf != 0, 0, idx)
        eq = (idx[:, None] == bin_ids[None, :]) & valid[:, None]
        cnt += tl.sum(eq.to(tl.int32), axis=0)

    tl.store(y_ptr + bin_ids, cnt, mask=bin_mask)


def _num_vectorcore():
    import torch_npu  # noqa: F401

    props = torch.npu.get_device_properties(torch.npu.current_device())
    return getattr(props, "num_vectorcore", None) or getattr(props, "multi_processor_count", 32)


def histc(x: torch.Tensor, bins: int, min_v, max_v, impl: int = 1) -> torch.Tensor:
    """等宽直方图，y int32 [bins]。输入须 contiguous。语义对齐 GEIR HistogramFixedWidth。

    impl: 1=矩阵计数（主路径，bins 小）；2=标量原子加（bins 大时兜底，慢但正确）；
          0=向量 scatter 原子 —— 实测重复地址丢更新，禁用。
    """
    assert x.is_contiguous()
    assert max_v > min_v, "range max 必须 > min（tiling:108-115）"
    if impl == 0:
        raise NotImplementedError(
            "impl=0 向量 scatter tl.atomic_add 在本后端实测错值：向量内重复地址更新被静默丢弃"
            "（全同地址 1024→128=1/8，910_9372+CANN 9.1.0）。用 impl=1（bins 小）或 impl=2（bins 大）。"
        )
    numel = x.numel()
    y = torch.zeros(bins, dtype=torch.int32, device=x.device)
    if numel == 0 or bins == 0:
        return y

    is_int = not x.is_floating_point()
    if is_int:
        mn, mx = int(min_v), int(max_v)
        rng = mx - mn
        inv = 0.0
    else:
        mn, mx = float(min_v), float(max_v)
        rng = 0
        inv = float("inf") if (mx == float("inf") or mn == float("-inf")) else 1.0 / (mx - mn)
        # 注：min=-inf 或 max=+inf 时 kernel 走 where 短路，inv_range 不参与有效计算

    if impl == 2:
        grid = (min(numel, _num_vectorcore()),)
        _histc_atomic_scalar_kernel[grid](x, y, numel, mn, mx, inv, bins, rng, IS_INT=is_int)
    else:
        # BLOCK_X×BLOCK_BINS ≤ 128×64：128×128 实测 ub overflow（1606400 > 1572864 bits）
        BLOCK_BINS = min(64, triton.next_power_of_2(bins))
        BLOCK_X = 128
        grid = (triton.cdiv(bins, BLOCK_BINS),)
        _histc_matrix_kernel[grid](x, y, numel, mn, mx, inv, bins, rng,
                                   IS_INT=is_int, BLOCK_X=BLOCK_X, BLOCK_BINS=BLOCK_BINS)
    return y


if __name__ == "__main__":
    # P2.5 ②正确性门：dtype × impl × 特殊值组合对 golden（y int32 精确相等）
    import os
    import sys

    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch_npu  # noqa: F401

    device = "npu:0"

    def golden(x, bins, mn, mx):
        """spec.py golden 的同构 numpy 移植（fp32 域同序 / int64 域同式 / ±inf 三分支）"""
        import numpy as np
        xn = x.cpu().numpy()
        res = np.zeros(bins, dtype=np.int32)
        if x.dtype in (torch.float16, torch.float32):
            x32 = xn.astype(np.float32)
            mn32, mx32 = np.float32(mn), np.float32(mx)
            if mn32 == np.float32(-np.inf) or mx32 == np.float32(np.inf):
                valid = x32[~((x32 == np.float32(np.inf)) | np.isnan(x32))]
                if mn32 == np.float32(-np.inf):
                    res[bins - 1] = valid.size
                else:
                    res[0] = valid.size
                return torch.from_numpy(res)
            mask = ~((x32 == np.float32(np.inf)) | np.isnan(x32) | (x32 == np.float32(-np.inf)))
            xc = x32[mask]
            if xc.size == 0:
                return torch.from_numpy(res)
            range_v = np.float32(mx32 - mn32)
            inv = np.float32(np.float32(1.0) / range_v)
            bf = np.float32(bins)
            clamped = np.clip(xc, mn32, mx32)
            scaled = np.float32(np.float32(clamped - mn32) * inv * bf)
            idx = np.clip(scaled.astype(np.int32), 0, bins - 1)
        else:
            x64 = xn.astype(np.int64)
            mn64, mx64 = np.int64(mn), np.int64(mx)
            rng = np.int64(mx64 - mn64)
            clamped = np.clip(x64, mn64, mx64)
            idx = ((clamped - mn64) * np.int64(bins) // rng).astype(np.int32)
            idx = np.clip(idx, 0, bins - 1)
        counts = np.bincount(idx.ravel(), minlength=bins)
        res[: len(counts)] = counts[:bins].astype(np.int32)
        return torch.from_numpy(res)

    gen = torch.Generator().manual_seed(20260918)
    n_pass = n_fail = 0

    def check(tag, x, bins, mn, mx):
        global n_pass, n_fail
        for impl in (1, 2):  # impl=0 向量 scatter 原子实测错值，已禁用（见文件头注）
            yn = histc(x.to(device), bins, mn, mx, impl=impl).cpu()
            gn = golden(x, bins, mn, mx)
            ok = bool((yn == gn).all())
            n_pass += ok
            n_fail += (not ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {tag:34s} impl={impl} "
                  f"sum_y={int(yn.sum())}/{int(gn.sum())}")

    # —— 常规 case：4 dtype × 3 组 (numel, bins, range) ——
    for dt in (torch.float32, torch.float16):
        for numel, bins, mn, mx in ((4096, 64, -3.0, 3.0), (10007, 7, 0.0, 100.0), (500, 200, -1.0, 1.0)):
            x = (torch.randn(numel, generator=gen) * (mx - mn) / 4 + (mn + mx) / 2).to(dt)
            check(f"fp numel={numel} bins={bins}", x, bins, mn, mx)
    for dt in (torch.int32, torch.int64):
        for numel, bins, mn, mx in ((4096, 64, -100, 100), (10007, 7, 0, 1000), (500, 200, -5, 5)):
            x = torch.randint(mn - 20, mx + 20, (numel,), generator=gen, dtype=dt)
            check(f"int({str(dt)[6:]}) numel={numel} bins={bins}", x, bins, mn, mx)
    # —— 特殊值 case（浮点）——
    x = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.5, -0.5, 3.0, -3.0, 1e9] * 100)
    check("special: nan/±inf/越界", x, 10, -1.0, 1.0)
    check("special: min=-inf", x, 10, float("-inf"), 1.0)
    check("special: max=+inf", x, 10, -1.0, float("inf"))
    check("special: 全 NaN", torch.full((64,), float("nan")), 8, 0.0, 1.0)
    # —— bins=1 与大 bins（触发路线 1 多 program）——
    check("bins=1", torch.randn(2048, generator=gen), 1, -5.0, 5.0)
    check("bins=500>128 多 program", torch.randn(8192, generator=gen), 500, -4.0, 4.0)

    print(f"\n{n_pass} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)
