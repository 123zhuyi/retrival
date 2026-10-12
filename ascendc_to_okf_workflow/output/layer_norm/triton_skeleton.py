# P2 骨架 — layer_norm（CANN layer_norm_v4 转化）
#
# 语义：y = (x − mean)·rstd·gamma + beta，rstd = 1/sqrt(var + eps)，沿尾部 normalized_shape 折叠轴 R 归一化。
# 源锚点：src/ops-nn/norm/layer_norm_v4/
#         op_kernel/arch35/layer_norm_v4_two_pass_perf.h（全开实现，P2 主参照）
# 算法：in-kernel two-pass（官方 TwoPass 家族同构）——Σx→mean → Σ(x−mean)²→var→rstd → 归一化。
#   tl.sum 树形归约天然具备二分累加 O(log n) 误差特性（设计文档 §4 精度收益免费获得）。
# 计算域：全 fp32（f16/bf16 x 与 gamma/beta load 升 fp32，y store 降回——perf 全开实证）。
# 分核：行主语 persistent-core，grid=min(M, nvc)，program 独占一行串行扫 R（官方 R 不跨核同构）。
# 纪律：C-026 规避用 1D tile 块状 load（行内 tile 循环非标量行循环）；无 %；stride 运行传参（C-017）；
#   HAS_GAMMA/HAS_BETA/NEED_MEAN_RSTD constexpr（C-023）；rsqrt 未取证 → 用 1/tl.sqrt（T1 [SRC]）。

import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    x_ptr, y_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr,
    M, R, eps,
    HAS_GAMMA: tl.constexpr,
    HAS_BETA: tl.constexpr,
    NEED_MEAN_RSTD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 每行 3 遍扫 R（官方 TwoPass 家族同形态；官方 perf 的 (x−mean) UB 复用为 UB 容量机制 ❌ 不译）
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for m in range(pid, M, num_progs):
        row = x_ptr + m * R
        # pass 1: mean = Σx / R（fp32 域）
        s = tl.zeros([BLOCK_N], dtype=tl.float32)
        for r0 in range(0, R, BLOCK_N):
            offs = r0 + tl.arange(0, BLOCK_N)
            xv = tl.load(row + offs, mask=offs < R, other=0).to(tl.float32)
            s += xv
        mean = tl.sum(s, axis=0) / R
        # pass 2: var = Σ(x−mean)² / R → rstd = 1/sqrt(var+eps)
        v = tl.zeros([BLOCK_N], dtype=tl.float32)
        for r0 in range(0, R, BLOCK_N):
            offs = r0 + tl.arange(0, BLOCK_N)
            msk = offs < R
            xv = tl.load(row + offs, mask=msk, other=0).to(tl.float32)
            # ⚠️ 掩码元素 load 为 0（other=0）→ d=xv−mean=−mean 会误加 mean² 进 var
            #   （非 2 幂 R 尾 tile 高估 pad·mean²/R，②门实测抓出）→ 掩码处 d 必须归零
            d = tl.where(msk, xv - mean, 0.0)
            v += d * d
        var = tl.sum(v, axis=0) / R
        rstd = 1.0 / tl.sqrt(var + eps)
        # pass 3: y = (x − mean)·rstd·gamma + beta
        for r0 in range(0, R, BLOCK_N):
            offs = r0 + tl.arange(0, BLOCK_N)
            mask = offs < R
            xv = tl.load(row + offs, mask=mask, other=0).to(tl.float32)
            yv = (xv - mean) * rstd
            if HAS_GAMMA:
                g = tl.load(gamma_ptr + offs, mask=mask, other=0).to(tl.float32)
                yv = yv * g
            if HAS_BETA:
                b = tl.load(beta_ptr + offs, mask=mask, other=0).to(tl.float32)
                yv = yv + b
            tl.store(y_ptr + m * R + offs, yv.to(y_ptr.dtype.element_ty), mask=mask)
        if NEED_MEAN_RSTD:
            tl.store(mean_ptr + m, mean)
            tl.store(rstd_ptr + m, rstd)


def _num_vectorcore():
    import torch_npu  # noqa: F401

    props = torch.npu.get_device_properties(torch.npu.current_device())
    return getattr(props, "num_vectorcore", None) or getattr(props, "multi_processor_count", 32)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps: float = 1e-5,
               return_mean_rstd: bool = False):
    """y = LayerNorm(x)。对齐 GEIR LayerNormV4（def 表：mean/rstd 恒 f32；gamma/beta OPTIONAL）。

    normalized_shape: int 或 tuple（尾部维），须与 x 尾部维逐维相等。
    weight/bias: (normalized_shape,) 可选；None=恒 1/恒 0（nullptr 语义）。
    return_mean_rstd: True 时返回 (y, mean, rstd)（mean/rstd 恒 f32，(M,)）。
    """
    ndim = x.dim()
    normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    assert 0 < len(normalized_shape) <= ndim, "normalized_shape len 校验（aclnn:200-204 同构）"
    for sz, ns in zip(x.shape[-len(normalized_shape):], normalized_shape):
        assert sz == ns, f"normalized_shape 尾部维不匹配（aclnn:218-230 同构）"

    M = R = 1
    for s in x.shape[:-len(normalized_shape)]:
        M *= s
    for s in normalized_shape:
        R *= s

    device, dt = x.device, x.dtype
    has_g = weight is not None
    has_b = bias is not None
    if not x.is_contiguous():
        x = x.contiguous()

    # 空 tensor 语义（aclnn:354-374 同构）：M=0 全空；R=0 → y 空、mean=0、rstd=NaN
    if M == 0 or R == 0:
        y = x.reshape(M, R).clone()
        if return_mean_rstd:
            mean = torch.zeros(M, dtype=torch.float32, device=device)
            rstd = torch.full((M,), float("nan"), dtype=torch.float32, device=device)
            return y, mean, rstd
        return y

    y = torch.empty(M, R, dtype=dt, device=device)
    if has_g:
        w = weight.reshape(R).contiguous()
    else:
        w = x  # 占位，HAS_GAMMA=0 不读
    if has_b:
        b = bias.reshape(R).contiguous()
    else:
        b = x
    mean = torch.empty(M, dtype=torch.float32, device=device)
    rstd = torch.empty(M, dtype=torch.float32, device=device)

    nvc = _num_vectorcore()
    grid = (min(M, nvc),)
    _ln_kernel[grid](
        x, y, mean, rstd, w, b, M, R, eps,
        HAS_GAMMA=has_g, HAS_BETA=has_b, NEED_MEAN_RSTD=return_mean_rstd,
        BLOCK_N=min(1024, triton.next_power_of_2(R)),
    )
    y = y.reshape(x.shape)
    if return_mean_rstd:
        return y, mean, rstd
    return y


if __name__ == "__main__":
    # P2.5 ②正确性门：3 dtype × gamma/beta 四组合 × R 三档 × 多形态，对 fp64 参照 + torch F.layer_norm
    import os

    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch_npu  # noqa: F401
    import torch.nn.functional as F

    device = "npu:0"
    gen = torch.Generator().manual_seed(20260918)

    def ref(x, w, b, eps):
        """fp64 参照：mean/var 手算防抵消；y 用 torch F.layer_norm 双对照"""
        xf = x.reshape(M_base, R_base).double()
        mean = xf.mean(dim=1)
        var = xf.var(dim=1, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + eps)
        return mean.float(), rstd.float()

    def close(a, b, rtol, atol):
        return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)

    def run_case(tag, shape, nshape, xf, w, b, eps=1e-5, ret_std=False):
        global M_base, R_base
        nshape = (nshape,) if isinstance(nshape, int) else tuple(nshape)
        pre = shape[:-len(nshape)]
        M_base = int(torch.tensor(pre).prod()) if pre else 1
        R_base = int(torch.tensor(nshape).prod())
        x = xf.to(device)
        wd = w.to(device) if w is not None else None
        bd = b.to(device) if b is not None else None
        res = layer_norm(x, nshape, wd, bd, eps, return_mean_rstd=True)
        yn, myn, ryn = res
        yn = yn.cpu(); myn = myn.cpu(); ryn = ryn.cpu()
        total = M_base * R_base
        m_r, r_r = ref(xf, w, b, eps)
        # y 对照 torch（权重/偏置/换轴）
        xc = xf.reshape(shape).to(x.dtype)
        wc = w.reshape(nshape) if w is not None else None
        bc = b.reshape(nshape) if b is not None else None
        yt = F.layer_norm(xc, nshape, wc, bc, eps)
        # 容差按 dtype；mean/rstd 恒 f32 两态对比
        tol = {torch.float32: (1e-4, 1e-4), torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-2, 1e-2)}[x.dtype]
        ok = yn.shape == yt.shape and close(yn.reshape(-1), yt.reshape(-1), *tol)
        ok &= myn.shape == m_r.shape and close(myn, m_r, 1e-4, 1e-5)
        ok &= ryn.shape == r_r.shape and close(ryn, r_r, 1e-4, 1e-5)
        _r[0] += ok
        _r[1] += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {tag:44s} shape={tuple(shape)} ns={tuple(nshape)} "
              f"w={'Y' if w is not None else 'n'} b={'Y' if b is not None else 'n'} maxy={abs(yn.float()-yt.float()).max() if ok else float('nan'):.2e}")

    _r = [0, 0]
    f32 = lambda *s: torch.randn(*s, generator=gen)
    # —— f32：wg+wb / wg / wb / 无 ——
    run_case("f32 R=128 双参", (256, 128), 128, f32(256, 128), f32(128), f32(128))
    run_case("f32 R=64 仅 gamma", (512, 64), 64, f32(512, 64), f32(64), None)
    run_case("f32 R=32 仅 beta", (1024, 32), 32, f32(1024, 32), None, f32(32))
    run_case("f32 R=1000 无参", (40, 1000), 1000, f32(40, 1000), None, None)
    # —— R 三档：小(≤64) / 中(1024) / 大(非 2 幂) ——
    run_case("f32 R=64 小", (8, 64), 64, f32(8, 64), f32(64), f32(64))
    run_case("f32 R=2048 大", (16, 2048), 2048, f32(16, 2048), f32(2048), f32(2048))
    run_case("f32 R=700 大非2幂", (8, 700), 700, f32(8, 700), f32(700), f32(700))
    # —— 多维尾部 normalized_shape ——
    run_case("f32 ns=(16,32) 尾部多维", (6, 16, 32), (16, 32), f32(6, 16, 32), f32(16, 32), f32(16, 32))
    run_case("f32 ns=(2,4,8) 3维尾部", (3, 2, 4, 8), (2, 4, 8), f32(3, 2, 4, 8), None, None)
    # —— 恒定行 var=0 ——
    cx = torch.full((16, 128), 1.0)
    run_case("f32 恒定行 var=0", (16, 128), 128, cx, None, f32(128))
    # —— 大数值行 抗抵消 ——
    big = f32(8, 256) * 1e6
    run_case("f32 大数值 1e6", (8, 256), 256, big, None, None)
    # —— f16 / bf16 ——
    run_case("f16 R=1024 双参", (64, 1024), 1024, f32(64, 1024).half(), f32(1024).half(), f32(1024).half())
    run_case("f16 R=700 无参", (16, 700), 700, f32(16, 700).half(), None, None)
    run_case("bf16 R=512 双参", (32, 512), 512, f32(32, 512).bfloat16(), f32(512).bfloat16(), f32(512).bfloat16())
    # —— 混合精度 x=f16 gamma=f32 ——
    run_case("f16 x / f32 gamma/beta", (8, 128), 128, f32(8, 128).half(), f32(128), f32(128))
    # —— return_mean_rstd=False ——
    x1 = f32(32, 64).to(device)
    y1 = layer_norm(x1, 64, return_mean_rstd=False)
    ok1 = y1.shape == (32, 64)
    _r[0] += ok1; _r[1] += (not ok1)
    print(f"[{'PASS' if ok1 else 'FAIL'}] {'return_mean_rstd=False 仅 y':30s}")

    print(f"\n{_r[0]} PASS, {_r[1]} FAIL")
    import sys
    sys.exit(1 if _r[1] else 0)