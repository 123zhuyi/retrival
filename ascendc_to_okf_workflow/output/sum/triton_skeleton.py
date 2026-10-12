# P2 骨架 — sum / reduce_sum（CANN ops-math 转化）
#
# 语义：y = Σ x over axes，keep_dims 控制维保留，空 axes 默认 noop 直返 x。
# 源锚点：src/ops-math/math/reduce_sum/op_kernel/arch35/reduce_sum_dag.h（DAG）
#         model/opbase-master-src-op_common/.../atvoss/reduce/reduce_tiling.cpp（分核/跨核）
# 路线：T=尾轴规约 (M,R)→(M,)；M=中间轴规约 (M,R,N)→(M,N)；均 A 轴全载（groupR=1 形态）。
#   非连续段 axes → wrapper permute+contiguous 归一化后走路线 T。
#   跨核两阶段（GM workspace+ScheduleMode=1）❌ 不译：单 kernel A 全载，确定性对齐 README。
# 计算域（DAG Cast→ReduceSumOp→Cast 同构）：f16/bf16/f32→fp32 累加；i32→i32；i64/bool→i64。
# 纪律：无 %（// 与 a-(a//b)*b）；stride 运行传参（C-017）；ACC 走 constexpr 码（C-023）；
#   禁原子归约（C-024+确定性声明）；i64 vreduce A2/A3 标量化预警（性能两态，①门附取证）。

import torch
import triton
import triton.language as tl


@triton.jit
def _rs_tail_kernel(
    x_ptr, y_ptr, M, R,
    ACC: tl.constexpr,      # 0=fp32 域, 1=i32 域, 2=i64 域
    BLOCK_R: tl.constexpr,
):
    # y[m] = Σ_r x[m, r]——persistent-core 扫 m，R 分块累加（A 全载）
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for m in range(pid, M, num_progs):
        if ACC == 0:
            acc = tl.zeros([BLOCK_R], dtype=tl.float32)
        elif ACC == 1:
            acc = tl.zeros([BLOCK_R], dtype=tl.int32)
        else:
            acc = tl.zeros([BLOCK_R], dtype=tl.int64)
        for r0 in range(0, R, BLOCK_R):
            offs = r0 + tl.arange(0, BLOCK_R)
            mask = offs < R
            if ACC == 0:
                t = tl.load(x_ptr + m * R + offs, mask=mask, other=0).to(tl.float32)
            elif ACC == 1:
                t = tl.load(x_ptr + m * R + offs, mask=mask, other=0)
            else:
                t = tl.load(x_ptr + m * R + offs, mask=mask, other=0).to(tl.int64)
            acc += t
        s = tl.sum(acc, axis=0)
        tl.store(y_ptr + m, s.to(y_ptr.dtype.element_ty))


@triton.jit
def _rs_mid_kernel(
    x_ptr, y_ptr, M, R, N,
    ACC: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # y[m, n] = Σ_r x[m, r, n]——program 独占 (m, n-tile)，R 分块 2D tile 累加。
    # ⚠️ 初版「标量 r 循环 + 每迭代换 base 的行 load」在 i32/N=128 实测非确定错值
    # （同输入两次运行结果不同，单 program 也复现；f16/i64 同构未触发）——疑似 C-021
    # 同族流水失同步，改 2D tile 块状加载（尾轴 kernel 同构）后 23/23 正确。
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    nb_n = tl.cdiv(N, BLOCK_N)
    NUM_TASKS = M * nb_n
    for t in range(pid, NUM_TASKS, num_progs):
        m = t // nb_n
        nb = t - m * nb_n  # C-001：禁 %，用 a-(a//b)*b
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        if ACC == 0:
            acc = tl.zeros([BLOCK_R, BLOCK_N], dtype=tl.float32)
        elif ACC == 1:
            acc = tl.zeros([BLOCK_R, BLOCK_N], dtype=tl.int32)
        else:
            acc = tl.zeros([BLOCK_R, BLOCK_N], dtype=tl.int64)
        for r0 in range(0, R, BLOCK_R):
            rows = r0 + tl.arange(0, BLOCK_R)
            ptrs = x_ptr + (m * R + rows[:, None]) * N + offs_n[None, :]
            lmask = (rows[:, None] < R) & mask_n[None, :]
            if ACC == 0:
                tv = tl.load(ptrs, mask=lmask, other=0).to(tl.float32)
            elif ACC == 1:
                tv = tl.load(ptrs, mask=lmask, other=0)
            else:
                tv = tl.load(ptrs, mask=lmask, other=0).to(tl.int64)
            acc += tv
        s = tl.sum(acc, axis=0)
        tl.store(y_ptr + m * N + offs_n, s.to(y_ptr.dtype.element_ty), mask=mask_n)


def _num_vectorcore():
    import torch_npu  # noqa: F401

    props = torch.npu.get_device_properties(torch.npu.current_device())
    return getattr(props, "num_vectorcore", None) or getattr(props, "multi_processor_count", 32)


def _acc_code(dt: torch.dtype) -> int:
    if dt in (torch.float16, torch.bfloat16, torch.float32):
        return 0
    if dt == torch.int32:
        return 1
    return 2  # int64 / bool(int8 化)


def reduce_sum(x: torch.Tensor, axes=None, keep_dims: bool = False,
               noop_with_empty_axes: bool = True, dtype: torch.dtype = None) -> torch.Tensor:
    """y = Σ x over axes。语义对齐 GEIR ReduceSum（def 表：bool→int64，其余同 x）。

    axes: None=全轴；()=依 noop_with_empty_axes（默认直返 x）；int/tuple 支持负轴。
    dtype: 显式指定输出 dtype（store 端 cast；与 aclnn「先 cast 输入」中间域不同，登记）。
    """
    ndim = x.dim()
    if axes is None:
        ax = list(range(ndim))
    else:
        ax = [axes] if isinstance(axes, int) else list(axes)
    if len(ax) == 0:
        if noop_with_empty_axes:
            return x.to(dtype) if dtype is not None else x.clone()
        ax = list(range(ndim))
    ax = sorted(a + ndim if a < 0 else a for a in ax)

    out_dt = dtype if dtype is not None else (torch.int64 if x.dtype == torch.bool else x.dtype)
    kept = [i for i in range(ndim) if i not in ax]
    out_shape = [x.shape[i] for i in kept]
    if keep_dims:
        out_shape = [1 if i in ax else x.shape[i] for i in range(ndim)]

    if x.dtype == torch.bool:
        x = x.to(torch.int8)
    if not x.is_contiguous():
        x = x.contiguous()

    # 连续段折叠：axes 成 [a..b] 连续段 → (M,R,N)；否则 permute+contiguous 归一化
    if ax == list(range(ax[0], ax[-1] + 1)):
        a, b = ax[0], ax[-1]
        M = 1
        for i in range(0, a):
            M *= x.shape[i]
        R = 1
        for i in range(a, b + 1):
            R *= x.shape[i]
        N = 1
        for i in range(b + 1, ndim):
            N *= x.shape[i]
        xf = x
    else:
        perm = kept + ax
        xf = x.permute(perm).contiguous()
        M = 1
        for i in kept:
            M *= x.shape[i]
        R = 1
        for i in ax:
            R *= x.shape[i]
        N = 1

    numel_out = M * N
    y = torch.empty(numel_out, dtype=out_dt, device=x.device)
    if numel_out == 0:
        return y.reshape(out_shape)
    if R == 0:  # 空规约轴 → 全零（语义登记）
        return torch.zeros(out_shape, dtype=out_dt, device=x.device)

    acc = _acc_code(x.dtype)
    nvc = _num_vectorcore()
    if N == 1:
        grid = (min(M, nvc),)
        _rs_tail_kernel[grid](xf, y, M, R, ACC=acc, BLOCK_R=256)
    else:
        grid = (min(M * triton.cdiv(N, 128), nvc),)
        _rs_mid_kernel[grid](xf, y, M, R, N, ACC=acc, BLOCK_R=8, BLOCK_N=128)
    return y.reshape(out_shape)


if __name__ == "__main__":
    # P2.5 ②正确性门：6 dtype × 多形态 axes 组合对 golden（官方 golden.py 同构）
    import os
    import sys

    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch_npu  # noqa: F401

    device = "npu:0"

    def golden(x, axes, keep_dims):
        """官方 golden.py 同构：f16/bf16 升 fp32 求和后回原型；bool→int64；i32 按 np 默认升 i64 再回"""
        dims = None if axes is None else (tuple(axes) if isinstance(axes, (list, tuple)) else (axes,))
        xc = x.cpu()
        if x.dtype in (torch.float16, torch.bfloat16):
            return xc.to(torch.float32).sum(dim=dims, keepdim=keep_dims).to(x.dtype)
        if x.dtype == torch.bool:
            return xc.to(torch.int64).sum(dim=dims, keepdim=keep_dims)
        if x.dtype == torch.int32:
            return xc.to(torch.int64).sum(dim=dims, keepdim=keep_dims).to(torch.int32)
        return xc.sum(dim=dims, keepdim=keep_dims)

    def close(a, b, dt):
        if dt in (torch.float32,):
            return torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-4)  # R 大时累加序噪声 ~5e-5
        if dt == torch.float16:
            return torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-3)
        if dt == torch.bfloat16:
            return torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-2)
        return bool((a == b).all())

    gen = torch.Generator().manual_seed(20260918)
    n_pass = n_fail = 0

    def check(tag, x, axes, keep_dims=False, noop=True):
        global n_pass, n_fail
        yn = reduce_sum(x.to(device), axes, keep_dims=keep_dims, noop_with_empty_axes=noop).cpu()
        if isinstance(axes, (list, tuple)) and len(axes) == 0 and noop:
            gn = x.clone()
        else:
            gn = golden(x, axes, keep_dims)
        ok = yn.shape == gn.shape and close(yn, gn, x.dtype)
        n_pass += ok
        n_fail += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {tag:40s} shape={tuple(x.shape)} axes={axes} kd={keep_dims}")

    f32 = lambda *s: torch.randn(*s, generator=gen)
    # —— 浮点家族 ——
    check("f32 尾轴 (2048,4096)→a1", f32(2048, 4096), 1)
    check("f32 首轴 (2400,16)→a0", f32(2400, 16), 0)
    check("f32 中间轴 4D (4,8,16,32)→a2", f32(4, 8, 16, 32), 2)
    check("f32 相邻多轴 4D→(1,2)", f32(8, 16, 32, 64), (1, 2))
    check("f32 非相邻多轴 4D→(0,2)", f32(4, 8, 4, 8), (0, 2))
    check("f32 负轴 -1", f32(64, 128), -1)
    check("f32 全轴 None", f32(16, 32), None)
    check("f32 keep_dims", f32(8, 16, 32), (0, 2), keep_dims=True)
    check("f32 大 R (3,100000)", f32(3, 100000), 1)
    check("f16 尾轴 (512,2048)", f32(512, 2048).half(), 1)
    check("f16 中间轴 (8,64,128)", f32(8, 64, 128).half(), 1, keep_dims=True)
    check("bf16 尾轴 (256,1024)", f32(256, 1024).bfloat16(), 1)
    check("bf16 非相邻 (2,4,2,4)→(0,2)", f32(2, 4, 2, 4).bfloat16(), (0, 2))
    # —— 整型家族 ——
    i32 = lambda *s: torch.randint(-100, 100, s, generator=gen, dtype=torch.int32)
    check("i32 尾轴 (512,2048)", i32(512, 2048), 1)
    check("i32 中间轴 (8,64,128)", i32(8, 64, 128), 1)
    check("i32 非相邻 (4,8,4,8)→(0,2)", i32(4, 8, 4, 8), (0, 2))
    i64 = lambda *s: torch.randint(-1000, 1000, s, generator=gen, dtype=torch.int64)
    check("i64 尾轴 (128,256)（标量化预警）", i64(128, 256), 1)
    check("i64 中间轴 (4,32,64)", i64(4, 32, 64), 1)
    # —— bool ——
    check("bool 尾轴 (64,128)→i64", torch.rand(64, 128, generator=gen) > 0.5, 1)
    check("bool 全轴", torch.rand(8, 16, generator=gen) > 0.5, None)
    # —— 特殊语义 ——
    check("空 axes noop 直返", f32(4, 8), ())
    check("空 axes noop=False→全轴", f32(4, 8), (), noop=False)
    check("1D (1024,)", f32(1024), 0)

    print(f"\n{n_pass} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)
