# P2 骨架 — cumsum（CANN ops-math 转化）
#
# 语义：沿 dim 轴前缀和 y_i = Σ_{j≤i} x_j；exclusive/reverse 属性；f16/bf16 升 fp32 中间域。
# 源锚点：src/ops-math/math/cumsum/op_kernel/arch35/cumsum_base/cumsum_oneway_sklansky.h
# 分核：扫描线级 persistent-core（mapping §1）——每 program 负责 (m, n 条带) 的整条 R；
#       官方 CORE_SS 跨核 SyncAll 档 ❌ 降级为放弃跨核并行（M×cdiv(N,BLOCK_N) 不足时性能两态）。
# 扫描：块内 tl.cumsum(axis=0) + 块间 carry（= UB_SS 同构）。
#
# 【P2.5 ①编译门实测结论——本骨架按此定型】
#  1. tl.cumsum → func::CallOp 库调用（tl_op_lowering.md:70 [SRC]），UB 放大 ~48×：
#     **tile 元素数 R_CHUNK×BLOCK_N 必须 ≤1024**，否则 bishengir "ub overflow,
#     requires 6324224 bits"（需求与 tile 尺寸解耦，恒定 ~6.3Mbit 报错）。
#  2. 行序循环备选（逐行 load→carry+=row→store(carry)）**废弃**：store 与下一行 carry
#     更新在 UB 上构成 WAR，后端流水偶发插同步不足 → 输出系统性前移一行且逐 run 非确定。
#  3. Triton 前端不支持 or 链；.is_int() 本地分支也不 constexpr 折叠（跨块 dtype 不一致
#     报 Mismatched type）——dtype 分支一律走 wrapper 传 tl.constexpr 参数（IS_INT）。
# exclusive = inclusive − x（代数恒等；整型精确，fp ≤1ulp 注记）；
# reverse = 后缀和两遍法 total−prefix（纯升序寻址；降序行 tile 后端存取错误——①编译门实测）。
# 模块级常数一律 tl.constexpr（gelu JIT 教训）；无 %；mask 整型比较；int64 仅标量基址。

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 实测硬约束：R_CHUNK * BLOCK_N <= 1024（tl.cumsum CallOp UB 放大）
        triton.Config({"R_CHUNK": 16, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"R_CHUNK": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"R_CHUNK": 8, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["R", "N", "EXCLUSIVE", "REVERSE", "IS_INT"],
)
@triton.jit
def _cumsum_kernel(
    x_ptr,
    y_ptr,
    M,          # 扫描轴外维乘积
    R,          # 扫描轴长度
    N,          # 扫描轴内维乘积
    NUM_TASKS,  # M * cdiv(N, 32)（grid 先于 autotune 定，取 BLOCK_N=32 上界）
    EXCLUSIVE: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_INT: tl.constexpr,
    R_CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    per_prog = tl.cdiv(NUM_TASKS, num_progs)
    start = pid * per_prog
    nb_n = tl.cdiv(N, BLOCK_N)
    # NUM_TASKS 是按 BLOCK_N=32 估的上界；用真实 task 数钳制，防 m 越界 OOB
    end = tl.minimum(tl.minimum(start + per_prog, NUM_TASKS), M * nb_n)

    for task in range(start, end):
        # 任务解码：(m, n 条带)——标量 div，不进 mask 表达式
        m = task // nb_n
        nb = task - m * nb_n

        cols = nb * BLOCK_N + tl.arange(0, BLOCK_N)          # int32 块内偏移
        col_mask = cols < N
        base = m.to(tl.int64) * R * N                        # int64 标量基址

        # carry：fp 一律 fp32 中间域（PromoteType 同构）；整型原生
        if IS_INT:
            carry = tl.zeros([BLOCK_N], dtype=x_ptr.dtype.element_ty)
        else:
            carry = tl.zeros([BLOCK_N], dtype=tl.float32)

        num_chunks = tl.cdiv(R, R_CHUNK)
        rr = tl.arange(0, R_CHUNK)
        if REVERSE:
            # 后缀和 = total − 前缀（纯升序寻址两遍；降序行 tile 后端存取错误——实测）
            # pass1：只求 total
            total = carry
            for c in range(0, num_chunks):
                rows = c * R_CHUNK + rr
                mask = (rows < R)[:, None] & col_mask[None, :]
                tile = tl.load(x_ptr + base + rows.to(tl.int64)[:, None] * N + cols[None, :],
                               mask=mask, other=0)
                if not IS_INT:
                    tile = tile.to(tl.float32)
                total += tl.sum(tile, axis=0)
            # pass2：out = total − inclusive_prefix (+x 若 inclusive)
            carry2 = carry  # 重新从 0 起
            for c in range(0, num_chunks):
                rows = c * R_CHUNK + rr
                mask = (rows < R)[:, None] & col_mask[None, :]
                off = base + rows.to(tl.int64)[:, None] * N + cols[None, :]
                tile = tl.load(x_ptr + off, mask=mask, other=0)
                if not IS_INT:
                    tile = tile.to(tl.float32)
                s = tl.cumsum(tile, axis=0) + carry2[None, :]
                if EXCLUSIVE:
                    out = total[None, :] - s
                else:
                    out = total[None, :] - s + tile
                tl.store(y_ptr + off, out.to(y_ptr.dtype.element_ty), mask=mask)
                carry2 += tl.sum(tile, axis=0)
        else:
            for c in range(0, num_chunks):
                rows = c * R_CHUNK + rr
                mask = (rows < R)[:, None] & col_mask[None, :]
                off = base + rows.to(tl.int64)[:, None] * N + cols[None, :]
                tile = tl.load(x_ptr + off, mask=mask, other=0)
                if not IS_INT:
                    tile = tile.to(tl.float32)
                s = tl.cumsum(tile, axis=0) + carry[None, :]
                if EXCLUSIVE:
                    s = s - tile
                tl.store(y_ptr + off, s.to(y_ptr.dtype.element_ty), mask=mask)
                carry += tl.sum(tile, axis=0)


def _num_vectorcore():
    import torch_npu  # noqa: F401  # 必须：注册 NPU backend

    props = torch.npu.get_device_properties(torch.npu.current_device())
    return getattr(props, "num_vectorcore", None) or getattr(props, "multi_processor_count", 32)


def cumsum(x: torch.Tensor, dim: int, exclusive: bool = False, reverse: bool = False) -> torch.Tensor:
    """沿 dim 前缀和。输入须 contiguous。"""
    assert x.is_contiguous(), "骨架要求 contiguous 输入"
    rank = x.dim()
    d = dim % rank
    shape = x.shape
    M = 1
    for s in shape[:d]:
        M *= s
    R = shape[d]
    N = 1
    for s in shape[d + 1:]:
        N *= s
    numel = x.numel()
    assert numel < 2**31, "numel ≥ 2³¹ 超 int32 安全域（4e）：按 before 维切块调用"

    y = torch.empty_like(x)
    if numel == 0 or R == 0:
        return y

    # grid 先于 autotune 配置确定：按 BLOCK_N=32 取 NUM_TASKS 上界；kernel 内按真实 nb_n 钳制。
    num_tasks_ub = M * triton.cdiv(N, 32)
    grid = (min(num_tasks_ub, _num_vectorcore()),)

    _cumsum_kernel[grid](
        x, y, M, R, N, num_tasks_ub,
        EXCLUSIVE=exclusive, REVERSE=reverse,
        IS_INT=(not x.is_floating_point()),
    )
    return y


if __name__ == "__main__":
    # P2.5 ②正确性门：dtype × exclusive × reverse × dim 全组合小 shape 对 golden
    import os
    import sys

    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch_npu  # noqa: F401

    device = "npu:0"

    def golden(x, dim, exclusive, reverse):
        xf = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        t = xf.flip(dim) if reverse else xf
        t = torch.cumsum(t, dim=dim)
        if exclusive:
            t = torch.zeros_like(t).index_copy(
                dim, torch.arange(1, t.shape[dim], device=t.device),
                t.narrow(dim, 0, t.shape[dim] - 1))
        if reverse:
            t = t.flip(dim)
        return t.to(x.dtype)

    shapes_dims = [((64, 48), -1), ((4, 3, 40), 1), ((2, 5, 4, 24), 2), ((37, 300), 0)]
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32]
    tols = {torch.float32: 2e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.int32: 0.0}

    n_pass = n_fail = 0
    for shape, dim in shapes_dims:
        for dt in dtypes:
            for ex in (False, True):
                for rv in (False, True):
                    gen = torch.Generator().manual_seed(20260917)
                    if dt == torch.int32:
                        x = torch.randint(-50, 50, shape, generator=gen, dtype=torch.int32)
                    else:
                        x = torch.randn(shape, generator=gen, dtype=torch.float32).to(dt)
                    xn = x.to(device)
                    yn = cumsum(xn, dim, exclusive=ex, reverse=rv)
                    gn = golden(x, dim, ex, rv).to(device)
                    if dt == torch.int32:
                        ok = bool((yn.cpu() == gn.cpu()).all())
                        err = 0.0 if ok else float("inf")
                    else:
                        err = (yn.float() - gn.float()).abs().max().item()
                        ok = err <= tols[dt]
                    tag = "PASS" if ok else "FAIL"
                    if ok:
                        n_pass += 1
                    else:
                        n_fail += 1
                    print(f"[{tag}] dtype={str(dt).split('.')[-1]:8s} "
                          f"shape={str(shape):18s} dim={dim} ex={int(ex)} rv={int(rv)} max_err={err:.3e}")
    print(f"\n{n_pass} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)
