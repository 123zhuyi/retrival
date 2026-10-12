"""P2 Triton 骨架 — abs（2026-09-17）

y = |x| 逐元素（实数域；complex 域为 ❌ 扩展，降级方案见 mapping.md §6）。

设计决策（mapping.md 检索记录，P2.5 skipped: low-value）：
- **分档精度路径（与源码 tilingkey 三键同构）**：
  bf16 → 升 fp32 计算 + RTNE 回写（vabs 无 BF16 [SRC dtype_support:143]，
  与 AbsDag<bfloat16_t,float> / golden 升 fp32 完全一致）；
  fp16/fp32/int → 原生 tl.abs，**不升 fp32**（官方路径 AbsDag<U,U>，
  fp16 升 fp32 是偏离而非改进，extraction §7-5）。
  实现 = 单 kernel + wrapper 传 constexpr PROMOTE_F32（对应 host 侧分 key）。
- persistent-core 连续段（mapping §1 [EXP]）；int64 标量基址 + int32 块内偏移（4e）；
  mask 整型比较（§3 规则 2）；无 %；multibuffer 默认承接；无 num_stages/num_warps。
- BLOCK {1024,2048,4096,8192}：VV 32B 对齐全满足；8192 bf16 UB ~96KB < 192/248KB。

preflight（backend_contracts 8 项 + 4a~4e，2026-09-17）：0 真命中
（int64 输入 dtype 时 tl.abs 命中 vabs-i64 标量化 [SRC]——性能预警已记卡片 §10，
非违禁写法；骨架不为 int64 特化）。
"""

import torch
import triton
import triton.language as tl

try:
    import triton.runtime.driver as driver
    NUM_VEC_CORE = driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"]
except Exception:
    NUM_VEC_CORE = 48  # 兜底；查询失败时 P4 冒烟须确认


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}),
        triton.Config({"BLOCK": 2048}),
        triton.Config({"BLOCK": 4096}),
        triton.Config({"BLOCK": 8192}),
    ],
    key=["NUMEL"],
)
@triton.jit
def abs_kernel(
    x_ptr, y_ptr,
    NUMEL,
    PROMOTE_F32: tl.constexpr,   # bf16 → fp32 域（vabs 无 bf16 [SRC]）；其余 dtype False
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(NUMEL, BLOCK)
    per_prog = tl.cdiv(NUM_BLOCKS, num_progs)
    blk_lo = pid * per_prog
    blk_hi = tl.minimum(blk_lo + per_prog, NUM_BLOCKS)

    for blk in range(blk_lo, blk_hi):
        base = blk.to(tl.int64) * BLOCK           # 4e：int64 仅标量基址
        cols = tl.arange(0, BLOCK)
        rem = tl.minimum(NUMEL - base, BLOCK).to(tl.int32)
        mask = cols < rem                          # load/store mask 保整型（§3 规则 2）

        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        if PROMOTE_F32:
            # bf16 路径 = Cast NONE 升 fp32 → Abs<float> → Cast RINT 回写（abs_dag.h:28-32）
            y = tl.abs(x.to(tl.float32)).to(x_ptr.dtype.element_ty)
        else:
            # 实数原生路径 = AbsDag<U,U>（两次 Cast 恒等）
            y = tl.abs(x)
        tl.store(y_ptr + base + cols, y, mask=mask)


def abs_(x):
    """y = |x|，x: bf16/fp16/fp32/int8/int16/int32/int64（def.cpp:25）。

    非连续输入由调用方 .contiguous()。complex 不支持（❌ 域，mapping §6 降级方案）。
    """
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32,
                       torch.int8, torch.int16, torch.int32, torch.int64)
    x = x.contiguous()
    y = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:
        return y
    grid = (min(triton.cdiv(numel, 1024), NUM_VEC_CORE),)
    abs_kernel[grid](x.reshape(-1), y.reshape(-1), numel,
                     PROMOTE_F32=(x.dtype == torch.bfloat16))
    return y


def ref_abs(x):
    """golden：对齐 tests/assets/golden.py——bf16 升 fp32 计算后回原 dtype。"""
    if x.dtype == torch.bfloat16:
        return x.float().abs().to(x.dtype)
    return x.abs()
