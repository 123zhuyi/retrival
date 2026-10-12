"""P2 Triton 骨架 — gelu（2026-09-17）

tanh 近似 GELU（sigmoid 形态，gelu_dag.h:29-30）：
  y = x / (1 + exp(NEG_SQRT_EIGHT_OVER_PI * (x³ + x / 0.044715)))
    ≡ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

设计决策（mapping.md 检索记录，证据 [SRC]/[DOC]，P2.5 skipped: low-value）：
- 单 kernel 无 dtype 分支：指针推导 dtype，load → .to(fp32) → 计算 → .to(out_dtype)
  回写（RTNE 默认舍入 = 源码 Cast RINT，gelu_dag.h:80）。fp32 时 cast 恒等。
- **计算域恒 fp32**：vexp/vdiv 向量指令仅 F16/F32（dtype_support.md:157,52 [SRC]），
  bf16/fp16 必须升 fp32——与源码 DAG「Cast NONE 进 / fp32 计算 / RINT 出」天然同构。
- 常数照抄源码表达式（1/0.044715、-1.595769121*0.044715），与官方舍入路径逐位同构；
  trace 期折叠，无运行时开销。mul+add 可能被 TRITON_DEFAULT_FP_FUSION 折 fma（= Axpy 语义）。
- persistent-core 连续段：grid = min(cdiv(NUMEL, BLOCK), vector_core_num)，
  per_prog = cdiv(NUM_BLOCKS, num_progs)，禁交织 grid-stride（mapping §1 [EXP]）。
- 大 numel 防御（4e）：int64 仅限标量块基址 blk.to(tl.int64)*BLOCK，块内偏移 int32；
  mask 保持整型比较（dtype_support §3 规则 2：load/store mask 正确性优先）。
- 无 % / // 解码（一维线性寻址，C-001）；无 num_stages/num_warps（NPU 不生效）；
  双缓冲由 multibuffer 默认 true 承接 [SRC Options.td]。
- BLOCK 候选 {1024,2048,4096,8192}：全满足 VV 尾轴 32B 对齐（C-010）；
  8192 时 UB 峰值 ~96KB < 192KB(A2)/248KB(950)（mapping §4 字节校核）。

preflight（backend_contracts 8 项 + dtype_support 4a~4e，2026-09-17 对本文件执行）：
0 真命中（mask 为 int32 向量比较 = §3 规则 2 明文允许的已知例外；int64 仅标量基址）。
"""

import torch
import triton
import triton.language as tl

# 源码常数（gelu_dag.h:25-26），tl.constexpr 注解（JIT 全局变量访问要求），trace 期折叠
TANH_APPROX_FACTOR: tl.constexpr = 1.0 / 0.044715
NEG_SQRT_EIGHT_OVER_PI: tl.constexpr = -1.595769121 * 0.044715

try:
    import triton.runtime.driver as driver
    NUM_VEC_CORE = driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"]
except Exception:
    NUM_VEC_CORE = 48  # 兜底；查询失败时 grid 可能错配——P4 冒烟须确认查询成功


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
def gelu_kernel(
    x_ptr, y_ptr,
    NUMEL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(NUMEL, BLOCK)
    # 每 program 一段【连续】block 区间（mapping §1：连续段优于交织 grid-stride）
    per_prog = tl.cdiv(NUM_BLOCKS, num_progs)
    blk_lo = pid * per_prog
    blk_hi = tl.minimum(blk_lo + per_prog, NUM_BLOCKS)

    for blk in range(blk_lo, blk_hi):
        base = blk.to(tl.int64) * BLOCK           # 4e：int64 仅标量基址
        cols = tl.arange(0, BLOCK)                # int32 块内偏移
        # mask 保整型比较（§3 规则 2 已知例外）；rem 先夹回 int32 避免 i64 向量比较
        rem = tl.minimum(NUMEL - base, BLOCK).to(tl.int32)
        mask = cols < rem

        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        # Reg 链语义（gelu_dag.h:55-62）：x³ + x/0.044715 → ×(-√(8/π)·0.044715)
        # → exp → +1 → x/(·)，全程 fp32
        x3 = x * x * x
        t = (x3 + x * TANH_APPROX_FACTOR) * NEG_SQRT_EIGHT_OVER_PI
        y = x / (1.0 + tl.exp(t))
        tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def gelu(x):
    """y = GELU(x, approximate='tanh')，x: fp32/fp16/bf16 任意形状（0~8 维）。

    非连续输入由调用方 .contiguous()（aclnn 层 Contiguous 语义，不内嵌 kernel）。
    """
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16)
    x = x.contiguous()
    y = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:
        return y
    grid = (min(triton.cdiv(numel, 1024), NUM_VEC_CORE),)
    gelu_kernel[grid](x.reshape(-1), y.reshape(-1), numel)
    return y


def ref_gelu(x):
    """golden：复刻 tests/assets/golden.py:36-44，低精度升 fp32 计算后回原 dtype。"""
    orig = x.dtype
    m = torch.nn.GELU(approximate="tanh")
    return m(x.float()).to(orig)
