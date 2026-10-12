"""P2 Triton 骨架 — swi_glu（2026-09-17）

SwiGLU：y = SiLU(A)·B，A/B = x 沿 dim 对半切（README）。
kernel 实现形态（sigmoid 形）：y = a / (1 + e^{−a}) · b
  —— β=−1.0 由 SwigluSingle::InitGmBuffer 覆写（single.hpp:45），
     impl.hpp:29 的 beta=1.0 成员初值是诱饵（extraction §1 陷阱 1）。

设计决策（mapping.md 检索记录，P2.5 skipped: low-value）：
- **恒 fp32 计算域**：fp16 也升 fp32（SwigluVectorBF16<half,float,half>，
  与 golden 一致；与 abs 的 fp16 原生相反——陷阱 2）。vexp 仅 F16/F32
  [SRC dtype_support:157]，fp32 域同时也是后端硬约束。
- 结合序取 bf16 链：a/(1+e^{−a}) 一步 × b（少一次乘法，同低精度标杆路径）。
- 地址折算（mapping §2）：输出展平 1D，offs → i = offs // HA →
  a_off = offs + i·HA，b_off = a_off + HA（HA = half·after）。
  代数化简消掉 j/k 解码；无 %（C-001）；// 不进 mask 表达式。
- persistent-core 连续段（mapping §1 [EXP]）；mask 整型比较（§3 规则 2）；
  双缓冲由 multibuffer 默认承接（对应 isDoubleBuffer/bufferNum=2）；
  2D base-tile 优选 / DataCopyPad / DataCopy 硬件上限不复刻（mapping §6）。
- int32 安全域：numel_x < 2³¹（wrapper 断言，mapping §5）。

preflight（backend_contracts 8 项 + 4a~4e，2026-09-17）：0 真命中
（i32 向量 `//` 一次 = vdiv-i32 [SRC] 安全路径，非 mask 表达式）。
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
    ],
    key=["NUMEL_OUT", "HA"],
)
@triton.jit
def swi_glu_kernel(
    x_ptr, y_ptr,
    NUMEL_OUT, HA,                      # HA = half · after（i32 安全域内）
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(NUMEL_OUT, BLOCK)
    per_prog = tl.cdiv(NUM_BLOCKS, num_progs)
    blk_lo = pid * per_prog
    blk_hi = tl.minimum(blk_lo + per_prog, NUM_BLOCKS)

    for blk in range(blk_lo, blk_hi):
        base = blk.to(tl.int64) * BLOCK           # 标量基址
        cols = tl.arange(0, BLOCK)
        rem = tl.minimum(NUMEL_OUT - base, BLOCK).to(tl.int32)
        mask = cols < rem                          # 整型 mask（§3 规则 2）

        offs = (base + cols).to(tl.int32)          # 安全域内（wrapper 断言 < 2³¹）
        i = offs // HA                             # before 段号，vdiv-i32 [SRC]
        a_off = offs + i * HA                      # = (i·2H + j)·after + k（化简证明见 mapping §2）
        b_off = a_off + HA                         # B = A 沿 splitDim 的后半

        a = tl.load(x_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
        # sigmoid 形 SiLU 门控（bf16.hpp:62-88 结合序）：a/(1+e^{−a})·b
        y = a / (1.0 + tl.exp(-a)) * b
        tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def swi_glu(x, dim=-1):
    """y = SiLU(A)·B，A/B = x 沿 dim 对半切；x: fp16/fp32/bf16。

    dim 轴长度须为偶数；numel 须 < 2³¹（int32 安全域，mapping §5）。
    非连续输入由调用方 .contiguous()。
    """
    assert x.dtype in (torch.float16, torch.float32, torch.bfloat16)
    x = x.contiguous()
    d = dim if dim >= 0 else x.dim() + dim
    assert 0 <= d < x.dim() and x.shape[d] % 2 == 0
    before, half, after = 1, x.shape[d] // 2, 1
    for s in x.shape[:d]:
        before *= s
    for s in x.shape[d + 1:]:
        after *= s
    numel_x = x.numel()
    assert numel_x < 2**31, "int32 安全域：numel_x 须 < 2³¹（超限处置见卡片 §9）"
    numel_out = numel_x // 2

    out_shape = list(x.shape)
    out_shape[d] = half
    y = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if numel_out == 0:
        return y
    grid = (min(triton.cdiv(numel_out, 1024), NUM_VEC_CORE),)
    swi_glu_kernel[grid](x.reshape(-1), y.reshape(-1), numel_out, half * after)
    return y


def ref_swi_glu(x, dim=-1):
    """golden：复刻 tests/assets/golden.py——chunk → F.silu(a)*b，fp16/bf16 升 fp32。"""
    import torch.nn.functional as F
    orig = x.dtype
    xf = x.float()
    a, b = torch.chunk(xf, 2, dim=dim)
    return (F.silu(a) * b).to(orig)
