"""P2 Triton 骨架伪码 — fused_bias_leaky_relu
对应 mapping.md §6。y = LeakyRelu(x + bias, negative_slope) * scale
- 全程 fp32 计算域，fp16 输入仅末段单次 RTNE 舍入（对齐 kernel h:158-165）
- 属性 negative_slope/scale 为 runtime 标量（对应 tilingdata 透传），不进 constexpr
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_stages=2),
        triton.Config({"BLOCK": 4096}, num_stages=3),
        triton.Config({"BLOCK": 8192}, num_stages=3),
        triton.Config({"BLOCK": 16384}, num_stages=4),
    ],
    key=["numel"],
)
@triton.jit
def fused_bias_leaky_relu_kernel(
    x_ptr, bias_ptr, y_ptr,
    numel,
    negative_slope, scale,          # runtime 标量，对应 tilingdata.negativeSlope/scale
    BLOCK: tl.constexpr,
):
    # persistent-core 分核（2026-09-07 P4 修订，与源码 blockFactor/ubFactor 两级切分同构）：
    # grid = min(cdiv(numel, BLOCK), vector_core_num)，每 program 处理一段【连续】block 区间
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_blocks = tl.cdiv(numel, BLOCK)
    per_prog = tl.cdiv(total_blocks, num_progs)
    start_block = pid * per_prog
    end_block = tl.minimum(start_block + per_prog, total_blocks)

    for blk in range(start_block, end_block):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel             # 尾块统一 mask（对应 DataCopyPad 越界保护）

        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)      # Cast CAST_NONE 精确提升
        b = tl.load(bias_ptr + offs, mask=mask).to(tl.float32)

        t = x + b                                                 # Add
        u = tl.where(t >= 0, t, t * negative_slope)              # LeakyRelu（t=0 两分支同值）
        y = u * scale                                             # Muls

        # 可选强度折减（mapping.md §6 末行）：
        #   y = t * tl.where(t >= 0, scale, scale * negative_slope)
        # 少一次乘，但浮点舍入序列与上式不同（(t*ns)*scale vs t*(ns*scale)），
        # 验收阈值需容忍 1 ulp 差异；默认用上式保持与标杆逐点对齐。
        # benchmark 场景亦可将 negative_slope/scale 改 tl.constexpr 换常量折叠（注意重编译）。
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)  # RTNE 回写


def _vector_core_num() -> int:
    try:
        import torch_npu
        return torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except Exception:
        return 40


def fused_bias_leaky_relu(x: torch.Tensor, bias: torch.Tensor,
                          negative_slope: float = 0.2,
                          scale: float = 1.414213562373) -> torch.Tensor:
    assert x.shape == bias.shape, "no broadcast: bias must have identical shape to x"
    assert x.dtype == bias.dtype and x.dtype in (torch.float16, torch.float32)
    x = x.contiguous()
    bias = bias.contiguous()          # 对应 def.cpp AutoContiguous
    out = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:                    # 空 tensor 短路（tiling L155-164 等价）
        return out
    num_cores = _vector_core_num()    # 动态读核数，禁止硬编码
    grid = lambda meta: (min(triton.cdiv(numel, meta["BLOCK"]), num_cores),)
    fused_bias_leaky_relu_kernel[grid](x, bias, out, numel, negative_slope, scale)
    return out


# ---- 自建标杆（benchmark 库未收录本算子，§9 验收用）----
def ref_fused_bias_leaky_relu(x, bias, negative_slope=0.2, scale=1.414213562373):
    t = x.float() + bias.float()                       # fp32 计算域
    ref = torch.where(t >= 0, t, t * negative_slope) * scale
    return ref.to(x.dtype)                             # 末段 RTNE，与 kernel 对齐
