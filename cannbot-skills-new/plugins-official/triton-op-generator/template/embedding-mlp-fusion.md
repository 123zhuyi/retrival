---
name: embedding-mlp-fusion
description: Embedding + MLP 融合类算子（如 FluxTimestepGuidanceProjectionEmbedding）的 Triton Ascend 优化经验，含多 kernel 拆分、中间 dtype 强制舍入与 matmul tiling 策略。
metadata:
  type: reference
---

# Embedding + MLP 融合类算子优化经验

## 适用范围

- timestep/text embedding 后接多层 Linear + 激活 + Elementwise 的融合算子
- 典型代表：`FluxTimestepGuidanceProjectionEmbedding`
- 输入 dtype：fp16/bf16；bias dtype：fp16/bf16/fp32

## §1 Layer 1: 设计约束（硬性边界）

### L1.1 必须按 PyTorch 参考实现显式拆分独立 kernel，强制中间 dtype 舍入

- 当参考实现中存在 `output.to(dtype)` 时，Triton 编译器若将前后计算融合为单一 fp32 计算会跳过中间舍入，导致逐 bit 不匹配。
- **必须**将每个显式 cast 点拆分为独立 kernel，通过 GM store+load 往返强制中间 dtype 舍入。
- 例如：linear2/text_embedder 输出 cast 到 bias dtype，最终 add 输出 cast 到 `result_type(...)`，至少拆分为 5 个 kernel。

### L1.2 fp16/bf16 输入必须在 kernel 内升精度到 fp32 计算

- 与 transformer-inference.md T2 一致：embedding、matmul、激活、加法均应在 fp32 下完成，再 cast 回目标 dtype。
- 但 `tl.dot` 可直接对 fp16/bf16 输入使用 fp32 accumulator；若参考实现明确要求 fp32 乘法则需将输入 cast 到 fp32 后再 dot。

### L1.3 禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）

- `torch.result_type(...)`、`min()` 等计算操作必须替换为等价的 Python 分支或条件表达式。
- 仅允许 buffer 分配、形状操作（view/reshape/transpose/contiguous）、kernel 启动。

### L1.4 grid 必须钳制到物理核数，且 `num_cores` 入参等于 grid_size

- matmul-heavy kernel 使用 cube_core_num；elementwise/vector kernel 使用 vector_core_num。
- 禁止 grid 远超核数导致串行调度，也禁止 `num_cores` 与 grid_size 不匹配。

### L1.5 所有长度参数必须为 `tl.constexpr`

- `INNER_DIM`、`TIME_EMBED_DIM`、`FREQ_DIM` 等固定维度必须以 constexpr 传入，确保 `tl.arange` 和循环边界为编译期常量。

## §2 Layer 2: 算法骨架

### §2.1 Host 侧流程

```
输入 timestep[B], pooled[B, INNER_DIM], freqs[FREQ_DIM]
以及 timestep_linear1/2 的 weight/bias、text_embedder 的 weight/bias

1. 预转置权重：w1_t = timestep_linear1_weight.t().contiguous()
                w2_t = timestep_linear2_weight.t().contiguous()
                w3_t = text_embedder_weight.t().contiguous()
2. 分配中间 buffer：
   - t_emb_pre      [B, TIME_EMBED_DIM]  fp32
   - t_emb_mid      [B, INNER_DIM]       fp32
   - t_emb_out      [B, TIME_EMBED_DIM]  timestep_linear2_bias.dtype
   - text_emb       [B, TIME_EMBED_DIM]  text_embedder_bias.dtype
   - conditioning   [B, TIME_EMBED_DIM]  result_type(bias2_dtype, bias3_dtype)
3. 依次启动 5 个 kernel（见 §2.2）
```

### §2.2 Kernel 拆分骨架

| Kernel | 输入 | 输出 | 关键说明 |
|--------|------|------|----------|
| K1 timestep_embed | timestep[B], freqs[FREQ_DIM] | t_emb_pre[B, TIME_EMBED_DIM] fp32 | sin/cos/concat，全 fp32 |
| K2 linear1 + SiLU | t_emb_pre, w1_t, b1 | t_emb_mid[B, INNER_DIM] fp32 | matmul fp32 + bias + SiLU |
| K3 linear2 | t_emb_mid, w2_t, b2 | t_emb_out[B, TIME_EMBED_DIM] | cast 到 bias2 dtype |
| K4 text_embedder | pooled, w3_t, b3 | text_emb[B, TIME_EMBED_DIM] | cast 到 bias3 dtype |
| K5 add + cast | t_emb_out, text_emb | conditioning | fp32 相加后 cast 到 out_dtype |

### §2.3 Matmul kernel 骨架

```python
@triton.jit
def linear_kernel(x_ptr, w_t_ptr, b_ptr, out_ptr,
                  B, N: tl.constexpr, K: tl.constexpr,
                  ..., NUM_CORES: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    num_m_blocks = tl.cdiv(B, BLOCK_M)
    total_blocks = num_m_blocks * num_n_blocks
    for block_idx in range(pid, total_blocks, NUM_CORES):
        block_m = block_idx // num_n_blocks
        block_n = block_idx - block_m * num_n_blocks
        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < B
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_tile in range(0, K, BLOCK_K):
            offs_k = k_tile + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            b = tl.load(w_t_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                        mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.dot(a, b)
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
        # 可选 SiLU
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 acc.to(out_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])
```

## §3 Layer 3: 关键技巧

### §3.1 手动替换 `torch.result_type`

```python
# 正确：用 Python 分支在 host 侧推导最终 dtype，不调用 torch
if (bias2_dtype == torch.float32) or (bias3_dtype == torch.float32):
    out_dtype = torch.float32
elif bias2_dtype == bias3_dtype:
    out_dtype = bias2_dtype
else:
    out_dtype = torch.float32
conditioning = torch.empty((B, TIME_EMBED_DIM), device=device, dtype=out_dtype)
```

### §3.2 2D tile 选择经验

- `BLOCK_M=16, BLOCK_N=256, BLOCK_K=64` 是本算子的稳健起点：
  - b tile `[64, 256]` 约 64 KB，acc `[16, 256]` 约 16 KB，a `[16, 64]` 约 4 KB，总 UB 约 100 KB，安全。
  - `TIME_EMBED_DIM=768` 和 `INNER_DIM=3072` 均可被 256 整除，mask 覆盖率低。
- 尝试 `BLOCK_N=512` 在本算子上未带来提升，可能因为 UB 压力增大或 cube 利用率未改善。

### §3.3 对输入已是 fp32 的 matmul 保持 fp32 dot

- `linear2` 的输入 `t_emb_mid` 为 fp32，必须将 weight 也 cast 到 fp32 后再 `tl.dot`，否则 dtype 不匹配。
- `text_embedder` 的输入 `pooled_projections` 为 fp16/bf16，可直接用原生 dtype dot（fp32 accumulator），但经测试在本算子上精度与 fp32 dot 路径相近，性能提升有限。

### §3.4 grid 计算避免 `min()`

```python
# AST 预检查会 flag min() 为禁止计算；改用 if/else
tmp = m_blocks * n_blocks
if tmp < cube_cores:
    grid = tmp
else:
    grid = cube_cores
```

### §3.5 性能基准

- 设备：ascend910b3
- 60 cases 全通过
- Phase 3 基线：几何平均 `0.0493x`
- Phase 4 最佳（opt_iter_0）：几何平均 `0.4391x`
- 相对基线提升：`8.9067x`
- 未达目标加速比 `0.8x`

## §4 常见陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|----------|
| 单 kernel 融合导致中间 cast 被消除 | 编译器融合跳过 dtype 舍入 | 按参考实现的 cast 点拆 kernel |
| forward 中调用 `torch.result_type` / `min` | AST 预检查 Type-3 退化失败 | 用 Python 分支手动推导 |
| `tl.dot` 输入 dtype 不一致 | fp32 x 与 fp16 weight 混用 | 统一 cast 后再 dot |
| BLOCK_M=1 导致 cube 利用率极低 | 每个 program 只做 1 行，dot 维度太小 | 使用 BLOCK_M=16/32 的 2D tile |
| grid 远超核数 | 小 shape 上 launch 开销主导 | 钳制 grid 到 cube/vector core 数量并循环处理 |
