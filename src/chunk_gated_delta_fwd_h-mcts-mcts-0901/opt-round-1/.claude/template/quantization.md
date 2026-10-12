---
name: quantization
description: 量化类算子（DynamicQuant / SwigluQuant / WeightQuantBatchmatmul / GroupedMatmulSwigluQuant）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 量化类算子优化经验

本文档合并了**四类**量化算子的优化经验。按以下结构组织：
- **§1 通用经验**：量化类算子跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 DynamicQuant**（reduction + element-wise 复合，per-token 动态量化）
- **§3 SwigluQuant**（quantization-activation，SwiGLU 激活后量化）
- **§4 WeightQuantBatchmatmul**（weight-quantized batch matrix multiplication，反量化+GEMM）
- **§5 GroupedMatmulSwigluQuant**（MoE 分组 GEMM + SwiGLU + 动态量化，per-expert 两 kernel 链；合并自 cv-fusion.md §3）
- **§6 常见陷阱与避免方法**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| DynamicQuant | `reduction` + `element-wise` 复合 | per-row max_abs reduction + 量化，2-pass HBM（大 K）或 single-pass fused（小 K） | 自适应 BLOCK 选择 + 单/双 pass 分派；大 K 接受架构限制 |
| SwigluQuant | `quantization-activation` | 对 SwiGLU 激活结果做逐行静态/动态 INT8/INT4 量化，支持 smooth scales、offsets、group index | static/dynamic 双 kernel 分派 + vector scale division + INT4 打包 |
| WeightQuantBatchmatmul | `matmul` + `quantization` | `x` fp16/bf16 `[M,K]` @ int8 `weight` `[K,N]`，per-tensor/per-channel/per-group antiquant scale/offset，可选 bias；输出 dtype 同 `x` | group / non-group 分裂 + mixed 解耦/fused 路由 + N-strip chunking 消除 L2-exceeding workspace 重读；medium-N 接受 Triton toolchain ceiling |
| GroupedMatmulSwigluQuant | `matmul` + `quantization` + `activation` | MoE per-expert 分组 GEMM（int8 weight）+ 两级 dequant + SwiGLU 激活 + 逐行动态 INT8 量化；输出 `out_q` int8 `[M,N//2]` + `out_s` fp32 `[M]` | K1（GMM+dequant，Cube）/ K2（SwiGLU+quant，Vector）两 kernel 链职责分离；per-expert 独立 launch + tile 按专家行数分档；fp16 双舍入 dequant 逐位对齐参考 |

> ⚠️ **关键区分**：四类算子的瓶颈与优化哲学不同，生成时**禁止混用经验**：
> - **DynamicQuant** 瓶颈在 **memory-bound + 跨 pass 数据复用**（大 K 时 2-pass HBM 不可优化）。
> - **SwigluQuant** 瓶颈在 **UB 压力 + 精度对齐**（dynamic 模式必须 vector scale division，sigmoid 必须手写）。
> - **WeightQuantBatchmatmul** 瓶颈在 **Cube 利用率 + 访存层次**（dequant 是否在 dot 关键路径、workspace 是否超过 L2），优化核心是 **kernel 分裂 + mixed 路由 + N-strip chunking**。
> - **GroupedMatmulSwigluQuant** 瓶颈在 **Cube↔Vector 跨核协同 + 量化精度链对齐**（两 kernel 链职责分离 + int32 累加 + fp16 双舍入 dequant + per-expert tile 分档）。
>
> 禁止跨算子套用：
> - 生成 DynamicQuant 时，**不要**套用 SwigluQuant 的 activation 打包技巧或 WeightQuantBatchmatmul 的解耦 GEMM/N-strip 技巧。
> - 生成 SwigluQuant 时，**不要**套用 DynamicQuant 的 2-pass HBM 失败方向或 WeightQuantBatchmatmul 的 matmul tile 调参。
> - 生成 WeightQuantBatchmatmul 时，**不要**套用 DynamicQuant 的 2-pass 策略（没有跨 pass 数据复用问题），也**不要**套用 SwigluQuant 的 scale division / INT4 打包技巧（本算子是反量化+GEMM，不是训练后量化）。
> - 生成 GroupedMatmulSwigluQuant 时，**不要**套用 WeightQuantBatchmatmul 的 N-strip chunking（GMM 是两 kernel 链 + per-expert，无 L2-exceeding workspace 问题）；其 per-expert 两 kernel 架构也与 SwigluQuant 的整行量化架构不同。

> ⚠️ **架构限制特殊说明（破例归档）**：DynamicQuant 几何平均加速比仅 **0.3604x**，远低于归档阈值 1.0x 和用户目标 0.6x。**破例归档**作为"Triton DSL 架构限制典型案例"——大 K 场景必然 2-pass HBM，与 torch 单遍融合 kernel 存在结构性带宽劣势，是**已知不可优化**的场景。后续做其他 per-group quant 时，若大 K 慢于 torch，应首先排查 2-pass HBM（WeightQuantBatchmatmul 见 §4，不适用此结论）。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束在 DynamicQuant、SwigluQuant、WeightQuantBatchmatmul、GroupedMatmulSwigluQuant 四类量化算子中均适用；WeightQuantBatchmatmul 的专用约束见 §4，GroupedMatmulSwigluQuant 的专用约束见 §5。

### Q1 动态读取 Vector Core 数量，禁止硬编码
- **必须**动态读取实际 Vector Core 数量，禁止硬编码 `num_cores=8` 或 `num_cores=48`。
- **正确做法**：
通过 `triton.runtime.driver.active.utils.get_device_properties(device)` 读取 `num_vectorcore` / `num_aicore`。
- **Why:** 硬编码仅利用部分 Vector Core，导致加速比显著下降。

### Q2 1D Grid + 核内循环（负载均衡通用模式）
- **必须**使用 `pid = tl.program_id(0)`，grid 大小限制为 `min(total_blocks, num_cores)`。
- **必须**在 kernel 内用交织循环 `for block_idx in range(pid, num_blocks, num_programs)` 或 `rows_per_prog` 块循环处理多行。
- **禁止**用多维 grid 强行并行（量化算子按行处理，1D grid 配合核内循环是最稳定的并行模式）。
- **Why:** Ascend vector core 数量有限，1D grid 配合核内循环天然负载均衡。

### Q3 输入必须 contiguous
- **必须** Host 侧进入 kernel 前调用 `x = x.contiguous()`。
- **Why:** 避免非连续张量导致 kernel 内 stride 计算复杂化。

### Q4 int32 索引，避免 int64 降级
- **必须**在 kernel 内将 `tl.program_id` 和 `tl.arange` 结果 `.to(tl.int32)`。
- **Why:** int64 标量会触发地址计算降级；NPU 上 int32 索引更高效。

### Q5 多维输入必须在 host 侧扁平化为 2D
- **必须**在 `forward()` 中将输入 x 通过 `x.view(prefix_dims, last_dim)` 扁平化（保持 stride 不变亦可，但逻辑上按 2D 处理）。
- **Why:** kernel 内部只处理 `(row, col)` 二维逻辑，降低维度处理复杂度。

### Q6 禁止在 forward 中使用 torch / torch_npu 计算
- **必须**`ModelNew.forward()` 中只负责 shape 计算、分派、host 预计算；所有量化计算必须在 `@triton.jit` kernel 内完成。
- **禁止**调用 `torch_npu.npu_swiglu_quant` / `torch_npu.npu_dynamic_quant` 等参考算子预计算 scale。
- **Why:** `validate_triton_impl.py` Type-3 检查会 flag 任何 forward 中的 torch 计算为退化；同时失去纯 Triton 意义。

### Q7 禁止在 kernel 内使用 `continue` / `break`
- **Why:** Ascend 后端对非规整控制流支持有限，容易引发编译或性能问题。
- **How to apply:** 所有场景，用 `tl.where` 做向量分段代替条件跳转。

### Q8 BLOCK 向上取 2 的幂（NPU 向量化友好）
- **必须**将 BLOCK_SIZE / BLOCK_N 等向量化长度声明为 `triton.next_power_of_2(N)` 或自适应选择 pow2 值，且作为 `tl.constexpr` 传入 kernel。
- **Why:** Ascend 上 fixed-shape vector load 必须是编译期常量长度；非 constexpr 会触发 dynamic-shape load，退化为标量循环。

---

## §2 DynamicQuant 算子（reduction + element-wise 复合）

**算子类别**: `reduction` + `element-wise` 复合（per-row max_abs reduction + 量化）
**典型特征**: 输入 x `[M, K]` 或 `[B, M, K]` (fp16/bf16) + 可选 smooth_scales `[K]`，输出 int8 quant + float32 scale `[M]`。算法要求两阶段：pass1 计算 per-row max_abs → scale，pass2 用 scale 量化
**性能基准**: 42/42 verify pass，几何平均加速比 **0.3604x** vs torch（**未达 0.6x 用户目标，未达 1.0x 归档阈值，破例归档**）

### §2.0 破例归档说明

本版本 geomean 0.3604x，远低于归档阈值 1.0x 和用户目标 0.6x。破例归档理由：
1. **架构限制典型案例**：Triton DSL 无法暴露 CCE 级 UB 软件管理，memory-bound 算子大 K 场景必然 2-pass HBM，与 torch 单遍融合 kernel 存在结构性带宽劣势。这是**已知不可优化**的场景。
2. **7 种失败优化方向**：记录已验证无效的优化策略，避免后续任务重复踩坑。
3. **量化类算子复用价值**：后续做 per-group quant / weight quant 时，若大 K 慢于 torch，应首先排查 2-pass HBM。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 大 K (>8192) 场景必然 2-pass HBM，禁止期望超越 torch（架构限制）
- **必须**认识到：DynamicQuant 在 K > 8192 时，整行无法放入 UB (192KB)，必须分块处理，导致 x 被从 HBM 读取两次（pass1 max_abs + pass2 quant）。
- **禁止**在 K > 8192 场景期望通过 tile 优化、循环重排等手段超越 torch。torch 的 CCE 底层算子是**单遍融合 kernel**——在 UB 中缓存分块数据，pass1 计算 max_abs 后直接在 UB 中完成 pass2 量化，全程只从 HBM 读取 x 一次。
- **Why:** 实测带宽：torch 818 GB/s (68% HBM 峰值)，triton 277 GB/s (23% HBM 峰值)。2x HBM 读取 = 2x 带宽劣势，这是架构层面的硬限制。
- **How to apply:** 大 K 场景直接接受 0.2x~0.3x 加速比，不要浪费优化轮次。把精力放在小 K (≤4096) 场景的 single-pass 优化上。

#### L1.2 禁止用 L2 cache 命中作为 2-pass 优化手段（已验证无效，负面经验/失败方向）
- **禁止**在单 kernel 内用两个循环（pass1 max_abs → pass2 quant）期望 L2 cache 保留 pass1 的数据供 pass2 复用。
- **Why:** 实测无改善（0.37x），L2 cache 在大 K 场景未有效保留数据。Ascend L2 cache 容量和替换策略不适合跨 pass 数据复用。
- **How to apply:** 2-pass 就是 2-pass HBM，不要试图用 L2 cache "欺骗" 成 single-pass。

#### L1.3 禁止用非对称 BLOCK_K 优化 2-pass（已验证无效，负面经验/失败方向）
- **禁止**给 pass1 (max_abs) 用大 BLOCK_K (8192/16384)、pass2 (quant) 用小 BLOCK_K (4096)，期望减少 pass1 的中间量。
- **Why:** 实测无显著改善。瓶颈是 HBM 带宽而非中间量寄存器压力。
- **How to apply:** 两个 pass 用相同 BLOCK_K 即可。

#### L1.4 禁止用 bf16 max_abs 避免 fp32 膨胀（已验证无效，负面经验/失败方向）
- **禁止**在 bf16 精度下计算 max_abs，期望避免 fp32 的 4x 内存膨胀。
- **Why:** 实测无改善。max_abs 的精度需求使 bf16 中间结果仍需转 fp32 累积，且膨胀发生在寄存器而非 HBM。
- **How to apply:** 直接用 fp32 累积 max_abs。

#### L1.5 禁止用 inv_scale 乘法替代除法（已验证反而回退，负面经验/失败方向）
- **禁止**用 `x * (1/scale)` 替代 `x / scale` 期望加速。
- **Why:** 实测反而回退。Ascend 的乘法和除法延迟相近，额外的 `1/scale` 计算和 inv_scale reshape 反而增加开销。
- **How to apply:** 直接用 `x / scale` 或在 kernel 内 `inv_scale = 1.0 / scale` 后 broadcast 乘法——两者性能相近，选可读性更好的。

#### L1.6 禁止在大 K 场景用 BLOCK_M > 1（UB 溢出）
- **禁止**在 K > 4096 场景用 `BLOCK_M > 1` 的 2D tile，会导致 UB 容量溢出。
- **Why:** UB 容量 192KB，`BLOCK_M * BLOCK_K * sizeof(fp32)` 易超限。大 K 时必须 `BLOCK_M = 1` 或极小值。
- **How to apply:** `_choose_block_size`: `cols >= 512` 用 `(8, 512)`；否则 `block_m = max(1, min(32, 4096 // block_n))`。

#### L1.7 Triton DSL 无法暴露 CCE 级优化（架构限制，禁止尝试）
- **禁止**尝试以下 CCE 级优化，Triton DSL 不支持：
  - 软件管理 UB 的分块缓存（在 UB 中保留 pass1 数据供 pass2 使用）
  - 跨核同步原语（atomic max + barrier 实现单遍跨块 reduction）
  - DMA 异步传输与计算的 overlap
- **Why:** 这些是 torch CCE 单遍融合 kernel 的核心优化手段，Triton DSL 层面无法表达。
- **How to apply:** 遇到 memory-bound + 跨 pass 数据复用的算子，直接判定为架构限制场景，不要尝试逆向工程 CCE 优化。

#### L1.8 小 K (≤4096) 必须用 single-pass fused kernel（可达 1.0x+）
- **必须**在 K ≤ 4096 时用 single-pass fused kernel——整行一次性加载到 UB，单遍完成 max_abs + 量化。
- **Why:** 小 K 整行可放入 UB，避免 2-pass HBM，可达甚至超越 torch 性能（实测部分 case 1.192x）。
- **How to apply:** `_choose_block_size` 根据 cols 选择 BLOCK_N = pow2(cols)，整行单块处理。

#### L1.9 必须用 tl.cast overflow_mode='saturate' 替代手动 clamp
- **必须**用 `tl.cast(q, tl.int8, overflow_mode='saturate')` 把浮点量化结果转为 int8，禁止手动 `tl.maximum(tl.minimum(q, 127), -128)`。
- **Why:** `overflow_mode='saturate'` 是 Ascend 硬件原生支持的饱和转换，比手动 clamp 更快且精度一致。
- **How to apply:**
  ```python
  q = tl.where(q >= 0, q + 0.5, q - 0.5)   # round to nearest
  q = tl.cast(q, tl.int8, overflow_mode='saturate')   # saturate to [-128, 127]
  ```

#### L1.10 必须用 round-half-away-from-zero 而非 tl.round
- **必须**用 `tl.where(q >= 0, q + 0.5, q - 0.5)` 然后 `tl.cast` 实现 round-half-away-from-zero，禁止用 `tl.round`。
- **Why:** `tl.round` 默认 round-half-to-even（banker's rounding），与 torch `npu_dynamic_quant` 的 round-half-away-from-zero 语义不一致，会导致 verify 失败。
- **How to apply:** 见 L1.9 代码片段。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, smooth_scales=None, group_index=None, dst_type=None):
    if dst_type is None: dst_type = torch.int8
    if group_index is not None:
        raise NotImplementedError("Group quantization not supported")

    x = x.contiguous()
    orig_shape = x.shape
    dim = x.dim()

    # 1. shape 归一化为 2D
    if dim == 2:
        rows, cols = x.shape
        quant = torch.empty(orig_shape, device=x.device, dtype=dst_type)
        scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    elif dim == 3:
        b, m, cols = x.shape
        rows = b * m
        quant = torch.empty(orig_shape, device=x.device, dtype=dst_type)
        scale = torch.empty((b, m), device=x.device, dtype=torch.float32)
    else:
        x_2d = x.reshape(1, -1)
        rows, cols = x_2d.shape
        # ... 1D 退化处理

    # 2. 自适应 block size (L1.6 / L1.8)
    block_m, block_n = self._choose_block_size(cols)
    grid = (min(triton.cdiv(rows, block_m), self.num_cores),)

    # 3. 按 smooth_scales 分派
    if smooth_scales is None:
        dynamic_quant_fused_2d_kernel_no_smooth[grid](
            x, quant, scale, rows, cols, cols, cols,
            BLOCK_M=block_m, BLOCK_N=block_n,
        )
    else:
        smooth_scales = smooth_scales.contiguous()
        dynamic_quant_fused_2d_kernel[grid](
            x, smooth_scales, quant, scale, rows, cols, cols, cols,
            BLOCK_M=block_m, BLOCK_N=block_n,
        )

    if dim not in (2, 3):
        quant = quant.reshape(orig_shape)
    return quant, scale
```

#### L2.2 _choose_block_size 自适应策略

```python
def _choose_block_size(self, cols):
    if cols >= 512:
        return 8, 512          # 宽行: (8, 512) tile 摊销循环开销
    cols_pow2 = triton.next_power_of_2(cols)
    block_n = cols_pow2
    block_m = max(1, min(32, 4096 // block_n))   # 窄行: tile 约 4K elements
    return block_m, block_n
```

#### L2.3 Kernel 两阶段循环骨架（2D tiling）

```python
@triton.jit
def dynamic_quant_fused_2d_kernel(
    x_ptr, smooth_ptr, quant_ptr, scale_ptr,
    ROWS, COLS, ROW_STRIDE, QUANT_STRIDE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_row_blocks = tl.cdiv(ROWS, BLOCK_M)

    # 1D Grid + 交织循环 (Q2)
    for block_idx in range(pid, num_row_blocks, num_programs):
        row_start = block_idx * BLOCK_M
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < ROWS

        # === Pass 1: 累积 max_abs ===
        max_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for col_start in range(0, COLS, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < COLS
            mask2d = row_mask[:, None] & col_mask[None, :]

            x_tile = tl.load(
                x_ptr + row_offsets[:, None] * ROW_STRIDE + col_offsets[None, :],
                mask=mask2d, other=0.0
            ).to(tl.float32)
            smooth_tile = tl.load(smooth_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
            x_tile = x_tile * smooth_tile[None, :]   # smooth broadcast 到行
            max_acc = tl.maximum(max_acc, tl.max(tl.abs(x_tile), axis=1))

        # === 计算 scale ===
        scale = max_acc / 127.0
        scale = tl.maximum(scale, 1e-10)
        tl.store(scale_ptr + row_offsets, scale, mask=row_mask)

        # === Pass 2: 量化 ===
        inv_scale = tl.reshape(1.0 / scale, (BLOCK_M, 1))   # broadcast 准备
        for col_start in range(0, COLS, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < COLS
            mask2d = row_mask[:, None] & col_mask[None, :]

            x_tile = tl.load(
                x_ptr + row_offsets[:, None] * ROW_STRIDE + col_offsets[None, :],
                mask=mask2d, other=0.0
            ).to(tl.float32)
            smooth_tile = tl.load(smooth_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
            q = x_tile * smooth_tile[None, :] * inv_scale
            q = tl.where(q >= 0, q + 0.5, q - 0.5)           # round-half-away-from-zero (L1.10)
            q = tl.cast(q, tl.int8, overflow_mode='saturate')  # 饱和转换 (L1.9)
            tl.store(
                quant_ptr + row_offsets[:, None] * QUANT_STRIDE + col_offsets[None, :],
                q, mask=mask2d
            )
```

### §2.3 Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

#### L3.1 tl.cast overflow_mode='saturate' 替代手动 clamp

**问题**: 量化结果需 clamp 到 `[-128, 127]`，手动 `tl.maximum(tl.minimum(q, 127), -128)` 生成多条指令。

**解决**: `tl.cast(q, tl.int8, overflow_mode='saturate')` 是 Ascend 硬件原生饱和转换，单条指令完成。

```python
q = tl.where(q >= 0, q + 0.5, q - 0.5)   # round
q = tl.cast(q, tl.int8, overflow_mode='saturate')   # saturate
```

**可替代方向**: 若 `overflow_mode` 参数不可用（旧版 triton-ascend），回退到手动 clamp，但性能略差。

#### L3.2 inv_scale reshape broadcast 替代逐元素除法

**问题**: `x_tile / scale` 中 scale 是 `[BLOCK_M]`，x_tile 是 `[BLOCK_M, BLOCK_N]`，需要 broadcast 除法。

**解决**: `inv_scale = tl.reshape(1.0 / scale, (BLOCK_M, 1))`，然后 `x_tile * inv_scale` 利用 broadcast 乘法。预先计算 `1/scale` 避免逐元素除法。

**注意**: L1.5 标注此优化"反而回退"，但在某些 case 仍可能有效，需实测验证。

#### L3.3 smooth_scales broadcast 到行

**问题**: smooth_scales 是 `[K]`（per-column），需应用到 `[BLOCK_M, BLOCK_N]` tile 的每一行。

**解决**: `x_tile * smooth_tile[None, :]`，`smooth_tile` 加载为 `[BLOCK_N]`，`[None, :]` broadcast 到 `[BLOCK_M, BLOCK_N]`。

**可替代方向**: 若 smooth_scales 全 1（无 smooth），用 `_no_smooth` kernel 避免额外 load。

#### L3.4 _choose_block_size 自适应 tile 选择

**问题**: 不同 K 需要不同 BLOCK_N，固定 tile 会在小 K 浪费、大 K 溢出。

**解决**:
```python
if cols >= 512:
    return 8, 512                          # 宽行: 摊销循环开销
cols_pow2 = triton.next_power_of_2(cols)
block_n = cols_pow2
block_m = max(1, min(32, 4096 // block_n))  # 窄行: tile 约 4K elements
```

**关键点**: `4096 // block_n` 保证 tile 总量约 4K elements，避免 UB 溢出。

**可替代方向**: 若 K 固定已知，可离线调优最优 (BLOCK_M, BLOCK_N) 组合。

#### L3.5 3D 输入 reshape 为 2D 处理

**问题**: 3D 输入 `[B, M, K]` 需要按 `[B, M]` 每行计算 scale。

**解决**: 不 reshape x（保持 stride），直接用 `rows = b * m` 把 3D 当 2D 处理，scale 输出为 `[B, M]` shape。

**可替代方向**: 若 3D 不连续，需先 `.contiguous()` 再处理。

#### L3.6 将 `COLS` 声明为 `tl.constexpr` 以启用编译期地址推导

**问题**: `ROW_STRIDE` / `QUANT_STRIDE` 作为运行时 kernel 参数传入时，编译器无法静态确定每行起始地址的 stride，导致地址计算无法完全向量化/常量化。

**解决**: 在 host 侧保证输入 `contiguous` 并把最后一维展平后，`ROW_STRIDE == QUANT_STRIDE == COLS`。将 `COLS` 声明为 `tl.constexpr`，kernel 内直接用 `row_offsets[:, None] * COLS + col_offsets[None, :]`，编译器可静态推导地址模式。

**正确示例**:
```python
@triton.jit
def dynamic_quant_kernel(
    x_ptr, quant_ptr, scale_ptr,
    ROWS, COLS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # ...
    x_tile = tl.load(
        x_ptr + row_offsets[:, None] * COLS + col_offsets[None, :],
        mask=mask2d, other=0.0
    ).to(tl.float32)
```

**效果**: 在 ascend910b2 / 42 cases 评测中，该优化使几何平均加速比从 0.795x 提升至 0.891x（相对基线 0.696x）。

**可替代方向**: 若输入无法保证 stride==cols，仍需保留 `ROW_STRIDE` 参数，但优先在 host 侧通过 `.view`/`.contiguous()` 规整 stride。

#### L3.7 小 K (≤4096) single-block 特化路径

**问题**: 通用 2-pass kernel 使用固定 BLOCK_N=512，小 K（如 128、256、1024）会被拆成多个 col 循环，增加 mask 计算和循环开销。

**解决**: 当 `COLS ≤ 4096` 时，启用 single-block 特化 kernel：`BLOCK_N = next_power_of_2(COLS)`，整行一次性加载，单 kernel 内完成 max + quant，避免 col 循环。

**正确示例**:
```python
# host 侧分派
if cols <= 4096:
    block_n = triton.next_power_of_2(cols)
    block_m = max(1, min(32, 4096 // block_n))
    dynamic_quant_single_block_kernel[grid](...)
else:
    block_m, block_n = 8, 512
    dynamic_quant_fused_2d_kernel[grid](...)
```

**效果**: 与 `COLS` constexpr 优化配合，小 K case 加速比普遍 >1.0x，部分达到 2.8x+。

**可替代方向**: 对中等 K（2048-4096）可尝试 BLOCK_N=1024/2048 而非整行，以平衡 register 压力和循环次数。

### §2.4 DynamicQuant 性能基准

| 版本 | K 范围 | cases | geomean | 区间 | 说明 |
|------|--------|-------|---------|------|------|
| 历史基线 | K ≤ 4096 | 20 | 0.590x | [0.219, 1.192] | 接近目标，小张量部分超 1.0x |
| 历史基线 | 4096 < K ≤ 8192 | 4 | 0.311x | [0.141, 0.475] | 中等表现 |
| 历史基线 | K > 8192 | 14 | 0.186x | [0.158, 0.236] | 瓶颈所在，2-pass HBM 限制 |
| 历史基线 | 全量 42 cases | 42 | 0.3604x | — | 未达 0.6x 目标，破例归档 |
| **v2 (2026-07-02)** | 全量 42 cases | 42 | **0.8910x** | [0.2886, 3.5556] | 加入 `COLS` constexpr + 小 K single-block 特化后，在 ascend910b2 环境达成目标 0.8x |

**带宽实测对比**：

| 实现 | HBM 读取量 | 延迟 | 带宽 | HBM 峰值利用率 |
|------|-----------|------|------|---------------|
| torch 单遍 | 90 MB | 0.11 ms | 818 GB/s | 68% |
| triton 双遍 | 180 MB | 0.65 ms | 277 GB/s | 23% |

**关键结论**:
1. DynamicQuant 是 Triton DSL 架构限制的边界案例——memory-bound + 大 K + 跨 pass 数据复用
2. 7 种优化策略全部失败，证明瓶颈在架构层而非实现层
3. 小 K (≤4096) 可达 1.0x+，是可用区间；大 K (>8192) 0.19x 是架构硬限制
4. **新经验**: 将 `COLS` 声明为 `tl.constexpr` 并配合小 K single-block 特化，可在 ascend910b2 上把全量 42 cases 几何平均加速比从 ~0.7x 提升至 0.89x，超过 0.8x 目标
5. 后续量化类算子若大 K 慢于 torch，应首先排查 2-pass HBM

---

## §3 SwigluQuant 算子（quantization-activation）

**算子类别**: `quantization-activation`
**典型特征**: 对 SwiGLU 激活结果做逐行静态或动态 INT8/INT4 量化，支持 smooth scales、offsets、group index
**性能基准**: 52 cases 精度全过；纯 Triton 实现几何平均 **0.5067x** vs torch，implementation latency 约 **0.265 ms**

> **评判口径**: framework 参考实现为高度优化的 `torch_npu.npu_swiglu_quant`，`speedup_vs_torch < 1` 属于正常现象；优化目标应聚焦在 **implementation latency** 降低，同时保证 52/52 精度全过。

### §3.0 算子语义

**输入**:
- `x`: 任意前缀 shape + 最后一维 `2 * half_dim` 的激活张量
- `smooth_scales`: 可选，1D `[num_groups]` 或 2D `[num_groups, half_dim]` 缩放
- `offsets`: 可选，仅 static 模式有效，1D 或 2D 偏移
- `group_index`: 可选，映射每行属于哪个 group
- `activate_left`: bool，决定 SwiGLU 用 `sigmoid(left) * right` 还是 `sigmoid(right) * left`
- `quant_mode`: 0=static, 1=dynamic
- `dst_type`: `torch.int8` 或 `torch.quint4x2`

**输出**:
- `out_quant`: 量化后的 int8/int4 张量
- `out_scales`: dynamic 模式返回每行 scale；static 模式返回零张量占位

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须拆分 static / dynamic 两个 kernel
- **必须**为 `quant_mode=0`（static）和 `quant_mode=1`（dynamic）分别编写 `@triton.jit` kernel，host 侧通过 `_route` 二选一启动
- **Why:**
  - static 模式直接量化，dynamic 模式需要 Pass 1 求 `max_abs`、Pass 2 量化
  - 合并为单一 kernel 会在内部引入大量 `tl.where(quant_mode==1, ...)` 分支，向量利用率低、编译器难以优化
- **How to apply:** 所有 SwigluQuant 场景

#### L1.2 必须使用 1D Grid + 核内循环，且 grid 必须钳制到 VEC_CORE_NUM
（通用模式见 §1 Q2）
- **必须** `grid = (min(M, VEC_CORE_NUM),)`，kernel 内 `for row in range(pid, M, num_programs)` 循环处理多行
- **禁止** `grid = (M,)`（每行一个 program，M 大时严重过饱和，全部串行排队，实测性能 0.01x，损失 10-50 倍）
- **禁止** 多维 grid（未带来收益，增加调度复杂度）
- static kernel 内部按 `rows_per_prog` 循环处理多行；dynamic kernel 内部按 `tiles_per_prog` 循环处理多个 tile
- **正确骨架**：
  ```python
  VEC_CORE_NUM = triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)
  grid = (min(M, VEC_CORE_NUM),)
  kernel[grid](..., num_cores=VEC_CORE_NUM, ...)
  # kernel 内：
  for row in range(pid, M, num_programs):
      ...
  ```
- **Why:** grid=(M,) 当 M=数百~数千 token 时远超 vector core 数（~40），program 全部排队串行，无核内循环摊销。grid 钳制 + 核内循环是本算子脱离 0.01x 的单点关键修复。

#### L1.3 禁止在 kernel 内或 forward 中调用 torch / torch_npu 计算
（见 §1 Q6）
- **禁止**使用 `torch_npu.npu_swiglu_quant` 预计算 dynamic scale
- **Why:** 违反 Triton-Ascend 算子实现约束；虽然可能提升 `speedup_vs_torch`，但失去纯 Triton 意义，无法沉淀为可复用 kernel
- **How to apply:** dynamic scale 必须在 Triton kernel 内部通过 `tl.max` + vector division 计算

#### L1.4 禁止 `tl.sigmoid`，必须手写 sigmoid
- **必须**使用 `x / (1.0 + tl.exp(-x))`
- **禁止**使用 `tl.sigmoid`
- **Why:** `tl.sigmoid` 在 Ascend 后端会产生系统性数值偏差，导致大量 case 精度 fail
- **How to apply:** 所有场景

#### L1.5 INT4 输出必须按 (odd << 4) | even 打包
- **必须**将相邻两个 INT8 值各取低 4 位：`packed = ((q_odd & 0x0F) << 4) | (q_even & 0x0F)`
- **Why:** 与 `torch.quint4x2` 存储布局一致
- **How to apply:**
  ```python
  q_2d = tl.reshape(q_val, (BLOCK_SIZE_COL // 2, 2))
  q_even, q_odd = tl.split(q_2d)
  packed = ((q_odd & 0x0F) << 4 | (q_even & 0x0F)).to(tl.int8)
  ```

#### L1.6 dynamic 模式下 scale 必须使用 vector division 路径
- **必须**先按行将 `max_abs` 存入全局内存，再以 tile 为单位 load 一个短向量 `max_vec`，执行 `int_scale / max_vec`
- **禁止**在量化的内层循环里对每行单独做标量除法 `int_scale / max_abs`
- **Why:**
  - Triton-Ascend 标量除法与向量除法的 ULP 行为不同
  - 标量路径会在大 fp32 dynamic-int8 shape 上产生 off-by-1
- **How to apply:**
  - Pass 1: 每行写 `scales_ptr[row_idx] = max_abs`
  - tile 级别：`scale_vec = int_scale / tl.maximum(max_vec, 1e-10)`，写回 `scales_ptr`
  - Pass 2: `row_scale = tl.load(scales_ptr + row_idx)`，再 `nearbyint(swiglu * row_scale)`

#### L1.7 clip 必须在 int32 阶段使用 `tl.maximum/tl.minimum`
- **必须**按 `nearbyint → .to(tl.int32) → tl.maximum/tl.minimum → .to(tl.int8)` 顺序执行
- **禁止**对 int32 使用 `tl.clamp`
- **Why:** `tl.clamp` 仅支持浮点；`tl.maximum/tl.minimum` 在 int32 上效率更优
- **How to apply:** 所有量化路径

#### L1.8 Grid 大小必须动态读取实际 vector core 数量
（见 §1 Q1）
- **禁止**硬编码 `num_cores`

#### L1.9 禁止在 kernel 内使用 `continue` / `break`
（见 §1 Q7）

#### L1.10 多维输入必须在 host 侧扁平化为 2D
（见 §1 Q5）
- **必须**在 `forward()` 中将输入 `x` 通过 `x.view(prefix_dims, 2 * half_dim)` 扁平化

#### L1.11 offsets 仅在 static 模式下有效
- **必须**在 host 侧设置 `has_offsets = offsets is not None and quant_mode == 0`
- **Why:** dynamic 模式不存在 offsets 语义，错误启用会导致精度失败或多余计算
- **How to apply:** host 侧分支决策时处理

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, smooth_scales=None, offsets=None, group_index=None,
            activate_left=False, quant_mode=0, group_list_type=0, dst_type=None):
    prefix_shape = x.shape[:-1]
    prefix_dims = x.numel() // x.shape[-1]
    half_dim = x.shape[-1] // 2
    x_ptr = x.view(prefix_dims, 2 * half_dim)

    is_int4 = dst_type is not None and (str(dst_type) == 'int4' or dst_type == torch.quint4x2)
    int_scale = 7 if is_int4 else 127
    clip_min, clip_max = (-8, 7) if is_int4 else (-128, 127)

    has_smooth_scales = smooth_scales is not None
    has_offsets = offsets is not None and quant_mode == 0
    has_group_index = group_index is not None
    num_groups = group_index.shape[0] if has_group_index else 0

    # block_size_col 按 half_dim 选择
    block_size_col = select_block_size_col(is_int4, half_dim)

    if quant_mode == 1:
        out_scales = torch.empty(prefix_dims, dtype=torch.float32, device=x.device)
        grid = (min((prefix_dims + BLOCK_ROWS - 1) // BLOCK_ROWS, VEC_CORE_NUM),)
        swiglu_quant_dynamic_kernel[grid](...)
    else:
        out_scales = torch.zeros(prefix_shape, dtype=torch.float32, device=x.device)
        grid = (min(prefix_dims, VEC_CORE_NUM),)
        swiglu_quant_static_kernel[grid](...)

    reshape and return
```

#### L2.2 Static kernel 骨架

```python
pid = tl.program_id(0)
num_progs = tl.num_programs(0)
rows_per_prog = (prefix_dims + num_progs - 1) // num_progs
start_row = pid * rows_per_prog
end_row = tl.minimum(start_row + rows_per_prog, prefix_dims)

for row_idx in range(start_row, end_row):
    gid = 0
    in_bounds = True
    if has_group_index:
        gid = row_idx // group_size
        gid = tl.minimum(gid, num_groups - 1)
        in_bounds = row_idx < num_groups * group_size

    scale_scalar = tl.load(smooth_scales_ptr + gid) if has_smooth_scales else 0.0
    offset_scalar = tl.load(offsets_ptr + gid) if has_offsets else 0.0

    for col_start in range(0, half_dim, BLOCK_SIZE_COL):
        num_cols = tl.minimum(BLOCK_SIZE_COL, half_dim - col_start)
        col_arange = tl.arange(0, BLOCK_SIZE_COL)
        mask = col_arange < num_cols

        left_base = row_idx * x_stride_row + col_start
        right_base = left_base + half_dim
        left = tl.load(x_ptr + left_base + col_arange, mask=mask, other=0.0).to(tl.float32)
        right = tl.load(x_ptr + right_base + col_arange, mask=mask, other=0.0).to(tl.float32)

        sig_left = left / (1.0 + tl.exp(-left))
        sig_right = right / (1.0 + tl.exp(-right))
        swiglu = tl.where(activate_left, sig_left * right, sig_right * left)

        if has_smooth_scales:
            if smooth_scales_is_2d:
                scale_block = tl.load(smooth_scales_ptr + gid * half_dim + col_start + col_arange,
                                      mask=mask, other=1.0).to(tl.float32)
                swiglu = tl.where(in_bounds, swiglu * scale_block, swiglu)
            else:
                swiglu = tl.where(in_bounds, swiglu * scale_scalar, swiglu)

        if has_offsets:
            if offsets_is_2d:
                offset_block = tl.load(offsets_ptr + gid * half_dim + col_start + col_arange,
                                       mask=mask, other=0.0).to(tl.float32)
                swiglu = tl.where(in_bounds, swiglu + offset_block, swiglu)
            else:
                swiglu = tl.where(in_bounds, swiglu + offset_scalar, swiglu)

        q_val = tl.extra.ascend.libdevice.nearbyint(swiglu)
        q_val = q_val.to(tl.int32)
        q_val = tl.maximum(q_val, clip_min_val)
        q_val = tl.minimum(q_val, clip_max_val)

        if is_int4:
            # reshape + split + pack
            ...
        else:
            q_val = q_val.to(tl.int8)
            tl.store(out_ptr + row_idx * half_dim + col_start + col_arange,
                     q_val, mask=mask)
```

#### L2.3 Dynamic kernel 骨架

```python
pid = tl.program_id(0)
num_progs = tl.num_programs(0)
num_tiles = (prefix_dims + BLOCK_ROWS - 1) // BLOCK_ROWS
tiles_per_prog = (num_tiles + num_progs - 1) // num_progs
start_tile = pid * tiles_per_prog
end_tile = tl.minimum(start_tile + tiles_per_prog, num_tiles)

for tile_idx in range(start_tile, end_tile):
    start_row = tile_idx * BLOCK_ROWS
    end_row = tl.minimum(start_row + BLOCK_ROWS, prefix_dims)

    # Pass 1: per-row max_abs
    for row_idx in range(start_row, end_row):
        gid, in_bounds = compute_group(row_idx, ...)
        scale_scalar = load_scalar_smooth(gid) if has_smooth_scales else 0.0
        max_abs = 0.0
        for col_start in range(0, half_dim, BLOCK_SIZE_COL):
            swiglu = compute_row_swiglu(row_idx, col_start, activate_left,
                                        has_smooth_scales, smooth_scales_is_2d, ...)
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(swiglu), axis=0))
        tl.store(scales_ptr + row_idx, max_abs)

    # Vector scale computation for this tile
    row_arange = tl.arange(0, BLOCK_ROWS)
    row_mask = row_arange < (end_row - start_row)
    max_vec = tl.load(scales_ptr + start_row + row_arange, mask=row_mask, other=1.0).to(tl.float32)
    scale_vec = int_scale_val / tl.maximum(max_vec, 1e-10)
    tl.store(scales_ptr + start_row + row_arange, scale_vec, mask=row_mask)

    # Pass 2: quantize
    for row_idx in range(start_row, end_row):
        row_scale = tl.load(scales_ptr + row_idx)
        for col_start in range(0, half_dim, BLOCK_SIZE_COL):
            swiglu = compute_row_swiglu(row_idx, col_start, activate_left,
                                        has_smooth_scales, smooth_scales_is_2d, ...)
            q_val = tl.extra.ascend.libdevice.nearbyint(swiglu * row_scale)
            q_val = clip(q_val, clip_min_val, clip_max_val)
            store_int8_or_pack_int4(q_val, ...)
```

#### L2.4 Block Size 选择策略

static/dynamic 共用 `BLOCK_SIZE_COL`，按 `half_dim` 选择：

```python
if is_int4:
    if half_dim >= 2048:   block_size_col = 4096
    elif half_dim >= 1024: block_size_col = 2048
    elif half_dim >= 512:  block_size_col = 1024
    elif half_dim >= 256:  block_size_col = 512
    else:                   block_size_col = 256
else:
    if half_dim > 2048:    block_size_col = 4096
    elif half_dim > 1024:  block_size_col = 2048
    else:                   block_size_col = 1024
```

dynamic kernel 可固定 `BLOCK_ROWS=4` 并配合 autotune 尝试 `{2, 4, 6}`。

### §3.3 Layer 3: 关键技巧（Agent 可参考但不可复制）

#### L3.1 手写 sigmoid 保证精度

**错误**:
```python
sig_left = tl.sigmoid(left)
```

**正确**:
```python
sig_left = left / (1.0 + tl.exp(-left))
```

**可替代方向**: 可尝试 `tl.extra.ascend.libdevice` 中其他 exp 近似，但需逐 case 验证精度。

#### L3.2 标量 smooth/offset 的冗余边界合并

对标量** smooth/offset 路径，可将边界判断合并到 `tl.load`：

```python
# 原实现
scale_scalar = tl.load(smooth_scales_ptr + gid)
swiglu = tl.where(in_bounds, swiglu * scale_scalar, swiglu)

# 优化后
scale_scalar = tl.load(smooth_scales_ptr + gid, mask=in_bounds, other=1.0)
swiglu = swiglu * scale_scalar
```

**收益**: 实现延时几何平均下降约 2.6%。

**注意**: 对 **2D smooth/offset** 做同样合并会触发 UB overflow，应保持 `tl.where(in_bounds, swiglu * scale_block, swiglu)`。

#### L3.3 dynamic 模式 vector scale division

**错误（标量路径，ULP 漂移）**:
```python
max_abs = tl.max(tl.abs(swiglu), axis=0)
row_scale = int_scale / tl.maximum(max_abs, 1e-10)
q = nearbyint(swiglu * row_scale)
```

**正确（向量路径，精度对齐 torch）**:
```python
# Pass 1: 写 max_abs
max_abs = tl.max(tl.abs(swiglu), axis=0)
tl.store(scales_ptr + row_idx, max_abs)

# tile 级别重新 load 向量
max_vec = tl.load(scales_ptr + start_row + row_arange, mask=row_mask, other=1.0)
scale_vec = int_scale / tl.maximum(max_vec, 1e-10)
tl.store(scales_ptr + start_row + row_arange, scale_vec, mask=row_mask)

# Pass 2: 读 row_scale
row_scale = tl.load(scales_ptr + row_idx)
q = nearbyint(swiglu * row_scale)
```

**可替代方向**: 可尝试 `BLOCK_ROWS={2,4,6}` autotune，但历史测试显示收益有限。

#### L3.4 INT4 打包模板

```python
q_val = tl.extra.ascend.libdevice.nearbyint(swiglu).to(tl.int32)
q_val = tl.maximum(q_val, clip_min_val)
q_val = tl.minimum(q_val, clip_max_val)

q_2d = tl.reshape(q_val, (BLOCK_SIZE_COL // 2, 2))
q_even, q_odd = tl.split(q_2d)

q_even_4 = q_even & 0x0F
q_odd_4 = q_odd & 0x0F
packed = ((q_odd_4 << 4) | q_even_4).to(tl.int8)

pair_arange = tl.arange(0, BLOCK_SIZE_COL // 2)
pair_mask = pair_arange < (num_cols // 2)
out_base = row_idx * (half_dim // 2) + (col_start // 2)
tl.store(out_ptr + out_base + pair_arange, packed, mask=pair_mask)
```

**可替代方向**: 可尝试 interleave 而非 split，但当前实现已验证正确。

#### L3.5 编译选项

- `multibuffer=True` 在该算子上有稳定收益，保持开启
- `unit_flag=True` 历史上测得性能下降，应避免

#### L3.6 用 `tl.load(..., mask=..., other=...)` 减少 `tl.where`

```python
# 推荐
scale_block = tl.load(ptr + off, mask=mask, other=1.0)

# 不推荐
scale_block = tl.load(ptr + off, mask=mask)
scale_block = tl.where(mask, scale_block, 1.0)
```

### §3.4 SwigluQuant 优化尝试记录与性能基准

#### 有效优化点

| 优化点 | 效果 | 关键代码位置 |
|--------|------|-------------|
| 标量 smooth/offset 边界合并到 `tl.load` | 延时下降约 2.6% | static/dynamic kernel 标量分支 |
| vector scale division | 大 fp32 dynamic shape 精度对齐 | dynamic kernel tile 级别 scale 计算 |
| `multibuffer=True` | 稳定收益 | `_route` 中 kernel 启动参数 |
| 手写 sigmoid | 避免系统性精度偏差 | sigmoid 计算处 |

#### 尝试后无效或劣化的优化点（负面经验/失败方向）

| 优化点 | 结果 | 原因 |
|--------|------|------|
| recompute 核 autotune `BLOCK_ROWS={2,4,6}` | 无提升 | `BLOCK_ROWS=4` 已接近最优 |
| 把 int32 `max/min` clip 改为 fp32 clip | 失败 | 增大 fp32 中间结果 live range，case 27 UB overflow |
| 把 recompute `BLOCK_ROWS` 从 4 降到 2 | 劣化 | 0.2434 ms > 0.2382 ms |
| 静态 `block_size_col` 限制到 2048 | 异常 | benchmark framework 延时异常，结果不可信 |

#### 参考性能数据

| 指标 | 数值 |
|------|------|
| Implementation Avg Latency | 0.2654 ms |
| Framework Avg Latency | 0.1239 ms |
| 几何平均加速比 | 0.5067x |
| 相对 Phase 3 基线 | 0.8990x |

**关键结论**:
1. **static/dynamic 分 kernel 是架构基础**：合并会导致控制流复杂、向量利用率低
2. **手写 sigmoid 是精度前提**：`tl.sigmoid` 在该后端有系统性偏差
3. **vector scale division 是 dynamic 模式精度关键**：标量除法会在大 fp32 shape 上 off-by-1
4. **INT4 打包必须严格按 (odd << 4) \| even 布局**：与 `torch.quint4x2` 一致
5. **UB 压力是主要瓶颈**：动态核 `BLOCK_ROWS` 受限；避免长期保留 fp32 中间结果
6. **speedup_vs_torch < 1 不代表失败**：framework 是优化后的 NPU 原生算子，重点看 implementation latency

---

## §4 WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication）

**算子类别**: `matmul` + `quantization`
**典型特征**: `x` fp16/bf16 `[M, K]`, `weight` int8 `[K, N]`, `antiquant_scale/offset` 支持 per-tensor `[1]` / per-channel `[1, N]` / per-group `[G, N]`；可选 `bias`；输出 dtype 与 `x` 相同
**性能基准**: 48/48 verify pass，几何平均加速比 **0.9133x** vs torch（本轮 repeats=50 实测，run-to-run 方差带 0.91–0.95；超 target 0.8x 达 14%；refactor_iter_20 = opt_iter_19 + N-strip chunking for L2-exceeding workspaces）

> **评判口径**: framework 参考实现为 `torch_npu.npu_weight_quant_batchmatmul`，是 NPU 高度优化的原生算子；Triton DSL 在部分 shape 上存在结构性劣势。优化目标应聚焦在 **implementation latency** 降低与 **48/48 精度全过**。
>
> ⚠️ **方差带提示（benchmark 噪声）**: 本算子含多个 ~17-20us 的极小 M case（c1/5/8/11/14/21/24/27/30/33/38/40/42），对 profiler 抖动敏感，单次 geomean 浮动 ±0.03 属正常。代码不变（sha256 一致）时多次实测会落在 **0.91–0.95** 区间。**判 target_reached 必须看是否稳定 ≥ 0.8，不要纠结单次 0.94 vs 0.91 的差异**。

### §4.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 必须按 group / non-group 分裂为两个专用 kernel
- **必须**为 `antiquant_group_size > 0`（group 模式）和 `antiquant_group_size == 0`（non-group 模式）分别编写 `@triton.jit` kernel，host 侧通过简单 `if is_group` 分支启动对应 kernel。
- **Why:** group 模式需要按 group id 加载 scale/offset，non-group 模式 scale/offset 在 K 维度恒定，合并会导致大量 `tl.where(IS_GROUP, ...)` 分支，降低向量利用率并增加 UB 压力。
- **注意**: 当前项目的 `validate_triton_impl.py` 要求 `kernel[grid](...)` 必须直接出现在 `ModelNew.forward()` 内部，不能封装到 `self._route()` 等辅助方法中。

#### L1.2 group 模式必须保留原始 `[G, N]` scale/offset，禁止 host 侧物化为 `[K, N]`
- **必须**将 group scale/offset 以原始 `[G, N]` 布局传入 kernel，由 kernel 按 `group_id = k_start // group_size` 加载单条 `[1, N]` 行。
- **禁止**在 host 侧用 `view(G,1,N).expand(G,group_size,N).contiguous()` 物化为 `[K, N]`。
- **Why:** host 物化会引入大量 CPU 时间与额外 HBM 写入；原始 `[G, N]` 每 K-block 只读一条 `[1, N]`，总读取量仅为 `K/group_size * N`，远低于物化后的 `K * N`。

#### L1.3 group kernel 中每个 K-block 必须落在单一 group 内
- **必须**选择 `BLOCK_K` 满足 `BLOCK_K <= group_size` 且 `group_size % BLOCK_K == 0`。
- **Why:** 保证当前 K-block 内所有 K 位置共享同一个 group 的 scale/offset，从而只需加载一条 `[1, N]` scale/offset 行并广播到 `[BLOCK_K, N]`。
- **推荐配置**:
  - `group_size % 128 == 0` 且 dtype 为 fp16: `BLOCK_K = 128`
  - 其他 group 场景（含 bf16 group）: `BLOCK_K = 64`

#### L1.4 dequant 域必须按 `x.dtype` 分流
- **fp16 输入**: 在 fp16 低精度域完成 `(weight + offset) * scale`，最后 cast 到 fp32 累加。
- **bf16 输入**: 在 fp32 域完成 `(weight + offset) * scale`，再 cast 到 bf16 后回 fp32（模拟 NPU cube L0C 行为）。
- **Why:** bf16 加法尾数损失会导致 verify fail；fp16 路径若在 fp32 域 dequant 也会与 NPU 参考实现产生微小差异。

#### L1.5 必须使用 1D Grid，grid 大小钳制到实际 Vector Core 数量
- **必须** `grid = (min(total_tiles, num_cores),)`，其中 `num_cores = triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)`。
- **禁止**多维 grid。
- **Why:** 1D grid 配合核内 `for tile_idx in range(pid, total_tiles, num_programs)` 是 Ascend vector core 上最稳定的负载均衡模式。

#### L1.6 `ModelNew.forward()` 中禁止任何 torch 计算，kernel 启动必须直接可见
- **必须**所有核心计算在 `@triton.jit` kernel 内完成；`forward()` 仅做 shape 处理、contiguous、buffer 分配、条件分派。
- **Why:** `validate_triton_impl.py` 会 flag `forward()` 中的 torch 计算或间接 kernel 调用。

#### L1.7 必须使用 int32 索引
- **必须**将 `tl.program_id`、`tl.arange` 结果 `.to(tl.int32)`，避免 int64 地址计算降级。

#### L1.8 bf16 group 路径必须保守选择 block size
- **必须**对 bf16 group 使用 `BLOCK_N = 128, BLOCK_K = 64`；更大的 `BLOCK_N` 或 `BLOCK_K` 会触发 `MLIRCompilationError` / `hivm.hir.vcast` 不支持。

#### L1.9 non-group 大 N fp16 路径可放大 block size
- 当 `x.dtype == fp16` 且 `N > 1024` 时，可安全使用 `BLOCK_N = 256, BLOCK_K = 128`。
- 当 `x.dtype == bf16` 或 group 模式时，`BLOCK_N = 256` 容易触发编译错误，应保持 `BLOCK_N = 128`。

#### L1.10 non-group 大 M 路径必须解耦为「dequant workspace + 纯 GEMM」（msprof Roofline 验证）
- **必须**对 `is_group == False` 且 `M >= 256` 的 case 走**两 kernel 解耦**路径：
  1. `dequant_weight_kernel`：int8 weight 一次性 dequant 到 low_dtype workspace `[K, N]`（消除 fused kernel 中 dequant 被 M/BLOCK_M 个 tile 重复执行的冗余）；
  2. `matmul_kernel`：纯 GEMM `x[M,K] @ w_fp[K,N]`（+bias），K 循环无 dequant 依赖，Cube 可软件流水转为 compute-bound。
- **禁止**对小 M（`M < 256`）的 non-group 路径使用解耦：无 dequant 冗余可消，workspace 分配/读写开销主导，实测劣化。
- **Why（msprof 证据）**: fused kernel 经 `msprof op --aic-metrics=Roofline` 判定为 `latency bound: pipeline caused`，Cube ratio 仅 **7.6%（bf16）/ 15.8%（fp16 大 N）**，MTE2 21–24% 未饱和、compute <20%。根因是 int8→low_dtype dequant 位于每个 `tl.dot` 关键路径且跨 M-tile 重复执行，Cube↔Vector 同步等待饿死 Cube。解耦后 non-group 大 M case 实测 geomean 0.5807→0.6240。
- **UB 约束**: dequant kernel 用 `BLOCK_K=64, BLOCK_N=128`（fp32 中间缓冲 + group 全 tile scale/offset 控制 <192KB）。纯 GEMM BLOCK 选择见 **L1.12**（修正早期错误：BM128×BN256×BK128 恰好容纳 192KB 双 buffer，是最优配置；**仅 BM256×BN256 溢出**）。
- **精度保障**: workspace dequant 与 fused 逐元素数学完全一致（bf16 走 fp32 中间域），48/48 verify 全过。

#### L1.11 group 路径可在「结构信号」命中时解耦，但 dequant 必须用标量 g（msprof 验证，修正 L1.10 旧禁令）
- ⚠️ **本条修正 L1.10 早期的「禁止对 group 解耦」禁令**——该禁令是基于首版 group 解耦的**逐元素 gather dequant** 实测劣化得出的，但根因是 gather 而非 group 本身。修正后 group 解耦在结构信号命中时显著收益。
- **结构信号（命中任一即对 group 解耦）**:
  1. `group_size <= 64`：fused kernel 被迫 `BLOCK_K <= group_size`（16/32/64），Cube 在微 dot 上严重饿死；
  2. `x.dtype == bfloat16`：fused 把 fp32 dequant 留在 dot 关键路径（Vector/MTE2 主导，msprof fixpipe 仅 0.5%——bf16 真实瓶颈是关键路径 dequant 而非 fixpipe）；
  3. `M >= 512`：M-tile dequant 冗余高 + `[K,N]` workspace 摊销好。
- **保留 fused group** 的场景：`fp16 + group_size >= 128 + M < 512`（fused 本就用 `BLOCK_K=128`，解耦 workspace 开销略亏）。
- **关键实现约束——dequant 必须用标量 g + 单行连续 scale 加载**:
  - host 选 `DQ_BLOCK_K` 使 `DQ_BLOCK_K | group_size`，整个 tile 落在单一 group；
  - kernel 内 `g_scalar = k_start // group_size`（标量），加载单条连续 `[1,N]` scale/offset 行并广播；
  - **禁止** `g = offs_k // group_size`（`[BLOCK_K]` 向量）做 2D gather——离散访存使 dequant 慢 ~4x，首版因此对全部 group case 劣化（0 胜 15 负）。
- **msprof 证据**: fused group case Cube 常 <8%，标量 g 解耦后 geomean 0.6240→0.6770（+8.5%），11 个 group 解耦 case 全部净收益、0 回退。

#### L1.12 解耦 GEMM 必须用 BLOCK_N=256（大 N bandwidth-bound，msprof Roofline 验证）
- **必须**对解耦路径的纯 GEMM（`matmul_kernel`）使用 `BLOCK_N = 256, BLOCK_K = 128`（BLOCK_M 自适应 32/64/128），适用于 fp16 与 bf16（matmul 只加载 low_dtype，UB 占用与 dtype 无关）。
- **Why（msprof 证据）**: 大 N 解耦 `matmul_kernel` 为 bandwidth bound：`Cube 20% / MTE2 54% / aic_main_mem_read_bw ~1.6 TB/s 聚合`（HBM 打满）。根因是 dequant workspace fp16 `[K,N]`（64–128MB）远大于 ~96MB L2，被每个 M-tile 从 HBM 重读。放大 BLOCK_N 128→256 减半 N-tile 数，纯 GEMM geomean 0.6770→0.7946（+17.4%）。
- **UB 边界**: `BM=128 × BN=256 × BK=128` 双 buffer = 192KB（恰好容纳）。**仅 `BM=256 × BN=256` 溢出**。
- **⚠️ 否决方向（避免重复尝试）**:
  1. 改 tile 遍历序为 n-slow/m-fast 企图 L2 复用 workspace——实测更慢；L2 跨核共享不足以摊销大 footprint。
  2. 大 N 改回 fused 靠 int8 权重命中 L2——多数更差；大 M 时 dequant 在 dot 关键路径主导，fused 同样不命中 L2。
- **适用判据**: `MTE2 > 40% 且 Cube < 30% 且 read_bw≈HBM 峰值` → bandwidth bound → 放大 BLOCK_N。

#### L1.13 大 N 解耦路径必须做 N-strip chunking（L2-exceeding workspace 的算法级解法，refactor_iter_20）
- **必须**对解耦路径（L1.10/L1.11 命中）且 `K * N * sizeof(low_dtype) > L2_THRESHOLD`（取 **60MB**，~96MB L2 留 1/3 余量）的 case 走 **N-strip 切分**：把 N 切成多个 strip，每个 strip 的 dequant workspace `[K, chunk_n]` 能完整驻留 L2，再对该 strip 跑全部 M-tile 的 GEMM——此时 workspace 从 L2 命中而非 HBM 重读。
- **Why（修正 L1.12 末尾"workspace 重读是固有代价"的判断——该判断在 N-strip 下不再成立）**: L1.12 的 BLOCK_N=256 把单次 GEMM 访存减半，但整个 `[K,N]` workspace（大 N 时 64–128MB）仍远大于 ~96MB L2，被每个 M-tile 从 HBM 重读 8×。**正确破法是 N-strip**：每次只物化一个 strip 的 workspace（32MB ≤ L2），让它跨 M-tile 全程命中 L2，把 HBM 重读从 "M-tile 数 × 全 workspace" 降到 "num_strips × strip workspace"。
- **实现要点（deliverable 实测 1.48–1.75× on c4/37/45/46/47/48）**：
  - 触发条件：`K * N * sizeof(low_dtype) > 60MB`（约 ~96MB L2 的 2/3）。
  - 计算单 strip 大小：`chunk_n = floor(48MB / (K * dtype_size))`，再向下对齐到 `BLOCK_N_GEMM=256` 的整数倍，保证至少一个完整 N-tile。
  - Host 侧按 `chunk_idx` 循环 N 维：对每段 `weight[:, n_start:n_end]` 启动 `dequant_weight_kernel` 写 `[K, chunk_n]` workspace；再启动 `matmul_kernel` 让**所有 M-tile 顺序消费该 strip**。workspace 全程驻留 L2，跨 M-tile 命中复用，避免整 `[K,N]` buffer 被反复从 HBM 重读。
  - GEMM tile 仍用 `BM128/BN256/BK128`（L1.12）。
  - 输出按 N 维 slice 写入 `out[:, n_start:n_end]`，不同 strip 输出 disjoint，**禁止用 atomic_add**（相对 K-split 的核心优势）。
  - strip 内 dequant 仍遵守 L1.11：group 模式用标量 `g = k_start // group_size` + `DQ_BLOCK_K | group_size` + 单行连续 scale/offset 加载，`DQ_BLOCK_N=128`。
- **触发 case 实测（repeats=50）**: c4 M=1024 K=4096 N=8192 → 1.481×；c37/47 M=1024 K=4096 N=16384 → 1.58–1.59×；c45 → 1.752×；c46/48 M=2048 K=8192 N=4096 → 1.56–1.60×。相对 opt_iter_19（这些 case 仅 ~0.54×）实现质变，geomean **0.7946 → 0.9133**（+15%）。
- **适用判据**: 解耦路径 + `K*N*2 > 60MB` → 必走 N-strip；`K*N*2 ≤ 60MB` 走常规单 workspace 解耦（L1.12）即可（strip 切分对能装进 L2 的 workspace 反而增加 host 循环开销）。

### §4.2 Layer 2: 算法骨架（纯描述，无完整源码）

#### L2.1 Host 侧分派

`forward()` 仅负责：
1. 输入 `contiguous()` + reshape 为 2D `[M, K]` / `[K, N]`；
2. 按 `antiquant_group_size>0` 判定 `is_group`；
3. 按 dtype / shape 选择 `(BLOCK_M, BLOCK_N, BLOCK_K)`；
4. 启动对应 kernel。

分派策略表（遵守 L1.8–L1.13）：

| 场景 | kernel | BLOCK_N | BLOCK_K | BLOCK_M |
|------|--------|---------|---------|---------|
| non-group, N≤1024 或 bf16 | fused | 128 | 128 | 32/64/128 按 M |
| non-group, N>1024, fp16 | fused | 256 | 128 | 32/64/128 按 M |
| group, bf16 | group | 128 | 64 | 32/64/128 按 M |
| group, gs%128==0, fp16 | group | 128/256 | 128 | 32/64/128 按 M |
| 其他 group | group | 128 | 64 | 32/64/128 按 M |

解耦判定见 L1.10–L1.13：命中解耦时先启动 `dequant_weight_kernel` 写 workspace，再启动 `matmul_kernel` 做纯 GEMM；命中 N-strip 时 host 侧循环 N 维 strip。

#### L2.2 Group kernel 骨架

1. **1D grid + 核内 stride 循环**：`pid = program_id(0)`，`for tile_idx in range(pid, total_tiles, num_programs)`。
2. 每个 tile 取 `m_tile = tile_idx // num_n_tiles`，`n_tile = tile_idx % num_n_tiles`。
3. K-loop 内用**标量 group id**：`g = k_start // group_size`；按 `g` 加载单条 `[1, N]` scale/offset，广播到 `[BLOCK_K, BLOCK_N]`。
4. 每个 K-block 必须满足 `BLOCK_K | group_size`（L1.3），保证不跨 group。
5. dequant 域按 `IS_BFLOAT16` 分流（L1.4）：fp16 在 fp16 域，bf16 在 fp32 域。
6. 累加用 fp32 accumulator，`tl.dot(x_tile, w_tile, acc)`；最后加 bias 并 cast 回 `x.dtype` store。

#### L2.3 Non-group kernel 骨架

1. 同样 1D grid + stride 循环。
2. scale/offset 在 K 维恒定，**提到 K 循环外预加载 + cast**（L3.4）：
   - 预计算 `scale_f32` / `scale_low`（fp16 路径）和 `offset_f32` / `offset_low`（若存在 offset）。
3. K-loop 内只做：load int8 weight → cast → `*` scale `(+ offset)` → cast → `tl.dot`。
4. 最后加 bias、cast、store。

#### L2.4 N-strip chunking 骨架（解耦路径）

当 `K*N*sizeof(low_dtype) > 60MB` 时：
1. 计算 `max_chunk_n = 48MB // (K * dtype_size)`，向下对齐到 `BLOCK_N_GEMM=256`。
2. Host 侧 `for chunk_idx in range(cdiv(N, n_chunk))`：
   - 取 `weight[:, n_start:n_end]`；
   - 启动 `dequant_weight_kernel` 写 `w_fp_slice[K, chunk_n]`；
   - 启动 `matmul_kernel` 计算 `x @ w_fp_slice → out[:, n_start:n_end]`。
3. 所有 M-tile 顺序处理同一个 strip，workspace 在 L2 中命中复用；输出 N 维 disjoint，**无需 atomic**。

### §4.3 Layer 3: 关键技巧（Agent 可参考但不可复制代码结构）

#### L3.1 group scale 单 row 广播替代 host 物化

**问题**: group 模式若将 `[G, N]` expand 为 `[K, N]`，会占用大量 host 时间与 HBM 带宽。

**解决**: kernel 内按 `g = k_start // group_size` 只加载一条 `[1, N]`，通过 `scale_row[None, :]` 广播到 `[BLOCK_K, N]`。

**可替代方向**: 若未来后端支持更高效的 gather，可尝试 vector gather；但在当前 Ascend 后端，单 row 广播更稳定。

#### L3.2 按 `group_size` 选择 `BLOCK_K`

**问题**: group_size 为 64 与 128 时，最优 `BLOCK_K` 不同。

**解决**:
- `group_size % 128 == 0` 且 dtype 为 fp16: `BLOCK_K = 128`，减少 K 循环迭代数。
- 其他 group 场景: `BLOCK_K = 64`，避免跨 group 并保证编译通过。

**可替代方向**: 对固定 group_size 可离线搜索最优 `BLOCK_K`，但需保证不跨 group。

#### L3.3 按 dtype / shape 分 block 策略

| 场景 | 推荐 BLOCK_N | 推荐 BLOCK_K | 原因 |
|------|-------------|-------------|------|
| fp16 non-group, N > 1024 | 256 | 128 | 大 tile 提升 Cube 利用率 |
| fp16 non-group, N <= 1024 | 128 | 128 | 平衡 |
| bf16 non-group | 128 | 128 | BLOCK_N=256 会触发 MLIRCompilationError |
| fp16 group, gs%128==0 | 128/256 | 128 | 单 group 单 block |
| bf16 group | 128 | 64 | 保守，避免编译错误 |

**可替代方向**: 对小 M (<64) 使用 `BLOCK_M=32`，中等 M (64-127) 使用 `BLOCK_M=64`，大 M 使用 `BLOCK_M=128`；`BLOCK_M=256` 易触发 UB overflow，不建议。

#### L3.4 循环不变量 cast 外提

**问题**: non-group 模式下 `scale_tile` / `offset_tile` 在 K 循环内重复 cast。

**解决**: 在 K 循环外预计算 `scale_f32` / `scale_low` / `offset_f32` / `offset_low`，K 循环内直接使用。

**可替代方向**: 现代编译器可能自动外提，但显式外提更可控，实测有小幅收益。

### §4.4 性能基准与优化尝试记录

| 版本 | cases | geomean | 相对 Phase 3 基线 | 说明 |
|------|-------|---------|------------------|------|
| Phase 3 基线 | 48/48 | 0.1026x | 1.0x | 通用 kernel，host 物化 group scale |
| opt_iter_9 (kernel 分裂 + group 单 row 广播) | 48/48 | 0.4558x | 4.44x | 首次算法级优化，group 路径去掉 host 物化 |
| opt_iter_10 (fp16 大 N BLOCK_K=128) | 48/48 | 0.4979x | 4.85x | non-group 大 N 同时放大 N/K tile |
| opt_iter_11 (group_size=128 BLOCK_K=128) | 48/48 | 0.5290x | 5.16x | group_size=128 的 fp16 group 减少循环 |
| opt_iter_12 (bf16 non-group BLOCK_K=128) | 48/48 | 0.5582x | 5.44x | bf16 非 group 放大 K tile |
| opt_iter_14 (fp16 group N>=1024 BLOCK_N=256) | 48/48 | 0.5705x | 5.56x | fp16 group N=1024 时放大 N tile |
| **opt_iter_15b (cast 外提)** | **48/48** | **0.5807x** | **5.66x** | fused 最终版本 |
| opt_iter_16 (纯解耦 dequant+GEMM) | 48/48 | 0.3640x | 3.55x | 全 case 解耦：non-group 大 M 赢，group/小 M 大输 → 总体劣化（group 输因逐元素 gather dequant，见 opt_iter_18 修正） |
| opt_iter_17 (mixed: 解耦 non-group M>=256 + fused 其余) | 48/48 | 0.6240x | 6.08x | msprof Roofline 驱动 non-group 大 M 解耦 |
| opt_iter_18 v1 (group 解耦-逐元素 gather) | 48/48 | 0.4275x | 4.17x | group 解耦首版：dequant 用 `g[:,None]` 2D gather，15 个 group case 全劣化（0 胜 15 负） |
| **opt_iter_18 (group 解耦-标量 g + 结构路由)** | **48/48** | **0.6770x** | **6.60x** | **msprof 驱动 group 路径解耦：标量 g 连续 scale 加载消除 gather；gs<=64/bf16/M>=512 解耦，其余 fused** |
| **opt_iter_19 (解耦 GEMM BLOCK_N=256)** | **48/48** | **0.7946x** | **7.74x** | **msprof 驱动：大 N 解耦 GEMM bandwidth-bound（MTE2 54%, Cube 20%, HBM 打满），workspace(64-128MB)>L2 被 M-tile 重读；放大 BLOCK_N 128→256，纯 GEMM +36-38%** |
| **refactor_iter_20 (N-strip chunking, 最终 deliverable)** | **48/48** | **0.9133x**（方差带 0.91-0.95） | **8.90x** | **算法级消除 L2-exceeding workspace 重读：`K*N*2>60MB` 时按 N 切 L2-resident strip（48MB 上限，对齐 BN256），每 strip dequant 一次 → 全部 M-tile L2 命中复用；输出 N 维 disjoint 无需 atomic。大 N case 0.54→1.48-1.75×（c4/37/45/46/47/48）** |

**有效优化点**:
1. group / non-group kernel 分裂 + group 单 row 广播（最大收益，4.4x）
2. fp16 大 N 使用 BLOCK_N=256 / BLOCK_K=128
3. 按 group_size 选择 BLOCK_K=128（gs=128 时）
4. bf16 non-group 使用 BLOCK_K=128
5. 循环不变量 cast 外提（小幅收益）
6. **non-group 大 M 解耦为 dequant workspace + 纯 GEMM**（msprof 驱动，0.5807→0.6240）：消除 dequant 跨 M-tile 重复执行，Cube 从 latency-bound(7.6-15.8%) 转 compute-bound
7. **group 路径解耦（标量 g + 结构路由，0.6240→0.6770）**：gs<=64/bf16/M>=512 命中时解耦；dequant 用标量 g（`DQ_BLOCK_K | group_size`）+ 单行连续 scale 加载，禁用逐元素 2D gather；fp16/gs>=128/M<512 保留 fused
8. **解耦 GEMM 放大 BLOCK_N=256（大 N bandwidth-bound，0.6770→0.7946）**：msprof 显示大 N 解耦 GEMM 是 bandwidth bound（MTE2 54% / Cube 20% / HBM 打满），workspace(64-128MB)>L2 被 M-tile 重读；放大 BLOCK_N 128→256，纯 GEMM +36-38%（case37/46/4）
9. **N-strip chunking（L2-exceeding workspace 算法级解法，0.7946→0.9133）**：L1.12 的 BLOCK_N=256 只减半单次访存，整 workspace 仍被 M-tile 重读。N-strip 把 N 切成 L2-resident 的 strip（48MB 上限），每 strip dequant 一次供全部 M-tile L2 复用，输出 N 维 disjoint 无需 atomic。大 N case 0.54→1.48-1.75×（详见 L1.13）

**已验证无效/失败的优化方向**:

| 优化方向 | 结果 | 原因 |
|----------|------|------|
| group scale host 物化为 `[K, N]` | 劣化 | host CPU 时间与额外 HBM 写入 |
| group 模式 in-kernel per-K gather（未广播） | 劣化 | gather 开销大，0.0735x |
| bf16 non-group / bf16 group **fused** BLOCK_N=256 | 编译失败 | `hivm.hir.vcast` unsupported / UB overflow（仅限 **fused** kernel 的 fp32 dequant 中间体；解耦 GEMM 的 BN=256 不受此限，见 L1.12） |
| bf16 group BLOCK_K=128 | 编译失败 | `hivm.hir.vcast` unsupported |
| BLOCK_M=256 × BLOCK_N=256 | UB overflow | `cc overflow`（仅 BM256×BN256 组合溢出；BM128×BN256×BK128 恰好容纳 192KB） |
| 解耦 GEMM tile 遍历序改 n-slow/m-fast（L2 复用 workspace N-strip） | 更慢 | case37 GEMM 0.94 vs 0.85ms；L2 跨核共享不足以摊销 128MB footprint |
| 大 N 改回 fused（靠 int8 权重 64MB 命中 L2） | 多数更差 | 仅 case45 略胜；大 M 时 dequant 在 dot 关键路径主导，fused 重读 int8 8× 同样不命中 L2 |
| **scale-hoisted fused / fold-out（post-deliverable 重构）** | 35/48 挂 | 把 per-channel scale 移出 K-loop 直读 int8 权重省 workspace 往返。no-offset 精度超阈（fold-out 比 torch_npu 更精）；offset 触发 `hivm.hir.vcast root alloc` UB overflow。精度与编译双重阻塞 |
| **BLOCK_M=64 并行填充（post-deliverable 重构）** | 0.9437→0.7903(-0.15) | 推翻并行假设：decoupled matmul 是 per-tile Cube 效率受限而非 aggregate-MLP 受限，[128,128]@[128,256] 是高效 Cube dot shape，BM=64 崩 Cube 利用率 |

**关键结论**:
1. **Kernel 分裂是架构基础**：group 与 non-group 的 scale 访问模式差异巨大，必须分 kernel。
2. **group scale 单 row 广播是核心优化**：避免 host 物化和 gather，是收益最大的单点改进。
3. **block size 必须按 dtype/shape 分派**：bf16 和 group 模式对大块更敏感，fp16 non-group 可激进放大。
4. **BLOCK_M=256 仅在与 BLOCK_N=256 组合时溢出**：`BM=256×BN=128` 单独可行；但解耦 GEMM 实测 `BN=256×BM=128` 更优（N-tile 减半胜于 M-tile 减半），故 BLOCK_M 仍保持 128。
5. **dequant 域必须按 dtype 分流**：fp16 在 fp16 域，bf16 在 fp32 域，否则精度 fail。
6. **msprof Roofline 是判定 latency-bound 的权威手段**：fused kernel 经 profiling 判定 `latency bound: pipeline caused`（Cube 7.6-15.8%），直接驱动「non-group 大 M 解耦 + 纯 GEMM」重构，geomean 0.5807→0.6240。后续 quant 类 matmul 若 Cube<20% 且带宽未饱和，应优先考虑解耦 dequant。
7. **解耦须按 case 分流（mixed strategy），且 group 解耦成败取决于 dequant 访存模式**：早期「group 解耦必劣化」的结论是**错误**的——根因是首版用了逐元素 2D gather dequant（离散访存慢 ~4x）。改用标量 g + 连续行加载后，group 解耦在结构信号（gs<=64 / bf16 / M>=512）命中时显著收益（opt_iter_18：11 个 group case 全净收益，0 回退）。仍须保留 fused 的场景：fp16+gs>=128+M<512（fused 本就 BLOCK_K=128，workspace 开销略亏）。
8. **bf16 真实瓶颈是关键路径 dequant 而非 fixpipe**：msprof 显示 fused bf16 的 fixpipe 仅 0.5%，bf16 慢是因为 fp32 dequant 留在 dot 关键路径（Vector/MTE2 主导）。解耦即消除——最差 bf16 group case 由 group 解耦一并修复（c28 0.283→0.519, c29 0.279→0.607, c36 0.518→0.910）。
9. **目标 0.8x 已达成（refactor_iter_20 实测 0.9133x，方差带 0.91-0.95，超目标 14%）**：N-strip chunking（L1.13）是破局点——把大 N 解耦 case 从 opt_iter_19 的 ~0.54 拉到 1.48-1.75×，彻底解决 L1.12 遗留的 "workspace > L2 被 M-tile 重读" 问题。剩余拖低 geomean 的 laggard 转移到 **medium-N 解耦 case（c2/12/25/28/41/44，0.62-0.79）**，这些 case 见结论 11。
10. **bandwidth-bound 与 compute-bound 的 msprof 区分（L1.12）**：解耦后 GEMM 的瓶颈从 fused 的「Cube 饿死（latency-bound, Cube<16%）」转为「HBM 打满（bandwidth-bound, MTE2 54% / Cube 20% / read_bw 接近峰值）」。前者靠解耦/放大 tile 提升 Cube，后者放大 BLOCK_N 减半访存次数。**判据**：msprof `MTE2>40% 且 Cube<30% 且 read_bw≈HBM 峰值` → bandwidth bound → 放大 BLOCK_N；若 workspace 体积>L2，重读无法靠遍历序（n-slow/m-fast）消除（已实测更慢）。
11. **medium-N laggard 已到 Triton toolchain 天花板**：refactor_iter_20 达 0.9133x 后，剩余 laggard 集中在 M∈[200,512] 的 medium-N 解耦路径（0.62–0.79）。msprof 显示该路径 `latency bound: memory caused` 且 MTE2 带宽余量 <80% peak，说明有 headroom 但 Triton 抓不到。两条算法重构均证伪：scale-hoisted fused 被精度（no-offset）和编译（offset UB overflow）双重阻塞；BLOCK_M=64 并行填充 -0.15 崩 Cube 效率。medium-N 剩余 gap 到 1.0 是 **torch_npu 硬件 dequant-on-load 单 kernel 的结构性优势**——Triton 无法软件流水 dequant-across-dot（vmul/vcast root-alloc rejection），也无法访问硬件 dequant 路径。**后续 medium-N 量化 matmul 若解耦路径 <0.8x，不要调参/fold-out/BM缩放，直接标注 toolchain ceiling**。

---

## §5 GroupedMatmulSwigluQuant 算子（`gmm-swiglu-quant`，MoE 分组 GEMM + SwiGLU + 动态量化）

**算子类别**: `gmm-swiglu-quant`（per-expert 分组 GEMM + SwiGLU + 动态量化）
**典型特征**: `x` fp16 `[M,K]` + per-expert int8 `weight[E,K,N]` + `weight_scale[E,1,N]` + `x_scale[M]` + `group_list`（专家行边界）；输出 `out_q` int8 `[M, N//2]` + `out_s` fp32 `[M]`（逐行动态量化 scale）
**性能基准**: 60/60 verify pass；**两 kernel 链一次成型**几何平均 **1.6967x**（超 target 0.8x 达 112%，per-shape 区间 [1.106, 3.504]）；**fusion-search 10 轮遗传进化最终 2.1153x**（split-K GEMM + epilogue 架构，权威 full-60 expert_max geomean，进化轨迹与基因清单见 §5.6）

> 📌 本节由 `cv-fusion.md §3` 合并而来。`cv-fusion.md` 为 CV 融合类卡，同时收录 QuantMatmul（§2）与 5_LightningIndexer（§4）。本算子的**权威类别映射**：`gmm-swiglu-quant` → `cv-fusion.md §3`（其 §9 判别表）。生成时优先读 `cv-fusion.md §3`；本节是 `quantization.md` 内的自包含副本，保证按 quantization 分类时也能拿到正确经验，两处内容需保持同步。

> ⛔ **反例（0812 批跑，重生成教训）**：GroupedMatmulSwigluQuant 的骨架按 `cv-fusion.md §3` 逐字复刻（两 kernel 链 / tile 分档 / fp16 双舍入 / 标量 store trick 全对），但 K 归约被改写成 `acc += tl.dot(out_dtype=...)` + `tl.arange` 基向量内联两处"等价改写"后，**60/60 精度全过、几何平均跌到 0.5076x**（同 case impl 0.076→13.8ms，182x），Phase 4 仅扫 BLOCK_K（→0.6596x）未能挽回。⇒ 骨架正确≠细节正确：§5.5 CV1 的累加器融合形式与 CV7b 的基向量外提是**写法级硬约束**。

### §5.0 首次生成必读：两 kernel 链 + per-expert 分组是"一次成型"的关键

本算子 **Phase 3 一轮就达 1.6060x**（已超 target 0.8x 两倍），靠的是首次生成就把架构写对：**K1（GMM+dequant，Cube）/ K2（SwiGLU+quant，Vector）职责分离 + 每个 expert 独立 launch + tile 按专家行数 m_e 分档**。后续 Phase 4 只做微调（1.6060→1.6967）。Designer 必须在草图阶段就定下这个两 kernel 结构，不要试图单 kernel 承担全部。

> ⚠️ **首代体检阈值（生成后立即核对）**：架构写对时，**首代 geomean 应 ≥1.5x**（0811 首代 1.6060x）。若首代 **<1.0x**，说明架构偏离了 L1——按 §6.3 的预检指纹自查：`acc += tl.dot(` / 基向量内联 / 扁平任务表 / `torch.stack` 拼接 / tile <4 档 / BLOCK_K≠128。**不要带着错误架构直接进 Phase 4 扫参**（0818 首代 0.6551x 正是带偏架构进 Phase 4，越优化越偏，最终 0.83x）。

### §5.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须拆成 K1（GMM+dequant）/ K2（SwiGLU+dynamic quant）两 kernel 链
- **必须** K1 在 Cube 上做 per-expert `gmm_dequant_kernel`（int8@int8 → int32 累加 → 两级 dequant → fp16 workspace `y[M,N]`）；K2 在 Vector 上做 `swiglu_dquant_kernel`（读 y → SwiGLU → 逐行 max_abs → 动态量化 → int8 out_q + fp32 out_s）。
- **禁止** 把 SwiGLU+量化塞进 GEMM kernel（Cube↔Vector 混在一个 kernel 里同步代价大）；**禁止** 把 dequant 拆成第三个 kernel（多一次 HBM 往返）。
- **Why**：MoE FFN 的两部分计算特征完全不同（K1 是 Cube-bound GEMM，K2 是 Vector-bound reduce+量化），分两个 kernel 让各自走对硬件路径；中间 fp16 workspace 是必要的解耦点。
- 配套：K1 按 expert 逐个 launch（`_launch_gmm_dequant_all_experts`），K2 一次性处理全部 M 行。

#### L1.2 K1 用 int32 累加 + 按 weight dtype 分 i8/i32w 两个 kernel（见 §5.5 CV1）
- weight 是 int8 → `gmm_dequant_kernel_i8`（int32 累加器，操作数保持 int8）；weight 是 int32 → `gmm_dequant_kernel_i32w`（操作数 cast fp32 后 fp32 累加）。
- host 侧按 `w_dtype = weight[0].dtype` 分派。
- **必须** K1 的 K 归约写成累加器融合形式 `acc = tl.dot(a, b, acc)`、`tl.arange` 基向量外提到 tile 循环外（见 §5.5 CV1 / CV7b）——0812 重生成版仅这两处"等价改写"（`acc += tl.dot(out_dtype=...)` + 基向量内联）就把 1.6060x 拖到 **0.5076x**（同 case impl 0.076ms → 13.8ms，182x），且 60/60 精度全过零报错。

#### L1.3 ★ 两级 dequant 必须 fp16 双舍入，逐位对齐 Ascend 参考链
- **必须** dequant 走 `int32 acc → fp32 → *ws → fp16 → fp32 → *xs → fp16`（两级，每级 cast 回 fp16 舍入），**不是** 一次性 fp32 乘完。
```python
f = acc.to(tl.float32)
f = f * ws[None, :]        # weight_scale [N]
f = f.to(tl.float16)       # ← 第一级 fp16 舍入（参考链在此舍入）
f = f.to(tl.float32)
f = f * xs[:, None]        # x_scale [M]
f = f.to(tl.float16)       # ← 第二级 fp16 舍入
```
- **禁止** 跳过中间 fp16 cast（"算得更准"反而 fail，见 §5.5 CV2）。

#### L1.4 K2 手写 sigmoid（fp32）+ round-half-away-from-zero + int32 clamp + 寄存器内 per-row scale（见 §5.5 CV2）
- **必须** `sig = gate_f32 / (1.0 + tl.exp(-gate_f32))`（fp32，禁止 `tl.sigmoid`）；`sw = sig * up_f32 → fp16 → fp32`（对齐参考的 fp16 舍入点）。
- **必须** per-row `max_abs = tl.max(tl.abs(sw), axis=0)` → `scale = 127.0 / max(max_abs, 1e-10)` 在寄存器内算（K2 是 Vector kernel，整行常驻）。
- **必须** round-half-away-from-zero：`q = sw*scale; q = tl.where(q>=0, q+0.5, q-0.5); q = q.to(int32); q = tl.maximum(tl.minimum(q, 127), -128); q = q.to(int8)`。

#### L1.5 per-expert 分组 launch + tile 按专家行数 m_e 分档（见 §5.5 CV4）
- **必须** host 侧按 `group_list` 遍历 expert，每个 expert 取 `m_e = curr - prev` 行子张量独立 launch K1。
- **必须** tile 按 `(m_e, N)` 分档：
```python
if (N % 512 != 0) and (896 <= N <= 1536):           BLOCK_M=64,  BLOCK_N=128
elif (m_e < 128) and (512 <= N <= 1024):            BLOCK_M=16,  BLOCK_N=128
elif (M >= 500) and (N >= 2048):
    tiles_512 = cdiv(m_e,64)*cdiv(N,512)
    if 2*tiles_512 <= cube_cores:                   BLOCK_M=64,  BLOCK_N=256
    else:                                            BLOCK_M=64,  BLOCK_N=512
else:                                                BLOCK_M=64,  BLOCK_N=256
BLOCK_K = 128
```
- **禁止** 把 per-expert 逐次 launch 改成**扁平任务表 + 单次 launch**（host 构建 task 数组 + kernel 内 flat loop 取任务）。0818/4 批跑实测：省 launch 的开销 < 任务表每 tile 4 次全局 load（e/m_off/n_off/m_end）+ `e*K*N` 大偏移寻址 + tile 适配性下降的代价，**1.70x → 0.83x**，精度 60/60 全过却性能掉 2x（详见 §6.3 反例）。
- **禁止** 用 `torch.stack([w for w in weight], 0)` 把 per-expert weight 拼成 `[E,K,N]` 连续——每 forward 一次 `[E,K,N]` host 拷贝，且丢弃了窄子张量的相对寻址。0811/1.70x 用 Python list 直接传（`weight[e]` + 各 stride）。
- **禁止** 把 BLOCK_K 从 128 改成 256：int8 dot 的 K-tile 增大 UB 占用（a/b tile 各大一倍），实测 128 是 GMM 最优。
- **Why**：专家行数 m_e 忽大忽小（有的专家几百行、有的几千行），固定 tile 会让小专家空转、大专家溢出；按 m_e 分档是 per-expert 结构的必然要求。0811/1.70x 与 0818/0.83x 的唯一实质差异就在这几处架构细节——**per-expert + 4 档 tile + BLOCK_K=128 + 窄子张量传参**是硬约束，不是 Phase 4 可优化的候选。

#### L1.6 K1/K2 grid 收缩 + 核内步长，cube/vec 核数动态读取（见 §5.5 CV3）
- K1：`total_tiles = cdiv(m_e,BM)*cdiv(N,BN)`，`grid = min(cube_cores, total_tiles)`，kernel 内 `for tile_idx in range(pid, num_tiles, NUM_CORES)`。
- K2：`grid = M if M < 4*vec_cores else 4*vec_cores`，kernel 内 `for row_idx in range(pid, M, num_programs)`。
- `cube_core_num` / `vector_core_num` 从 `torch_npu.npu.npu_config.get_device_limit(0)` 动态读取。

#### L1.7 标量 scale store 的 broadcast trick
- per-row scale `out_s[M]` 是标量写回，用 `r1 = tl.arange(0, 1); tl.store(out_s_ptr + row_idx + r1, (max_abs/127.0) + r1*0.0)` 借 1 元素张量完成标量 store（避免标量直接 store 的限制）。

#### L1.8 host 侧 contiguous + 无 `%`（见 §5.5 CV7）
- `x/weight/weight_scale/x_scale` 全部 `.contiguous()`；expert 行子张量用 `narrow(0, curr-m_e, m_e)` 取连续段。

### §5.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧两段分派
```python
def forward(self, x, weight, weight_scale, x_scale, group_list):
    M, K = x.shape; E = len(weight); N = weight[0].shape[1]; HALF_N = N // 2
    # contiguous；分配 y[M,N] fp16 workspace、out_q[M,HALF_N] int8、out_s[M] fp32
    cube_cores = get_device_limit('cube_core_num', 24); vec_cores = get_device_limit('vector_core_num', 48)
    # 路由 gate（v4s / v4-wide / fallback）；首版 fallback 走两 kernel 链
    _launch_gmm_dequant_all_experts(x, weight, weight_scale, x_scale, y, group_list,
                                    K, N, HALF_N, M, E, cube_cores, w_dtype)   # K1
    grid_k2 = M if M < 4*vec_cores else 4*vec_cores
    swiglu_dquant_kernel[(grid_k2,)](y, out_q, out_s, M, N, HALF_N,
                                     y.stride(0), out_q.stride(0), out_q.stride(1),
                                     BLOCK_N=triton.next_power_of_2(HALF_N), NUM_VEC_CORES=grid_k2)  # K2
    return out_q, out_s
```

#### L2.2 K1 per-expert 循环骨架
```python
def _launch_gmm_dequant_all_experts(x, weight, weight_scale, x_scale, y, group_list, K, N, HALF_N, M, E, cube_cores, w_dtype):
    prev = 0
    for e in range(E):
        curr = int(group_list[e].item()); m_e = curr - prev; prev = curr
        if m_e == 0: continue
        x_e = x.narrow(0, curr-m_e, m_e); w_e = weight[e]; ws_e = weight_scale[e]
        xs_e = x_scale.narrow(0, curr-m_e, m_e); y_e = y.narrow(0, curr-m_e, m_e)
        BLOCK_M, BLOCK_N = _select_gmm_tile(m_e, N, M, cube_cores)        # L1.5
        total_tiles = cdiv(m_e,BLOCK_M)*cdiv(N,BLOCK_N)
        grid = cube_cores if cube_cores < total_tiles else total_tiles
        (gmm_dequant_kernel_i8 if w_dtype==torch.int8 else gmm_dequant_kernel_i32w)[(grid,)](
            x_e, w_e, ws_e, xs_e, y_e, m_e, K, N, HALF_N, ...,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=128, NUM_CORES=grid)
```

kernel 侧 K 归约与基向量的**强制形态**（§5.5 CV1 / CV7b，缺一即退化）：
```python
    offs_m_base = tl.arange(0, BLOCK_M)          # ← 基向量外提，tile 循环外（CV7b）
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    for tile_idx in range(pid, num_tiles, NUM_CORES):
        ...
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(...); b = tl.load(...)
            acc = tl.dot(a, b, acc)              # ← 累加器融合形式，禁止 acc += tl.dot(...)（CV1）
```

#### L2.3 K2 SwiGLU+quant 骨架
```python
@triton.jit
def swiglu_dquant_kernel(y_ptr, out_q_ptr, out_s_ptr, M, N, HALF_N, stride_ym, stride_qm, stride_qn,
                         BLOCK_N: tl.constexpr, NUM_VEC_CORES: tl.constexpr):
    pid = tl.program_id(0); num_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    for row_idx in range(pid, M, num_programs):
        gate = tl.load(y_ptr + row_idx*stride_ym + offs_n,         mask=(offs_n<HALF_N), other=0.0)
        up   = tl.load(y_ptr + row_idx*stride_ym + HALF_N + offs_n, mask=(HALF_N+offs_n<N), other=0.0)
        sig  = gate.to(fp32) / (1.0 + tl.exp(-gate.to(fp32)))      # L1.4 手写 sigmoid
        sw   = (sig * up.to(fp32)).to(fp16).to(fp32)               # 对齐参考 fp16 舍入
        max_abs = tl.max(tl.abs(sw), axis=0)
        scale = 127.0 / tl.maximum(max_abs, 1e-10)
        q = tl.where(sw*scale >= 0, sw*scale+0.5, sw*scale-0.5).to(int32)
        q = tl.maximum(tl.minimum(q, 127), -128).to(int8)
        tl.store(out_q_ptr + row_idx*stride_qm + offs_n, q, mask=(offs_n<HALF_N))
        r1 = tl.arange(0,1); tl.store(out_s_ptr + row_idx + r1, (max_abs/127.0) + r1*0.0)   # L1.7
```

### §5.3 Layer 3: 关键技巧（Agent 可参考但不可复制结构）

#### L3.1 两 kernel 链 + fp16 workspace 解耦
K1（Cube）与 K2（Vector）经 fp16 `y[M,N]` workspace 解耦，各自走对硬件路径；workspace 是必要的中间落地点，不要试图消除（消除=单 kernel=Cube↔Vector 强耦合，更慢）。

#### L3.2 两级 dequant 的 fp16 双舍入
见 §5.1 L1.3。坏味道：dequant 注释写"int32 → fp32 → *ws*xs → fp16"（一次舍入）⇒ 与参考两级舍入不符，精度 fail。

#### L3.3 per-expert tile 分档 + grid 按 tiles/cores 自适应
见 §5.1 L1.5/L1.6。大专家走 BN=512、小专家走 BM=16，避免空转与溢出。

#### L3.4 路由 gate 留扩展位
`_v4s_eligible` / `_v4wide_eligible` / fallback 三类分支首版都落 fallback（两 kernel 链），为后续把"小 HALF_N 单 kernel 直出"等更激进融合留扩展位——**首版不要急着上 v4s/v4-wide**，先把两 kernel 链跑稳。

### §5.4 性能基准与优化记录

| 版本 | cases | geomean | 相对 Phase3 | 说明 |
|------|-------|---------|------------|------|
| Phase 3（生成） | 60/60 | 1.6060x | 1.0x | 两 kernel 链 + per-expert 分档 + int32 累加 + fp16 双舍入，**一次成型**已超 target 0.8x 两倍 |
| opt_iter_0 | 60/60 | 1.6967x | 1.0561x | 微调 tile 分档（per-shape max 3.504x @ 256×8192） |
| **fusion-search R4（G014 split-K）** | 60/60 | **1.2084x** | — | 首次真实 >1.0x；split-K GEMM + int32 atomic_add epilogue 双 kernel，cube_util 23%→68% |
| **fusion-search R10（最终）** | 60/60 | **2.1153x** | — | 10 轮遗传进化收敛；per-shape tiling + LARGE_M_BLK + per-class 参数 + epilogue 解耦，权威 full-60 expert_max geomean |

**有效优化点**: ① 两 kernel 链职责分离（架构基础）② per-expert tile 分档 ③ int32 累加 ④ 两级 fp16 双舍入 dequant ⑤ 手写 sigmoid + round-half-away ⑥ grid 按 tiles/cores 自适应。
**关键结论**: 本算子是 **"设计对了一次成型"** 的典型——架构（两 kernel + per-expert）写对就 1.6x，Phase 4 只剩微调空间；Designer 应在草图阶段就定下两 kernel 结构，不要在 Phase 3/4 反复改架构。**Phase 4 优化范围仅限于调 tile 档位边界**（如 0811 从 BM64/BN256 微调到大专家 BN512），**禁止改动调度层架构**（per-expert → 扁平任务表）、**禁止改 BLOCK_K**（128 定死）、**禁止加 host 数据拼接**（torch.stack）——这些在 L1.5 已列为禁令，0818 违反后 1.70x → 0.83x。

> ②③④⑤⑥ 是**两 kernel 链 1.6-1.7x 路径**的收益来源（§5.6 的 fusion-search 路径在其之上叠加 split-K 架构重构再 +25%）。两套经验互补：单轮 Phase 3/4 应走本节结论，多轮遗传进化应走 §5.6。

### §5.5 共享通用经验（CV 系，GMM 适用子集）

本算子与 QuantMatmul 共享 `cv-fusion.md §1` 的 CV 通用经验；下列为对 GMM-Swiglu-Quant **必须遵守**的子集（完整版见 `cv-fusion.md §1`）。**CV10-CV12 为检索/select 类（5_LightningIndexer）专属，不适用本算子。**

#### CV1 ★ 低精度 GEMM 必须用 `tl.dot` + int32 累加器融合形式，禁止逐维外积
- **必须** `tl.dot(a, b, acc)`，int8@int8 用 `tl.int32` 累加器；**必须** 累加器融合形式 `acc = tl.dot(a, b, acc)`。
- **禁止** `acc += tl.dot(a, b, out_dtype=...)`（每次 K 迭代物化 `[BM,BN]` 全尺寸 dot 输出再相加，UB 占用翻倍、MMAD 链断开）；**禁止** 操作数升 fp32 再 dot（异构路径，慢）；**禁止** 逐维标量循环代替矩阵乘。
- **判别信号（代码指纹，精度全过也命中）**：K 循环体内出现 `acc += tl.dot(` ⇒ 立即改 `acc = tl.dot(a, b, acc)`。

#### CV2 ★★ 量化精度链必须逐位复刻参考实现，禁止"算得更准"
- **必须** 在参考会舍入的每个位置都 cast 回对应 dtype（典型：两级 dequant 的 fp16 双舍入，见 §5.1 L1.3）。
- **禁止** `tl.sigmoid`（Ascend 后端系统性偏差）→ 手写 `x / (1.0 + tl.exp(-x))`。
- **禁止** `tl.round`（round-half-to-even）→ 用 `tl.where(q >= 0, q + 0.5, q - 0.5)` 实现 **round-half-away-from-zero**（与 `npu_*_quant` 语义一致）。
- **禁止** 对 int32 用 `tl.clamp`（仅支持浮点）→ int32 阶段用 `tl.maximum/tl.minimum`。
- **判别信号**：失败集合**严格与 dtype/精度路径相关** ⇒ 直接查算术路径，不要去搜 tiling。

#### CV3 ★ grid 收缩到 `min(核数, tiles)` + 核内步长循环，核数必须动态读取
- **必须** host 侧算 `total_tiles`，`grid = (min(NUM_CORES, total_tiles),)`；kernel 内 `for tile in range(pid, total_tiles, NUM_PROGS)`。
- **必须** 动态读核数：Cube kernel 读 `cube_core_num`，Vector kernel 读 `vector_core_num`；**禁止** `grid=(total_tiles,)`、**禁止** 硬编码 `num_cores`。
- `NUM_CORES` 必须作为 `tl.constexpr` 传入（随 case 变化的值会触发编译爆炸）；改变任务空间必须同步改 grid（否则静默算少 → AccuracyError）。

#### CV4 ★ tile 必须 host 侧动态分档，禁止全局固定 tile
- **必须** 按 (M, N, K, dtype, 后处理组合) 在 host 侧选 `(BLOCK_M, BLOCK_N, BLOCK_K)`，典型 3~5 档（GMM 按 per-expert `m_e` 分 4 档，见 §5.1 L1.5）。
- **禁止** 全局一个固定 tile（大 shape 溢出 UB、小 shape 空转）；**禁止** 只拿大 case 扫参结果全局套用。
- 分档判定里**必须含一档小 shape 收缩**，否则固定大 tile 在 `M/BLOCK_M < 1` 时纯空转。

#### CV6 多后处理留在 `tl.dot` epilogue 单 kernel 内，顺序按精度链
- **必须** 把 dequant 后处理接在 `tl.dot` 累加器之后、同一 kernel（K1）内完成，按参考精度链顺序。
- **禁止** 把 dequant 拆成独立 kernel（多一次 HBM 往返）。

#### CV7 host 侧 flatten batch → 2D/3D，kernel 内禁用 `%` 模运算
- **必须** 多维/批量输入在 `forward()` 里 reshape/flatten 成 2D，kernel 只处理二维逻辑。
- **禁止** kernel 内用 `%` 拆 tile 坐标 → 用 `//` + 减法（`bb = tile // T; rem = tile - bb*T`）。

#### CV7b ★ 持久化 tile 循环的 `tl.arange` 基向量必须外提，禁止迭代内重算
- **必须** 把 `offs_m_base = tl.arange(0, BLOCK_M)` / `offs_n_base` / `offs_k_base` 声明在 tile 循环体之外，体内只做 `offs_m = m_start + offs_m_base` 加法。
- **禁止** 在 `for tile_idx in range(pid, num_tiles, NUM_CORES)` 体内重新 `tl.arange(...)`（每 tile 每维重算序列生成，累计发射开销可观，且阻断归纳优化）。
- **判别信号（代码指纹）**：tile 循环体内出现 `tl.arange` ⇒ 外提；与 `acc += tl.dot(` 并列为本算子静态预检指纹。

#### CV8 后处理开关必须 `tl.constexpr` 特化，禁止运行时标量分支
- **必须** `HAS_PERTOKEN` / `HAS_BIAS` / `DO_CLAMP` 等布尔以 `tl.constexpr` 传入，让编译器 DCE 掉不适用的后处理段。
- **禁止** 作为运行时标量在 kernel 内 `if` 分支（保留全部分支的指令与 UB 占用）。

#### CV9 ★ 中间标量必须 `.to(tl.int32)`，避免 int64 地址降级
- **必须** `tl.program_id` / `tl.arange` 结果 `.to(tl.int32)`；地址计算全程 int32。
- **Why**：int64 标量触发地址计算降级；NPU 上 int32 索引更高效（本算子 tile 坐标计算多，尤其重要）。

### §5.6 fusion-search 10 轮遗传进化的核心学习（0.2869x → 2.1153x）

> 本节记录 `0819-0817-4` run 的完整进化轨迹，是**跨 1.0x 的必读路径**。适用于一切 grouped-matmul + 复杂 epilogue（SwiGLU/动态量化）算子。10 轮全成功，最终 2.1153x = 2.2x 目标的 96.2%。
>
> 与 §5.4 两 kernel 链（1.6967x）的**差异在于架构**：§5.4 是全融合单 kernel（GMM+SwiGLU+quant），§5.6 是 **split-K GEMM + epilogue 双 kernel**（把 SwiGLU+quant 从 GEMM 拆出，换来 cube 利用率 3× 提升）。二者共享 §5.1/§5.5 的精度链约束（fp16 双舍入、手写 sigmoid、round-half-away、int32 累加）。

**进化轨迹**：R1 0.774 → R2 0.8199 → R3 0.8259 → **R4 1.2084**（首个真实 >1.0x）→ R5 1.2017 → **R6 1.735** → **R7 1.985** → **R8 2.0262** → **R9 2.0931** → **R10 2.1153**

#### L6.1 架构分水岭：split-K GEMM + epilogue 双 kernel（G014）

- **前 R1-R3 天花板 ~0.83x**：单 fused kernel（GMM+SwiGLU+dquant 全融合）是 **memory/traffic-bound**（sw16 HBM 往返 + 每 M-block 全量重载 W，aic_mte2 0.82-0.90，aic_mac 仅 0.20-0.28，cube_util 4.6-8.7%）。within-encoding 调优（BLOCK_M/BLOCK_K/num_warps）已验证耗尽。
- **R4 突破 = 拆 kernel**：S=4 个程序做 split K-reduction，用 `tl.atomic_add`(int32) 逐位精确累加 partial GEMM；**独立 epilogue kernel** 读全 K 的 int32 accumulator，组合 + 反量化(weight_scale 按通道、x_scale 按 token) + SwiGLU + online-max 量化。**cube_util 23-30% → 68-78%**，+50.4% 到 1.2084x。
- **铁律**：epilogue 的 `tl.atomic_add` 必须在 int32 域做（bit-exact 已验证，float 域会破坏量化逐位一致性）。后处理顺序保持 `acc * ws → fp16 → * xs → fp16 → silu(gate)*up → fp16 → online max → 127/max → round-half-away + clamp`。

#### L6.2 权威 metric：expert_max geomean full-60（拒绝 fwd_sum）

- **`fwd_sum` 对 byte-identical kernel 虚高 1.3206**——forward 内 8 次发射的 kernel 只统计 1/8 时间，是测量假象。
- **只认 expert_max 的 full-60 geomean**（单次基准会漂移 ~1%，时钟节流可污染 3-5×）。
- 所有 route 固定到**单空闲全频设备**（device-6 曾 800MHz 节流导致 0.99→0.1951 失真，route 被标 negative-result 拒绝）。

#### L6.3 晋升基因清单（Promoted）

| 基因 | 内容 | 收益/作用 |
|------|------|-----------|
| **G014** | split-K GEMM + int32 atomic_add epilogue | R4 首个 >1.0x（+50.4%），cube_util 翻 3× |
| **G015** | per-shape GRID_FILL / S_MAX / EPI_BLOCK_M（tiny/small/medium/large 分档） | 小 shape 欠填充修复 |
| **G018** | LARGE_M_BLK 64→96/128 | 修复 large-M 回归（case-1 1.2263→1.9415），削减 W-reload MTE2 |
| **G019** | epilogue EPI_HALF_BLOCK_N=256 与 GEMM tile **解耦** + per-shape EPI_BLOCK_M/num_warps | R7 +14.4% 主推手 |
| **G023** | per-class S_MAX/GRID_FILL（large GF24/S6、tiny S8） | R8 +2.08%；修复 case-21 0.83→1.07、case-7 0.87→1.16、case-28 0.90→1.21 |
| **G024** | case-51 small-row 5-band split | 0.908→1.251 |
| **G025** | X-Large (1024,2048] split-K 边界 | 修复 champion 最差 2 回归 |
| **G028** | BLOCK_K=512（受控 A/B +3.49%） | 修复 case-51 small-row |
| **G030** | xlarge GRID_FILL=30 / S_MAX=6 / GRID_CAP=48 | R10 最终增量 |

**R9 关键验证**：三个合并基因（G023×G024×G025）正交——合并后**无 case < 1.0**（min 1.08），证明基因正交性可叠加。**LARGE_M_BLK 96→128** 独立 +3.1%。

#### L6.4 负基因清单（已验证丢弃，勿重试）

| 基因 | 失败原因 |
|------|----------|
| G005 | BLOCK_M 删除 -6.2% |
| G009 | wide-dot 丢弃 |
| G011/G012 | W-reuse UB 溢出 / in-UB sw16 cbuf-cbuf+tl.dot 冲突 |
| G013 | multi-kernel split 退化 |
| G016/G017 | sw16 消除 net-negative / 两级 max net-negative |
| G020 | micro S-scaling neutral（小-M int8 W-bandwidth-bound） |
| G021 | fused W-reuse compute-limited |
| G022 | tile sweep wash |
| G026 | last-split epilogue 不可行 |
| G027 | tie 折叠入 G019 |
| G029 | medium S=6/8 回归 |
| G031/G032/G033 | xlarge 并行度 probe / BLOCK_K 激进 / sw16 re-opt 均无净收益 |

#### L6.5 最终编码与残余墙

- **最终 best_config**：SMALL_M_BLK=32 / LARGE_M_BLK=128 / FUSED_LARGE_M_BLK=64 / ADAPT_THRESHOLD=256 / BLOCK_K=512 / HALF_BLOCK_N=128 / FUSED_HALF_BLOCK_N=128 / EPI_HALF_BLOCK_N=256 / EPI_BLOCK_M{tiny:8,small:8,med_lo:16,med_hi:32,xlarge:32,large:32} / XLARGE_GRID_FILL=30 / XLARGE_S_MAX=6 / XLARGE_GRID_CAP=48 / SMALL_HI_GRID_CAP=18 / SHAPE_CFG 6-class。
- **残余 3.8% 缺口（2.1153→2.2）不可达**：split-K **W-reload MTE2 墙**（UB-bound，BLOCK_K/split-count 不变）+ case-21 xlarge m≈1887 并行度下限。当前编码空间内已收敛。
- **R10 教训**：num_stages_gemm=3 病态 1.7221（k_iters≤1）、medium GRID_FILL=12 欠填充 20-AICore 1.6974 均回滚——**病态参数值必须回滚，不能保底**。

---

## §6 常见陷阱与避免方法

### §6.1 DynamicQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 期望优化大 K 场景超越 torch | 在 K > 8192 场景反复尝试 tile 优化、循环重排，期望超越 torch | 这是架构限制 (L1.1)，2-pass HBM 无法避免；直接接受 0.2x~0.3x，把精力放在小 K 优化 |
| 用 L2 cache 期望跨 pass 数据复用 | 单 kernel 两循环期望 L2 cache 保留 pass1 数据 | 已验证无效 (L1.2)，L2 cache 不保留跨 pass 数据 |
| 用非对称 BLOCK_K 优化 2-pass | pass1 用大 BLOCK_K、pass2 用小 BLOCK_K | 已验证无效 (L1.3)，瓶颈是 HBM 带宽而非中间量寄存器压力 |
| 用 bf16 max_abs 避免 fp32 膨胀 | 在 bf16 精度下计算 max_abs | 已验证无效 (L1.4)，max_abs 仍需转 fp32 累积 |
| 用 inv_scale 乘法替代除法 | `x * (1/scale)` 替代 `x / scale` | 已验证反而回退 (L1.5)，乘除延迟相近 |
| 用 `tl.round` 导致 verify 失败 | `tl.round` 默认 round-half-to-even，与 torch 的 round-half-away-from-zero 不一致 | 用 `tl.where(q >= 0, q + 0.5, q - 0.5)` + `tl.cast` (L1.10) |
| 手动 clamp 到 [-128, 127] | `tl.maximum(tl.minimum(q, 127), -128)` 生成多条指令 | `tl.cast(overflow_mode='saturate')` 硬件原生饱和转换 (L1.9) |
| 大 K 场景 BLOCK_M > 1 导致 UB 溢出 | 2D tile `BLOCK_M * BLOCK_K * sizeof(fp32)` 超 192KB UB | `_choose_block_size` 限制 `block_m = max(1, min(32, 4096 // block_n))` (L1.6 / L3.4) |
| 误把 DynamicQuant 当纯 reduction 优化 | 按 reduction 优化思路尝试向量化、tiling | DynamicQuant 瓶颈是跨 pass 数据复用 (2-pass HBM)；reduction 优化对大 K 无效 |
| 试图逆向工程 CCE 单遍融合 kernel | 试图用 Triton 复现 torch CCE 的 UB 缓存 + 跨核同步 | Triton DSL 不支持这些 CCE 级原语 (L1.7)，直接判定为架构限制 |
| profiler 失败误判为代码错误 | 4 个 case 报 `RuntimeError: 无法从 profiler 提取有效时延数据` | 这是 profiling 基础设施问题，verify 阶段 42/42 全部通过 |
| 忽略 smooth_scales 分支 | 只写一个 kernel，用 `if smooth_ptr is None` 分支，导致 kernel 内有条件分支 | 分两个 kernel (`_fused_2d_kernel` 和 `_no_smooth`)，host 侧分派 (L2.1) |
| 3D 输入处理不当 | 3D 输入 `[B, M, K]` 直接按 2D 处理，scale shape 错误 | `rows = b * m`，scale 输出为 `[B, M]` shape (L3.5) |

### §6.2 SwigluQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 使用 `tl.sigmoid` 导致精度系统性 fail | `tl.sigmoid` 在 Ascend 后端产生系统性数值偏差 | 始终使用 `x / (1.0 + tl.exp(-x))` (L1.4) |
| dynamic 模式标量除法 ULP 漂移 | `int_scale / max_abs` 在标量路径下与 torch 的 `aclnnDiv` 偏差 1-2 ULP，经 `nearbyint` 后 off-by-1 | 先存 `max_abs`，tile 级别以向量 load 再做除法 (L1.6 / L3.3) |
| 对 int32 使用 `tl.clamp` | 报 `Only floating point clamp is supported` | 用 `tl.maximum/tl.minimum` 在 int32 阶段做 clip (L1.7) |
| INT4 打包顺序错误 | `quint4x2` 期望低位为 even、高位为 odd，顺序错误会导致量化结果整体错位 | 严格按 `((q_odd & 0x0F) << 4) \| (q_even & 0x0F)` 打包 (L1.5 / L3.4) |
| 2D smooth/offset 的边界合并导致 UB overflow | 将 `mask & in_bounds` 合并到 `tl.load` 可减少一次 `tl.where`，但会增大 fp32 张量 live range，在大 shape 上 UB overflow | 2D 路径保持 `tl.where(in_bounds, swiglu * scale_block, swiglu)`；仅标量路径可合并 (L3.2) |
| 硬编码 `num_cores` | 不同 NPU 核数不同 | 动态读取 `triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)` (Q1) |
| offsets 在 dynamic 模式下误启用 | dynamic 模式没有 offsets 语义，启用会导致参考实现不一致 | host 侧强制 `has_offsets = offsets is not None and quant_mode == 0` (L1.11) |
| 把 int32 clip 改为 fp32 clip | 增大 fp32 中间结果 live range，触发 UB overflow (case 27) | 保持 int32 阶段 `tl.maximum/tl.minimum` clip，避免 fp32 中间结果长期保留 |
| 把 dynamic `BLOCK_ROWS` 从 4 降到 2 | 劣化，0.2434 ms > 0.2382 ms | `BLOCK_ROWS=4` 已接近最优，autotune `{2,4,6}` 收益有限 |

### §6.3 GroupedMatmulSwigluQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 单 kernel 承担 GMM+SwiGLU+quant 全部 | Cube↔Vector 单 kernel 强耦合，同步代价大 | 必须两 kernel 链（§5.1 L1.1）：K1 Cube GMM+dequant，K2 Vector SwiGLU+quant |
| 把 dequant 拆成独立第三 kernel | 多一次 HBM 往返 | dequant 留在 K1 GEMM epilogue（§5.1 L1.3） |
| per-expert 小专家空转 / 大专家溢出 | 固定 tile 不适应 m_e 跨度 | tile 按 m_e 分档（§5.1 L1.5） |
| **精度全过但比历史同架构慢 4x+（重生成/移植后）** | K 归约被"等价改写"成 `acc += tl.dot(out_dtype=...)`：每次迭代物化 `[BM,BN]` 全尺寸 dot 输出，MMAD 链断开；大 tile 档叠加 UB 压力，放大到 22~182x | **§5.5 CV1**：`acc = tl.dot(a, b, acc)` 累加器融合形式，配合 CV7b 基向量外提，列为静态预检指纹 |
| tile 循环体内出现 `tl.arange`（持久化 kernel） | 基向量未外提，每 tile 每维重算序列生成 | **§5.5 CV7b**：`offs_*_base` 声明在 tile 循环外，体内只做加法 |
| Phase 4 只扫 BLOCK_K/BLOCK_N、性能仍差一个量级 | 根因在代码形态（累加器形式/基向量外提）不在 tile 参数，扫参闭合不了 | 先跑 §6.3 上面的静态指纹预检，再进参数扫描 |
| 两级 dequant 漏舍入 / 用 `tl.sigmoid` / `tl.round` / int32 `tl.clamp` | 精度 fail（失败集 = 全部 fp16 case） | §5.1 L1.3-L1.4 + §5.5 CV2：fp16 双舍入、手写 sigmoid、round-half-away、int32 用 max/min |
| 改了 tile 没同步改 grid | 静默算少 → AccuracyError | §5.5 CV3：改任务空间必须同步改 grid |
| **精度全过但比历史慢 2x（0818/4 批跑实锤）**：per-expert 逐次 launch 被"优化"成**扁平任务表 + 单次 launch**，tile 4 档退化成 2 档、BLOCK_K 128→256、weight 改 host `torch.stack` 拼接 | Phase 4 优化 agent 凭直觉改架构，**背离 §5.1 L1.5 的 per-expert 分档约束**：任务表 kernel 每 tile 多 4 次全局 load（e/m_off/n_off/m_end）+ `e*K*N` 大偏移寻址；BM16 小专家适配丢失（小专家空转）；BN512 大 N 适配丢失；`torch.stack` 每 forward 一次 `[E,K,N]` host 拼接。净效果：省 launch 的开销 < 任务表寻址 + host 拼接 + tile 适配性下降的代价，1.70x → 0.83x | **模板约束写进 precheck ≠ Phase 4 遵守**。Layer 1 只在 Phase 2 Step 3 被门禁核对，Phase 4 opt_iter 不检查——优化 agent 改架构前必须回查 §5.1 L1.5：per-expert 逐次 launch + 4 档 tile（BM16 小专家/BN512 大 N）+ BLOCK_K=128 是**硬约束**，不是可优化的候选 |
| **单 fused kernel 性能天花板 ~0.83x（10 轮验证）**：GMM+SwiGLU+dquant 全融合单 kernel 在 60-case 多组输入上是 memory/traffic-bound（sw16 HBM 往返 + 每 M-block 全量重载 W） | 单 fused kernel 把 Cube-bound GEMM 与 Vector-bound 量化焊在一起，sw16 workspace 往返 + W 重载主导（aic_mte2 0.82-0.90，aic_mac 仅 0.20-0.28，cube_util 4.6-8.7%） | 跨 1.0x 必须 **split-K GEMM + epilogue 双 kernel**（§5.6 L6.1）：S=4 程序 split K-reduction + int32 `tl.atomic_add`（逐位精确）+ 独立 epilogue kernel 做 dequant+SwiGLU+quant，cube_util 23%→68-78% |
| **`tl.atomic_add` 用 fp32 域做 partial GEMM 归约** | 浮点归约破坏量化逐位一致性，verify 失败 | 必须 int32 域做 atomic_add（bit-exact）；epilogue 再按精度链 fp16 双舍入（§5.6 L6.1 铁律） |
| **采信 fwd_sum 或单次基准** | `fwd_sum` 对 byte-identical kernel 虚高 1.3206（8 次发射只统计 1/8）；单次基准漂移 ~1%，时钟节流可污染 3-5× | **只认 expert_max geomean full-60**；固定到单空闲全频设备，记录时钟节流（§5.6 L6.2） |
| **split-K 后残余 <0.9x（R10 收敛）**：W-reload MTE2 墙（UB-bound，BLOCK_K/split-count 不变）+ xlarge 并行度下限 | 参数空间已收敛，2.2x 目标的 96.2% 处无法再推进 | 标记 toolchain ceiling，勿重复扫参；病态参数值（num_stages_gemm=3 / GRID_FILL 欠填充）必须回滚不能保底（§5.6 L6.5） |
