---
id: technique-flash-attention-backward-recompute
title: FlashAttention backward recomputation and FlashSoftmaxGrad on Ascend NPU
type: technique
tags: [flash_attention, attention, backward, recompute, softmax_grad, online_softmax]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_buffer, l1_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [flash_attention_score_grad]
symptoms: [precision_mismatch, performance_bottleneck]
confidence: inferred
reproducibility: snippet
prerequisites: [了解FlashAttention前向计算流程(QK->softmax->PV)、softmax反向梯度公式、Ascend CUBE/VECTOR分工、在线softmax]
related: [technique-flash-attention-pipeline, technique-flash-attention-tiling, kernel-flash-attention-score-grad, kernel-flash-attention-score]
sources: [ops-transformer-flash-attention-score-grad]
---

# FlashAttention 反向重算与 FlashSoftmaxGrad

FlashAttention 反向不保存前向的完整注意力矩阵 P（O(N²) 显存），而是在反向**重算** P，再以 6 步公式得到 dQ/dK/dV。该方法是架构无关的可迁移算法，CANN `FlashAttentionScoreGrad`（FAG）即按此实现（来源: `docs/FAG算子设计介绍.md` L9-14）。

## 1. 反向 6 步公式

设前向：`S=Mask(QK^T/√d + pse, atten_mask)`，`P=Dropout(Softmax(S),keep_prob)`，`Y=PV`。

| 步 | 公式 | 计算单元 | 说明 |
|----|------|---------|------|
| 1 重算 p | `p = SimpleSoftmax(Mask(Matmul(query, key^T) + pse) * scale)` | Cube+Vector | 重算前向 softmax 结果 P，存片上 |
| 2 dp | `dp = Dropout(Matmul(dy, value^T))` | Cube | dY·V^T |
| 3 ds | `ds = p * Sub(dp, FlashSoftmaxGrad(dy, attention_in))` | Vector | softmax 反向 |
| 4 dq | `dq = Matmul(ds, key) * scale` | Cube | ds·K |
| 5 dk | `dk = Matmul(ds^T, query) * scale` | Cube | ds^T·Q |
| 6 dv | `dv = Matmul(Dropout(p)^T, dy)` | Cube | Dropout(P)^T·dY |

## 2. FlashSoftmaxGrad 推导（可迁移数学）

`ds = P ⊙ (dP − rowsum(dP ⊙ P))`，其中 `dP = dY·V^T`（步 2 的 dp）。推导：`Y=PV` ⇒ `dP=dY·V^T`；`P=Softmax(S)` ⇒ `dS = P⊙(dP − ⟨dP,P⟩)`（softmax 反向，`⟨·,·⟩` 为按行内积/行和）。CANN 用 `FlashSoftmaxGrad(dy, attention_in)` 计算归一化项，其中 `attention_in=Y=PV`（FAG doc L11）。该公式与 GPU FlashAttention 反向一致，架构无关。

## 3. 重算策略

- **重算 P**：反向重新执行前向的 QK^T + mask + softmax，不依赖前向保存的 P（O(N²)）。前向输出的 `softmax_max`/`softmax_sum` 作为**可选输入**可加速重算（避免再次在线归约），见 `op_host/flash_attention_score_grad_def.cpp` L72-83。
- **FP32 累加**：BF16/FP8 输入下，QK^T、dY·V^T、ds·K 等在 FP32 累加，输出再 cast 回 dtype（对应 CANN L0C/UB FP32 中间）。
- **在线 softmax 重算**：与前向相同的 running max/sum + rescale，按 S2 轴分块在线更新（见 [[technique-flash-attention-tiling]] 的 FlashSoftmax 在线刷新）。

## 4. matmul 编排（4 次 Cube matmul）

| matmul | 表达式 | 用途 |
|--------|--------|------|
| mm1 | Q·K^T | 步 1 重算 P 的前半（score） |
| mm2 | dY·V^T | 步 2 dp |
| mm3 | ds·K（dq）/ ds^T·Q（dk） | 步 4-5 |
| mm4 | P^T·dY | 步 6 dv |

CANN 三阶段流水：Cube1(`ComputeMM1`, 步1) -> Vector1(`ComputeVec`, 步2-3) -> Cube2(`ComputeMMDqkv`, 步4-6)，见 FAG doc L20-23。dK/dV 跨 query 块累加（多核同址累加；确定性模板用特定分核避免非确定性，FAG doc L357-366）。

## 5. Triton-Ascend 抽象

```python
# 表达设计意图，非可编译（reproducibility: snippet/pseudocode）
# 步 1: 在线 softmax 重算 P（running max/sum + rescale）
for start_n in range(0, Skv, BLOCK_N):
    s = tl.dot(q, tl.trans(k)) * sm_scale       # mm1: QK^T
    s = apply_mask(s, ...)                        # atten/causal/padding
    # online softmax -> p (本块), 更新 m_i, l_i
# 步 2: dp = dY @ V^T
dp = tl.dot(dy, tl.trans(v))
# 步 3: ds = P_norm * (dp - rowsum(dp * P_norm))   # FlashSoftmaxGrad
ds = p_norm * (dp - tl.sum(dp * p_norm, axis=1)[:, None])
# 步 4-6: dQ/dK/dV
acc_dq += tl.dot(ds, k) * sm_scale                # 本地累加
tl.atomic_add(dk_ptrs, tl.dot(tl.trans(ds), q) * sm_scale)  # 跨块累加
tl.atomic_add(dv_ptrs, tl.dot(tl.trans(p_norm), dy))
```

映射：重算 P 的 mm1 workspace -> on-chip `tl` tensor / drop；FP32 累加 -> `tl.dot(...).to(tl.float32)`；dK/dV 跨块累加 -> `tl.atomic_add`（确定性场景改用分核避免 atomic）。完整 skeleton 见 `artifacts/flash-attention-score-grad/flash-attention-score-grad-triton-skeleton.py`。

## 6. 不迁移的 CANN 机制

MTE/BufferID/queue/event、`fixp_l0c_to_ub`/`copy_ub_to_l1`、手动 Cube/Vector 流水编排、ND↔NZ 格式转换——这些是 CANN 实现机制，不作为 Triton 实现步骤，仅抽象为 tile 级复用与 fused load/store。

## 7. Agent actions

1. 反向按 6 步公式编排，重算 P 而非保存。
2. FlashSoftmaxGrad 用 `ds = P⊙(dP − rowsum(dP⊙P))`，FP32 累加。
3. 若前向已输出 softmax_max/sum，作为可选输入加速重算。
4. 4 次 matmul 按 CV 流水编排（参考 [[technique-flash-attention-pipeline]]）。
5. dK/dV 跨块累加用 atomic；确定性场景改分核策略。

来源与详细分析见 [[ops-transformer-flash-attention-score-grad]]。
