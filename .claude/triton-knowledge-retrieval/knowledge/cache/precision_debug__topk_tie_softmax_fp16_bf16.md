---
query_type: precision_debug
query_keywords: [topk, tie, softmax, fp16, bf16]
created: 2026-06-11
last_updated: 2026-06-11T00:00:00
hit_count: 0
context_summary: "TopK on logits vs softmax probabilities causes tie-breaking mismatch in fp16/bf16; tl.max/argmax support tie_break_left=True for PyTorch-compatible ordering"
top_units: [P1_topk_logit_softmax_order_001, P1_tie_break_left_001, P0_exp_precision_001]
stale: false
---

## 检索问题（Problem Statement）

**错误症状**：MoE Gating TopK Softmax kernel 精度验证失败，19/50 case 输出不一致，pass_rate ~99.7%，失败 1-2 个元素。

**触发场景**：
1. 仅在 fp16/bf16 case 下失败，fp32 全部通过
2. 失败表现为 topk 选择的索引与 PyTorch 参考不同
3. PyTorch 参考在 `softmax(probs)` 上做 topk，Triton 在 logits 上做 topk

**根本原因分析**：
- softmax 是单调函数，非 ties 情况下 logits 排序 = softmax 概率排序
- **ties 情况下**：bf16/fp16 输入在转换为 fp32 后，原本在 logits 中 tie 的值在 softmax 概率中可能因 `tl.exp` 的微小精度差异而不再 tie，导致排序改变
- 或 logits 中不 tie 但非常接近的值，在 softmax 概率中 tie-breaking 方向不同
- 核心：必须**在 softmax 概率上做 topk**（和 PyTorch 参考完全一致），而非在 logits 上做 topk

## 知识单元列表

### 核心修复知识单元

**P0_exp_precision_001** `[阻塞修复]` fp32 softmax 概率上做 topk，而非 logits
- **内容**: Triton `tl.exp` 在 fp32 下与 PyTorch `torch.exp` 可能存在 1-ULP 差异。当输入值非常接近（ties 边缘）时，这种差异可能改变 softmax 概率的排序。
  - 错误: 在 logits 上做 topk，然后从 softmax 概率中 gather 值
  - 修复: 先计算完整 fp32 softmax 概率数组，然后在 softmax 概率数组上做 topk
  - 修复: `softmax_vals = tl.exp(x_f32 - max_val) / sum_val`，然后 `for kk in range(k): curr_max, curr_idx = tl.max(softmax_vals, axis=0, return_indices=True, tie_break_left=True)`
  - 注意: `tl.max(return_indices=True)` 返回的索引类型是 fp32，需 `.to(tl.int32)`
- **来源**: 精度调试分析 + reduction_ops API 文档

**P1_tie_break_left_001** `[重要修复]` 使用 `tie_break_left=True` 匹配 PyTorch topk 行为
- **内容**: PyTorch `torch.topk` 在 ties 时选择最左侧（最小索引）元素。Triton `tl.max` / `tl.argmax` / `tl.min` 均支持 `tie_break_left` 参数。
  - `tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)` — 返回最大值 + 最左侧索引
  - `tl.argmax(x, axis, tie_break_left=True)` — 直接返回最左侧最大值索引
  - 索引返回类型为 fp32，需显式 `.to(tl.int32)` 转换
- **来源**: triton_api/Reduction_Ops/max.md + argmax.md

**P1_topk_logit_softmax_order_001** `[重要修复]` logits topk ≠ softmax topk 的边界情况
- **内容**: 虽然 softmax 是单调递增函数，但 ties 时 tie-breaking 依赖于具体数值。
  - 场景 A: logits 中值 A 和 B 在 bf16 精度下 tie，但 fp32 转换后 A > B → softmax 中 A > B
  - 场景 B: logits 中 A > B，但 exp 精度差异导致 softmax 中 A < B（概率极低，仅在 1-ULP 边缘）
  - 结论：确保 Triton topk 和 PyTorch topk 在**同一数组**上执行
- **来源**: 数值稳定性分析

---

## 修复代码示例

```python
# 1. 计算 fp32 softmax（与 PyTorch 参考完全一致）
x_f32 = x_row.to(tl.float32)
max_val = tl.max(x_f32, axis=0)
x_f32 = x_f32 - max_val
exp_x = tl.exp(x_f32)
sum_val = tl.sum(exp_x, axis=0)
softmax_vals = exp_x / sum_val

# 2. 在 softmax 概率上做 topk（非 logits）
MAX_K: tl.constexpr = 16
topk_vals = tl.full((MAX_K,), float('-inf'), tl.float32)
topk_idx = tl.zeros((MAX_K,), tl.int32)

for kk in range(k):
    # tl.max with return_indices, tie_break_left=True 匹配 PyTorch topk
    curr_max, curr_idx_fp32 = tl.max(
        softmax_vals, axis=0,
        return_indices=True,
        return_indices_tie_break_left=True
    )
    curr_idx = curr_idx_fp32.to(tl.int32)
    
    pos_mask = tl.arange(0, MAX_K) == kk
    topk_vals = tl.where(pos_mask, curr_max, topk_vals)
    topk_idx = tl.where(pos_mask, curr_idx, topk_idx)
    
    # 排除已选位置（所有 tie 位置一起排除）
    softmax_vals = tl.where(softmax_vals == curr_max, float('-inf'), softmax_vals)
```
