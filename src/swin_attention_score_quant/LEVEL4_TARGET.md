# LEVEL4 目标算子对照说明（swin_attention_score_quant）

> 本目录材料来源：`src/ops-transformer/attention/swin_attention_score_quant`（gitcode.com/cann/ops-transformer @ d28d8f99a，CANN Open Software License 2.0）。
> **定位：CrossformerAttention 的结构参考（次要）**，主参考在 `src/flash_attention_score/`（pse_shift + atten_mask 路径）。

## 服务的 level4 算子

### CrossformerAttention（level4_target/CrossformerAttention/）
- **为什么选它**：Swin/CrossFormer 同为窗口化多头注意力，「B×N×C 分组后逐窗口做 attention + 加性 mask」的结构与 level4 参考实现同构。
- **限制**：本算子是**量化版**（quant），dtype 通路（int8/fp16 混合）与 level4 目标（fp16/bf16/fp32，fp32 累加 softmax）不一致；只抽取窗口分块组织方式，不参考量化路径。
- **缺口**：level4 的 `pos_bias`（相对位置偏置，逐头加性 bias）在本算子中无对应，需回到 FAS 的 pse_shift 路径取证。

## 材料清单（P0 五类）
| 类别 | 位置 |
|---|---|
| kernel 入口/模板 | `op_kernel/` |
| host tiling | `op_host/` |
| aclnn API / 文档 / 测试 | `op_api/`、`docs/`、`tests/` |
