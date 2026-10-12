# LEVEL4 目标算子对照说明（flash_attention_score）

> 本目录材料来源：`src/ops-transformer/attention/flash_attention_score`（gitcode.com/cann/ops-transformer @ d28d8f99a，CANN Open Software License 2.0），与 CANN 9.1.0 安装包
> `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/flash_attention_score/` 内 device 源码同源；本仓多带 host 侧 op_host/tiling、op_api、docs、tests。

## 服务的 level4 算子

### 1. SelfAttention（level4_target/SelfAttention/）
- **对应度**：注意力主体 1:1（softmax(QK^T·scale)·V，fp32 累加）。
- **差异**：level4 定义把 q/k/v/out 四个 `nn.Linear`（seed=42 随机权重）包进了算子，参考实现时需先拆出投影（对应 CANN MatMul），FAS 只覆盖注意力核。
- **形状风险**：workload 为小尺寸非对齐形状（如 B=2,S=7,D=72,n_heads=3 → head_dim=24），FAS 工业实现不保证支持；P1 抽取时以算法结构（S1S2 分块、online softmax、Cube/Vec 流水）为主，不可照搬规格约束。

### 2. CrossformerAttention（level4_target/CrossformerAttention/）
- **对应度**：部分对应。FAS 的 **pse_shift（加性位置偏置）** 与 **atten_mask（加性 mask）** 路径覆盖该算子的 pos_bias/mask 语义。
- **缺口**：窗口/分组重排（`feature_shape` + `group_size` 的 view/permute）CANN 无现成实现，属于 host 侧 layout 变换，不在 kernel 内。
- **补充参考**：`src/swin_attention_score_quant/`（窗口注意力结构同类，但为量化版）。

## 材料清单（P0 五类）
| 类别 | 位置 |
|---|---|
| kernel 入口 | `op_kernel/flash_attention_score.cpp`、`flash_attention_score_apt.cpp` |
| kernel 模板 | `op_kernel/arch22/*.h`（s1s2_bn2gs1 主实现、var_len、drop_mask_adapter 等） |
| host tiling | `op_host/`（安装包缺失部分，本仓补齐） |
| aclnn API | `op_api/` |
| 文档/测试 | `docs/`、`examples/`、`tests/` |
