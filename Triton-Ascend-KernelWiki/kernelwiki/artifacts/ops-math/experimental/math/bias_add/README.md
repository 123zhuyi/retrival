# BiasAdd

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：为输入张量的每一个元素加上对应通道（channel）的偏差值。
- 计算公式：

$$out_i = x_i + bias_{c(i)}$$

  其中 $c(i)$ 为元素 $i$ 在 `data_format` 指定的 C（通道）维上的下标。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| x | 输入 | 待计算的输入张量 | FLOAT、FLOAT16、BFLOAT16、INT32 | NCHW、NHWC、NDHWC、NCDHW、ND |
| bias | 输入 | 累加偏差，长度等于 x 的 C 维大小 | 同 x | ND |
| y | 输出 | 计算结果 | 同 x | 同 x |
| data_format | 属性（可选） | 指定 C 维位置的数据排布，默认 "NHWC" | string | - |

## 约束说明

- 仅支持 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>。
- `bias` 的长度必须等于 `x` 在 `data_format` 指定的 C 维上的大小。

## 实现说明

按输入 shape 的结构属性在 host 侧运行时分派多条调度路径（schMode），kernel 侧据此选择对应实现：

- **TINY / TINY_NOQUEUE**：小 total，受 kernel launch / prologue 固定开销限制，走最薄路径。
- **THIN_TINY_VECTOR_BROADCAST**：小非对齐 NHWC，逐元素标量广播改为向量 `Copy + GatherMask` UB 广播。
- **BROADCAST_UB_TILE**：整 tile 物化 bias 后做整宽向量 `Add`；大 C 非对齐（cBlocks 较大）经向量广播处理，避免标量串行化。
- bfloat16 作为 dtype decorator，额外套 `Cast`（bf16↔fp32）链。

## 调用说明

| 调用方式 | 说明 |
| -------- | ---- |
| aclnn 单算子调用 | 见 `examples/test_aclnn_bias_add.cpp`（API 由算子定义自动生成 `aclnnBiasAdd`）。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 杨镇泽（@gcw_5x5Ew5Ms） | 重庆邮电大学（CQUPT） | BiasAdd | 2026/06/16 | 新增 BiasAdd 算子（Ascend 910B AscendC 实现） |
