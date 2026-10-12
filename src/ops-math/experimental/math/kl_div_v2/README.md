# KLDivV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：计算输入 `x` 和 `target` 之间的 Kullback-Leibler 散度。

- 计算公式：

  $$
  loss(x, target) = target \cdot (\log(target) - x)
  $$

  当 `log_target=true` 时：

  $$
  loss(x, target) = \exp(target) \cdot (target - x)
  $$

## 参数说明

| 参数名    | 输入/输出 | 说明                                                                 |
| --------- | --------- | -------------------------------------------------------------------- |
| x         | 输入      | 公式中的输入x，数据类型支持FLOAT16、FLOAT、BF16，数据格式支持ND。   |
| target    | 输入      | 公式中的target，数据类型支持FLOAT16、FLOAT、BF16，数据格式支持ND。  |
| reduction | 属性      | 归约方式，支持"none"(0)、"mean"(1)、"sum"(2)、"batchmean"(3)，默认"mean"。 |
| log_target| 属性      | target是否已取对数，默认false。                                      |
| y         | 输出      | 输出张量。reduction为"none"时shape与x相同，否则为标量[1]。           |

## 约束与限制

- x和target的shape需可广播（broadcastable）。
- 输入维度不超过8维。
- 数据格式仅支持ND。
- 数据类型仅支持float32、float16、bfloat16，且x和target类型必须一致。

## 调用说明

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)

| 目录 | 描述 |
| ---- | ---- |
| [test_aclnn_kl_div_v2.cpp](./examples/test_aclnn_kl_div_v2.cpp) | 通过aclnn调用的方式调用KLDivV2算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|--------|--------|---------|---------|---------|
| Xzz | 西工大智能感知交互实验室 | KLDivV2 | 2026/07/12 | 新增KLDivV2算子 |
