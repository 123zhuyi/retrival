# AsinGrad

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：计算反正弦函数 Asin 的反向梯度。
- 计算公式：

```text
z = dy / sqrt(1 - y * y)
```

其中 `y` 表示 Asin 反向计算的输入张量，`dy` 表示上游梯度张量，`z` 表示输出梯度张量。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape规格 |
| --- | --- | --- | --- | --- | --- |
| y | 输入 | 表示 Asin 反向计算的输入张量，对应计算公式中的 `y`。 | FLOAT、FLOAT16、BFLOAT16 | ND | 0-8维 |
| dy | 输入 | 表示上游梯度张量，对应计算公式中的 `dy`。 | 数据类型与 `y` 保持一致 | ND | Shape与 `y` 保持一致 |
| z | 输出 | 表示输出梯度张量，对应计算公式中的 `z`。 | 数据类型与 `y` 保持一致 | ND | Shape与 `y` 保持一致 |

## 约束说明

- `y`、`dy`、`z` 的数据类型需要保持一致，支持 FLOAT、FLOAT16、BFLOAT16。
- `y`、`dy`、`z` 的 shape 需要保持一致，不支持 broadcast。
- 仅支持 ND 数据格式。
- 支持 0-8 维输入。
- FLOAT16 路径在 half 精度上直接计算；BFLOAT16 路径将输入转换为 FLOAT 后计算，再将结果转换回 BFLOAT16。
- 当 `1 - y * y` 小于 0 或分母为 0 时，结果遵循硬件 `sqrt` 和 `div` 指令对 NaN/Inf 的处理行为。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_asin_grad.cpp](examples/test_aclnn_asin_grad.cpp) | 通过 [aclnnAsinGrad](docs/aclnnAsinGrad.md) 两段式接口调用 AsinGrad 算子。 |

从仓库根目录可使用如下命令编译算子包：

```bash
bash build.sh --pkg --experimental --soc=ascend910b --ops=asin_grad -j16
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=asin_grad -j16
```

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
|[@qq_61939128](https://gitcode.com/qq_61939128) | 西北工业大学智能感知交互实验室 | AsinGrad | 2026/7/23 | AsinGrad算子适配开源仓 |

欢迎参与 AsinGrad 算子的功能增强、问题修复和文档完善。提交贡献前，请参考仓库贡献指南 [CONTRIBUTING.md](../../../CONTRIBUTING.md) 和文档贡献说明 [CONTRIBUTING_DOCS.md](../../../docs/CONTRIBUTING_DOCS.md)。

## 参考资源

- [aclnnAsinGrad接口文档](docs/aclnnAsinGrad.md)
