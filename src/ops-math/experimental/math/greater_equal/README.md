# GreaterEqual

## 产品支持情况

| 产品                                                   | 是否支持 |
|:-----------------------------------------------------|:----:|
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |  √   |

## 功能说明

- 算子功能：逐元素比较 `self` 与 `other`（tensor 或 scalar），判断 `self >= other`，输出 BOOL 张量。
- 计算公式：

    $$out_i = (self_i \geq other_i)$$

## 参数说明（Tensor 形态）

| 参数名   | 输入/输出/属性 | 描述                               | 数据类型 |
|-------|----------|----------------------------------|------|
| self  | 输入       | 参与比较的 tensor。                 | FLOAT、FLOAT16、BFLOAT16、INT32、INT64、UINT64、INT8、UINT8、BOOL 等（以接口校验为准） |
| other | 输入       | 参与比较的 tensor 或标量。          | 与 self 可广播、可类型提升 |
| out   | 输出       | 比较结果。                         | BOOL |

## 约束说明

以 `aclnn_ge_tensor` / `aclnn_ge_scalar` 实现中的维度与 dtype 校验为准（例如 broadcast、最大维度等）。

## 调用说明

| 调用方式    | 调用样例                                                            | 说明                                                                                                      |
|---------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_ge_scalar](./examples/test_aclnn_ge_scalar.cpp)     | 通过 [aclnnGeScalar / aclnnInplaceGeScalar](./docs/aclnnGeScalar&aclnnInplaceGeScalar.md) 调用。   |
| aclnn调用 | [test_aclnn_ge_tensor](./examples/test_aclnn_ge_tensor.cpp)     | 通过 [aclnnGeTensor / aclnnInplaceGeTensor](./docs/aclnnGeTensor&aclnnInplaceGeTensor.md) 调用。   |

## 贡献说明

| 贡献者        | 贡献方   | 贡献算子       | 贡献时间      | 贡献内容              |
|------------|-------|------------|-----------|-------------------|
| GreaterEqual | 个人开发者 | GreaterEqual | 2026/5/14 | GreaterEqual 算子适配开源仓 |
