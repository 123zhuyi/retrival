# TensorMove 算子设计说明

## 产品支持情况
| 产品 | 是否支持 |
| --- | --- |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
## 功能说明
- 算子功能：将输入张量 `x` 原样复制到输出张量 `y`，不涉及数值计算、广播、数据重排或数据类型转换。
- 计算公式：

$$
y = x
$$

## 参数说明
| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x | 输入 | 输入张量，表示需要被复制的数据。 | BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT16、BFLOAT16、FLOAT、DOUBLE | ND |
| y | 输出 | 输出张量，表示 TensorMove 算子的输出结果。输出张量的 shape、dtype、format 与输入张量 `x` 保持一致。 | BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT16、BFLOAT16、FLOAT、DOUBLE | ND |

## 约束说明
- 输入张量 `x` 与输出张量 `y` 的 shape 必须一致。
- 输入张量 `x` 与输出张量 `y` 的 dtype 必须一致。
- 当前支持 ND 数据格式。
- 不支持广播。
- 不支持数据类型转换。
- 不支持数据重排。
- 无算子属性参数。
- 当前算子适配 Atlas A2 训练系列产品。

## 调用说明
| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn API | `examples/test_aclnn_tensor_move.cpp` | 通过 `aclnnTensorMoveGetWorkspaceSize` 和 `aclnnTensorMove` 两阶段接口调用 TensorMove 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
|[@qq_61939128](https://gitcode.com/qq_61939128) | 西北工业大学智能感知交互实验室 | TensorMove | 2026/7/8 | TensorMove算子适配开源仓 |
