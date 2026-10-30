# CaseCondition

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：根据三个输入值与阈值的关系，输出离散case编号。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `x` | 输入 | 长度为3的一维张量，依次表示`i`、`j`、`k`。 | INT32、INT64、UINT64 | ND |
| `algorithm` | 属性 | 当前仅支持`LU`。 | STRING | - |
| `y` | 输出 | 输出case编号。 | INT32 | ND |

## 约束说明

- `x`必须为shape`[3]`的向量。
- `algorithm`仅支持`LU`。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_case_condition](./examples/test_geir_case_condition.cpp) | 通过[算子IR](./op_graph/case_condition_proto.h)构图方式调用CaseCondition算子。 |
