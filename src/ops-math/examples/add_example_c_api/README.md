# AddExampleCApi

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：完成加法计算，kernel使用C API（asc_simd）开发方式。

- 计算公式：

$$
y = x1 + x2
$$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| x1 | 输入 | 待进行加法计算的入参，公式中的x1。 | FLOAT | ND |
| x2 | 输入 | 待进行加法计算的入参，公式中的x2。 | FLOAT | ND |
| y | 输出 | 待进行加法计算的出参，公式中的y。 | FLOAT | ND |

## 约束说明

输入输出仅支持4维，仅支持FLOAT数据类型。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|---------|------|
| aclnn调用 | [test_aclnn_add_example_c_api](./examples/test_aclnn_add_example_c_api.cpp) | 参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成算子编译和验证。 |

## Kernel开发方式

本示例kernel使用C API（asc_simd）方式开发，区别于C++类方式的AscendC API。核心接口包括：

- `asc_init()` - 初始化
- `asc_copy_gm2ub_sync()` / `asc_copy_ub2gm_sync()` - 数据搬运
- `asc_add_sync()` - 向量计算
