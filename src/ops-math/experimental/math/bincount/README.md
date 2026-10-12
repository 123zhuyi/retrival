# Bincount

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：统计一维整数张量中每个值的出现次数；带 `weights` 时按权重累加。
- **计算公式**：

对每个输入元素 $i$（要求 $self_i \ge 0$），有权重时：
$$out[self_i] = out[self_i] + weights_i$$

无权重（计数）时：
$$out[self_i] = out[self_i] + 1$$

输出长度 $L = \max(\max_i self_i + 1,\ minlength)$，输出下标 $k$ 即原值 $k$。若 $\min_i self_i < 0$，算子运行期检查到负数后报错并结束计算，不产出结果。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续 |
| --- | --- | --- | --- | --- | --- | --- |
| array | 输入 | 输入整数张量，公式中的 self_i | INT8、INT16、INT32、INT64、UINT8 | ND | 1 | √ |
| size | 属性 | 输出最小长度，默认 0 | int64_t | - | - | - |
| weights | 输入 | 权重，可为空指针，公式中的 weights_i | FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL | ND | 1 | √ |
| bins | 输出 | 输出张量，公式中的 out | INT32、INT64、FLOAT、DOUBLE | ND | 1 | √ |

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn 调用 | [test_aclnn_bincount](examples/test_aclnn_bincount.cpp) | 通过 `aclnnBincount` 接口调用，演示非负输入计数 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| ddplys | 个人开发者 | Bincount | 2026/07 | 基于 Ascend C 实现 Bincount |
