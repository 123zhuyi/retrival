# Transpose

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    √     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：对tensor的任意维度进行调换。如输入self是shape为[2, 3, 5]的tensor，dims为(2, 0, 1)，则输出是shape为[5, 2, 3]的tensor。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型 | 数据格式 |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | 输入张量       | 需要进行维度置换的输入张量。                                 | 见下方   | ND       |
| perm   | 输入张量       | 表示x的维度的排列。取值需在[0，self的维度数量-1]范围内。 | INT64、INT32    | -        |
| y    | 输出           | 维度最大不超过8维，shape由dims和原self的shape共同决定，dtype需要与self一致。 | 同x   | ND       |

- <term>GPU支持的数据类型</term>：数据类型支持DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN。
- <term>CPU支持的数据类型</term>：数据类型支持同GPU。

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn接口 | [test_aclnn_channel_shuffle](examples/test_aclnn_channel_shuffle.cpp) | 通过[aclnn_channel_shuffle](docs/aclnnChannelShuffle.md)接口方式调用transpose算子。 |
| aclnn接口 | [test_aclnn_permute](examples/test_aclnn_permute.cpp) | 通过[aclnn_permute](docs/aclnnPermute.md)接口方式调用transpose算子。 |
