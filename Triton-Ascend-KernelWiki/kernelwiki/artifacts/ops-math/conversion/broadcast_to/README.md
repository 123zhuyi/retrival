# BroadcastTo

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

- 算子功能：将输入tensor广播到指定的shape。如输入tensor的shape为(1, 4)，指定的shape为(2, 4)，则输出是shape为(2, 4)的tensor。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型 | 数据格式 |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | 输入张量       | 需要被广播的张量。  | BF16、FLOAT16、FLOAT、 DOUBLE、INT64、INT32、UINT8、UINT16、UINT32、UINT64、INT8、INT16、COMPLEX32、COMPLEX64、COMPLEX128、QINT8、QUINT8、QINT16、QUINT16、QINT32、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、STRING   | ND       |
| shape   | 输入张量       | 表示x广播后的shape大小。| INT64、INT32    | -        |
| y    | 输出           | 维度最大不超过8维，shape由shape输入决定，dtype需要与self一致。 | 同x   | ND       |

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 图模式调用 | [test_geir_broadcast_to](./examples/test_geir_broadcast_to.cpp)   | 通过[算子IR](./op_graph/broadcast_to_proto.h)构图方式调用broadcast_to算子。 |
