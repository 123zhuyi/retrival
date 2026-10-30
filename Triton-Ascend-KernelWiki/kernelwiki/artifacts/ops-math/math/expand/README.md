# Expand

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |      √   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×    |
| <term>Atlas 推理系列产品</term>                       |     √    |
| <term>Atlas 训练系列产品</term>                       |     √    |

## 功能说明

- 算子功能：将输入tensor广播到指定的shape。如输入tensor的shape为(1, 4)，指定的shape为(2, 4)，则输出是shape为(2, 4)的tensor。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型 | 数据格式 |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | 输入张量       | 需要被广播的张量。  | FLOAT16、FLOAT、INT32、INT64、INT8、UINT8、BOOL、BF16   | ND       |
| shape   | 输入张量       | 表示x广播后的shape大小。| INT64、INT32    | -        |
| y    | 输出           | 维度最大不超过8维，shape由shape输入决定，dtype需要与self一致。 | 同x   | ND       |

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 图模式调用 | [test_geir_expand](./examples/test_geir_expand.cpp) | 通过GE IR图模式调用Expand算子 |
| aclnn接口 | [test_aclnn_expand](./examples/test_aclnn_expand.cpp)   | 通过[aclnnExpand](./docs/aclnnExpand.md)接口方式调用Expand算子。 |
