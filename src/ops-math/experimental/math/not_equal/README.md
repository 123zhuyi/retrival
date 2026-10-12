# NotEqual

## 产品支持情况

| 产品                                                   | 是否支持 |
|:-----------------------------------------------------|:----:|
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |  √   |

## 功能说明

- 算子功能：逐元素比较两个输入tensor，判断对应元素是否不相等。返回一个BOOL型tensor。
- 计算公式：

    $$out_i = (self_i \neq other_i)$$

## 参数说明

| 参数名   | 输入/输出/属性 | 描述                               | 数据类型                                               | 数据格式 |
|-------|----------|----------------------------------|----------------------------------------------------|------|
| self  | 输入       | 待进行not_equal计算的入参，公式中的$self_i$。  | FLOAT16,FLOAT,INT32,INT8,UINT8,BOOL,INT64,BFLOAT16 | ND   |
| other | 输入       | 待进行not_equal计算的入参，公式中的$other_i$。 | FLOAT16,FLOAT,INT32,INT8,UINT8,BOOL,INT64,BFLOAT16 | ND   |
| out   | 输出       | 待进行not_equal计算的出参，公式中的$out_i$。   | BOOL                                               | ND   |

## 约束说明

不支持广播，不支持int64。

## 调用说明

| 调用方式    | 调用样例                                                            | 说明                                                                                                      |
|---------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logical_xor](./examples/test_aclnn_logical_xor.cpp) | 通过[aclnnLogicalXor](./docs/aclnnLogicalXor.md)接口方式调用NotEqual算子。                                         |
| aclnn调用 | [test_aclnn_ne_scalar](./examples/test_aclnn_ne_scalar.cpp)     | 通过[aclnnNeScalar/aclnnInplaceNeScalar](./docs/aclnnNeScalar&aclnnInplaceNeScalar.md)接口方式调用NotEqual算子。   |
| aclnn调用 | [test_aclnn_ne_tensor](./examples/test_aclnn_ne_tensor.cpp)     | 通过[aclnnNeTensor/aclnnInplaceNeTensor](./docs/aclnnNeTensor&aclnnInplaceNeTensor.md)接口方式调用NotEqual算子。 |

## 贡献说明

| 贡献者      | 贡献方   | 贡献算子     | 贡献时间      | 贡献内容            |
|----------|-------|----------|-----------|-----------------|
| NotEqual | 个人开发者 | NotEqual | 2026/2/13 | NotEqual算子适配开源仓 |
