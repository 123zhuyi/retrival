# NotEqualV2

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
| self  | 输入       | 待进行not_equal_v2计算的入参，公式中的$self_i$。  | FLOAT16,FLOAT,INT32,INT8,UINT8,BOOL,BFLOAT16 | ND   |
| other | 输入       | 待进行not_equal_v2计算的入参，公式中的$other_i$。 | FLOAT16,FLOAT,INT32,INT8,UINT8,BOOL,BFLOAT16 | ND   |
| out   | 输出       | 待进行not_equal_v2计算的出参，公式中的$out_i$。   | BOOL                                               | ND   |

## 约束说明

- `aclnnNeTensor`、`aclnnInplaceNeTensor` 和 `aclnnLogicalXor` 动态执行路径支持 broadcast，`self` 与 `other` 需满足广播规则，`out` 的 shape 需与广播后的 shape 一致。
- `aclnnInplaceNeTensor` 需额外满足广播后的 shape 与 `selfRef` 的 shape 一致。
- 当前对外文档所列输入数据类型范围内，不支持 `int64`。

## 调用说明

| 调用方式    | 调用样例                                                            | 说明                                                                                                      |
|---------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logical_xor](./examples/test_aclnn_logical_xor.cpp) | 通过[aclnnLogicalXor](./docs/aclnnLogicalXor.md)接口方式调用NotEqualV2算子。                                         |
| aclnn调用 | [test_aclnn_ne_scalar](./examples/test_aclnn_ne_scalar.cpp)     | 通过[aclnnNeScalar/aclnnInplaceNeScalar](./docs/aclnnNeScalar&aclnnInplaceNeScalar.md)接口方式调用NotEqualV2算子。   |
| aclnn调用 | [test_aclnn_ne_tensor](./examples/test_aclnn_ne_tensor.cpp)     | 通过[aclnnNeTensor/aclnnInplaceNeTensor](./docs/aclnnNeTensor&aclnnInplaceNeTensor.md)接口方式调用NotEqualV2算子。 |

## 贡献说明

| 贡献者      | 贡献方   | 贡献算子     | 贡献时间      | 贡献内容            |
|----------|-------|----------|-----------|-----------------|
| NotEqualV2 | 个人开发者 | NotEqualV2 | 2026/2/13 | NotEqualV2算子适配开源仓 |
