# AssignSub

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：计算 `var - value` 并将结果写入输出。

- 计算公式：

  $$
  var\_out = var - value
  $$

## 参数说明

| 参数名   | 输入/输出 | 说明                                                                 |
| -------- | --------- | -------------------------------------------------------------------- |
| var      | 输入      | 被减数张量，数据类型支持FLOAT16、INT8、FLOAT、INT32、UINT8、BF16、INT64，数据格式支持ND。 |
| value    | 输入      | 减数张量，数据类型与var一致，shape与var一致，数据格式支持ND。         |
| var_out  | 输出      | 输出张量，shape与var一致，数据类型与var一致，数据格式支持ND。         |

## 约束与限制

- var和value的shape及数据类型必须完全一致（不支持broadcast）。
- 数据格式仅支持ND。
- int8/uint8的减法溢出按模256环绕处理。
- int64类型输入值范围限制在int32可表示范围内（[-2^31+1, 2^31-1]）。

## 调用说明

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)

| 目录 | 描述 |
| ---- | ---- |
| [test_aclnn_assign_sub.cpp](./examples/test_aclnn_assign_sub.cpp) | 通过aclnn调用的方式调用AssignSub算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|--------|--------|---------|---------|---------|
| Xzz | 西工大智能感知交互实验室 | AssignSub | 2026/07/12 | 新增AssignSub算子 |
