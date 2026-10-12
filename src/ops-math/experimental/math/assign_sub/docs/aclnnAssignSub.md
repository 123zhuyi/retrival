# aclnnAssignSub

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：计算 `var - value` 并将结果写入输出。

- 计算公式：

  $$
  var\_out = var - value
  $$

## 函数原型

- 每个算子分为两段式接口，必须先调用 "aclnnAssignSubGetWorkspaceSize" 接口获取入参并根据计算流程计算所需workspace大小，再调用 "aclnnAssignSub" 接口执行计算。

  ```Cpp
  aclnnStatus aclnnAssignSubGetWorkspaceSize(
    const aclTensor* var,
    const aclTensor* value,
    const aclTensor* varOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
  ```

  ```Cpp
  aclnnStatus aclnnAssignSub(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream);
  ```

## 参数说明

### aclnnAssignSubGetWorkspaceSize

| 参数名       | 输入/输出 | 说明                                                                 |
| ------------ | --------- | -------------------------------------------------------------------- |
| var          | 输入      | 被减数张量，数据类型支持FLOAT16、INT8、FLOAT、INT32、UINT8、BF16、INT64。 |
| value        | 输入      | 减数张量，数据类型与var一致，shape与var一致。                         |
| varOut       | 输出      | 输出张量，shape与var一致。                                           |
| workspaceSize| 输出      | 返回需要在Device侧申请的workspace大小。                              |
| executor     | 输出      | 返回op执行器。                                                       |

## 约束与限制

- var和value的shape及数据类型必须一致。
- 数据类型支持FLOAT16、INT8、FLOAT、INT32、UINT8、BF16、INT64。
