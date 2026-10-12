# aclnnKlDiv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：计算输入 `x` 和 `target` 之间的 Kullback-Leibler 散度。

- 计算公式：

  $$
  loss(x, target) = target \cdot (\log(target) - x)
  $$

  当 `log_target=true` 时：

  $$
  loss(x, target) = \exp(target) \cdot (target - x)
  $$

## 函数原型

- 每个算子分为两段式接口，必须先调用 "aclnnKlDivGetWorkspaceSize" 接口获取入参并根据计算流程计算所需workspace大小，再调用 "aclnnKlDiv" 接口执行计算。

  ```Cpp
  aclnnStatus aclnnKlDivGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* target,
    const char* reduction,
    bool logTarget,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
  ```

  ```Cpp
  aclnnStatus aclnnKlDiv(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream);
  ```

## 参数说明

### aclnnKlDivGetWorkspaceSize

| 参数名       | 输入/输出 | 说明                                                                 |
| ------------ | --------- | -------------------------------------------------------------------- |
| x            | 输入      | 公式中的输入x，数据类型支持FLOAT16、FLOAT、BF16，数据格式支持ND。   |
| target       | 输入      | 公式中的target，数据类型支持FLOAT16、FLOAT、BF16，数据格式支持ND。  |
| reduction    | 输入      | 归约方式，支持"none"、"mean"、"sum"、"batchmean"，默认"mean"。       |
| logTarget    | 输入      | target是否已取对数，默认false。                                      |
| out          | 输出      | 输出张量。reduction为"none"时shape与x相同，否则为标量[1]。           |
| workspaceSize| 输出      | 返回需要在Device侧申请的workspace大小。                              |
| executor     | 输出      | 返回op执行器。                                                       |

## 约束与限制

- x和target的shape需可广播（broadcastable）。
- 数据类型仅支持FLOAT16、FLOAT、BF16。
