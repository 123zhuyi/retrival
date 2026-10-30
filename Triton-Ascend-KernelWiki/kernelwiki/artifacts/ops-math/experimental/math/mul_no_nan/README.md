# MulNoNan 自定义算子

## 产品支持情况

| 产品型号 | 支持的 CANN 版本 |
|---------|----------------|
| Ascend 950PR / Ascend 950DT | CANN 9.0.0 及以上 |

## 功能说明

MulNoNan 是一种安全乘法算子，完成两个张量的逐元素乘法，当乘数 y 为零时强制输出 0（即使被乘数 x 为 NaN 或无穷大）。支持 broadcast。

计算公式：

$$z_i = \begin{cases} 0 & \text{if } y_i = 0 \\ x_i \times y_i & \text{otherwise} \end{cases}$$

核心语义：当 y=0 时，无论 x 为何值（包括 NaN、Inf、-Inf），输出均为 0。

## 参数说明

### 输入

| 参数名 | 数据类型 | 数据格式 | shape | 说明 |
|-------|---------|---------|-------|------|
| x | FLOAT16, FLOAT, BFLOAT16 | ND | 0-8 维，任意 shape | 被乘数张量，支持空 Tensor，支持 broadcast |
| y | FLOAT16, FLOAT, BFLOAT16 | ND | 0-8 维，任意 shape | 乘数张量，支持空 Tensor，支持 broadcast。y=0 时输出强制为 0 |

### 输出

| 参数名 | 数据类型 | 数据格式 | shape | 说明 |
|-------|---------|---------|-------|------|
| z | FLOAT16, FLOAT, BFLOAT16 | ND | x 和 y broadcast 后的 shape | 输出张量，dtype 与输入一致。不支持空 Tensor |

## 约束说明

- x 与 y 的数据类型必须一致，不支持混合 dtype 输入。
- z 的数据类型须与 x 一致。
- x 和 y 的 shape 必须满足 broadcast 规则（遵循标准 NumPy 广播规则：从右向左对齐，每个维度要么相等、要么其中一个为 1）。
- z 的 shape 为 x 和 y broadcast 后的 shape。
- 维度范围：0-8 维。
- 输入 x、y 支持空 Tensor（元素数为 0），输出为对应 shape 的空 Tensor。
- aclnnMulNoNan 默认确定性实现。

## 调用说明

### ACLNN 接口（两段式）

```cpp
// 第一段：获取 workspace 大小和执行器
aclnnStatus aclnnMulNoNanGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    aclTensor *z,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 第二段：执行计算
aclnnStatus aclnnMulNoNan(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

调用示例参见 [examples/test_aclnn_mul_no_nan.cpp](examples/test_aclnn_mul_no_nan.cpp)。
