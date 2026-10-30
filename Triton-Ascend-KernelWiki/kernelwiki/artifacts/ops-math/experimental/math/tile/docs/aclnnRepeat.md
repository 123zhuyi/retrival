# aclnnRepeat

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2系列产品 | √ |

## 功能描述

- 算子功能：将输入张量按指定倍数在各维度重复拼接，等价于 `numpy.tile(input, multiples)` 或 PyTorch `tensor.repeat(multiples)`。
- 计算公式：

  $$out[i_0, i_1, ..., i_N] = self[i_0 \bmod s_0, i_1 \bmod s_1, ..., i_N \bmod s_N]$$

  其中 $s_0, s_1, ..., s_N$ 为输入各维度大小。

## 实现原理

调用 Ascend C 的 DataCopy 接口实现纯数据搬运，通过 Host 侧维度合并优化和多核并行来提升性能。

## 函数原型

Tile 算子对应内置 aclnnRepeat 接口（[两段式接口](../../../../docs/zh/context/two_phase_api.md)）：

```cpp
aclnnStatus aclnnRepeatGetWorkspaceSize(
    const aclTensor  *self,
    const aclIntArray *repeats,
    aclTensor        *out,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnRepeat(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

### aclnnRepeatGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 | 数据类型 | 是否必选 |
|--------|----------|------|----------|----------|
| self | 输入 | 输入张量 | aclTensor* | 是 |
| repeats | 输入 | 各维度重复倍数 | aclIntArray* | 是 |
| out | 输出 | 输出张量 | aclTensor* | 是 |
| workspaceSize | 输出 | workspace 大小 | uint64_t* | 是 |
| executor | 输出 | 算子执行器 | aclOpExecutor** | 是 |

### aclnnRepeat

| 参数名 | 输入/输出 | 描述 | 数据类型 | 是否必选 |
|--------|----------|------|----------|----------|
| workspace | 输入 | workspace 内存地址 | void* | 是 |
| workspaceSize | 输入 | workspace 大小 | uint64_t | 是 |
| executor | 输入 | 算子执行器 | aclOpExecutor* | 是 |
| stream | 输入 | 计算流 | aclrtStream | 是 |

## 支持的数据类型

| self 数据类型 | out 数据类型 |
|-------------|------------|
| FLOAT | FLOAT |
| FLOAT16 | FLOAT16 |
| BF16 | BF16 |
| INT32 | INT32 |
| INT16 | INT16 |
| INT8 | INT8 |
| UINT8 | UINT8 |
| UINT16 | UINT16 |
| UINT32 | UINT32 |
| UINT64 | UINT64 |
| BOOL | BOOL |
| COMPLEX64 | COMPLEX64 |

## 约束说明

- 输入张量维度范围 1-8，仅支持 ND 格式
- repeats 各维度值须为非负整数（≥0），出现 0 时输出对应维度为 0
- int64、double 数据类型暂不支持

## 调用示例

请参考 [test_aclnn_tile.cpp](../examples/test_aclnn_tile.cpp)。
