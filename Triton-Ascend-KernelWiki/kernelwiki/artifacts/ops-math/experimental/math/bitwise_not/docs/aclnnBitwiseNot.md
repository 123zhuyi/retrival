# aclnnBitwiseNot

> 本文档为 BitwiseNot 算子（`experimental/math/bitwise_not`，Ascend910B 原生 AscendC 实现）的 aclnn API 用户接口文档，面向算子使用者。本算子为 BitwiseNot 的原生 AscendC 实现，OpType 为 `BitwiseNot`，对外提供两段式接口 `aclnnBitwiseNot`。

## 1. 产品支持情况

| 产品 | 是否支持 |
|-----|---------|
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品（Ascend910B） | √ |

> 本算子为 `experimental/math` 投放区的 Ascend910B 原生 AscendC 贡献实现。当前交付范围仅 **Ascend910B**。

## 2. 功能说明

- **算子功能**：对输入 tensor 逐元素求 BitwiseNot（`out = ~self`），语义按 dtype 分支：
  - **整型（INT8/INT16/INT32/INT64/UINT8）**：进行**按位非（按位补码）**运算。
  - **BOOL**：进行**逻辑非**运算（`0 ↔ 1`，即 `(self == 0) ? 1 : 0`，非裸位翻转）。
- **计算公式**：

$$
out_i = \lnot\,self_i
$$

  其中：对 BOOL 为逻辑非 $out_i = (self_i = 0) ? 1 : 0$；对整型为按位非（有符号整型等价于 $out_i = -self_i - 1$）。

- **对标框架**：`torch.bitwise_not(input)` / `~input` / `numpy.invert`。

## 3. 函数原型

每个算子分为两段式接口，必须先调用 "aclnnBitwiseNotGetWorkspaceSize" 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 "aclnnBitwiseNot" 接口执行计算。

```cpp
aclnnStatus aclnnBitwiseNotGetWorkspaceSize(
    const aclTensor* self,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);

aclnnStatus aclnnBitwiseNot(
    void*            workspace,
    uint64_t         workspaceSize,
    aclOpExecutor*   executor,
    aclrtStream      stream);
```

> 头文件：`aclnn_bitwise_not.h`（位于自定义算子包 `op_api/include/aclnnop/`）。
> 库：`libcust_opapi.so`（位于自定义算子包 `op_api/lib/`），链接时需 `-lcust_opapi -lnnopbase`。

## 4. 参数说明

### 4.1 aclnnBitwiseNotGetWorkspaceSize

| 参数名 | 输入/输出 | 数据类型 | 描述 | 使用说明 |
|-------|----------|---------|------|---------|
| self | 输入 | const aclTensor* | 公式中的 self（待按位取反的输入张量） | 数据类型支持 INT8、INT16、INT32、INT64、UINT8、BOOL；数据格式支持 ND；支持非连续 Tensor |
| out | 输出 | aclTensor* | 公式中的 out（按位取反结果） | shape 与 self 相同；数据类型需与 self 一致；数据格式支持 ND，且需与 self 一致 |
| workspaceSize | 输出 | uint64_t* | 返回需要在 Device 侧申请的 workspace 大小 | - |
| executor | 输出 | aclOpExecutor** | 返回 op 执行器，包含算子计算流程 | - |

### 4.2 aclnnBitwiseNot

| 参数名 | 输入/输出 | 数据类型 | 描述 |
|-------|----------|---------|------|
| workspace | 输入 | void* | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | uint64_t | 在 Device 侧申请的 workspace 大小，由第一段接口 aclnnBitwiseNotGetWorkspaceSize 获取 |
| executor | 输入 | aclOpExecutor* | op 执行器，包含了算子计算流程 |
| stream | 输入 | aclrtStream | 指定执行任务的 Stream |

## 5. 返回值

aclnnStatus：返回状态码，具体参见 aclnn 返回码。

### 5.1 第一段接口错误码

第一段接口完成入参校验，出现以下场景时报错：

| 返回码 | 错误码 | 描述 |
|-------|-------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的 self 或 out 是空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | self 的数据类型不在支持范围之内（非 INT8/INT16/INT32/INT64/UINT8/BOOL） |
| ACLNN_ERR_PARAM_INVALID | 161002 | self 与 out 的数据类型不同 |
| ACLNN_ERR_PARAM_INVALID | 161002 | self 与 out 的 shape 不同 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入/输出为私有格式（仅支持 ND/NCHW/NHWC/HWCN/NDHWC/NCDHW） |

## 6. 约束说明

- **数据类型**：本算子（Ascend910B）支持 INT8、INT16、INT32、INT64、UINT8、BOOL；`self.dtype` 必须与 `out.dtype` 一致（无类型提升）。
  - UINT16/UINT32/UINT64 不在本算子（Ascend910B）支持范围内。
- **shape**：`out.shape` 必须与 `self.shape` 完全相同（逐元素，无 broadcast）；维度数 ≤ 8。
- **格式**：仅支持 ND 类公开格式，禁止私有格式。
- **空 Tensor**：支持空 Tensor（element 数为 0），此时 workspace 为 0 并直接返回成功。
- **语义分支**：BOOL 类型执行逻辑非（结果规整为 0/1），整型执行按位非；这与 PyTorch `torch.bitwise_not` 行为一致。
- **精度**：整数/逻辑算子，结果与参考实现（`torch.bitwise_not` / numpy `~`）**按位精确逐元素相等**（atol = 0，rtol = 0）。
- **确定性计算**：aclnnBitwiseNot 默认确定性实现（逐元素算子，天然确定性）。
- **单 tensor 字节数上限**：当前实现的 GM 偏移/长度计算使用 `uint32`，单 tensor 的元素总字节数须 ≤ `UINT32_MAX`（约 4 GB）；超大 tensor 暂不支持。

## 7. 调用示例

核心调用片段（取自 [`../examples/test_aclnn_bitwise_not.cpp`](../examples/test_aclnn_bitwise_not.cpp)，单输入 self → 单输出 out）：

```cpp
#include "aclnn_bitwise_not.h"

// 0. 初始化
aclInit(nullptr);

// 1. 构造 self / out（dtype/shape 相等，ND，连续）
aclTensor *self = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                  ACL_FORMAT_ND, shape.data(), shape.size(), selfDev);
aclTensor *out  = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                  ACL_FORMAT_ND, shape.data(), shape.size(), outDev);

// 2. 第一段：workspace + 执行器
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnBitwiseNotGetWorkspaceSize(self, out, &workspaceSize, &executor);
void *workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 3. 第二段：执行；之后 D2H 回拷 outDev 即得 ~self
aclnnBitwiseNot(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

完整可运行示例（含 device/stream 初始化、H2D/D2H、int32/int8/uint8/bool 四分支 CPU golden 自验 == `numpy.invert`、资源释放）见：

- **eager 两段式 aclnnBitwiseNot**：[`../examples/test_aclnn_bitwise_not.cpp`](../examples/test_aclnn_bitwise_not.cpp)
- **图模式（GE IR）**：[`../examples/test_geir_bitwise_not.cpp`](../examples/test_geir_bitwise_not.cpp)（构造单算子图 `BitwiseNot`，经 GE Session `AddGraph`/`RunGraph` 调度）
- **一键编译运行脚本**：[`../examples/run.sh`](../examples/run.sh)

```bash
cd experimental/math/bitwise_not/examples
bash run.sh                  # 编译 eager + geir 两个示例；在真实 NPU 上运行 eager 示例
```

真实 NPU 上 eager 示例预期输出（节选）：

```bash
[PASS] int32  ~self bitwise-exact
[PASS] int8   ~self bitwise-exact (boundary ~(-128)=127)
[PASS] uint8  ~self bitwise-exact (~0=255 ~255=0)
[PASS] bool   !self (0<->1) bitwise-exact
==== BitwiseNot eager example: ALL PASS ====
```
