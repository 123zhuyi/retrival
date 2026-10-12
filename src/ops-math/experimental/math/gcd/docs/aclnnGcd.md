# aclnnGcd

[查看源码](https://gitcode.com/cann/ops-math/tree/master/experimental/math/gcd)

## 产品支持情况

| 产品 | 是否支持 | 说明 |
|---|:---:|---|
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ | 统一 Ascend C 实现 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ | 统一 Ascend C 实现 |

## 功能说明

`aclnnGcd` 对 `self` 和 `other` 做逐元素最大公约数计算。输入 shape 需要满足 broadcast 关系，输出 shape 需要等于 broadcast 后的 shape。

当前实现支持 `float32`、`float16`、`bfloat16`、`int8`、`uint8`、`int16`、`int32`、`int64`。输入可为 mixed dtype，ACLNN wrapper 先按 `PromoteType(self, other)` 转为内部 same-dtype Gcd 输入，再按需要 cast 到 `out` dtype。浮点有限数值按向 0 截断转换为整数后执行 Gcd；`NaN`/`Inf` 不属于当前有效输入域。

源码结构参考 `experimental/math/bitwise_and` 的扁平布局：host tiling 使用 `op_host/gcd_tiling.cpp`，kernel 使用 `op_kernel/gcd.cpp` 和 `op_kernel/gcd.h`，不再按架构拆分子目录或在 kernel 入口内按编译架构分叉。L0/API 使用单一 AiCore dtype 支持集合，不根据运行 NPU arch 切换 dtype 合同。

## 实现边界

- 当前 `aclnnGcd` 计算路径不调用 Python reference、PyTorch/torch_npu 等价 Gcd、官方 ACLNN Gcd、CANN 内置 TBE Gcd 或 CPU fallback。

## 算子设计

### ACLNN 与非连续 Tensor

`aclnnGcdGetWorkspaceSize` 先检查空指针、dtype、rank 和 broadcast shape。ACLNN wrapper 按原 Gcd 规则对普通 mixed 输入做 dtype promote，内部 `l0op::Gcd` 只接收同 dtype 输入，并在需要时 cast 到 `out` dtype。L2 在输入转换后显式校验两路 kernel 输入为受支持 same dtype；L0 在加入 launcher 前再次校验 same-dtype 签名，或校验 mixed 签名是否属于 `op_host/gcd_def.cpp` 注册的三组组合。未注册签名返回错误，不会进入 kernel。

输入通过 `l0op::Contiguous` 转为连续 Tensor。输出分派如下：

| 条件 | 执行路径 | 输出所有权 |
| --- | --- | --- |
| `self`、`other`、连续 `out` 在 promote 后为相同 dtype | `GcdToOutput` | Gcd kernel 直接写用户 `out`，避免临时 Tensor 和 `ViewCopy` |
| 元素数不超过 4096，且 dtype 属于已注册的三组 mixed 签名 | `GcdWithOutputType -> ViewCopy` | mixed Gcd kernel 写指定 dtype 临时结果，再复制到 `out` |
| 其他合同内输入 | `Gcd -> 按需 Cast -> ViewCopy` | same-dtype Gcd 写临时结果，L2 完成输出 dtype 和非连续 layout 适配 |

因此 `GcdToOutput` 不用于支持 mixed 输入或不同输出 dtype；其职责仅是省去同 dtype、连续输出场景的一次临时结果与复制。

### Host tiling

`op_host/gcd_tiling.cpp` 生成 `GcdTilingData`，内容包括：

- `rank` 和 broadcast 后的 `outputDims`，rank 限制为 1 到 8；
- `x1Strides`、`x2Strides`，stride 为 0 表示该维度按 broadcast 复用同一个输入元素；
- `totalNum`，即输出元素总数。

host tiling 按输出 storage word 数选择 `blockDim`：`int8/uint8` 每 4 个元素对应 1 个 32-bit word，`int16/float16/bfloat16` 每 2 个元素对应 1 个 32-bit word，`float32/int32` 每个元素对应 1 个 word，`int64` 每个元素对应 2 个 word；再按每 block 至少约 256 个 word 和运行时 AIV core 数取上限。该策略允许大 shape 使用多核，小 shape 保持较少 block 以降低调度开销。

### Kernel 数据流

`op_kernel/gcd.cpp` 使用 `op_kernel/gcd.h` 中的 GM scalar broadcast kernel：

- kernel 保持 ACLNN 生成 wrapper 兼容的五参数 ABI，但实际 tiling 数据在参数区内联传入；入口通过 `get_para_base()` 读取 tensor 地址，并从 `paramBase + 5` 解释 `GcdTilingData`。
- 每个 block 处理连续的输出 storage-word chunk，chunk 边界按 16 个 32-bit word 对齐，避免多核在相邻 byte/half lane 或 GM 写回粒度上交错写同一片区域。
- 当 `x1Strides` 和 `x2Strides` 都表示相对输出 dense contiguous same-shape 访问时，kernel 使用线性地址快路径，直接按输出线性位置读取两路输入并写回输出；存在 broadcast 维度时保持通用 cursor 路径。
- broadcast 不在 kernel 内物化输入。kernel 通过 `OffsetCursor` 从输出线性坐标推进 `x1Offset/x2Offset`，按 tiling stride 读取实际输入元素。
- `float32` 走普通模板路径，通过 IEEE-754 符号位、指数和尾数显式解码向 0 截断的 `uint64` 绝对值，避免将超出 `int64` 值域的 `float` 隐式转换为有符号整数；超出 `uint64` 可表示范围的有限值按 `UINT64_MAX` 封顶，无符号 Euclid GCD 结果直接 cast 回 `float`，不再经过 `int64`。
- `float16` 和 `bfloat16` 走 raw 16-bit lane 路径，解码有限浮点 bit pattern 的整数部分，计算 GCD 后编码回原浮点 dtype 的整数值。
- `int8/uint8` 以 32-bit word 读写 4 个 byte lane，`int16` 以 32-bit word 读写 2 个 halfword lane；`int32/int64` 使用通用 GM scalar 路径。signed integer 先得到绝对值，再计算 GCD。
- `int8/uint8` 在所有有效 lane 都位于 innermost dimension 内时使用 base-plus-lane-stride 快路径；否则退回逐 lane cursor 推进。这是基于 shape/stride 元数据的通用路径，不依赖 public case id 或输入值分布。
- 标量 GCD 内部对 `0..255` 小幅值使用小质因数分解路径、对大幅值使用 Euclid modulo；两条路径都按实际输入值计算同一个数学 GCD，不依赖 public case id、shape 编号或输出模式。

## 浮点输入语义

任务书要求覆盖 `fp32/fp16/bf16` dtype，但 GCD 的数学定义是整数域运算。本实现对所有三种浮点 dtype 使用同一条任务语义：

1. 输入必须是有限浮点值，`NaN` 和 `Inf` 不属于有效输入域。
2. 每个输入元素先按向 0 截断得到整数语义值，例如 `7.9 -> 7`、`-7.9 -> -7`、`0.75 -> 0`；`float32` 使用位级解码避免隐式窄化，绝对值超出 `uint64` 范围时按 `UINT64_MAX` 封顶。
3. 对截断后的整数执行 `gcd(abs(self), abs(other))`，并保留 `gcd(0, 0) = 0`。
4. 结果 cast/编码回原浮点 dtype，因此输出是原 dtype 中的整数值表示。

这是一种明确的工程语义，不等同于覆盖任意 fp32/fp16/bf16 bit pattern。不能把该路径宣称为支持 `NaN`、`Inf` 或未定义转换范围的任意浮点数。

## 函数原型

每个算子分为两段式接口，先调用 `aclnnGcdGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnGcd` 执行计算。

这里的“两段式”是 ACLNN 算子 API 的常见调用方式，不表示有两个 GCD 算法：

1. 第一段 `aclnnGcdGetWorkspaceSize` 做参数校验、shape/broadcast 检查、输入连续化和内部执行图构建，返回本次调用需要的 device workspace 大小以及 `aclOpExecutor`。
2. 调用方按第一段返回的 `workspaceSize` 申请 workspace。
3. 第二段 `aclnnGcd` 接收 workspace、executor 和 stream，通过 `CommonOpExecutorRun` 把第一段构建好的计算提交到指定 ACL stream 上执行。

因此，第一段主要是准备/构图/查询 workspace，第二段才是真正的异步执行提交；两段必须配套使用。

```cpp
aclnnStatus aclnnGcdGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* other,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnGcd(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnGcdGetWorkspaceSize 参数

| 参数名 | 输入/输出 | 说明 | dtype | shape | format | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `self` | 输入 | 第一个输入 Tensor，与 `other` 满足 broadcast 关系，参与 dtype promote | `float32`, `float16`, `bfloat16`, `int8`, `uint8`, `int16`, `int32`, `int64` | rank 1-8 | ND | √ |
| `other` | 输入 | 第二个输入 Tensor，与 `self` 满足 broadcast 关系，参与 dtype promote | 同 `self` 支持集合 | rank 1-8 | ND | √ |
| `out` | 输出 | Gcd 计算结果，shape 为 broadcast 后 shape；可由 promote 结果 cast 到支持 dtype | 同 `self` 支持集合 | rank 1-8 | ND | √ |
| `workspaceSize` | 输出 | Device 侧 workspace 大小 | - | - | - | - |
| `executor` | 输出 | 算子执行器 | - | - | - | - |

## aclnnGcd 参数

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| `workspace` | 输入 | Device 侧 workspace 起始地址 |
| `workspaceSize` | 输入 | workspace 大小，由第一段接口返回 |
| `executor` | 输入 | 第一段接口返回的执行器 |
| `stream` | 输入 | ACL stream |

## 返回值与错误场景

返回 `aclnnStatus`。第一段接口会在以下场景返回参数错误：

- `self`、`other` 或 `out` 为空指针。
- `self`、`other` 或 `out` dtype 不在支持集合内，或 `self`/`other` 无法按 `PromoteType` 推导出支持的内部 dtype。
- dtype 不在 `float32/float16/bfloat16/int8/uint8/int16/int32/int64` 范围内。
- 输入 rank 超过 8。
- `self` 与 `other` 不满足 broadcast 关系。
- `out` shape 与 broadcast 后 shape 不一致。

## 约束说明

- 浮点路径仅定义有限数值的向 0 截断 Gcd 语义。
- 当前实现支持原 Gcd promote/cast；内部 AiCore Gcd kernel 仍只接收 ACLNN wrapper 转换后的 same-dtype 输入。
- 当前实现不提供整算子 fallback；不支持的 dtype、rank、broadcast 或浮点值域会按上述合同校验失败。

## 调用示例

完整 ACLNN 调用示例见 `experimental/math/gcd/examples/test_aclnn_gcd.cpp`。
