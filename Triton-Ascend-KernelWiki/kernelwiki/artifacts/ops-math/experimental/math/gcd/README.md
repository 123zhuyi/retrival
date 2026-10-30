# Gcd

## 功能概述

`aclnnGcd` 对 `self` 和 `other` 做逐元素最大公约数计算，输入 shape 需要满足 broadcast 关系，输出 shape 为 broadcast 后的结果。当前 experimental 实现支持 ND、rank 1-8、非连续输入/输出和以下数据类型：

- `float32`
- `float16`
- `bfloat16`
- `int8`
- `uint8`
- `int16`
- `int32`
- `int64`

浮点输入按任务验证口径处理：有限数值计算前按 C/C++ 整数转换规则向 0 截断，再执行整数 Gcd，结果可 cast 回请求的输出 dtype。`NaN`/`Inf` 不属于当前有效输入域。混合 dtype 输入支持 `float32`、`float16`、`bfloat16`、`int8`、`uint8`、`int16`、`int32`、`int64` 范围内的原 Gcd promote/cast 规则。

## 产品与源码路径

| 产品/架构 | 编译 SoC | host tiling | kernel | 支持 dtype |
|---|---|---|---|---|
| Atlas A2/A3 | `ascend910b`, `ascend910_93` | `op_host/gcd_tiling.cpp` | `op_kernel/gcd.cpp`, `op_kernel/gcd.h` | `float32`, `float16`, `bfloat16`, `int8`, `uint8`, `int16`, `int32`, `int64` with promote/cast |

源码结构参考 `experimental/math/bitwise_and` 的扁平布局：host tiling 只有 `op_host/gcd_tiling.cpp`，kernel 只有 `op_kernel/gcd.cpp` 和 `op_kernel/gcd.h`，不再按架构建子目录或在 kernel 入口内分支。ACLNN wrapper 先按 `PromoteType(self, other)` 将普通 mixed 输入转换为内部 same-dtype Gcd，再按需要 cast 到 `out` dtype；仅三组在 `op_host/gcd_def.cpp` 注册的小 mixed 签名可走融合 kernel。L2 在 launch 前校验 promote 后的输入为受支持 same dtype，L0 再次校验最终签名已注册，未匹配的签名不会进入 kernel。L0/API 使用单一 AiCore dtype 支持集合，不根据运行 NPU arch 切换 dtype 合同。

当 promote 后的 `self`、`other` 与连续 `out` dtype 完全相同时，`GcdToOutput` 直接把同 dtype Gcd 结果写入用户输出，避免临时结果和一次 `ViewCopy`；它不负责 mixed 输入或不同输出 dtype。其他情况使用“Gcd -> 按需 Cast -> ViewCopy”通用路径。

## 实现说明

- 当前 Ascend C 实现的 host tiling 在 `op_host/gcd_tiling.cpp` 内生成 rank、broadcast 后 shape、两路输入 stride 和输出元素数；stride 为 0 表示该维度 broadcast 复用同一输入元素。
- host tiling 按输出 storage word 数选择 `blockDim`：`int8/uint8` 每 4 个元素 1 个 word，`int16/float16/bfloat16` 每 2 个元素 1 个 word，`float32/int32` 每元素 1 个 word，`int64` 每元素 2 个 word；大 shape 可使用多 AIV core，小 shape 可保持较少 block。
- kernel 侧使用 GM scalar broadcast 计算，不物化输入 broadcast。入口保持 ACLNN wrapper 的五参数 ABI，通过 `get_para_base()` 读取 tensor 地址，并从 `paramBase + 5` 读取内联 tiling 数据。
- 多核切分使用连续 storage-word chunk，边界按 16 个 32-bit word 对齐，避免 subword dtype 的 byte/half lane 在跨核写回时互相覆盖。
- 当两路输入相对 broadcast 输出都是 dense contiguous same-shape 布局时，kernel 走线性 GM 地址快路径，直接以输出线性坐标读写输入和输出；存在 broadcast stride 时仍走通用 `OffsetCursor` 路径。
- 单元素 GCD 计算保持全域正确：`0..255` 小幅值输入使用小质因数分解路径减少标量 modulo/循环，大幅值输入保留 Euclid modulo 路径。
- `float32` 通过 IEEE-754 位级解码向 0 截断到 `uint64` 整数语义，避免超出 `int64` 值域时的隐式转换，超出 `uint64` 范围时按 `UINT64_MAX` 封顶，GCD 结果直接转回 `float32` 而不经过 `int64`；`float16/bfloat16` 通过 raw 16-bit lane 解码有限浮点的整数部分，计算后编码回原浮点 dtype。
- `int8/uint8/int16` 使用 raw lane 读写和 packed 32-bit 输出；`int32/int64` 使用通用 GM scalar 路径。signed dtype 先得到绝对值，再执行 Euclid GCD。
- ACLNN 层先将非连续输入转为连续 Tensor，输出仍通过 ACLNN copy/view 流程写回用户 out，因此输入和输出均可覆盖非连续场景。
- 当前实现不调用 Python reference、PyTorch/torch_npu 等价 Gcd、官方/vendor Gcd、CANN TBE Gcd 或 CPU fallback 作为计算路径。

## 限制说明

- 浮点支持只表示 `float32/float16/bfloat16` 有限值按向 0 截断后执行 GCD，并 cast 回原 dtype；不表示支持 `NaN`、`Inf` 或任意未定义浮点 bit pattern。
- 完整支持边界以任务书/API 文档的 dtype、rank、broadcast、非连续、整数 promote/cast 和浮点有限值语义为准。

## 测试

ACLNN 调用示例位于 `examples/test_aclnn_gcd.cpp`。L2 UT 位于 `tests/ut/op_api/test_aclnn_gcd.cpp`，通过仓库统一的 `math_op_api_ut` 目标执行。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| 于森浩、刘崇威 | 深圳河套学院、大连理工大学、香港中文大学 | Gcd | 2026/07/16 | Gcd 算子适配开源仓 |
