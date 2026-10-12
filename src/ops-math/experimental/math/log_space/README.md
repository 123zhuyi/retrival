# LogSpace

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

> 整型输出（INT8/INT16/INT32/UINT8）仅 Atlas A2/A3 支持；Ascend 950PR/Ascend 950DT 维持原有的 FLOAT/FLOAT16/BFLOAT16 输出。

## 功能说明

- 算子功能：创建一个大小为 $steps$ 的一维张量，其值在 $base^{start}$ 到 $base^{end}$ 之间按对数尺度均匀间隔（含端点），以 $base$ 为底。在原 FLOAT/FLOAT16/BFLOAT16 基础上**新增 INT8/INT16/INT32/UINT8 输出**（整型按向零取整）。
- 计算公式：

  $$ result_i = base^{\left(start + i \cdot \frac{end - start}{steps - 1}\right)},\quad i \in [0, steps) $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| -- | -- | -- | -- | -- |
| start | 输入（aclScalar*） | 对数序列的起始指数 | FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、INT16、INT32、UINT8 | ND |
| end | 输入（aclScalar*） | 对数序列的结束指数 | FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、INT16、INT32、UINT8 | ND |
| steps | 输入（int64_t） | 序列中的元素数量 | int64_t | - |
| base | 输入（double） | 对数空间的底数 | double | - |
| result | 输出（aclTensor*） | 对数间隔序列张量 | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、UINT8（整型仅 Atlas A2/A3） | ND |

> 上表为 aclnn 接口 `aclnnLogSpace(start, end, steps, base, result, ...)` 的签名；算子定义 `op_host/log_space_def.cpp` 与之不同：LogSpace 是纯生成类算子，**无输入 tensor**，start/end/steps/base 四者均为算子属性（Host 侧已由 aclScalar/double 转为 float 下发），result 为唯一输出。

## 约束说明

- `result` 为一维张量，长度 = `steps`。
- 整型输出（INT8/INT16/INT32/UINT8）仅 Atlas A2/A3 支持，在 Ascend 950PR/Ascend 950DT 上不支持。
- 整型输出按 `base^x` 向零取整（与 torch `.to(int)` 一致）；溢出按饱和处理，应保证 `base^x` 落在输出 dtype 范围内。
- UINT8 输出要求 `base^x ≥ 0`（`base > 0` 时恒成立）。
- `base > 0`，`0 ≤ steps ≤ UINT32_MAX`。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -- | -- | -- |
| aclnn API 调用 | `examples/test_aclnn_log_space.cpp` | 覆盖 7 种 dtype + steps=0/1 边界；接口说明见 `docs/aclnnLogSpace.md` |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 开源社区贡献者 | 开源社区 | LogSpace | 2026/07/16 | LogSpace 算子在原有 Ascend950 基础上新增 Atlas A2/A3 支持及整型输出 |
