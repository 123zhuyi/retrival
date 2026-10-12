# Slogdet（experimental / 原生 AscendC 实现）

> 本算子为 `experimental/math/slogdet` 贡献的 **原生 AscendC 实现**（registry-invoke 方式，自带 Kernel/Tiling），目标芯片 **Ascend 910B / 910C（Atlas A2/A3，fp32）**。
> 接口与仓库内 `math/slogdet`（aclnn-only 转发壳，经 `LogMatrixDeterminant` 转发）真值源保持一致；本实现以 **带部分主元的 LU 分解** 在 NPU 上直接计算行列式的符号与对数绝对值，对标 `torch.linalg.slogdet`。

## 产品支持情况

> 本次原生 AscendC 实现交付 **Atlas A2/A3 训练/推理系列产品（Ascend 910B / 910C，910C/A3 对应构建参数 `ascend910_93`），数据类型仅 FLOAT(fp32)**。其余产品/数据类型沿用 `math/slogdet` 的 aclnn 转发实现，不在本次交付范围内。

| 产品                                                          | 是否支持 |
| :------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                        |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：计算输入方阵 `self` 的行列式的 **符号** `signOut` 与 **行列式绝对值的自然对数** `logOut`，支持 batch 方阵输入，对标 `torch.linalg.slogdet`（单输入、双输出）。

- 计算公式：

  $$
  signOut = sign(det(self))
  $$

  $$
  logOut = \log(|det(self)|)
  $$

  其中 `det` 表示方阵行列式，`|·|` 表示绝对值（实数取绝对值）。若 `det(self) = 0`（矩阵奇异），则 `logOut = -inf` 且 `signOut = 0`。

- 数学算法（LU 分解 + 部分主元）：

  行列式不按定义展开（O(n!) 不可行），而是经 **带部分主元（partial pivoting）的 LU 分解** 求得，与 LAPACK / torch CPU 实现一致：

  1. 对每个 `n×n` 方阵做 `P·A = L·U` 分解（`L` 单位下三角、`U` 上三角、`P` 行置换）。第 k 步在第 k 列 k..n-1 行中选绝对值最大元素作主元（部分主元），与第 k 行交换（记录置换奇偶性），再消去下方元素。
  2. `det(A) = (-1)^{行交换次数} · ∏_i U_{ii}`。
  3. 由 `U` 对角元一次性合成双输出：
     - `logOut = Σ_i \log|U_{ii}|`（**对数域累加**，规避大 n 连乘的上溢/下溢）；
     - `signOut = (置换符号) · ∏_i sign(U_{ii})`（实数 ±1；置换符号为 `(-1)^{行交换次数}`）；
     - 若任一主元绝对值低于奇异阈值（`det = 0`）：`logOut = -inf`、`signOut = 0`。

  **数值稳定性**：部分主元保证消元因子 `|L_{ij}| ≤ 1`，抑制中间放大；log 域累加规避连乘溢出；奇异判定采用 LAPACK 风格相对阈值（`≈ n · FLT_EPSILON(fp32) · maxAbs`）。

## 参数说明

> 接口参数顺序以 L0 真值源 `aclnn_slogdet_native.h` 为准：`aclnnSlogdetGetWorkspaceSize(self, signOut, logOut, ...)`，即 **signOut 在前、logOut 在后**。

| 参数名  | 输入/输出/属性 | 描述                                                         | 数据类型 | 数据格式 |
| :------ | :------------: | :----------------------------------------------------------- | :------- | :------- |
| self    |      输入      | 计算公式中的输入 `self`，方阵。shape 满足 `(*, n, n)` 形式，`*` 为 0 或更多维 batch，`n` 为正整数（**首版上界 `1 ≤ n ≤ 4095`**，见下「约束说明」）。支持空 Tensor、支持非连续 Tensor。 | FLOAT    | ND       |
| signOut |      输出      | 计算公式中的输出 `signOut`，行列式符号。shape 与 `self` 的 batch 一致（`self` 去掉最后两维）。数据类型与 `self` 一致。 | FLOAT    | ND       |
| logOut  |      输出      | 计算公式中的输出 `logOut`，行列式绝对值的自然对数。shape 与 `self` 的 batch 一致。数据类型与 `self` 一致。 | FLOAT    | ND       |

- **shape 规则**：输入 `[*, n, n]` → 两输出均为 batch 形状 `[*]`。例：`self=[3,2,2]` → `signOut=[3]`、`logOut=[3]`；`self=[2,2]`（无 batch）→ 两输出为标量形状 `[]`。
- 本次原生 AscendC 实现首版仅支持 **FLOAT**。`math/slogdet` 转发实现额外支持 DOUBLE、COMPLEX64、COMPLEX128（self 为 COMPLEX 时 signOut/logOut 须为 COMPLEX），不在本实现范围内。

## 约束说明

- **数据类型**：本实现仅支持 **FLOAT(fp32)**；`self`、`signOut`、`logOut` 数据类型需一致。
- **shape 约束**：`self` 维度必须 ≥ 2，且最后两维必须相等（**方阵**），否则报 `ACLNN_ERR_PARAM_INVALID`；`signOut`/`logOut` 的 shape 必须等于 `self` 去掉最后两维的 batch 形状。
- **方阵维度 n 上界（首版 `n ≤ 4095`）**：BLOCKED（large-n）路径列 gather 用 `DataCopyPad` 的 `blockCount`（uint16，取值范围 `[1,4095]`，= 子列长度 `n-k ≤ n`）；`n > 4095` 会使该参数静默越界，故 host tiling 对 `n > 4095` **显式返回错误**（`GRAPH_FAILED`），不静默错误。功能 ST 验证至 `n=512`；`513..4095` 走相同 BLOCKED 路径、无新增参数约束。后续若需更大 `n`，可将列 gather 按 `blockCount≤4095` 分段放宽。
- **batch**：batch 维各方阵相互独立，按核切分并行计算（核数运行时由 `GetBlockNum()` 动态获取，不写死）。
- **奇异矩阵**：当 `det(self) = 0` 时，对应 batch 位置的 `logOut = -inf`、`signOut = 0`。
- **空 Tensor**：`self` 为空（`numel == 0`）时 `workspaceSize = 0`，直接返回成功，不计算。
- **溢出值**：输入数据中不支持存在 `Inf` / `NaN`，否则行为未定义。
- **确定性**：`aclnnSlogdet` 默认确定性实现。列消元、log 域累加顺序固定，batch 按核独立切分（核间无累加依赖），相同输入产生 bitwise 一致的输出。
- **精度**：fp32 浮点社区标准（双万分之一，rtol/atol = 1e-4 量级）；奇异位 `-inf`/`0` 精确匹配，不参与 rtol 比对。

## 性能特性

本实现按 **单矩阵核内驻留策略** 自动选路（host 侧 Tiling 由运行时 UB 容量推导 `N_RESIDENT_MAX`，不写死），分两条路径：

| 路径 | 触发条件 | 适用场景 | 说明 |
| :--- | :------- | :------- | :--- |
| **FULL**（全驻留 LU，MEM_STRATEGY=0） | `n ≤ N_RESIDENT_MAX`（单矩阵 `n×n` fp32 + 临时 buffer ≤ 可用 UB，约 184KB） | 小/中等 n（主路径，覆盖绝大多数用例 n≤~200） | U 工作区一次性搬入 UB，主元搜索/消元全程在 UB；SCALAR-bound（LU 串行依赖 + 主元/log 标量）。 |
| **BLOCKED**（核内分块 LU，MEM_STRATEGY=1） | `n > N_RESIDENT_MAX` | 大 n（UB 容量不足以全驻留 `n×n`） | U 工作区按列块分块、消元行块化（ROW_BLOCK=16 批量 DMA）搬运于 GM↔UB；MTE2/MTE3 微小 DMA 延迟-bound。 |

- 两条路径共用同一套 batch 按核切分与 LU 主循环结构，差异仅在 U 工作区是否全驻留 UB；按 `n` 与 UB 容量自动选路，对调用方透明。
- 性能量级（真机 Ascend 910B3，满频 1800MHz，`msprof op` 采集）：FULL 路径 n=4/16/64 约 5.8/15.3/139 us；BLOCKED 路径 n=256/512 约 3.1/14.1 ms（行块化 DMA 优化后较优化前提速 3.6~3.9×）。batch 维多核并行有效摊薄单矩阵耗时（如 [8,32,32] 8 核、[3,4,5,2,2] 40 核）。
- 本算子为带数据依赖的串行 LU（单矩阵内无法并行，算法固有），并非计算密集型；首版以正确性/精度优先，性能无硬性指标。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn 调用 | [examples/test_aclnn_slogdet.cpp](examples/test_aclnn_slogdet.cpp) | 通过 [aclnnSlogdet](docs/aclnnSlogdet.md) 两段式接口调用 Slogdet 算子（真机 NPU 运行 + CPU LU golden 比对，含奇异 -inf/0 用例）。 |
| 图模式 (GE IR) | [examples/test_geir_slogdet.cpp](examples/test_geir_slogdet.cpp) | 通过 GE IR 构图调用 `Slogdet` 算子（单输入 `self`，双输出 `signOut`/`logOut`）。**仅保证编译/构图通过**：运行时与 CANN 内置 `Slogdet`（`linalg_ops.h`，输入名 `x`/输出 `sign`,`y`）IR 定义冲突，experimental 自定义注册（输入名 `self`）无法 RunGraph，如实说明见示例注释。 |

示例的编译与运行：

```bash
# 前置：source CANN set_env.sh；并设置自定义算子包 vendor 目录
export SLOGDET_CUSTOM_OPP=/path/to/vendors/custom_math    # 含 op_api/{include,lib}、op_proto/inc、op_impl

cd examples
bash run.sh --eager      # aclnn 调用示例（默认，真机 NPU 运行）
bash run.sh --graph      # 图模式 (GE IR) 调用示例（编译 + 尝试 RunGraph）
bash run.sh --clean      # 清理构建目录
```

> 链接说明：aclnn 示例用 `-Wl,--no-as-needed ${CUSTOM_OP_LIBRARY}` 将自定义 `libcust_opapi.so` 置于链接列表最前，保证运行时命中本实现的自定义 AscendC kernel（`Slogdet`），而非上游 `libopapi.so` 的 `math/slogdet` 转发壳（`LogMatrixDeterminant`）。
