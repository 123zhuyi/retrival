# ZerosLike

> experimental 自包含实现（仅 ascend910b / DAV_2201 / Atlas A2）。本目录为内建 `conversion/zeros_like`（当前仅注册 ascend950 / arch35 DAG kernel）补齐 **Ascend910B 原生 AscendC AI Core kernel** 的开源参考实现。

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

- 本 experimental 实现仅在 `op_host/zeros_like_def.cpp` 中 `AddConfig("ascend910b", ...)`，目标芯片为 Ascend910B（SocVersion `ASCEND910B`，NpuArch `DAV_2201`，即 arch32）。
- 950（DAV_3510 / arch35）由内建 `conversion/zeros_like` 覆盖，不在本目录范围内（差异见下文「与内建 ascend950 版差异」）。

## 功能说明

- 算子功能: 创建一个与给定张量形状相同、且所有元素都为 0 的新张量（等价于 `torch.zeros_like` / `tf.zeros_like`）。
- 计算公式：

  $$
    y_{i} = 0,\quad \text{shape}(y) = \text{shape}(x),\quad \text{dtype}(y) = \text{dtype}(x)
  $$

- 计算范式: **Elementwise 一元（退化形态）** —— 仅按元素写常量 0，**纯写出、不读取 `x` 的数据值**。`x` 仅用于推导输出的 shape / dtype（host tiling 据此切分），kernel 入参含 `GM_ADDR x` 但不解引用。退化数据流为 `UB(Duplicate 0) → GM(DataCopy/DataCopyPad)`，无 CopyIn。
- 示例：

  ```text
  输入x：
  tensor([[1, 2],
          [3, 4]])
  输出y：
  tensor([[0, 0],
          [0, 0]])
  ```

## 算子原型设计

| 参数名 | 类别     | 描述                                         | 数据类型                             | 数据格式 |
| :----- | :------- | :------------------------------------------- | :----------------------------------- | :------- |
| x      | 输入张量 | 输入张量，仅用于推导输出 shape/dtype。       | FLOAT16、BF16、FLOAT32、INT32、INT64、INT8、UINT8、BOOL | ND       |
| y      | 输出张量 | 与输入张量形状、dtype 相同的新张量，所有元素都为 0。 | 与x一致                              | ND       |

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持 FLOAT16、BF16、FLOAT32、INT32、INT64、INT8、UINT8、BOOL，共 **8 种**。

### 支持的数据类型（8 种）

| # | dtype    | ge::DataType   | 字节宽度桶（TilingKey） |
| - | -------- | -------------- | :---------------------: |
| 1 | FLOAT16  | DT_FLOAT16     | 2B |
| 2 | BF16     | DT_BF16        | 2B |
| 3 | FLOAT32  | DT_FLOAT       | 4B |
| 4 | INT32    | DT_INT32       | 4B |
| 5 | INT64    | DT_INT64       | 8B |
| 6 | INT8     | DT_INT8        | 1B |
| 7 | UINT8    | DT_UINT8       | 1B |
| 8 | BOOL     | DT_BOOL        | 1B |

> **全 0 二进制等价原则**：所有 dtype 的「全 0」在 bit 层面都是全 0 字节，故 kernel 按**字节宽度桶（1/2/4/8B）** 写 0 即可，无需逐 dtype 计算路径。dtype 仅决定字节宽度（→ TilingKey）。
>
> aclnn 接口层 L2 dtype 校验列表为 13 种超集（额外含 DOUBLE/INT16/UINT16/COMPLEX64/COMPLEX128，与 `math/zero_op` 真值源逐字节一致），但仅上述 8 种命中本仓 910b AI Core kernel，其余走 AICPU 兜底，不在本实现验收范围。

## aclnn 接口

本算子通过 `aclnnInplaceZero` 两段式接口调用（in-place：`selfRef` 既是输入又是输出）。接口详情见 [docs/aclnnInplaceZero.md](docs/aclnnInplaceZero.md)。

```Cpp
// 第一段：计算 workspace 大小 + 构造执行器
aclnnStatus aclnnInplaceZeroGetWorkspaceSize(
    aclTensor      *selfRef,
    uint64_t       *workspaceSize,
    aclOpExecutor **executor);

// 第二段：执行计算
aclnnStatus aclnnInplaceZero(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## 约束说明

- selfRef 维度（rank）范围 0~8，支持空 tensor（0 元素，此时 workspace 为 0），支持非连续 Tensor（内部 Contiguous）。
- 确定性计算：纯写常量 0，无 Reduce / 累加，输出与核数 / 切分无关 → 天然 bitwise 确定可复现。
- 精度判据：全 0 bitwise（rtol = atol = 0，metric = bitwise_equal）。
- **验证路径**：以 **aclnn / eager** 为主验证路径。图模式（graph）非主验证路径；本算子 OpType `ZerosLike` 的图原型签名（`INPUT(x) → OUTPUT(y)`，同形同 dtype）与 CANN 内置 `REG_OP(ZerosLike)` 一致，不存在图模式 IR 签名冲突；如需图模式，建议单列为独立验收项。

## 打包命令

构建 experimental 树的自定义运行包（含本算子的 op_host / op_kernel / op_api）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=zeros_like
```

产物：`build_out/cann-ops-math-custom_linux-<arch>.run`。安装：

```bash
bash build_out/cann-ops-math-custom_linux-<arch>.run --quiet --install-path=<install_dir>
# 自定义算子安装到 <install_dir>/vendors/custom_math/
```

## 调用说明

| 调用方式  | 样例代码                                                | 说明                                           |
| :-------- | :------------------------------------------------------ | :--------------------------------------------- |
| aclnn接口 | [test_aclnn_zeros_like](examples/test_aclnn_zeros_like.cpp) | 通过 aclnnInplaceZero 两段式接口调用 ZerosLike 算子（参考示例代码）。 |

## 与内建 ascend950（arch35 DAG）版差异

本 experimental 实现与内建 `conversion/zeros_like`（950 版）共享同一 aclnn 接口语义（`aclnnInplaceZero`）与同一 L2 dtype 校验超集（来自 `math/zero_op`）；差异仅在目标芯片、kernel 形态、AI Core 实际编译的 dtype 集合。

| 维度 | 内建 `conversion/zeros_like`（950 / DAV_3510 / arch35） | 本 experimental（910b / DAV_2201 / arch32） |
|------|----------------------------------------------------------|---------------------------------------------|
| 目标芯片 | Ascend950DT/PR | Ascend910B（A2） |
| OpDef AddConfig | `AddConfig("ascend950", ...)` | `AddConfig("ascend910b", config910b)` |
| kernel 形态 | arch35 **DAG kernel**（`op_kernel/arch35/zeros_like_dag.h`，入口 `zeros_like_apt.cpp`，`opFile.value="zeros_like_apt"`） | arch32 **原生 AscendC kernel**（`op_kernel/zeros_like.cpp`，显式 `opFile.value="zeros_like"`） |
| OpDef dtype | 13 种（含 fp8_e5m2/fp8_e4m3fn/hifloat8/fp4_e1m2/fp4_e2m1 窄浮点） | **8 种**（不扩窄浮点；窄浮点不在 910b AI Core 范围） |
| 写 0 实现 | DAG 框架 | 单块零缓冲 `Duplicate(0)` 复用 + DataCopy/DataCopyPad |
| 目录布局 | 跨目录（kernel 在 conversion，L0 aclnn 在 `math/zero_op`） | experimental 自包含单目录 |
| op_graph | `math/zero_op/op_graph/zero_op_proto.h`（内建原型） | 无（承接内建原型；InferShape/InferDataType 在本目录 op_host） |

## 目录结构

```text
experimental/conversion/zeros_like/
├── op_host/                       # 算子信息库 + 推导 + tiling
│   ├── zeros_like_def.cpp           # OpDef: 8 dtype, AddConfig("ascend910b")
│   ├── zeros_like_infershape.cpp    # InferShape / InferDataType（直传）
│   ├── zeros_like_tiling.{cpp,h}    # 顶层 tiling（字节宽度桶切分）
├── op_kernel/                     # 设备侧 AscendC AI Core kernel（arch32 原生）
│   ├── zeros_like.cpp               # kernel 入口（opFile.value="zeros_like"）
│   ├── zeros_like_tiling_data.h
│   └── zeros_like_tiling_key.h      # 字节宽度 4 桶 ASCENDC_TPL_UINT_DECL
├── op_api/                        # aclnn C 接口封装
│   ├── aclnn_zero.{cpp,h}           # aclnnInplaceZero 两段式 + L2 dtype 校验
│   └── zero_op.{cpp,h}              # L0 内部接口 + 910b AI Core 分发
├── examples/                      # aclnn 调用示例
│   └── test_aclnn_zeros_like.cpp    # aclnnInplaceZero 两段式调用
├── docs/                          # aclnnInplaceZero.md（aclnn 接口参考）
├── tests/                         # ut（op_host/op_api/op_kernel）
├── README.md
└── CMakeLists.txt
```

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| - | 个人开发者 | ZerosLike | 2026/5/28 | ZerosLike 算子适配开源仓（experimental，ascend910b 原生 AscendC kernel） |

## 已知限制 / Known Limitations

以下为本实现的已知限制，均不影响 Atlas A2（Ascend910B）上的功能与精度。

### 限制 1 — Tiling `usableUb` 在极小 UB 下的无符号下溢（理论）

- **位置**：`op_host/zeros_like_tiling.cpp`（`ComputeZerosLikeTileBytes`）。
- **现象**：`usableUb = (uint64_t)ubSize - ZL_RESERVED_UB(8192) - sizeof(ZerosLikeTilingData)`，前置仅校验 `ubSize > 0`。当 `ubSize < 8232` 时 uint64 回绕为巨大值。
- **缓解**：下游 `tileBytes` 受 `ZL_TILE_BYTES_LIMIT`(64KB) 截断并被 `ZL_BLOCK_BYTES`(32) 下限钳住，不会真正分配巨型 UB；目标 DAV_2201 UB=192KB ≫ 8232，实际不触发，属理论健壮性问题。
- **加固建议**：加显式下界保护：

  ```cpp
  uint64_t reserved = ZL_RESERVED_UB + sizeof(ZerosLikeTilingData);
  uint64_t usableUb = (static_cast<uint64_t>(compileInfo->ubSize) > reserved)
                          ? (static_cast<uint64_t>(compileInfo->ubSize) - reserved) : 0;
  ```
