# BitwiseNot

`experimental/math/bitwise_not` 是 BitwiseNot 的原生 AscendC 实现，对输入张量逐元素求 BitwiseNot（`out = ~self`），OpType 为 `BitwiseNot`，对外提供两段式接口 `aclnnBitwiseNot`，目标芯片 **Ascend910B**。

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品 / Atlas 800I A2 推理产品（Ascend910B） | √ |

> 本算子为 `experimental/math` 投放区的 Ascend910B 原生 AscendC 贡献实现，当前仅支持 **Ascend910B**。

## 功能说明

- **算子功能**：对输入张量 `self` 逐元素求 BitwiseNot（`out = ~self`），按 dtype 分两条语义分支：

  | dtype 分支 | 语义 | 公式 | 示例 |
  | ---- | ---- | ---- | ---- |
  | 整型（INT8 / INT16 / INT32 / INT64 / UINT8） | **按位非（按位补码）** | `out = ~self`（有符号等价 `-self - 1`） | `~0=-1`、`~(-128)=127`（int8）、`~0=255`、`~255=0`（uint8） |
  | BOOL | **逻辑非**（结果规整为 0/1，非裸位翻转） | `out = (self == 0) ? 1 : 0` | `~true=false`、`~false=true`（0↔1） |

- **对标框架**：`torch.bitwise_not(input)` / `~input` / `numpy.invert`。
- **精度**：整数/逻辑算子，结果与参考实现（`numpy.invert`）**按位精确逐元素相等**（atol = 0，rtol = 0）。

- **原型信息**：

  | 项 | 值 |
  | ---- | ---- |
  | 算子类型（OpType） | `BitwiseNot` |
  | aclnn API 名称 | `aclnnBitwiseNot`（两段式入口） |
  | 核函数名（opFile.value） | `bitwise_not` |
  | 输入 | `x`：ND tensor，支持 0D~8D，dtype ∈ {int8, int16, int32, int64, uint8, bool} |
  | 输出 | `y`：ND tensor，dtype 与 shape 均与 `x` 相同 |

## 参数说明

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | 描述 |
| ---- | ---- | ---- | ---- | ---- |
| x (self) | 输入 | int8 / int16 / int32 / int64 / uint8 / bool | ND | 待按位/逻辑取反的输入张量 |
| y (out) | 输出 | int8 / int16 / int32 / int64 / uint8 / bool | ND | 取反结果；shape 与 dtype 须与 x 一致 |

约束：

- **shape**：`out.shape` 必须与 `self.shape` 完全相同（逐元素，无 broadcast）；维度数 ≤ 8。
- **数据类型**：`self.dtype` 必须与 `out.dtype` 一致（无类型提升）。UINT16/UINT32/UINT64 不在本算子（Ascend910B）支持范围内。
- **格式**：仅支持 ND 类公开格式，禁止私有格式。
- **空 Tensor**：支持空 Tensor（element 数为 0），直接返回成功且 out 为空。
- **确定性**：逐元素算子，天然确定性（同输入逐位可复现）。

## 约束说明

- **单 tensor 字节数上限**：当前实现的 GM 偏移与长度计算使用 `uint32`，因此单个 tensor 的元素总字节数须 ≤ `UINT32_MAX`（4,294,967,295 字节，约 4 GB）；超大 tensor 暂不支持。换算到元素数：

  | dtype（字节/元素） | 元素数上限 |
  | ---- | ---- |
  | int8 / uint8 / bool（1 字节） | ≤ ~42.9 亿 |
  | int16（2 字节） | ≤ ~21.4 亿 |
  | int32（4 字节） | ≤ ~10.7 亿 |
  | int64（8 字节） | ≤ ~5.37 亿 |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| aclnn 两段式（eager） | [examples/test_aclnn_bitwise_not.cpp](./examples/test_aclnn_bitwise_not.cpp) | 经 `aclnnBitwiseNotGetWorkspaceSize` / `aclnnBitwiseNot` 调用本算子（含 int32/int8/uint8/bool 四分支 CPU golden 自验 == `numpy.invert`） |
| 图模式（GE IR） | [examples/test_geir_bitwise_not.cpp](./examples/test_geir_bitwise_not.cpp) | 构造单算子图 `BitwiseNot`，经 GE Session AddGraph/RunGraph 调度 |

接口语义详见 [docs/aclnnBitwiseNot.md](./docs/aclnnBitwiseNot.md)。

### 编译部署

编译运行前，请参考[《CANN 软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境部署，并 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。

```bash
# 进入仓库根
cd ${git_clone_path}/ops-math

# 编译本算子自定义包（仅 bitwise_not）
bash build.sh --pkg --experimental --soc=ascend910b --ops=bitwise_not

# 安装自定义包并导出算子包环境
./build_out/cann-ops-math-custom_linux-<arch>.run
source <install-path>/vendors/custom_math/bin/set_env.bash
```

### 算子调用

可用配套脚本一键编译运行示例：

```bash
cd experimental/math/bitwise_not/examples
bash run.sh                  # 编译 eager + geir 两个示例；在真实 NPU 上运行 eager 示例（两段式 aclnnBitwiseNot）
bash run.sh --eager-only     # 仅 eager 示例（编译 + 运行）
bash run.sh --geir-build-only  # 仅编译 geir 示例（纯编译验收，不实跑图）
bash run.sh --run-geir       # 额外实跑 geir 示例（需完整 GE 图执行环境）
```

eager 示例在真实 NPU 上的预期输出（节选）：

```bash
[PASS] int32  ~self bitwise-exact
[PASS] int8   ~self bitwise-exact (boundary ~(-128)=127)
[PASS] uint8  ~self bitwise-exact (~0=255 ~255=0)
[PASS] bool   !self (0<->1) bitwise-exact
==== BitwiseNot eager example: ALL PASS ====
```

也可走仓库统一入口（在 ops-math 根目录）：

```bash
bash build.sh --run_example bitwise_not eager cust --experimental --vendor_name=custom  # 编译并运行 eager 示例
bash build.sh --run_example bitwise_not graph --experimental                            # 编译 graph 示例
```
