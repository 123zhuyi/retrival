# Bernoulli

`experimental/random/bernoulli` 提供 Atlas A2/A3 上 `aclnnBernoulli` 和
`aclnnInplaceBernoulli` 标量概率路径的低内存 Ascend C 实现。Tensor 概率
接口以及非 DAV_2201 平台继续使用既有兼容路径。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| m0_69357246 | 个人开发者 | Bernoulli | 2026/07 | A2/A3 标量概率路径的内存与性能优化 |

## 产品支持

| 产品 | 构建参数 | 架构 | 支持状态 |
| --- | --- | --- | :---: |
| Atlas A2 训练/推理系列产品 | `ascend910b` | DAV_2201 | 支持 |
| Atlas A3 训练/推理系列产品 | `ascend910_93` | DAV_2201 | 支持 |

两个产品使用同一份 Ascend C Kernel 和 `arch22` tiling。核数与 UB 大小由
`PlatformAscendC` 在运行时获取，不依赖固定型号参数。
本任务面向 CANN 8.5.0 及以上版本。

## 功能说明

对于标量概率 `p`，输出元素满足：

```text
y[i] in {0, 1}
P(y[i] = 1) = p
P(y[i] = 0) = 1 - p
```

`self` 的数值不参与计算，只提供输出的 shape、dtype、format 和 view 信息。
`seed` 与 `offset` 决定随机序列；实现继续使用 `DSAGenBitMask`，不改变既有
随机数生成协议和 packed mask bit 顺序。

## 接口与支持范围

| 接口 | 概率类型 | 本目录中的处理方式 |
| --- | --- | --- |
| `aclnnBernoulli` | `aclScalar*` | DAV_2201 使用低内存 Bernoulli Kernel |
| `aclnnInplaceBernoulli` | `aclScalar*` | 复用标量概率实现并写回 `selfRef` |
| `aclnnBernoulliTensor` | `aclTensor*` | 保持既有 `StatelessBernoulli` 路径 |
| `aclnnInplaceBernoulliTensor` | `aclTensor*` | 保持既有 Tensor 概率路径 |

标量概率接口的参数范围如下：

| 参数 | 类型与格式 | 约束 |
| --- | --- | --- |
| `self` / `out` | FP16、FP32、FP64、BF16、UINT8、INT8、INT16、INT32、INT64、BOOL；ND | shape 和 dtype 一致，rank 0-8，支持空 Tensor 和非连续 Tensor |
| `prob` | FP16、FP32、FP64、BF16 标量 | 有限值且 `0 <= prob <= 1` |
| `seed` | `int64_t` | 用于确定随机序列 |
| `offset` | `int64_t` | 必须满足 `offset % 4 == 0` |

完整签名、错误码和调用约束见：

- [`aclnnBernoulli&aclnnInplaceBernoulli.md`](docs/aclnnBernoulli&aclnnInplaceBernoulli.md)
- [`aclnnBernoulliTensor&aclnnInplaceBernoulliTensor.md`](docs/aclnnBernoulliTensor&aclnnInplaceBernoulliTensor.md)

## 实现与内存优化

原标量概率链路会物化 `Fill`、`DropoutDoMask` 和部分 `Cast` 全尺寸中间
Tensor。本实现保留 DSA 随机 mask 生成，将 mask 展开融合为单个 Bernoulli
Kernel：

```text
self / prob / seed / offset
  -> DSAGenBitMask
  -> Bernoulli (packed mask -> target dtype 0/1)
  -> out
```

对于满足直接写入条件的连续输出，`DSAGenBitMask` 将 packed mask 写入输出
storage 的低地址区域，Bernoulli Kernel 按从高地址到低地址的 stage 原地展开。
每个 stage 在写回前完成 mask 搬入，多核路径使用 `SyncAll`，避免结果覆盖尚未
读取的 mask。该路径不再申请独立 packed mask、Fill 或 DropoutDoMask 全尺寸
Tensor。

以下场景使用兼容处理：

- 非连续输出通过 executor 临时结果和 `ViewCopy` 保持 view 语义；
- 极小 Tensor 若输出容量不足以容纳 DSA 的最小 128-bit mask block，扩大
  executor-owned storage，但公共 view shape 不变；
- `prob=0` 和 `prob=1` 根据 dtype 选择常量 AIV Kernel 或既有
  `ZerosLike`/`OnesLike` 路径；
- Tensor 概率接口和非 DAV_2201 平台不进入本次融合路径。

私有 Bernoulli Kernel 使用 512 B 同步 workspace；tiling 根据实际 Vector Core
数、UB 容量、dtype 和 shape 联合选择 block、stage 与 tile。

## 目录结构

```text
bernoulli/
|-- CMakeLists.txt
|-- README.md
|-- docs/                 ACLNN 接口文档
|-- examples/             out-of-place 与 in-place 调用样例
|-- op_api/               公共 ACLNN 实现和私有 L0 launcher
|-- op_graph/             私有 Bernoulli 原型
|-- op_host/              OpDef、tiling 与二进制配置
|-- op_kernel/            Ascend C Kernel 与 TilingData
`-- tests/
    |-- st/               ACLNN JSON/Python 系统测试
    `-- ut/op_api/        参数、格式、边界和路径选择 UT
```

## 构建与安装

在 `ops-math` 仓库根目录执行。`ASCEND_HOME_PATH` 应指向 CANN Toolkit
安装根目录。

```bash
source "${ASCEND_HOME_PATH}/set_env.sh"

# Atlas A2
bash build.sh --pkg --experimental --vendor_name=experimental \
  --soc=ascend910b --ops=bernoulli -j16

# Atlas A3
bash build.sh --pkg --experimental --vendor_name=experimental \
  --soc=ascend910_93 --ops=bernoulli -j16
```

安装包生成在 `build_out/`。安装到自定义目录的示例：

```bash
./build_out/cann-ops-math-experimental_linux-<arch>.run \
  --install-path=<install-path> --force
source <install-path>/vendors/experimental_math/bin/set_env.bash
```

## 调用样例

| 调用方式 | 样例 |
| --- | --- |
| out-of-place `aclnnBernoulli` | [`test_aclnn_bernoulli.cpp`](examples/test_aclnn_bernoulli.cpp) |
| in-place `aclnnInplaceBernoulli` | [`test_aclnn_inplace_bernoulli.cpp`](examples/test_aclnn_inplace_bernoulli.cpp) |

构建和运行说明见 [`examples/README.md`](examples/README.md)。

## 测试

### Op API UT

UT 公共框架需要 Python 开发头。如果 CMake 未自动找到 `Python.h`，先执行：

```bash
export CPLUS_INCLUDE_PATH="$(python3 -c \
  'import sysconfig; print(sysconfig.get_paths()["include"])'):${CPLUS_INCLUDE_PATH:-}"
```

```bash
bash build.sh -u --opapi --experimental --soc=ascend910b --ops=bernoulli
bash build.sh -u --opapi --experimental --soc=ascend910_93 --ops=bernoulli
```

27 个 UT 覆盖输出 dtype、ND 及现有兼容 format 路径、空 Tensor、非连续 Tensor、单元素
storage alias、in-place、Tensor 概率兼容路径，以及非法 rank、offset、NaN
概率和私有 format。

### ACLNN ST

[`tests/st/aclnnBernoulli`](tests/st/aclnnBernoulli) 包含 ATK JSON 用例和
PyTorch `torch.bernoulli` golden executor。当前用例集包含 200 个浮点用例，
覆盖 FP16/FP32/FP64/BF16、rank 1-7、概率端点和一般概率，以及多个合法
offset。执行方式和判定说明见 [`tests/st/README.md`](tests/st/README.md)。

### 覆盖对应关系

| 检查项 | 随代码提交的覆盖 |
| --- | --- |
| 接口签名、参数、约束、错误码 | `docs/` 下两份 ACLNN 接口文档 |
| 标量、in-place、Tensor 概率及异常分支 | 27 个 Op API UT |
| 浮点 dtype、shape、概率与 offset 组合 | 200 个 ACLNN ATK ST |
| A2/A3 构建与打包 | `ascend910b`、`ascend910_93` 两套配置 |

竞赛任务书给出的 50 个专项用例用于最终内存、精度与性能验收，不替代本目录
随代码提交的 UT/ST。
