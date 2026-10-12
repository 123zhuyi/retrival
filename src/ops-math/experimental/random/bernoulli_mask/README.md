# BernoulliMask

## 概述

本目录提供面向 Atlas A2/A3 训练系列产品的 `aclnnBernoulli` 和
`aclnnInplaceBernoulli` 低内存实现。算子保留 `DSAGenBitMask` 的
`seed/offset` 随机序列生成逻辑，由 Ascend C `BernoulliMask` Kernel 将压缩
bit mask 直接展开为目标数据类型的 `0` 或 `1`。

原实现中的全尺寸 `Fill`、`DropoutDoMask` 和部分 `Cast` 中间张量不再参与一般
概率路径。连续输出在满足存储条件时复用输出空间保存压缩 mask；小张量或非连续
输出使用独立 mask/连续结果缓冲区，以保持边界与视图语义。`prob=0` 和 `prob=1`
分别沿用 `ZerosLike` 和 `OnesLike` 快速路径。

## 产品支持

| 产品 | 构建目标 | 支持状态 |
| :--- | :---: | :---: |
| Atlas A2 训练系列产品 | `ascend910b` | 支持 |
| Atlas A3 训练系列产品 | `ascend910_93` | 支持 |

## 功能与约束

- `self/out` 支持 FP16、FP32、FP64、BF16、UINT8、INT8、INT16、INT32、
  INT64 和 BOOL；
- `prob` 支持 FP16、FP32、FP64 和 BF16，必须为有限值且满足
  `0 <= prob <= 1`；
- 支持 0～8 维、标量、空 Tensor、连续 Tensor 和非连续 Tensor；
- `self` 与 `out` 的 shape、dtype 必须一致；
- `offset` 必须满足 `offset % 4 == 0`；
- 相同的 `seed/offset` 生成可复现的随机序列。

## 接口

```cpp
aclnnStatus aclnnBernoulliGetWorkspaceSize(
    const aclTensor* self,
    const aclScalar* prob,
    int64_t seed,
    int64_t offset,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnBernoulli(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

`aclnnInplaceBernoulliGetWorkspaceSize` 和 `aclnnInplaceBernoulli` 使用相同的
参数语义，输出写回 `selfRef`。

## 实现说明

一般概率路径的数据流如下：

```text
self / prob / seed / offset
  -> DSAGenBitMask
  -> BernoulliMask（packed bit -> 目标 dtype 的 0/1）
  -> out
```

对于容量和布局满足条件的连续输出，DSA packed mask 与输出复用同一块设备
内存。Kernel 从高地址到低地址分波展开，每一波完成后同步，避免输出覆盖尚未
读取的 mask。其他布局使用独立缓冲区；非连续输出通过 `ViewCopy` 写回目标
view。FP64 非连续写回使用 INT64 原始位模式 view，避免数值转换。

FP16/FP32 使用向量 `Select` 生成结果；整数类型和 BF16 通过向量类型转换输出；
FP64 将 double 的两个 32 位字向量化构造后通过 `Gather` 交织。

## 目录结构

```text
bernoulli_mask/
├── op_api/       ACLNN 接口、路径选择与 BernoulliMask L0 launcher
├── op_host/      算子定义、InferShape 与 Tiling
├── op_kernel/    Ascend C Kernel、TilingData 与 TilingKey
├── examples/     ACLNN 调用样例
└── tests/
    ├── assets/   TTK golden
    ├── st/       ACLNN 系统测试
    ├── ttk/      Kernel 通用及存储复用用例
    └── ut/       Host UT
```

## 构建

在 `ops-math` 仓库根目录执行：

```bash
# Atlas A2
bash build.sh --pkg --experimental --soc=ascend910b \
  --ops=bernoulli_mask --build-type=Release

# Atlas A3
bash build.sh --pkg --experimental --soc=ascend910_93 \
  --ops=bernoulli_mask --build-type=Release
```

运行包生成在 `build_out/`。安装后按照安装器提示加载 `custom_math` 的
`op_api/lib` 环境。

## 调用样例

构建并安装当前源码生成的自定义算子包后，在仓库根目录执行：

```bash
bash experimental/random/bernoulli_mask/examples/run.sh
```

样例使用标准两段式 ACLNN 接口执行 FP32 用例，并检查输出值域、固定随机状态
的可复现性和样本均值。其他安装路径及仅编译方式见
[`examples/README.md`](examples/README.md)。

## 测试

### Host UT

```bash
bash build.sh -u --ophost --experimental --soc=ascend910b \
  --ops=bernoulli_mask
```

### ACLNN ST

从当前 checkout 构建并安装自定义算子包后执行：

```bash
bash experimental/random/bernoulli_mask/tests/st/run.sh
```

ST 覆盖全部输出 dtype、rank 0～8、空 Tensor、连续与非连续 view、
out-of-place/in-place、概率和 offset 边界、随机状态重现性，以及
mask/output 存储复用边界。环境变量和加载方式见
[`tests/st/README.md`](tests/st/README.md)。

### Kernel TTK

`tests/ttk/bernoulli_mask.csv` 包含 26 个通用 Kernel 用例，覆盖全部输出
dtype、packed-bit 顺序、mask/tile 边界和多核大 shape；
`tests/ttk/bernoulli_mask_alias.csv` 包含 8 个 fallback/alias 配对用例，覆盖
15、16、257 元素和百万元素多核 `SyncAll` 路径。

安装当前 checkout 构建的 Release 包后，从 TTK 3.0 根目录执行：

```bash
OPS_MATH_ROOT=/path/to/ops-math

python -m ttk kernel \
  -i "${OPS_MATH_ROOT}/experimental/random/bernoulli_mask/tests/ttk/bernoulli_mask.csv" \
  -d false -b release \
  --plugin "${OPS_MATH_ROOT}/experimental/random/bernoulli_mask/tests/assets/bernoulli_mask.py" \
  --compare binary --seed 20260725 \
  --device-whitelist=0 --pc=1 --proc-timeout=300 --proc-no-reuse \
  -o bernoulli_mask_ttk_result.csv

python -m ttk kernel \
  -i "${OPS_MATH_ROOT}/experimental/random/bernoulli_mask/tests/ttk/bernoulli_mask_alias.csv" \
  -d false -b release \
  --plugin "${OPS_MATH_ROOT}/experimental/random/bernoulli_mask/tests/assets/bernoulli_mask.py" \
  --compare binary --seed 20260725 \
  --device-whitelist=0 --pc=1 --proc-timeout=300 --proc-no-reuse \
  -o bernoulli_mask_alias_ttk_result.csv
```

BF16 golden 依赖 `ml-dtypes==0.5.4`。如果设置了
`ASCEND_RT_VISIBLE_DEVICES`，`--device-whitelist` 使用映射后的逻辑 device；
否则填写物理 device。

## 测试覆盖

随算子提交的测试覆盖以下内容：

- 10 种输出 dtype 和 4 种 `prob` dtype；
- rank 0～8、空 Tensor，以及 127/128/129、255/256/257 mask 边界；
- 连续、转置/切片 view、非零 storage offset 和 in-place；
- `prob=0/1`、一般概率、合法/非法 offset；
- seed/offset 可复现性和大样本统计分布；
- A2/A3 Kernel 通用路径与 mask/output 存储复用路径。
