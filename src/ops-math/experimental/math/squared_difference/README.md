# SquaredDifference

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品（Ascend 910B） | √ |

本算子当前在 `op_host/squared_difference_def.cpp` 中声明为 `ascend910b`，其它产品暂不在本算子交付范围内。

## 功能说明

SquaredDifference 对两个输入执行逐元素平方差：

```text
y = (x1 - x2) * (x1 - x2)
```

输入支持 NumPy broadcast 语义。输入从尾维对齐；同一维的长度必须相等，或至少有一个输入的长度为 1。输出 shape 是两个输入 shape 的逐维最大值。例如：

```text
x1: [2, 1, 3]
x2: [1, 4, 3]
y : [2, 4, 3]
```

## 参数说明

| 参数 | 输入/输出 | 描述 | 数据类型 | 格式 | rank |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `x1` | 输入 | 平方差的第一个操作数 | FLOAT、FLOAT16、BFLOAT16、INT32、INT64 | ND | 0-16（合轴后最多 8） |
| `x2` | 输入 | 平方差的第二个操作数 | FLOAT、FLOAT16、BFLOAT16、INT32、INT64 | ND | 0-16（合轴后最多 8） |
| `y` | 输出 | 平方差结果，shape 为 broadcast 后的输出 shape | FLOAT、FLOAT16、BFLOAT16、INT32、INT64 | ND | 与输出 shape 一致 |

输入和输出 dtype 必须一致。输入 shape 不可 broadcast、dtype 不支持或合轴后的维度超过 8 时，shape 推导或 tiling 返回失败。空 Tensor 的输出长度为 0，workspace 大小为 0。

## 实现结构

```text
experimental/math/squared_difference/
├── CMakeLists.txt                         # 接入 ops-math 统一构建
├── op_host/
│   ├── squared_difference_def.cpp         # 算子 schema 和 910B 注册
│   ├── squared_difference_infershape.cpp  # broadcast 输出 shape 推导
│   └── squared_difference_tiling.cpp      # 运行时 tiling 和 10 个 tiling key
├── op_kernel/
│   ├── squared_difference.cpp              # kernel 入口，按 key 选择 dtype
│   ├── squared_difference.h                # OneDim/BRC kernel 实现
│   ├── squared_difference_tiling_data.h   # host/kernel 共享 tiling 数据
│   └── squared_difference_tiling_key.h    # 5 dtype x 2 mode
├── examples/
│   └── test_aclnn_squared_difference.cpp  # ACLNN 两段式调用示例
├── tests/ut/
│   ├── op_host/test_squared_difference_tiling.cpp
│   └── op_kernel/test_squared_difference.cpp
└── docs/aclnnSquaredDifference.md         # ACLNN 接口说明
```

Tiling key 编码如下：

| key | dtype | 路径 |
| :---: | :---: | :--- |
| 0/1 | FP32 | OneDim / BRC |
| 2/3 | FP16 | OneDim / BRC |
| 4/5 | BF16 | OneDim / BRC |
| 6/7 | INT32 | OneDim / BRC |
| 8/9 | INT64 | OneDim / BRC |

`OneDim` 用于合轴后单维计算以及标量广播；`BRC` 用于多维广播。非 INT64 的单轴广播优先使用 BRCFast 路径；普通多维广播使用 BRC 路径。INT64 使用标量计算回退，并在需要时启用 N 方向切分或广播轴优化。所有路径均不申请用户 workspace。

## 构建

先加载 CANN 环境，再从仓库根目录使用统一 `build.sh`。只执行普通 host 构建不会生成 kernel 二进制；需要 kernel 产物时显式传入 `--opkernel` 和 `--soc`：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh

# 构建实验性 host 库（仓库级目标）
bash build.sh --experimental --ophost

# 构建 910B kernel 二进制
bash build.sh --experimental --opkernel --ops=squared_difference --soc=ascend910b

# 构建包含 host、kernel 和 ACLNN 头文件的完整 run 包
bash build.sh --pkg --experimental --ops=squared_difference --soc=ascend910b
```

启用仓库统一 UT：

```bash
bash build.sh -u --experimental --ophost --ops=squared_difference --soc=ascend910b
bash build.sh -u --experimental --opkernel --ops=squared_difference --soc=ascend910b
```

`--ophost` 与 `--opkernel` 的 UT 目标分别是 `math_op_host_ut` 和 `math_op_kernel_ut`。kernel UT 使用 tikicpulib；若只做编译检查，可附加 `--noexec`。

### 测试数据脚本

`tests/ut/op_kernel/squared_difference_data/gen_data.py` 和 `compare_data.py` 用于离线设备数据验证：前者生成 BF16 输入和 golden 文件，后者对设备导出的 `*output*.bin` 与 golden 文件进行精度比对。它们不参与 gtest 的编译或运行，gtest 使用内存中构造的确定性数据，避免 UT 依赖 Python、NumPy 或外部文件。

需要进行离线数据验证时，可在数据目录执行：

```bash
cd experimental/math/squared_difference/tests/ut/op_kernel/squared_difference_data
python3 gen_data.py
# 将设备输出保存为 bfloat16_output*.bin 后执行
python3 compare_data.py bfloat16
```

## 调用示例

ACLNN 示例位于 [`examples/test_aclnn_squared_difference.cpp`](./examples/test_aclnn_squared_difference.cpp)，接口签名和两段式调用顺序见 [`docs/aclnnSquaredDifference.md`](./docs/aclnnSquaredDifference.md)。示例使用 BF16、shape `[1, 19]`，输入均为 1，输出均为 0。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| lorcas | 个人开发者 | SquaredDifference | 2026/08/25 | SquaredDifference 算子适配开源仓，补充统一构建、UT、示例和文档 |
| 圣皇心痛天使 | 个人开发者 | SquaredDifference | 2026/08/25 | SquaredDifference 算子适配开源仓，补充统一构建、UT、示例和文档 |
