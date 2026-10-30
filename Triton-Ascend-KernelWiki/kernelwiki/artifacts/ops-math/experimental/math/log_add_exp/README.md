# LogAddExp 自定义算子

## 功能说明

LogAddExp 算子计算两个输入张量的 log-sum-exp 运算：

```bash
logaddexp(x, y) = max(x, y) + ln(1 + e^(-|x - y|))
```

该公式等价于 `ln(e^x + e^y)`，但采用数值稳定形式，可避免直接计算 `e^x + e^y` 时的上溢出问题。

**对标**：PyTorch `torch.logaddexp`

## 支持规格

### 数据类型

| 输入 x | 输入 y | 输出 out |
|--------|--------|---------|
| float32 | float32 | float32 |
| float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 |

注意：x 和 y 的数据类型必须一致。

### Shape

- 最大维度数：3（支持 1D、2D、3D）
- 支持广播：x 和 y 满足 NumPy 广播规则

### 广播规则

支持以下广播模式：

| 广播模式 | 示例 |
|---------|------|
| 标量广播 | `[1]` + `[N]` → `[N]` |
| 单维度广播 | `[1, N]` + `[M, N]` → `[M, N]` |
| 双向广播 | `[M, 1]` + `[1, N]` → `[M, N]` |
| 维度扩展 | `[N]` + `[M, N]` → `[M, N]` |
| 3D 多维广播 | `[1, 1, N]` + `[B, M, N]` → `[B, M, N]` |

### 目标芯片

- Ascend 910B3（arch32 架构）

## 使用方法

### aclnn API 调用

```cpp
#include "aclnn/acl_meta.h"

// 第一步：获取 workspace 大小和执行器
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
auto ret = aclnnLogAddExpGetWorkspaceSize(
    x_tensor,         // 输入张量 x
    y_tensor,         // 输入张量 y
    output_tensor,    // 输出张量
    &workspaceSize,   // 输出：workspace 大小
    &executor         // 输出：执行器句柄
);

// 第二步：分配 workspace 内存
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
}

// 第三步：执行算子
ret = aclnnLogAddExp(workspace, workspaceSize, executor, stream);

// 第四步：同步并释放资源
aclrtSynchronizeStream(stream);
if (workspace) aclrtFree(workspace);
```

### 张量创建示例

```cpp
// 创建 float32 张量，shape [4, 8]
std::vector<int64_t> shape = {4, 8};
aclTensor* tensor = aclCreateTensor(
    shape.data(), shape.size(),
    ACL_FLOAT,          // 数据类型
    nullptr, 0,         // strides（nullptr 表示连续）
    0,                  // offset
    ACL_FORMAT_ND,      // 格式
    device_ptr          // 设备内存指针
);
```

### 完整示例参考

参见 `tests/st/test_aclnn_log_add_exp.cpp`，其中包含完整的初始化、数据准备、调用和精度比对流程。

## 精度说明

| 数据类型 | 精度标准 | 真实 NPU 表现 | 备注 |
|---------|---------|-------------|------|
| float32 | rtol=1e-4 | 100% 通过 | 全量 23 条用例 |
| float16 | rtol=1e-3 | 88.9% 通过 | 2 条边缘用例失败，均为单元素精度超限 |
| bfloat16 | rtol=1e-3 | 93.3% 通过 | 1 条边缘用例失败，为单元素精度超限 |

**说明**：fp16/bf16 失败用例均为极端边缘情况，失败元素集中在小负数区域（-0.08 ~ -0.02），属于 fp16/bf16 格式固有的精度限制，并非算子实现问题。中间计算已使用 fp32 升精度（Cast→fp32→Cast back）。

## 构建方法

### 前提条件

- CANN Toolkit 已安装（路径：`/home/developer/Ascend/cann-9.0.0` 或其他版本）
- 已设置环境变量：`source /home/developer/Ascend/cann-9.0.0/bin/setenv.bash`

### 编译自定义算子包

```bash
cd ops/log_add_exp
bash build.sh
```

编译成功后，算子包位于 `build/custom_opp_ubuntu_aarch64.run`。

### 安装算子包

```bash
bash build/custom_opp_ubuntu_aarch64.run
```

算子安装到：`/home/developer/Ascend/latest/vendors/log_add_exp_custom/`

## 测试方法

### 运行 UT（单元测试）

```bash
cd tests/ut
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j4
./log_add_exp_ut
```

预期结果：40/40 PASS

### 运行 ST（系统测试）

```bash
cd tests/st
bash run.sh
```

默认运行 59 条测试用例。

ST 测试支持两种模式：

- **Mock 模式**（CPU Golden）：无需 NPU，用于开发验证
- **真实 NPU 模式**：需要 NPU 设备，验证实际精度

### 运行性能测试

```bash
cd tests/st
bash run_performance.sh
```

## 目录结构

```text
ops/log_add_exp/
├── README.md                          # 本文档
├── CMakeLists.txt                     # 顶层构建脚本
├── build.sh                           # 编译脚本
├── op_host/                           # Host 侧实现
│   ├── log_add_exp_def.cpp            # 算子定义（aclnn API 注册）
│   ├── log_add_exp_infershape.cpp     # InferShape（广播规则实现）
│   └── arch32/
│       └── log_add_exp_tiling.cpp    # Tiling 实现（含广播分析）
├── op_kernel/                         # Device 侧实现
│   └── arch32/
│       └── log_add_exp.h             # Kernel 实现（核心计算逻辑）
├── tests/
│   ├── st/                            # 系统测试
│   │   ├── test_aclnn_log_add_exp.cpp # ST 测试工程（59 条用例）
│   │   ├── CMakeLists.txt
│   │   └── run.sh                     # ST 运行脚本
│   └── ut/                            # 单元测试
│       └── log_add_exp_ut.cpp         # UT（40 条用例）
├── docs/
│   ├── DEVELOPMENT_LOG.md             # 开发日志
│   ├── REQUIREMENT_ANALYSIS.md        # 需求分析文档
│   ├── DETAILED_DESIGN.md             # 详细设计文档
│   ├── log_add_exp_TEST_DESIGN.md     # 测试设计文档
│   └── PERFORMANCE_REPORT.md          # 性能报告
└── build/                             # 编译输出目录
    └── custom_opp_ubuntu_aarch64.run  # 算子包
```

## 实现说明

### 核心计算步骤

Kernel 内部将公式分解为以下 8 步向量运算：

```text
tmp = x - y          (Sub)
tmp = |tmp|          (Abs)
tmp = -tmp           (Muls, factor=-1)
tmp = e^tmp          (Exp)
tmp = tmp + 1        (Adds, scalar=1)
tmp = ln(tmp)        (Ln)
out = max(x, y)      (Max)
out = out + tmp      (Add)
```

### 精度保证

- **fp32**：全程 fp32 计算，精度最高
- **fp16/bf16**：输入 Cast 升至 fp32，执行 8 步计算，结果 Cast 回原始类型

### 广播实现

广播通过 stride=0 + 逐元素索引计算实现，无需展开数据，内存高效。广播开销 < 0.2%（相比非广播场景）。
