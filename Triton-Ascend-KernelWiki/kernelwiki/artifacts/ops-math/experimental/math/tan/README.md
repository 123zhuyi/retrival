# Tan 自定义算子

## 功能说明

Tan 算子计算输入张量的逐元素正切值：

```bash
tan(x) = sin(x) / cos(x)
```

**对标**：PyTorch `torch.tan`

## 支持规格

### 数据类型

| 输入 x | 输出 out |
|--------|---------|
| float32 | float32 |
| float16 | float16 |

注意：输出 out 的数据类型与输入 x 一致。

### Shape

- 支持任意维度（标量、1D ~ 8D 均已验证）
- 输出 shape 与输入 shape 相同
- 不支持广播（单输入算子）
- 支持空 tensor（元素数为 0）
- 支持动态 shape / 动态 rank

### 目标芯片

- Ascend 910B3（arch32 架构）

## 使用方法

### aclnn API 调用

```cpp
#include "acl/acl.h"
#include "aclnn_tan.h"

// 第一步：获取 workspace 大小和执行器
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
auto ret = aclnnTanGetWorkspaceSize(
    x_tensor,         // 输入张量 x
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
ret = aclnnTan(workspace, workspaceSize, executor, stream);

// 第四步：同步并释放资源
aclrtSynchronizeStream(stream);
if (workspace) aclrtFree(workspace);
```

### 张量创建示例

```cpp
// 创建 float32 张量，shape [2, 4]
std::vector<int64_t> shape = {2, 4};
std::vector<int64_t> strides = {4, 1};
aclTensor* tensor = aclCreateTensor(
    shape.data(), shape.size(),
    ACL_FLOAT,                 // 数据类型
    strides.data(), 0,         // strides + offset
    ACL_FORMAT_ND,             // 格式
    shape.data(), shape.size(),
    device_ptr                 // 设备内存指针
);
```

### 完整示例参考

- **aclnn 调用示例**：`examples/test_aclnn_tan.cpp`
- **GE IR 图模式示例**：`examples/test_geir_tan.cpp`

运行示例：

```bash
cd examples
bash run.sh              # 运行 aclnn 调用示例（默认）
bash run.sh --graph      # 运行图模式 (GE IR) 调用示例
```

## 精度说明

| 数据类型 | 精度标准 | 真实 NPU 表现 | 备注 |
|---------|---------|-------------|------|
| float32 | rtol=1e-4, atol=1e-6 | 100% 通过 | 全量 36 条用例 |
| float16 | rtol=1e-3, atol=1e-3 | 100% 通过 | 全量 20 条用例 |

**说明**：float16 路径内部先 Cast 升至 float32 进行 Sin/Cos/Div 计算，再 Cast 回 float16，保证精度。

## 构建方法

### 前提条件

- CANN Toolkit 已安装（路径：`/home/developer/Ascend/cann-9.0.0` 或其他版本）
- 已设置环境变量：`source /home/developer/Ascend/ascend-toolkit/set_env.sh`

### 编译自定义算子包

```bash
cd ops/tan
bash build.sh --soc=ascend910b --pkg
```

编译成功后，算子包位于 `build/custom_opp_ubuntu_aarch64.run`。

### 安装算子包

```bash
bash build/custom_opp_ubuntu_aarch64.run
```

算子安装到：`$ASCEND_HOME_PATH/opp/vendors/tan_custom/`

## 测试方法

### 运行 UT（单元测试）

```bash
cd tests/ut
bash run.sh
```

### 运行 ST（系统测试）

```bash
cd tests/st
bash run.sh
```

全量 56 条测试用例（36 条 float32 + 20 条 float16）。

ST 测试支持两种模式：

- **Mock 模式**（CPU Golden）：无需 NPU，用于开发验证
- **真实 NPU 模式**：需要 NPU 设备，验证实际精度

### 一键编译 + 测试

```bash
bash build.sh --soc=ascend910b --pkg -a    # 编译 + UT + ST
bash build.sh --soc=ascend910b --pkg -u    # 编译 + 仅 UT
bash build.sh --soc=ascend910b --pkg -s    # 编译 + 仅 ST
```

## 目录结构

```text
ops/tan/
├── README.md                              # 本文档
├── CMakeLists.txt                         # 顶层构建脚本
├── build.sh                               # 编译脚本
├── op_host/                               # Host 侧实现
│   ├── CMakeLists.txt
│   ├── tan_def.cpp                        # 算子定义（aclnn API 注册）
│   ├── tan_infershape.cpp                 # InferShape（输出 shape = 输入 shape）
│   └── arch32/
│       └── tan_tiling.cpp                 # Tiling 实现（多核切分 + UB 切分）
├── op_kernel/                             # Device 侧实现
│   ├── CMakeLists.txt
│   ├── tan_arch32.cpp                     # Kernel 入口（模板分发）
│   └── arch32/
│       ├── tan.h                          # Kernel 实现（核心计算逻辑）
│       ├── tan_tiling_data.h              # Tiling 数据结构
│       └── tan_tiling_key.h               # Tiling Key 定义
├── op_graph/
│   └── tan_proto.h                        # GE IR 算子原型注册
├── examples/                              # 调用示例
│   ├── CMakeLists.txt                     # aclnn 模式构建脚本
│   ├── CMakeLists_geir.txt                # GE IR 模式构建脚本
│   ├── run.sh                             # 统一运行脚本（--eager / --graph）
│   ├── test_aclnn_tan.cpp                 # aclnn 调用示例
│   └── test_geir_tan.cpp                  # GE IR 图模式调用示例
├── tests/
│   ├── st/                                # 系统测试
│   │   ├── test_aclnn_tan.cpp             # ST 测试工程（56 条用例）
│   │   ├── CMakeLists.txt
│   │   └── run.sh                         # ST 运行脚本
│   └── ut/                                # 单元测试
│       ├── run.sh                         # UT 运行脚本
│       ├── CMakeLists.txt
│       └── op_host/
│           ├── CMakeLists.txt
│           ├── test_op_host_main.cpp
│           ├── test_tan_infershape.cpp    # InferShape UT
│           └── test_tan_tiling.cpp        # Tiling UT
├── docs/
│   ├── aclnnTan.md                        # aclnn API 接口文档
│   ├── REQUIREMENT_ANALYSIS.md            # 需求分析文档
│   ├── DETAILED_DESIGN.md                 # 详细设计文档
│   ├── TEST_DESIGN.md                     # 测试设计文档
│   ├── PRECISION_VERIFICATION_REPORT.md   # 精度验收报告
│   └── DEVELOPMENT_LOG.md                 # 开发日志
└── build/                                 # 编译输出目录
    └── custom_opp_ubuntu_aarch64.run      # 算子包
```

## 实现说明

### 核心计算步骤

#### float32 路径

```text
sinVal = Sin(x)          // 计算 sin(x)
cosVal = Cos(x)          // 计算 cos(x)
y      = Div(sinVal, cosVal)  // tan(x) = sin(x) / cos(x)
```

#### float16 路径（升精度计算）

```text
x_fp32 = Cast(x, CAST_NONE)        // half -> float32
cosVal = Cos(x_fp32)               // 先算 cos（避免输入被覆盖）
sinVal = Sin(x_fp32)               // 再算 sin
result = Div(sinVal, cosVal)       // tan = sin / cos
y      = Cast(result, CAST_ROUND)  // float32 -> half
```

### Tiling 策略

- **多核切分**：总元素数均匀分配到各 AI Core
  - `blockFactor = CeilDiv(totalNum, coreNum)`
  - `usedCoreNum = CeilDiv(totalNum, blockFactor)`
- **UB 切分**：每个 Core 内按 UB 容量分块处理
  - float32：`ubFactor = FloorAlign(ubCanUse / 4 / 6, ubBlockSize)`（6 块 buffer）
  - float16：`ubFactor = FloorAlign(ubCanUse / 4 / 4, ubBlockSize)`（4 块 buffer）
- **流水线**：双 buffer（BUFFER_NUM=2），CopyIn → Compute → CopyOut 三级流水
