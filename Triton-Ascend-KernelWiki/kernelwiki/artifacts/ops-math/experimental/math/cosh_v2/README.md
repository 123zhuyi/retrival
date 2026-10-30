# CoshV2 算子

Ascend C 自定义算子，逐元素计算输入 tensor 的双曲余弦值。对标 PyTorch `torch.cosh(input)` 接口。

## 算子说明

### 数学公式

```bash
out = cosh(self) = (exp(self) + exp(-self)) / 2
```

实际实现使用数值稳定公式，避免中间结果溢出：

```bash
cosh(x) = exp(|x| - ln2) + exp(-|x|) / 2
```

### 支持规格

| 项目 | 说明 |
|------|------|
| 目标芯片 | Ascend 910B3 (arch32, DAV_2201) |
| CANN 版本 | 9.0.0 |
| 数据类型 | float16, float32, bfloat16 |
| 调用方式 | ACLNN 两段式接口 |
| 输入维度 | 任意 shape，最大 8 维 |
| 数据格式 | ND |

### ACLNN 接口

```cpp
// 第一段：计算 workspace 大小
aclnnStatus aclnnCoshV2GetWorkspaceSize(
    const aclTensor* self,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

// 第二段：执行计算
aclnnStatus aclnnCoshV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

### 精度标准

| 数据类型 | 精度标准 | 说明 |
|---------|---------|------|
| float16 | 双千分之一 | 相对误差 < 0.001 的比例 >= 99.9% |
| float32 | 双万分之一 | 相对误差 < 0.0001 的比例 >= 99.99% |
| bfloat16 | 双千分之一 | 中间计算使用 float32 |

### 特殊值处理

| 输入 | 输出 | 说明 |
|------|------|------|
| 0 | 1.0 | cosh(0) = 1 |
| +inf | +inf | 正无穷 |
| -inf | +inf | cosh 为偶函数 |
| NaN | NaN | 保持 NaN |

## 目录结构

```text
cosh/
├── CMakeLists.txt              # 顶层构建配置
├── build.sh                    # 一键构建脚本
├── op_host/                    # Host 侧实现
│   ├── cosh_v2_def.cpp            # 算子原型注册
│   ├── cosh_v2_infershape.cpp     # Shape 推导
│   ├── CMakeLists.txt
│   └── arch32/
│       └── cosh_v2_tiling.cpp     # Tiling 实现 (Ascend910B)
├── op_kernel/                  # Kernel 侧实现
│   ├── cosh_v2_arch32.cpp         # Kernel 入口
│   ├── CMakeLists.txt
│   └── arch32/
│       ├── cosh.h              # Kernel 类定义与计算逻辑
│       ├── cosh_v2_tiling_data.h  # TilingData 结构体
│       └── cosh_v2_tiling_key.h   # TilingKey 模板参数定义
├── tests/                      # 测试
│   ├── ut/                     # 单元测试 (Host 侧 Tiling + InferShape)
│   └── st/                     # 系统测试 (ACLNN 端到端)
├── probe/                      # 穿刺验证工程
└── docs/                       # 文档
    ├── DEVELOPMENT_LOG.md      # 开发日志
    ├── REQUIREMENT_ANALYSIS.md # 需求分析
    ├── cosh_DETAILED_DESIGN.md # 详细设计
    ├── cosh_ITERATION_PLAN.md  # 迭代计划
    ├── cosh_TEST_DESIGN.md     # 测试设计
    ├── final_precision_report.md   # 精度验收报告
    └── final_code_review_report.md # 代码检视报告
```

## 构建方法

### 环境要求

- CANN 9.0.0 已安装
- `ASCEND_HOME_PATH` 环境变量已设置（如 `/home/developer/Ascend/ascend-toolkit/latest`）
- GCC 9.4.0+, C++17

### 编译算子包

```bash
# 编译（生成 Kernel 二进制 + 安装包）
bash build.sh --soc=ascend910b -j8

# 清理构建产物
bash build.sh --make_clean
```

编译成功后产物：

- Kernel 二进制：`build/op_kernel/ascendc_kernels/binary/ascend910b/`
- 安装包：`build/custom_opp_ubuntu_aarch64.run`

### 安装算子包

```bash
# 安装自定义算子包
./build/custom_opp_ubuntu_aarch64.run --full
```

### 运行时配置

```bash
# 配置自定义算子库路径
export LD_LIBRARY_PATH=<安装路径>/vendors/cosh_v2_custom/op_api/lib/:$LD_LIBRARY_PATH
```

## 测试方法

### 运行单元测试 (UT)

```bash
# 方法一：通过 build.sh
bash build.sh --soc=ascend910b -u

# 方法二：直接运行
cd tests/ut && ./run.sh
```

UT 测试内容：

- InferShape 测试（6 条）：1D/多维/4D/动态/大 shape/标量
- Tiling 测试（73 条）：fp16/fp32/bf16 各路径、单/双缓冲、边界值、阈值、多核分配

### 运行系统测试 (ST)

```bash
# 方法一：通过 build.sh（需先安装算子包）
bash build.sh --soc=ascend910b -s

# 方法二：直接运行
cd tests/st && bash run.sh

# 方法三：一次性运行全部测试
bash build.sh --soc=ascend910b -a
```

ST 测试内容：

- 端到端 ACLNN 接口调用，在真实 NPU 上执行
- 71 条用例覆盖 3 种 dtype、1D~8D 维度、特殊值、边界值
- 精度对比 CPU golden (std::cosh)

### 测试用例文件

| 文件 | 说明 |
|------|------|
| `tests/st/testcases/l0_test_cases.csv` | L0 门槛用例 23 条 |
| `tests/st/testcases/l1_test_cases.csv` | L1 功能用例 500 条 |

## 实现要点

1. **模板化设计**：通过 `<typename T, int BUFFER_MODE>` 模板参数支持 6 个 TilingKey 组合
2. **数值稳定性**：使用 `exp(|x|-ln2)` 替代 `exp(x)` 避免中间溢出
3. **统一 fp32 中间计算**：fp16/bf16 均 Cast 到 fp32 计算后 Cast 回原类型
4. **双缓冲流水**：totalNum > 1024 时启用双缓冲，MTE2 与 Vector 计算并行
5. **DataCopyPad**：安全处理非对齐尾块数据搬运
6. **动态多核**：使用 `GetCoreNumAiv()` 获取核数，禁止写死

## 测试结果

| 指标 | 结果 |
|------|------|
| UT 通过率 | 79/79 = 100% |
| ST 通过率 | 71/71 = 100% |
| fp16 精度 | 满足双千分之一 |
| fp32 精度 | 满足双万分之一 |
| bf16 精度 | 满足双千分之一 |
