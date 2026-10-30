# ComplexV3 算子

将两个实数张量（real 和 imag）组合为一个复数张量，即 `output[i] = real[i] + imag[i] * j`。

## 支持规格

| 项目 | 说明 |
|------|------|
| 输入数据类型 | float32, float16 |
| 输出数据类型 | complex64（对应 float32）, complex32（对应 float16） |
| 支持芯片 | Ascend910B（910B3） |
| 架构 | arch32 |
| 广播 | 支持 NumPy 广播语义（单维、双向、维度补齐、标量广播，最高 8D） |
| ACLNN 接口 | `aclnnComplexV3(real, imag, out)` 两段式接口 |

## 目录结构

```text
ops/complex_v3/
├── CMakeLists.txt                  # 根构建配置
├── build.sh                        # 一键编译脚本
├── README.md                       # 本文件
├── op_host/
│   ├── CMakeLists.txt
│   ├── complex_v3_def.cpp             # 算子定义与注册
│   ├── complex_v3_infershape.cpp      # Shape 推导（NumPy 广播）
│   └── arch32/
│       └── complex_v3_tiling.cpp      # Tiling 策略（多核切分 + UB 切分）
├── op_kernel/
│   ├── CMakeLists.txt
│   ├── complex_v3_arch32.cpp          # Kernel 入口
│   └── arch32/
│       ├── complex.h               # Kernel 实现（无广播 + 广播路径）
│       ├── complex_v3_tiling_data.h   # TilingData 结构体
│       └── complex_v3_tiling_key.h    # TilingKey 定义
├── tests/
│   ├── ut/                         # 单元测试（43 例）
│   │   ├── CMakeLists.txt
│   │   ├── common/                 # UT 公共框架
│   │   └── op_host/
│   │       ├── test_complex_v3_tiling.cpp      # Tiling UT（26 例）
│   │       └── test_complex_v3_infershape.cpp  # InferShape UT（17 例）
│   └── st/                         # 系统测试（58 例）
│       ├── CMakeLists.txt
│       ├── test_aclnn_complex_v3.cpp  # ST 测试工程
│       ├── run.sh                  # ST 运行脚本
│       ├── perf_test_v2.cpp        # 性能测试程序
│       └── testcases/
│           ├── L0_test_cases.csv   # L0 用例（32 条）
│           └── L1_test_cases.csv   # L1 用例（500 条）
└── docs/
    ├── REQUIREMENT_ANALYSIS.md     # 需求分析
    ├── DETAILED_DESIGN.md          # 详细设计
    ├── TEST_DESIGN.md              # 测试设计
    ├── PERFORMANCE_REPORT.md       # 性能报告
    ├── CODE_REVIEW_REPORT.md       # 代码检视报告
    └── DEVELOPMENT_LOG.md          # 开发日志
```

## 编译

```bash
cd ops/complex
bash build.sh
```

编译成功后生成安装包 `build/custom_opp_ubuntu_aarch64.run`。

安装算子包：

```bash
./build/custom_opp_ubuntu_aarch64.run --install
```

默认安装到 `$ASCEND_OPP_PATH/vendors/complex_v3_custom/`。

## 测试

### 单元测试（UT）

```bash
cd tests/ut
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./bin/ut_test
```

共 43 例（Tiling 26 例 + InferShape 17 例），覆盖全 dtype、广播、边界场景。

### 系统测试（ST）

```bash
cd tests/st
bash run.sh --mode=real
```

支持两种模式：

- `--mode=mock`：CPU 模拟执行（无需 NPU）
- `--mode=real`：NPU 真实执行（需安装算子包）

可通过 `ASCEND_DEVICE_ID` 环境变量指定 NPU 设备。

共 58 例，精度标准为 bit-exact（diff_thd=0, pct_thd=0）。

## 性能数据

| 场景 | DType | 输出元素数 | Kernel 时间 (us) | 带宽 (GB/s) | 带宽利用率 |
|------|-------|-----------|-----------------|------------|-----------|
| 无广播 | fp32 | 16M | 2,900 | 92.58 | 5.8% |
| 无广播 | fp16 | 16M | 2,856 | 47.00 | 2.9% |
| 广播(行) | fp32 | 1M | 1,120 | 11.24 | 0.7% |

测试环境：Ascend910B (910B3)，ACL Event Timing，10 次 warmup + 50 次重复取中位数。

广播路径采用 DataCopyPad 预加载输入到 UB + LocalTensor 逐元素索引方案（arch32 不支持 Scatter/Gather 向量指令），性能受架构限制。无广播路径通过宽类型打包写入 + 4x 循环展开优化，相比初始版本提升 4.1x。

## 设计要点

- TilingKey 双参数：`(D_T, BROADCAST_MODE)`，共 4 个变体
- 无广播路径：连续段 DataCopyPad + 交织写入，支持多核并行
- 广播路径：DataCopyPad 将输入整体预加载到 UB，按输出线性索引反推输入索引，通过 LocalTensor::GetValue 从 UB 读取（避免直接访问 GM）
- 空 tensor 保护：Tiling 早期返回 + Kernel 越界核跳过
- 纯数据搬运算子，无浮点运算，精度为 bit-exact
