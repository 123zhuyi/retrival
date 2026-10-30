# Atanh 自定义算子

## 算子功能

逐元素计算输入 tensor 的反双曲正切值（inverse hyperbolic tangent），数学公式为：

```bash
y = atanh(x) = 0.5 * ln((1 + x) / (1 - x))
```

- 输入值域：开区间 (-1, 1)
- 输出值域：(-inf, +inf)
- 边界行为：x = +/-1 时输出 +/-inf，|x| > 1 时输出 NaN
- 对标：PyTorch `torch.atanh`

## 支持芯片与数据类型

| 芯片 | 架构 | 数据类型 |
|------|------|---------|
| Ascend910B | arch32 (DAV_2201) | float32, float16 |

### 精度标准

| 数据类型 | 相对容差 (rtol) | 绝对容差 (atol) |
|---------|----------------|----------------|
| float32 | 1e-4 | 1e-4 |
| float16 | 1e-2 | 1e-2 |

## 工程目录结构

```text
atanh/
├── CMakeLists.txt                          # 顶层构建配置
├── build.sh                                # 编译脚本
├── README.md                               # 本文件
├── op_graph/
│   └── atanh_proto.h                       # 算子原型定义
├── op_host/                                # Host 侧实现
│   ├── CMakeLists.txt
│   ├── atanh_def.cpp                       # 算子定义（dtype/format 注册）
│   ├── atanh_infershape.cpp                # Shape 推导（output_shape = input_shape）
│   └── arch32/
│       └── atanh_tiling.cpp                # Tiling 参数计算与 TilingKey 选择
├── op_kernel/                              # Kernel 侧实现
│   ├── CMakeLists.txt
│   ├── atanh_arch32.cpp                    # Kernel 入口（模板实例化）
│   └── arch32/
│       ├── atanh.h                         # Kernel 类（Init/Process/CopyIn/Compute/CopyOut）
│       ├── atanh_tiling_data.h             # TilingData 结构体
│       └── atanh_tiling_key.h              # TilingKey 模板参数声明
├── docs/                                   # 文档
│   ├── LOG.md                              # 开发日志
│   ├── REQUIREMENTS.md                     # 需求分析文档
│   ├── DESIGN.md                           # 详细设计文档
│   ├── PLAN.md                             # 迭代执行计划
│   ├── TEST.md                             # 测试设计文档
│   └── precision-report.md                 # 最终精度验收报告
└── tests/                                  # 测试
    ├── ut/                                 # 单元测试
    │   ├── CMakeLists.txt
    │   ├── run.sh                          # UT 运行脚本
    │   └── op_host/
    │       ├── CMakeLists.txt
    │       ├── test_atanh_infershape.cpp   # InferShape 单元测试
    │       ├── test_atanh_tiling.cpp       # Tiling 单元测试
    │       └── test_op_host_main.cpp       # UT 主入口
    ├── st/                                 # 集成测试（ST）
    │   ├── CMakeLists.txt
    │   ├── run.sh                          # ST 运行脚本
    │   └── test_aclnn_atanh.cpp            # ST 测试代码（aclnn 两段式接口）
    └── reports/                            # 迭代测试报告
        ├── iter1-integration-report.md
        ├── iter1-acceptance-report.md
        ├── iter2-integration-report.md
        ├── iter2-acceptance-report.md
        ├── iter3-integration-report.md
        └── iter3-acceptance-report.md
```

## 编译

```bash
bash build.sh --soc=ascend910b
```

编译成功后，算子包 `custom_opp_ubuntu_aarch64.run` 生成在 `build/` 目录下，并自动安装至 CANN OPP 目录。

## 运行 UT

```bash
cd tests/ut && bash run.sh
```

UT 覆盖 Host 侧的 InferShape 和 Tiling 逻辑，共 5 条用例：

| 测试模块 | 用例数 | 说明 |
|---------|--------|------|
| InferShape | 2 | 验证输出 shape 推导正确性 |
| Tiling | 3 | 验证 TilingData 计算和 TilingKey 分支选择 |

## 运行 ST

```bash
cd tests/st && bash run.sh
```

ST 在真实 NPU 上通过 aclnn 两段式接口执行精度验证，共 51 条用例：

| 测试分类 | 用例数 | 说明 |
|---------|--------|------|
| Shape 维度覆盖 | 20 | 1D~5D 多种 shape，数据规模 1~8192 |
| 值域分布覆盖 | 17 | 全零、近零、中等、近边界等多种值域 |
| 特殊模式覆盖 | 7 | 交替正负、单调递增/递减、随机混合 |
| Float16 类型覆盖 | 7 | float16 多种 shape 和值域 |

## 验证结果汇总

| 测试类型 | 通过数/总数 | 通过率 |
|---------|-----------|--------|
| UT | 5/5 | 100% |
| ST | 51/51 | 100% |

### ST 精度验证详情

| 数据类型 | 用例数 | 通过数 | 精度标准 |
|---------|--------|--------|---------|
| float32 | 44 | 44 | rtol=1e-4, atol=1e-4 |
| float16 | 7 | 7 | rtol=1e-2, atol=1e-2 |

### TilingKey 覆盖

| TilingKey | 数据类型 | 缓冲模式 | 触发条件 | 覆盖用例数 |
|-----------|---------|---------|---------|-----------|
| TK0 | float32 | 单缓冲 | totalNum <= 1024 | 33 |
| TK1 | float32 | 双缓冲 | totalNum > 1024 | 11 |
| TK2 | float16 | 单缓冲 | totalNum <= 1024 | 5 |
| TK3 | float16 | 双缓冲 | totalNum > 1024 | 2 |
