# Pdist

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品               |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品               |    √     |

## 功能说明

- 算子功能：计算输入二维 tensor 各行之间的 p-范数成对距离，等价于 PyTorch 的 `torch.nn.functional.pdist`。

计算公式：

- **0 < p < ∞**：dist(i, j) = (Σ|x_ik − x_jk|^p)^(1/p)（闵可夫斯基距离）
- **p = 0**：dist(i, j) = Σ𝟙(x_ik ≠ x_jk)（汉明距离）
- **p = ∞**：dist(i, j) = max|x_ik − x_jk|（切比雪夫距离）

## 算子原型设计

| 参数名 | 类别     | 描述                                        | 数据类型       | 数据格式 |
| :----- | :------- | :------------------------------------------ | :------------- | :------- |
| x      | 输入张量 | 二维输入张量，形状 (N, M)，N ≥ 2, M ≥ 1。  | FLOAT16、FLOAT32 | ND       |
| p      | 属性     | 距离参数，p ≥ 0，含 inf。可选，默认 2.0。   | FLOAT          | -        |
| y      | 输出张量 | 一维输出张量，形状 (N*(N-1)/2,)。            | 与 x 一致      | ND       |

- Atlas A2/A3 系列产品：数据类型支持 FLOAT16、FLOAT32。

## 约束说明

- 输入 x 必须为二维 tensor，且 dim(0) ≥ 2。
- 输出 y 的数据类型必须与 x 一致。
- p 的取值范围为 [0, +∞]。
- N 上限为 65535（computeNum 使用 uint64_t，N 过大时输出规模超出实际内存限制）。

## 调用说明

| 调用方式          | 样例代码                                                           | 说明                                     |
| :---------------- | :----------------------------------------------------------------- | :--------------------------------------- |
| aclnn 接口        | [test_aclnn_pdist](examples/test_aclnn_pdist.cpp)                  | 通过 aclnnPdist 接口调用，p 以 float 传入。 |
| aclnn Forward 接口 | [test_aclnn_pdist_forward](examples/test_aclnn_pdist_forward.cpp)  | 通过 aclnnPdistForward 接口调用，p 以 aclScalar 传入。 |

## 目录结构

```text
pdist/
├── README.md
├── op_host/                     # Host 侧：算子定义、Tiling、InferShape
│   ├── pdist_def.cpp
│   ├── pdist_infershape.cpp
│   └── pdist_tiling.cpp
├── op_kernel/                   # Kernel 侧：NPU 计算逻辑
│   ├── pdist.cpp
│   ├── pdist.h
│   ├── pdist_constants.h
│   ├── pdist_tiling_data.h
│   └── pdist_tiling_key.h
├── op_api/                      # ACLNN 接口层
│   ├── aclnn_pdist.h
│   ├── aclnn_pdist.cpp
│   ├── aclnn_pdist_forward.h
│   └── aclnn_pdist_forward.cpp
├── docs/                        # 接口文档
│   ├── aclnnPdist.md
│   └── aclnnPdistForward.md
├── examples/                    # 调用示例
│   ├── test_aclnn_pdist.cpp
│   └── test_aclnn_pdist_forward.cpp
└── tests/
    ├── ut/                      # 单元测试
    │   ├── op_host/             # Tiling + InferShape UT (14 cases)
    │   ├── op_api/              # ACLNN 接口 UT (9 cases)
    │   └── op_kernel/           # Kernel CPU 模拟器 UT
    ├── st/                      # 系统测试 (97 cases, Real NPU)
    │   ├── test_aclnn_pdist.cpp
    │   └── run.sh
    └── reports/
        └── iter3-acceptance-report.md
```

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| xiaoy2459 | 个人开发者 | Pdist | 2026/6/3 | Pdist算子适配开源仓 |
