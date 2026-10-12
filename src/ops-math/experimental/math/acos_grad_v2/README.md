# AcosGradV2

AcosGrad 算子的 A2（Atlas A2 训练/推理系列产品，Ascend910B / DAV_2201）适配版本，基于
`math/acos_grad` 的功能定义实现，采用 aclnn（registry-invoke）工程结构，便于在 A2 设备上
编译、部署与精度验证。

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

> 本版本（v2）面向 A2（ascend910b）适配；原 `math/acos_grad` 面向 Ascend950（arch35）。

## 功能说明

- 算子功能：计算 Acos（反余弦）算子的反向梯度。
- 算子公式：

  $$
  z_i = -1 \cdot dy_i \cdot \dfrac{1}{\sqrt{1 - y_i^2}}
  $$

  其中：
  - $y_i$ 为前向 Acos 算子的输入张量；
  - $dy_i$ 为上游传入的梯度；
  - $z_i$ 为对原始输入张量的梯度，等于上游梯度乘以 $-1/\sqrt{1 - y_i^2}$。

- 超出定义域（$|y_i| > 1$）时：$1 - y_i^2 < 0$，平方根结果为 NaN，结果同样为 NaN。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                          | 数据类型                  | 数据格式 |
| :----: | :------------: | :------------------------------------------- | :----------------------- | :------: |
|   y    |      输入      | 前向 Acos 算子的输入张量。值域期望落在 [-1, 1]。 | FLOAT16, FLOAT32, BFLOAT16 |    ND    |
|   dy   |      输入      | 上游传入的梯度张量，shape 与 dtype 与 y 一致。   | FLOAT16, FLOAT32, BFLOAT16 |    ND    |
|   z    |      输出      | 对原始输入张量的梯度，shape 与 dtype 与 y 一致。 | FLOAT16, FLOAT32, BFLOAT16 |    ND    |

## 约束说明

- y 与 dy 的 shape 必须完全一致。
- y 与 dy 的 dtype 必须完全一致。
- 仅支持 ND 格式。

## 性能说明

CANN 当前未提供同名官方 `aclnnAcosGrad` 单算子，故以 **torch_npu 的 `torch.acos` 反向路径**
（用户实际调用 `autograd.grad` 时 NPU 上执行的实现）作为标杆。该路径在 NPU 上被拆分为多算子链：
`Acos（前向重算）+ Mul×2 + Neg×2 + Rsqrt + Adds`。本算子将其融合为单个 kernel。

加速比 = 标杆反向链 device 核时之和 / 本算子单 kernel device 核时（均由 profiler 采集，
shape = [1024, 1024]，10 轮平均）。

| 数据类型 | 标杆反向链 (us) | 本算子 (us) | 加速比 |
| :------: | :-------------: | :---------: | :----: |
| FP32     | 26.4            | 6.97        | 3.8x   |
| FP16     | 22.6            | 7.90        | 2.9x   |
| BF16     | 26.1            | 8.09        | 3.2x   |

- 三种数据类型平均加速比约 **3.3x**。
- 加速主要来自融合：将反向链的多次基础算子访存与启动开销合并为单次，减少 HBM 来回搬运。

## A2 适配要点

- 算子定义 `op_host/acos_grad_v2_def.cpp` 仅注册 `ascend910b`（AICore 配置）。
- `CMakeLists.txt` 中 `COMPUTE_UNIT=ascend910b`、`TILING_DIR=arch22`，并通过
  `add_kernel_sources` 显式注册 `arch22/acos_grad_v2.cpp` 入口。
- Kernel 使用标准 Ascend C 高阶向量 API（Cast/Mul/Muls/Adds/Sqrt/Div 等），
  FP16/BF16 先 Cast 到 FP32 计算再 Cast 回原类型，FP32 直接计算；这些 API 在 A2 上原生支持。

## 调用说明

| 调用方式   | 调用样例                                          | 说明                                              |
| ---------- | ------------------------------------------------- | ------------------------------------------------- |
| aclnn 调用 | [test_aclnn_acos_grad_v2](./examples/test_aclnn_acos_grad_v2.cpp) | 两段式 aclnn 调用并在 NPU 上验证精度。 |
