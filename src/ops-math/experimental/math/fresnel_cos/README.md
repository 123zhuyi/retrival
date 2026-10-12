# FresnelCos 算子用户指南

## 快速开始

### 算子概述

**FresnelCos** 是一个 Ascend C 自定义算子，对输入 tensor 的每个元素计算 Fresnel 余弦积分 C(x)。

**数学定义**：

```text
C(x) = ∫_0^x cos(π/2 * t^2) dt
```

C(x) 是奇函数（C(-x) = -C(x)），在 |x| → ∞ 时收敛到 ±0.5。

### 关键特性

- **多类型支持**：FLOAT、FLOAT16、BFLOAT16
- **算法**：Cephes 风格的三段近似（小值多项式 / 大值渐近展开 / 极限饱和），fp16/bf16 在内部使用 fp32 中间精度
- **数值健壮**：对 |x| > 阈值的输入做饱和裁剪，避免中间 fp32 计算溢出
- **多架构支持**：Ascend950（DAV_3510 / arch35）
- **精度达标**：FP32 atol/rtol=1e-4，FP16/BF16 atol/rtol=1e-3

---

## 安装与编译

### 环境要求

| 项目 | 要求 |
|------|------|
| **CANN 版本** | >= 9.0.0 |
| **Ascend 芯片** | Ascend950 |
| **操作系统** | Ubuntu 18.04+ / CentOS 7.6+ |
| **编译工具** | g++ 7.3+，cmake 3.13+ |

### 编译步骤

```bash
cd ~/ops-math
bash build.sh --pkg --experimental --soc=ascend950 --ops=fresnel_cos -j16
./build_out/cann-ops-math-custom_linux-x86_64.run
```

### 运行调用样例

```bash
bash build.sh --experimental --run_example fresnel_cos eager cust --vendor_name=custom --soc=ascend950
```

期望输出：`run test_aclnn_fresnel_cos, execute samples success`

---

## ACLNN 接口

### 接口定义

```cpp
aclnnStatus aclnnFresnelCosGetWorkspaceSize(
    const aclTensor *x,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnFresnelCos(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### 参数说明

| 参数 | 方向 | 类型 | 约束 |
|------|------|------|------|
| `x` | 输入 | aclTensor* | DT_FLOAT / DT_FLOAT16 / DT_BF16；最高 8 维 |
| `out` | 输出 | aclTensor* | dtype 与 shape 必须与 `x` 一致 |

---

## 目录结构

```text
fresnel_cos/
├── CMakeLists.txt
├── README.md
├── examples/arch35/test_aclnn_fresnel_cos.cpp
├── op_api/
│   ├── aclnn_fresnel_cos.{h,cpp}    # L2 接口
│   └── fresnel_cos.{h,cpp}          # L0 接口
├── op_host/
│   ├── fresnel_cos_def.cpp
│   ├── fresnel_cos_infershape.cpp
│   └── arch35/fresnel_cos_tiling.cpp
├── op_kernel/
│   ├── fresnel_cos.cpp              # Kernel 入口
│   └── arch35/
│       ├── fresnel_cos.h
│       ├── fresnel_cos_tiling_data.h
│       └── fresnel_cos_tiling_key.h
└── tests/
```
