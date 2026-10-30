# RealV2

## 功能说明

RealV2 算子提取输入 tensor 的实部，对标 PyTorch `torch.real` 接口语义。

- 对于复数类型输入（COMPLEX64/COMPLEX32），提取其实部并以对应的浮点类型输出。
- 对于实数类型输入（FLOAT/FLOAT16），直接返回输入值（实数的实部等于自身）。

**计算公式**:

- 复数输入: `out_i = Re(self_i) = a_i`，其中 `self_i = a_i + b_i * j`
- 实数输入: `out_i = self_i`

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR / Ascend 950DT | Yes |

## 参数说明

### aclnnRealV2GetWorkspaceSize

```cpp
aclnnStatus aclnnRealV2GetWorkspaceSize(
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| self | 输入 | 输入 tensor。支持 FLOAT、FLOAT16、COMPLEX64、COMPLEX32 数据类型。支持 0-8 维，ND 格式。 |
| out | 输出 | 输出 tensor，包含 self 的实部。shape 与 self 一致，dtype 由 self 推导确定。 |
| workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 |
| executor | 输出 | 返回 op 执行器，包含了算子计算流程。 |

### aclnnRealV2

```cpp
aclnnStatus aclnnRealV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址。 |
| workspaceSize | 输入 | workspace 大小，由 aclnnRealV2GetWorkspaceSize 获取。 |
| executor | 输入 | op 执行器。 |
| stream | 输入 | 指定执行任务的 AscendCL Stream。 |

### 数据类型映射

| 输入 dtype (self) | 输出 dtype (out) | 说明 |
| :--- | :--- | :--- |
| FLOAT (float32) | FLOAT (float32) | 实数直传 |
| FLOAT16 (half) | FLOAT16 (half) | 实数直传 |
| COMPLEX64 | FLOAT (float32) | 提取实部 |
| COMPLEX32 | FLOAT16 (half) | 提取实部 |

## 约束说明

1. 本算子仅支持 Ascend 950PR/Ascend 950DT 产品。
2. self 的数据类型仅支持 FLOAT、FLOAT16、COMPLEX64、COMPLEX32。
3. out 的数据类型由 self 推导确定，不可自由指定。
4. out 的 shape 必须与 self 的 shape 完全一致。
5. 支持 0-8 维 tensor。
6. 支持空 tensor（0 元素），此时 out 也为空 tensor。
7. 精度标准为二进制一致（Bitwise Match），无数值计算误差。

## 目录结构

```text
real_v2/
├── CMakeLists.txt
├── README.md
├── examples/
│   └── test_aclnn_real_v2.cpp
├── op_host/
│   ├── CMakeLists.txt
│   ├── real_v2_def.cpp
│   ├── real_v2_infershape.cpp
│   └── real_v2_tiling.cpp
├── op_kernel/
│   ├── real_v2.cpp
│   ├── real_v2.h
│   ├── real_v2_tiling_data.h
│   └── real_v2_tiling_key.h
└── tests/
    └── .gitkeep
```
