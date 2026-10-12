# AccumulateNv2V2

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR/Ascend 950DT | Yes |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | No |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | No |
| Atlas 200I/500 A2 推理产品 | No |
| Atlas 推理系列产品 | No |
| Atlas 训练系列产品 | No |

## 功能说明

完成 N 个输入 tensor 的逐元素累加计算。将输入的 N 个 tensor 逐元素相加，输出一个与输入 broadcast 后 shape 相同的结果 tensor。

计算公式：

$$output = \sum_{i=0}^{N-1} tensors[i]$$

其中 N 为输入 tensor 的个数（通过 DYNAMIC_INPUT 机制传入），各 tensor 逐元素累加。

aclnn 对外接口名称为 `aclnnSumV2`（两段式：`aclnnSumV2GetWorkspaceSize` + `aclnnSumV2`）。内部调用 AccumulateNv2V2 kernel 完成计算。

## 目录结构

```text
accumulate_nv2_v2/
├── CMakeLists.txt
├── README.md
├── examples/
│   └── test_aclnn_sum_v2.cpp
├── op_host/
│   ├── CMakeLists.txt
│   ├── accumulate_nv2_v2_def.cpp
│   ├── accumulate_nv2_v2_infershape.cpp
│   └── accumulate_nv2_v2_tiling.cpp
├── op_kernel/
│   ├── accumulate_nv2_v2.cpp
│   ├── accumulate_nv2_v2.h
│   ├── accumulate_nv2_v2_tiling_data.h
│   └── accumulate_nv2_v2_tiling_key.h
└── tests/
    └── .gitkeep
```

## 参数说明

### aclnnSumGetWorkspaceSize

| 参数名 | 输入/输出 | 数据类型 | 描述 |
| :----- | :-------: | :------- | :--- |
| tensors | 输入 | const aclTensorList* | 需要累加的输入 tensor 列表，包含 N 个 tensor。支持 FLOAT16、FLOAT、INT8、INT32、UINT8。各 tensor 的 shape 需满足 broadcast 关系，维度不超过 8。支持非连续 Tensor 和空 Tensor。 |
| out | 输出 | aclTensor* | 输出 tensor，shape 为各输入 tensor broadcast 后的 shape，dtype 与输入相同。不支持空 Tensor。 |
| workspaceSize | 输出 | uint64_t* | 返回 Device 侧所需 workspace 大小。 |
| executor | 输出 | aclOpExecutor** | 返回 op 执行器，包含算子计算流程。 |

### aclnnSumV2

| 参数名 | 输入/输出 | 数据类型 | 描述 |
| :----- | :-------: | :------- | :--- |
| workspace | 输入 | void* | Device 侧 workspace 内存地址。 |
| workspaceSize | 输入 | uint64_t | Device 侧 workspace 大小，由第一段接口获取。 |
| executor | 输入 | aclOpExecutor* | op 执行器，包含算子计算流程。 |
| stream | 输入 | aclrtStream | 指定执行任务的 Stream。 |

## 约束说明

- tensors 列表中各 tensor 的 dtype 必须相同，且与 out 的 dtype 相同。
- tensors 列表中各 tensor 的 shape 必须满足 broadcast 关系。
- 各 tensor 的维度不超过 8。
- 支持非连续 Tensor。
- 支持空 Tensor（直接返回）。
- 支持 0 维 tensor（标量 tensor，内部 reshape 为 [1]）。
- 当输入 tensor 个数超过 16 时，内部自动分批累加。
- 确定性计算：aclnnSumV2 默认确定性实现。

## 调用说明

### aclnn 调用示例

详见 [examples/test_aclnn_sum_v2.cpp](examples/test_aclnn_sum_v2.cpp)
