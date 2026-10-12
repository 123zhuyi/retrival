# SliceWithAxes

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：沿指定axes从输入张量中提取切片，每个axis由offsets指定起始位置、size指定切片长度。

- 计算示例：
  - 输入 x shape = (10, 20, 30)，axes = [0, 2]，offsets = [1, 5]，size = [5, 10]
  - 输出 y shape = (5, 20, 10)
  - y[i, j, k] = x[1 + i, j, 5 + k]

- 计算公式：
  输出 y 的 shape 与输入 x 相同，仅 axes 中指定的维度被替换为对应的 size。对axes中的第k个轴axis_k：

  ```text
  y[..., axis_k, ...] = x[..., offsets[k] : offsets[k] + size[k], ...]

  当 size[k] == -1 时，表示从 offsets[k] 切到该轴末尾，即 size[k] = x.shape[axis_k] - offsets[k]
  ```

## 参数说明

| 参数名  | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| x      | 输入          | 输入张量 | INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL | ND |
| offsets| 输入          | 各axis的切片起始位置，1D张量，长度等于axes长度 | INT32、INT64 | ND |
| size   | 输入          | 各axis的切片长度，1D张量，长度等于axes长度。值为-1表示从offset切到该轴末尾 | INT32、INT64 | ND |
| axes   | 属性（必选）   | 指定切片的轴列表 | ListInt | - |
| y      | 输出          | 输出张量，与x具有相同的数据类型和格式 | 与x一致 | ND |

## 约束说明

- 输入张量x的维度范围为 [1, 8]。
- axes中的每个axis必须满足 0 <= axis < rank。
- 对每个切片轴，需满足 offsets[k] >= 0 且 offsets[k] + size[k] <= x.shape[axis_k]（size[k]为-1时自动取到末尾）。
- 输出各维度大小必须 >= 0。
- 支持动态Shape、动态Rank、动态编译静态。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| 图模式调用 | [test_geir_slice_with_axes](examples/test_geir_slice_with_axes.cpp) | 通过[算子IR](./op_graph/slice_with_axes_proto.h)构图方式调用SliceWithAxes算子。 |
