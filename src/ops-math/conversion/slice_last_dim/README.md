# SliceLastDim

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量的最后一维进行切片，通过start、end指定切片范围，stride指定步长。

- 计算示例：
  - 输入 x shape = (2, 3, 10)，start = 0，end = 10，stride = 2
  - 输出 y shape = (2, 3, 5)
  - y[i, j, k] = x[i, j, start + k * stride] = x[i, j, 2 * k]

- 计算公式：
  输出 y 的 shape 与输入 x 相同，仅最后一维变为：

  ```text
  y.shape[-1] = ceil((end - start) / stride)
  y[..., k] = x[..., start + k * stride]
  ```

  start和end支持负数索引（相对最后一维长度），start会被截断到 [0, lastDim]，end会被截断到 [0, lastDim]。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| x      | 输入          | 输入张量 | INT8、INT16、INT32、INT64、FLOAT16、FLOAT、BF16 | ND |
| start  | 属性（必选）   | 最后一维切片的起始索引，支持负数索引 | INT | - |
| end    | 属性（必选）   | 最后一维切片的结束索引，支持负数索引 | INT | - |
| stride | 属性（可选）   | 切片步长，默认为1，必须 >= 1 | INT | - |
| y      | 输出          | 输出张量，与x具有相同的数据类型和格式 | 与x一致 | ND |

## 约束说明

- 输入张量x的维度范围为 [1, 8]。
- stride 必须 >= 1。
- start和end支持负数索引，负数时加上最后一维长度转换为正索引；start截断到 [0, lastDim]，end截断到 [0, lastDim]。
- 当 end <= start 时，输出最后一维长度为0。
- 支持动态Shape、动态Rank、动态编译静态。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| 图模式调用 | [test_geir_slice_last_dim](examples/test_geir_slice_last_dim.cpp) | 通过[算子IR](./op_graph/slice_last_dim_proto.h)构图方式调用SliceLastDim算子。 |
