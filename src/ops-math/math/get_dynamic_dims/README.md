# GetDynamicDims

## 功能说明

GetDynamicDims算子用于根据`shape_info`中标记的未知维度，从动态输入张量中提取实际维度值并输出。

## 参数说明

| 参数名 | 输入/输出 | 参数类型 | 数据类型 | 说明 |
| - | - | - | - | - |
| input | 输入 | Tensor动态输入 | int32、int64 | 输入张量，每个输入表示一个shape向量。 |
| dims | 输出 | Tensor | int32、int64 | 输出未知维度的实际取值。 |
| shape_info | 属性 | ListInt | int64 | 各输入的shape信息，按`rank, dim0, dim1, ...`编码，`-1`表示未知维度。 |
| N | 属性 | Int | int64 | 动态输入个数。 |

## 约束说明

- 输入`input`数量必须与属性`N`一致。
- `shape_info`中描述的输入个数和每个输入rank必须与实际输入一致。
- 输出`dims`的数据类型支持int32和int64。

## 调用说明

| 调用方式 | 样例链接 | 说明 |
| - | - | - |
| 图模式调用 | [test_geir_get_dynamic_dims](./examples/test_geir_get_dynamic_dims.cpp) | 通过[算子IR](./op_graph/get_dynamic_dims_proto.h)构图方式调用GetDynamicDims算子。 |
