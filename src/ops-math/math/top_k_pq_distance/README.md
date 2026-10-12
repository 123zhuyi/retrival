# TopKPQDistance

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                                    |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品                    |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品                    |    √     |
| Atlas 200I/500 A2 推理产品                                   |    √     |
| Atlas 推理系列产品                                            |    √     |
| Atlas 训练系列产品                                            |    √     |

## 功能说明

- 算子功能：基于分组极值剪枝的TopK距离计算，用于向量检索场景。从多组PQ距离数据中找出最大或最小的k个距离及其对应的ivf和index。

- 计算流程：
  1. 从grouped_extreme_distance中选出TopK个极值（分组剪枝）
  2. 从选中的分组中展开计算精确TopK距离
  3. 输出TopK距离、ivf和index

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| actual_count | 输入（动态） | 每组实际元素个数 | INT32 | ND |
| pq_distance | 输入（动态） | PQ距离数据 | FLOAT16、FLOAT | ND |
| grouped_extreme_distance | 输入（动态） | 每组极值距离 | FLOAT16、FLOAT | ND |
| pq_ivf | 输入（动态） | IVF索引 | INT32 | ND |
| pq_index | 输入（动态） | PQ索引 | INT32 | ND |
| order | 属性（可选） | 排序方式，"ASC"升序或"DES"降序，默认"ASC" | STRING | - |
| k | 属性（必选） | TopK的k值 | INT | - |
| group_size | 属性（必选） | 分组大小 | INT | - |
| topk_distance | 输出 | TopK距离值 | FLOAT16、FLOAT | ND |
| topk_ivf | 输出 | TopK对应的ivf | INT32 | ND |
| topk_index | 输出 | TopK对应的index | INT32 | ND |

## 约束说明

- k不能为0，且不能大于actual_count的总和。
- group_size不能为0。
- actual_count必须能被group_size整除。
- 支持多组动态输入（data_batch），每组包含5个输入。

## 调用说明

| 调用方式   | 样例代码 | 说明 |
|------------|----------|------|
| 图模式调用 | [test_geir_top_k_pq_distance](./examples/test_geir_top_k_pq_distance.cpp) | 通过[算子IR](./op_graph/top_k_pq_distance_proto.h)构图方式调用TopKPQDistance算子。 |
