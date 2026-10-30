# StackBallQuery

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             | ×        |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     | √        |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √        |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×     |
| <term>Atlas 推理系列产品</term>                             | ×        |
| <term>Atlas 训练系列产品</term>                              | √        |
| Kirin X90 处理器系列产品|√|
| Kirin 9030 处理器系列产品|√|

## 功能说明

- 算子功能：Stack Ball Query是KNN的替代方案，用于查找点p1指定半径范围内的所有点（在实现中设置了K的上限）。
返回的点不一定是距离p1最近的点，而是半径范围内的前K个点。
和Ball Query算子相比，Stack Ball Query将Ball Query算子的输入进行了堆叠，center_xyz和xyz的维度从3变成了2。
优势：当半径范围内存在大量点时，该算法比KNN算法更快，同时保证了固定的区域尺度，使局部区域特征更通用。
使用模型：该算子在PointNet++模型中被提出，在该模型及其衍生模型中使用。
Stack Ball Query具体操作如下：
根据输入的center_xyz，对同一个batch内的每一个点计算和xyz之间的距离。
如果距离小于max_radius，保存xyz点的索引值。
寻找到sample_num个满足要求的索引后，退出循环。
输出保存的索引值。

## 参数说明

| 参数名               | 输入/输出/属性 | 描述                                         | 数据类型         | 数据格式 |
| -------------------- | -------------- | -------------------------------------------- | ---------------- | -------- |
| xyz                  | 输入           | 2D Tensor，经过堆叠的xyz的坐标值           | FLOAT16、FLOAT32 | ND       |
| center_xyz           | 输入           | 2D Tensor，经过堆叠的center_xyz的坐标值    | FLOAT16、FLOAT32 | ND       |
| xyz_batch_cnt        | 输入           | 1D Tensor，表示每个batch中的xyz点个数        | INT32、INT64     | ND        |
| center_xyz_batch_cnt | 输入           | 1D Tensor，表示每个batch中的center_xyz点个数 | INT32、INT64     | ND       |
| idx                  | 输出           | stack ball query后得到的索引值               | INT32            | ND       |
| max_radius           | 输入属性       | 最大半径值                                   | FLOAT            | -       |
| sample_num           | 输入属性       | 最大采样数                                   | INT              | -       |

## 约束说明

无。

## 调用说明

| 调用方式  | 样例代码                                                              | 说明                                                                        |
| --------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| aclnn接口 | [test_geir_stack_ball_query](./examples/test_geir_stack_ball_query.cpp) | 通过[算子IR](./op_graph/stack_ball_query_proto.h)构图方式调用StackBallQuery算子。 |
