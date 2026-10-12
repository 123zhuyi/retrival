# HistogramFixedWidth

## 产品支持情况

| 产品　　　　　　　　　　　　　　　　　　　　　　　　　　 | 是否支持 |
| :---------------------------------------------------------| :--------:|
| <term>Ascend 950PR/Ascend 950DT</term>　　　　　　　　　 | √　　　　|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √　　　　|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √　　　　|
| <term>Atlas 200I/500 A2 推理产品</term>　　　　　　　　　| ×　　　　|
| <term>Atlas 推理系列产品</term>　　　　　　　　　　　　　| √　　　　|
| <term>Atlas 训练系列产品</term>                      |    ×     |

## 功能说明

- 算子功能：计算张量直方图。与TensorFlow的histogram_fixed_width算子兼容。
- 计算公式：以range=[min,max]作为统计上下限，在min和max之间划出等宽的数量为nbins的区间，统计张量x中元素在各个区间的数量。小于min的元素会被统计到第一个区间，大于max的元素会被统计到最后一个区间。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述　　　　　　　　　　　　　　　　 | 数据类型　　　　　　　　　　 | 数据格式 |
| --------| ----------------| --------------------------------------| ------------------------------| ----------|
| x　　　| 输入　　　　　 | 输入张量，待统计的数据。　　　　　　 | FLOAT、FLOAT16、INT32、INT64 | ND　　　 |
| range　| 输入　　　　　 | 形状为[2]的张量，包含[min,max]。 | FLOAT、FLOAT16、INT32、INT64 | ND　　　 |
| nbins　| 输入　　　　　 | 标量张量，直方图区间数量。　　　　　 | INT32　　　　　　　　　　　　| ND　　　 |
| y　　　| 输出　　　　　 | 直方图结果，形状为[nbins]。　　　　 | INT32　　　　　　　　　　　　| ND　　　 |
|dtype   | 属性           |预留属性，默认值为3                 | INT64                         |-|

## 约束说明

输入张量range的max必须大于min，否则报错。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_histogram_fixed_width](./examples/test_geir_histogram_fixed_width.cpp) | 通过[算子IR](./op_graph/histogram_fixed_width_proto.h)构图方式调用HistogramFixedWidth算子。 |
