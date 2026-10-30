# Coordinates1DTo2D

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                   |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    √     |
| Atlas 推理系列产品                          |    √     |
| Atlas 训练系列产品                          |    √     |

## 功能说明

- 算子功能：将1D坐标转换为2D坐标，根据shape信息计算行列索引。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
<col style="width: 140px">
<col style="width: 140px">
<col style="width: 180px">
<col style="width: 213px">
<col style="width: 100px">
</colgroup>
<thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    </tr></thead>
<tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>1D坐标索引值。</td>
    <td>INT32、INT64、UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>shape</td>
    <td>输入</td>
    <td>形状信息，包含4个元素(N, D, H, W)。</td>
    <td>INT32、INT64、UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output_row</td>
    <td>输出</td>
    <td>输出的行索引。</td>
    <td>INT32、INT64、UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output_col</td>
    <td>输出</td>
    <td>输出的列索引。</td>
    <td>INT32、INT64、UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output_n</td>
    <td>输出</td>
    <td>输出的列数(W维度值)。</td>
    <td>INT32、INT64、UINT64</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- input[shape]元素个数必须为4
- input[x]与input[shape]数据类型必须一致
- shape[3]（W维度）不能为0

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_coordinates_1d_to_2d](examples/test_geir_coordinates_1d_to_2d.cpp) | 通过[算子IR](op_graph/coordinates_1d_to_2d_proto.h)接口方式调用Coordinates1DTo2D算子。 |
