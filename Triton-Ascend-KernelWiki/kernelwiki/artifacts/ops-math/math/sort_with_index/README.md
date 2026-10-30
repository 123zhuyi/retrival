# SortWithIndex

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                    |    √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×      |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×      |
| <term>Atlas 200I/500 A2 推理产品</term>                    |    ×      |
| <term>Atlas 推理系列产品</term>                            |   ×       |
| <term>Atlas 训练系列产品</term>                            |   √       |

## 功能说明

- 算子功能：将输入tensor按照元素值大小进行排序，index值跟随对应元素值进行排序。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>待进行元素值排序的输入张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>待进行元素值排序的输入张量。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>用来作为排序标准的维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>输入</td>
      <td>控制排序顺序，True为降序，False为升序。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>输入</td>
      <td>是否稳定排序, True为稳定排序，False为非稳定排序。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>    
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>元素值排序后的输出结果。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_index</td>
      <td>输出</td>
      <td>索引值随元素值排序后的输出结果。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x和index的shape必须一致

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_sort_with_index](./examples/test_geir_sort_with_index.cpp) | 通过[算子IR](../less/op_graph/less_proto.h)构图方式调用SortWithIndex算子。 |
