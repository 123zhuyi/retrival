# SearchSorted

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：在一个已排序的张量sorted_sequence中查找给定张量values应该插入的位置。返回与values相同大小的张量，其中每个元素表示给定值在原始张量中应该插入的位置。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>sorted_sequence</td>
      <td>输入</td>
      <td> 输入Tensor ，其最后一维的数值按升序排列。</td>
      <td>FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>待插入的Tensor ，数据类型必须与输入sorted_sequence相同，除了最后一维外，其余维度必须与sorted_sequence相同。</td>
      <td>FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorter</td>
      <td>可选输入</td>
      <td>可选输入Tensor ，其shape与未排序的sorted_sequence相同，包含将其在最内层维度升序排序的索引序列。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>可选属性</td>
      <td>
          • 指定输出Tensor的数据类型，仅支持INT64 / INT32。<br>
          • 默认值为INT64。
      </td>
      <td>TYPE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>right</td>
      <td>可选属性</td>
      <td>
          • 可选布尔值，false表示当序列中存在相同值、插入位置不唯一时，插入位置靠左对齐；true表示在这种情况下插入位置靠右对齐。<br>
          • 默认值为false。
      </td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出Tensor，shape与输入values相同。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                              | 说明                                                                     |
|--------|-------------------------------------------------------------------|------------------------------------------------------------------------|
| 图模式调用 | [test_geir_search_sorted](./examples/test_geir_search_sorted.cpp)   | 通过[算子IR](./op_graph/search_sorted_proto.h)构图方式调用SearchSorted算子。 |
