# Sort

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：将输入tensor中的元素根据指定维度进行升序/降序排序，并且返回对应的排序结果值和索引值。

- 计算公式：

$$y1 = \operatorname{sort}(x, \text{axis}, \text{descending}, \text{stable})$$

$$y2 = \operatorname{argsort}(x, \text{axis}, \text{descending}, \text{stable})$$

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
      <td>待进行排序的输入tensor。</td>
      <td>FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示tensor在指定维度上排序的结果值，与x具有相同的类型和格式。</td>
      <td>FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示排序后每个元素在原tensor中的索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>指定排序的维度，默认为-1（最后一维）。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>属性</td>
      <td>是否降序排序，true为降序，false为升序，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>属性</td>
      <td>是否稳定排序，true为稳定排序，false为非稳定排序，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y2_dtype</td>
      <td>属性</td>
      <td>y2的数据类型，默认为DT_INT32。</td>
      <td>Type</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sort](./examples/test_aclnn_sort.cpp) | 通过[aclnnSort](./docs/aclnnSort.md)接口方式调用Sort算子。 |
| aclnn调用 | [test_aclnn_argsort](./examples/test_aclnn_argsort.cpp) | 通过[aclnnArgsort](./docs/aclnnArgsort.md)接口方式调用Sort算子。 |
