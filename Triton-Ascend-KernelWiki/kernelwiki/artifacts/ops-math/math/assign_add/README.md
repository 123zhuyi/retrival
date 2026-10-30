# AssignAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：完成在原有tensor上的加法计算。

- 计算公式：

$$out_i = ref_i + value_i$$

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
      <td>ref</td>
      <td>输入</td>
      <td>输入张量，公式中的ref_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>value</td>
      <td>输入</td>
      <td>输入张量，表示要加到ref_i上的值，公式中的value_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ref</td>
      <td>输出</td>
      <td>输出张量，公式中的out_i，与ref同地址。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、INT64、UINT8</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>可选，是否使用锁来保护更新操作。</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                | 说明                                                             |
|--------------|-----------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_assign_add.cpp](examples/test_geir_assign_add.cpp)   | 通过[算子IR](op_graph/assign_add_proto.h)构图方式调用AssignAdd算子。                   |
