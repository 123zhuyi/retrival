# RightShift

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

- 算子功能：按元素计算输入张量x与y的按位右移操作。

- 计算公式：

$$
z_i = x_i >> y_i
$$

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
      <td>x</td>
      <td>输入</td>
      <td>输入x是一个k维张量，是待进行right_shift计算的入参，公式中的x_i 。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>待进行right_shift计算的入参，公式中的y_i 。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>待进行right_shift计算的出参，公式中的z_i 。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                          | 说明                                                          |
|--------|---------------------------------------------------------------|-------------------------------------------------------------| 
| aclnn调用 | [test_aclnn_rightshift](./examples/test_aclnn_right_shift.cpp) | 通过[aclnnRightshift](./docs/aclnnRightShift.md)接口方式调用rightshift算子。|  
| 图模式调用 | [test_geir_right_shift](./examples/test_geir_right_shift.cpp) | 通过[算子IR](./op_graph/right_shift_proto.h)构图方式调用RightShift算子。 |
