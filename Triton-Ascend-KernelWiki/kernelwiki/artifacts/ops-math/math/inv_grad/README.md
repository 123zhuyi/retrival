# InvGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算Inv（取倒数）算子的反向梯度，对应前向`y = 1 / original_x`的梯度计算。
- 算子公式：

  $$
  y_i = -1 \cdot x_i \cdot x_i \cdot grad_i
  $$

  其中：
  - $x_i$ 为前向Inv算子的输出（即 $1 / \mathrm{original\_x}_i$）；
  - $grad_i$ 为上游传入的梯度；
  - $y_i$ 为对原始输入 $\mathrm{original\_x}_i$ 的梯度。

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
      <td>输入张量。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>对应的输入梯度。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输入张量x的导数梯度</td>
      <td>FLOAT16, FLOAT32, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_inv_grad](./examples/test_geir_inv_grad.cpp)   | 通过[算子IR](./op_graph/inv_grad_proto.h)构图方式调用inv_grad算子。 |
