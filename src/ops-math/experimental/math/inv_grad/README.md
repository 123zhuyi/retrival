# InvGrad

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：计算 Inv（取倒数）算子的反向梯度，对应前向 `y = 1 / x` 的梯度计算。
- 计算公式：

  $$dx_i = -dy_i \cdot y_i^2$$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 250px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>y</td>
    <td>输入</td>
    <td><ul><li>公式中的输入 y（前向 Inv 的输出）。</li><li>支持空 Tensor。</li><li>支持非连续 Tensor。</li><li>shape 维度：0-8。</li></ul></td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>输入</td>
    <td><ul><li>公式中的输入 dy（下游传回的梯度）。</li><li>shape 和 dtype 必须与 y 完全一致。</li></ul></td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dx</td>
    <td>输出</td>
    <td><ul><li>公式中的输出 dx（对前向输入 x 的梯度）。</li><li>shape 和 dtype 必须与 y/dy 完全一致。</li></ul></td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>
