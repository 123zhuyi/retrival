# RsqrtGrad

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 算子功能：完成梯度平方根计算。
- 计算公式：

$$
z = y * y * y * dy * (-0.5)
$$

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
      <td>y</td>
      <td>输入</td>
      <td>待进行rsqrt_grad计算的入参，公式中的y。</td>
      <td>DT_FLOAT,DT_FLOAT_16,DT_BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>待进行rsqrt_grad计算的入参，公式中的dy。</td>
      <td>DT_FLOAT,DT_FLOAT_16,DT_BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>待进行rsqrt_grad计算的出参，公式中的z。</td>
      <td>DT_FLOAT,DT_FLOAT_16,DT_BF16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 调用样例                                                  | 说明                                                                            |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------------------------------- |
| aclnn调用 | [test_rsqrt_grad](./examples/test_aclnn_rsqrt_grad.cpp) | 使用自动生成的aclnn接口调用RsqrtGrad算子。 |
