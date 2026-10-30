# TanhGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：Tanh的反向实现。

- 计算公式：

$$
d = tanh(x)= (\frac{e^{x} - {e^{-x}}}{e^{x} + {e^{-x}}})  \tag{1}
$$
$$
dy = 1 -tan(x)^2  \tag{2}
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
      <td>正向的输出。</td>
      <td>FLOAT16、FLOAT、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>反向上一个算子的梯度。</td>
      <td>FLOAT16、FLOAT、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>complex_conj</td>
      <td>可选属性</td>
      <td>用于指示是否对复数类型进行共轭操作。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>反向的输出。</td>
      <td>FLOAT16、FLOAT、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_tanh_backward](./examples/test_aclnn_tanh_backward.cpp) | 通过[aclnnTanhBackward](./docs/aclnnTanhBackward.md)接口方式调用Abs算子。 |
