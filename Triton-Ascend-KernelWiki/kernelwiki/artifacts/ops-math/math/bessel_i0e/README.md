# BesselI0e

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

- 算子功能：计算输入Tensor中每个元素对应的指数缩放第一类零阶修正贝塞尔函数值。

- 计算公式：

$$y = e^{-|x|} \cdot I_0(x)$$

其中 $I_0(x)$ 是第一类零阶修正贝塞尔函数。

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
      <td>待进行BesselI0e计算的入参，公式中的x。</td>
      <td>FLOAT16、FLOAT、BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行BesselI0e计算的出参，公式中的y。</td>
      <td>FLOAT16、FLOAT、BF16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入与输出的数据类型和形状必须一致。
- 仅支持图模式调用，不支持aclnn接口。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_bessel_i0e](./examples/test_geir_bessel_i0e.cpp) | 通过图模式调用BesselI0e算子。 |

## 第三方框架兼容性

- 对标 TensorFlow 的 `tf.math.bessel_i0e` 算子。
