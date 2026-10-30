# NanToNum

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将输入中的NaN、正无穷大和负无穷大值分别替换为nan、posinf、neginf指定的值。

- 计算公式：

$$y=\begin{cases} nan & \text{if } x \text{ is NaN} \\ posinf & \text{if } x \text{ is } +\infty \\ neginf & \text{if } x \text{ is } -\infty \\ x & \text{otherwise} \end{cases}$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 320px">
  <col style="width: 290px">
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
      <td>待进行nan_to_num计算的入参，公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行nan_to_num计算的出参，公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nan</td>
      <td>属性</td>
      <td>替换tensor元素中NaN的值。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>posinf</td>
      <td>属性</td>
      <td>替换tensor元素中正无穷大的值。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>neginf</td>
      <td>属性</td>
      <td>替换tensor元素中负无穷大的值。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_nan_to_num](./examples/test_aclnn_nan_to_num.cpp) | 通过[aclnnNanToNum](./docs/aclnnNanToNum&aclnnInplaceNanToNum.md)接口方式调用NanToNum算子。 |
