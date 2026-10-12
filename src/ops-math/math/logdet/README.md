# Logdet

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算输入self的行列式的自然对数。

- 计算公式：

  $$
  out = log(det(self))
  $$

    - 如果$det(self)$的结果是0，则$out = -inf$。
    - 如果$det(self)$的结果是负数，则$out = nan$。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>self</td>
      <td>输入</td>
      <td>计算公式中的输入self。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算公式中的输出out。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

输入数据中不支持存在溢出值Inf/Nan。

## 调用说明

| 调用方式 | 调用样例                                                            | 说明                                               |
|--------------|-----------------------------------------------------------------|--------------------------------------------------|
| aclnn调用 | [test_aclnn_logdet.cpp](examples/test_aclnn_logdet.cpp) | 通过[aclnnLogdet](docs/aclnnLogdet.md)接口方式调用Logdet算子。 |
