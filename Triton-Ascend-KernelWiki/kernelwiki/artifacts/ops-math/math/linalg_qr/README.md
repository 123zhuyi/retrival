# LinalgQr

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入Tensor进行正交分解。

- 计算公式：

  $$
  A = QR
  $$

  其中$A$为输出Tensor，维度至少为2， A可以表示为正交矩阵$Q$与上三角矩阵$R$的乘积的形式

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
      <td>A</td>
      <td>输入</td>
      <td>计算公式中的输入A。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Q</td>
      <td>输出</td>
      <td>计算公式中的输出Q。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>R</td>
      <td>输出</td>
      <td>计算公式中的输出R。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式 | 调用样例                                                   | 说明                                               |
|--------------|--------------------------------------------------------|--------------------------------------------------|
| aclnn调用 | [test_aclnn_linalg_qr.cpp](examples/test_aclnn_linalg_qr.cpp) | 通过[aclnnLinalgQr](docs/aclnnLinalgQr.md)接口方式调用LinalgQr算子。 |
