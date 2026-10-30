# Rfft1D

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对输入张量进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。
- 计算公式：

  $$
  y = W \cdot x
  $$

  其中W为DFT矩阵，其定义为$W_{jk}=e^{-ij2\pi{\tfrac{k}{N}}n}$
  
- 示例：

  假设x为 {1, 2, 3, 4} 则y = {10, 0, -2, 2, -2, 0} = {10 + 0j, -2 + 2j, -2 + 0j} 

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>公式中的输入张量x。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dft</td>
      <td>输入</td>
      <td>输入的DFT矩阵或者终止矩阵，使用该矩阵则Rfft1D变换可表示为矩阵乘法。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>n</td>
      <td>输入</td>
      <td>表示信号长度，n取值范围为[1, 4096]时dft应传入DFT矩阵，否则应传入终止矩阵。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>norm</td>
      <td>输入</td>
      <td>表示归一化模式。支持取值：1表示不归一化，2表示按1/n归一化，3表示按1/sqrt(n)归一化。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示公式中的输出</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_acl_rfft1_d](./examples/test_acl_rfft1d.cpp) | 通过[aclRfft1D](./docs/aclRfft1D.md)接口方式调用Rfft1D算子。    |
