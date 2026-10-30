# StatelessNormal

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：根据指定的均值（mean）和标准差（stdev）生成服从正态分布的随机数张量，使用基于计数器的Philox4x32-10随机数生成算法和Box-Muller变换，严格对标GPU精度（相同seed+offset产生相同随机序列），支持mean和stdev为标量或张量。

- 计算公式：

  $$
  y = BoxMuller(Philox(seed, offset)) \times stdev + mean
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
      <td>shape</td>
      <td>输入</td>
      <td>输出张量的形状。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>Philox随机数生成器的种子。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>Philox随机数生成器的偏移量，必须为4的倍数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>正态分布的均值，支持标量或与输出同形状的张量。</td>
      <td>FLOAT、FLOAT16、BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stdev</td>
      <td>输入</td>
      <td>正态分布的标准差，支持标量或与输出同形状的张量。</td>
      <td>FLOAT、FLOAT16、BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出指定形状的正态分布随机值。</td>
      <td>FLOAT、BF16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- offset必须为4的倍数。
- mean和stdev的数据类型与输出y锁步一致。
- mean和stdev为tensor时，shape必须与输出y的shape一致（不支持广播）。
- 仅支持SIMT编程模型（Ascend 950系列）。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_stateless_normal.cpp](./examples/arch35/test_aclnn_stateless_normal.cpp) | 通过[aclnnInplaceNormal](../dsa_random_normal/docs/aclnnInplaceNormal.md)接口方式调用StatelessNormal算子，mean和stdev均为标量。 |
