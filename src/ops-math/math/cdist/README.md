# Cdist

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：计算两个向量集合中每对向量之间的p范数距离。
- 计算公式：对于输入张量x1和x2，设其shape分别为$[B, P, M]$和$[B, R, M]$，输出张量y的shape为$[B, P, R]$。

  当$p>0$时：

  $$
  y_{b,i,j}=\left(\sum_{k=1}^{M}|x1_{b,i,k}-x2_{b,j,k}|^p\right)^{\frac{1}{p}}
  $$

  当$p=0$时，输出为对应向量中不相等元素的数量；当$p=+\infty$时，输出为对应向量元素差值绝对值的最大值。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>x1</td>
      <td>输入</td>
      <td>第一个输入张量，shape为[B, P, M]。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>第二个输入张量，shape为[B, R, M]。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>p范数距离计算结果，shape为[B, P, R]。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>可选属性</td>
      <td>范数阶数，取值范围为[0, +∞]，默认值为2.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- x1、x2和y的数据类型必须一致。
- x1和x2的维度范围为2~8，最后一维长度必须相同。
- x1和x2除最后两维外的其他维度必须满足broadcast关系；输出对应维度为broadcast后的shape。
- 若x1的倒数第二维为P，x2的倒数第二维为R，则输出的最后两维为[P, R]。
- p必须大于或等于0，支持+∞，常用取值为0、1.0、2.0和+∞。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_cdist](./examples/test_aclnn_cdist.cpp) | 通过[aclnnCdist](./docs/aclnnCdist.md)接口方式调用Cdist算子。 |
