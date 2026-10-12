# StatelessExponential

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：基于 Philox4x32-10 PRNG 生成服从参数为 `lambd` 的指数分布随机数，将输入张量 `self` 原地填充为服从 `Exp(lambd)` 分布的随机数。本算子是 `aclnnMultinomialTensor` 接口在 Ascend 950 场景下无放回采样路径使用的内部服务算子（生成指数分布随机扰动，供后续 `div + argmax/topk` 完成无放回多项采样）。
- 计算公式：

  $$
  x = -\frac{1}{\lambda} \ln(u), \quad u \sim U(0, 1]
  $$

  其中 `u` 由 Philox4x32-10 PRNG 生成，`lambd > 0` 为指数分布速率参数。

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
      <td>self</td>
      <td>输入/输出</td>
      <td>待填充的张量（原地写入），决定随机数生成总量与输出 dtype；≤8 维。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>随机数生成器的种子，影响生成的随机数序列。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>随机数生成器的偏移量，必须是 4 的倍数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>属性</td>
      <td>指数分布速率参数 λ，必须大于 0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

1. `self` 仅支持 FLOAT16/BFLOAT16/FLOAT，ND 格式，维度数 ≤8；非连续输入通过 AutoContiguous 转为连续。
2. `lambd` 必须 > 0。
3. `offset` 必须为 4 的倍数。
4. 仅支持 Ascend 950；不支持入图（op_graph）、不支持 L2 接口、不支持非连续 Tensor。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_multinomial_tensor](../stateless_sample_multinomial/examples/test_aclnn_multinomial_tensor.cpp) | 通过[aclnnMultinomialTensor](../stateless_sample_multinomial/docs/aclnnMultinomialTensor.md)接口构建计算流程时，内部调用StatelessExponential服务算子。 |
