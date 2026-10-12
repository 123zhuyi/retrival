# Mod

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Admin05210 | 个人开发者 | Mod | 2026/07/16 | Mod算子增强：新增INT16同/混合数据类型计算支持、大商场景数值稳定性增强、连续核与广播路径优化 |

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：返回 self 除以 other 的余数。

- 计算公式：

  对于入参 self 和比较标量 other，Fmod 可以用如下数学公式表示：

  $$
  out_{i} = self_{i} - (other \times trunc(self_{i}/other))
  $$

- 精度说明：针对 self/other 商值较大（大 \|self/other\|）的场景，AICore 计算路径引入了数值稳定性增强算法，相比朴素截断取余（trunc-mod）实现降低了大商场景下的精度损失风险；该增强与 INT16/混合数据类型能力一并限定于 Atlas A2/A3（其余产品的既有算法与精度行为不变）。

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
      <td>输入</td>
      <td>待进行mod计算的入参，公式中的self_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32、INT16*</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行mod计算的入参，公式中的other。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32、INT16*</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行mod计算的出参，公式中的out_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32、INT16*</td>
      <td>ND</td>
    </tr>
  </tbody></table>

\* INT16 同数据类型计算，以及 self/other 分别为 INT16 与 BFLOAT16/FLOAT16/FLOAT32 的混合数据类型计算，仅 Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品支持；其余产品不适用（BFLOAT16/FLOAT16/FLOAT32/INT32 的既有支持不受影响）。

## 约束说明

1. aclnn 层支持 DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、INT16 类型推导；AICore kernel 覆盖 BFLOAT16、FLOAT16、FLOAT32、INT32，其余类型走 AICPU fallback。其中 **INT16 同数据类型计算、以及 INT16 与 BFLOAT16/FLOAT16/FLOAT32 的混合数据类型计算为 Atlas A2/Atlas A3 专属增强**（由 AICore 支持）；其余产品上该增强不适用，已有的 BFLOAT16/FLOAT16/FLOAT32/INT32 同数据类型计算及 DOUBLE/INT64/INT8/UINT8 的 AICPU 回退行为保持不变。
2. self和out的shape必须一致。
3. 数据维度不支持8维以上。

## 调用说明

| 调用方式 | 样例代码  | 说明  |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn接口 | [test_aclnn_fmod_scalar](examples/test_aclnn_fmod_scalar.cpp) | 通过[aclnnFmodScalar](docs/aclnnFmodScalar&aclnnInplaceFmodScalar.md)接口方式调用Mod算子。 |
| aclnn接口 | [test_aclnn_fmod_tensor](examples/test_aclnn_fmod_tensor.cpp) | 通过[aclnnFmodTensor](docs/aclnnFmodTensor&aclnnInplaceFmodTensor.md)接口方式调用Mod算子。 |
