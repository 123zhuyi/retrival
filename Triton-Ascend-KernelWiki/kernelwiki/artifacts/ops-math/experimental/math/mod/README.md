# Mod

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
      <td>self</td>
      <td>输入</td>
      <td>待进行mod计算的入参，公式中的self_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行mod计算的入参，公式中的other。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行mod计算的出参，公式中的out_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. aclnn 层支持 DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8 类型推导；AICore kernel 覆盖 BFLOAT16、FLOAT16、FLOAT32、INT32，其余类型走 AICPU fallback。
2. self和out的shape必须一致。
3. 数据维度不支持8维以上。

## 调用说明

| 调用方式 | 样例代码  | 说明  |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn接口 | [test_aclnn_fmod_scalar](examples/test_aclnn_fmod_scalar.cpp) | 通过[aclnnFmodScalar](docs/aclnnFmodScalar&aclnnInplaceFmodScalar.md)接口方式调用Mod算子。 |
| aclnn接口 | [test_aclnn_fmod_tensor](examples/test_aclnn_fmod_tensor.cpp) | 通过[aclnnFmodTensor](docs/aclnnFmodTensor&aclnnInplaceFmodTensor.md)接口方式调用Mod算子。 |
