# stateless_bernoulli

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

- 算子功能：
    从伯努利分布中提取二进制随机数（0或1），prob为生成二进制随机数的概率，输入的张量用于指定shape。

- 计算公式：

  $$
  out∼Bernoulli(prob)
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
      <td>随机数的shape。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prob</td>
      <td>输入</td>
      <td>保活系数。</td>
      <td>FLOAT16、BF16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>获取随机种子。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>获取值的步长。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>获取输出的tensor。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_bernoulli_tensor](./examples/arch35/test_aclnn_bernoulli_tensor.cpp) | 通过[aclnnBernoulliTensor](./docs/aclnnBernoulliTensor&aclnnInplaceBernoulliTensor.md)接口方式调用stateless_bernoulli算子。 |
| aclnn调用 | [test_aclnn_bernoulli](./examples/arch35/test_aclnn_bernoulli.cpp) | 通过[aclnnBernoulli](./docs/aclnnBernoulli&aclnnInplaceBernoulli.md)接口方式调用stateless_bernoulli算子。 |
