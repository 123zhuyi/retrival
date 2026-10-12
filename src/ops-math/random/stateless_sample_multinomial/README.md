# StatelessSampleMultinomial

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

- 算子功能：本算子是`aclnnMultinomialGetWorkspaceSize`和`aclnnMultinomialTensorGetWorkspaceSize`接口构建计算流程时使用的内部服务算子，用于Ascend 950场景下的有放回多项分布采样路径。算子根据输入的累积分布和随机种子，从每个多项分布中抽取`num_samples`个样本，并将样本的类别索引存储到输出张量中。
- 计算公式：

  对于第$d$个分布的第$j$次抽样，生成随机数：

  $$
  u_{d,j} \sim U(0, 1]
  $$

  输出满足如下条件的最小类别索引$k$：

  $$
  x_{d,k-1} < u_{d,j} \le x_{d,k}
  $$

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
      <td>x</td>
      <td>输入</td>
      <td>输入累积分布张量，shape为(C)或(N, C)，最后一维表示类别的累积概率。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>norm_probs</td>
      <td>可选输入</td>
      <td>归一化概率张量，形状和数据类型与x相同。传入时用于采样结果的零概率类别回退。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
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
      <td>随机数生成器的偏移量，影响生成的随机数序列的位置。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_samples</td>
      <td>属性</td>
      <td>从每个多项分布中抽取的样本数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出样本的类别索引，shape为(num_samples)或(N, num_samples)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. `x`为一维或二维张量，最后一维为类别维，其长度不超过$2^{24}$。
2. `norm_probs`为可选输入。传入时，其形状和数据类型与`x`一致，`x`为`norm_probs`沿最后一维计算累加和的结果；缺省时，通过比较`x`的相邻元素进行零概率类别回退。
3. `num_samples`为正整数。
4. `offset`为4的整数倍。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_multinomial](./examples/test_aclnn_multinomial.cpp) | 通过[aclnnMultinomial](./docs/aclnnMultinomial.md)接口构建计算流程时，内部调用StatelessSampleMultinomial服务算子。 |
| aclnn接口 | [test_aclnn_multinomial_tensor](./examples/test_aclnn_multinomial_tensor.cpp) | 通过[aclnnMultinomialTensor](./docs/aclnnMultinomialTensor.md)接口构建计算流程时，内部调用StatelessSampleMultinomial服务算子。 |
