# CosineSimilarity

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

- 算子功能：计算两个输入张量沿指定维度的余弦相似度。余弦相似度衡量两个向量之间的夹角余弦值，值域为 [-1, 1]，常用于度量向量之间的相似程度。

- 计算公式：

$$
\text{cosine\_similarity}(x_1, x_2, \text{dim}, \text{eps}) = \frac{\sum(x_1 \cdot x_2, \text{dim})}{\max(\sqrt{\sum(x_1^2, \text{dim})}, \text{eps}) \cdot \max(\sqrt{\sum(x_2^2, \text{dim})}, \text{eps})}
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
      <td>input_x1</td>
      <td>输入</td>
      <td>第一个输入张量，任意维度。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_x2</td>
      <td>输入</td>
      <td>第二个输入张量，与input_x1同shape（或可广播）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td>沿dim维度reduce后的输出张量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>计算余弦相似度的维度，默认值为1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>数值稳定性参数，防止除零，默认值为1e-8。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量最大支持8维。
- 输入支持float32数据类型。
- 支持广播（x1和x2 shape不同时自动广播）。
- dim属性支持负数索引（如dim=-1表示最后一维）。

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| 图模式调用 | [test_geir_cosine_similarity](./examples/test_geir_cosine_similarity.cpp) | 通过[算子IR](./op_graph/cosine_similarity_proto.h)构图方式调用CosineSimilarity算子 |
