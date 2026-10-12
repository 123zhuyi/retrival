# CosineSimilarity

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
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
      <td>input_x1</td>
      <td>输入</td>
      <td>第一个输入张量，rank范围为1~8。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_x2</td>
      <td>输入</td>
      <td>第二个输入张量，必须与input_x1的shape完全相同。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td>沿dim维度reduce后的输出张量，shape为输入shape删除dim对应维后的结果。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>计算余弦相似度的维度，取值范围为[-rank, rank-1]，默认值为1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>非负的数值稳定性参数，防止除零，默认值为1e-8。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入和输出仅支持FLOAT（float32）数据类型。
- 两个输入的rank必须相同，且rank范围为1~8；不支持0D Tensor。
- 两个输入的每一维长度必须相同，即`input_x1.shape == input_x2.shape`。
- 不支持任何形式的广播，包括维度为1的扩展和不同rank的左侧补1。
- 不支持shape中含长度为0的维度。
- `dim`必须位于输入rank对应的`[-rank, rank-1]`范围内；负数索引按对应正数维度处理，例如`dim=-1`表示最后一维。
- 输出shape为两个输入的共同shape删除`dim`对应维后的结果。
- `eps`必须为非负值。
- 若两个输入的rank为1，则`dim`必须显示指定在`[-1, 0]`范围内。

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| 图模式调用 | [test_geir_cosine_similarity](./examples/test_geir_cosine_similarity.cpp) | 通过[算子IR](./op_graph/cosine_similarity_proto.h)构图方式调用CosineSimilarity算子 |
