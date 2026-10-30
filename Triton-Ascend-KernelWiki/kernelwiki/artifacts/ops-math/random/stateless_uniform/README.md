# StatelessUniform

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

- 算子功能：基于Philox4x32-10伪随机数生成算法，生成服从均匀分布的随机数张量，随机值范围为[from, to)。

- 计算公式：

  $$
  y_i = u_i \times (to - from) + from, \quad u_i \sim \text{Uniform}(0, 1]
  $$

  其中 $u_i$ 由Philox4x32-10算法生成，归一化方式与竞品curand_uniform一致：$u = x \times 2^{-32} + 2^{-33}$。

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
      <td>输出张量的形状，1-D tensor。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>Philox算法的随机数种子，0-D标量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>Philox算法的偏移量，0-D标量。必须是4的倍数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>from</td>
      <td>输入</td>
      <td>均匀分布随机范围的下界（包含），0-D标量。</td>
      <td>DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>to</td>
      <td>输入</td>
      <td>均匀分布随机范围的上界（不包含），0-D标量。</td>
      <td>DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出指定形状的均匀分布随机值，值域为[from, to)。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>输出数据类型，默认为FLOAT32。</td>
      <td>Type</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- offset必须是4的倍数。
- 输出张量维度支持0~8维。
- from必须小于等于to，且to - from不能超出输出数据类型的表示范围。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_stateless_uniform](./examples/arch35/test_aclnn_stateless_uniform.cpp) | 通过[aclnnInplaceUniform](../dsa_random_uniform/docs/aclnnInplaceUniform.md)接口方式调用StatelessUniform算子。 |
| 图模式调用 | [test_geir_stateless_uniform](./examples/test_geir_stateless_uniform.cpp) | 通过[算子IR](./op_graph/stateless_uniform_proto.h)构图方式调用StatelessUniform算子。 |
