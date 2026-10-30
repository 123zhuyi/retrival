# ClipByNormNoDivSum

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：执行四输入元素的ClipByNorm变体计算，最后不做除以norm的归一化，而是通过Select/Max链实现分段映射。公式描述如下：

- 计算公式：

$$y_i = \max(\text{select\_ones}(x_i \leq \text{greater\_zeros}_i, x_i, \text{sqrt}(\text{select\_ones}(x_i > \text{greater\_zeros}_i, x_i, \text{select\_ones}_i))), \text{maximum\_ones}_i)$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 230px">
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
      <td>第一个输入tensor，公式中的x。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>greater_zeros</td>
      <td>输入</td>
      <td>第二个输入tensor，比较阈值，公式中的greater_zeros。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>select_ones</td>
      <td>输入</td>
      <td>第三个输入tensor，条件选择备用值，公式中的select_ones。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maximum_ones</td>
      <td>输入</td>
      <td>第四个输入tensor，最大值裁剪边界，公式中的maximum_ones。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算结果tensor，公式中的y。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入输出的数据类型必须一致。
- 所有输入支持Broadcast语义，输出shape为四个输入broadcast后的shape。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|----------|------|
| 图模式调用 | [test_geir_clip_by_norm_no_div_sum](./examples/test_geir_clip_by_norm_no_div_sum.cpp) | 通过[算子IR](./op_graph/clip_by_norm_no_div_sum_proto.h)构图方式调用ClipByNormNoDivSum算子。 |
