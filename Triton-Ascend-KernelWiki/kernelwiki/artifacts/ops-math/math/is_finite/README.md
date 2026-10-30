# IsFinite

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Kirin X90 处理器系列产品</term>                         |    √     | 
| <term>Kirin 9030 处理器系列产品</term>                   | √ |

## 功能说明

- 算子功能：判断张量中哪些元素是有限数值，即不是inf、-inf或nan。
- 计算公式：
  $$
  y_i=(x_i != \pm inf) \: and \: (x_i!= nan)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>公式中的输入张量x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品: 不支持BFLOAT16。
- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_is_finite](./examples/test_aclnn_is_finite.cpp) | 通过[aclnnIsFinite](./docs/aclnnIsFinite.md)接口方式调用IsFinite算子。    |
| 图模式调用 | [test_geir_is_finite](./examples/test_geir_is_finite.cpp)   | 通过[算子IR](./op_graph/is_finite_proto.h)构图方式调用IsFinite算子。 |
