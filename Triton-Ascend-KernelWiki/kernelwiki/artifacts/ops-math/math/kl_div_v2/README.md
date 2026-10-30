# aclnnKlDivV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

* 算子功能：计算KL散度
* 计算公式：
  * 定义loss_pointwise，保存中间结果。

    $$
    loss\_pointwise_i=\begin{cases}
    NaN & \text{ if }&logTarget=false \text{ and } target_i <= 0,  \\
    target_i * \left ( \log{(target_i)}- self_i  \right )  & \text{ if }& logTarget=false， \\
    \exp^ {target_i} * \left ( target_i- self_i \right )  & \text{ else. }
    \end{cases}
    $$

  * out计算公式为：

    $$
    out=\begin{cases}
    \bar{loss\_pointwise}  & \text{ if }& reduction= 1, \\
    \sum loss\_pointwise & \text{ elif }& reduction= 2,\\
    \frac{\sum loss\_pointwise}{self.size(0)} & \text{ elif }& reduction= 3,\\
    loss\_pointwise & \text{ else. }
    \end{cases}
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
      <td>target</td>
      <td>输入</td>
      <td>公式中的输入张量x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>可选属性</td>
      <td><ul><li>公式中的reduction。</li><li>默认值为mean。</li></ul></td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_target</td>
      <td>可选属性</td>
      <td><ul><li>公式中的logTarget。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_kl_div_v2](./examples/test_aclnn_kl_div_v2.cpp) | 通过[aclnnKlDiv](./docs/aclnnKlDiv.md)接口方式调用KlDivV2算子。    |
| 图模式调用 | [test_geir_kl_div_v2](./examples/test_geir_kl_div_v2.cpp)   | 通过[算子IR](./op_graph/kl_div_v2_proto.h)构图方式调用KlDivV2算子。 |
