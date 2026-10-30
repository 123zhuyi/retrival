# Round

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

* 将输入张量的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。
* 将输入张量的元素四舍五入到指定的位数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>输入张量</td>
      <td>BFLOAT16、FLOAT16、 FLOAT32、 DOUBLE、 INT32、 INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>decimals</td>
      <td>属性</td>
      <td>一个可选的整数属性，指定要保留的小数位数。默认值为0</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量</td>
      <td>同输入张量x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* 当输入值在[-0.5, -0]之间时，输出值为0。
* 针对decimals不为0的场景：输入数据超过(-347000, 347000)范围，精度可能会有影响。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_round](./examples/test_aclnn_round.cpp) | 通过[aclnnRound](./docs/aclnnRound&aclnnInplaceRound.md)接口方式调用Round算子    |
| 图模式调用 | [test_geir_round](./examples/test_geir_round.cpp)   | 通过[算子IR](./op_graph/round_proto.h)构图方式调用Round算子 |
