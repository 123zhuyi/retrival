# Maximum

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：返回两个输入张量中的最大值组成的新张量。

- 计算公式：

$$out[i] = max(x_1[i], x_2[i])$$

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
      <td>x1</td>
      <td>输入</td>
      <td>待进行Maximum计算的入参，公式中的x1。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行Maximum计算的入参，公式中的x2。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>tensor逐元素取最大值结果，公式中的out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_maximum](./examples/test_aclnn_maximum.cpp) | 通过[aclnnMaximum](./docs/aclnnMaximum.md)接口方式调用aclnnMaximum算子。 |
| aclnn调用 | [test_aclnn_maxn](./examples/test_aclnn_maxn.cpp) | 通过[aclnnMaxN](./docs/aclnnMaxN.md)接口方式调用aclnnMaxN算子。 |
