# SignBitsUnpack

## 产品支持情况

| 产品           | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √    |

## 功能说明

- 算子功能：对输入进行unpack。

当位置为1时取1.0，位置为0时取0.0

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
      <td>self</td>
      <td>输入</td>
      <td>待进行SignBitsUnpack计算的入参，公式中的x1。</td>
      <td>uint8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>参数</td>
      <td>reshape时输出张量的第一个维度</td>
      <td>int64</td>
      <td>1</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>参数</td>
      <td>决定输出的数据类型</td>
      <td>int64</td>
      <td>1</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行SignBitsUnpack计算的出参，公式中的输出。</td>
      <td>float16,float</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 调用样例                                                                  | 说明                                                                                     |
| --------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_sign_bits_unpack](./examples/test_aclnn_sign_bits_unpack.cpp) | 通过[aclnnSignBitsUnpack](./docs/aclnnSignBitsUnpack.md)接口方式调用SignBitsUnpack算子。 |

## 贡献说明

| 贡献者      | 贡献方     | 贡献算子       | 贡献时间  | 贡献内容                     |
| ----------- | ---------- | -------------- | --------- | ---------------------------- |
| ilovescrapy | 个人开发者 | SignBitsUnpack | 2026/4/21 | SignBitsUnpack算子适配开源仓 |
