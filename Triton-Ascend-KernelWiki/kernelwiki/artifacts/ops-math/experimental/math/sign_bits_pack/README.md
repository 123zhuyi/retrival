# SignBitsPack

## 产品支持情况

| 产品                                                               | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 |    √    |

## 功能说明

- 算子功能：将float16类型或者float32类型的输入的符号位打包为uint8。

## 算子原型

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
      <td>x</td>
      <td>输入</td>
      <td>入参x,计算的1D张量</td>
      <td>float16,float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>属性</td>
      <td>表示处理维度，reshape时输出张量的第一个维度</td>
      <td>int</td>
      <td>scale</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>出参y, 只支持二维, x元素个数不被8整除时为（x元素个数 // 8） + 1，在被8整除时为（x元素个数/8）</td>
      <td>uint8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 调用样例                                                      | 说明                                                                                        |
| --------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_sign_bits_pack](./examples/test_aclnn_sign_bits_pack.cpp) | 通过[aclnnSignBitsPack](./docs/aclnnSignBitsPack.md)接口方式调用SignBitsPack算子。 |

## 贡献说明

| 贡献者      | 贡献方     | 贡献算子 | 贡献时间  | 贡献内容               |
| ----------- | ---------- | -------- | --------- | ---------------------- |
| infinity | 个人开发者 | SignBitsPack | 2026/4/26 | SignBitsPack算子适配开源仓 |
