# Sign

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：对输入逐元素取符号，正数为1，负数为−1，0为0。
- 计算公式：

  $$
  \text{out}_{i}= 
  \begin{cases}
  1  & \text{if } \text{input}_{i}>0 \\[4pt]
  0  & \text{if } \text{input}_{i}=0 \\[4pt]
  -1 & \text{if } \text{input}_{i}<0
  \end{cases}
  $$

 **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1330px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>待取符号的Tensor。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>与input同形状的符号结果。</td>
      <td>数据类型、shape需与input一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16</td>
      <td>同input</td>
      <td>ND</td>
      <td>√</td>
    </tr>
   </tbody>
  </table>

## 约束说明

输入 input 和输出 output 的 Shape 必须严格保持一致。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sign](./examples/test_aclnn_sign.cpp) | 通过[aclnnSign](./docs/aclnnSign.md)接口方式调用Sign算子。 |    

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| hth810 | 个人开发者 | Sign | 2025/12/12 | Sign算子适配开源仓 |
| hth810 | 个人开发者 | Sign | 2026/5/12 | Sign算子添加int16支持 |
