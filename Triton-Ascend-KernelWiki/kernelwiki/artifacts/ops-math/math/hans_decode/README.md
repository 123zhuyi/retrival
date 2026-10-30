# HansDecode

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对压缩后的张量基于PDF进行解码，同时基于mantissa（尾数）重组恢复张量。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 310px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>mantissa</td>
      <td>输入</td>
      <td>表示输入的待解压张量的尾数部分。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pdf</td>
      <td>输入</td>
      <td>表示压缩时采用的指数位所在字节的概率密度分布，shape要求为(1, 256)。其中每一个元素的值表示其对应的索引，在input中出现的次数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>fixed</td>
      <td>输入</td>
      <td>表示压缩的第一段输出。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输入</td>
      <td>表示压缩时超过fixed空间后的部分。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示解压缩后的指数存放位置。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reshuff</td>
      <td>可选属性</td>
      <td><ul><li>表示是否在连续内存上进行重新填充。</li><li>默认值为false。</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_hans_decode](./examples/test_aclnn_hans_decode.cpp) | 通过[aclnnHansDecode](./docs/aclnnHansDecode.md)接口方式调用HansDecode算子。    |
