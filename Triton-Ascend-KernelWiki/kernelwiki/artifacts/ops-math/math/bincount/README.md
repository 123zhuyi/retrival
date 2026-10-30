# Bincount

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

- 算子功能：计算非负整数数组中每个数的频率。minlength为输出tensor的最小size；当weights为空指针时，self中的self[i]每出现一次，则其频率加1，当weights不为空时，self[i]每出现一次，其频率加weights[i]，最后存放到out的self[i]+1位置上；因此out大小为(self的最大值+1)与minlength中的最大值。

- 计算公式：

  如果n是self在位置i上的值，如果指定了weights，则

  $$
  out[self_i] = out[self_i] + weights_i
  $$

  否则：

  $$
  out[self_i] = out[self_i] + 1
  $$

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
      <td>array</td>
      <td>输入</td>
      <td>输入tensor，公式中的self_i。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>指定输出tensor最小长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>输入tensor的权重，公式中的weights_i。</td>
      <td>FLOAT</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>bins</td>
      <td>输出</td>
      <td>输出tensor，公式中的out_i。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_bincount](./examples/test_aclnn_bincount.cpp) | 通过[aclnnBincount](./docs/aclnnBincount.md)接口方式调用Bincount算子。 |
