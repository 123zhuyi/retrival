# FeedsRepeat

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：
  
  对于输入feeds，根据输入feeds_repeat_times,将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维padding到output_feeds_size的大小

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 885px"><colgroup>
<col style="width: 137.333333px">
<col style="width: 104.333333px">
<col style="width: 369.333333px">
<col style="width: 205.333333px">
<col style="width: 68.333333px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0pky">描述</th>
    <th class="tg-0pky">数据类型</th>
    <th class="tg-0pky">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">feeds</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">等待处理的对象。</td>
    <td class="tg-0pky">BFLOAT16、FLOAT16、FLOAT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">feeds_repeat_times</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">描述对feeds每行的repeat次数。</td>
    <td class="tg-0pky">INT32、INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">y</td>
    <td class="tg-0lax">输出</td>
    <td class="tg-0lax">repeat完成后的张量，dtype与feeds保持一致。</td>
    <td class="tg-0lax">BFLOAT16、FLOAT16、FLOAT32</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">output_feeds_size</td>
    <td class="tg-0lax">属性</td>
    <td class="tg-0lax">输出的第0维的数值等于该数值。</td>
    <td class="tg-0lax">INT</td>
    <td class="tg-0lax"></td>
  </tr>
</tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

在算子实现过程中，feeds_repeat_times需搬入ub并做精度转换和累加，故feeds_repeat_times数据规模不能超过ub大小的四分之一。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| geir接口 | [test_geir_feeds_repeat](./examples/test_geir_feeds_repeat.cpp) | 通过[feeds_repeat_proto](op_graph/feeds_repeat_proto.h)接口方式调用feeds_repeat算子。 |
