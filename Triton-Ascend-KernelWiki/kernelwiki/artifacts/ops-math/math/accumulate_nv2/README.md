# AccumulateNv2

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：返回输入tensors列表中每个输入tensor依次做add求和。AccumulateNv2与AddN执行相同的运算，但不等待所有输入就绪后再开始求和，可在输入就绪时间不同时节省内存。

- 计算公式：

$$y = \sum_{i=0}^{N-1} x_i$$

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
      <td>x</td>
      <td>输入（动态输入）</td>
      <td>需要求和的输入tensor列表，公式中的x_i。各tensor的shape需满足broadcast关系。</td>
      <td>FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>求和结果tensor，公式中的y。</td>
      <td>FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>N</td>
      <td>属性（必填）</td>
      <td>输入tensor列表的大小。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：aclnnSum默认确定性实现。
- 输入tensors列表中各tensor的shape需满足broadcast关系，broadcast后的shape需与输出y的shape一致。
- 输入tensors与输出的数据类型需相同。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sum](./examples/test_aclnn_sum.cpp) | 通过[aclnnSum](./docs/aclnnSum.md)接口方式调用AccumulateNv2算子。 |
