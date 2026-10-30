# Sinkhorn

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：

  计算Sinkhorn距离，可以用于MoE模型中的专家路由。
- 计算公式：

  $$
  p=Sinkhorn(cost, tol)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 100px">
  <col style="width: 120px">
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
      <td>cost</td>
      <td>输入</td>
      <td>
        <ul>
          <li>表示成本张量，公式中的。<code>cost</code>，Device侧的aclTensor。</li>
          <li>输入为二维矩阵且列数不超过4096。</li>
          <li>支持<a href="../../docs/zh/context/non_contiguous_tensor.md">非连续的Tensor</a>。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>tol</td>
      <td>输入</td>
      <td>
        <ul>
          <li>计算Sinkhorn的误差。</li>
          <li>如果传入空指针，则tol取0.0001。</li>
        </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>输出</td>
      <td>
        <ul>
          <li>表示最优传输张量，公式中的<code>p</code>，Device侧的aclTensor。</li>
          <li>如果传入空指针，则tol取0.0001。</li>
          <li>shape维度为2，不支持<a href="../../docs/zh/context/non_contiguous_tensor.md">非连续的Tensor</a>。</li>
          <li>数据类型和shape与入参<code>cost</code>的数据类型和shape一致。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sinkhorn](./examples/test_aclnn_sinkhorn.cpp) | 通过[aclnnSinkhorn](./docs/aclnnSinkhorn.md)接口方式调用Sinkhorn算子。    |
