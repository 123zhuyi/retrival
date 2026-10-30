# SelectV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：实现张量的条件选择功能，根据条件从两个输入张量中选择对应位置的元素。

- 计算公式：

$out_i=if(condition_i) ? self_i : other_i$

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
      <td>condition</td>
      <td>输入</td>
      <td>待进行select_v2计算的入参，公式中的condition。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行select_v2计算的入参，公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT16、UINT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行select_v2计算的入参，公式中的other。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT16、UINT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行select_v2计算的出参，公式中的out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT16、UINT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_s_where.cpp](./examples/test_aclnn_s_where.cpp) | 通过[test_aclnn_s_where](./docs/aclnnSWhere.md)接口方式调用SelectV2算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| infinity | 个人开发者 | SelectV2 | 2025/11/25 | SelectV2算子适配开源仓 |
