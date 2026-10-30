# Split

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：Split（切分）算子用于根据 sections 或 indices_or_sections 对输入 Tensor 进行切分.

- 计算公式：
    - 根据轴 axis 对输入张量进行等分或不等分切片
    - 输出为一个 TensorList，每个 Tensor 对应一个分段

- 等分模式（isEven = true）:当传入 sections = k 时，把输入沿指定轴平均切成 k 份。
- 非等分模式（isEven = false）:当传入数组 indices_or_sections = [a0, a1, …] 时，按照给定区间切分。

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
      <td>输入</td>
      <td>待进行split计算的入参，被切分的tensor。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices_or_sections</td>
      <td>输入</td>
      <td>只有一个输入时以其为均分数量进行均分，否则进行索引切分。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>指定切分轴，默认为-1时展成一维进行切分</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行split计算的出参，输出的tensorlist。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 索引切分输入限制十个以下
- 目前支持FLOAT和INT32

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_split](./examples/test_aclnn_split.cpp) | 通过aclnnSplit接口方式调用Split算子。 |
