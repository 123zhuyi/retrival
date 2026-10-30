# Atan

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入的每个元素进行反正切操作后输出。

- 计算公式：

  $$
  y_{i}=tan^{-1}(x_{i})
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>公式中的输入张量x。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、BOOL、FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR/Ascend 950DT</term>：支持FLOAT、FLOAT16、BFLOAT16。
- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：不支持BFLOAT16。

## 约束说明

- 输入类型为INT8、INT16、INT32、INT64、UINT8、BOOL时，转化为FLOAT32进行运算，输出FLOAT32类型。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，非连续的Tensor维度不大于8，且输入与输出的shape需要一致。

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_Atan](./examples/test_aclnn_atan.cpp) | 通过[aclnnAtan](./docs/aclnnAtan&aclnnInplaceAtan.md)接口方式调用Atan算子。  |
| 图模式调用 | [test_geir_Atan](./examples/test_geir_atan.cpp)   | 通过[算子IR](./op_graph/atan_proto.h)构图方式调用Atan算子。                                      |
