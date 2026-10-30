# Trace

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- |:----:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |  √   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |  √   |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |  √   |
| <term>Atlas 200I/500 A2 推理产品</term>                      |  ×   |
| <term>Atlas 推理系列产品 </term>                             |  √   |
| <term>Atlas 训练系列产品</term>                              |  √   |

## 功能说明

- 算子功能：
  Trace算子用于计算矩阵从左上角开始的主对角线元素的和。

- 计算公式：  

  $$
  out=sum(diag(self))
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1464px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 300px">
  <col style="width: 324px">
  <col style="width: 118px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示输入张量，公式中的输入self。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、INT32、INT64、INT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示输出张量，公式中的输出out。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、INT64</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| aclnn调用  | [test_aclnn_trace](./examples/arch35/test_aclnn_trace.cpp) | 通过[aclnnTrace](./docs/aclnnTrace.md)接口方式调用Trace算子。   |
| 图模式调用 | [test_geir_trace](./examples/test_geir_trace.cpp) | 通过[算子IR](./op_graph/trace_proto.h)构图方式调用Trace算子 |
