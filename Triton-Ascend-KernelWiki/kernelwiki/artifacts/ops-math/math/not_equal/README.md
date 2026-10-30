# NotEqual

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

* 算子功能：逐元素比较两个输入张量是否不相等
* 计算公式：

$$
\text{y}_i = 
\begin{cases}
True,&\text{if }x_{i, 1} \ne x_{i,2} \\
False,&\text{otherwise}
\end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>DOUBLE、FLOAT16、FLOAT、BFLOAT16、INT64、INT32、INT8、UINT8、BOOL、INT16、COMPLEX64、COMPLEX128、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_ne_tensor](./examples/test_aclnn_ne_tensor.cpp) | 通过[aclnnNeTensor](./docs/aclnnNeTensor&aclnnInplaceNeTensor.md)接口方式调用NotEqual算子    |
| 图模式调用 | [test_geir_not_equal](./examples/test_geir_not_equal.cpp)   | 通过[算子IR](./op_graph/not_equal_proto.h)构图方式调用NotEqual算子 |
