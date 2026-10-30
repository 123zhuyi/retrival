# AsStrided

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：允许用户通过制定新的形状（size）和步长（stride）来创建一个与原张量共享相同数据内存的张量视图。

- 计算公式：

$$
out_i=input_{\text{storage\_offset}+\sum_{d=0}^{D-1}(i_d\cdot \text{strided}[d])}
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的input_i。</td>
      <td>INT64、UINT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BF16、INT16、UINT16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出张量的形状。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>stride[d]是输入张量在第d维的步幅。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>storage_offset</td>
      <td>输入</td>
      <td>是out_i中相对于原张量input_i存储的偏移量。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的out_i。</td>
      <td>INT64、UINT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BF16、INT16、UINT16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| 图模式调用 | [test_geir_as_strided](./examples/test_geir_as_strided.cpp)   | 通过[算子IR](./op_graph/as_strided_proto.h)构图方式调用as_strided算子
