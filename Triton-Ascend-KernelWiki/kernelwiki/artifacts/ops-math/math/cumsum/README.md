# Cumsum

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中。
- 计算公式：$x\_{i}$是输入张量self中，从维度dim视角来看的某个元素（其它维度下标不变，只dim维度下标依次递增），$y\_{i}$是输出张量out中对应位置的元素，则：

$$
y_{i} = x_{1} + x_{2} + x_{3} + ...... + x_{i}
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
      <td>x</td>
      <td>输入</td>
      <td>输入Tensor。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
      FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>需要进行依次累加的维度。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>属性</td>
      <td>默认值为false，表示执行包含性累积求和（inclusive cumsum），即输出的第一个元素与输入的第一个元素相同；<br> 
      true表示执行排除性累计求和（exclusive cumsum），即输出的第一个元素为0，后续元素为输入的前缀和。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reverse</td>
      <td>属性</td>
      <td>默认值为false，表示从张量的开头向末尾进行累积求和（正向计算）；<br> 
      true表示从张量的末尾向开头进行累积求和（反向计算）。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>按指定维度累加后的输出Tensor。</td>
      <td>输入Tensor相同x</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn调用 | [test_aclnn_cumsum_v2](./examples/test_aclnn_cumsum_v2.cpp) | 通过[aclnnCumsumV2](./docs/aclnnCumsumV2.md)接口方式调用cumsum算子。    |
| 图模式调用 | [test_geir_cumsum](./examples/test_geir_cumsum.cpp)   | 通过[算子IR](./op_graph/cumsum_proto.h)构图方式调用cumsum算子。 |
