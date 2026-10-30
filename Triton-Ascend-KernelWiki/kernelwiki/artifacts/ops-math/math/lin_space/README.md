# LinSpace

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Kirin X90 处理器系列产品</term>                      | √ | 
| <term>Kirin 9030 处理器系列产品</term>                       | √ |

## 功能说明

- 算子功能：生成一个等间隔数值序列。创建一个大小为num的1维向量，其值从start起始到stop结束（包含）线性均匀分布。

- 计算公式：

$$
output = (start, start + \frac{stop - start}{num - 1},...,start + (num - 2) * \frac{stop - start}{num -1}, stop)
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
      <td>start</td>
      <td>输入</td>
      <td>公式中的输入张量start，必须为1维张量。</td>
      <td>INT8、UINT8、INT16、INT32、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stop</td>
      <td>输入</td>
      <td>公式中的输入张量stop，必须为1维张量。</td>
      <td>INT8、UINT8、INT16、INT32、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num</td>
      <td>输入</td>
      <td>公式中的输入张量num，必须为1维张量。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>公式中的输出张量output。</td>
      <td>INT8、UINT8、INT16、INT32、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品、Atlas 训练系列产品、Atlas 推理系列产品：不支持BFLOAT16、INT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_lin_space](./examples/test_aclnn_lin_space.cpp) | 通过[aclnnLinspace](./docs/aclnnLinspace.md)接口方式调用LinSpace算子。 |
| 图模式调用 | [test_geir_lin_space](./examples/test_geir_lin_space.cpp)   | 通过[算子IR](./op_graph/lin_space_proto.h)构图方式调用LinSpace算子。 |
