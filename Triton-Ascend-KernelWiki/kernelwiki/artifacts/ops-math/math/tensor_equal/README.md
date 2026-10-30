# TensorEqual

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

- 接口功能：计算两个Tensor是否有相同的大小和元素，返回一个Bool类型。
- 计算表达式：

  $$
  out = (self == other)  ?  True : False
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1549px"><colgroup>
  <col style="width: 168px">
  <col style="width: 136px">
  <col style="width: 258px">
  <col style="width: 271px">
  <col style="width: 311px">
  <col style="width: 116px">
  <col style="width: 142px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示第一个输入。</td>
      <td>self与other的数据类型满足数据类型推导规则（参见<a href="../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、DOUBLE、INT64、INT16、UINT16、UINT32、UINT64、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>表示第二个输入。</td>
      <td>other与self的数据类型满足数据类型推导规则（参见<a href="../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8、BOOL、DOUBLE、INT64、INT16、UINT16、UINT32、UINT64、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示输出。输出一个数据类型为BOOL，一维包含一个元素的Tensor。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT、DOUBLE。

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码                                                | 说明                                                             |
|--------------|-----------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_equal.cpp](examples/test_aclnn_equal.cpp) | 通过[aclnnEqual](docs/aclnnEqual.md)接口方式调用TensorEqual算子。 |
