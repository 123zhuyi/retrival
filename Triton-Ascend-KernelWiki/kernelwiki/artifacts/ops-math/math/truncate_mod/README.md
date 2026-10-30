# TruncateMod

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：返回self除以other的余数。

- 计算公式：

  将张量self，和标量或张量other，进行广播成相同shape的张量后，TruncateMod可以用如下数学公式表示：

  $$
  out_{i} = self_{i} - trunc \left(\frac{self_{i}}{other_{i}}\right) * other_{i}
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
      <td>self</td>
      <td>输入</td>
      <td>待进行TruncateMod计算的入参，公式中的self_i。</td>
      <td>DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行TruncateMod计算的入参，公式中的other_i。</td>
      <td>DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行TruncateMod计算的出参，公式中的out_i。</td>
      <td>DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 数据类型需满足数据类型推导规则，推导后的数据类型需在支持的数据类型范围内。
- self和other的shape必须满足 [广播规则](../../docs/zh/context/broadcast_relationship.md#广播规则)。
- 数据维度不支持8维以上。

## 调用说明

| 调用方式 | 样例代码  | 说明  |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| geir接口 | [test_geir_truncate_mod](examples/test_geir_truncate_mod.cpp) | 通过geir接口方式调用TruncateMod算子。 |
