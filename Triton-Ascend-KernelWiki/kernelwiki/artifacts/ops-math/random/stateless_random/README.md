# StatelessRandom

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                             |    ×     |

## 功能说明

- 算子功能：返回从[from, to - 1]范围中抽取离散均匀分布的随机数。
- 计算公式：
  $$
  output[i] = X \% (to - from) + from
  $$
  其中X为Philox随机数生成器产生的uint32随机数。

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
      <td>shape</td>
      <td>输入</td>
      <td>输出张量的形状维度。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>随机数种子。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>随机数偏移量，必须是4的倍数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>from</td>
      <td>输入</td>
      <td>可选参数，随机数范围的下界。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>to</td>
      <td>输入</td>
      <td>可选参数，随机数范围的上界。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，包含指定范围内的随机值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT64、INT32、INT16、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>输出张量的数据类型。</td>
      <td>DataType</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- offset是4的倍数。
- 输出张量维度支持0~8维
- from必须小于to, from/to-1指定范围不可超出self数据类型的范围。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn接口 | [test_aclnn_inplace_random](../dsa_random_uniform/examples/test_aclnn_inplace_random.cpp) | 通过[aclnnInplaceRandom](../dsa_random_uniform/docs/aclnnInplaceRandom.md)或[aclnnInplaceRandomTensor](../dsa_random_uniform/docs/aclnnInplaceRandomTensor.md)接口方式调用StatelessRandom算子。 |
| 图模式调用 | [test_geir_stateless_random](./examples/arch35/test_geir_stateless_random.cpp)   | 通过[算子IR](./op_graph/stateless_random_proto.h)构图方式调用StatelessRandom算子。 |
