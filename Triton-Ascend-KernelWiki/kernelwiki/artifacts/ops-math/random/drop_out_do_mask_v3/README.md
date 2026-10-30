# DropOutDoMaskV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：训练过程中，按照概率keep_prob随机保留输入中的元素，并将输出按照1/keep_prob的比例放大。若mask对应比特位为1，则输入中相应的元素放大。若mask中比特位为0，则gradOutput相应的元素置零。特别地，若keep_prob为0，则将所有元素置为0；若keep_prob为1，则不改变gradOutput的元素。

- 计算公式：

$$
gradOutput_i=\begin{cases}
0,&mask_i=0 \\
\frac{1}{keep\_prob}self_i, &mask_i=1
\end{cases}
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
      <td>公式中的输入self，shape支持0-8维。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>bit类型并使用UINT8类型存储的mask数据，shape需要为(align(self的元素个数,128)/8)。</td>
      <td>UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keep_prob</td>
      <td>输入</td>
      <td>用于计算输出数据缩放比例的概率值，shape需要为(1)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的gradOutput，数据类型需要是self可转换的数据类型，shape需要与self一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. keep_prob的值必须在0和1之间。
2. mask的shape必须满足条件：$$\text{mask\_shape} = \frac{\text{align}(\text{num}(self), 128)}{8}$$
3. 数据维度支持0-8维。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式调用 | [test_geir_drop_out_do_mask_v3](./examples/test_geir_drop_out_do_mask_v3.cpp)   | 通过[算子IR](./op_graph/drop_out_do_mask_v3_proto.h)构图方式调用DropOutDoMaskV3算子。 |
