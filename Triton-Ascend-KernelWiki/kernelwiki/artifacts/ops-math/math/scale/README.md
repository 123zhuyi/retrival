# Scale

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |   √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |   ×     |

## 功能说明

- 算子功能：对输入Tensor进行scale和bias计算。若不输入bias，则 $y = x \cdot scale$；若输入bias，则 $y = x \cdot scale + bias$。
- 算子公式：

  若不输入bias，则

  $$
  y_i = x_i \cdot scale_i
  $$

  若输入bias，则

  $$
  y_i = x_i \cdot scale_i + bias_i
  $$

  其中：
  - $x_i$为输入Tensor；
  - $scale_i$为缩放因子Tensor，支持与x的broadcast；
  - $bias_i$为可选的偏置Tensor，shape与scale保持一致；
  - $y_i$为输出Tensor，shape与x一致。

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
      <td>算子输入的Tensor。支持空Tensor。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>缩放因子Tensor。数据类型需与x一致，shape满足broadcast要求。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>偏置Tensor。不为空时数据类型需与scale一致，shape与scale保持一致。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>指定进行scale的起始轴。取值范围[-x_rank, x_rank)。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_axes</td>
      <td>属性</td>
      <td>指定进行scale的轴长度。取值范围>=-1，-1表示从axis轴scale到最后一轴。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_from_blob</td>
      <td>属性</td>
      <td>True：使用numAxes + axis推导scale shape；False：从axis开始按scale的rank推导，忽略numAxes。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出Tensor。shape和数据类型与x一致。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x与y的shape必须完全一致。
- x、scale、bias、y的dtype必须一致（均为FLOAT16/FLOAT32/BFLOAT16之一）。
- bias不为空时，bias与scale的shape必须一致。
- scale的shape需满足broadcast规则（参见功能说明）。
- x和scale的shape维度不大于8。
- axis取值范围为[-x_rank, x_rank)。
- numAxes取值范围>=-1。
- 仅支持ND格式。
- 支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，非连续的Tensor维度不大于8。

## 调用说明

| 调用方式 | 调用样例                                              | 说明                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_scale](./examples/test_aclnn_scale.cpp) | 通过[aclnnScale](./docs/aclnnScale.md)接口方式调用Scale算子。 |
