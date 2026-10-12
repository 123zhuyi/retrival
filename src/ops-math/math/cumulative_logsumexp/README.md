# CumulativeLogsumexp

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：对输入张量 `x` 沿 `axis` 指定维度计算累积 log-sum-exp，并将结果保存到输出张量 `y` 中。
- 计算公式：设 $i$ 是 `axis` 维度上的下标，$S_i$ 是参与第 $i$ 个输出元素计算的前缀或后缀下标集合，则：

$$
y_i = \log \sum_{j \in S_i} e^{x_j}
$$

当 `exclusive` 为 `false` 时，$S_i$ 包含当前位置；当 `exclusive` 为 `true` 时，$S_i$ 不包含当前位置。当 `reverse` 为 `false` 时按下标递增方向累积；当 `reverse` 为 `true` 时按下标递减方向累积。

## 参数说明

<table>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入Tensor。shape支持1-6维，指定axis维度长度必须大于0。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>需要进行累积log-sum-exp的维度，必须为编译期常量标量，取值范围为[-rank(x), rank(x)-1]。</td>
      <td>INT32、INT64、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>属性</td>
      <td>默认值为false。false表示包含当前位置；true表示不包含当前位置。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reverse</td>
      <td>属性</td>
      <td>默认值为false。false表示正向累积；true表示反向累积。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出Tensor，shape与x相同。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `axis` 必须为常量标量。
- `x` 不支持标量输入；`axis` 指定维度长度为0时返回参数错误。
- `y` 的shape和数据类型需要与 `x` 一致。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| 图模式调用 | [test_geir_cumulative_logsumexp.cpp](./examples/test_geir_cumulative_logsumexp.cpp) | 通过算子IR构图方式调用CumulativeLogsumexp算子。 |
