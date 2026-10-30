# AmpUpdateScale

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：实现AMP（Automatic Mixed Precision）训练中的动态Scale更新逻辑。根据当前scale值、growth tracker计数器以及是否发现Inf/NaN，动态调整loss scale大小。
- 计算公式：

  $$
  \text{updated\_scale} = \begin{cases}
  \text{current\_scale} \times \text{backoff\_factor} & \text{if found\_inf} \neq 0 \\
  \text{current\_scale} \times \text{growth\_factor} & \text{if growth\_tracker + 1 = growth\_interval and new\_scale is finite} \\
  \text{current\_scale} & \text{otherwise}
  \end{cases}
  $$

  $$
  \text{updated\_growth\_tracker} = \begin{cases}
  0 & \text{if found\_inf} \neq 0 \text{ or growth triggered} \\
  \text{growth\_tracker} + 1 & \text{otherwise}
  \end{cases}
  $$

  其中：
  - $\text{current\_scale}$：当前的loss scale值
  - $\text{found\_inf}$：是否检测到Inf/NaN的标志（0表示正常，非0表示发现Inf/NaN）
  - $\text{growth\_tracker}$：连续未出现Inf/NaN的步数计数器
  - $\text{growth\_factor}$：scale增长因子（通常为2.0）
  - $\text{backoff\_factor}$：scale回退因子（通常为0.5）
  - $\text{growth\_interval}$：触发scale增长的间隔步数

- 使用场景：AMP训练中的动态损失缩放（Dynamic Loss Scaling），用于在FP16/BF16混合精度训练中防止梯度下溢。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 170px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 220px">
  <col style="width: 160px">
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
      <td>current_scale</td>
      <td>输入</td>
      <td>当前的loss scale值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>growth_tracker</td>
      <td>输入</td>
      <td>连续未出现Inf/NaN的步数计数器。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>found_inf</td>
      <td>输入</td>
      <td>是否检测到Inf/NaN的标志，0表示正常，非0表示发现Inf/NaN。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>growth_factor</td>
      <td>属性</td>
      <td>scale增长因子，通常设置为2.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backoff_factor</td>
      <td>属性</td>
      <td>scale回退因子，通常设置为0.5。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>growth_interval</td>
      <td>属性</td>
      <td>触发scale增长的间隔步数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>updated_scale</td>
      <td>输出</td>
      <td>更新后的loss scale值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updated_growth_tracker</td>
      <td>输出</td>
      <td>更新后的growth tracker计数器。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- shape约束：所有输入输出张量均为标量，shape为 [1]。
- 数据类型约束：current_scale与found_inf、updated_scale的数据类型需一致。

## 约束说明

无
