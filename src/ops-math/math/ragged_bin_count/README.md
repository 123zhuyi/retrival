# RaggedBinCount

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：RaggedBinCount按 `splits` 划分的不规则行（ragged row），统计每个bin的出现次数或权重和。设
  `B = numel(splits) - 1`、`M = size[0]`，第 `b` 行对应
  `values[splits[b]:splits[b + 1]]`，输出shape为 `[B, M]`。

  对满足 `0 <= values[i] < M` 的元素：`binary_output=false` 且 `weights` 为空时输出对应位置累加 `1.0`；
  `binary_output=false` 且 `weights` 非空时累加 `weights[i]`；`binary_output=true` 时出现过的bin
  输出 `1.0` 并忽略权重值。大于等于 `M` 的bin会被忽略，负bin属于非法输入。

  使用场景：对变长序列批次（ragged batch）逐行做直方图统计，如按样本统计词表命中次数或类别权重和。

- 计算公式：

  非二值（`binary_output = false`）：

$$
output_{b,m}
= \sum_{i=splits_b}^{splits_{b+1}-1} \mathbb{1}[values_i = m] \cdot w_i
$$

  其中 $w_i = weights_{flat}[i]$（`weights` 元素个数非0）或 $w_i = 1.0$（`weights` 元素个数为0）。

  二值（`binary_output = true`），忽略 `weights`：

$$
output_{b,m}
= \mathbb{1}\left[\exists\, i \in [splits_b,\ splits_{b+1}),\ values_i = m\right]
$$

  其中 $b \in [0, B)$、$m \in [0, M)$，$\mathbb{1}[\cdot]$ 为指示函数（成立取1，否则取0）。

## 参数说明

<table style="table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 150px">
  <col style="width: 160px">
  <col style="width: 650px">
  <col style="width: 180px">
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
      <td>splits</td>
      <td>输入</td>
      <td>严格1D row-splits tensor，元素数至少为2。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>0D、1D或2D bin index tensor，按连续元素顺序展平；各产品支持的rank见产品差异说明。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>1D且元素个数为1（即shape为<code>[1]</code>），<code>size[0]</code>为非负bin数量，dtype与values相同。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>0D、1D或2D权重tensor，必传。元素个数为0（空Tensor）时表示不使用权重，等价于全1；否则元素个数必须与values相同，按values展平后的顺序逐元素对应。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>binary_output</td>
      <td>属性</td>
      <td>是否输出二值结果，默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>严格2D，shape为<code>[numel(splits)-1, size[0]]</code>。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

### 产品差异说明

各支持产品的静态shape与动态shape均使用ND格式，支持的数据类型组合和其余shape约束一致，仅`values`的rank范围不同。

<table><thead>
  <tr>
    <th>产品</th>
    <th>values支持的rank</th>
  </tr></thead>
<tbody>
  <tr>
    <td><term>Ascend 950PR/Ascend 950DT</term></td>
    <td>1～2</td>
  </tr>
  <tr>
    <td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br>
      <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term><br>
      <term>Atlas 训练系列产品</term></td>
    <td>0～2</td>
  </tr>
</tbody></table>

## 约束说明

- 各输入的数据类型必须整组匹配下表之一，不允许跨行组合（`size` 的dtype必须与 `values` 相同）：

| splits | values | size | weights | output | 数据格式 |
| --- | --- | --- | --- | --- | --- |
| INT64 | INT32 | INT32 | FLOAT | FLOAT | ND |
| INT64 | INT64 | INT64 | FLOAT | FLOAT | ND |

- `splits[0]` 必须为0，`splits` 必须单调不降，`splits[-1]` 必须等于values元素数。
- `splits` 越界或不满足上述数据值约束，以及`values`含负数，均属于非法输入，不得依赖非法输入的输出内容。
- `weights` 自身维数不得超过2；在此范围内只按**元素个数**校验，不要求shape或维数与`values`相同：元素个数为0即表示不使用权重，非0时必须与`values`的元素个数相同（按展平顺序对应）。
- 空Tensor支持情况，按轴分别说明：
  - `size[0]` 为0：输出为空Tensor `[numel(splits)-1, 0]`，不下发有效计算，直接返回成功。
  - `values` 元素数为0：此时 `splits` 各元素必须全为0，输出各位置均为 `+0.0`。
  - `weights` 元素个数为0（如shape为`[0]`、`[0,3]`、`[2,0]`、`[0,0]`）：表示无权重，等价于全1，均合法。
  - `splits` **不支持**空Tensor，元素数必须至少为2，否则校验失败返回错误。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE图模式</td>
    <td><a href="./examples/test_geir_ragged_bin_count.cpp">test_geir_ragged_bin_count</a></td>
    <td>通过<a href="./op_graph/ragged_bin_count_proto.h">算子IR</a>构图方式调用RaggedBinCount算子。</td>
  </tr>
</tbody></table>

本算子不提供同名aclnn接口，仅支持上表的GE图模式调用。
