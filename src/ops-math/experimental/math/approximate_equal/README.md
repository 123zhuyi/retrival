# ApproximateEqual

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：逐元素判断两个输入张量是否近似相等。
- 计算公式：

  $$
  y_i = |x1_i - x2_i| < tolerance
  $$

  其中 tolerance 为非负浮点数，默认值为 1e-5。输出为 BOOL 类型张量，元素值为 true（近似相等）或 false（不近似相等）。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>x1</td>
    <td>输入</td>
    <td>公式中的输入 x1。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>公式中的输入 x2，数据类型和 shape 需与 x1 一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>tolerance</td>
    <td>可选属性</td>
    <td><ul><li>近似相等的判定阈值。</li><li>默认值为 1e-5。</li><li>必须为非负有限浮点数。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>逐元素比较结果，shape 与 x1 相同。</td>
    <td>BOOL</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- x1 和 x2 的数据类型必须一致，不支持隐式类型转换。
- x1 和 x2 的 shape 必须严格相等，不支持广播。
- tolerance 必须为非负有限浮点数（不允许 NaN 或 Inf）。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/arch35/test_aclnn_approximate_equal.cpp">test_aclnn_approximate_equal</a></td>
    <td>参见算子调用完成算子编译和验证。</td>
  </tr>
</tbody></table>
