# ApproximateEqual

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：逐元素判断两个输入张量是否近似相等。
- 计算公式：

  $$
  y_i = |x1_i - x2_i| < tolerance
  $$

  其中tolerance为非负浮点数，默认值为1e-5。输出为BOOL类型张量，元素值为true（近似相等）或false（不近似相等）。

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
    <td>公式中的输入x1。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>公式中的输入x2，数据类型和shape需与x1一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>tolerance</td>
    <td>可选属性</td>
    <td><ul><li>近似相等的判定阈值。</li><li>默认值为1e-5。</li><li>必须为非负有限浮点数。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>逐元素比较结果，shape与x1相同。</td>
    <td>BOOL</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- x1和x2的数据类型必须一致，不支持隐式类型转换。
- x1和x2的shape必须严格相等，不支持广播。
- tolerance必须为非负有限浮点数（不允许NaN或Inf）。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_approximate_equal.cpp">test_geir_approximate_equal</a></td>
    <td>通过<a href="./op_graph/approximate_equal_proto.h">算子IR</a>构图方式调用ApproximateEqual算子。</td>
  </tr>
</tbody></table>
