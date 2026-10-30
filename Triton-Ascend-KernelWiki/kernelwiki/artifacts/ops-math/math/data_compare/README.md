# DataCompare

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

- 算子功能：逐元素比较两个相同shape和dtype的输入张量`x1`和`x2`，统计差异超出容差范围的元素总个数（All Reduce），输出float32标量表示不匹配元素总数。
- 计算公式：

  逐元素判断是否不匹配：

  $$
  diff_i = |x1_i - x2_i|
  $$

  $$
  threshold_i = atol + rtol \times |x2_i|
  $$

  $$
  mismatch_i = \begin{cases} 1.0, & diff_i > threshold_i \\ 0.0, & diff_i \leq threshold_i \end{cases}
  $$

  归约输出（不匹配元素总数）：

  $$
  output = \sum_{i} mismatch_i \quad (\text{dtype: float32})
  $$

- 输出为`0`：所有元素均在容差范围内，两个张量视为匹配。
- 输出`> 0`：存在差异超差的元素，值越大说明差异越大。
- 整数类型特殊处理：当输入为int8/uint8/int32时，算子内部先将输入cast到float32再执行比较计算，避免atol/rtol截断为0导致误判。

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
    <td>第一个输入张量，对应公式中的x1。shape和dtype必须与x2完全一致，不支持broadcast。支持0-8维张量，支持空Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>第二个输入张量，对应公式中的x2。shape和dtype必须与x1完全一致，不支持broadcast。支持0-8维张量，支持空Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>atol</td>
    <td>属性</td>
    <td><ul><li>绝对容差，对应公式中的atol。</li><li>需>=0。</li><li>默认值为1e-5。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rtol</td>
    <td>属性</td>
    <td><ul><li>相对容差，对应公式中的rtol。</li><li>需>=0。</li><li>默认值为1e-3。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>num</td>
    <td>输出</td>
    <td>不匹配元素总数，对应公式中的output。固定为float32标量（0维张量）。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- x1和x2的shape必须完全相同，不支持broadcast。
- x1和x2的dtype必须完全相同，不支持混合dtype。
- 整数类型（INT8/UINT8/INT32）输入时，算子内部先cast到FLOAT再执行比较计算。
- 归约方式为All Reduce（对所有轴归约），用户不可指定axis。
- 输出固定为FLOAT标量。
- 确定性实现：相同输入保证产生相同输出。
- 当输入元素总数超过2^24（约16M）时，由于float32精确表示范围限制，计数值可能不精确。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_data_compare](./examples/arch35/test_geir_data_compare.cpp)   | 通过[算子IR](./op_graph/data_compare_proto.h)构图方式调用DataCompare算子。
