# SquareSumAll

## 产品支持情况

| 产品                                                      | 是否支持 |
| :-------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                    |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                   |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

- 算子功能：对两个形状相同的输入张量`x1`、`x2`分别求全元素平方和，输出两个标量`y1`、`y2`。

- 计算公式：

$$
y_1=\sum_{i=0}^{N-1}x_{1,i}^{2},\qquad
y_2=\sum_{i=0}^{N-1}x_{2,i}^{2}
$$

其中 $N$ 为输入张量的元素总数。

## 参数说明

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 420px">
  <col style="width: 120px">
  <col style="width: 410px">
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
      <td>表示第一路待求平方和的张量，对应公式中的x1。</td>
      <td>FLOAT</td>
      <td>FRACTAL_Z、C1HWNCoC0、NC1HWC0、ND、NCHW、NHWC</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示第二路待求平方和的张量，对应公式中的x2。</td>
      <td>FLOAT</td>
      <td>FRACTAL_Z、C1HWNCoC0、NC1HWC0、ND、NCHW、NHWC</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td><ul><li>表示x1全部元素的平方和，对应公式中的y1。</li><li>输出为0维Tensor（rank为0），含1个FLOAT元素。</li></ul></td>
      <td>FLOAT</td>
      <td>FRACTAL_Z、C1HWNCoC0、NC1HWC0、ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td><ul><li>表示x2全部元素的平方和，对应公式中的y2。</li><li>数据类型、shape与输出y1保持一致。</li></ul></td>
      <td>FLOAT</td>
      <td>FRACTAL_Z、C1HWNCoC0、NC1HWC0、ND</td>
    </tr>
  </tbody></table>

### 产品差异说明

下表中的“输入格式→输出格式”同时适用于x1→y1和x2→y2，两路输入必须使用相同的数据格式。

| 产品 | 静态shape格式组合 | 动态shape格式组合 | rank限制 |
| :--- | :--- | :--- | :--- |
| <term>Ascend 950PR/Ascend 950DT</term> | ND→ND、NCHW→ND、NHWC→ND | ND→ND、NCHW→ND、NHWC→ND | ND输入的rank取值范围为[0, 8]；NCHW、NHWC输入的rank必须为4。 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term><br><term>Atlas 200I/500 A2 推理产品</term><br><term>Atlas 推理系列产品</term><br><term>Atlas 训练系列产品</term> | FRACTAL_Z→FRACTAL_Z、C1HWNCoC0→C1HWNCoC0、NC1HWC0→NC1HWC0、ND→ND | ND→ND | 输入的rank取值范围为[0, 8]。 |

## 约束说明

- x1与x2不支持空Tensor；rank为0时表示含1个元素的标量输入，rank大于0时每一维长度必须大于0。
- x1与x2的执行期rank及每一维长度必须完全相同，不支持广播。shape`[2, 3]`与`[6]`即使元素数相同也不合法。
- 本算子不做隐式数据类型转换与输入数据格式转换，四个参数的数据类型和数据格式必须满足产品差异说明中的合法组合。
- NaN、Inf及有限值平方上溢按FLOAT语义自然传播；两路累加器与两个输出互不污染，`x1`中的非有限值不会影响`y2`，反之亦然。
- 本算子仅提供GE图模式注册，构建配置为`aclnn_exclude`，不提供aclnn接口。

## 调用说明

| 调用方式   | 调用样例                                                                            | 说明                                                                                    |
| :--------- | :---------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| 图模式调用 | [test_geir_square_sum_all](./examples/arch35/test_geir_square_sum_all.cpp)          | 通过[算子IR](./op_graph/square_sum_all_proto.h)构图方式调用SquareSumAll算子。            |
