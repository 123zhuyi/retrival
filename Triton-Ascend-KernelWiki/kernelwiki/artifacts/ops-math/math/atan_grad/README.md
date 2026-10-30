# AtanGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算反正切函数（atan）的输入梯度，用于神经网络反向传播中的梯度传递。
- 计算公式（命名与op_graph/atan_grad_proto.h一致）：

  $$
  z_i = dy_i \times \frac{1}{1 + y_i^2}
  $$

  其中 $y$ 为前向计算的输入张量（atan函数自变量），$dy$ 为上游传入的梯度张量，$z$ 为输出的输入梯度张量。

- 等效分步计算（NPU实现采用`Div`替代`Reciprocal+Mul`，避免INTRINSIC模式精度不足）：
  1. $t_i = y_i \times y_i$（计算 $y^2$）
  2. $g_i = t_i + 1.0$（计算 $1 + y^2$）
  3. $z_i = dy_i / g_i$（直接除法，等价于 $dy_i \times (1 / (1+y^2))$）

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>y</td>
    <td>输入</td>
    <td>前向计算输入张量，对应公式中y，为atan函数自变量。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>输入</td>
    <td>上游传入的梯度张量，对应公式中dy。数据类型须与y完全一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>z</td>
    <td>输出</td>
    <td>输出的输入梯度张量，对应公式中z。数据类型须与y完全一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- y、dy、z三者数据类型必须完全一致，不支持隐式类型转换。
- y、dy、z三者shape必须完全相同，不支持广播（broadcast）。
- 支持空Tensor（元素个数为0）。
- 支持0-8维Tensor，0维表示标量（scalar），此时dy和z也必须为0维。
- 当y取值极大（如fp16最大值）时，$y^2$ 可能溢出为inf，此时 $1/\text{inf}=0$，z=0，属于正常数值行为。
- FP16/BF16输入：NPU kernel走 **FP16/BF16 → FP32中间计算 → FP16/BF16** 的Cast模式（CANN kernel默认实现约定）。FP32输入：直接FP32计算。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_atan_grad.cpp](examples/test_geir_atan_grad.cpp) | 通过图模式方式调用AtanGrad算子。 |
