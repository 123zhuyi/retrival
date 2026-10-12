# AtanGrad

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：计算反正切函数（atan）的输入梯度，用于神经网络反向传播中的梯度传递。
- 计算公式：

  $$
  dx_i = dy_i \times \frac{1}{1 + x_i^2}
  $$

  其中 $x$ 为前向计算的输入张量（atan 函数自变量），$dy$ 为上游传入的梯度张量，$dx$ 为输出的输入梯度张量。

- 等效分步计算：
  1. $t_i = x_i \times x_i$（计算 $x^2$）
  2. $g_i = t_i + 1.0$（计算 $1 + x^2$）
  3. $r_i = 1 / g_i$（取倒数，即 atan 函数的导数）
  4. $dx_i = dy_i \times r_i$（乘以上游梯度）

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
    <td>x</td>
    <td>输入</td>
    <td>前向计算输入张量，对应公式中 x，为 atan 函数自变量。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>输入</td>
    <td>上游传入的梯度张量，对应公式中 dy。数据类型须与 x 完全一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dx</td>
    <td>输出</td>
    <td>输出的输入梯度张量，对应公式中 dx。数据类型须与 x 完全一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- aclnnAtanGrad 默认确定性实现。
- x、dy、dx 三者数据类型必须完全一致，不支持隐式类型转换。
- x、dy、dx 三者 shape 必须完全相同，不支持广播（broadcast）。
- 支持空 Tensor（元素个数为 0）。
- 支持 0-8 维 Tensor，0 维表示标量（scalar），此时 dy 和 dx 也必须为 0 维。
- 当 x 取值极大（如 fp16 最大值）时，$x^2$ 可能溢出为 inf，此时 $1/\text{inf}=0$，dx=0，属于正常数值行为。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用</td>
    <td><a href="examples/arch35/test_aclnn_atan_grad.cpp">test_aclnn_atan_grad</a></td>
    <td>调用前需完成自定义算子包的编译与安装（bash build.sh --soc=ascend910b）。</td>
  </tr>
</tbody></table>

## 参考资源

接口文档和设计文档待补充。
