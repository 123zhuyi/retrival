# Eltwise

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

- 算子功能：Eltwise 算子对 1~32 个同 shape、同 dtype 的输入张量执行逐元素操作，支持三种计算模式。

- 计算公式：

  - **mode=0 (PRODUCT)**：逐元素乘积

    $$out_i = x^{(0)}_i \times x^{(1)}_i \times \cdots \times x^{(n-1)}_i$$

  - **mode=1 (SUM)**：逐元素加权求和

    $$out_i = c_0 \cdot x^{(0)}_i + c_1 \cdot x^{(1)}_i + \cdots + c_{n-1} \cdot x^{(n-1)}_i$$

    其中 $c_k$ 为第 $k$ 个输入的加权系数（coeff），默认值为 1.0。

  - **mode=2 (MAX)**：逐元素取最大值

    $$out_i = \max(x^{(0)}_i, x^{(1)}_i, \ldots, x^{(n-1)}_i)$$

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
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>inputs</td>
    <td>输入</td>
    <td>输入张量列表，包含 1~32 个张量。所有张量必须同 shape、同 dtype。支持空 Tensor（0 元素）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>属性</td>
    <td><ul><li>计算模式。</li><li>0=PRODUCT（逐元素乘积），1=SUM（逐元素加权求和），2=MAX（逐元素取最大值）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>coeff</td>
    <td>可选属性</td>
    <td><ul><li>加权系数数组，仅 mode=1 时有效。</li><li>长度必须与 inputs 中张量个数一致。</li><li>可传空指针，此时默认所有系数为 1.0。</li><li>mode=0 和 mode=2 时忽略此参数。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>输出张量。shape 与输入张量相同，dtype 与输入张量相同。不支持空 Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 所有输入张量必须同 shape、同 dtype，不支持广播。
- 输入张量个数范围为 [1, 32]。
- 输出 dtype 和 shape 必须与输入完全一致。
- 仅支持 ND 数据格式，shape 维度范围：0~8。
- mode 仅支持 0（PRODUCT）、1（SUM）、2（MAX）三个取值。
- mode=1 时，若提供 coeff，其长度必须等于输入张量个数；若不提供（传空指针），默认系数为 1.0。
- 空 Tensor（0 元素）：正常接受，返回空 out，不进行计算。
- 确定性计算：是（逐元素操作，不涉及 Reduce，输出结果确定）。
