# AddMatMatElements

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |

## 功能说明

- 算子功能：对输入张量 `a`、`b` 进行逐元素相乘后，与输入张量 `c` 按标量 `alpha`、`beta` 进行加权求和，结果写入输出张量 `cOut`。

- 计算公式：

  $$
  cOut_i = c_i \times \beta + \alpha \times a_i \times b_i
  $$

  其中 `a`、`b`、`c`、`cOut` 具有相同的 shape 和 dtype，`alpha`、`beta` 为标量。

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 240px">
  <col style="width: 210px">
  <col style="width: 120px">
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
      <td>a</td>
      <td>输入</td>
      <td>公式中的输入张量 a，参与逐元素相乘。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>公式中的输入张量 b，与 a 同 shape、同 dtype。</td>
      <td>同 a</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输入</td>
      <td>公式中的输入张量 c，与 a 同 shape、同 dtype。</td>
      <td>同 a</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入（aclScalar）</td>
      <td>标量缩放系数，对 a × b 的乘积进行缩放。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入（aclScalar）</td>
      <td>标量缩放系数，对 c 进行缩放。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cOut</td>
      <td>输出</td>
      <td>公式中的输出张量 cOut，与 a 同 shape、同 dtype。</td>
      <td>同 a</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `a`、`b`、`c`、`cOut` 四个张量必须具有完全一致的 shape 和 dtype。
- 张量最大维度为 8。
- 仅支持 ND 格式，不支持私有格式。
- `alpha`、`beta` 不可为空指针。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | [test_aclnn_add_mat_mat_elements](./examples/arch35/test_aclnn_add_mat_mat_elements.cpp) | 通过 aclnnAddMatMatElements 两段式接口调用 AddMatMatElements 算子，覆盖 FP16 / FP32 / BF16 三种 dtype |
