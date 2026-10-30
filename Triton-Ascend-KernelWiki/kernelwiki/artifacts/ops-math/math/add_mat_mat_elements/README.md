# AddMatMatElements

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

- 算子功能：对输入张量`a`、`b`进行逐元素相乘后，与输入张量`c`按标量`alpha`、`beta`进行加权求和，结果写入输出张量`cOut`。
- 计算公式：

  $$
  cOut_i = c_i \times \beta + \alpha \times a_i \times b_i
  $$

  其中`a`、`b`、`c`支持PyTorch addcmul / numpy风格的广播，`cOut`的shape等于broadcast(a, b, c)的统一shape，`alpha`、`beta`为float标量。

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
    <td>c</td>
    <td>输入</td>
    <td>公式中的输入张量c，按beta缩放后参与累加。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>a</td>
    <td>输入</td>
    <td>公式中的输入张量a，参与逐元素相乘。</td>
    <td>同c（shape支持与b/c广播）</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>输入</td>
    <td>公式中的输入张量b，与a进行逐元素相乘。</td>
    <td>同c（shape支持与a/c广播）</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入（1-element标量tensor）</td>
    <td>标量缩放系数，对c进行缩放。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>输入（1-element标量tensor）</td>
    <td>标量缩放系数，对a × b的乘积进行缩放。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cOut</td>
    <td>输出</td>
    <td>公式中的输出张量cOut，shape必须等于broadcast(a, b, c)。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- `a`、`b`、`c`、`cOut`四个张量必须具有完全一致的dtype。
- `a`、`b`、`c`支持PyTorch addcmul / numpy风格的广播（尾部对齐、size=1维度扩展）。
- `cOut`的shape必须等于broadcast(a, b, c)的统一shape，否则参数非法。
- 张量最大维度为8。
- 仅支持ND格式，不支持私有格式。
- `alpha`、`beta`不可为空指针。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_add_mat_mat_elements.cpp](examples/test_geir_add_mat_mat_elements.cpp) | 通过图模式方式调用AddMatMatElements算子。 |
