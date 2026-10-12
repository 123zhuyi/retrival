# AddMatMatElements

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量`a`、`b`进行逐元素相乘后，与输入张量`c`按`alpha`、`beta`加权求和，结果写入输出张量。
- 计算公式：

  $$
  out_i = c_i \times \beta + \alpha \times a_i \times b_i
  $$

- 输出shape固定为`c.shape`。`a`、`b`分别按照尾部对齐规则广播到`c.shape`，不支持通过统一广播扩大`c`的shape。
- `alpha`、`beta`是shape为`(1,)`的Tensor，不是属性。

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
    <td>c</td>
    <td>输入</td>
    <td>输出shape的锚点，按beta缩放后参与累加，rank为1~8。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>a</td>
    <td>输入</td>
    <td>参与逐元素相乘，shape必须能够广播到c.shape。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>输入</td>
    <td>参与逐元素相乘，shape必须能够广播到c.shape。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>对c进行缩放的系数Tensor，shape必须为(1,)。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>输入</td>
    <td>对a × b的乘积进行缩放的系数Tensor，shape必须为(1,)。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>c</td>
    <td>输出</td>
    <td>计算结果，shape等于输入c.shape。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- `c`、`a`、`b`、`beta`、`alpha`及输出必须具有完全一致的数据类型，仅支持`FLOAT16`和`FLOAT`，不支持数据类型提升。
- `c`的rank取值范围为1~8；`a`、`b`的rank不能大于`c`的rank。
- `a`、`b`分别尾部对齐到`c.shape`；补齐后的每个维度必须为1或等于`c`的对应维度。
- 输出shape固定为`c.shape`。例如`c.shape=(1, 3)`、`a.shape=(2, 3)`不受支持，即使三输入能够得到统一广播shape`(2, 3)`。
- `alpha`、`beta`的shape必须严格为`(1,)`。
- 支持空Tensor，空Tensor场景仍需满足上述rank和广播约束。
- 支持动态shape和动态rank；运行时实际shape必须满足上述约束。
- 仅支持ND格式。输入输出配置了自动连续化，非连续Tensor由框架转换为连续Tensor后进入Kernel。

> 说明：当前InferShape实现按`broadcast(a, b, c)`进行形状推导，而Host Tiling以`c.shape`为输出锚点并校验`a`、`b`到`c.shape`的广播关系。实际执行约束以Host Tiling为准。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :------- | :------- | :--- |
| 图模式 | [test_geir_add_mat_mat_elements.cpp](examples/test_geir_add_mat_mat_elements.cpp) | 通过图模式调用AddMatMatElements算子。 |
