# Zeta

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                   |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    √     |
| Atlas 推理系列产品                          |    √     |
| Atlas 训练系列产品                          |    √     |

## 功能说明

- 算子功能：计算Hurwitz zeta函数，即 ζ(x, q) = Σ(n=0 to ∞) 1/(q+n)^x。

- 计算公式：

$$z = \zeta(x, q) = \sum_{n=0}^{\infty} \frac{1}{(q+n)^x}$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
<col style="width: 140px">
<col style="width: 140px">
<col style="width: 180px">
<col style="width: 213px">
<col style="width: 100px">
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
    <td>x</td>
    <td>输入</td>
    <td>输入张量，Hurwitz zeta函数的指数参数。</td>
    <td>FLOAT、DOUBLE</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>q</td>
    <td>输入</td>
    <td>输入张量，Hurwitz zeta函数的偏移参数，必须与x具有相同的数据类型。</td>
    <td>FLOAT、DOUBLE</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>z</td>
    <td>输出</td>
    <td>输出张量，Hurwitz zeta函数的计算结果，与x具有相同的类型和形状。</td>
    <td>FLOAT、DOUBLE</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- 输入x和q的数据类型必须一致。
- 输入x和q的数据大小必须一致（不支持broadcast）。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_zeta](examples/test_geir_zeta.cpp) | 通过[算子IR](op_graph/zeta_proto.h)接口方式调用Zeta算子。 |
