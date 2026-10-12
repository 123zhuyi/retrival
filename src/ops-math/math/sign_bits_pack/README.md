# SignBitsPack

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将浮点数的符号位提取并打包为uint8字节。每8个浮点数的符号位（0表示非负，1表示负数）打包为1个uint8字节，其中第1个浮点数的符号位对应bit7，第8个对应bit0。主要用于1-bit Adam优化器场景，将符号信息压缩存储。
- 计算公式：

$$
b_i = \begin{cases} 0 & x_i \geq 0 \\ 1 & x_i < 0 \end{cases}
$$

$$
y_j = \sum_{k=0}^{7} b_{8j+k} \cdot 2^{7-k}
$$

$$
\text{out} = \mathrm{reshape}\!\left(\{y_j\},\; (\text{size},\; L\,/\,\text{size})\right)
$$

其中$L = \lceil N / 8 \rceil$，$N$为输入元素数。当$N$不是8的倍数时，尾部用$-1.0$填充（符号位为1）。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>x</td>
      <td>输入</td>
      <td>待打包的浮点张量，1D，提取每个元素的符号位。非连续tensor会被自动转为连续。</td>
      <td>FLOAT16/FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>必选属性</td>
      <td><ul><li>输出第一维大小。</li><li>取值必须为正整数（size &ge; 1）。</li><li>须满足ceil(N/8) % size == 0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>符号位打包结果，2D，shape为[size, ceil(N/8)/size]，dtype恒为uint8。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 数据格式仅支持ND。
- 输入x必须为1D（rank=1），不支持标量或多维tensor。
- 类型组合固定为FLOAT16→UINT8或FLOAT→UINT8，不支持其他dtype组合。
- 属性size必须为正整数（size &ge; 1），且ceil(N/8)必须能被size整除。
- 支持空Tensor（N=0），直接返回shape为[size, 0]的空输出。
- 支持非连续tensor，框架自动转为连续后执行。
- +0和-0均视为非负（符号位为0）；NaN不支持，行为未定义。
- 默认确定性实现，相同输入始终产生相同输出。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 单算子调用 | [test_aclnn_sign_bits_pack](./examples/test_aclnn_sign_bits_pack.cpp) | 通过aclnn接口调用SignBitsPack算子。 |
| 图模式调用 | [test_geir_sign_bits_pack](./examples/test_geir_sign_bits_pack.cpp) | 通过GE-IR构图方式调用SignBitsPack算子。 |
