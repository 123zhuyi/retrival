# SignBitsUnpack

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：将UINT8类型的1-bit Adam符号位拆包为FLOAT32或FLOAT16类型的张量。

- 计算公式：

设输入`self`为长度为N的1D UINT8张量，每个字节包含8个符号位（按LSB优先顺序），第$j$个字节$b_j$的第$i$位（$i = 0, 1, \dots, 7$）解包规则如下：

$$
out_{j \cdot 8 + i} = \begin{cases} +1.0, & \text{bit}_i(b_j) = 1 \\ -1.0, & \text{bit}_i(b_j) = 0 \end{cases}
$$

输出张量共$N \times 8$个元素，并被reshape为二维，其shape为：

$$
out.shape = (size,\ \frac{N \times 8}{size})
$$

其中`size`为入参指定的输出第一维度大小。

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
      <td>self</td>
      <td>输入</td>
      <td>待进行SignBitsUnpack计算的入参，1D张量，每个UINT8元素包含8个符号位。支持空tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入属性</td>
      <td>表示维度处理，reshape时输出张量的第一个维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>输入属性</td>
      <td>表示量化输出Tensor的数据类型。</td>
      <td>FLOAT16、FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行SignBitsUnpack计算的出参，2D张量，数据类型由dtype参数决定。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `self`必须是1维张量，`out`必须是2维张量。
- `size`必须大于0，且（`self`的元素个数 × 8）能被`size`整除。
- `out`的第一维度必须等于`size`。
- `out`的数据类型必须与`dtype`参数一致。
- 数据格式仅支持ND。
- 支持空tensor场景。
- 确定性计算：aclnnSignBitsUnpack默认确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sign_bits_unpack](./examples/test_aclnn_sign_bits_unpack.cpp) | 通过[aclnnSignBitsUnpack](./docs/aclnnSignBitsUnpack.md)接口方式调用SignBitsUnpack算子。 |
| 图模式调用 | [test_geir_sign_bits_unpack](./examples/test_geir_sign_bits_unpack.cpp)   | 通过[算子IR](./op_graph/sign_bits_unpack_proto.h)构图方式调用SignBitsUnpack算子。 |
