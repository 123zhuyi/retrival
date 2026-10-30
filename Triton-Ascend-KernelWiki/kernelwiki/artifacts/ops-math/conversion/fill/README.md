# Fill

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

- **算子功能**：对张量进行填充操作，支持非连续的Tensor操作。
- **InplaceFillScalar**：使用标量值填充整个张量。
- **InplaceFillTensor**：使用张量值填充整个张量（value张量必须是0维或size=1的1维张量）。

## 参数说明

### InplaceFillScalar

<table style="undefined;table-layout: fixed; width: 922px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 202px">
<col style="width: 308px">
<col style="width: 102px">
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
    <td>selfRef</td>
    <td>输入/输出张量</td>
    <td>输入输出张量，支持非连续的Tensor，数据维度不支持8维以上。</td>
    <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、COMPLEX64、COMPLEX128、BOOL、BFLOAT16、UINT16、UINT32、UINT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入标量</td>
    <td>指定的标量值，数据类型需要与selfRef的数据类型满足数据类型推导规则。</td>
    <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、COMPLEX64、COMPLEX128、BOOL、BFLOAT16、UINT16、UINT32、UINT64</td>
    <td>-</td>
  </tr>
</tbody>
</table>

### InplaceFillTensor

<table style="undefined;table-layout: fixed; width: 922px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 202px">
<col style="width: 308px">
<col style="width: 102px">
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
    <td>selfRef</td>
    <td>输入/输出张量</td>
    <td>输入输出张量，支持非连续的Tensor，数据维度不支持8维以上。</td>
    <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、COMPLEX64、COMPLEX128、BOOL、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入张量</td>
    <td>填充值张量，必须是0维或size=1的1维张量。</td>
    <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、COMPLEX64、COMPLEX128、BOOL、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入张量的数据维度不支持8维以上。
- InplaceFillTensor的value参数必须是0维张量或size=1的1维张量。
- 数据类型需要满足互推导关系。

## 调用说明

| 调用方式  | 样例代码                                              | 说明                                                         |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_fill_scalar.cpp](./examples/test_aclnn_fill_scalar.cpp) | 通过[aclnnInplaceFillScalar](docs/aclnnInplaceFillScalar.md)接口方式调用InplaceFillScalar算子。 |
| aclnn接口 | [test_aclnn_fill_tensor.cpp](./examples/test_aclnn_fill_tensor.cpp) | 通过[aclnnInplaceFillTensor](docs/aclnnInplaceFillTensor.md)接口方式调用InplaceFillTensor算子。 |
