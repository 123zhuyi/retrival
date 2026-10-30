# MaskedFill

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

- 算子功能：将输入Tensor`x`中mask位置为`True`的元素填充指定的值。`mask`必须与`x`的shape相同或可广播。

- 计算公式：
  
  $$
  y_i = 
  \begin{cases}
  x_i, \, \, mask_i=False\\
  value_i, \, \, mask_i=True\\
  \end{cases}
  $$

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
      <td>公式中的输入x。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT8、INT32、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>公式中的输入mask。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的输入value。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT8、INT32、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT8、INT32、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* Atlas 训练系列产品、Atlas 推理系列产品、Atlas 200I/500 A2 推理产品：不支持BFLOAT16。

## 约束说明

* value的dtype与x不一致时，需保证进行类型转换时不会溢出。
* x和mask需要满足广播关系。
* value的数据类型能够转换为x的数据类型。
* y的shape可以由x、mask、value共同广播。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_inplace_masked_fill_scalar](examples/test_aclnn_inplace_masked_fill_scalar.cpp) | 通过[aclnnInplaceMaskedFillScalar](docs/aclnnInplaceMaskedFillScalar.md)接口方式调用MaskedFill算子。 |
