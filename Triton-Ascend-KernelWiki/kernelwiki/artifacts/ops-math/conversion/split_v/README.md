# SplitV

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √      |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √      |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：根据size_splits将张量沿维度split_dim拆分为num_split更小的张量。

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
      <td>需要切分的tensor列表。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、INT64、INT32、UINT8、UINT16、UINT32、UINT64、INT8、INT16、BOOL、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size_splits</td>
      <td>输入</td>
      <td>指定一个列表，其中包含沿分割维度的每个输出张量的大小。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>split_dim</td>
      <td>输入</td>
      <td>指定沿其分割的维度。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_split</td>
      <td>属性</td>
      <td>指定要分割的tensor个数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>FLOAT16、FLOAT32、DOUBLE、INT64、INT32、UINT8、UINT16、UINT32、UINT64、INT8、INT16、BOOL、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* Atlas 训练系列产品、Atlas 推理系列产品、Atlas 200I/500 A2 推理产品、Atlas 200/300/500推理产品：不支持BFLOAT16。

## 约束说明

* size_splits中的每个元素都大于或等于1。
* size_splits的长度等于num_split的值。
* size_splits中的元素总和为维度split_dim的大小。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_split_tensor](examples/test_aclnn_split_tensor.cpp) | 通过[aclnnSplitTensor](docs/aclnnSplitTensor.md)接口方式调用SplitV算子。 |
| aclnn接口 | [test_aclnn_split_with_size](examples/test_aclnn_split_with_size.cpp) | 通过[aclnnSplitWithSize](docs/aclnnSplitWithSize.md)接口方式调用SplitV算子。 |
