# BitwiseOr

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算输入张量self中每个元素和输入标量other的按位或。输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑或。
- 计算公式：

  $$
  \text{out}_i =
  \text{self}_i \, | \, \text{other}
  $$

## 参数说明
  
  <table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>公式中的输入self。</td>
      <td>INT16、UINT16、INT32、INT64、INT8、UINT8、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>公式中的输入other。</td>
      <td>INT16、UINT16、INT32、INT64、INT8、UINT8、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>INT16、UINT16、INT32、INT64、INT8、UINT8、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - Atlas 推理系列产品、Atlas 训练系列产品：数据类型不支持INT8、UINT8、UINT32、INT64、UINT64。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持INT8、UINT8、UINT32、UINT64。

## 约束说明

- 无

## 调用说明
    
| 调用方式   | 样例代码           | 说明                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_bitwise_or_scalar](examples/test_aclnn_bitwise_or_scalar.cpp) | 通过[aclnnBitwiseOrScalar](docs/aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar.md)接口方式调用BitwiseOr算子。 |
| aclnn接口  | [test_aclnn_bitwise_or_tensor](examples/test_aclnn_bitwise_or_tensor.cpp) | 通过[aclnnBitwiseOrTensor](docs/aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor.md)接口方式调用BitwiseOr算子。 |
| aclnn接口  | [test_aclnn_inplace_bitwise_or_scalar](examples/test_aclnn_inplace_bitwise_or_scalar.cpp) | 通过[aclnnInplaceBitwiseOrScalar](docs/aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar.md)接口方式调用BitwiseOr算子。 |
| aclnn接口  | [test_aclnn_inplace_bitwise_or_tensor](examples/test_aclnn_inplace_bitwise_or_tensor.cpp) | 通过[aclnnInplaceBitwiseOrTensor](docs/aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor.md)接口方式调用BitwiseOr算子。 |
