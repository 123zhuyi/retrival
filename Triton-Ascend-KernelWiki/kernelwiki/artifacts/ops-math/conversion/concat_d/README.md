# ConcatD

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                       |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品      |    √     |
| Atlas 200I/500 A2 推理产品                      |    √     |
| Atlas 推理系列产品                              |    √     |
| Atlas 训练系列产品                              |    √     |

## 功能说明

- 算子功能：将tensors中所有tensor按照维度dim进行级联，除了dim对应的维度以外的维度必须一致。

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
      <td>需要级联的tensor列表。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16、DOUBLE、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concat_dim</td>
      <td>属性</td>
      <td>需要级联的维度。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>N</td>
      <td>可选属性</td>
      <td>指定要级联的tensor个数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、BFLOAT16、DOUBLE、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品、Atlas 200I/500 A2 推理产品：不支持BFLOAT16。
- Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持UINT16、UINT32、UINT64、FLOAT4_E1M2、FLOAT4_E2M1。

## 约束说明

- x列表中元素的数据类型和数据格式必须在支持的范围之内。
- x列表可以做数据类型推导。
- 推导出的数据类型必须能转换为指定输出y的类型。
- 非级联维度shape必须一致。
- dim必须在x维度范围。
- 输入数据类型为FLOAT4_E1M2、FLOAT4_E2M1时，每个输入tensor在concat轴及之后各维度大小的乘积须为偶数。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_cat](examples/test_aclnn_cat.cpp) | 通过[aclnnCat](docs/aclnnCat.md)接口方式调用ConcatD算子。 |
