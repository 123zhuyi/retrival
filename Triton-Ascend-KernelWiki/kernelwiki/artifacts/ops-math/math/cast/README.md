# Cast

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

将输入tensor转换为指定的dtype类型。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 200px">
  <col style="width: 460px">
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
      <td>待进行cast计算的入参。</td>
      <td>BOOL、FLOAT16、FLOAT、INT8、INT32、UINT32、UINT8、INT64、UINT64、INT16、UINT16、DOUBLE、COMPLEX64、COMPLEX128、QINT8、QUINT8、QINT16、QUINT16、QINT32、BF16、UINT1、COMPLEX32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行cast计算的出参。</td>
      <td>BOOL、FLOAT16、FLOAT、INT8、INT32、UINT32、UINT8、INT64、UINT64、INT16、UINT16、DOUBLE、COMPLEX64、COMPLEX128、QINT8、QUINT8、QINT16、QUINT16、QINT32、BF16、UINT1、COMPLEX32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品：不支持BFLOAT16。
- Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持COMPLEX32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。

## 约束说明

- 针对数据类型从浮点数转换为整型的场景：
  输入数据中存在nan，则将nan转换为0。

- 针对输入数据类型为BOOL、COMPLEX32、COMPLEX64、COMPLEX128、FLOAT4_E2M1、FLOAT4_E1M2的场景：
  不支持输入为非连续。

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 针对数据类型从INT32转换为INT8的场景：
    只能保证输入数据在(-2048, 1920)范围内精度无误差。
  - 针对数据类型从FLOAT64/COMPLEX64/COMPLEX128转换为UINT8的场景：
    只能保证输入数据为非负数精度无误差。

- <term>Atlas 推理系列产品</term>：
  - 针对数据类型从FLOAT32转换为INT64和FLOAT32转换为UINT8的场景：
    只能保证输入数据在(-2147483648, 2147483583)范围内精度无误差。

  - 针对数据类型从INT64转换为FLOAT32的场景：
    只能保证输入数据在(-2147483648, 2147483647)范围内精度无误差。

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 针对输入、输出类型，涉及COMPLEX32、COMPLEX64、FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN的，只支持如下表格中的转换路径：

    | `self`数据类型 | `out`数据类型 |
    | -------- | -------- |
    | COMPLEX32 | COMPLEX64 |
    | COMPLEX64 | COMPLEX32 |
    | COMPLEX32 | FLOAT16 |
    | FLOAT16 | COMPLEX32 |
    | FLOAT32/BFLOAT16 | COMPLEX64 |
    | COMPLEX64 | FLOAT32/BFLOAT16 |
    | FLOAT32/FLOAT16/BFLOAT16 | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | FLOAT32/FLOAT16/BFLOAT16 |
    | FLOAT32/FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32/FLOAT16/BFLOAT16 |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_cast](./examples/test_aclnn_cast.cpp) | 通过[aclnnCast](./docs/aclnnCast.md)接口方式调用Cast算子。 |
| 图模式调用 | [test_geir_cast](./examples/test_geir_cast.cpp)   | 通过[算子IR](./op_graph/cast_proto.h)构图方式调用Cast算子。 |
