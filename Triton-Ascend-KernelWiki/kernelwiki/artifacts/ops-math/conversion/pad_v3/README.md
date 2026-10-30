# PadV3

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：对输入tensor做填充。
- 示例：

  ```text
  输入tensor([[0,1,2]])
  paddings([2,2])

  1.mode("REFLECT")
  输出为([[2,1,0,1,2,1,0]])

  2.mode("SYMMETRIC")
  输出为([[1,0,0,1,2,2,1]])

  3.mode("constant")
  输出为([[0,0,0,1,2,0,0]])

  4.mode("edge")
  输出为([[0,0,0,1,2,2,2]])
  ```

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
      <td>x</td>
      <td>输入</td>
      <td>待进行扩充的原始tensor。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>扩充的配置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>constant_values</td>
      <td>输入</td>
      <td>constant模式下填充常量的值。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>填充模式，包括常量"constant"、边缘"edge"、反射"REFLECT"以及对称"SYMMETRIC"四种选项。</td>
      <td>string</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings_contiguous</td>
      <td>属性</td>
      <td>用于控制paddings这个张量的存储是否是“连续的”。</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行扩充后的tensor。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：数据类型不支持BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型不支持HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。

- paddings
    paddings的第0维表示对输入x第0维的扩充配置，以此类推。每一行表示对应维度上的填充数量（左/右、前/后、上/下等）。对于每一行[a, b]，a表示在该维度的开头填充的元素数，b表示在该维度的末尾填充的元素数(在paddings_contiguous为true的情况下)。

- mode
    constant模式下，填充常量值，常量值在输入constant_values中表示，当constant_values为空时表示填充0。
    edge模式下，以输入x的边缘值进行填充。
    REFLECT模式下，以镜像的方式填充，镜像时不会包括边界本身。
    在SYMMETRIC模式下，以镜像的方式填充，镜像时会包含边界本身。
- paddings_contiguous

    若为true时，paddings按 行主序(row-major)排列。
    例如[[2,2],[1,1]]表示：

        第0维：左边填2，右边填2。
        第1维：左边填1，右边填1。

    若为false时，paddings按 列主序(column-major)排列。
    例如[[2,1],[2,1]]表示：

        第0维：左边填2，右边填2。
        第1维：左边填1，右边填1。

## 约束说明

在paddings_contiguous = true即行主序的情况下：
paddings的形状需要为[rank, 2]，其中rank为输入x的维度数。
对于每一行paddings[i]，填充元素数不得超过其对应x[i]的元素数。即在REFLECT模式下，paddings[i][0]和paddings[i][1] <= x[i].length-1；在SYMMETRIC模式下，paddings[i][0]和paddings[i][1] <= x[i].length。

在paddings_contiguous = false即列主序的情况下，REFLECT模式和SYMMETRIC模式的paddings约束对应同理。

CONSTANT模式下，如果x的数据类型为HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2，则paddings所有维度均要为非负数；如果x的数据类型为FLOAT4_E2M1、FLOAT4_E1M2，还需满足输入x的最后一维的paddings都要为偶数。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----|----|----|
| aclnn调用 | [test_aclnn_constant_pad_nd](./examples/test_aclnn_constant_pad_nd.cpp) | 通过[aclnnConstantPadNd](./docs/aclnnConstantPadNd.md)接口方式调用constant模式的pad_v3算子。 |
| aclnn调用 | [test_aclnn_replication_pad_1d](./examples/test_aclnn_replication_pad_1d.cpp) | 通过[aclnnReplicationPad1d](./docs/aclnnReplicationPad1d.md)的接口方式调用edge模式下的pad_v3算子，填充输入tensor的最后一维|
| aclnn调用 | [test_aclnn_replication_pad_2d](./examples/test_aclnn_replication_pad_2d.cpp) | 通过[aclnnReplicationPad2d](./docs/aclnnReplicationPad2d.md)的接口方式调用edge模式下的pad_v3算子，填充输入tensor的最后两维|
| aclnn调用 | [test_aclnn_replication_pad_3d](./examples/test_aclnn_replication_pad_3d.cpp) | 通过[aclnnReplicationPad3d](./docs/aclnnReplicationPad3d.md)的接口方式调用edge模式下的pad_v3算子，填充输入tensor的最后三维|
