# TensorRedirect

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    √     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：将输入张量 `x` 的数据完整拷贝到输出张量 `outputX`，即逐元素恒等映射。本接口不做任何算术运算、类型转换、广播、归约或layout变换，仅按元素的bit位模式原样搬运，因此输出与输入**逐bit严格相等**（bit-exact）。NaN、Inf、负零（-0.0）、非规格化数均原样透传，不做规范化。

  使用场景：图优化 / 引用重定向场景下，将某节点的输出数据搬运到独立的目标张量。本接口的语义与 `aclnnTensorMove` 等价，区别仅在于对应不同的IR算子（`TensorRedirect`）。

- 计算公式：

$$outputX_i = x_i$$

其中 $i$ 遍历 $x$ 的全部元素。

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
      <td>待拷贝的源张量。支持空Tensor，仅支持连续Tensor，rank取值范围 [1, 8]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outputX</td>
      <td>输出</td>
      <td>拷贝结果张量。数据类型、shape、数据格式均需与x严格一致。</td>
      <td>与x保持一致</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x与outputX的数据类型必须完全一致，本接口不做数据类型推导或转换。
- x与outputX的shape必须完全一致（逐维相等），维度范围为1-8维，不涉及broadcast。
- x与outputX的内存地址不允许重叠（非in-place），重叠时结果未定义。
- 仅支持连续Tensor，数据格式支持ND。
- 精度：恒等拷贝，输出与输入逐bit严格相等（bit-exact，rtol=atol=0）。浮点类型的NaN、Inf、负零（-0.0）及非规格化数均按bit位模式原样透传。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| 图模式调用 | [test_geir_tensor_redirect](./examples/test_geir_tensor_redirect.cpp) | 通过[算子IR](./op_graph/tensor_redirect_proto.h)构图方式调用TensorRedirect算子。 |
