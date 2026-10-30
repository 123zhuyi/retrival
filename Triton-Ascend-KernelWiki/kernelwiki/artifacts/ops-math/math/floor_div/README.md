# FloorDiv

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

- 算子功能：完成除法计算，对结果向下取整。
- 计算公式：

  $$
  out_i = floor(\frac{self_i}{other_i})
  $$

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1496px"><colgroup>
  <col style="width: 149px">
  <col style="width: 120px">
  <col style="width: 205px">
  <col style="width: 305px">
  <col style="width: 317px">
  <col style="width: 121px">
  <col style="width: 134px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td>
        <ul>
          <li>数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/deduction_relationship.md)）。</li>
          <li>shape需要与other满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
      <td>ND</td>
      <td>不超过8维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>公式中的输入other。</td>
      <td>
        <ul>
          <li>数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/deduction_relationship.md)）。</li>
          <li>shape需要与other满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
      <td>ND</td>
      <td>不超过8维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>公式中的alpha。</td>
      <td>数据类型需要可转换成self与other推导后的数据类型。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>
        <ul>
          <li>数据类型需要是self与other推导之后可转换的数据类型（参见[互转换关系](../../docs/zh/context/conversion_relationship.md)）。</li>
          <li>shape需要是self与other broadcast之后的shape。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
      <td>ND</td>
      <td>不超过8维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas 训练系列产品</term>：不支持BFLOAT16数据类型。

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_floor_divide](./examples/test_aclnn_floor_divide.cpp) | 通过[aclnnFloorDivide和aclnnFloorDivide](./docs/aclnnFloorDivide&aclnnInplaceFloorDivide.md)接口方式调用FloorDiv算子  |
