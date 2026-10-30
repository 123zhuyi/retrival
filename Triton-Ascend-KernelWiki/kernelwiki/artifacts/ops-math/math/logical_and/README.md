# LogicalAnd

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

- 算子功能：对两个输入张量的对应元素执行「与逻辑」判断，输出布尔型张量（True/False）。

- 计算公式：

$out=input1∧input2$

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
      <td>input_1</td>
      <td>输入</td>
      <td>待进行logical_and计算的入参，公式中的input_1。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_2</td>
      <td>输入</td>
      <td>待进行logical_and计算的入参，公式中的input_2。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行logical_and计算的出参，公式中的out。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品、Atlas 推理系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logical_and](./examples/test_aclnn_logical_and.cpp) | 通过[aclnnLogicalAnd](./docs/aclnnLogicalAnd&aclnnInplaceLogicalAnd.md)接口方式调用LogicalAnd算子。 |
| 图模式调用 |  |  |

本目录仅包含LogicalAnd算子对应的aclnn接口；如您想要贡献该算子的AscendC实现，请参考[贡献流程](../../CONTRIBUTING.md)。
