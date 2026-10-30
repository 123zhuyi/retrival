# LogicalAnd

## 产品支持情况

| 产品                                                                                     | 是否支持 |
| :--------------------------------------------------------------------------------------- | :------: |
| `<term>`Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 `</term>` |    √    |

## 功能说明

- 算子功能：对两个输入张量进行逻辑和运算。
- 计算公式：

$$
out_i=self_i && other_i
$$

- 实际运算逻辑：

将两个输入，类型转换为fp16，进行Mul运算，得到结果之后转换为int8.

## 参数说明

 <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
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
      <td>待进行logical_and计算的入参，公式中的self。</td>
      <td>无</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行logical_and计算的入参，公式中的other。</td>
      <td>无</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>  
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行logical_and计算的出参，公式中的out。</td>
      <td>shape与self相同。</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>0-8</td>
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
  
## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logical_and.cpp](./examples/test_aclnn_logical_and.cpp) | 通过[test_aclnn_logical_and](./docs/aclnnLogicalAnd.md)接口方式调用SelectV2算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Nice_try | 个人开发者 | LogicalAnd | 2025/12/13 | LogicalAnd算子适配开源仓 |
