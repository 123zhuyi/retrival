# Expandv

## 产品支持情况

| 产品                                                                                     | 是否支持 |
| :--------------------------------------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：将输入张量在长度为 1 的维度上广播到目标 shape，返回广播后的新张量。
- 计算规则（遵循 NumPy 广播规则）：

$$
y_i = x_{\mathrm{offset}(i)}
$$

其中 $\mathrm{offset}(i)$ 表示输出索引 $i$ 映射到输入张量的对应位置。

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
      <td>待进行广播的输入张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、BOOL、INT16、UINT16、INT32、UINT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>必选属性</td>
      <td>目标广播形状（输出形状）。</td>
      <td>INT64 列表</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>广播后的输出张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、BOOL、INT16、UINT16、INT32、UINT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入与输出维度需满足广播规则（从尾维对齐）：
   - 输入维度等于目标维度，或
   - 输入维度为 1（可广播）。
2. 目标 shape 中新增的高维必须为正整数。
3. 数据格式当前支持 ND。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn 调用 | [test_aclnn_expandv](./examples/test_aclnn_expandv.cpp) | 通过 [aclnnExpandv](./docs/aclnnExpandv.md) 接口方式调用 Expandv 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| OpenBOAT（HIT）团队 | 个人开发者 | Expandv | 2025/12/12 | 新增 Expandv 算子 |
| fulltower | 个人开发者 | Expandv | 2026/04/01 | 补充 README 文档 |
