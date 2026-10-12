# Log

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：完成自然对数的计算。

- 计算公式：

$$
output_i=log_e(self_i)
$$

- 结合下面参数，变形后的公式如下：

$$
y = logbase(x * scale + shift)
$$

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
      <td>待进行log计算的入参，公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行完log计算的出参，公式中的output。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>base</td>
      <td>属性</td>
      <td>对数的底数（自定义对数类型）</td>
      <td>FLOAT，默认值-1.0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>缩放因子（输入的线性缩放）</td>
      <td>FLOAT，默认值1.0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>属性</td>
      <td>偏移量（输入的线性偏移）</td>
      <td>FLOAT，默认值0.0</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_log.cpp](./examples/test_aclnn_log.cpp) | 通过[test_aclnn_log](./docs/aclnnLog&aclnnInplaceLog.md)接口方式调用log算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| fulltower | 个人开发者 | log | 2025/12/15 | Log算子适配开源仓 |
