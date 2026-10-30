# Polar

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：由模长（abs）与幅角（angle）构造极坐标复数张量，对应 PyTorch 接口 `torch.polar(abs, angle)`。

- 计算公式：

  $$
  out_i = input_i \times (\cos(angle_i) + i \cdot \sin(angle_i))
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
      <td>input</td>
      <td>输入</td>
      <td>极坐标模长分量，公式中的input。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>angle</td>
      <td>输入</td>
      <td>极坐标幅角（弧度），公式中的angle。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算结果，公式中的out；shape为input与angle广播后的shape。</td>
      <td>COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- input与angle的数据类型必须一致（FLOAT）。
- input与angle的shape满足NumPy广播关系，out的shape为两者广播后的shape。
- input、angle的维度数不超过8维。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_polar](./examples/test_aclnn_polar.cpp) | 通过[aclnnPolar](./docs/aclnnPolar.md)接口方式调用Polar算子。 |
| 图模式调用 | | |

本目录包含Polar算子的aclnn接口及AscendC实现；关于该算子的设计与自测，请参考[设计文档](./docs/design.md)与[自测报告](./tests/自测报告.md)。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 开源社区贡献者 | 开源社区 | Polar | 2026/06/30 | Polar算子适配开源仓 |
