# AsStrided

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：根据输入张量 `x`、输出形状 `size`、步长 `stride` 和存储偏移 `storage_offset`，生成一个按照指定步长访问输入存储的输出张量，对应 PyTorch `torch.as_strided` 语义。

- 计算公式：

  $$
  y_i=x_{\text{storage\_offset}+\sum_{d=0}^{D-1}(i_d\cdot \text{stride}[d])}
  $$

  其中 `D` 为 `size` 的长度，`i_d` 为输出元素 `i` 在第 `d` 维上的坐标。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 280px">
  <col style="width: 325px">
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
      <td>输入张量。</td>
      <td>INT64、UINT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BF16、INT16、UINT16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>输出张量的形状。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>输出张量各维度映射到输入存储时使用的步长。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>storage_offset</td>
      <td>输入</td>
      <td>输出首元素相对于输入存储起始位置的偏移量。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，shape由size决定，数据类型与x一致。</td>
      <td>INT64、UINT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BF16、INT16、UINT16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 仅支持 ND 格式。
- `size` 和 `stride` 的长度必须一致，且长度范围为 1 到 8。
- `size`、`stride` 中的元素必须为非负整数。
- `storage_offset` 必须为非负整数，且当前 aclnn 调用方式需要传入长度为 1 的 `aclIntArray`。
- 当输出元素个数不为 0 时，`storage_offset + sum((size[d] - 1) * stride[d])` 不能超出输入张量存储范围。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn接口 | [test_aclnn_as_strided](./examples/test_aclnn_as_strided.cpp) | 通过[aclnnAsStrided](./docs/aclnnAsStrided.md)接口方式调用AsStrided算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| [Asahi](https://gitcode.com/Mars_Cheng_cys) | 西北工业大学智能感知交互实验室 | AsStrided | 2026/7/9 | AsStrided算子适配开源仓 |
