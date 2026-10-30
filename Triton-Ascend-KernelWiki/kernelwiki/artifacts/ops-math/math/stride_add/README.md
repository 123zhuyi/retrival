# StrideAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：NC1HWC0格式5D张量的局部逐元素加法。从x1和x2的C1维度指定偏移位置开始，取出c1_len个C1块数据进行加法运算。

- 计算公式：

$$
y[n, c, h, w, c0] = x1[n, x1\_c1\_offset + c, h, w, c0] + x2[n, x2\_c1\_offset + c, h, w, c0]
$$

其中 $c = 0, 1, ..., c1\_len - 1$

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
      <td>x1</td>
      <td>输入</td>
      <td>第一个输入张量，NC1HWC0格式5D张量，shape为(N, C1_x1, H, W, C0)。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>NC1HWC0</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>第二个输入张量，NC1HWC0格式5D张量，shape为(N, C1_x2, H, W, C0)，dtype与x1一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>NC1HWC0</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，NC1HWC0格式5D张量，shape为(N, c1_len, H, W, C0)，dtype与x1一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>NC1HWC0</td>
    </tr>
    <tr>
      <td>x1_c1_offset</td>
      <td>属性</td>
      <td>x1在C1维度的偏移（单位：C1块数），必须≥0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2_c1_offset</td>
      <td>属性</td>
      <td>x2在C1维度的偏移（单位：C1块数），必须≥0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>c1_len</td>
      <td>属性</td>
      <td>输出y的C1维度长度（单位：C1块数），必须>0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入输出仅支持5维NC1HWC0格式。
- x1和x2的dtype必须一致。
- c1_len必须>0。
- x1_c1_offset + c1_len ≤ C1_x1，x2_c1_offset + c1_len ≤ C1_x2。
- x1_c1_offset ≥ 0，x2_c1_offset ≥ 0。
- x1和x2的N、H、W、C0维度必须相同。
- 当前实现仅支持C0=16。
- bfloat16输入内部使用float32中间计算（类型提升），输出转回bfloat16。

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| 图模式调用 | [test_geir_stride_add.cpp](./examples/test_geir_stride_add.cpp) | 通过[算子IR](./op_graph/stride_add_proto.h)构图方式调用StrideAdd算子 |
