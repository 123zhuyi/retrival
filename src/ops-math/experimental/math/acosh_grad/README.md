# AcoshGrad

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |

## 功能说明

- 算子功能：AcoshGrad 是 Acosh 算子的反向梯度算子。输入前向 Acosh 的输出 `y` 和上游梯度 `dy`，输出对前向输入的梯度 `dx`。
- 计算公式：

  $$
  dx_i = \frac{dy_i}{\sinh(y_i)}
  $$

  当前 AscendC 实现保留 TBE DSL 路径，使用 `y/8` 上的 Taylor 多项式和 3 次 `sqrt` 倍角公式近似计算 `sinh(y)`，再执行 `dy / sinh(y)`。

## 参数说明

<table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 340px">
  <col style="width: 240px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>前向 Acosh 算子的输出 tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>上游传播的梯度 tensor，shape 和 dtype 必须与 y 相同。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>反向梯度输出 tensor，shape 和 dtype 与 y 相同。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `y` 和 `dy` 的数据类型必须一致，仅支持 FLOAT16、FLOAT32、BFLOAT16。
- `y` 和 `dy` 的 shape 必须完全相同，不支持 broadcast。
- `dx` 的 shape 和 dtype 与 `y` 保持一致。
- 支持 ND 格式，元素数需大于 0。
- `sinh(y)==0` 时结果按硬件除法行为产生 `inf` 或 `nan`，接口层不做额外截断。
- FLOAT16 和 BFLOAT16 输入在 kernel 内部转换为 FLOAT32 计算，输出前 cast 回原 dtype。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :-- | :-- | :-- |
| aclnn 接口 | [test_aclnn_acosh_grad.cpp](examples/test_aclnn_acosh_grad.cpp) | 通过 [aclnnAcoshGrad](docs/aclnnAcoshGrad.md) 接口方式调用 AcoshGrad 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :-- | :-- | :-- | :-- | :-- |
| [GMOW](https://gitcode.com/gcw_8p1hhlB0) | 西北工业大学智能感知交互实验室 | AcoshGrad | 2026/7/25 | AcoshGrad 算子 AscendC 实现|
