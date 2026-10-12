# Arange

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |

## 功能说明

- 算子功能：从 `start` 起始、以 `step` 为步长、到 `end` 结束（**左闭右开**，不含 `end`），生成一个一维等差序列张量并写入 `out`。`start`、`end`、`step` 均为 Host 侧标量（aclScalar），`out` 为一维输出张量（aclTensor）。功能与昇腾内置 `aclnnArange`、PyTorch `torch.arange` 一致。

- 计算公式：

  序列元素：

  $$
  \text{out}_i = \text{start} + i \times \text{step}, \quad i = 0, 1, \dots, N-1
  $$

  输出元素个数 N（左闭右开，**向上取整**）：

  $$
  N = \left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil
  $$

  > 取整口径为 `ceil`（左闭右开），与昇腾内置 `aclnnArange` / PyTorch `torch.arange` 一致。其中 `out` 的元素个数 N 由**调用方**按上式计算，并据此分配并构造 `out` 张量（shape 为 `[N]`）；算子本身不重新计算或校验 N（详见“约束说明 - 调用方前置约束”）。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 130px">
  <col style="width: 300px">
  <col style="width: 320px">
  <col style="width: 130px">
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
      <td>start</td>
      <td>输入</td>
      <td>Host 侧的 aclScalar，取值范围的起始位置，对应公式中的 start。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>输入</td>
      <td>Host 侧的 aclScalar，取值范围的结束位置（左闭右开，不含 end），对应公式中的 end。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>step</td>
      <td>输入</td>
      <td>Host 侧的 aclScalar，取值的步长，对应公式中的 step。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>一维输出张量，存放等差序列，shape 为 [N]，对应公式中的 out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

> `start`、`end`、`step`、`out` 四者的数据类型必须保持一致，不做跨数据类型推导。

> **关于 INT32 / INT64**：算子原型实际注册 8 种数据类型（FLOAT / FLOAT16 / BFLOAT16 / INT8 / UINT8 / INT16 / **INT32 / INT64**）。其中 INT8 / UINT8 / INT16 为必测数据类型；**INT32 / INT64 为兼容保留项**（INT32 同时是 examples 与性能对标的主用例）。INT32 / INT64 同样走 FP32 中间域计算（详见“功能说明”），由于 FP32 尾数仅 24 位，当序列值的绝对值 **超过 2^24（16777216）** 时存在精度损失（无法精确表示该量级的整数），调用方应在该约束内使用，或避免对超大值域使用 INT32 / INT64。

## 约束说明

### 算子约束

- `start`、`end`、`step`、`out` 四者的数据类型必须保持一致，且数据格式只支持 ND。
- `out` 不支持空 Tensor（要求 N ≥ 1）。
- 整数类型（INT8、UINT8、INT16）输出当序列值超出对应类型值域时，按硬件 Cast **饱和（clamp）** 语义处理（例如 INT8 越界值截断到 [-128, 127]）。该行为已在 NPU 上实测确认，调用方应保证序列值落在目标类型值域内以获得与 CPU 标杆一致的结果。
- 确定性：算子为纯逐元素等差序列生成（`out[i] = start + i*step`），无 Reduce、无核间累加，相同输入恒产生相同输出，默认确定性实现。

### 调用方前置约束（值级，由调用方保证；接口不做值级校验）

`aclnnArange` 接口的入参校验仅覆盖**数据类型**（白名单 + 四者一致性）与**空指针**；以下**值级**约束属于**调用方前置条件**，接口不做值级校验。调用方须在调用前自行保证，否则行为未定义：

| 前置约束 | 调用方须保证 |
|---------|-------------|
| step ≠ 0 | step 非零 |
| step 符号匹配 | step > 0 时 start < end；step < 0 时 start > end（即 (end - start) 与 step 同号，N ≥ 1） |
| UINT8 非负 | out 为 UINT8 时，start / end / step 均需为非负，且需 step > 0、start < end（UINT8 不可表示负值） |
| N 由调用方计算 | out 的元素个数 N = ceil((end - start) / step)，由调用方按该公式计算并据此分配、构造 out 张量；算子不重新计算或校验 N |
| N ≥ 1 | 不支持空 Tensor，N ≤ 0 为非法输入 |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn 调用 | [test_aclnn_arange](./examples/test_aclnn_arange.cpp) | 通过 `aclnnArange` 两段式接口（`aclnnArangeGetWorkspaceSize` + `aclnnArange`）调用 Arange 算子，覆盖 FLOAT 升序、负 step 降序、INT8 等多组用例。 |

测试命令调用方式：`bash build.sh --run_example arange eager cust --vendor_name=custom --experimental`（参考 [build.sh 调用说明](../../../docs/zh/invocation/quick_op_invocation.md)）。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| forge | 个人贡献者 | Arange | 2026-06 | 扩展 INT8/UINT8/INT16 数据类型；动态多核 former/tail 切分；ArithProgression 等性能优化 |
