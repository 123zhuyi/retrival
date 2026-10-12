# Ndtri

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Kirin xxx</term> | × |

## 功能说明

- 算子功能：逐元素计算标准正态分布累积分布函数（CDF）的反函数（又称 probit / 分位数函数），将概率张量映射为对应的标准正态分位点。
- 计算公式：

  $$
  y = \mathrm{ndtri}(p) = \Phi^{-1}(p) = \sqrt{2} \cdot \mathrm{erf}^{-1}(2p - 1), \quad p \in (0, 1)
  $$

  其中 $\Phi$ 为标准正态分布的累积分布函数，$\mathrm{erf}^{-1}$ 为逆误差函数。

- 计算策略：基于 Cephes 数学库的分区间有理逼近实现。设 $\mathrm{val\_sub} = e^{-2} \approx 0.1353$，$\mathrm{res\_exp} = 1 - e^{-2} \approx 0.8647$，按下表选择计算路径：

  | 区间 | 条件 | 计算方法 |
  | :--: | :--- | :--- |
  | 中心区 | $e^{-2} \le p \le 1 - e^{-2}$ | `cal_p0`：$y \approx \sqrt{2\pi}\cdot((p-0.5) + (p-0.5)^2 \cdot P_0/Q_0)$ |
  | 左尾 | $p < e^{-2}$ | `cal_sub + cal_p12`：$x=\sqrt{-2\ln p}$，$y \approx -(x - \ln x / x - P_{1/2}/Q_{1/2})$ |
  | 右尾 | $p > 1 - e^{-2}$ | `cal_sub + cal_p12`：$x=\sqrt{-2\ln(1-p)}$，$y \approx +(x - \ln x / x - P_{1/2}/Q_{1/2})$ |

- 特殊值处理（对齐 PyTorch `torch.special.ndtri` / Cephes）：

  | 输入 p | 输出 y |
  | :---: | :---: |
  | 0 | $-\infty$ |
  | 1 | $+\infty$ |
  | $p < 0$ 或 $p > 1$ | $\mathrm{NaN}$ |
  | $\mathrm{NaN}$ / $\pm\infty$ | $\mathrm{NaN}$ |

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>self</td>
    <td>输入</td>
    <td>公式中的输入概率张量 <i>p</i>，期望元素取值范围 <i>p</i> ∈ (0, 1)，越界或非法值按特殊值规则处理。支持 0-8 维，支持空 Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>公式中的输出 <i>y</i>，标准正态分位点。shape 与 dtype 均与 self 一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- out 与 self 必须同 shape、同 dtype、同 format。
- 支持维度范围 0 ~ 8；支持空 Tensor（numel=0 时 short-circuit 返回）。
- 非连续 Tensor 由 aclnn L2 层通过 Contiguous 兜底处理，Kernel 仅处理连续内存。
- fp16 / bf16 输入在 Kernel 内部统一 Cast 到 fp32 参与多项式计算，结果再 Cast 回原 dtype。
- 精度标准（浮点计算类社区标准，并发判定 MERE < Threshold 且 MARE < 10×Threshold）：

  | 数据类型 | Threshold |
  | :---: | :---: |
  | FLOAT32 | 2<sup>-13</sup> |
  | FLOAT16 | 2<sup>-10</sup> |
  | BFLOAT16 | 2<sup>-7</sup> |

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/arch35/test_aclnn_ndtri.cpp">test_aclnn_ndtri</a></td>
    <td>参见<a href="../../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody></table>

## 参考资源

- [TensorFlow: tf.math.ndtri](https://www.tensorflow.org/api_docs/python/tf/math/ndtri)（对标基线）
- [Cephes Mathematical Library - ndtri](https://www.netlib.org/cephes/)
- [SciPy: scipy.special.ndtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtri.html)
- [PyTorch: torch.special.ndtri](https://pytorch.org/docs/stable/special.html#torch.special.ndtri)
