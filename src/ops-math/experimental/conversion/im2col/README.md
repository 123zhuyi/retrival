# Im2col

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：将三维或四维 NCHW 输入中的滑动窗口展开为二维列矩阵；四维输入保留 batch 维。
- 计算公式：

$$
outD = \left\lfloor\frac{inD + 2 \times paddingD - dilationD \times (kernelD - 1) - 1}{strideD}\right\rfloor + 1
$$

对于三维输入 $[C,H,W]$，输出 shape 为
$[C \times kernelH \times kernelW, outH \times outW]$；对于四维输入
$[N,C,H,W]$，输出 shape 为
$[N,C \times kernelH \times kernelW, outH \times outW]$。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| self | 输入 | 三维 $[C,H,W]$ 或四维 $[N,C,H,W]$ Tensor。 | FLOAT16、FLOAT、BFLOAT16、BOOL | ND |
| kernelSize | 属性 | 卷积核大小 $[kernelH,kernelW]$，长度为 2。 | INT64 | - |
| dilation | 属性 | 卷积核膨胀系数 $[dilationH,dilationW]$，长度为 2。 | INT64 | - |
| padding | 属性 | H/W 两侧的对称填充 $[paddingH,paddingW]$，长度为 2。 | INT64 | - |
| stride | 属性 | 滑动步长 $[strideH,strideW]$，长度为 2。 | INT64 | - |
| out | 输出 | 展开后的 Tensor，数据类型与 `self` 相同。 | FLOAT16、FLOAT、BFLOAT16、BOOL | ND |

## 约束说明

- `self` 的 C、H、W 必须大于 0；四维输入的 N 必须大于等于 0。
- `kernelSize`、`dilation`、`stride` 的元素必须大于 0，`padding` 的元素必须大于等于 0。
- 计算得到的 `outH` 和 `outW` 必须大于 0，`out` 的 shape 和数据类型必须与推导结果一致。
- 支持非连续输入和输出，aclnn 层负责连续化和结果回写。
- 本目录实现注册在 Atlas A2、Atlas A3 产品，不复用仓库中 `conversion/im2col` 的 Ascend 950 实现。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn 调用 | [test_aclnn_im2col](./examples/test_aclnn_im2col.cpp) | 通过 [aclnnIm2col](./docs/aclnnIm2col.md) 两段式接口调用。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| gcw_jgWZA4ay | 个人开发者 | Im2col | 2026/07/30 | Im2col 算子适配开源仓，新增 Atlas A2/A3 AscendC 实现及 BOOL 支持。 |
