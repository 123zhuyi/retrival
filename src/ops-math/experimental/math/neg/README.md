# Neg

## 贡献说明

| 贡献者    | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|--------|------------------|-------|-----------|-----------|
| skywang2 | 个人开发者 | Neg | 2025/12/31 | 新增Neg算子 |
| CyndiZ | 个人开发者 | Neg | 2026/5/11 | Neg算子新增int64/int16/uint8支持 |

### 算子描述

`Neg`算子对输入的数值型数据执行取负操作（y = -x）。

### 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Neg</th></tr>
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr> <tr><td rowspan="1" align="center">算子输入</td>
<td align="center">x</td><td align="center">tensor</td> <td align="center">int64, int32, int16, int8, uint8, float16, bfloat16, float32</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td> <td align="center">与输入相同</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">neg</td></tr>
</table>

### 支持的产品型号

本样例支持如下产品型号：

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品

### 环境要求

编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 算子调用

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td>test_aclnn_neg.cpp</td>
        <td>通过aclnn调用的方式调用Neg算子</td>
    </tr>
</table>
