# ClipByValueV2

## 贡献说明

| 贡献者    | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|--------|------------------|-------|-----------|-----------|
| skywang2 | 个人开发者 | ClipByValueV2 | 2025/12/31 | 新增ClipByValueV2算子 |

### 支持的产品型号

本样例支持如下产品型号：

- Atlas A2 训练系列产品

### 算子描述

`ClipByValueV2`算子用于将一个张量值剪切到指定的最小值和最大值之间。

### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">ClipByValueV2</td></tr>
<tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">7, 2045</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">clip_value_min</td><td align="center">1</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">clip_value_max</td><td align="center">1</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">7, 2045</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">clip_by_value_v2</td></tr>
</table>

### 环境要求

编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 算子调用

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)

<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_clamp.cpp"> test_aclnn_clamp.cpp</a></td><td>通过aclnn调用的方式调用ClipByValueV2算子。</td>
    </tr>
</table>
