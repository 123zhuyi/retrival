# Cast

## 贡献说明

| 贡献者    | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|--------|------------------|-------|-----------|-----------|
| skywang2 | 个人开发者 | Cast | 2025/12/31 | 新增Cast算子 |

### 算子描述

`Cast`算子提供将tensor从源数据类型转换为目标数据类型的功能。

### 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Cast</th></tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float16, float32, int32, int8, uint8, bool, int64, bfloat16, int16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">tensor</td><td align="center">float16, float32, int32, int8, uint8, bool, int64, bfloat16, int16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子属性</td><td align="center">dstType</td><td align="center">attr</td><td align="center">int64</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cast</td></tr>
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
        <td><a href="./examples/test_aclnn_cast.cpp"></a> test_aclnn_cast.cpp</td><td>通过aclnn调用的方式调用Cast算子</td>
    </tr>
</table>
