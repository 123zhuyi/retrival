# Sin

## 贡献说明

| 贡献者    | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|--------|------------------|-------|-----------|-----------|
| skywang2 | 个人开发者 | Sin | 2025/12/31 | 新增Sin算子 |

### 算子描述

`Sin`算子返回输入数据经过开方运算的结果。

### 算子规格描述

<table style="border-collapse: collapse;">
  <tr>
    <th align="center";"></th>
    <th colspan="4" align="center";">Sin</th>
  </tr>
  <tr>
    <td align="center";"> </td>
    <td align="center";">name</td>
    <td align="center";">Type</td>
    <td align="center";">data type</td>
    <td align="center";">format</td>
  </tr>
  <tr>
    <td align="center";">算子输入</td>
    <td align="center";">x</td>
    <td align="center";">tensor</td>
    <td align="center";">float32, float16, bfloat16</td>
    <td align="center";">ND</td>
  </tr>
  <tr>
    <td align="center";">算子输出</td>
    <td align="center";">y</td>
    <td align="center";">tensor</td>
    <td align="center";">与输入相同</td>
    <td align="center";">ND</td>
  </tr>
  <tr>
    <td align="center";">核函数名</td>
    <td colspan="4" align="center";">Sin</td>
  </tr>
</table>

### 支持的产品型号

本样例支持如下产品型号：

- Atlas A2 训练系列产品
- Atlas 200I/500 A2 推理产品

### 环境要求

编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 算子调用

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)

<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_sin.cpp"> test_aclnn_sin.cpp</a></td><td>通过aclnn调用的方式调用Sin算子。</td>
    </tr>
</table>
