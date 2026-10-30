# Fill

## 贡献说明

| 贡献者    | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|--------|------------------|-------|-----------|-----------|
| skywang2 | 个人开发者 | Fill | 2025/12/31 | 新增Fill算子 |

### 支持的产品型号

- Atlas A2 训练系列产品

### 算子描述

- 功能描述

  Fill算子创建一个形状由输入dims指定的张量，并用标量值value填充所有元素。该算子常用于初始化张量为特定值。

- 原型信息

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Fill</th></tr>
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td rowspan="2" align="center">算子输入</td></tr>

<tr><td align="center">dims</td><td align="center">attr_tuple</td><td align="center">int64</td><td align="center">-</td></tr>
<tr><td rowspan="2" align="center">算子输入</td></tr>

<tr><td align="center">value</td><td align="center">scalar</td><td align="center">float32,float16,bfloat16,int8,bool,int64,int32</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,int8,bool,int64,int32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">Fill</td></tr>
</table>

- 约束与限制

  dims的数据类型支持INT64。

  value的数据类型支持FLOAT16、FLOAT32、INT8、INT32、INT64，BOOL、BFLOAT16。

  y的数据类型支持FLOAT16、FLOAT32、INT8、INT32、INT64，BOOL、BFLOAT16。，数据格式只支持ND。

### 环境要求

编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 运行验证

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_fill.cpp"> test_aclnn_fill.cpp</a></td><td>通过aclnn调用的方式调用Fill算子。</td>
    </tr>
</table>
