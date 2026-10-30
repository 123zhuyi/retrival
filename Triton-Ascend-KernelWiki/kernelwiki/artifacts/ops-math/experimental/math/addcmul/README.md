# Addcmul

## 贡献说明

| 贡献者       | 贡献方              | 贡献算子      | 贡献时间 | 贡献内容 |
|-----------|------------------|-----------|------|------|
| skywang2 | 个人开发者 | Addcmul | 2025/12/31 | 新增Addcmul算子 |

## 支持的产品型号

- Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述

- 功能描述

  `Addcmul`算子实现了向量x1乘向量x2，乘标量value后的结果再加上向量input_data，返回计算结果的功能。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Addcmul</th></tr>
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td rowspan="5" align="center">算子输入</td></tr>

    <tr><td align="center">input_data</td><td align="center">tensor</td><td align="center">float32,float16,int32,bfloat16</td><td align="center">ND</td></tr>
    <tr><td align="center">x1</td><td align="center">tensor</td><td align="center">float32,float16,int32,bfloat16</td><td align="center">ND</td></tr>
    <tr><td align="center">x2</td><td align="center">tensor</td><td align="center">float32,float16,int32,bfloat16</td><td align="center">ND</td></tr>
    <tr><td align="center">value</td><td align="center">scalar</td><td align="center">float32,float16,int32,bfloat16</td><td align="center">-</td></tr>

    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,int32,bfloat16</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">addcmul</td></tr>
  </table>

## 约束与限制

- input_data, x1, x2, y, value, out的数据类型只支持float32,float16,int32,bfloat16，数据格式只支持ND

### 运行验证

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_addcmul.cpp"> test_aclnn_addcmul.cpp</a></td><td>通过aclnn调用的方式调用Addcmul算子。</td>
    </tr>
</table>
