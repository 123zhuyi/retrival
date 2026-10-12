# Rsqrt

## 支持的产品型号

- Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述

- 功能描述

  `Rsqrt`算子将数据进行开方并取倒数运算。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Rsqrt</th></tr>
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td rowspan="2" align="center">算子输入</td></tr>

    <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,int8,int16,int32,uint8,bool</td><td align="center">ND</td></tr>

    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,int8,int16,int32,uint8,bool</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">rsqrt</td></tr>
  </table>

## 约束与限制

- x，y，out的数据类型只支持 float32,float16,bfloat16,int8,int16,int32,uint8,bool，数据格式只支持ND

### 运行验证

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/test_aclnn_rsqrt.cpp"> test_aclnn_rsqrt.cpp</a></td><td>通过aclnn调用的方式调用Rsqrt算子。</td>
    </tr>
</table>

## 贡献说明

| 贡献者       | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容                         |
|-----------|------------------|-------|-----------|------------------------------|
| skywang2 | 个人开发者 | Rsqrt | 2026/03/30 | 新增Rsqrt算子 |
| Nice_try | 个人开发者 | Rsqrt | 2026/07/06 | Rsqrt新增支持整型数据支持（A2/A3） |
