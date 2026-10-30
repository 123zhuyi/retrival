# Equal

## 贡献说明

| 贡献者   | 贡献方              | 贡献算子  | 贡献时间      | 贡献内容      |
|-------|------------------|-------|-----------|-----------|
| skywang2 | 个人开发者 | Equal | 2026/6/21 | 新增Equal算子 |

## 支持的产品型号

- Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 算子描述

- 功能描述

  `Equal`算子将输入的两个向量数据进行各对应位置的判等运算，返回结果向量。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Equal</th></tr>
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td rowspan="2" align="center">算子输入</td><td align="center">x1</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,uint8,int8,uint32,int32</td><td align="center">ND</td></tr>
    <tr><td align="center">x2</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,uint8,int8,uint32,int32</td><td align="center">ND</td></tr>
    <tr><td align="center">算子输出</td><td align="center">y</td><td align="center">tensor</td><td align="center">bool</td><td align="center">ND</td></tr>
    <tr><td align="center">核函数名</td><td colspan="4" align="center">equal</td></tr>
  </table>

## 约束与限制

- x,y,out的数据类型仅支持float32,float16,bfloat16,uint8,int8,uint32,int32，数据格式仅支持ND

### 运行验证

测试命令调用方式：[build.sh](../../../docs/zh/invocation/quick_op_invocation.md)
<table>
    <tr><th>目录</th><th>描述</th></tr>
    <tr>
        <td><a href="./examples/test_aclnn_equal.cpp">test_aclnn_equal.cpp</a></td><td>通过aclnn调用的方式调用Equal算子。</td>
    </tr>
</table>
