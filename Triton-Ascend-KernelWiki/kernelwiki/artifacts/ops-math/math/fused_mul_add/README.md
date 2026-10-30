# FusedMulAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将`Mul`、`Add`子图融合为单个算子，对三个输入按NumPy广播规则对齐后逐元素计算乘加。

- 计算公式：

  $$
  y = x_1 \times x_2 + x_3
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的乘法输入张量x1。</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的乘法输入张量x2，shape需与x1可广播。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>公式中的加法输入张量x3，shape需与x1*x2的结果可广播。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y，shape为x1、x2、x3广播后的统一形状。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x1、x2、x3、y必须为同一种数据类型，不支持混合数据类型。
- 支持任意NumPy广播形态（含标量、单维broadcast、跨rank broadcast），支持动态shape与动态rank。

## 实现方案

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 计算图原型 | `op_graph/fused_mul_add_proto.h` | `REG_OP(FusedMulAdd)`，三输入一输出 |
| 算子定义 | `op_host/fused_mul_add_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/fused_mul_add_infershape.cpp` | 复用`Ops::Base::InferShape4Broadcast(ctx, 3)` |
| Tiling | `op_host/arch35/fused_mul_add_tiling_arch35.{h,cpp}` | 按dtype分支调用`Ops::Base::BroadcastBaseTiling<OpDag>` |
| DAG | `op_kernel/arch35/fused_mul_add_dag.h` | fp32/fp16通路在fp32中间精度下用`Vec::Mul + Vec::Add`；int32通路用`Vec::Mul + Vec::Add` |
| Struct | `op_kernel/arch35/fused_mul_add_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel入口 | `op_kernel/fused_mul_add_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp16 / fp32通路

```pseudocode
In0/In1/In2 -- CopyInBrc -- Cast(->fp32) -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- Cast(->T,RINT) -- CopyOut -- Out0
```

全部输入先Cast到fp32，再用`Vec::Mul + Vec::Add`两段式按公式
`y = x1 * x2 + x3`顺序计算，最后Cast回T写出。

> 此处不使用`Vec::FusedMulAdd`。该API底层实现会in-place写回src2 buffer，
> 在broadcast大tensor跨tile复用输入UB buffer时会污染下一tile的输入，
> 导致fp32/fp16 huge broadcast场景精度错误。使用普通`Vec::Mul + Vec::Add`
> 保证所有placeholder派生的cast结果只读。

### int32通路

```pseudocode
In0/In1/In2 -- CopyInBrc -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- CopyOut -- Out0
```

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式 | [test_geir_fused_mul_add](examples/arch35/test_geir_fused_mul_add.cpp) | 通过[算子IR](op_graph/fused_mul_add_proto.h)构图方式调用FusedMulAdd算子。 |
