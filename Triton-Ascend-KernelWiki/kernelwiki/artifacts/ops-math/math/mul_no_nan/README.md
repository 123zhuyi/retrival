# MulNoNan

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

- 算子功能：完成安全乘法计算，当x2为0时返回0，从而屏蔽`0 * inf = NaN`、`0 * NaN = NaN`两类异常。等价于TensorFlow的`tf.math.multiply_no_nans`。

- 计算公式：

  $$
  y = \begin{cases}
  0, & \text{if } x2 = 0 \\
  x1 \times x2, & \text{if } x2 \neq 0
  \end{cases}
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
      <td>FLOAT16, FLOAT, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的乘法输入张量x2，作为判0主体，shape需与x1可广播。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y，shape为x1、x2广播后的统一形状。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x1、x2、y必须为同一种数据类型，不支持混合数据类型。
- 仅判x2一侧，`x2 = -0`同样进入零臂（IEEE-754 `-0 == 0`）；`x2 != 0`时所有IEEE行为按普通乘法保留（NaN / Inf正常传播）。
- 支持任意NumPy广播形态（含标量、单维broadcast、跨rank broadcast），支持动态shape与动态rank。

## 实现方案

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 计算图原型 | `op_graph/mul_no_nan_proto.h` | `REG_OP(MulNoNan)`，二输入一输出 |
| 算子定义 | `op_host/mul_no_nan_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/mul_no_nan_infershape.cpp` | 复用`Ops::Base::InferShape4Broadcast` |
| Tiling | `op_host/arch35/mul_no_nan_tiling_arch35.{h,cpp}` | 按dtype分支调用`Ops::Base::BroadcastBaseTiling<OpDag>` |
| DAG | `op_kernel/arch35/mul_no_nan_dag.h` | fp32/int32通路原生计算；fp16/bf16通路提升fp32中间精度 |
| Struct | `op_kernel/arch35/mul_no_nan_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel入口 | `op_kernel/mul_no_nan_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp32 / int32通路

```pseudocode
In0/In1 -- CopyInBrc --+
                       +-- Mul(x1,x2) --+
                                        +-- Select(mask, MulRes, Zero) -- CopyOut -- Out0
            Const(0)  -- Duplicate -- Zero --+
                                             |
                       +-- Compare(NE, x2, Zero) -- mask
```

`Vec::Compare<u8, T, NE>`输出位掩码，`Vec::Select<u8, T, TENSOR_TENSOR>`按
`mask ? MulRes : Zero`逐元素选择。`x2 == 0`时即使`MulRes`是`NaN`（`0 · inf`
/ `0 · NaN`）也被Select丢弃，输出0。

### fp16 / bf16通路

```pseudocode
In0/In1 -- CopyInBrc -- Cast(->fp32) --+
                                       +-- Mul --+
                                                 +-- Select -- Cast(->T,RINT) -- CopyOut -- Out0
            Const(0,fp32) -- Duplicate -- Zero --+
                                                 |
                              +-- Compare(NE) -- mask
```

把fp16 / bf16整体提升到fp32做cmp/sel/mul，末端用`CAST_MODE_RINT`
（round-to-nearest-even）回退。原因：

1. 与DSL `mul_no_nan.py`在fp16 `vcmpsel`不可用时fallback到fp32的行为一致；
2. 避免fp16中间溢出（如`3e4 · 3e4`在fp16中先inf再被select处理会
   引入额外的saturation不确定性），fp32中间有充足动态范围。

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式 | [test_geir_mul_no_nan](examples/arch35/test_geir_mul_no_nan.cpp) | 通过[算子IR](op_graph/mul_no_nan_proto.h)构图方式调用MulNoNan算子。 |
