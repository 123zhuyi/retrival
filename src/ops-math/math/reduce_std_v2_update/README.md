# ReduceStdV2Update

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

> 产品与芯片映射关系参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。其中 <term>Ascend 950PR/Ascend 950DT</term> 为本次新增 Ascend C Kernel 实现的目标平台；其余支持产品由已有 TBE binary 覆盖。

## 功能说明

- 算子功能：接收原始输入 `x` 和已计算好的均值 `mean`（已广播到 `x` 的 shape），沿指定维度 `dim` 归约计算平方偏差和，再除以 `N - correction` 得到方差或标准差。支持贝塞尔校正（correction）与 keepdim。ReduceStdV2Update 是 CANN l0op 内部算子，被 `aclnnVar`/`aclnnVarCorrection`/`aclnnVarMean` 在非 regbase 路径下内部调用，使 `aclnnVarMean` 可复用 mean 结果避免重复求均值。
- 计算公式：

  $$
  \text{diff} = x - \mu
  $$

  $$
  \text{sq\_sum} = \sum_{i \in D} \text{diff}_{i}^{2}
  $$

  $$
  \text{var} = \frac{\text{sq\_sum}}{N - \text{correction}}
  $$

  $$
  \text{output} = \begin{cases} \sqrt{\text{var}} & \text{if } \text{if\_std} = \text{true} \\ \text{var} & \text{if } \text{if\_std} = \text{false} \end{cases}
  $$

  其中 $D$ 为归约维度集合，$N$ 为归约维度元素个数，$\mu$ 为外部传入的均值，correction=0 为有偏估计（分母 N），correction=1 为无偏估计（分母 N-1）。

- 接口间区别：本算子提供两个等价 L0 接口。`ReduceStdV2Update` 使用 `bool unbiased` 参数（true=无偏 N-1，false=有偏 N），被 `aclnnVar` 调用；`ReduceStdV2UpdateCorrection` 使用 `int64_t correction` 参数（0=有偏，1=无偏），被 `aclnnVarCorrection` 调用。两者数学等价，仅参数表达方式不同。L0 接口固定输出方差（if_std=false）；GE IR 图模式可通过 `if_std` 属性控制输出方差或标准差。

## 参数说明

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 280px">
  <col style="width: 150px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（self）</td>
      <td>输入</td>
      <td>原始输入数据，对应公式中 x。0-8 维 ND。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>已计算好的均值，对应公式中 μ，须已 Expand 广播到 self 的 shape。dtype 与 self 一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>归约维度列表，对应公式中 D，支持负索引。dim 为空时归约所有维度。</td>
      <td>INT64（ListInt）</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>if_std</td>
      <td>属性</td>
      <td>false=输出方差，true=输出标准差（开方）。默认 false。aclnn 接口固定 false。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unbiased</td>
      <td>属性</td>
      <td>是否使用贝塞尔校正。true=无偏（N-1），false=有偏（N）。默认 true。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>属性</td>
      <td>是否保留归约维度。true=设为 1，false=移除。默认 false。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>correction</td>
      <td>属性</td>
      <td>贝塞尔校正因子。0=有偏（N），1=无偏（N-1）。默认 1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output（output_var）</td>
      <td>输出</td>
      <td>方差或标准差，对应公式中 output。dtype 与 x 一致，shape 为 x 沿 dim 归约后的 shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入数据类型限制：`x` 与 `mean` 仅支持 FLOAT、FLOAT16、BFLOAT16，且二者数据类型须一致；不支持 DOUBLE、复数、整型。
- shape 约束：`mean` 必须已通过 Expand 广播到 `x` 的 shape；output shape 为 `x` 沿 `dim` 归约后的 shape（keepdim=true 维度设 1，false 移除）。
- correction/unbiased 约束：仅支持 correction=0（有偏）和 correction=1（无偏）两种语义；correction>1 由上层 `aclnnVarCorrection` 处理。
- 边界情况：空 Tensor、单元素且 correction≥1 的场景由上层 `aclnnVar`/`aclnnVarCorrection` 提前拦截返回 NAN/INF，不会到达本算子；kernel 实现仍需防御性处理这些边界。
- 精度约束（950 实现严格对齐 canndev 原型）：
  - FP16/BF16 输入固定提升到 FP32 累加（固定提升，无 GetPromoteType），FP32 输入直接计算；结果转回原 dtype。
  - BF16 输出回转使用 round 模式（对应 AscendC Cast round），保留 BF16 舍入语义。
  - if_std=true 时使用 high_precision vsqrt 开方。
  - 输出 dtype 与输入 x 一致，不在接口层做类型提升。
- 确定性说明：本算子默认确定性实现。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| GE图模式 | [test_geir_reduce_std_v2_update](./examples/arch35/test_geir_reduce_std_v2_update.cpp) | 通过[算子IR](./op_graph/reduce_std_v2_update_proto.h)构图方式调用ReduceStdV2Update算子，支持if_std属性控制输出方差或标准差。 |

> 注：本算子为l0op内部子算子，不对外暴露独立aclnn接口；L0接口固定if_std=false输出方差，GE IR图模式可通过if_std属性控制输出方差或标准差。上层`aclnnVar`/`aclnnVarCorrection`/`aclnnVarMean`在非regbase路径下内部调用本算子。
