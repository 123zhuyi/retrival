# SubMulConcat

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

SubMulConcat是内部融合算子，由CANN既有图融合流程将Sub、Mul和Concat/ConcatV2/ConcatV2D组成的原始图融合生成；该算子没有公开ACLNN接口，也不支持用户直接构造SubMulConcat节点。

- `[general]` 本目录保留内部OpDef和InferShape，仅用于图融合后的内部节点；本次不交付融合规则，也不新增公开REG_OP或ACLNN接口。
- `[existing]` Atlas A2对应的Ascend910B实现继续由canndev维护；本次不将其arch32 Tiling、Kernel或UT迁入本目录。
- `[RegBase-native]` 本目录新增且仅交付Ascend950的arch35 Host Tiling和基于RegTensor的Kernel实现。

对于数据类型均为float32、数据格式均为ND且形状均为$[N,H,W]$的输入$x$和$y$，计算公式为：

$$
z = \operatorname{Concat}\left([x,\ y,\ x-y,\ x\odot y],\ \mathrm{axis}=-1\right),
\qquad z\in\mathbb{R}^{N\times H\times 4W}.
$$

其中$\odot$表示逐元素乘法。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>公式中的输入张量$x$，形状为$[N,H,W]$。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输入</td>
<td>公式中的输入张量$y$，形状为$[N,H,W]$且与$x$相同。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>z</td>
<td>输出</td>
<td>公式中的输出张量$z$，形状为$[N,H,4W]$。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>axis</td>
<td>可选属性</td>
<td>内部拼接轴，默认值为-1。源图的拼接轴可为-1或2，融合后统一规范化为-1。</td>
<td>INT64</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- 输入`x`和`y`仅支持FLOAT数据类型和ND格式，必须为形状相同的三维张量$[N,H,W]$。
- 在既有图融合规则下，原始图必须按`x`、`y`、`x-y`、`x*y`的顺序拼接4个输入，Concat/ConcatV2/ConcatV2D的拼接轴必须为-1或2。
- Sub和Mul的输出只能由该拼接节点消费，否则既有图融合规则不会生成SubMulConcat节点。
- SubMulConcat仅供既有图融合流程生成，没有公开ACLNN接口，不支持用户直接构造其内部节点。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 仅支持由CANN既有图融合流程将公开的Sub、Mul和Concat/ConcatV2/ConcatV2D原始图转换为内部节点；本目录不交付融合规则，也不提供SubMulConcat直接构造接口。 |

SubMulConcat不提供aclnn API或PyTorch API，也不提供GE图模式下直接构造内部节点的接口。
