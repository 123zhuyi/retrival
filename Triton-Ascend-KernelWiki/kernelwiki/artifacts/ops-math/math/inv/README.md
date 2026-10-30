# Inv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量的每个元素逐元素计算倒数。

- 计算公式：

$$
out_{i}=\frac{1}{self_{i}}
$$

- 浮点（FLOAT、FLOAT16、BFLOAT16）按IEEE-754真值倒数计算（除零得 $\pm\text{Inf}$、$\pm\text{Inf}$ 得 $\pm0$、NaN得NaN）。

- INT32按截断向零的整数倒数计算，整型下 $1/self$ 仅当 $|self|=1$ 时非零，等价为三值映射：

$$
out_{i}=
\begin{cases}
+1 & self_{i}=+1 \\
-1 & self_{i}=-1 \\
0  & \text{其他（含 } self_{i}=0、|self_{i}|\ge 2、\text{INT\_MIN}、\text{INT\_MAX)}
\end{cases}
$$

  其中 $self_{i}=0$ 映射为 $0$ 且不抛异常，$\text{INT\_MIN}(-2147483648)$、$\text{INT\_MAX}(2147483647)$ 均映射为 $0$ 且不溢出。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>self</td>
      <td>输入</td>
      <td>待进行inv计算的入参，公式中的self。支持空Tensor、非连续Tensor，shape维度0-8。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行inv计算的出参，公式中的out。shape和数据类型与self一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- self与out必须shape、dtype相同，逐元素计算，无广播。
- 浮点路径精度按社区标准：FLOAT rtol/atol = 1e-4，FLOAT16 = 1e-3，BFLOAT16 = 4e-3。
- INT32路径为精确匹配（bitwise，rtol = atol = 0），采用纯整型比较 + 选择实现三值映射，不经float32中转、不对INT_MIN取负，故INT_MIN/INT_MAX不溢出、self = 0不抛异常。

## 调用说明

| 调用方式 | 说明                                                           |
|--------------|--------------------------------------------------------------|
| 图模式调用 | 通过[算子IR](op_graph/inv_proto.h)构图方式调用Inv算子。样例参见[test_geir_inv](./examples/test_geir_inv.cpp)。 |
