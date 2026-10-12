# SwinTransformerLnQKV

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Kirin X90 处理器系列产品</term> | √ |
|<term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：完成fp16权重场景下的Swin Transformer 网络模型的Q、K、V 的计算。

- 计算公式：

    $$
    (Q,K,V)=((Layernorm(inputX)).transpose() * weight).transpose().split()
    $$  

  其中，weight 是 Q、K、V 三个矩阵权重的拼接。
  
## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>inputX</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr> 
  <tr>
    <td>query_output</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key_output</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value_output</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr> 
</tbody>
</table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT16。

## 约束说明

- 当前不支持用户直接调用
