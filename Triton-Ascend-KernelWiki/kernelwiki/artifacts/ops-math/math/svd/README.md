# Svd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：计算一个或多个矩阵的奇异值分解。

  当输入张量的维度大于2时，会将高维张量视为一批矩阵进行处理。对于形状为(..., M, N)的输入张量，将倒数第二维之前的维度(...)视为批处理维度，对每个(M, N)矩阵独立进行奇异值分解计算。

- 计算公式：

$$
\mathbf{input} = \mathbf{U} \times \mathrm{diag}(\boldsymbol{sigma}) \times \mathbf{V}^T
$$

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>input</td>
      <td>输入</td>
      <td>需要进行奇异值分解的张量，对应公式中的input。<br>不支持空Tensor。shape维度至少为2。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>输出</td>
      <td>输出张量，对应公式中的sigma。<br>shape需要根据input的shape及fullMatrices参数进行推导。数据类型需要和input保持一致。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>U</td>
      <td>输出</td>
      <td>输出张量，对应公式中的U。<br>shape需要根据input的shape及fullMatrices参数进行推导。数据类型需要和input保持一致。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>V</td>
      <td>输出</td>
      <td>输出张量，对应公式中的V。<br>shape需要根据input的shape及fullMatrices参数进行推导。数据类型需要和input保持一致。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>compute_uv</td>
      <td>属性</td>
      <td>可选属性,默认true。表示是否计算输出张量u和v。当设为true时，输出张量sigma、u、v均会被计算。当设为false时，只计算sigma。</td>
      <td>BOOL</td>
      <td>true/false</td>
    </tr>
     <tr>
      <td>full_matrices</td>
      <td>属性</td>
      <td>可选属性，表示是否完整计算输出张量u和v，默认false</td>
      <td>BOOL</td>
      <td>true/false</td>
    </tr>
  </tbody></table>

## 约束说明
  
  无
  
## 调用说明
  
  | 调用方式 | 调用样例                                                          | 说明                                                          |
  |--------|---------------------------------------------------------------|-------------------------------------------------------------| 
  | aclnn调用 | [test_aclnn_svd](./examples/test_aclnn_svd.cpp) | 通过[aclnnSvd](./docs/aclnnSvd.md)接口方式调用Svd算子。|  
