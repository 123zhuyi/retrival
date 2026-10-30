# ExpSegsumGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：[Segsum](../segsum/README.md)的反向计算。
- 计算公式（以5D输入为例）:

  1. 输入gradOutput（N1,N2,N3,N4,N4）与输入gradSelf（正向的输出）相乘。
     
     $$
     out\_mul = gradOutput * gradSelf
     $$
  2. 生成（N4,N4）类型为bool的三角矩阵A，上三角为True，下三角为False，对角线为True。
  3. 用0填充输入$out\_mul$里面与矩阵A中值为True的位置相对应的元素。
    
     $$
     out\_mul_i=
     \begin{cases}out\_mul_i,\quad A_i==False
     \\0, \quad A_i==True
     \end{cases}
     $$
  4. 对out_mul的倒数第二维进行倒序生成out_flip。
  5. 以$out\_flip$的倒数第二维进行cumsum累加。从维度视角来看的某个元素（其它维度下标不变，当前维度下标依次递增），$out\_cumsum\_{i}$是输出张量中对应位置的元素。
     
     $$
     out\_cumsum_{i} = out\_flip_{1} + out\_flip_{2} + out\_flip_{3} + ...... + out\_flip_{i}
     $$
  6. 对$out\_cumsum$的-2维进行倒序生成out_flip2。
  7. 生成（N4,N4）类型为bool的三角矩阵B，上三角为True，下三角为False，对角线为True。
  8. 用0填充$out\_flip2$里面与矩阵B中值为True的位置相对应的元素。
    
     $$
     out\_flip2_i=
     \begin{cases}out\_flip2_i,\quad B_i==False
     \\0, \quad B_i==True
     \end{cases}
     $$
  9. 返回gradInput为out\_flip2最后一维每行的和。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>grad_output</td>
      <td>输入</td>
      <td>表示进行反向计算的梯度，对应公式中的`gradOutput`。输入维度支持4D或5D，且最后两维的维度大小相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_self</td>
      <td>输入</td>
      <td>表示正向计算的输出，对应公式中的`gradSelf`。shape和数据类型与输入`grad_output`保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示反向计算的输出，对应公式中的`out`。输出维度必须比输入维度少一维，支持3D或4D。当输入`grad_output`为4D时，输出的维度大小与`grad_output`的前3维保持一致；当输入`grad_output`为5D时，输出的维度大小与`grad_output`的前4维保持一致。数据类型与输入`grad_output`保持一致。
      </td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_exp_segsum_backward](examples/test_aclnn_exp_segsum_backward.cpp) | 通过[aclnnExpSegsumBackward](docs/aclnnExpSegsumBackward.md)接口方式调用ExpSegsumGrad算子。 |
