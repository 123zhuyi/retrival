# DropOutV3Grad

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：反向专用Dropout算子，训练过程中，根据mask中对应bit位的值，将输入gradY中的元素按照scale放大或者置零。
- 计算公式：

  $$
  gradX_i = gradY_i * mask_i * scale
  $$

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
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
      <td>grad_y</td>
      <td>输入</td>
      <td>公式中的输入gradY_i，反向梯度输入，shape支持0-8维。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>公式中的输入mask_i，对应gradY第i个元素的掩码位，取值为0（丢弃）或1（保留）。bit类型并使用UINT8类型存储，shape需要为(align(grad_y的元素个数,128)/8)。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>公式中的输入scale，用于计算输出数据缩放比例的缩放因子。不做范围校验，按gradY*mask*scale直接计算；正常业务下scale来自前向1/(1-p)，即为0或大于等于1。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>公式中的gradX_i，反向梯度输出，数据类型需要是grad_y可转换的数据类型，shape需要与grad_y一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. scale不做范围校验，任意值均按gradY$\times$mask$\times$scale直接计算；正常业务下scale来自前向1/(1-p)，即为0或大于等于1。
2. mask的数据类型为UINT8，其shape必须满足条件：

   $$
   \text{mask\_shape} = \frac{\text{align}(\text{num}(grad\_y), 128)}{8}
   $$

3. 数据维度支持0-8维。

## 调用说明

| 调用方式  | 调用样例                                                                         | 说明                                                                                |
| --------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_drop_out_v3_grad](./examples/arch35/test_aclnn_drop_out_v3_grad.cpp) | 通过[aclnnDropoutV3Grad](docs/aclnnDropoutV3Grad.md)接口方式调用DropOutV3Grad算子。 |
