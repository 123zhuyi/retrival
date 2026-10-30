# AdaCast

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：面向 HDRnet（High Dynamic Range Network）实时图像增强网络的输入预处理算子。将传感器输出的 uint16 HDR 图像数据转换为 float16 张量，并按白电平（pixel）完成归一化，等价于把「整型→浮点类型转换」与「按除数缩放」两步融合为单次逐元素运算。
- 计算公式：

$$
out_{i} = \mathrm{Cast\_FP16}^{sat+round}\bigl(\mathrm{Cast\_FP32}(self_{i}) \times (1 / pixel)\bigr)
$$

其中末步采用饱和（clip 到 fp16 值域 [-65504, 65504]）+ 四舍五入（round-to-nearest-even）方式转换到 float16。

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
      <td>x</td>
      <td>输入</td>
      <td>待归一化的 HDR uint16 图像数据，公式中的self_i，shape 支持 1D~4D。</td>
      <td>UINT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pixel</td>
      <td>可选属性</td>
      <td><ul><li>白电平归一化基数，缩放系数为 1/pixel。</li><li>默认值为 65535。</li><li>取值必须为正整数（pixel &gt; 0）。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>归一化后的 float16 张量，公式中的out_i，shape 与 x 完全一致。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 数据格式仅支持 ND。
- 输入 x 的 rank 范围为 [1, 4]，输出 y 的 shape 必须与 x 完全一致（element-wise 语义，不支持广播）。
- 类型组合固定为 UINT16 → FLOAT16，不支持其他 dtype 组合。
- 属性 pixel 必须为正整数（pixel > 0），否则返回 ACLNN_ERR_PARAM_INVALID。
- 支持空 Tensor（0 元素），直接返回。
- 默认确定性实现，相同输入始终产生相同输出。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_ada_cast](./examples/test_geir_ada_cast.cpp) | 通过 GE-IR 构图方式调用 AdaCast 算子。 |
