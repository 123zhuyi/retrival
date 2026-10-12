# Log

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |     ×    |

## 功能说明

- 算子功能：对输入张量x的元素，逐元素进行对数计算，并将结果保存到输出张量y中。可支持多种类型输入张量，并支持对输入张量x进行缩放（shift/scale）操作。
- 计算公式：

    $$
    y = log_{base}(shift + scale * x)
    $$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
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
      <td>x</td>
      <td>输入</td>
      <td>输入张量，公式中的x。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、FLOAT16、FLOAT、DOUBLE <br>
      BOOL、BFLOAT16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>base</td>
      <td>可选属性</td>
      <td><ul><li>对数运算的底。</li><li>默认为-1.0（表示自然对数e）。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选属性</td>
      <td><ul><li>输入张量x的缩放因子。</li><li>默认为1.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>可选属性</td>
      <td><ul><li>输入张量x的偏移量。</li><li>默认为0.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算对数后的输出张量，公式中的y。</td>
      <td>当输入类型为整数类型（INT类型）时，输出类型为FLOAT；<br>
      其他情况与输入张量x相同</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- base参数必须大于0；默认值-1表示以e为底数。
- 输入值在特定范围内(0,0.01]或[0.95,1.05)时，输出精度可能不稳定
- 整数输入会自动转换为FLOAT类型输出

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn调用 | [test_aclnn_log](./examples/test_aclnn_log.cpp) <br> [test_aclnn_inplace_log](./examples/test_aclnn_inplace_log.cpp) | 通过[aclnnLog或aclnnInplaceLog](./docs/aclnnLog&aclnnInplaceLog.md)接口方式调用Log算子。    |
| aclnn调用 | [test_aclnn_log2](./examples/test_aclnn_log2.cpp) <br> [test_aclnn_inplace_log2](./examples/test_aclnn_inplace_log2.cpp) | 通过[aclnnLog2或aclnnInplaceLog2](./docs/aclnnLog2&aclnnInplaceLog2.md)接口方式调用Log算子。    |
| aclnn调用 | [test_aclnn_log10](./examples/test_aclnn_log10.cpp) <br> [test_aclnn_inplace_log10](./examples/test_aclnn_inplace_log10.cpp) | 通过[aclnnLog10或aclnnInplaceLog10](./docs/aclnnLog10&aclnnInplaceLog10.md)接口方式调用Log算子。    |
| 图模式调用 | [test_geir_log](./examples/test_geir_log.cpp)   | 通过[算子IR](./op_graph/log_proto.h)构图方式调用Log算子。 |
