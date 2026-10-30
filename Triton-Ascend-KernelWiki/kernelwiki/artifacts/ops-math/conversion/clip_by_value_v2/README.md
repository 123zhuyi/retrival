# ClipByValueV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能: 将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于clipValueMax则限制为clipValueMax，若元素小于clipValueMin则限制为clipValueMin，否则等于元素本身。
- 计算公式：

$$
{y}_{i} = max(min({{x}_{i}},{max\_value}_{i}),{min\_value}_{i})
$$

## 参数说明

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
      <td>x</td>
      <td>输入</td>
      <td>输入张量。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64 <br> 
      FLOAT16、FLOAT、DOUBLE、BOOL、BFLOAT16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_value_min</td>
      <td>输入</td>
      <td>限制元素范围的最小值。</td>
      <td>同输入张量x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_value_max</td>
      <td>输入</td>
      <td>限制元素范围的最大值。</td>
      <td>同输入张量x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>将输入张量x裁剪后的输出张量。</td>
      <td>同输入张量x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn调用 | [test_aclnn_clamp](./examples/test_aclnn_clamp.cpp) | 通过[aclnnClamp](./docs/aclnnClamp.md)接口方式调用clip_by_value_v2算子。    |
| aclnn调用 | [test_aclnn_clamp_max](./examples/test_aclnn_clamp_max.cpp) <br> [test_aclnn_inplace_clamp_max](./examples/test_aclnn_inplace_clamp_max.cpp) | 通过[aclnnClampMax&aclnnInplaceClampMax](./docs/aclnnClampMax&aclnnInplaceClampMax.md)接口方式调用clip_by_value_v2算子。    |
| aclnn调用 | [test_aclnn_clamp_max_tensor](./examples/test_aclnn_clamp_max_tensor.cpp) <br> [test_aclnn_inplace_clamp_max_tensor](./examples/test_aclnn_inplace_clamp_max_tensor.cpp) | 通过[aclnnClampMaxTensor&aclnnInplaceClampMaxTensor](./docs/aclnnClampMaxTensor&aclnnInplaceClampMaxTensor.md)接口方式调用clip_by_value_v2算子。    |
| aclnn调用 | [test_aclnn_clamp_min](./examples/test_aclnn_clamp_min.cpp) | 通过[aclnnClampMin](./docs/aclnnClampMin.md)接口方式调用clip_by_value_v2算子。    |
| aclnn调用 | [test_aclnn_clamp_min_tensor](./examples/test_aclnn_clamp_min_tensor.cpp) <br> [test_aclnn_inplace_clamp_min_tensor](./examples/test_aclnn_inplace_clamp_min_tensor.cpp) | 通过[aclnnClampMinTensor&aclnnInplaceClampMinTensor](./docs/aclnnClampMinTensor&aclnnInplaceClampMinTensor.md)接口方式调用clip_by_value_v2算子。    |
| aclnn调用 | [test_aclnn_clamp_tensor](./examples/test_aclnn_clamp_tensor.cpp) | 通过[aclnnClampTensor](./docs/aclnnClampTensor.md)接口方式调用clip_by_value_v2算子。    |
| aclnn调用 | [test_aclnn_hardtanh](./examples/test_aclnn_hardtanh.cpp) <br> [test_aclnn_inplace_ardtanh](./examples/test_aclnn_inplace_ardtanh.cpp) | 通过[aclnnHardtanh&aclnnInplaceHardtanh](./docs/aclnnHardtanh&aclnnInplaceHardtanh.md)接口方式调用clip_by_value_v2算子。    |
| 图模式调用 | [test_geir_clip_by_value_v2](./examples/test_geir_clip_by_value_v2.cpp)   | 通过[算子IR](./op_graph/clip_by_value_v2_proto.h)构图方式调用clip_by_value_v2算子。 |
