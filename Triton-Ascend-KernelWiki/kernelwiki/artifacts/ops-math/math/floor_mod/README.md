# FloorMod

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能: 将scalar self进行broadcast成和tensor other一样shape的tensor以后，其中的每个元素都转换为除以other的对应元素以后得到的余数。该结果与除数other同符号，并且该结果的绝对值是小于other的绝对值。
  实际计算remainder(self, other)等效于以下公式：

  $$
  out_i = self - floor(self / other_i) * other_i
  $$

- 示例：

```text
self = 5.0   # float
other = tensor([[-1, -2],
                [-3, -4]]).type(int32)
result = remainder(self, other)

# result的值
# tensor([[ 0., -1.],
#         [-1., -3.]])  float

# 对于元素other中的-4来说，计算结果为5 - floor(5 / -4) * -4 = -3
# 可以看到，最终结果-3的绝对值小于原来的-4的绝对值。
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入张量self_i</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入张量other_i</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量out_i</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_remainder_scalar_tensor](./examples/test_aclnn_remainder_scalar_tensor.cpp) | 通过[aclnnRemainderScalarTensor](./docs/aclnnRemainderScalarTensor.md)接口方式调用FloorDiv算子  |
