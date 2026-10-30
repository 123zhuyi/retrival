# MirrorPad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                        |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                        |    ×     |
| <term>Atlas 推理系列产品</term>                                |    √     |
| <term>Atlas 训练系列产品</term>                                |    √     |

## 功能说明

- 算子功能：以镜像方式填充输入tensor的边界。
- 示例：

  ```text
  输入tensor([[0,1,2]])
  paddings([2,2])

  1.mode("REFLECT")
  输出为([[2,1,0,1,2,1,0]])

  2.mode("SYMMETRIC")
  输出为([[1,0,0,1,2,2,1]])
  ```

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
      <td>待进行镜像扩充的原始tensor。</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BF16, FLOAT16, FLOAT, DOUBLE, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>扩充的配置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>填充模式，可以是反射"REFLECT"或对称"SYMMETRIC"。</td>
      <td>string</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行扩充后的tensor。</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BF16, FLOAT16, FLOAT, DOUBLE, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- paddings  
  paddings的第0维表示对输入x第0维的扩充配置，以此类推。每一行表示对应维度上的填充数量（左/右、前/后、上/下等）。对于每一行[a, b]，a表示在该维度的开头填充的元素数，b表示在该维度的末尾填充的元素数。

- mode  
  REFLECT模式下，镜像时不会包括边界本身。
  SYMMETRIC模式下，镜像时会包含边界本身，示例中可见。

## 约束说明

paddings的形状需要为[rank, 2]，其中rank为输入x的维度数。
对于每一行paddings[i]，填充元素数不得超过其对应x[i]的元素数。即在反射模式下，paddings[i][0]和paddings[i][1] <= x[i].length-1；在对称模式下，paddings[i][0]和paddings[i][1] <= x[i].length。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----|----|----|
| aclnn调用 | [test_aclnn_reflection_pad_1d](./examples/test_aclnn_reflection_pad_1d.cpp) | 通过[aclnnReflectionPad1d](./docs/aclnnReflectionPad1d.md)接口方式调用REFLECT情况下的mirror_pad算子，填充输入tensor的最后一维。 |
| aclnn调用 | [test_aclnn_reflection_pad_2d](./examples/test_aclnn_reflection_pad_2d.cpp) | 通过[aclnnReflectionPad2d](./docs/aclnnReflectionPad2d.md)接口方式调用REFLECT情况下的mirror_pad算子，填充输入tensor的最后两维。 |
| aclnn调用 | [test_aclnn_reflection_pad_3d](./examples/test_aclnn_reflection_pad_3d.cpp) | 通过[aclnnReflectionPad3d](./docs/aclnnReflectionPad3d.md)接口方式调用REFLECT情况下的mirror_pad算子，填充输入tensor的最后三维。 |
