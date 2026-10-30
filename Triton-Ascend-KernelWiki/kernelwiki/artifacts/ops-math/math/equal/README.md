# Equal

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |   ×    |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √    |

## 功能说明

- 算子功能：计算两个Tensor是否有相同的大小和元素，返回一个Bool类型。
- 计算表达式：

  $$
  out = (self == other)  ?  True : False
  $$

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>公式中的输入self。</td>
      <td>FLOAT、BFLOAT16、FLOAT16、INT32、INT8、UINT8、DOUBLE、INT16、INT64、COMPLEX64、COMPLEX128、QUINT8、QINT8、QINT32、STRING、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入other。</td>
      <td>FLOAT、BFLOAT16、FLOAT16、INT32、INT8、UINT8、DOUBLE、INT16、INT64、COMPLEX64、COMPLEX128、QUINT8、QINT8、QINT32、STRING、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），场景为最后2轴合轴小于16，前面的轴合轴超大。

## 调用说明

| 调用方式 | 调用样例                                                   | 说明                                               |
|--------------|--------------------------------------------------------|--------------------------------------------------|
| aclnn调用 | [test_aclnn_equal.cpp](../tensor_equal/examples/test_aclnn_equal.cpp) | 通过[aclnnEqual](../tensor_equal/docs/aclnnEqual.md)接口方式调用Equal算子。 |
| 图模式调用 | [test_geir_equal.cpp](examples/test_geir_equal.cpp) | 通过[算子IR](op_graph/equal_proto.h)构图方式调用Equal算子。   |
