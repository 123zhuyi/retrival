# Pack

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：通过沿轴维度打包，将值中的张量列表打包成一个比值中的每个张量高一维度的张量。给定形状为（A，B，C）的张量的长度为N的列表；如果轴==0，则输出张量将具有形状（N，A，B，C）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
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
      <td>x</td>
      <td>输入</td>
      <td>需要级联的tensor列表。</td>
      <td>COMPLEX128、COMPLEX64、BOOL、DOUBLE、FLOAT32、FLOAT16、BFLOAT16、INT16、INT32、INT64、INT8、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td>指定沿其打包的维度。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>N</td>
      <td>可选属性</td>
      <td>指定要打包的tensor个数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>COMPLEX128、COMPLEX64、BOOL、DOUBLE、FLOAT32、FLOAT16、BFLOAT16、INT16、INT32、INT64、INT8、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* Atlas 训练系列产品、Atlas 推理系列产品：不支持BFLOAT16。

## 约束说明

无。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_stack](examples/test_aclnn_stack.cpp) | 通过[aclnnStack](docs/aclnnStack.md)接口方式调用Pack算子。 |
