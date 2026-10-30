# CompareAndBitpack

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                   |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    √     |
| Atlas 推理系列产品                          |    √     |
| Atlas 训练系列产品                          |    √     |

## 功能说明

- 算子功能：将输入张量x中的每个元素与阈值threshold进行比较，将比较结果（大于为1，否则为0）打包为uint8类型的位域输出。每8个连续输入元素的比较结果打包为一个uint8值。

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
    <td>待比较的输入张量，维度至少为1维。</td>
    <td>FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>threshold</td>
    <td>输入</td>
    <td>标量阈值，用于与x中的元素进行比较。</td>
    <td>FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>打包后的位域输出张量，最内维大小为输入最内维的1/8。</td>
    <td>UINT8</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- threshold必须为标量（scalar）
- x必须至少为一维张量（vector或更高维度），不支持标量输入
- x的最内维（最后一个维度）大小必须能被8整除
- x和threshold的数据类型必须一致

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_compare_and_bit_pack](examples/test_geir_compare_and_bit_pack.cpp) | 通过[算子IR](op_graph/compare_and_bit_pack_proto.h)接口方式调用CompareAndBitpack算子。 |
