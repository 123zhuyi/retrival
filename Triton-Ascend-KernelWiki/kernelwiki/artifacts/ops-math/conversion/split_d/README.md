# SplitD

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

- 算子功能：将张量沿指定维度split_dim平均拆分为num_split份更小的张量。与Split算子不同，split_dim作为属性而非输入提供。

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
    <td>需要切分的tensor。</td>
    <td>FLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>split_dim</td>
    <td>属性</td>
    <td>指定沿其分割的维度。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>num_split</td>
    <td>属性</td>
    <td>指定要分割的tensor个数。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出结果，动态输出，输出个数由num_split决定。</td>
    <td>FLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- split_dim必须在[0, value维度数)范围内，若为负数则从末尾开始计数
- value在split_dim维度上的大小必须能被num_split整除
- num_split必须大于等于1

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_split_d](examples/test_geir_split_d.cpp) | 通过[算子IR](op_graph/split_d_proto.h)接口方式调用SplitD算子。 |
