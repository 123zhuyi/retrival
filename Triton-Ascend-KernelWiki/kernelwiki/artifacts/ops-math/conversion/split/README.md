# Split

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

- 算子功能：将张量沿指定维度split_dim平均拆分为num_split份更小的张量。

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
    <td>FLOAT16、FLOAT32、INT8、INT16、INT64、INT32、UINT8、UINT16、UINT32、UINT64、BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>split_dim</td>
    <td>输入</td>
    <td>指定沿其分割的维度。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
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
    <td>输出结果。</td>
    <td>FLOAT16、FLOAT32、INT8、INT16、INT64、INT32、UINT8、UINT16、UINT32、UINT64、BOOL</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_split](examples/test_geir_split.cpp) | 通过[算子IR](op_graph/split_proto.h)接口方式调用Split算子。 |
