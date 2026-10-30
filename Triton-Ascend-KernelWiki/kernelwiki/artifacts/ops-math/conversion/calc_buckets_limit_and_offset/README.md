# CalcBucketsLimitAndOffset

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

- 算子功能：根据bucket_list索引，从ivf_counts和ivf_offset中获取对应桶的计数和偏移量，并根据total_limit限制各桶的计数总和，输出限制后的计数和对应偏移量。

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
    <td>bucket_list</td>
    <td>输入</td>
    <td>1-D tensor，桶索引列表，值为ivf_counts和ivf_offset的下标。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>ivf_counts</td>
    <td>输入</td>
    <td>1-D tensor，每个桶的计数值。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>ivf_offset</td>
    <td>输入</td>
    <td>1-D tensor，每个桶的偏移量。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>total_limit</td>
    <td>属性</td>
    <td>所有桶计数总和的上限值。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>buckets_limit</td>
    <td>输出</td>
    <td>1-D tensor，限制后的各桶计数值，总和不超过total_limit。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>buckets_offset</td>
    <td>输出</td>
    <td>1-D tensor，与bucket_list对应的偏移量。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- bucket_list中的值必须在[0, ivf_counts元素个数)和[0, ivf_offset元素个数)范围内
- ivf_counts和ivf_offset的元素个数应大于等于bucket_list中的最大索引值
- total_limit必须为非负整数

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_calc_buckets_limit_and_offset](examples/test_geir_calc_buckets_limit_and_offset.cpp) | 通过[算子IR](op_graph/calc_buckets_limit_and_offset_proto.h)接口方式调用CalcBucketsLimitAndOffset算子。 |
