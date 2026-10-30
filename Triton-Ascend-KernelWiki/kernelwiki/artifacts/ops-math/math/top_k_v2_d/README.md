# TopKV2D

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

- 算子功能：沿指定维度找出输入张量中最大或最小的k个元素及其索引。与TopK相比，TopKV2D多了一个assist_seq辅助输入，兼容TensorFlow的TopKV2算子。

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
    <td>输入张量。</td>
    <td>FLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>k</td>
    <td>输入</td>
    <td>要取出的元素个数。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>assist_seq</td>
    <td>输入</td>
    <td>辅助序列张量。</td>
    <td>FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sorted</td>
    <td>属性</td>
    <td>是否对输出结果排序，默认为true。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dim</td>
    <td>属性</td>
    <td>指定沿哪个维度进行操作，默认为-1（最后一维）。</td>
    <td>Int</td>
    <td>-</td>
    </tr>
    <tr>
    <td>largest</td>
    <td>属性</td>
    <td>是否取最大值，默认为true。若为false则取最小值。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>values</td>
    <td>输出</td>
    <td>输出的top-k个元素值。</td>
    <td>FLOAT16、FLOAT、DOUBLE、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>indices</td>
    <td>输出</td>
    <td>输出的top-k个元素在输入张量中的索引。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- dim必须在[-x维度数, x维度数)范围内
- k必须大于等于0且小于等于x在dim维度上的大小

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_top_k_v2_d](examples/test_geir_top_k_v2_d.cpp) | 通过[算子IR](op_graph/top_k_v2_d_proto.h)接口方式调用TopKV2D算子。 |
