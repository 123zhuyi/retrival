# StridedSlice

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：按照指定的起始、结束位置和步长，从输入张量中提取一个子张量。

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
    <td>输入的张量</td> 
    <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>begin</td>
    <td>输入</td>
    <td>每个维度的起始值</td>
    <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>end</td>
    <td>输入</td>
    <td>每个维度的结束值</td>
    <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>strides</td>
    <td>输入</td>
    <td>每个维度上每个点取值的跨度</td>
    <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>begin_mask</td>
    <td>属性</td>
    <td>这个值指定哪些维度的begin被忽略</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>end_mask</td>
    <td>属性</td>
    <td>这个值指定哪些维度的end被忽略</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>ellipsis_mask</td>
    <td>属性</td>
    <td>从指定的维度开始全选，直到遇到用户指定begin才退出</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>new_axis_mask</td>
    <td>属性</td>
    <td>在指定位上增加维度为1的shape</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>shrink_axis_mask</td>
    <td>属性</td>
    <td>把对应索引处维度强制降为1</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出张量</td>
    <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无  

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_strided_slice](./examples/arch35//test_aclnn_strided_slice.cpp) | 通过[aclnnStridedSlice](./docs/aclnnStridedSlice.md)接口方式调用StridedSlice算子。 |
| 图模式调用 | [test_strided_slice](./examples/test_geir_strided_slice.cpp) | 通过[算子IR](./op_graph/strided_slice_proto.h)构图方式调用StridedSlice算子    |
