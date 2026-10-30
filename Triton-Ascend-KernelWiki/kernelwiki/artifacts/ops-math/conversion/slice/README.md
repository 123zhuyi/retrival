# Slice

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √      |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √      |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：从张量中提取一个切片。此操作是从张量“x”中提取大小为“size”的切片，其中起始位置由“offsets”指定。

$$
y = x[offsets:offsets + size]
$$

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
    <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>offsets</td>
    <td>输入</td>
    <td>slice起始位置</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>size</td>
    <td>输入</td>
    <td>slice的大小</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出张量</td>
    <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- 对于输入x的每一个维度n，需要满足：0 <= offsets[i] <= offsets[i] + size[i] <= x_dim[i] for i in [0,n]。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_slice](examples/test_aclnn_slice.cpp) | 通过[aclnnSlice](docs/aclnnSlice.md)方式调用Slice算子    |
