# SliceWrite

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

- 算子功能：将value tensor写入x tensor的指定位置（由begin指定偏移）。这是一个原地操作（in-place operation），输出与输入x共用同一块内存。

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
    <td>目标tensor，将被原地修改。</td>
    <td>FLOAT16、FLOAT、DOUBLE、INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>begin</td>
    <td>输入</td>
    <td>写入起始位置偏移量。1D tensor，最多2个元素（row_offset, col_offset）。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>value</td>
    <td>输入</td>
    <td>要写入的tensor。维度必须与x相同。</td>
    <td>FLOAT16、FLOAT、DOUBLE、INT32、INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x</td>
    <td>输出</td>
    <td>输出结果，与输入x同一块内存（原地操作）。</td>
    <td>FLOAT16、FLOAT、DOUBLE、INT32、INT64</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- x维度必须 <= 2
- value维度必须与x维度相同
- begin必须是1D tensor，元素个数 <= 2
- value shape + begin offset必须在x shape范围内
- 输入x和输出x必须是同一块内存

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_slice_write](examples/test_geir_slice_write.cpp) | 通过[算子IR](op_graph/slice_write_proto.h)接口方式调用SliceWrite算子。 |
