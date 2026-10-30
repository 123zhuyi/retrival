# ChunkCat

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                       |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品      |    √     |
| Atlas 200I/500 A2 推理产品                      |    ×     |
| Atlas 推理系列产品                              |    ×     |
| Atlas 训练系列产品                              |    ×     |

## 功能说明

将tensors中所有tensor先按照维度dim切分为num_chunks块，再按照dim后一维进行级联，最后转换为out的数据类型。

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
      <td>输入tensor列表。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>需要切分块的维度。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_chunks</td>
      <td>属性</td>
      <td>指定要切分块的数量。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

* x列表中元素的数据类型和数据格式不在支持的范围之内。
* x列表中元素的数据类型不一致。
* x列表中tensor的shape不在1~8维。
* dim不为0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_chunk_cat](examples/test_aclnn_chunk_cat.cpp) | 通过[aclnnChunkCat](docs/aclnnChunkCat.md)接口方式调用ChunkCat算子。 |
