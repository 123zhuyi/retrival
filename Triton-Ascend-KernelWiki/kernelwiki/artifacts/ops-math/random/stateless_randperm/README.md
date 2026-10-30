# StatelessRandperm

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                      |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                      |    √     |
| Atlas 推理系列产品                              |    √     |
| Atlas 训练系列产品                              |    √     |

## 功能说明

- 算子功能：返回从0到N-1的整数随机排列。

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
      <td>n</td>
      <td>输入</td>
      <td>生成的结果个数，对应算子功能中的N。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>随机数种子。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>偏移值。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>属性</td>
      <td>存储格式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>指定输出的数据类型。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>随机排列后的元素值。</td>
      <td>INT64、INT32、INT16、UINT8、INT8、FLOAT16、FLOAT、DOUBLE、BF16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_randperm](examples/test_aclnn_randperm.cpp) | 通过[aclnnRandperm](docs/aclnnRandperm.md)接口方式调用StatelessRandperm算子。 |
