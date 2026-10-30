# SplitV

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √      |

## 功能说明

- 算子功能：根据size_splits将张量沿维度split_dim拆分为num_split更小的张量。

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
      <td>需要切分的tensor列表。</td>
      <td>BOOL、INT8、UINT8、BFLOAT16、FLOAT16、INT16、UINT16、FLOAT32、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size_splits</td>
      <td>输入</td>
      <td>指定一个列表，其中包含沿分割维度的每个输出张量的大小。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>split_dim</td>
      <td>输入</td>
      <td>指定沿其分割的维度。</td>
      <td>INT64</td>
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
      <td>BOOL、INT8、UINT8、BFLOAT16、FLOAT16、INT16、UINT16、FLOAT32、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- size_splits中的每个元素都大于或等于1。
- size_splits的长度等于num_split的值。
- size_splits中的元素总和为维度split_dim的大小。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_split_tensor](examples/test_aclnn_split_tensor.cpp) | 通过[aclnnSplitTensor](docs/aclnnSplitTensor.md)接口方式调用SplitV算子。 |
| aclnn接口 | [test_aclnn_split_with_size](examples/test_aclnn_split_with_size.cpp) | 通过[aclnnSplitWithSize](docs/aclnnSplitWithSize.md)接口方式调用SplitV算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| [GMOW](https://gitcode.com/gcw_8p1hhlB0) | 西北工业大学智能感知交互实验室 | SplitV | 2026/6/23 | SplitV算子适配开源仓 |
