# ViewCopy

## 产品支持情况

| 产品                                                      | 是否支持 |
| :-------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：将 `src` 中由 `src_size`、`src_stride`、`src_storage_offset` 描述的逻辑视图数据拷贝到 `dst` 中由 `dst_size`、`dst_stride`、`dst_storage_offset` 描述的逻辑视图位置，输出更新后的 `dst`。
- 支持非连续 view copy、transpose view copy、连续 copy 以及存在目标地址重叠的 view copy 场景。

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
      <td>dst</td>
      <td>输入</td>
      <td>原始目标存储tensor，输出结果会继承该tensor的原始内容，并在目标视图位置写入src视图数据。</td>
      <td>BOOL、INT8、UINT8、BFLOAT16、FLOAT16、INT16、UINT16、FLOAT32、INT32、UINT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_size</td>
      <td>输入</td>
      <td>目标逻辑视图shape，元素个数表示视图rank。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_stride</td>
      <td>输入</td>
      <td>目标逻辑视图stride，元素个数需要与dst_size一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_storage_offset</td>
      <td>输入</td>
      <td>目标逻辑视图相对dst存储起始位置的offset，元素个数为1。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>输入</td>
      <td>源存储tensor。</td>
      <td>BOOL、INT8、UINT8、BFLOAT16、FLOAT16、INT16、UINT16、FLOAT32、INT32、UINT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_size</td>
      <td>输入</td>
      <td>源逻辑视图shape，需要与dst_size一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_stride</td>
      <td>输入</td>
      <td>源逻辑视图stride，元素个数需要与src_size一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_storage_offset</td>
      <td>输入</td>
      <td>源逻辑视图相对src存储起始位置的offset，元素个数为1。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst</td>
      <td>输出</td>
      <td>拷贝完成后的目标存储tensor，shape与输入dst一致。</td>
      <td>BOOL、INT8、UINT8、BFLOAT16、FLOAT16、INT16、UINT16、FLOAT32、INT32、UINT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持ND数据格式。
- `dst`、`src`、输出`dst`的数据类型必须一致。
- `dst_size`、`dst_stride`、`dst_storage_offset`、`src_size`、`src_stride`、`src_storage_offset`的数据类型必须一致，且仅支持INT32或INT64。
- `dst_size`和`src_size`需要一致，当前支持rank范围为1到8。
- `dst_stride`和`src_stride`的元素个数必须等于rank，`dst_storage_offset`和`src_storage_offset`的元素个数必须为1。
- size和stride中的元素需要为正数，不支持广播语义。

## 调用说明

| 调用方式  | 样例代码                                      | 说明                                                     |
| --------- | --------------------------------------------- | -------------------------------------------------------- |
| aclnn接口 | [test_aclnn_view_copy](examples/test_aclnn_view_copy.cpp) | 通过[aclnnViewCopy](docs/aclnnViewCopy.md)接口方式调用ViewCopy算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| [Andy Zhang](https://gitcode.com/hehe7758511) | 西北工业大学智能感知交互实验室 | ViewCopy | 2026/7/9 | ViewCopy算子适配开源仓 |
