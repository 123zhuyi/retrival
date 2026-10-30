# CoalesceSparse

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    ×     |

## 功能说明

- 算子功能：将相同坐标点的value进行累加求和，进而减少Coo_Tensor的内存大小。

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
      <td>unique_len</td>
      <td>输入</td>
      <td>去重后的索引数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>unique_indices</td>
      <td>输入</td>
      <td>去重后的索引数组。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引数组。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>每个坐标对应的元素值。</td>
      <td>INT32、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_indices</td>
      <td>输出</td>
      <td>合并后的索引数组。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_values</td>
      <td>输出</td>
      <td>合并后的元素值。</td>
      <td>INT32、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

只能从PyTorch侧调用，如想使用aclnn接口需要手动构造unique_indices入参，并确保调用Kernel前new_values Tensor已置为全零。

重索引后的indices值不能超过int32上限。

确定性计算：

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas 推理系列产品</term>、 <term>Atlas 训练系列产品</term>：aclnnCoalesceSparse默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
- <term>Ascend 950PR/Ascend 950DT</term>：aclnnCoalesceSparse默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn调用 | [test_aclnn_coalesce_sparse](./examples/test_aclnn_coalesce_sparse.cpp)   | 通过aclnnCoalesceSparse接口方式调用CoalesceSparse算子。 |
