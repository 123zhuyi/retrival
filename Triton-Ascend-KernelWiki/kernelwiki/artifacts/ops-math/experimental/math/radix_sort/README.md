# RadixSort

## 贡献说明

| 贡献者      | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------|------------|----------|
| CANN-BOT SIMT | radix_sort | 2026/06/22 | 新增radix_sort算子 |

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：使用 LSD（Least Significant Digit）基数排序算法对一维整数 Tensor 执行稳定排序，同时返回排序后的值和对应的原始索引。

- 计算公式：

算法按字节（8-bit）从低位到高位逐趟扫描，每趟使用 256 个桶进行直方图统计 → 前缀和 → 逆序散射。有符号整数通过 XOR 符号位翻转映射到无符号域排序。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>待排序的一维整数数组。</td>
      <td>INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_values</td>
      <td>输出</td>
      <td>排序后的值，与输入具有相同的类型和形状。</td>
      <td>INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_indices</td>
      <td>输出</td>
      <td>排序后每个元素对应的原始索引。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>属性</td>
      <td>排序方向。false 为升序，true 为降序。默认值为 false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持一维（1D）输入。
- 使用 LSD 基数排序算法（每趟 8-bit，256 bins）。
- 排序结果保证稳定性（stability），重复值的原始索引顺序保持不变。
- 支持的数据类型：int32、uint32、int64、uint64。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td>不支持</td>
    <td rowspan="2">参见<a href="../../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_radix_sort.cpp">test_geir_radix_sort</a></td>
  </tr>
</tbody>
</table>
