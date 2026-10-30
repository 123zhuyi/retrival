# DynamicPartition

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

接口功能：用于根据分区索引将输入数据动态分割成多个张量。示例：
假设输入Tensor为[1, 6, 3, 8, 2, 9]，将一维张量按条件分成两组（如元素值≤5和>5），值如下所示：

```text
>>> x = tf.constant([1, 6, 3, 8, 2, 9])
>>> partitions = tf.cast(data ≤ 5, tf.int32)
>>> tf.dynamic_partition(x, partitions, num_partitions=2)
[array([1, 3, 2]), array([6, 8, 9])]
```

支持对多维张量的分区操作，例如将矩阵按行或列拆分：

```text
>>> x = tf.reshape(tf.range(12), [3, 4])
// 按行分区
>>> tf.dynamic_partition(matrix, [0, 1, 0], num_partitions=2)
array([[0, 1, 2, 3],
       [8, 9, 10, 11]])
```

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1528px"><colgroup>
  <col style="width: 132px">
  <col style="width: 120px">
  <col style="width: 256px">
  <col style="width: 253px">
  <col style="width: 333px">
  <col style="width: 126px">
  <col style="width: 160px">
  <col style="width: 145px">
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
      <td>待分区的原始输入张量</td>
      <td>FLOAT、DOUBLE、FLOAT16、COMPLEX64、UINT8、INT8、INT16、INT32、INT64、UINT16、UINT32、UINT64、BOOL、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>partitions</td>
      <td>输入</td>
      <td>分区索引张量，决定每个元素归属的分区编号</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_partitions</td>
      <td>可选属性</td>
      <td><ul><li>指定总分区的数量</li><li>默认值为1。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>分区后的输出张量</td>
      <td>FLOAT、DOUBLE、FLOAT16、COMPLEX64、UINT8、INT8、INT16、INT32、INT64、UINT16、UINT32、UINT64、BOOL、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_dynamic_partition](./examples/test_geir_dynamic_partition.cpp) | 通过[算子IR](./op_graph/dynamic_partition_proto.h)构图方式调用DynamicPartition算子。 |
