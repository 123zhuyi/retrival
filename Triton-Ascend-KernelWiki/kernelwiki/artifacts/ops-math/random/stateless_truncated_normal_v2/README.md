# StatelessTruncatedNormalV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    ×     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    ×     |
| Atlas 200I/500 A2 推理产品                      |    ×     |
| Atlas 推理系列产品                              |    ×     |
| Atlas 训练系列产品                              |    ×     |

## 功能说明

- 算子功能：无状态版截断正态分布随机数生成器。使用Philox4x32算法，以key和counter为输入生成均匀分布的uint32随机数，通过Box-Muller变换将均匀分布转换为标准正态分布，生成服从标准正态分布N(0,1)的随机数，绝对值大于2.0的样本被截断丢弃并重新采样。
- 无状态特性：随机数生成完全由外部传入的key/counter决定，相同输入保证产生相同输出（确定性）。
- 对标：TensorFlow `tf.raw_ops.StatelessTruncatedNormalV2`

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
      <td>shape</td>
      <td>输入</td>
      <td>输出张量的形状</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>Philox随机数生成器密钥，shape=[1]</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>输入</td>
      <td>Philox随机数生成器计数器，shape=[2]</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alg</td>
      <td>输入</td>
      <td>随机数算法ID（1=Philox，当前仅支持Philox）</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>指定输出的数据类型（0=float32, 1=float16, 27=bfloat16 ,11=float64）</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>生成的截断正态分布随机数序列</td>
      <td>FLOAT16、FLOAT、BF16、FLOAT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- alg参数当前仅支持Philox算法（alg=1）
- key的shape必须为[1]
- counter的shape必须为[2]（Philox算法要求）
- 输出shape中元素数为0时直接返回空tensor

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式调用 | [test_geir_stateless_truncated_normal_v2](./examples/test_geir_stateless_truncated_normal_v2.cpp) | 通过[算子IR](./op_graph/stateless_truncated_normal_v2_proto.h)构图方式调用StatelessTruncatedNormalV2算子。 |
