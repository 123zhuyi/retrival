# SimThreadExponential

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：生成服从参数为lambda的指数分布随机数，并将其填充到self张量中。
- 计算公式：

  $$
  f(x) = -\frac{1}{\lambda} \ln(1 - u), u \sim \text{Uniform}(0, 1]
  $$

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
      <td>self</td>
      <td>输入/输出</td>
      <td>输入输出tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>属性</td>
      <td>生成的随机数元素个数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambda</td>
      <td>属性</td>
      <td>公式中的lambda。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>属性</td>
      <td>随机数种子。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>属性</td>
      <td>随机数偏置值。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- lambd必须大于0
- offset必须为4的倍数（Ascend 950）
- count必须等于self张量的元素总数（Ascend 950）
- 仅支持ND格式

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_sim_thread_exponential](tests/ut/op_kernel/test_sim_thread_exponential.cpp) | 通过[aclnnSimThreadExponential](docs/aclnnSimThreadExponential.md)接口方式调用SimThreadExponential算子。 |
