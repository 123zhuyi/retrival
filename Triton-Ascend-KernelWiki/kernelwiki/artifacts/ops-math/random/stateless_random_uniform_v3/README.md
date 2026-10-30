# StatelessRandomUniformV3

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

- 算子功能：返回一个随机数张量，该随机数是从独立均匀分布中获取。

- 计算公式：

  $$
  UniformResult = StatelessRandomUniformV2 * (to - from) + from
  $$

  $$
  RandomResult = StatelessRandomUniformV2 * to - (StatelessRandomUniformV2 * from - from)
  $$

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
      <td>shape</td>
      <td>输入</td>
      <td>输入张量的形状。</td>
      <td>INT64、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>用于基于计数器的随机数生成算法的秘钥。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>输入</td>
      <td>用于基于计数器的随机数生成算法的初始计数值。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>from</td>
      <td>输入</td>
      <td>用于随即范围的下界（包含）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>to</td>
      <td>输入</td>
      <td>用于随即范围的上界（不包含）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出指定形状的随机值。</td>
      <td>FLOAT、BF16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

无
