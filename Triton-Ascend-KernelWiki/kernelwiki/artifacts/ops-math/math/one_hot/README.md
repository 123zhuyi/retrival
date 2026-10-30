# OneHot

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √    |
| <term>Atlas 推理系列产品</term>                             |    √    |
| <term>Atlas 训练系列产品</term>                              |    √    |

## 功能说明

- 算子功能：对长度为n的输入self，经过one_hot的计算后得到一个元素数量为n*k的输出out，其中k的值为numClasses。
- 计算公式： 
  $$
  out[i][j]=\left\{
  \begin{aligned}
  onValue,\quad self[i] = j \\
  offValue, \quad self[i] \neq j
  \end{aligned}
  \right.
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
      <td>x</td>
      <td>输入</td>
      <td>表示索引张量。</td>
      <td>UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>depth</td>
      <td>输入</td>
      <td>类别数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>on_value</td>
      <td>输入</td>
      <td>索引位置的填充值。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>off_value</td>
      <td>输入</td>
      <td>非索引位置的填充值。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入属性</td>
      <td>编码向量的插入维度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示one-hot张量。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn接口 | [test_aclnn_one_hot](examples/test_aclnn_one_hot.cpp) | 通过[aclnnOneHot](docs/aclnnOneHot.md)接口方式调用OneHot算子。 |
