# IsClose

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |     ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：返回一个带有布尔元素的新张量，判断给定的self和other是否彼此接近，如果值接近，则返回True，否则返回False。
- 计算公式：
  返回一个新张量out，其数据类型为BOOL，表示输入self的每个元素是否“close”输入other的对应元素。
  closeness定义为：
  
  $$
  \left | self_{i}-other_{i}\right | \le  atol + rtol\times \left | other_{i} \right |
  $$
    
  当self和other都是有限值时，以上公式成立。当self和other都是非有限时，且仅当它们是相等时，结果为True。当equal_nan为True，NaNs被认为是close的；当equal_nan为False，NaNs被认为是不close。

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
      <td>x1</td>
      <td>输入</td>
      <td>待进行is_close计算的入参，公式中的self_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行is_close计算的入参，公式中的other_i。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rtol</td>
      <td>输入</td>
      <td>待进行is_close计算的入参，公式中的rtol。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>atol</td>
      <td>输入</td>
      <td>待进行is_close计算的入参，公式中的atol。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>equal_nan</td>
      <td>输入</td>
      <td>待进行is_close计算的入参。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行is_close计算的出参。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_isclose](./examples/test_aclnn_isclose.cpp) | 通过[aclnnIsClose](./docs/aclnnIsClose.md)接口方式调用IsClose算子。 |
