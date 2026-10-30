# Range

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：从start起始到end结束按照step的间隔获取值，并保存到输出的1维张量，其中数据范围为[start,end)。
- 计算公式： 
  $$
  out_{i+1} = out_i+step
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
      <td>start</td>
      <td>输入</td>
      <td>获取值的范围的起始位置。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>limit</td>
      <td>输入</td>
      <td>获取值的范围的结束位置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>输入</td>
      <td>获取值的步长。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>is_closed</td>
      <td>输入属性</td>
      <td>是否包含limit成闭区间。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>递增序列输出。</td>
      <td>FLOAT16、FLOAT、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn接口 | [test_aclnn_range](examples/test_aclnn_range.cpp) | 通过[aclnnRange](docs/aclnnRange.md)接口方式调用Range算子。 |
| aclnn接口 | [test_aclnn_arange](examples/test_aclnn_arange.cpp) | 通过[aclnnArange](docs/aclnnArange.md)接口方式调用Range算子。 |
