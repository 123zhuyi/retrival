# CdistGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：完成Cdist的反向。。
- 计算公式：

  $$
  \begin{aligned}
  out&=grad \cdot y' \\
  &= grad \cdot \left( \sqrt[p]{\sum (x_1 - x_2)^p} \right)' \\
  &= grad \cdot \frac{1}{p} \times \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p}-1} \times p \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{-(p-1)}{p}} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \left( \sum (x_1 - x_2)^p \right)^{\frac{1}{p} \times (-(p-1))} \times (x_1 - x_2)^{p-1} \\
  &= grad \cdot \frac{diff^{p-1}}{cdist^{p-1}} \\
  &= grad \cdot \frac{diff \times |diff|^{p-2}}{cdist^{p-1}}
  \end{aligned}
  $$

  - $\mathrm{diff} = x_1 - x_2$ ：变量差值
  - $\mathrm{cdist} = \sqrt[p]{\sum (x_1 - x_2)^p}$ ： $p$ -范数距离

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
      <td>grad</td>
      <td>输入</td>
      <td>公式中的输入grad。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cdist</td>
      <td>输入</td>
      <td>公式中的输入cdist。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
      <td>p</td>
      <td>属性</td>
      <td>公式中的输入p。</td>
      <td>float</td>
      <td>ND</td>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明
    
| 调用方式   | 样例代码           | 说明                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_cdist_backward](examples/test_aclnn_cdist_backward.cpp) | 通过[aclnnCdistBackward](docs/aclnnCdistBackward.md)接口方式调用CdistGrad算子。 |
