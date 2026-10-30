# NanToNum

## 产品支持情况

| 产品                                                               | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √    |

## 功能说明

- 算子功能：对x中的nan、inf、-inf替换成指定的数据。

## 算子原型

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
      <td>待进行NanToNum计算的入参x。</td>
      <td>float16,bf16,float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nan</td>
      <td>属性</td>
      <td>待进行NanToNum计算的属性nan</td>
      <td>float</td>
      <td>1</td>
    </tr>
    <tr>
      <td>posinf</td>
      <td>属性</td>
      <td>待进行NanToNum计算的属性posinf</td>
      <td>float</td>
      <td>1</td>
    </tr>
    <tr>
      <td>neginf</td>
      <td>属性</td>
      <td>待进行NanToNum计算的属性neginf</td>
      <td>float</td>
      <td>1</td>
    </tr>  
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行NanToNum计算的出参y。</td>
      <td>float16,bf16,float</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 调用样例                                                      | 说明                                                                                        |
| --------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_nan_to_num](./examples/test_aclnn_nan_to_num.cpp) | 通过[aclnnNanToNum](./docs/aclnnNanToNum&aclnnInplaceNanToNum.md)接口方式调用NanToNum算子。 |

## 贡献说明

| 贡献者      | 贡献方     | 贡献算子 | 贡献时间  | 贡献内容               |
| ----------- | ---------- | -------- | --------- | ---------------------- |
| ilovescrapy | 个人开发者 | NanToNum | 2026/3/16 | NanToNum算子适配开源仓 |
