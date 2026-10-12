# SwinAttentionFFN

## 产品支持情况

<table style="undefined;table-layout: fixed; width: 700px"><colgroup>
<col style="width: 600px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th style="text-align: center;">产品</th>
    <th style="text-align: center;">是否支持</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Ascend 950PR/Ascend 950DT</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A3 训练系列产品/Atlas A3 推理系列产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A2 训练系列产品/Atlas A2 推理系列产品</td>
    <td style="text-align: center;">√</td>
  </tr>
  <tr>
    <td>Atlas 200I/500 A2 推理产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas 推理系列加速卡产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas 训练系列产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Kirin X90 处理器系列产品</td>
    <td style="text-align: center;">√</td>
  </tr>
  <tr>
    <td>Kirin 9030 处理器系列产品</td>
    <td style="text-align: center;">√</td>
  </tr>
</tbody>
</table>

## 功能说明

- 算子功能：全量推理场景的FlashAttention算子，支持sparse优化、支持actualSeqLengthsKv优化、支持int8量化功能，支持高精度或者高性能模式选择。

- 计算公式：

    $$
    y=x1*x2+bias +x3
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x1</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x3</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>  
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

  当前不支持用户直接调用
