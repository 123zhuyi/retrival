# RadixTopK

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                      |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                              |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：返回输入Tensor在指定维度上的k个极值及索引。RadixTopK算子支持通过输入Tensor指定k值。

- 计算说明：
  - 当largest=True时，返回指定维度上最大的k个值及其索引
  - 当largest=False时，返回指定维度上最小的k个值及其索引

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
      <td>待进行TopK计算的输入张量，支持1-8维度。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>0D标量Tensor，表示计算维度上输出的极值个数。取值范围为[0, x.size(dim)]。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输出</td>
      <td>TopK计算的输出值，数据类型与x保持一致。shape排序轴与k一致，非排序轴与x一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>TopK计算的输出索引，数据类型由indices_dtype属性指定。shape排序轴与k一致，非排序轴与x一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>属性</td>
      <td>可选布尔型，默认为True。True表示输出结果排序，False表示输出结果不排序。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>可选整型，默认为-1。表示计算维度。取值范围为[-x.dim(), x.dim())。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>largest</td>
      <td>属性</td>
      <td>可选布尔型，默认为True。True表示返回最大的k个元素，False表示返回最小的k个元素。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices_dtype</td>
      <td>属性</td>
      <td>可选整型，默认为DT_INT32(3)。表示输出indices的数据类型，支持DT_INT32(3)或DT_INT64(9)。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无
