# ReduceMean

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对一个多维向量按照指定的维度求平均值。
- 示例：
  - 示例1：
    定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对shape为(2, 3)的二维矩阵进行运算，对第一维计算数据求平均值，输出结果为[2.5, 3.5, 4.5]；对第二维计算数据求平均值，输出结果为[2, 5]。

    图1 ReduceMean按第一个维度计算示例
    ![alt text](./docs/fig/reduce_mean-by-dim-1.png)

    图2 ReduceMean最后一个维度计算示例
    ![alt text](./docs/fig/reduce_mean-by-dim-2.png)

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1034px">
<colgroup>
<col style="width: 82px">
<col style="width: 123px">
<col style="width: 352px">
<col style="width: 352px">
<col style="width: 125px">
</colgroup>
<thead>
  <tr>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">参数名</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">输入/输出/属性</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">描述</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据类型</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据格式</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入张量</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">complex128, complex64, double, float32, float16, int64, int32, int16, int8 <br> 
    uint64, uint32, uint16, uint8, bfloat16</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
 <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">axes</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">要缩减的维度。支持int、list、tuple或NoneType。数据类型必须为int32或int64。<br>如果为None，则缩减所有维度。取值范围为 [-rank(x), rank(x)]。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">int32, int64</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
 </tr>
 <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">keep_dims</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">属性</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">可选布尔值，默认为false。若为true，则保留长度为1的缩减维度；<br>若为false，则对每个axis中的条目，张量的秩减1。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bool</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">noop_with_empty_axes</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">属性</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">可选布尔值，默认为true。若为true，当axes = [] 时不进行缩减；<br> 若为false，当axes = [] 时缩减所有维度。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bool</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">y</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输出</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输出张量，与输入张量x的类型和格式相同</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">与输入张量x相同</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr> 
</tbody>
</table>

## 约束说明

- axes维度须大于0。
- 输入向量维度：最小1维，最大8维。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn调用 | [test_aclnn_mean_v2](./examples/test_aclnn_mean_v2.cpp) | 通过[aclnnMeanV2](./docs/aclnnMeanV2.md)接口方式调用ReduceMean算子。    |
| aclnn调用 | [test_aclnn_global_average_pool](./examples/test_aclnn_global_average_pool.cpp) | 通过[aclnnGlobalAveragePool](./docs/aclnnGlobalAveragePool.md)接口方式调用ReduceMean算子。    |
| 图模式调用 | [test_geir_reduce_mean](./examples/test_geir_reduce_mean.cpp)   | 通过[算子IR](./op_graph/reduce_mean_proto.h)构图方式调用ReduceMean算子。 |
