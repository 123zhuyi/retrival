# Asinh

> 反双曲正弦（Inverse Hyperbolic Sine）算子，对应PyTorch `torch.asinh` / `torch.asinh_`。
> 在线aclnn API文档：[docs/aclnnAsinh&aclnnInplaceAsinh.md](docs/aclnnAsinh&aclnnInplaceAsinh.md)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |      √   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×    |
| <term>Atlas 推理系列产品</term>                       |     √    |
| <term>Atlas 训练系列产品</term>                       |     √    |

> 说明：Asinh仅在Ascend950平台（arch35 / DAV_3510）落地，不向下兼容Atlas A2 / A3 / Ascend910等其他平台。

## 功能说明

- 接口功能：对输入Tensor中的每个元素进行反双曲正弦操作后输出。

- 计算公式：

$$
y_{i}=ln(x_{i} + \sqrt{x_{i}^2 + 1})
$$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 240px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>公式中的输入x，dtype与y一致，shape任意（0~8维），支持空Tensor，支持非连续Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出y，shape与x完全一致；非原地接口专有。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selfRef</td>
    <td>输入/输出</td>
    <td>原地接口专有，既作输入又作输出的张量，计算结果原地写回。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

无

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用（非原地）</td>
    <td><a href="./examples/test_aclnn_asinh.cpp">examples/test_aclnn_asinh.cpp</a></td>
    <td rowspan="2">参考ops-math仓库根build.sh完成算子编译与算子包安装，再用examples验证（NPU真机）。</td>
  </tr>
  <tr>
    <td>图模式(GE IR)调用</td>
    <td><a href="./examples/test_geir_asinh.cpp">examples/test_geir_asinh.cpp</a></td>
  </tr>
</tbody></table>
