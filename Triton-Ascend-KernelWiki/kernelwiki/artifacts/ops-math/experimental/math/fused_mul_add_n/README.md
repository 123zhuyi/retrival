# FusedMulAddN

## 产品支持情况

> 本实验态（`experimental/math/`）实现仅适配 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>（ascend910b，DAV_2201）。内建算子 `math/fused_mul_add_n` 另行覆盖 Ascend 950 / Atlas A3 等产品，不在本实验态范围内。

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：逐元素融合的标量乘加（融合 `mul` 与 `addn(n=2)`，要求 addn 的 n 为 2，mul 的其中一个乘数必须是 scalar 或仅含一个数的 tensor）。`x3` 为单元素标量张量（ShapeSize = 1），仅其首元素 `x3[0]` 参与计算，按标量广播到 `x1` 的全部元素。

- 计算公式：

  $$
  y_i = x1_i \times x3[0] + x2_i
  $$

  其中 `x1`、`x2`、`y` 形状相同（逐元素，非矩阵乘）；`x3` 为单元素标量张量，仅取 `x3[0]` 作为标量乘数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 320px">
  <col style="width: 230px">
  <col style="width: 80px">
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
      <td>主张量，被标量 x3[0] 乘。对应公式中 x1。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>与 x1 同 dtype、同 shape，逐元素加到 x1 × x3[0] 上。对应公式中 x2。</td>
      <td>与 x1 保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>与 x1 同 dtype 的单元素标量张量（ShapeSize = 1），仅取 x3[0] 作为标量乘数。对应公式中 x3[0]。</td>
      <td>与 x1 保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算结果，与 x1 同 dtype、同 shape。对应公式中 y。</td>
      <td>与 x1 保持一致</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持 FLOAT、FLOAT16、BFLOAT16、INT32、INT16（共 5 类）。

## 约束说明

- `x1`、`x2`、`x3`、`y` 的数据类型必须完全一致（host tiling 强校验，不一致直接报错）。
- `x1`、`x2`、`y` 的形状必须一致（逐元素，非矩阵乘）。
- `x3` 必须为单元素标量张量（ShapeSize = 1），形态 `[1]` 与 `[1,1]` 等价（仅取 `x3[0]`）；否则 tiling 报错。
- 本算子无 attribute（无属性参数）。
- FLOAT16 / BFLOAT16 通过 cast 到 FLOAT 计算再 cast 回原 dtype，规避半精度中间累加精度损失；INT32 / INT16 直算，结果按整数语义（溢出按目标整型回绕，不做饱和）。
- 确定性说明：逐元素 FMA，不含 Reduce / 矩阵累加，默认确定性实现（相同输入恒产生相同输出）。

## 调用说明

本算子既可通过 aclnn 两段式接口（单算子）调用，也可通过图模式（GE IR）构图调用。

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用</td>
    <td><a href="./examples/test_aclnn_fused_mul_add_n.cpp">test_aclnn_fused_mul_add_n</a></td>
    <td>通过 aclnn 两段式接口 <code>aclnnFusedMulAddNGetWorkspaceSize</code> + <code>aclnnFusedMulAddN</code>（声明见 <a href="./op_host/op_api/aclnn_fused_mul_add_n.h">aclnn_fused_mul_add_n.h</a>，由自定义算子包 <code>custom_math</code> 导出 <code>libcust_opapi.so</code>）在真实 NPU 上单算子调用。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_fused_mul_add_n.cpp">test_geir_fused_mul_add_n</a></td>
    <td>通过<a href="./op_graph/fused_mul_add_n_proto.h">算子 IR</a>（<code>op::FusedMulAddN</code>）构图方式调用 FusedMulAddN 算子。</td>
  </tr>
</tbody></table>

### 编译运行

示例提供一键脚本 [examples/run.sh](./examples/run.sh) 与独立 [examples/CMakeLists.txt](./examples/CMakeLists.txt)。

aclnn 示例依赖自定义算子包 `custom_math`（含 `aclnnFusedMulAddN` + ascend910b kernel），先构建并解包：

```bash
# 在 worktree 根（source <CANN>/set_env.sh 后）
bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_mul_add_n
bash build_out/cann-ops-math-custom_linux-*.run --noexec --quiet \
     --extract=$(pwd)/build_out/installed_custom_math
```

随后一键编译运行（自动定位 custom_math 包、设置 `ASCEND_CUSTOM_OPP_PATH`/`LD_LIBRARY_PATH`）：

```bash
cd experimental/math/fused_mul_add_n/examples
bash run.sh            # 编译并运行 aclnn 示例 + 编译 geir 示例
bash run.sh aclnn      # 仅 aclnn 示例（编译 + 运行）
bash run.sh geir       # 仅 geir 示例
bash run.sh --noexec   # 仅编译，不运行
```

也可经仓库统一入口编译运行（`--vendor_name` 对应 `custom_math` 的前缀 `custom`）：

```bash
bash build.sh --run_example fused_mul_add_n eager cust --experimental --vendor_name=custom --soc=ascend910b  # aclnn
bash build.sh --run_example fused_mul_add_n graph --experimental --soc=ascend910b                            # 图模式
```

> 示例（ascend910b / 910B3）：`x1={1..8}`、`x2` 全 1、`x3[0]=2` 时 `y = x1×2+1 = {3,5,7,9,11,13,15,17}`，与公式一致。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| zhaohujie | 个人开发者 | FusedMulAddN | 2026/05/28 | FusedMulAddN 算子（A2/ascend910b）适配开源仓实验态 |
| zhaohujie | 个人开发者 | FusedMulAddN | 2026/06/01 | 补充 aclnn 两段式接口与 aclnn/图模式调用示例（examples） |
