# ReduceStdWithMean

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |

## 功能说明

- 算子功能：沿给定维度 `dim` 归约计算标准差和均值。两个 L2 API 覆盖不同场景：`aclnnStdMeanCorrection` 对应 `torch.std_mean`，计算标准差 + 均值，支持 Bessel 修正；`aclnnBatchNormStats` 对应 BatchNorm 统计量，计算均值 + 标准差倒数（`1/sqrt(var+eps)`）。

- L2 API 内部管线（以 `aclnnStdMeanCorrection` 为例）：

  ```text
  self ──→ ReduceMean ──→ mean ──→ Broadcast ──→ mean(expanded)
                │                                      │
                └──→ meanOut                           └──→ ReduceStdWithMean(self, mean) ──→ stdOut
  ```

  `ReduceMean` 和 `ReduceStdWithMean` 是 L0 kernel。`ReduceStdWithMean` 接收预计算的 `mean`（已 broadcast 到 `self` 同 shape），计算 `diff = self - mean` 进而得出方差和标准差。**预计算 mean 避免了在 kernel 内重复计算均值**——这是 Two-Pass 算法的设计意图，而非使用限制。

- 计算公式（`ReduceStdWithMean` L0 kernel）：

  $$
  \begin{aligned}
  \text{diff} &= \text{self} - \text{mean} \\
  \text{sum\_sqr} &= \sum (\text{diff})^2 \\
  \text{var} &= \frac{\text{sum\_sqr}}{\max(0,\ N - \text{correction})} \\
  \text{output} &= \begin{cases}
  \sqrt{\text{var}}, & \text{invert} = \text{false} \\[6pt]
  \dfrac{1}{\sqrt{\text{var} + \text{eps}}}, & \text{invert} = \text{true}
  \end{cases}
  \end{aligned}
  $$

  其中 `N` 为归约维度元素总数。

## 参数说明

<table style="table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 120px">
  <col style="width: 320px">
  <col style="width: 260px">
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
      <td>self (input)</td>
      <td>输入</td>
      <td>输入张量。对应公式中 self。支持 1-8 维。aclnnStdMeanCorrection 使用参数名 self，aclnnBatchNormStats 使用参数名 input。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>L0 kernel 的输入（L2 API 内部由 ReduceMean 自动计算并传入，调用方无需关心）。为预计算的均值张量，已 broadcast 到与 self 同 shape。</td>
      <td>与 self 保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>归约维度。支持单维或多维归约。L2 API 通过 Transpose 自动处理非连续归约维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>correction</td>
      <td>属性</td>
      <td>Bessel 修正值。0 表示总体方差（除以 N），>= 1 表示样本方差（除以 N - correction）。当 correction 超过归约维度元素数时，输出结果 clamp 为 0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>属性</td>
      <td>是否保留归约维度。true 时输出与输入同维数（归约维度长度为 1），false 时压缩归约维度。仅 aclnnStdMeanCorrection 使用。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>invert</td>
      <td>属性</td>
      <td>输出控制。false = 输出标准差（std），true = 输出标准差倒数（1/std）。仅 L0 API 使用；aclnnStdMeanCorrection 固定输出 std，aclnnBatchNormStats 固定输出 1/std。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>数值稳定性常数，加在方差上再开根号，避免除零。aclnnStdMeanCorrection 使用 FLOAT，aclnnBatchNormStats 使用 DOUBLE。</td>
      <td>FLOAT / DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stdOut (invstdOut)</td>
      <td>输出</td>
      <td>标准差结果。aclnnStdMeanCorrection 输出标准差（std），参数名为 stdOut；aclnnBatchNormStats 输出标准差倒数（invstd），参数名为 invstdOut。shape 取决于 keepdim 设置（aclnnStdMeanCorrection）或固定不保留维度（aclnnBatchNormStats）。</td>
      <td>与 self 保持一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>均值结果。aclnnStdMeanCorrection 输出归约后的均值，aclnnBatchNormStats 输出沿 batch 维归约的均值。shape 取决于 keepdim / 归约维度设置。</td>
      <td>与 self 保持一致</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持 FLOAT、FLOAT16、BFLOAT16（共 3 类）。输入、输出数据类型必须一致（非 RegBase 平台约束）。

## 约束说明

- `self` (input) 的维度数（rank）必须在 1 到 8 之间。
- 对于 L2 API，`mean` 由内部 `ReduceMean` 自动计算和 broadcast，调用方无需提供；对于 L0 API，`mean` 必须与 `self` 同 shape、同 dtype，由调用方预计算并传入。
- 支持多维归约，归约维度通过 `dim` 参数指定。
- 非连续归约维度：L2 API 通过 Transpose 自动将非连续归约维度移到末尾后再调度 kernel，调用方无需手动 transpose。
- correction 必须为非负整数。当 correction >= 归约维度元素数时，方差为 0，输出 0（std）或 eps 保护下的 1/sqrt(eps)（invstd），不会除零崩溃。
- FLOAT16 / BFLOAT16 中间计算全部升精度到 FLOAT32 执行（Cast → Sub → Mul → ReduceSum），规避半精度中间累加精度损失。
- 仅支持 ND 格式。
- 确定性说明：Pre-computed Mean Two-Pass 统一算法路径，invert 参数仅影响最终输出选择（sqrt 或 1/sqrt），核心计算路径完全相同。默认确定性实现（相同输入恒产生相同输出）。

## 调用说明

本算子通过 aclnn 两段式接口（单算子）调用。

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用（StdMeanCorrection）</td>
    <td><a href="./examples/test_aclnn_std_mean_correction.cpp">test_aclnn_std_mean_correction</a></td>
    <td>通过 aclnn 两段式接口 <code>aclnnStdMeanCorrectionGetWorkspaceSize</code> + <code>aclnnStdMeanCorrection</code>（声明见 <a href="./op_api/aclnn_std_mean_correction_experimental.h">aclnn_std_mean_correction_experimental.h</a>，由自定义算子包 <code>custom_math</code> 导出）计算标准差 + 均值。对应 PyTorch 的 <code>torch.std_mean(x, dim, correction=correction, keepdim=keepdim)</code>。</td>
  </tr>
  <tr>
    <td>aclnn 调用（BatchNormStats）</td>
    <td><a href="./examples/test_aclnn_batch_norm_status.cpp">test_aclnn_batch_norm_status</a></td>
    <td>通过 aclnn 两段式接口 <code>aclnnBatchNormStatsGetWorkspaceSize</code> + <code>aclnnBatchNormStats</code>（声明见 <a href="./op_api/aclnn_batch_norm_stats_experimental.h">aclnn_batch_norm_stats_experimental.h</a>）计算 BatchNorm 统计量（均值 + 标准差倒数）。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td>-</td>
    <td>暂不支持。本算子仅提供 aclnn 接口。</td>
  </tr>
</tbody></table>

### 编译运行

#### 构建自定义算子包

```bash
# 在仓库根目录（source <CANN>/set_env.sh 后）
source /usr/local/Ascend/cann-8.5.1/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=reduce_std_with_mean -j16
bash build_out/cann-ops-math-custom_linux-*.run --install-path=/usr/local/Ascend/cann-8.5.1

# 同步到 opp 目录（必须）
V=/usr/local/Ascend/cann-8.5.1/vendors/custom_math
O=/usr/local/Ascend/cann-8.5.1/opp/vendors/custom_math
cp $V/op_api/lib/libcust_opapi.so $O/op_api/lib/
patchelf --add-needed libopapi_math.so $O/op_api/lib/libcust_opapi.so
cp $V/op_impl/ai_core/tbe/op_tiling/liboptiling.so $O/op_impl/ai_core/tbe/op_tiling/
rm -rf $O/op_impl/ai_core/tbe/kernel/ascend910b/reduce_std_with_mean
cp -a $V/op_impl/ai_core/tbe/kernel/ascend910b/reduce_std_with_mean $O/op_impl/ai_core/tbe/kernel/ascend910b/

# 加载环境
source /usr/local/Ascend/cann-8.5.1/vendors/custom_math/bin/set_env.bash
```

#### UT 测试

```bash
# 在仓库根目录执行
bash build.sh -u --ophost --opapi --opkernel --ops=reduce_std_with_mean --experimental -j16
```

#### ST 测试（ATK）

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
source /usr/local/Ascend/cann-8.5.1/vendors/custom_math/bin/set_env.bash

# StdMeanCorrection ATK
atk aclnn --task accuracy --devices 0 \
  experimental/math/reduce_std_with_mean/tests/st/aclnnStdMeanCorrection/atk_aclnnStdMeanCorrection.json

# BatchNormStats ATK
atk aclnn --task accuracy --devices 0 \
  -p experimental/math/reduce_std_with_mean/tests/st/aclnnBatchNormStats/executor_aclnnBatchNormStats.py \
  experimental/math/reduce_std_with_mean/tests/st/aclnnBatchNormStats/atk_aclnnBatchNormStats.json
```

#### 示例代码编译运行

```bash
# 在仓库根目录执行（需先完成算子包构建和安装）
bash build.sh --run_example reduce_std_with_mean eager cust --experimental --vendor_name=custom --soc=ascend910b
```
