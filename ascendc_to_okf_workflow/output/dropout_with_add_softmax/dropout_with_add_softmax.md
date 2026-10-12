---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: DropoutWithAddSoftmax（axpy+softmax+dropout 三融合）组件合成 → Triton 语义转化卡
description: CANN 闭源内置融合算子（beta/停维护），语义由三个开源组件合成：AxpyV2（t = x1 + alpha·self，fp32 FMA）→ SoftmaxV2（沿 dim=-1，fp32 内部 + 减行最大）→ GenMask+DoMaskV3（philox mask + y=mask?x/keep_prob:0）。AIV-only。Triton 单 kernel 行式分核（W≤512 全载），一次 load 三次 store。唯一不可达点：mask bit-exact 复现（PTA seed 闭源 + uniform 转换式差异），降级为自生成 mask + 三层验收（softmax 精确 / mask 统计 / dropout 自洽）。
tags: [dropout, softmax, axpy, fused, philox, elementwise, reduce_row, aiv_only, ascend950, 合成型, ops_nn, ops_math]
confidence: inferred
status: draft
resource: zmm_source_retrieval/src/ops-math/{math/axpy_v2, random/stateless_drop_out_gen_mask, random/drop_out_do_mask_v3} + zmm_source_retrieval/src/ops-nn/activation/softmax_v2（组件合成，本体闭源）
created_at: "2026-09-07T00:00:00Z"
updated_at: "2026-09-07T00:00:00Z"
---

sources:
  - url: zmm_source_retrieval/src/ops-math/random/stateless_drop_out_gen_mask/
    role: primary
  - url: zmm_source_retrieval/src/ops-math/random/drop_out_do_mask_v3/
    role: primary
  - url: zmm_source_retrieval/src/ops-nn/activation/softmax_v2/
    role: primary
  - url: zmm_source_retrieval/src/ops-math/math/axpy_v2/
    role: primary

# DropoutWithAddSoftmax（axpy + softmax + dropout 三融合）

**合成型卡片**：算子本体是 CANN 闭源内置算子（官方标 beta/停维护，开源仓仅 ONNX 图解析插件 `ops-nn/common/src/framework/npu_dropout_with_add_softmax_onnx_plugin.cpp`），语义由三个开源组件算子合成 + PTA 文档契约 + 文档示例数值推断裁决。AIV-only（组件全部为向量算子）。Triton 侧单 kernel：一次 load、行内 softmax 归约、三次 store。

## 1. 功能与公式（详设 §1.1）

- **GEIR/aclnn 层**：无开源本体；PTA 原型 `npu_dropout_with_add_softmax(self, x1, alpha, prob, dim)`（文档契约）
- **语义分层**（合成链）：
  - 组件 1 AxpyV2：`t = x1 + alpha·self`（axpy_v2 README；PTA 文档 "y=x1+self*alpha"）
  - 组件 2 SoftmaxV2：`s = softmax(t, dim=-1)`（softmax_v2 README；axes 仅支持单轴默认 -1）
  - 组件 3a/3b GenMask+DoMaskV3：`result_dropout = mask ? s/keep_prob : 0`，keep_prob = 1 − prob
- **三输出**：mask（保留记录）、result_softmax（= s，未 dropout）、result_dropout
- ⚠️ **推断标注**：数据流 wiring（softmax 与 dropout 的先后/作用对象）无源码，裁决依据 = 文档示例数值分析（prob=0.9 示例输出量级仅与「dropout 作用在 softmax 输出 ×10」吻合）+ Bert 场景惯例；两处文档自相矛盾已裁决（缩放系数 1/keep_prob 非 1/prob；dropout 作用对象为 s 非 self）——**首次 device 验证时须用 prob=0 与 prob=0.9 两个边界 case 确认**（manifest 缺省项 1）

## 2. 原型与约束（详设 §2 / §2.1）

| 参数 | 方向 | 类型 | 约束 |
|---|---|---|---|
| self | 输入 | Tensor，fp16/fp32（bf16 组件支持，PTA 未列，待确认） | 4D (N,C,H,W)，ND |
| x1 | 输入 | 同 self | 4D 同 shape |
| alpha | 属性 | float | 正负浮点 |
| prob | 属性 | float | **丢弃概率** [0,1)，≥1 非法 |
| dim | 属性 | int | 仅支持 -1 |
| mask | 输出 | Tensor | 文档称 shape=(N,C,H,W)；⚠️ 组件 genmask 输出 bit-packed（8 元素/字节，pt.h:32），布局矛盾待 device 实测（缺省项 2） |
| result_softmax | 输出 | 同 self dtype | (N,C,H,W)，dim 维和为 1 |
| result_dropout | 输出 | 同 self dtype | (N,C,H,W) |

约束：H,W ∈ {128,256,384,512}；(N·C)%32==0；self/x1 同 shape。

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 锚点 |
|---|---|---|
| 支持 NPU | 组件 3a **仅 ascend950**（README 产品表 A2/A3 全 ×）；组件 1/2/3b 含 950+A2/A3 | 各 README |
| dtype | fp32/fp16 确定；bf16 待确认 | 组件 def |
| 确定性 | result_softmax 确定；mask/result_dropout 随机（philox） | — |
| 交付 | PTA torch_npu 接口（op-plugin 适配层，本体闭源） | — |

## 4. 分核策略（详设 §4.1.1 → Triton grid）

- 组件侧事实：elementwise 均分（domask tiling normalCoreProNum/tailCoreProNum）+ softmax A×R 分核六变体——**融合后全部不需要**（mapping §1）
- **Triton 决策**：任务轴 = 行（NUM_ROWS = N·C·H），`BLOCK_W = W` constexpr 单行全载（W≤512 ⇒ 无行内分切、无跨核规约）；grid = AIV 物理核数 + `for row in range(pid, NUM_ROWS, NUM_CORE)` 固定范式

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

- `BLOCK_W = W ∈ {128, 256, 384, 512}`（4 个 constexpr 实例；384 非 2 幂，Triton arange 需 2 幂 ⇒ 实际 `BLOCK_W = next_pow2(W)` + `offs_w < W` mask，512/256/128 直通）
- tilingkey：组件侧 schMode/变体分派全部丢弃（单 kernel 无分派）
- UB 预算：BLOCK_W=512 fp32 时 self/x1/t/e 约 8KB ≪ 192KB ✅

## 6. 片上内存预算（详设 §5）

- 组件侧双缓冲队列（BUFFER_NUM=2）→ Triton 隐式化；单 kernel 三输出共享一次 load，无 GM workspace

## 7. 流水编排与 Triton 可达性（详设 §6）

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级方案 | 预期影响 |
|---|---|---|---|---|
| axpy fp32 FMA 单舍入 | axpy_v2_dag.h:48-53 | ✅ | `x1 + alpha*self`（fp32；编译器自选 FMA） | 无 |
| softmax 减行最大 + fp32 归约 | softmax_v2_base.h:68-77, 493+ | ✅ | `tl.max` → `tl.exp(t-m)` → `tl.sum` → 除 | 无 |
| domask `mask ? x·(1/keep_prob) : 0`，倒数预算 | domask L97-98, 194 | ✅ | wrapper 预算 `inv_keep`，kernel `tl.where(keep, sm*inv_keep, 0)` | 无 |
| keep_prob==0/1 特判 | domask L256-280 | ✅ | prob=0 → keep_prob=1 → mask 全 1 + inv=1，kernel 自然成立无需特判 | 无 |
| philox4x32-10 + u=x·2⁻³²+2⁻³³ + u<keep_prob | pt.h:29,183-201,209 | ⚠️/❌ | `tl.rand(seed, offs) < keep_prob`（同 philox4x32-10，统计等价）；**bit-exact 不可达**：①PTA seed/offset 闭源 ②uniform 转换式不同 | 验收策略吸收（§9） |
| mask bit-packed 输出 | pt.h:32, 229-231 | ❌→替代 | unpacked uint8 0/1（PTA 文档 shape=(N,C,H,W)；待实测裁决） | 存储 8×，非性能瓶颈 |
| 组件各自 tiling/双缓冲/变体分派 | 各 tiling.cpp | ❌→不需要 | 单 kernel 行式分核覆盖 | 无 |

## 8. Triton 骨架（生成用伪码）

完整骨架见 `output/dropout_with_add_softmax/triton_skeleton.py`。核心：

```python
for row in range(pid, NUM_ROWS, NUM_CORE):
    offs = row * BLOCK_W + tl.arange(0, BLOCK_W)
    t  = tl.load(x1_ptr + offs).to(tl.float32) + alpha * tl.load(self_ptr + offs).to(tl.float32)
    sm = tl.exp(t - tl.max(t)) ; sm = sm / tl.sum(sm)          # fp32 全链
    tl.store(sm_ptr + offs, sm.to(out_dtype))
    keep = tl.rand(seed, offs) < keep_prob                      # 自生成 philox
    tl.store(do_ptr + offs, tl.where(keep, sm * inv_keep, 0.0).to(out_dtype))
    tl.store(mask_ptr + offs, keep.to(tl.uint8))
```

wrapper 要点：断言 4D/连续/W∈{128,256,384,512}/(N·C)%32==0；`keep_prob = 1-prob`、`inv_keep = 1/keep_prob` host 预算；grid = AIV 核数。

## 9. 精度标杆与验收（详设 §测试设计）

- **标杆**：device 上 `torch_npu.npu_dropout_with_add_softmax` 即 golden（无 torch 等价物）；benchmark case 需新建（ascendc-migration 风格 Model + json）
- **三层验收**（随机性吸收设计）：
  1. **result_softmax**：确定性，对 golden 按 evaluator 默认阈值精确比对（fp32 rtol=2⁻¹⁰/atol=2⁻¹⁶）
  2. **mask**：统计验收——keep 率 ≈ keep_prob，容差按 numel 取 ±3σ 量级
  3. **result_dropout**：自洽验收——`result_dropout == where(own_mask, result_softmax·inv_keep, 0)` 逐位复核
- **锚点 case**：`prob=0`（keep_prob=1 → 全保留确定性，三输出全部可精确比对）；`prob=0.9`（验证 scale=10 与作用对象，复核 manifest 数据流推断）
- 精度风险点：axpy FMA 单舍入 vs Triton 非融合乘加（fp32 域内差异 ≪ 阈值）；softmax fp32 全链与 golden 一致；b16 RTNE 回写对齐 domask castTrait（L75-77）

## 10. Agent actions（生成指引）

1. 识别类别：行归约 + 三融合 elementwise，AIV-only，无 tl.dot。
2. 分核：行式（NUM_ROWS=N·C·H），BLOCK_W=next_pow2(W) 全载 + 列 mask；grid=物理核数 grid-stride。
3. **必须保留的语义点**：fp32 全链（axpy FMA → softmax 减最大 → domask 缩放）；`keep_prob = 1−prob` 换算在 wrapper；`inv_keep` host 预算。
4. **不要复刻**：组件 tiling 分派、bit-packed mask、philox counter 布局复现。
5. **不要尝试 bit-exact 对齐 golden mask**——验收走 §9 三层策略。
6. mask 输出按 unpacked uint8 0/1；若 device golden 实测为 bit-packed，wrapper 侧补 pack。
7. 验收先跑 prob=0 / prob=0.9 两个锚点 case 确认数据流推断，再跑全量。

## 11. 复现性声明

本卡为**合成型**（confidence: inferred 中的弱推断级）：三组件语义有完整源码锚点（§7 表），但融合 wiring 无源码——依据 PTA 文档 + 文档示例数值推断 + Bert 惯例，未经 device 验证。残留不确定项：①mask 输出布局（packed vs unpacked）；②bf16 支持性；③PTA 层 keep_prob 换算（1−prob 为推断）。回放验证：未做（P4 待执行）。验证预案：benchmark case 首批 prob=0/0.9 边界 case 即裁决 ①②③。
