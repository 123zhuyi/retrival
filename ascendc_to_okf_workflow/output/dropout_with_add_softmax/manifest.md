# P0 材料清单 — dropout_with_add_softmax（合成型算子）

> 生成时间：2026-09-07｜**算子本体闭源**（CANN 内置算子，开源仓仅 ONNX 图解析插件 `ops-nn/common/src/framework/npu_dropout_with_add_softmax_onnx_plugin.cpp`，无 op_host/op_kernel 实现；官方文档标注 beta/停维护）
> **来源类型：三组件合成**（工作流首个非单算子源码的卡片）——语义由三个开源组件算子 + PTA 文档 + 文档示例数值推断合成。
> 验收标杆：无现成 benchmark case，需新建（见 §缺省项 4）。

## 数据流推断（Step 0 两个分歧的裁决）

融合算子语义 = 三组件串联：

```
t = x1 + alpha · self                     # 组件1 AxpyV2: out = self + alpha·other
s = softmax(t, dim=-1)                    # 组件2 SoftmaxV2（仅支持单轴，默认 -1）
mask = philox_genmask(keep_prob)          # 组件3a stateless_drop_out_gen_mask
result_softmax  = s
result_dropout  = s · mask / keep_prob    # 组件3b DropOutDoMaskV3
mask（输出1）= 保留/丢弃记录
```

**裁决依据**：

1. **dropout 作用对象 = softmax 输出 s（不是 self）**：文档示例 `self,x1~U(0,1), alpha=2, prob=0.9` → t∈(0,3)。若 dropout 作用在 self：值域 ≤ 1×scale；作用在 t：≤ 3×scale。示例 result_dropout 非零值全部 ∈ [0.02, 0.23]，典型值 0.06~0.15 ≈ softmax 典型值 (1/128≈0.008) × 10 ——只有「作用在 s 上 + scale=10」同时满足量级上限（≤0.23 远小于 1.1/30）和典型值（0.08 附近）。⚠️ 此推断需 device 验证（P4/建 case 时对 golden 验证一次 keep_prob=1 边界即可确认 scale，数据流用 prob=0.9 时输出量级确认）。
2. **缩放系数 = 1/keep_prob，文档"乘(1/prob)"表述错误**：组件 DropOutDoMaskV3 README 公式明确 `y = mask ? self/keep_prob : 0`（README L 功能说明），kernel `prob_reciprocal = 1.0/keepProb`（drop_out_do_mask_v3.h:194）；keep_prob=0 → 全零（L256），keep_prob=1 → 直通（L267）。PTA 侧 prob 是**丢弃**概率 → keep_prob = 1 − prob（示例 prob=0.9 → scale=10，与上条数值分析吻合）。

## 材料归集表

| 组件 | 路径 | 存在 | 关键锚点 | 承担语义 |
|---|---|---|---|---|
| 1. AxpyV2 | `ops-math/math/axpy_v2/` | ✅ | README 公式 `out = self + alpha·other`；kernel 结构同 add 族（DAG 壳，底座复用 op_common broadcast） | `t = x1 + alpha·self` |
| 2. SoftmaxV2 | `ops-nn/activation/softmax_v2/` | ✅ | def：`axes` 仅支持 1 维默认 -1；kernel 6 变体（ar/ara × full_load/recompute/small_r），全 fp32 内部计算（softmax_v2_base.h:32-44 cast trait，LastReduceSum 链 L437-572） | `s = softmax(t, dim=-1)` |
| 3a. GenMask | `ops-math/random/stateless_drop_out_gen_mask/` | ✅ | **仅注册 ascend950**（README 产品表 A2/A3 全 ×）；philox+compare 链见下 | `mask` 生成 |
| 3b. DoMaskV3 | `ops-math/random/drop_out_do_mask_v3/` | ✅ | 950 config 存在；kernel 语义见下 | mask 应用 + 缩放 |
| PTA 文档 | 用户提供（hiascend apilist_000145 同族文档） | ✅ | 原型 `npu_dropout_with_add_softmax(self, x1, alpha, prob, dim)`；约束：4D (N,C,H,W)，H,W∈{128,256,384,512}，(N·C)%32==0，dim=-1 | 接口契约 |
| benchmark | 无 | ❌ 需新建 | — | 验收标杆 |

## 组件 3a（GenMask）精确语义（stateless_drop_out_gen_mask_pt.h）

- **随机源**：`AscendC::PhiloxRandom<10>`（10 轮 philox4x32，pt.h:209）
  - key = (seed 低 32, seed 高 32)（pt.h:99-102，int64 seed 截断拆分）
  - counter = 4×uint32 ← 2×uint64 offset（pt.h:104-119）
  - 每 counter 值产 4 个 uint32（RESULT_ELEMENT_CNT=4，pt.h:28）；tile 间 `Skip(ceil(n/4))` 推进 counter（pt.h:218-235）→ **元素 i 的随机数 = philox(key, counter₀ + ⌊i/4⌋)[i%4]**
- **归一化**：`u = x·2⁻³² + 2⁻³³`（curand 风格开区间 (0,1)，CURAND_2POW32_INV=2.3283064365386963e-10，pt.h:29, 153-154, 183-184）
- **保留判定**：`mask = (u < prob)`，`prob` 为**保活系数 keep_prob**（CompareScalar CMPMODE::LT，pt.h:201）
- **输出布局**：bit-packed uint8，8 元素/字节（byteBitRatio=8，pt.h:32；字节偏移 = 元素偏移/8，pt.h:129, 229-231）

## 组件 3b（DoMaskV3）精确语义（drop_out_do_mask_v3.h）

- `keep_prob==0` → 全零；`keep_prob==1` → 直通拷贝（IsProbEqual eps 判定，L80-84, 256-280）
- 一般路径：`y = mask ? x·(1/keep_prob) : 0`（Muls + Select(select0=0)，L97-98）；`1/keep_prob` 先算倒数再乘（L194）
- **精度**：b16 输入升 fp32 乘、RTNE 回写（doMaskB16 L111-118，castTraitB32ToB16=CAST_RINT L75-77）；fp32 直算
- mask 读取：bit-packed uint8 → MaskDeInterleave 解包（L213-217）

## 组件 2（SoftmaxV2）精确语义

- 输入 fp16/bf16/fp32，**内部全 fp32**（CastToFp32From，softmax_v2_base.h:68-77），回写按 dtype
- 6 个 arch35 变体按 A（行数）/R（轴长）形态选择：full_load（R 全载）/ recompute（重算省 UB）/ small_r；对本算子 W≤512 ⇒ R 极小，full_load 路径即可覆盖
- 数值稳定：减行最大值后 exp（标准 safe softmax，与 torch 一致）

## 缺省项与注意点

1. **融合本体闭源**：数据流 wiring（§数据流推断）是「组件语义 + 文档 + 示例数值」三方推断，置信度高但未经 device 验证 → 建 benchmark case 时第一批先跑 prob=0/0.9 边界确认。
2. **mask 输出布局矛盾**：组件 3a 输出 bit-packed（8 元素/字节），但 PTA 文档称 fused op 的 mask 输出 shape=(N,C,H,W)（逐元素）——PTA 层可能解包。新建 case 时以 device golden 实测为准（看 mask.shape 和 dtype）。
3. **seed/offset 不可知**：PTA 层 philox key/counter 生成逻辑闭源 → **无法 bit-exact 复现 golden 的 mask 序列**，验收走三层策略（softmax 精确 + mask 统计 + dropout 自洽），详见卡片 §9。
4. **benchmark case 缺失**：需在 `ascendc-kernelgen-data/npu_benchmark/ascendc-migration/` 下新建（Task #15）。
5. **950-only 提示**：组件 3a 仅注册 ascend950；本任务目标平台 950，无影响；A2/A3 交付需注意。
6. **tilingkey/分核**：三组件各自 tiling 简单（纯 elementwise/行归约），融合后 Triton 单 kernel 行式分核即可，组件 tiling 细节不进卡片主体（P2 记录丢弃理由）。

## P0 质量门结论

三组件源码齐全 + 文档契约完整 + 数据流推断有数值旁证 → 通过，进 P1。风险集中在缺省项 1/2/3（融合层闭源），已全部显式列出并有验证预案。
