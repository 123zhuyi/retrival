# P2 语义转化 — dropout_with_add_softmax（三组件 → Triton 单 kernel）

> 核心结论：三组件融合在 Triton 侧是**单 kernel、一次 load、三次 store**。约束 W∈{128,256,384,512} ⇒ 一行（W 元素）单 tile 全载，行内归约无跨核依赖，是 softmax 类算子最简形态。
> 唯一❌项：philox mask 无法 bit-exact 复现 golden（PTA 层 seed/offset 生成闭源 + CANN/Triton uniform 转换式不同），降级为「自生成 mask + 统计验收 + 自洽验收」。

## 1. 分核 → grid

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| 组件各自元素均分（domask tiling normalCoreProNum/tailCoreProNum）；softmax A×R 分核 | 任务轴 = 行（row = N·C·H 共 NUM_ROWS 行，每行 W 元素）；`BLOCK_W = W` constexpr 全载；grid = AIV 物理核数 + `for row in range(pid, NUM_ROWS, NUM_CORE)` | ✅ |
| softmax_v2 六变体按 A/R 形态选择 | W≤512 单行全载覆盖全部 case，无需变体分派 | ✅（结构性简化） |

## 2. 计算链 → kernel 体

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| axpy：Cast fp32 → FusedMulAdd(x2·alpha+x1) → RINT 回写（axpy_v2_dag.h:48-53） | `t = x1_f32 + alpha * self_f32`（fp32；Triton 编译器自选 FMA 融合，精度等价单舍入） | ✅ |
| softmax：fp32 内部、减行最大、exp、ReduceSum、除（softmax_v2_base.h） | `m = tl.max(t)`；`e = tl.exp(t - m)`；`s = tl.sum(e)`；`sm = e / s` | ✅ |
| domask：`y = mask ? x·(1/keep_prob) : 0`，倒数预算（domask L97-98, 194） | `inv_keep = 1.0 / keep_prob`（wrapper 预算标量传入）；`tl.where(mask, sm * inv_keep, 0.0)` | ✅ |
| 边界：keep_prob==0 全零 / ==1 直通（domask L256-280） | wrapper 分支：prob==0 → mask 全 1、result_dropout = result_softmax（inv_keep=1.0，kernel 内自然成立，无需特判） | ✅ |
| b16 升 fp32 乘 + RINT 回写 | `.to(tl.float32)` 全程，store 前 `.to(out_dtype)`（RTNE 默认） | ✅ |

## 3. mask 生成 → philox 对齐判定

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| philox4x32-10：元素 i ← philox(key, counter₀+⌊i/4⌋)[i%4]；u = x·2⁻³²+2⁻³³；keep = (u < keep_prob)（pt.h:28-32, 183-201, 209-235） | **自生成**：`r = tl.rand(seed, offs)`（Triton 同为 philox4x32-10），`mask = r < keep_prob` | ⚠️ 统计等价 / bit-exact ❌ |
| bit-packed uint8 输出（8 元素/字节，pt.h:32） | unpacked uint8 0/1 存储（PTA 文档称 mask 输出 shape=(N,C,H,W)，与组件 bit-packed 矛盾，按文档走 unpacked；建 case 时以 device golden 实测裁决） | ⚠️ 待实测 |

**bit-exact 不可达的两层原因**：① PTA 层 seed/offset 生成闭源（key/counter 不可知）；② uniform 转换式不同（CANN `x·2⁻³²+2⁻³³` vs Triton fp32 转换取位方案不同）。两层任一成立即不可能 bit-exact → 验收不依赖 mask 逐位一致。

## 4. tilingkey / 内存 / 流水

- tilingkey：组件侧 schMode 等全部**丢弃**（单 kernel 无分派；理由记录于此）
- UB 预算：BLOCK_W=512 fp32 时 self/x1/t/e 四 tile ≈ 8KB ≪ 192KB ✅
- 流水：load→compute→store 编译器自动；三次 store 共享一次 load

## 5. 验收策略转化（随机性处理，卡片 §9）

| 输出 | 验收 |
|---|---|
| result_softmax | 确定性，对 golden 精确比对（evaluator 默认阈值） |
| mask | 统计：keep 率 ≈ keep_prob（容差 ±3σ/√numel 量级） |
| result_dropout | 自洽：`result_dropout == tl.where(own_mask, result_softmax·inv_keep, 0)` 逐位复核 + prob=0 case 全精确比对 |

## P2 质量门

- extraction 事实全部有去向：转化 5 条（axpy/softmax/domask/边界/精度）、简化 2 条（softmax 变体分派、组件 tiling）、❌ 1 条（mask bit-exact，降级方案 + 验收策略齐）✅
- 骨架见 `triton_skeleton.py` ✅ → 进 P3
