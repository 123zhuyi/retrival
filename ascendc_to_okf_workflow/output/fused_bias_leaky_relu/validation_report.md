# P4 回放验证报告 — fused_bias_leaky_relu

> 日期：2026-09-07｜验证方式：双盲对照（triton-op-generator 生成流水线）｜判定：**不升 verified，回 P2 补卡**
> 实验目录（cannbot-skills/plugins-official/triton-op-generator/triton_ascend_output/）：
> - 对照组（无知识）：`op_13_FusedBiasLeakyRelu_20260907_1929_4312/` → 最终 **4.5231x**
> - 实验组 A（KB 检索卡片）：`op_0_13_FusedBiasLeakyRelu_20260907_1956_1258/` → 最终 **4.3725x**
> - 实验组 B（用户直给同一卡片、禁 KB）：`op_13_FusedBiasLeakyRelu_20260907_2004_9431/` → 最终 **5.2816x**

## 1. 指标对照表

| 指标 | 对照组（无知识） | 实验组 A（KB 检索） | 实验组 B（直给卡） | 达标线 | 判定 |
|---|---|---|---|---|---|
| 首次生成可运行率 | 100%（iter_0 verify 10/10） | 100%（10/10） | 100%（10/10） | 实验组 ≥ 对照组 | 平（撞天花板） |
| 数值正确迭代轮数 | 1 | 1 | 1 | ≤3 | 平（撞天花板） |
| 精度 | 全 case pass | 全 case pass | 全 case pass | 卡片 §9 标准 | 平 |
| 最终性能 speedup | 4.5231x（5 轮优化） | 4.3725x（3 轮优化） | 5.2816x（4 轮优化） | 不低于对照组 | A 未达标，B 达标 |

## 2. 关键事实（从产物回溯）

1. **三组 iter_0 初版计算链完全一致**：`fp32 提升 → Add → where(t>=0,t,t*ns) → *scale → RTNE 回写`——对照组在无知识条件下也直接生成了卡片 §8 骨架的等价物。本算子的正确性知识完全在基座模型先验内，**卡片的信息增益 ≈ 0**。
2. 实验组 iter_0 骨架逐字来自卡片（autotune {2048,4096,8192}×num_stages{2,3,4}、runtime 标量、grid=cdiv(numel,BLOCK)），证明**卡片被检索到且被忠实执行**（实验组 A 代码头注释明确引用 `ops/fused-bias-leaky-relu.md`）——检索链路与指令遵循均无问题。
3. **性能主增量来自生成器 Phase-4 优化循环，与卡片无关**：三组最终版都收敛到「persistent-core：grid=min(cdiv(numel,BLOCK), vector_core_num) + 核内连续段循环」（生成器自带 references/vector_core_partition.md 的优化点 3，+35%）；实验组 B 的额外收益来自优化点 1（属性 constexpr 化）+ 优化点 13（autotune 扩到 {1024..16384}）。
4. **卡片 §4 的 grid 建议是次优的**：卡片写「grid 直接取 NUM_BLOCKS，一块一 program」——这正是三组初版被优化循环**修掉**的写法（grid 远超物理核数导致 block 排队）。讽刺的是 Ascend C 源码本身就是「每核一段连续 blockFactor + 核内 ubFactor 循环」（tiling L169 + kernel h:117-126），persistent-core 才是对源码的忠实翻译，卡片在 P2 把它过度抽象掉了。
5. **卡片 §10 第 5 条（属性勿烘焙常量）与实测最优相反**：实验组 B 把 negative_slope/scale 改 `tl.constexpr` 后获得常量折叠收益（scale=1.0 消 Muls）；实验组 A 的优化器明确拒绝该方向（"跨 case 变化会重编译——不命中"）。卡片该条只考虑了生产语义忠实性，未给 benchmark 场景的 perf 变体。
6. 测量噪音不可忽略：实验组 B 的 perf 测试 repeats=20，对照组/A 为 repeats=5；framework 侧 aclnn 延迟三组也有 ±4% 抖动。A 与对照组之差（-3.3%）在噪音带内，B 的优势（+16.8%）超出噪音带但部分来自更多优化轮次（随机性）。

## 3. 归因结论

**主要是算子问题，次要但有实质的是卡片内容问题**：

- **算子侧（主因）**：本算子是纯 elementwise 带宽受限、无广播、无规约、无 ❌ 不可达项的「最简实例」——对照组首轮即 10/10 正确，卡片的正确性价值无处发挥；性能天花板由 HBM 带宽决定（case7 实测 ~1.45TB/s 已贴边），剩余空间全在 grid/BLOCK 调优，属于生成器优化循环的固有职责。卡片设计时的预期价值点（mhc 类的 ❌ 降级预警、benchmark 语义超集、复杂 tiling）在本算子全部不存在。
- **卡片内容侧（次因，已修）**：① §4 grid 决策偏离源码结构（一块一 program vs 源码的 persistent-core 分段循环）；② §10 缺「属性 constexpr 化」perf 变体说明；这两条使卡片不仅无增益，还给初版指了次优起点。

## 4. 处置

- 卡片 `confidence` 维持 **inferred**（性能指标未达到「实验组显著优于对照组」）。
- 回 P2 补卡（2026-09-07 已完成，双副本同步）：
  - §4 grid 决策改为 persistent-core 范式（grid=min(cdiv(numel,BLOCK), 物理核数) + 核内连续段循环），并注明「此即源码 blockFactor/ubFactor 两级切分的忠实映射」；
  - §10 属性条目改为「runtime 标量为默认（生产语义）；benchmark 场景可 constexpr 化换常量折叠，注意属性变化触发重编译」。
- 工作流经验：简单 elementwise 算子的卡片价值趋近于零，P0 预检应加「算子复杂度预判」——无 ❌ 项、无广播/规约、对照组先验可覆盖的算子可直接标「低价值，跳过 P4」。
