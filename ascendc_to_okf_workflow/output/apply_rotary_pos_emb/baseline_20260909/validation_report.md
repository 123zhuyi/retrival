# ApplyRotaryPosEmb P4 回放验证报告

> 日期：2026-09-08/09 ｜ 结论：卡片 confidence 升 **verified**
> 三组双盲回放 + CANN 源码回头审计，全部 45/45 case 正确性通过。

## 1. 三组结果总览

| 组 | 实验目录 | 知识注入 | geomean 加速比 | 正确性 |
|---|---|---|---|---|
| 直给卡组 | op_0_5_ApplyRotaryPosEmb_20260908_1701_3967 | 工作流原版卡 + triton_skeleton.py（Agent 自动 ls 发现后随 inspirations 传入） | **4.2060** | 45/45 |
| KB 检索组 | op_0_5_ApplyRotaryPosEmb_20260908_1636_8759 | triton-knowledge-query 检索命中 KB 卡（score 28.37，relevance 0.94，全文 Read） | 3.5789 | 45/45 |
| 无知识对照 | op_0_5_ApplyRotaryPosEmb_20260908_1637_9195 | SKIPPED（user_explicit_override） | 3.1749 | 45/45 |

单调信号干净：**4.2060 > 3.5789 > 3.1749**，卡片指导力成立。

## 2. 知识价值定位

知识注入的收益集中在两处，而非均匀分布：

- **起点质量**：Phase3 初版 baseline 直给组 2.0494 / KB 组 1.3458 / 对照组 0.1126——卡片把「从哪出发」提升了 1~2 个数量级（对照组初版近乎不可用，靠优化循环爬回 3.17）。
- **大 case 地板**：case 32/33/34（大 shape）卡组 1.5~2.5x，对照组 0.58~0.78x——知识决定了「最坏情况不塌方」。geomean 差距（0.6~1.0x）反而低估知识价值。

## 3. P4 收敛的 kernel 结构（三组交叉验证后的定局）

- (b,s) 组为外循环的标量 persistent 循环（grid = min(BS, 物理核数)），组内 n 行展开为 `BLOCK_N = next_pow2(N)` 单 tile；
- cos/sin 按 (b,s) 组**外提**，每组只 load 一次（决定性单点杠杆，2.0956→4.2060，+100%；「cache 驻留 ≠ 复用」）；
- interleave 走连续 load + `tl.split`/`tl.interleave` 寄存器重排；**禁用 strided-2 访存**（三类后端 bug，见 §5.1）；
- 无 autotune、无 num_stages/num_warps（运行时不生效）；mask 全部 int32 比较；shapes/strides 全 constexpr；kernel 内禁用 `%`（写 `a-(a//b)*b`）；
- 行解码标量化（`b = bs // S; s = bs - b * S`），规避向量 div/mod × 特定访存的后端缺陷面。

## 4. KB 检索组差距归因（3.5789 vs 4.2060，四层）

① **autotune 残留**（主因）：KB 组终版保留 3 个 `@triton.autotune`（NB∈{1,2,4,8,16}，key=(S,N,RD)），µs 级 kernel do_bench 噪声系统性误选 per-shape 配置——per-case 比值呈双峰（1.00 vs 1.4~1.95x），34/45 case 显著劣化。直给组 1701 实测 autotune 轮 4.1228 < 静态轮 4.2060 后主动拒绝；1637 也记录过该噪声。且 NB<N 的重分块会重新引入 cos/sin 按 chunk 重载，部分抵消外提收益。
② **知识形态差距**：1701 拿到卡片全文 + 197 行可运行 triton_skeleton.py；1636 只拿到 KB 卡 §8 的 57 行浓缩伪码，且卡内「完整骨架见 output/.../triton_skeleton.py」是本地路径死引用（grep 确认 1636 全程未读骨架）。baseline 差距（2.05 vs 1.35）即在此。
③ **上下文稀释**：1636 注入 134KB（10 张 ops 卡 + 78KB preflight + 无关 runbook），优化器沿 kernel_splitting.md 走弯路白烧 2 轮回滚（0.2155x/2.117x）；1701 优化器读 loop-invariant-hoisting.md 一轮命中 +100%。
④ **对照不完美**：target_speedup 0.8 vs 8.0、device 5 vs 2 等环境差异存在，但不足以解释主趋势。

**流程整改（已落地）**：KB 卡 §8 改为自包含完整伪码、删除本地路径引用；README P5 增补「骨架是 bundle 资产 / 禁止死引用」规范；µs-kernel 禁 autotune 沉淀为 runbook 卡。

## 5. 回放暴露的卡片缺陷与修订（12 项知识使用审计摘要）

| # | 缺陷 | 类型 | 处置 |
|---|---|---|---|
| 1 | quarter 分支骨架笔误 `o4 = x4*c4 - x3*s4`（变量错位） | 正确性 | 卡片/骨架均已修正；README P2 增「骨架静态自检」动作 |
| 2 | cos/sin 外提曾被判「L1/L2 驻留等价、勿复刻」 | 错误断言 | 实测推翻（+100%），卡片 §7 改为「必须显式复刻」；README 增「勿复刻清单禁止无实测等价断言」 |
| 3 | strided-2 访存未标正确性风险 | 缺陷面缺失 | 沉淀 runbooks/memory-sram/triton-ascend-strided2-interleave-contract.md（3 类 bug + 安全写法） |
| 4 | shifted-load 劣化未记录 | 性能陷阱 | 沉淀 runbooks/compute-pipeline/triton-ascend-shifted-load-degradation.md |
| 5 | tl.trans 标量化 ~15x 未记录 | 性能陷阱 | 沉淀 runbooks/compute-pipeline/triton-ascend-trans-scalarization.md |
| 6 | num_stages/num_warps 静默不生效 | 后端契约 | 沉淀 runbooks/autotuning/triton-ascend-microkernel-autotune-noise.md（优先级高于库内旧卡） |
| 7 | fp32 mask 污染 / stride-0 memref / `%` 禁令 | 后端契约 | 沉淀 runbooks/portability/triton-ascend-mask-codegen-traps.md |
| 8 | (b,s) 组外提杠杆未显式写 | 优化决策缺失 | 沉淀 runbooks/compute-pipeline/loop-invariant-group-hoisting.md |
| 9 | KB 卡 §8 非自包含 + 本地路径死引用 | 入库规范 | KB 副本 §8 重写为完整伪码；README P5 成文 |
| 10 | Q/K 两 launch 融合疑问 | 实测裁决 | 融合 3.8252 < 拆分 4.2060，卡片 §4 注明实测定局 |
| 11 | in-place 语义缺图级证据 | 材料漏扫 | 补 tensormove 融合 pass 证据；README P0 材料表增「图级语义」行 |
| 12 | 官方验收容差/tiling UT 锚点缺失 | 材料漏扫 | 卡片 §9 补 ST 300 case（fp16_error=2⁻⁸、[-1,1] 填充）、tiling UT 950 期望 key（20030/20010）、UB=196608B@L1478、regbase UT 全注释 |

## 6. CANN 源码回头审计（6 项发现，已回流卡片）

1. 官方 ST 300 case 仅 V1 half-only，D∈{64,128}，`fp16_error = 2⁻⁸ = 0.00390625`，填充 [-1,1]——卡片 §9 验收标准对齐官方口径。
2. 官方 executor golden 仅对 bf16 上转 fp32（fp16 不上转）——golden 精度策略 caveats 已注明。
3. tiling UT Ascend950 锚点（test_apply_rotary_pos_emb_tiling.cpp:1459-1607，期望 key 20030/20010，40 AIV 核，UB=196608B @L1478）——卡片 §6 内存预算有官方锚点。
4. legacy ub_required 公式根因解释 D∈{64,128} 限制——卡片 §2 约束说明。
5. docs V2 空 tensor 行自相矛盾（q/k vs cos/sin）——卡片 §2 注明以代码为准。
6. regbase kernel UT 全注释（不活跃）——测试资产活性已标注。

## 7. 判定

实验组（直给卡）在迭代轮数与终值上显著优于对照组，KB 检索组介于两者之间且差距已归因、可修复（autotune 残留为主因，已沉淀 runbook）。**卡片 confidence: inferred → verified**（双副本同步），六张后端契约 runbook 卡入库并登记 index。
