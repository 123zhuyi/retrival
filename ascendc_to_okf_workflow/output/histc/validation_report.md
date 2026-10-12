# P4 验证报告 — histc（histogram_fixed_width）

> 2026-09-18｜环境：torch 2.10.0+torch_npu，Ascend910_9372，CANN 9.1.0，NPU Device 0（红线：TORCH_DEVICE_BACKEND_AUTOLOAD=0 + ASCEND_RT_VISIBLE_DEVICES=0）
> 判定回顾：非低价值（❌≥4：UB scratch 原子/SetAtomicAdd 写回/SyncAll/SIMT 模型）→ P2.5 降级①编译门+②正确性门，③③.5④ 跳过，BLOCK 选型/瓶颈论断两态，confidence=inferred。

## ① 编译门记录（4 轮，抓获 3 条后端事实 + 1 条源内矛盾）

| 轮 | 现象 | 结论与处置 |
|---|---|---|
| 1 | impl=0 编译**通过**但常规 case 全错值（sum_y≈1/4）；impl=1 在 (BLOCK_X=128, BLOCK_BINS=128) 配置 `ub overflow, requires 1606400 bits vs 1572864 available`（自动关 code-motion/multi-buffer 重编译仍失败） | 两条独立事实，见下 |
| 探针 | impl=0 单点探针：全同地址 1024→**128=恰好 1/8**；两值交替 4096→1280；地址全异 63/63 正确 | **向量 scatter atomic 重复地址丢更新**（疑 8-lane 冲突组只一 lane 生效）→ 路线 0 禁用，回流 **C-024**；矩阵路线 UB 硬上限 128×64（C-005 实例，维护记录登记） |
| 2 | impl=2 初版 0-d 标量算术编译失败：`failed to legalize unresolved materialization from () to 'tensor<1xf32>'`（ConvertTritonIRToLinalgIR） | **0-d 张量不可消化** → 改 1 宽向量承载标量语义（顺带无向量内重复地址），回流 **C-025** |
| 3 | 全 case 仅「special: max=+inf」双 impl 同错（500/600，恰丢 100 个 −inf 元素） | 非后端问题——**kernel/golden 源内矛盾**：DoBin:72-74 skip 链在 max==+inf 分支之前，kernel 丢 −inf；golden 保留进 bin 0（torch clamp 语义一致）。验收以 golden 为准 → valid 条件加 `~max_pinf`，矛盾登记 extraction §1/§9 + 卡片 §7 |
| 4 | 全绿 | — |

## ② 正确性门记录

- 覆盖：4 dtype（f32/f16/i32/i64）× 2 有效 impl（1 矩阵/2 标量原子）× 常规 12 组 + 特殊值 4 组（NaN/±inf/越界、min=−inf、max=+inf、全 NaN）+ bins=1 + bins=500（触发路线 1 多 program）= **36 check**。
- 基准：numpy 移植 spec.py golden（fp32 域同序 / int64 域同式 / ±inf 三分支），y int32 **精确相等**。
- 结果：**36/36 PASS**（日志 /tmp/histc_smoke4.log）。

## preflight 执行（契约清单 #1~#16 全扫）

- #15（scatter atomic）命中路线 0 kernel——**有意保留作证据**，wrapper impl=0 raise 禁用，注释明示；路线 1/2 不命中。
- #16（0-d 标量算术）不命中（路线 2 已改 1 宽向量）。
- #4（int64 加法/比较链）命中路线 1/2 的 int64 bin 计算——**登记豁免**：语义必需（`(c−min)·bins` 可超 int32），4a 标量化预警非回归（官方 SIMT 本就标量），卡片 §5 写明。
- 其余 #1~#3、#5~#14 不命中（无 %、无 autotune、无 scan、无降序寻址、无 loop-carried WAR、dtype 分支走 IS_INT constexpr）。

## 质量门结论

- [x] ①门 3 条新后端事实按 v1.4 第 5 条当次回流：C-024/C-025 入 backend_contracts.md（A/C 节）+ preflight #15/#16 + 维护记录 + tags；UB 128×64 上限判重为 C-005 实例不新增条目
- [x] ②门 36/36；卡片 §8 实测摘要与本文一致
- [x] ❌ 项（UB scratch/SetAtomicAdd/SyncAll/SIMT）全部有「结构性不可译+替代机制」论证（卡片 §6）
- [x] 源内矛盾（kernel vs golden 的 −inf/max=+inf 边界）双向登记：extraction §1/§9 + 卡片 §7
- [x] BLOCK 选型两态、confidence=inferred（降级纪律未破）
- **P4 通过，卡片可入 P5 待审队列。**
