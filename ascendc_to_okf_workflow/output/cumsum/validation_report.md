# P4 验证报告 — cumsum（非低价值：P2.5 降级①编译门+②正确性门，P4 正常执行）

> 2026-09-17｜策略：manifest 判非低价值（❌≥3）→ P2.5 降级执行①+②（③ simulator/③.5 board/④ IR 跳过，BLOCK 选型与瓶颈定型保持两态，confidence 不升 measured）。NPU Device 0（Ascend910_9372 + bishengir）真编译真执行。

## ① 编译门：3 配置 × 全 constexpr 组合真编译，PASS（抓到 4 个静态扫不出的后端事实）

| # | 事实 | 影响 | 骨架处置 |
|---|---|---|---|
| 1 | **`tl.cumsum`→`func::CallOp` UB 放大 ~48×，tile 元素数硬上限 1024**（16×64/32×32/8×128 过；16×128/32×64 起 "ub overflow, requires 6324224 bits"，需求恒定与尺寸解耦） | autotune 配置网格全部收敛到 ≤1024 | 3 配置 {(16,64),(32,32),(8,128)} |
| 2 | **行序循环备选（逐行 `carry+=row; store(carry)`）WAR 失同步**：输出系统性前移一行、逐 run 非确定（小 case 三连跑 2/3 错） | SCAN_IMPL=1 备选**废弃**；登记为后端模式级陷阱（loop-carried UB 值「同迭代先 store 后原地更新」不可靠） | 只留 tl.cumsum 路径（每迭代新 SSA 值 s，无 WAR） |
| 3 | **降序行地址 tile 存取错误**（reverse 初版 `rows=R−1−…`：只首行写出、值为总和） | reverse 不可镜像寻址 | 改**后缀和两遍法** total−prefix（纯升序） |
| 4 | 前端 `or` 链不支持；`.is_int()` 本地 if 不做 constexpr 折叠（跨块 dtype Mismatched type） | dtype 分支写法约束 | wrapper 传 `IS_INT: tl.constexpr` |

## ② 正确性门：64/64 PASS（4 shape×dim × 4 dtype × exclusive × reverse 全组合对 golden）

- shape 覆盖：尾轴 (64,48)/dim=-1、中间轴 (4,3,40)/dim=1、(2,5,4,24)/dim=2、跨 chunk 长 R (37,300)/dim=0（R=300 跨 19 个 R_CHUNK=16 块，验证 carry 链）。
- 误差分布：int32 全组合**精确相等**（exclusive 代数映射整型精确——闭环验证）；fp16/bf16 46 例零误差；fp32 ≤3.8e-6；最大 9.8e-4（bf16，容差 2e-2 内）。
- exclusive/reverse 代数映射（`s−tile` / 两遍法 `total−prefix`）全组合零超差——**降序 tile 错误的替代方案语义闭环**。

## 结论

P2.5 降级①+② 通过：①编译门 4 条实测后端事实全部转化为骨架硬约束（3 条为 backend_facts 库中**新增**的 scan 相关取证）；②正确性门 64/64。卡片 `cumsum.md` 的 BLOCK 选型、瓶颈定型、tl.cumsum 库调用性能、i64 标量化开销、CORE_SS 放弃跨核并行的性能差距**保持两态**（confidence: inferred，不升 measured(skeleton)）。

未覆盖（按降级策略显式放弃）：③ simulator 诊断（msprof）、③.5 board 时序、④ IR 取证；大 shape 性能；窄 N（官方 twoway 场景）利用率；M×cdiv(N,BLOCK_N)<核数的并行度不足场景；numel≥2³¹ 超域；i64/u64 dtype（骨架未禁，标量化预警已登记卡片）。
