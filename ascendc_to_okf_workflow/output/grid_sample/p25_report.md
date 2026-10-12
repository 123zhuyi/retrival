# P2.5 骨架实测门报告 — grid_sample（2026-09-17，首次正向全流程）

> 环境：910b（Phy-ID 2，避开坏卡 0/1/4），CANN 9.1.0，SOC=Ascend910_9372，
> `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。骨架 `output/grid_sample/triton_skeleton.py`。
> 全部数据标 [EXP-skeleton]（小 shape/微基准上下文，外推受 C-012 约束）。

## 四门结果总览

| 门 | 结果 | 产出 |
|---|---|---|
| ① 编译门 | **3 个编译/语义陷阱当场拦截** | libdevice.rint 不支持、反射早退控制流、int64 gather 索引 |
| ② 正确性门 | 48 组合（2D×3mode + 3D×2mode × 3pad × 2ac × 2dtype）**全过** | 3 个语义修复被 smoke 当场定位并验证 |
| ③ simulator 诊断门 | **采集失败（附原因）**：DBI init 失败，两次（含设备空闲重试） | 用微基准 + board 时序替代（见下）；94 骨架同时刻复测 simulator 正常 ⇒ **kernel 特异性失败（gather/SIMT 结构破坏 DBI 插桩）** |
| ③.5 board 时序门 | **骨架 vs CANN 原生（perf.json）= 0.01-0.03x** | gather 税的端到端量化 |
| ④ IR 取证门 | **gather 标量化的机制锚点找到** | HIVMVectorOps.td:1828-1859：「All gather operations are dispatched to the SIMT template regardless…」→ 新契约 C-018 |

## ① 编译门明细（三个拦截，全部已修）

| 陷阱 | 现象 | 修复 | 若流入回放的代价 |
|---|---|---|---|
| `libdevice.rint` 不支持 | TritonToLinalg「unsupported control flow」UNREACHABLE 崩溃（bisect k3 隔离） | 手写银行家舍入 `_rint`（floor+tie-to-even，无 %） | ≥1 轮 gen |
| `_reflect` 运行时标量早退 `if span<=0: return` | 同上编译崩溃（reflection 全组合 FAIL） | 无分支 where 吸收 | ≥1 轮 gen |
| int64 索引张量 gather | dtype_support 注 3 违规（gather 禁 I64）——写码时违反 4e 纪律 | 索引张量 int32 + int64 仅标量基址 | 潜在编译失败 |

**流程价值实证**：三个陷阱全部是「写卡时认为正确、一编译就崩」的类型——
编译门在提取期以分钟级成本拦截，对照 94/85 时代这类缺陷每个耗 1 轮 gen。

## ② 正确性门明细（smoke 对 torch 原生 = CANN 原生，逐组合）

初始矩阵暴露三个语义偏差，全部当场定位修复：

| 偏差 | max_diff | 根因 | 修复 |
|---|---|---|---|
| nearest+zeros | ~0.99 | 越界判定用了浮点坐标（x∈(−0.5,0) 被误杀） | rint 后**整数坐标**判界（torch 语义） |
| bicubic+border | ~0.2 | 浮点坐标先 clamp 改了 4 tap 权重（bicubic 权重非仿射可约；bilinear 恰好等价所以没暴露） | border 改 **tap 级夹取**，浮点不动 |
| bicubic+reflection | ~0.23 | 预折叠采样坐标 + tap 夹取，与 CANN bicubic 路径不符 | **权重用原始小数偏移 + 逐 tap 浮点折叠**（CANN bicubic PerLoopCompute 同构） |

**方法论发现**：bilinear 的「浮点夹取≈tap 夹取」等价性掩盖了 bicubic 的错误
——**同一 padding 语义在不同 mode 下实现路径不同**（CANN 也是分 mode 处理），
正确性门的全组合矩阵是唯一能可靠暴露这类交叉语义偏差的手段。

## ③.5 board 时序门：vs CANN 原生（perf.json，aclnn 实测口径）

| case | shape | mode/dtype | 原生 μs | 骨架 μs | 比 |
|---|---|---|---|---|---|
| 5 | [1,32,96,96] | bilinear/f16 | 51.7 | 5896.8 | 0.01x |
| 7 | [1,4,48,48] | bicubic/f16 | 229.4 | 8984.4 | 0.03x |
| 4 | [1,2,24,24,24]³ | bilinear/f16 | 85.0 | 15401.8 | 0.01x |
| 36 | [4,16,462,462] | bilinear/f16 | 1157.4 | 281946.3 | 0.004x |
| 32 | [1,32,680,680] | bilinear/f16 | 837.3 | 278769.0 | 0.003x |
| 50 | [4,32,460,460] | bilinear/f16 | 1535.0 | 506056.6 | 0.003x |
| 39 | [1,32,930,930] | bilinear/f32 | 1548.5 | 555336.6 | 0.003x |

## ③ 替代证据：gather 微基准（simulator 失败后的决定性测量）

| 形态（同搬运量 28MB fp32，C=16×HW=832×256） | 时间 | 结论 |
|---|---|---|
| 仿射连续 load（非 gather 对照） | **81μs** | 带宽正常（~346GB/s） |
| gather 恒等索引（地址连续） | 1931μs | **标量发射固定税 ~24x，与地址无关** |
| gather 块内连续块间随机 | 1917μs | 小范围局部性无收益 |
| gather 全随机（= 本套件 grid 分布） | 8952μs | **随机性再叠加 ~4.6x**；合计 ~110x |

## 结论（写卡依据）

1. **正确性**：骨架语义已全组合验证（48/48），可以直接进生成回放且预期
   iter_0 接近满分；
2. **性能**：gather 结构性税（~24x 发射 + ~4.6x 随机性）使本骨架对 CANN
   原生有 ~2 个数量级差距——**卡片必须明示这是后端 lowering 结构性问题
   （C-018），不是卡片质量问题**；突围方向只有访存重组（分桶/片上选择，
   ⚠️ [DERIVED] 未实测）或等后端 vgather 向量路径；
3. **P2.5 流程本身四门全部经受住检验**（③ 以微基准替代并记录失败原因，
   符合 skill 的覆盖门禁规则）。

## 产出索引

- 微基准/矩阵脚本：本目录 `p25_*.py`（部分在 /tmp，已归档关键数据于本文）
- simulator 失败现场：`p25_prof/`（仅 dump 无 CSV）、对照复测：
  `gdn2_fwd/p25_pilot/prof_retest/`（正常）
- 契约回流：backend_contracts.md **C-018** + preflight #11
