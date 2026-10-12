# P2 语义转化 — apply_rotary_pos_emb（Ascend C → Triton，新流程版）

> 输入：extraction.md（2026-09-14 复验版）。每条 P1 事实在此有去向（转化/合并/显式丢弃+理由）。
> 可达性三态：✅ 直接可达 / ⚠️ 编译器自动或需权衡 / ❌ 不可达（含降级方案）。
> **与 2026-09-08 旧版（baseline_20260909/mapping.md）的差异**：本版按『参考源与检索指引』
> 对六类后端论断强制取证（backend_facts → zh → IR 源码），每条标 [EXP]/[DOC]/[SRC]；
> §0.5 为六任务检索记录；旧版三处错误/缺证据论断在本版修正（见 §0.6）。

## 0. 总体转化决策：4 条 tiling 路径 → 单 kernel + constexpr

arch35 四个 kernel 类（AB/BAB/ABAAndBA）**计算数学完全相同**（都调 common.h 的
Half/Quarter/Interleave VF），差异只在：① 循环结构（BS 打平 vs B×S 二维）；
② cos/sin 广播是否载一次；③ Q/K 边界处理。三类差异由「strides 参数化 + constexpr 分支」
吸收——**单 kernel + `ROTARY_MODE: tl.constexpr` + 布局 strides**。layout 差异由 wrapper
算出 (stride_b, stride_s, stride_n) 传入（TND 时 stride_b=0、b=1）。

## 0.5 检索记录（新流程六任务，证据全标）

| # | 检索任务 | 命中 | 结论 |
|---|---|---|---|
| ① | 骨架各 tl op 的 lowering 分级 | `tl_op_lowering.md`；`dtype_support.md` 附录 A | tl.load/store/算术 = T1；`tl.split/tl.interleave` 对应 HIVM VDeinterleave/VInterleave，**fp32 向量原生，仅 i64 标量化** [SRC ShouldLowerToScalarLoops.cpp:71-72]；无 tl.scan/tl.gather（T4 空集） |
| ② | ⚠️/❌ 判定依据 | `backend_contracts.md` C-012；runbook strided2 契约（[EXP]）；`extension_apis.md` | strided-2 半宽 load/store 由旧版「⚠️ 默认」**降为 ❌ 禁用**（3 类后端 bug [EXP]）；DeInterleave 降级路径走 `tl.split+tl.interleave` ✅，无需 al/bl |
| ③ | 编译选项 | `compiler_options.yaml` by_operator_class.vector | 纯向量算子：multibuffer（bishengir 默认 true [SRC Options.td]，勿再配 num_stages——C-006）；grid 已为物理核数，auto_blockify_size 不需要；µs kernel 禁 autotune（C-007） |
| ④ | UB/对齐预算常数 | `hardware_platforms.md` | **950 UB=248KB 可用**（旧版按 192KB 校核，偏保守；UT mock 196608B 非硬件规格）；UB 对齐 32B [SRC NPUTargetSpec.td:223-235] |
| ⑤ | dtype 陷阱 | `dtype_support.md` §2.1/§3 | i64 全禁；**i32 不等值比较标量化，仅 i32 EQ/NE 向量安全** [SRC]——但 store mask 必须整型（C-002 正确性优先）；fp32 计算域与 common.h RegTensor\<float\> 一致 |
| ⑥ | 契约违禁写法 | `backend_contracts.md` C-001/C-002/C-006/C-011 | 禁 `%`（写 `a-(a//b)*b`）；mask 整型比较；无 num_stages/num_warps；persistent-core grid=物理核数 |

## 0.6 对旧版 P2 的三处修正

| # | 旧版论断 | 本版修正 | 依据 |
|---|---|---|---|
| 1 | §4「vs UB 192KB/核」 | 950 按 **248KB 可用**校核（192KB 是 A2 值；结论方向不变、富余更大） | hardware_platforms.md [SRC] |
| 2 | §3/§5「num_stages autotune {1,2,3}」「双缓冲→num_stages」 | **num_stages/num_warps 运行时不生效，骨架与候选配置一律不得出现**；双缓冲诉求由 multibuffer 选项承接（默认 true） | C-006 [EXP]；compiler_options.yaml [SRC] |
| 3 | §5「strided-2 两次半宽 load/store 为默认，split/interleave 为变体，实测择优」 | **strided-2 禁用**（静默错值 + masked store 相邻行污染 [EXP]）；默认即连续 load + `tl.split/tl.interleave` 寄存器重排（fp32 向量原生 [SRC]） | strided2 runbook [EXP]；dtype_support 附录 A [SRC] |

（另：旧版 §5「cos/sin 小 tile 自然驻留 L1/L2、勿复刻」已被其自身 P4 实测推翻（+100%），
并沉淀为契约 C-012「cache 驻留 ≠ 复用」——本版直接以**组级外提为默认设计**，见 §5。）

## 1. 分核策略 → grid 设计

| Ascend C 事实 | Triton 决策 | 可达性 | 依据 |
|---|---|---|---|
| AB：BS 打平 `blockFactor=CeilDiv(bs,coreNum)`（tiling_ab L69-71）；ABA/BA：B×S 二维分核；BAB：usedCoreNum 比选 | **(b,s) 组为最小任务单元**（组内 n 行+D 维展开为单 tile），persistent-core：grid=min(BS, vector_core_num)，核内连续段循环 | ✅ | C-011 范式 [EXP+DOC]；行间独立（rope 只耦合行内 D 维配对）；连续块划分使相邻 bs 共享 cos 行（L2 友好） |
| Q/K 两路一次 launch（核内先 Q 后 K 复用 cos/sin） | **两次 launch**（q、k 各一次） | ✅ | [EXP] 实测定局：融合 3.8252 < 拆分 4.2060（validation_report §5-10）；跨界拼接逻辑（ab.h:142-161）随之不需要 |
| 尾核少载（blockTailBS） | 连续段划分 `per_prog=cdiv(BS,num_progs)`，尾 program 自然短 | ✅ | 与 C-011 范式同构 |
| 核数来源 | `get_device_properties` / torch_npu 运行时查询，**禁止写死** | ✅ | hardware_platforms §3 规则 1 [SRC]：950 向量核 8~72 跨度 9 倍 |

## 2. tilingdata → BLOCK 超参

| tilingdata 字段 | 语义 | Triton 对应 | 推导式 |
|---|---|---|---|
| realDim (RD) / dSplitCoef | 旋转维长 / 配对系数（half=2, interleave=1, quarter=4） | `RD` 形参 + `SEG` + `ROTARY_MODE: tl.constexpr` | SEG=RD//2（half/interleave）；RD//4（quarter） |
| dAlign = CeilAlign(RD/coef, 32B/typeSize)×coef | UB 对齐行宽 | `BLOCK_DH: tl.constexpr` | `next_pow2(SEG)`；fp32 计算域下恒为 32B 倍数，C-010 天然满足 |
| ubFactorS/B/QN/KN | 核内 UB 切分 | `BLOCK_N: tl.constexpr`（组内 n 行单 tile） | `next_pow2(N)`（N≤128 主流；更大需改内层循环） |
| isPartialRope | RD<D | wrapper：`out=x.clone()` 保底直通段，kernel 只写 [0,RD) | RD==D 走 empty_like 快路径 |
| blockNum/Factor 系列 | 分核 | 不需要——persistent-core 隐式覆盖 | — |
| 4095 blockCount 上限 | DataCopy 硬件限制 | 丢弃 | Triton 无 DMA blockCount 概念 |

## 3. tilingkey / 模板分支 → constexpr

| tilingkey / 分支 | 分支语义 | Triton 处理 |
|---|---|---|
| 20010(ABA)/20011(BA)/20020(BAB) | cos/sin B 维广播与否 | `scb` 参数：CosB==1 传 0——广播退化为寻址 |
| 20030(AB) SBND cosb==1 时 b 折进 n | 维度折叠 | 不需要——strides 参数化覆盖 |
| rotaryMode∈{1,2,3} → 三 VF | 三种配对数学 | `ROTARY_MODE: tl.constexpr` 单 kernel 三分支 |
| BUFFER_NUM=2 双缓冲（4 队列） | 流水掩盖 | ⚠️ 编译器自动：`multibuffer`（bishengir 默认 true [SRC Options.td]）；**不用 num_stages**（C-006 [EXP]） |
| dtype 模板（DTYPE_QUERY） | fp16/bf16/fp32 实例化 | 指针 dtype 自动分派 + `.to(tl.float32)` 统一计算域 |
| legacy key 1/3/4 + BF16 cast 模板参 | 老平台路径 | 丢弃——目标平台 950；BF16 由统一 fp32 计算域覆盖 |

## 4. 片上内存 → 隐式化 + 预算校核

| Ascend C buffer（ABA 为例） | Triton 等效 tile | 字节数校核（BLOCK_N=8, RD=128 half→BLOCK_DH=64, fp16） |
|---|---|---|
| qIn/qOut：ubFactorB×S×N×dAlign×T×2 | [BLOCK_N, BLOCK_DH] 输入/输出 tile ×2 半 | 8×64×2B×2×2 ≈ 4KB |
| cos/sin：ubFactorS×dAlign×T×2 | [BLOCK_DH] ×4（c1/c2/s1/s2） | 512B，**组级外提后每 (b,s) 只载一次** |
| fp32 Reg 计算域 | 编译器分配的寄存器 tile | ≈12KB 量级 |
| **合计** | 每迭代活跃 < 32KB | **vs 950 UB 248KB 可用 [SRC]**：富余 ~8 倍；瓶颈不在容量 |

- D≤1024 → 单行 rope 区最大 1024 元素（fp32 4KB），单 tile 放得住。
- interleave 模式连续行宽 2×BLOCK_DH 同样远低预算。

## 5. 流水编排 → Triton 可达手段

| Ascend C 手段 | 可达性 | Triton 写法 / 依据 |
|---|---|---|
| CopyIn→Compute→CopyOut 三段式 | ✅ | (b,s) 组循环内 load→compute→store |
| LoopModeParams 多维跨步 DMA（BNSD） | ✅ | strides 参数化寻址，编译器生成 DMA（tl.load = T1 [DOC]） |
| DataCopyPad 越界保护 | ✅ | `tl.load(ptr+offs, mask=..., other=0)` |
| TQue 双缓冲 ×4 | ⚠️ 编译器自动 | multibuffer 默认开 [SRC]；无 num_stages（C-006） |
| **cos/sin 组级复用**（BA 广播载一次，aba_and_ba.h:185-196） | ✅ **必须显式复刻** | (b,s) 组循环不变量外提：c1/c2/s1/s2 在组首 load 一次。[EXP] +100%（2.0956→4.2060）；C-012 禁止「cache 驻留等价」断言 |
| MicroAPI DeInterleave/Interleave | ✅ | 连续 load + `tl.reshape(...,2)` + `tl.split` / `tl.interleave` 寄存器重排；HIVM VInterleave/VDeinterleave fp32 向量原生 [SRC] |
| ~~strided-2 半宽 load/store~~ | ❌ **禁用** | [EXP] 三类后端 bug：静默错值、并发 masked strided-2 store 相邻行污染、grid=1 不复现（strided2 runbook） |
| MicroAPI UpdateMask 逐 VL 尾 | ✅ | `mask = i < SEG` 整型比较（C-002） |
| RegTensor\<float\> 全程 fp32 | ✅ | `.to(tl.float32)` + 末段 `.to(out_dtype)` RTNE |
| partial rope 双段搬运 | ✅ | wrapper clone 保底 + kernel 只写 [0,RD) |
| Q/K 跨界块拼接 / workspace 16MB | 丢弃 | 两次 launch / Triton 不需要 |

**❌ 不可达项：0 个**（strided-2 是「禁用」而非「不可达」——有等价安全写法）。

## 6. Compute → Triton 计算骨架

统一式 `out = x·cos + rotate(x)·sin`，fp32 计算域：

| 模式（VF 锚点） | Triton 逐指令翻译 | 可达性 |
|---|---|---|
| half（common.h:111-119 已复验）：out₁=x₁c₁−x₂s₁，out₂=x₂c₂+x₁s₂ | 两个半宽 load（偏移 i 与 i+SEG）+ 乘加 | ✅ |
| quarter（common.h:193-209）：段内同 half，两段对 | 同构 ×2（偏移 2SEG/3SEG） | ✅ |
| interleave（common.h:270-280）：out[2i]=xe·ce−xo·se，out[2i+1]=xo·co+xe·so | 连续 load + split → 乘加 → interleave | ✅ |

骨架全文见 `triton_skeleton.py`。**生成前 preflight（backend_contracts 8 项 +
dtype_support 4a~4d 共 12 项）已对骨架执行，0 命中**——合规是构造出来的，不是测出来的
（记录见 ab_comparison.md §2）。

## 7. 精度策略 → 标杆与验收

- fp32 计算域全程保留（common.h 全部 RegTensor\<float\>）；唯一舍入点末段 `.to(out_dtype)`（RTNE）。
- torch golden：`out = x.float()*cos.float() + rotate(x.float())*sin.float()`，三模式参数化。
- 验收：fp32 rtol=1e-5/atol=1e-5；fp16/bf16 对 fp32 标杆 ≤1 ulp；官方 ST 口径
  fp16_error=2⁻⁸、填充 [-1,1]（validation_report §6-1 锚点）；**模式专项**（half 前后半场 /
  interleave 奇偶位 / quarter 四段）；**partial rope 专项**：[RD, D) bitwise 等于输入。
- 填充纪律：输入幅度 [-1,1]（fp16 大量级填充撞 evaluator max_abs cap 的教训 [EXP]）。

## 8. 显式丢弃 / 不涉及清单

| 项 | 处理 | 理由 |
|---|---|---|
| 4 个 tiling 模板 / 4 个 kernel 类 | 合并为单 kernel | §0 已论证 |
| blockNum/Factor/Tail、usedCoreNum 比选 | 丢弃 | persistent-core 隐式覆盖 |
| ubFactor 反解公式 | 丢弃（语义保留为 §4 校核） | Triton 无显式 UB 管理 |
| Q/K 跨界块、N 合并轴 | 丢弃 | 两次 launch [EXP 实测定局] |
| workspace | 丢弃 | 无跨核中间量 |
| legacy 3 key + BF16 cast | 丢弃（进卡片架构差异节） | 目标平台 950 |
| AutoContiguous | wrapper `.contiguous()` | 框架职责 |
| DeepSeekInterleave VF | 不涉及 | 在仓未用资产（extraction §1.1） |

## P2 质量门自查

- extraction §4.1.1~§6 每条事实有去向；❌=0，⚠️=1（multibuffer 编译器自动）。
- 六任务检索全部命中 backend_facts 首层（未下沉 zh/IR 原文——首层均已带锚点）。
- 旧版三处错误论断已修正（§0.6）；所有「等价/不必复刻」类断言均带 [EXP] 或 [SRC]（C-012 合规）。
- 骨架 preflight 12 项 0 命中。
