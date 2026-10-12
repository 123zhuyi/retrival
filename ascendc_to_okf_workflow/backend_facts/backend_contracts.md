---
schema_version: okf.v1
kind: runbook
type: backend_contract
source_family: curated
title: Triton-Ascend 后端契约卡（生成前必读）
description: Triton-Ascend 编译后端的硬性契约与静默陷阱清单，供 P2 语义转化核对与生成前 preflight 静态检查使用。每条契约标注严重性（正确性/性能）、禁用写法、安全替代与证据锚点；证据分四级：[EXP]=回放实测，[DOC]=model/zh 官方文档，[SRC]=AscendNPU-IR 源码锚点，[UNVERIFIED]=未在本仓源码确认。
tags: [triton_ascend, backend_contract, preflight, scalarization, ub_overflow, autotune, grid, alignment, scan_callop, descending_tile, war_hazard, atomic_scatter, zero_d_tensor, vsort_dtype_limit, argmax_key_dtype, selection_loop]
confidence: inferred
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-18T00:00:00Z"
---

# Triton-Ascend 后端契约卡

> 用法：P2 写骨架前通读本卡；P4 生成后按文末「preflight 检查清单」grep 终版代码。
> 争议仲裁顺序：[SRC] > [DOC] > [EXP] > [UNVERIFIED]。凡与本卡冲突的实测结果，回流修订本卡并记录日期。

## A. 正确性契约（违反 = 编译失败 / 结果错误 / 死锁）

### C-001 kernel 内禁用 `%` 取模运算符 [EXP]
- **严重性**：正确性（编译/运行失败）。
- **禁用**：`x = a % b`。
- **替代**：`x = a - (a // b) * b`。
- **锚点**：apply_rotary_pos_emb P4 回放经验（工作流 README 2026-09-08/09 增补，已沉淀 runbooks/portability）。
- **⚠️ 冲突待解**：zh 官方示例（`migration_guide/performance_guidelines.md` 指令并行示例 2）kernel 内仍使用 `%`。按仲裁顺序以 EXP 为准；待源码确认 `%` 的具体触发条件（dtype？场景？）后收窄或维持本契约。

### C-002 fp32 导出 mask 触发非确定性 store 污染 [EXP]
- **严重性**：正确性（结果偶发错误，难复现）。
- **现象**：用 fp32 张量构造 mask 时，后端生成非确定性 store，被掩盖 lane 可能写出脏数据。
- **替代**：mask 一律用整型比较产生（`offs < N`），避免 float 比较直接作 mask。
- **锚点**：工作流 README 2026-09-08/09 增补（runbooks/memory-sram）。

### C-003 stride-0 memref 触发常量折叠 MLIR 崩溃 [EXP]
- **严重性**：正确性（编译器崩溃）。
- **现象**：`expand` 产生的 stride-0 维进入某些 memref 路径时后端 MLIR 崩溃。
- **替代**：广播优先用 `tl.broadcast_to` + 显式 mask；wrapper 侧 `expand` 取 stride 的写法需在 P4 先跑通最小 case。
- **锚点**：工作流 README 2026-09-08/09 增补；add 卡「广播走 expand-stride 折算」增补。

### C-004 逻辑核数上限 65535；`TRITON_ALL_BLOCKS_PARALLEL` 的顺序敏感死锁风险 [DOC]
- **严重性**：正确性（超限报错 / 死锁）。
- **事实**：launch 核数 ≤ 65535；开启 `TRITON_ALL_BLOCKS_PARALLEL=1` 后允许 grid > 65535，编译器自动把逻辑核数调整为物理核数。
- **风险**：kernel 逻辑必须对执行顺序不敏感，否则死锁。
- **锚点**：zh `environment_variable_and_compiler_options_reference.md` 环境变量表（TRITON_ALL_BLOCKS_PARALLEL 行）；zh `migration_guide/architecture_difference.md`；pass 实现入口 [SRC] `AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/AutoBlockifyParallelLoop.cpp`。

### C-005 UB OVERFLOW 是编译期硬错误 [DOC]
- **严重性**：正确性（编译失败）。
- **事实**：A2 UB = 192KB（1572864 bits），默认开 double buffer 后可用容量减半。报错形如：
  `error: ub overflow, requires 3072256 bits while 1572864 bits available!`
- **处理**：减小 BLOCK，或改 for 循环分批搬运（长序列场景）。
- **锚点**：zh `programming_guide/index.md`（片上内存容量段、"如何在NPU上避免UB OVERFLOW"节）。
- **平台常数已有权威出处**：见 `hardware_platforms.md`（[SRC] NPUTargetSpec.td）——A2/A3=192KB，**950=248KB 可用（物理 256KB 保留 8KB）**；卡片按平台值校核，勿再手写 192KB。

### C-016 `tl.zeros` 作 loop-carried `tl.dot` 累加初值 → bishengir 段错误（上下文相关）[EXP]
- **严重性**：正确性（段错误，无诊断信息，排查成本高）。
- **现象**：`h = tl.zeros([BK, BV], dtype=tl.float32)` 作为 `for` 循环内
  `h += tl.dot(...)` 的初值时，bishengir 后端在 loop-carried 常量传播路径段错误。
- **证据**：chunk_gated_delta_rule_fwd_h 生成实验中 A1（Phase 3 修复记录）与 A2
  （代码注释）独立命中；B3 相似结构未触发（50/50 通过）——**上下文相关，非必现，
  不赌**。
- **替代**（二选一）：① 全 False mask 的 load 产零：
  `h = tl.load(p + offs, mask=mask & (idx < 0), other=0.0)`（不触碰内存）；② wrapper
  恒传零 buffer，kernel 恒按 USE_INITIAL 路径 load。
- **⚠️ 补充（2026-09-15，Gdn2Fwd 生成实验 iter_0 实证）**：写法①的产零 load
  **dtype 跟随指针**——占位指针若是 bf16（如拿 q 顶位），loop-carried 初值就是
  bf16，循环内 fp32 re-assign 触发「loop-carried 类型不一致」CompilationError
  （42/50；has_init 分支因真实指针是 fp32 而幸存，遮蔽了缺陷）。**各分支都必须
  显式 `.to(目标 dtype)`——即使该分支恒不访存**。
- **锚点**：chunk_gated_delta_rule_fwd_h A1/A2 实验存档（2026-09-09）；
  复盘 `output/chunk_gated_delta_rule_fwd_h/generation_retrospective_20260915.md`；
  `op_94_Gdn2Fwd_20260915_1713_7285/report.md` iter_0 根因。
- **⚠️ 修正（2026-09-21，flash_attention_score_grad P2.5 实证）**：C-016 的
  「tl.zeros 段错误」**不成立**——本套件多种 `tl.zeros` 累加器初值（dkdv 的 dv_acc/dk_acc、
  dq 的 dq_acc，fp32/单/多块）全部正常运行。早前 A1/A2 的段错误更可能来自**并发结构
  （persistent for-t 外层 × 叶子累加循环）或 fp32 运行时循环**，而非 tl.zeros 本身。真实
  罪魁区分见新增 **C-026**。**C-016 降级为「tl.zeros 初值正常可用；真正的杀伤在循环结构」**，
  preflight #9 措辞同步修订。

### C-017 stride 参数全量 `tl.constexpr` 化 → 非确定性 NaN 误编译 + UB 需求爆炸 [EXP]
- **严重性**：正确性（非确定性 NaN——同一 case 不同运行 NaN 数不同，h 条带特定 k 行
  整行 NaN）+ 编译失败（UB overflow 2359296 > 1572864 bits）。
- **现象**：kernel 的全部 stride 参数标 `tl.constexpr` 后，常量折叠使后端选择更大
  DMA 块/多缓冲，BK=128 配置 UB 需求爆炸；部分 case 产生非确定性 NaN（疑似未初始化
  UB 读取/错误代码生成）。NaN 非确定性 ⇒ 编译器误编译，非逻辑错误。
- **证据**：chunk_gated_delta_rule_fwd_h B4 复跑 opt_iter_0 的三变体 bisect
  （2026-09-15，ASCEND_RT_VISIBLE_DEVICES=2）：V1 仅 int32 寻址通过；V2 +标量
  （T/HV/G/NT/NVB）constexpr 通过（max_diff ≤ 3e-8）；V3 +30 个 stride constexpr
  = 非确定性 NaN + UB 爆炸。
- **规则**：**标量属性 constexpr 化可两态（C-013 不变）；stride/步长参数保持运行时
  传参**。单族实测证据（状态常驻递推核，chunk 循环 + 多 stride），按 C-011 证据规则
  其他族须两态实测后方可推广。
- **锚点**：`op_0_85_ChunkGatedDeltaRuleFwdH_20260915_1126_4718/output/opt_iter_0/log.md`。

### C-020 降序/非单调地址 tile 存取错误（只写首行）[EXP-skeleton]
- **严重性**：正确性（**静默错值**，无报错）。
- **现象**：load/store 的地址向量第一维非单调（如 reverse 镜像寻址
  `rows = R - 1 - (c*R_CHUNK + rr)` 降序）时，后端只写出首行（cumsum 实测：首行
  值为总和，其余行静默丢失）。
- **替代**：reverse/镜像语义改**代数两遍法**等纯升序寻址（cumsum：`out = total −
  prefix (+x)`，pass1 求 total、pass2 重算前缀，全部升序）。
- **锚点**：cumsum P2.5 ①编译门（2026-09-17，Ascend910_9372 + CANN 9.1.0），
  `output/cumsum/validation_report.md` ①门第 3 条。
- **上下文**：单点实测；具体触发条件（何种非单调模式、是否仅限 axis-0 降序）未做
  IR 取证——换平台/场景先跑最小 case 验证，方法性结论（**禁非单调地址 tile**）普适。

### C-021 loop-carried 值「同迭代先 store 后原地更新」WAR 失同步（逐 run 非确定）[EXP-skeleton]
- **严重性**：正确性（**偶发错值、逐 run 非确定——能骗过单次冒烟，最危险形态**）。
- **现象**：`for` 循环内 `carry += row; tl.store(ptr, carry)`——store 读的值在下一
  迭代被原地更新，UB 上构成 WAR；后端流水偶发同步插入不足 → 输出系统性前移一行
  （store 捕获到更新后的值），三连跑 2/3 出错。
- **替代**：**每迭代生成新 SSA 值再 store**——`s = f(carry, tile); tl.store(s);
  carry = g(carry, tile)`，使被 store 的值永不被后续迭代原地改写（`tl.cumsum` 块
  路径天然此结构）。显式 barrier 方案未验证。
- **锚点**：cumsum P2.5 ①编译门行序循环备选废弃记录（2026-09-17），
  `output/cumsum/validation_report.md` ①门第 2 条。与 C-016 同属 loop-carried
  陷阱家族。

### C-022 `tl.cumsum`/`tl.scan` → `func::CallOp`：UB 放大 ~48×，tile 元素数硬上限 [EXP-skeleton][SRC]
- **严重性**：正确性（编译失败 ub overflow）。
- **事实**：scan 族 lowering 为库调用（[SRC] `tl_op_lowering.md`:70），UB 需求与
  tile 尺寸解耦——实测（Ascend910_9372 + CANN 9.1.0）：`R_CHUNK×BLOCK_N` ≤1024
  （16×64/32×32/8×128）编译通过，≥2048（16×128/32×64）起报
  `ub overflow, requires 6324224 bits`（恒定 ~6.3Mbit 需求 vs 1.57Mbit 可用）。
- **规则**：**含 scan CallOp 的 kernel 禁用普通 UB 静态预算法**（C-005 的字节数
  校核对 CallOp 失效）；先小配置探编译包络、再把 autotune 网格收敛进包络。
- **上下文**：1024 为本栈单点实测值，换芯片/CANN 版本须重探——方法（探包络→收
  网格）普适，数值不普适。其他 lowering 为 CallOp 的 op（见 tl_op_lowering.md
  T3/T4 级）疑似同类，未逐个实测，引用时标「未经后端确认」。

### C-024 `tl.atomic_add` 向量 scatter：向量内重复地址的更新被静默丢弃 [EXP-skeleton]
- **严重性**：正确性（**静默错值**，编译通过、无报错）。
- **事实**（histc P2.5 ①门探针，2026-09-18，Ascend910_9372 + CANN 9.1.0）：
  `tl.atomic_add(y_ptr + idx, 1, mask=valid)`（idx 为数据相关向量索引）——
  全同地址 1024 次更新只落 **128 = 恰好 1/8**（疑 8-lane 冲突组内只一 lane 生效）；
  两值交替 4096→1280；**地址全异时 63/63 正确**。即：向量 scatter atomic 在
  地址可重复语义下（直方图/bincount/embedding 累加类）**不可用**。
- **替代**：① 重写为无原子算法（histc：program 独占 bin 的等值比较矩阵计数，
  所有权唯一免冲突）；② 必须原子时**逐元素 1 宽向量**（`off + tl.arange(0,1)`）
  循环发射——1 宽向量无向量内重复地址，跨 program 并发由 GM 硬件原子保证
  （= 官方 SIMT `asc_atomic_add` 同机制），实测正确。
- **锚点**：`output/histc/validation_report.md` ①门；与 C-018（gather 标量发射）
  同属离散向量索引访存陷阱家族。
- **上下文**：单点实测；1/8 因子为本栈数值，**「重复地址丢更新」判性结论普适**——
  任何 scatter atomic 直译前必须先过重复地址探针。

### C-026 标量行循环「逐迭代换 base 的行 load + i32 向量累加」非确定错值 [EXP-skeleton]
- **严重性**：正确性（**非确定错值——同输入两次运行结果不同，能骗过单次冒烟**）。
- **事实**（reduce_sum P2.5 ①门探针，2026-09-18，Ascend910_9372 + CANN 9.1.0）：
  中间轴规约 kernel 写成 `for r in range(0, R): tv = tl.load(x_ptr + (m*R+r)*N
  + offs_n); acc += tv`（标量 r 循环、每迭代换 base 的行式 load）——**i32（4B）、
  行宽 N=128（512B 行）时非确定错值**，128/128 元素全错、无部分和/移位规律；
  N=64/96/100（mask 部分假）正常；f16（2B）/i64（8B）同构形态正常；load 后加
  no-op `.to(int32)` 不解。**单 program（grid=1）也复现**——排除核间竞争，指向
  循环流水内 load→累加链的同步插入不足（C-021 同族，机制未 IR 取证）。
- **替代**：**R 维也块状化——2D tile (BLOCK_R, BLOCK_N) 整块 load 累加，末
  `tl.sum(acc, axis=0)`**（尾轴规约 kernel 同构；改写后 23/23 全正确，且更向量
  化）。规避写法的一般形式：循环内避免「标量迭代变量乘行宽」的逐行 load。
- **锚点**：`output/sum/validation_report.md` ①门第 2 条。

### C-027 fp32 运行时循环携带张量点积累加 → 错值/错乱；bf16/fp16 同构可跑 [EXP-skeleton]
- **严重性**：正确性（静默错值，编译通过无报错；fp32 尤其致命）。
- **事实**（flash_attention_score_grad P2.5 ②门，2026-09-21，Ascend910_9372+CANN 9.1.0）：
  - **A. fp32 运行时 n-loop + 张量累加**（即使恒单次迭代，`for n0 in range(0,Sk,BN)`
    包裹一个与 FLAT 等价的身体）：`dq_acc += tl.dot(ds,k_n,"ieee")` → max_abs~0.7-1.8
    错值。同形 **bf16/fp16 PASS**（`_fp32_tiled`：Sq32/Sk64 fp32→~1.7 vs fp16→PASS；
    fp32 Sq128/Sk256、Sq256/Sk512、causal/非 casual 全 FAIL）。
  - **B. persistent for-t 外层 + 叶子 n-loop 携带 acc**：grid=1→1.35e24、grid=8→0.74，
    而 DIRECT(grid=n_tasks，无 persistent) 精确 0.0（`_perloop`）。
  - **置信边界**：dkdv 的 **m-loop** 携带 fp32 acc 却**正常**（Sq1/Sq8/Sq32 fp32 全 PASS）——
  故缺陷不是「fp32 loop-carried acc」全称，而是**叶子 n-loop（循环驱动 reduction 维
  且累加形状=buffer 形状）**或 persistent×叶子组合的特定触发。fp3 2单叶 n-loop 也 FAIL ⇒
  触发更贴[运行时循环驱动出参矩阵的行内规约]；宽断言留待 IR 取证，引用标「机制未锚定」。
- **替代（本套件采用的混合策略）**：**fp32 走 FLAT 单 n-block**（BLOCK_N≥S_k，无循环；
  天然上界 = Cc 单 tile ⇒ S_k≲128@D64）；**bf16/fp16 走 TILED 叶子 n-loop**（任意 S_k、
  D≤128）。修复本缺陷可解锁 fp32 大 S_k 分块。
- **附：精度规则（同门取证）**：bf16/fp16 输入的收缩点积**不得**把 P/dS cast 回输入
  dtype 再 dot（bf16→7bit、fp16→11bit 量化，bad→0.2/0.02）；一律 promote 到 fp32 计算、
  末段一次 RTNE 回写，与 fp32 golden（fp32 einsum over 输入）逐位吻合（45/45 bad=0）。
- **锚点**：`output/flash_attention_score_grad/` 的 `_fp32_tiled.py` / `_perloop.py` /
  `_acc_exp.py` / `_tiled_probe.py` / `_smoke_valid.py`（45/45 综合验证）；p25_report.md。
- **上下文**：本单点实测；「fp32 循环内张量点积累加错误」判性结论普适，具体触发面待 IR。

### C-027 排序硬件指令 `hivm.hir.vsort` 仅 f16/f32：64-bit 键 `tl.sort` 编译失败 [EXP-skeleton]
- **严重性**：正确性（编译失败）。`tl.topk` 64-bit 键可用性**未独立复证**——首测
  错值系键映射笔误（负值分支 XOR 掩码 0x7FFFFFFF 误写，应为全 1）污染，标
  「未经后端确认」，sort 卡 ①门复测。
- **事实**（topk P2.5 ①门探针，2026-09-18，Ascend910_9372 + CANN 9.1.0）：
  triton_patch 的 `tl.sort` bitonic 网被后端模式匹配到硬件排序指令 `hivm.hir.vsort`，
  该 op **验证器只接受 16/32-bit 浮点元素类型**。实测：`tl.sort(int64 packed 键)` →
  编译失败 `error: 'hivm.hir.vsort' op failed to verify that operand ... should have
  element type 16-bit float or 32-bit float`；int32 packed 键装不下「32bit 值键+索引」
  （f32 值键已占满 32bit），packed-key 单趟排序路线整体废弃。
- **规则**：**joint (value,index) 排序/TopK 勿用 `tl.sort`/`tl.topk` 64-bit 键**；
  f16/f32 值域单趟 sort 可用性由 sort 卡 ①门另行取证。替代：迭代 max+argmax
  选择（见 C-028 的 int32 flag 写法）或两行段候选归并。
- **锚点**：`output/topk/validation_report.md` ①门（2026-09-18）；探针
  /tmp/topk_probe4.py（编译失败原文）。

### C-028 `tl.argmax` 的 int64 键：编译 >7min CPU 且结果错；int32 键正确快速 [EXP-skeleton]
- **严重性**：正确性（错值）+ 编译时间（近似挂死）。
- **事实**（topk P2.5 ①门探针，2026-09-18，Ascend910_9372 + CANN 9.1.0）：
  `tl.argmax(int64 键向量, axis=0)` 编译超 7 分钟 CPU 且输出错误位置；
  `tl.argmax(int32 键, axis=0)` 编译快、结果 exact（与 torch 一致）。迭代选择
  环（k 趟「max-reduce + 左most argmax + 掩码」）端到端验证：int32 键路线与
  「int64 max-reduce + int32 flag-argmax（`argmax(where(cur==mx, BLOCK-offs, 0))`
  取左most）」双路线均与 torch.topk 完全一致（含 k=8 全 exact）。
- **规则**：**argmax 键用 int32**（值键 ≤32bit 直接打包；64-bit 值域走「int64
  max-reduce + int32 比较标志 argmax」两步）。选择类算子（topk/kthvalue/分桶）
  优先迭代选择环，勿用 64-bit 键排序网络。
- **锚点**：`output/topk/validation_report.md` ①门；探针 /tmp/topk_probe5-7.py。

### C-029 f16/bf16 `tl.dot` 写 `input_precision="tf32"` 编译被拒；缺省走 `hf32` [EXP-skeleton]
- **严重性**：正确性（编译期断言报错，不静默）。
- **事实**（crossformer_attention P2.5 ①门，2026-09-21，Ascend910 + CANN 9.1.0）：
  `tl.dot(a, b, input_precision="tf32")` 抛
  `AssertionError input_precision == tf32 or tf32x3 is invalid, please use 'hf32' on Ascend`。
  注意 **fp32 输入也不得写 tf32**（C-023/operator_class_patterns §4.2 [EXP]：fp32 用
  `input_precision="ieee"`）；非 fp32（f16/bf16）缺省（不写 input_precision）即解析为
  `hf32`，**原生 fp32 累加**，无需显式 precision 参数。
- **规则**：写 `tl.dot` 时——fp32 输入给 `"ieee"`；f16/bf16 输入**不传 input_precision**
  （或仅测试写 `"hf32"`），禁写 `"tf32"`。
- **锚点**：`output/crossformer_attention/validation_report.md` ①门第 1 条。

### C-030 `p@V` 里 V 的加载方向须 `(BLOCK_N, BLOCK_D)`（key×feature），勿沿用 Kᵀ 方向 [EXP-skeleton]
- **严重性**：正确性（静默错值，rmse~O(1) 且无报错，极难定位）。
- **事实**（crossformer_attention P2.5 ②正确性门，2026-09-21）：flash 注意力里
  `s=dot(q,kᵀ)(M,D)(D,N)` 的 Kᵀ 加载应为 `(BLOCK_D, BLOCK_N)`（feature×key）；
  但 `o=dot(p,v)(M,N)(N,D)` 的 **V 必须按 `(BLOCK_N, BLOCK_D)`（key×feature）** 加载。
  两处若都套用 Kᵀ 的转置方向，V 被转置，`p·V` 收缩维错误 → 输出整体错值（本骨架 7 例全
  rmse~1.0，V 方向修正后全 rmse≤1.3e-4）。编译器不报错，只能靠 golden 抓。
- **规则**：flash 注意力中每个 `tl.dot` 的第二个操作数的收缩维必须对应输入张量的
  key 维——Kᵀ 是 (feature,key)、V 是 (key,feature)，加载维度按各自需要定。
- **锚点**：`output/crossformer_attention/validation_report.md` ②门第 2 条 / mapping §5。

### C-031 flash 在线 softmax 的 key 尾列（G<BLOCK_N/分块尾）必须显式 `-inf`，防假质量 [EXP-skeleton]
- **严重性**：正确性（静默错值；softmax 分母虚增 → 输出被 l 稀释，rmse~O(1)）。
- **事实**（crossformer_attention P2.5 ②正确性门，2026-09-21）：key 分块内超出序列长的列
  若不置 `-inf`，其 `Q·0=0` 的 score 会进入 `tl.max`/`tl.sum` → `exp(0−m_new)=1` 的假质量
  计入 softmax 分母 l，输出整体缩小。修复：`s = tl.where(n_ok, s, float("-inf"))` 后再
  softmax（exp→0），并确保掩码列对 `p·V` 的贡献为 0。
- **规则**：任何 key 分块带掩码的 flash 注意力，softmax 前对越界 key 列置 `-inf`
  （等价 atten_mask/causal 的 -inf 遮蔽语义，C-002/C-031；FAS atten_mask 亦是布尔
  select 置 -inf）。
- **锚点**：`output/crossformer_attention/validation_report.md` ②门第 3 条。

### C-032 大 D（head_dim>128）fp32 累加器 UB 溢出且 BLOCK_N 无效、BLOCK_M 是杠杆 [EXP-skeleton]
- **严重性**：正确性（编译期 `ub overflow` 硬失败，非静默错值）。自动多缓冲/code-motion 关闭仍失败。
- **事实**（crossformer_attention_v2 P2.5 ①编译门，2026-10-10，Ascend910b/9372 CANN 9.1.0）：
  flash 注意力 head_dim>128 时，即使降 BLOCK_N，`tl.dot` 侧 s/p 另加 **fp32 O 累加器（BLOCK_M×BLOCK_D×4B）**
  随 BLOCK_D 线性暴涨 ⇒ `ub overflow, requires 1587200 bits while 1572864 bits available`
  （192KB；超 ~4336 bits f16 hd192/hd256，~8000 bits fp32 hd256）。**BLOCK_N 对峰值影响甚微**
  （s/p 是 BLOCK_M×BLOCK_N，小）；**BLOCK_M 才是主杠杆**（O 累加器减半）。已验证逃逸：
  head_dim=256 f16 BLOCK_M=32/BLOCK_N=64/BLOCK_D=256 → PASS rmse=5e-8。
- **规则**：flash/注意力 kernel 的 head_dim 维受 UB 硬上限——192KB 下 head_dim≤128 用
  BLOCK_M=64/BLOCK_N=128/D=128；head_dim 128~256 须 **BLOCK_M=32/BLOCK_N=64**；>256 需 D-split
  （对齐 CANN arch35 D-workspace）。BLOCK_D 必须 ≥head_dim（勿硬编码小值——`d<HEAD_DIM` mask 会让
  越维部分静默不写，单头大 C 丢特征）。同类数据点：histc BLOCK_X×BLOCK_BINS 128×128 → 1606400 bits。
- **锚点**：`output/crossformer_attention_v2/{p25_report.md,mapping.md}` ①编译门。已回流 README v1.5 取证。

## B. 性能契约（违反 = 静默劣化，最难自查）

### C-006 `num_stages` / `num_warps` 在 Triton-Ascend 运行时不生效 [EXP][UNVERIFIED]
- **严重性**：性能（误导 autotune，系统性误选配置）。
- **现象**：设置后仅静默 WARNING，无任何效果。
- **规则**：骨架与终版代码均不得包含 `num_stages`/`num_warps`；候选配置里也不放。
- **锚点**：apply_rotary_pos_emb P4 三组回放经验（工作流 README）；本仓 triton 适配层未见该 WARNING 实现，待 triton-ascend Python backend 源码确认。

### C-007 µs 级 kernel 保留 `@triton.autotune` 会被 do_bench 噪声系统性误选 [EXP]
- **严重性**：性能（KB 组 34/45 case 显著劣化的实测）。
- **规则**：µs 级 kernel 终版交付前清算 autotune 残留，固化单一最优 config；NPU 计时抖动 ±3~5%，性能差 <5% 视为噪音带。
- **锚点**：apply_rotary_pos_emb P4 经验（工作流 README「回放产物必须清算 autotune 残留」条）。

### C-008 向量 ADD 不支持 int64 → 标量化 [DOC]
- **严重性**：性能（逐元素标量展开，悬崖式劣化）。
- **规则**：在不影响精度时，加法链路避免 int64，改 int32。
- **锚点**：zh `migration_guide/performance_guidelines.md`「数据类型优化核心原则」表。

### C-009 向量 CMP 不支持 int64/int32 → 标量化 [DOC]
- **严重性**：性能。**mask 运算是高发区**（load/store/where 的 mask 比较）。
- **规则**：mask 比较的操作数建议 fp32（如 `cols < N` 中让参与比较的张量为 fp32）。
- **锚点**：zh `migration_guide/performance_guidelines.md`（含完整 diff 示例）。

### C-010 尾轴对齐：VV 类 32B 整除，CV 类 512B 整除 [DOC]
- **严重性**：性能（不足自动补齐，如 shape (2048,3)/(2048,1) 明显恶化）。
- **规则**：tile 尾轴按对齐设计；不对齐场景考虑转置把对齐轴转到低维（「借轴转置」适用于 `numel % 256B == 0`）。
- **锚点**：zh `programming_guide/index.md` 对齐注意条（L122 附近）。

### C-011 分核范式：grid 默认 persistent-core；状态常驻递推核须两态实测 [EXP][DOC]
- **严重性**：性能（elementwise 家族 grid=123 vs 40 核实测 0.69x，负引导）。
- **默认范式**（适用：小块海量任务，per-task 足迹小、task 数远超核数）：
  ```python
  pid = tl.program_id(0)
  NUM_CORE = tl.num_programs(0)   # grid = 物理核数
  for blk in range(pid, NUM_BLOCKS, NUM_CORE):
      ...
  ```
  纯 Vector 算子 grid=`num_vectorcore`；CV 融合算子 grid=`num_aicore`（Cube 数，Vector 按 1:2 配比调用）。
- **⚠️ 适用范围修正（2026-09-15）**：上述默认的证据基是 elementwise 家族
  （fused_bias_leaky_relu 等）。对**片内状态常驻的递推核**（每 task 有大块常驻 tile、
  task 内为长串行循环，如 chunk 间状态递推），两个独立 [EXP] 数据点均是一块一
  program 胜出：
  - MCTS 0901（chunk_gated_delta_rule_fwd_h，910b1）：一块一 program 冠军 7.2335x；
  - 2026-09-15 生成实验（同算子，B 组卡包）：persistent-core 版 50 case 中 13 个
    UB 溢出（首轮 37/50），改一块一 program 后推进并最终通过。
  机制假设（**未验证**）：multibuffer 对外层领任务循环做流水会复制大块瞬态 tile，
  常驻 tile 活跃区间跨 task 延长 → UB 占用被放大；一块一 program 时流水只作用于
  内层循环的小搬运。注意混杂因素：该实验中 grid 改动与 BV 缩减同期发生，
  persistent 本身的纯净性能差未被隔离测量。
- **规则**：**同算子实测证据 > 本家族默认 > 跨家族外推**。状态常驻递推核的 grid
  形态必须两态实测，卡片两态并列呈现、不写死。
- **取核数**：`driver.active.utils.get_device_properties(device)` → `num_vectorcore` / `num_aicore`。
- **锚点**：zh `programming_guide/index.md`「设置最大硬件核数」节；fused_bias_leaky_relu P4 经验（工作流 README）；chunk_gated_delta_rule_fwd_h 生成实验复盘（`output/chunk_gated_delta_rule_fwd_h/generation_retrospective_20260915.md`）。

### C-012 「cache 驻留 ≠ 复用」：等价断言必须有实测 [EXP]
- **严重性**：性能（曾断言 cos/sin 外提可由 L1/L2 驻留等价，实测逐行重载慢一倍：2.0956→4.2060）。
- **规则**：凡写「不必复刻 X，因为 Y 机制等价」必须附实测或源码级论证，否则改写为「⚠️ 未验证，建议实测两态」。
- **锚点**：apply_rotary_pos_emb P4 经验（工作流 README「勿复刻清单禁止无实测等价断言」条）。

### C-013 float 属性透传给两态，不给禁令 [EXP]
- **严重性**：性能（`tl.constexpr` 常量折叠实测 +17.3%；给错绝对禁令会压住生成器不去试）。
- **规则**：生产语义默认 runtime 标量；benchmark/编译期稳定场景建议 `tl.constexpr`。卡片给两态权衡，不写死。
- **锚点**：fused_bias_leaky_relu P4 经验（工作流 README）。

### C-018 离散 gather load（tl.load 向量索引）标量发射：恒等索引即慢 ~24x [EXP-skeleton][SRC]
- **严重性**：性能（悬崖式劣化，gather 主导型算子与原生实现差 1-2 个数量级）。
- **现象**（微基准 [EXP-skeleton]，910b、CANN 9.1.0、fp32、C=16/HW=832×256）：
  仿射连续 load 81μs vs gather 恒等索引 1931μs（**慢 24x，与地址无关的固定
  标量发射税**）vs gather 块内连续块间随机 1917μs vs 全随机 8952μs（**随机性
  再叠加 ~4.6x 访存税**）。mask 有无、1D/2D 形态无关。
- **机制 [SRC]**：`AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/
  HIVMVectorOps.td:1828-1859` VGatherOp 注释——**「All gather operations are
  dispatched to the SIMT template regardless of whether the gather axis is the
  last dimension or not」**：gather 一律走 SIMT 模板逐元素发射，无向量 DMA 路径。
  （vgather 指令在 td 有定义且支持 I16/I32/F16/BF16/F32，但当前 lowering 不使用。）
- **规则**：gather 主导型算子（grid_sample、interpolate gather 模式、任意流场
  采样、离散索引读）提取时先判「能否重组为连续 load + 片上选择/分桶」——
  不能重组（随机流场）则**卡片必须明示与原生实现的预期数量级差距**，不得按
  常规访存算子预估性能。与 interpolate skill「禁止 gather」条款同源互证。
- **锚点**：`output/grid_sample/p25_pilot/`（微基准脚本与数据）；grid_sample
  骨架 vs CANN 原生实测 0.01-0.03x（p25_report）。
- **突围路径实证（2026-09-17，grid_sample 回放 [EXP]）**：**段连续 gather**
  （NHWC，[BHW,C] 段内连续）微基准 102μs≈连续 load 81μs（88 倍税消除），
  端到端 ×34.6（Phase 3 0.0060→0.2211 vs CANN 原生）；**片上 tl.gather
  全速**（74μs≈连续）——SIMT 税只在 GM 离散 gather。**结构地板**：段
  gather 发射 issue-bound ~98ns/64B 段（in-kernel 31GB/s ≪ 微基准
  213GB/s）；片上 gather 进 UB 的写法（gather_out_to_ub）编译失败
  （bishengir pipeline error 无详细行，待 IR 取证）。
- **待办**：分桶重构（按源行 histogram+scatter）为 ⚠️ [DERIVED] 未实测方向；
  段长 <32B（小 C）退化两态未标定。


## C. 用法约束（违反 = 功能静默失效）

### C-014 autotune 三个硬约束 [DOC]
- `@triton.autotune` 必须直接包在 `@triton.jit` 外层；
- tiling 参数影响 grid 时，grid 必须写成 lambda 形式；
- 参与调优的参数不要被提前固定。
- **扩展**：支持 `configs=[]` 自动生成候选（带 Ascend 硬件约束）、多 config 并行编译。
- **锚点**：zh `autotune_guide.md`。

### C-015 17 个 NPUOptions 的生效条件 [DOC][SRC]
- 部分选项有前置条件，如 `auto_blockify_size` 未设 `TRITON_ALL_BLOCKS_PARALLEL` 时被忽略；`enable_auto_bind_sub_block` / `enable_hivm_auto_cv_balance` / `sync_solver` / `tile_mix_*` 仅 CV 融合算子有效。
- **锚点**：zh `architecture_design_and_core_features.md` §3.2.1 选项表 + zh 环境变量文档编译选项表；对应 pass 实现 [SRC] `AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/`（`AutoBlockifyParallelLoop.cpp`、`EnableMultiBuffer.cpp`、`GraphSyncSolver/`、`TileCubeVectorLoop.cpp`、`CVPipelining.cpp` 等）。

---

### C-019 verify/benchmark 环境红线：`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 必设 [EXP]
- **严重性**：正确性（**大面积伪失败**——数值错位/NaN/整组报错，极易误判为代码 bug）。
- **现象**：未设时 torch 双 npu 后端注册冲突（报 Two accelerators cannot be used 或
  静默产生 NaN/错位）。**grid_sample 生成实验 iter_0 实证（2026-09-17）**：
  verify 报 44/50 失败（35 例「Framework 有 NaN」+ 9 例精度不达标），但同一代码
  在孤立环境全 50 case matched≥0.99 零 NaN——**伪失败率 88%**。
- **规则**：① 所有 verify/benchmark 运行环境显式 `export TORCH_DEVICE_BACKEND_
  AUTOLOAD=0`；② **此红线应内建于 verifier skill 的运行脚本，而非依赖 agent
  笔记**（template 臂 L1.9 仅以约束文本存在，本次事故证明笔记传承不可靠）；
  ③ 排查顺序：凡「参照侧 NaN/大面积精度失败但孤立复跑正常」，先查环境再查代码——
  **先孤立复跑，再改代码**（防把正确代码修错，B3 型风险）。
- **⚠️ 补充（2026-09-17，机制定案）**：grid_sample 事故的完整机制 =
  verify.py「**CPU golden 优先**」+ **CPU fp16 grid_sample 实现自身缺陷**
  （实测：CPU fp16 bilinear 产 49-136 个 NaN/54 万元素、nearest max_diff
  1.99；同输入 CPU fp32 / NPU aclnn 全正常）——fp16 case 的 golden 全是
  坏参照。**凡 verify 报「Framework 有 NaN / fp16 大面积精度失败」，
  先查 golden 跑在哪个设备**（CPU fp16 对部分算子是不可用参照）。
- **锚点**：template 臂 precheck L1.9（op_94_94_Gdn2Fwd_20260914_1219_8124）；
  grid_sample iter_0 伪失败分析（`op_12_12_grid_sample_20260917_1453_7647` vs
  孤立复跑实验记录）；CPU fp16 golden 缺陷实测（/tmp 微实验，2026-09-17）。

### C-023 前端语法两约束：`or` 链不支持；dtype 本地分支不 constexpr 折叠 [EXP-skeleton]
- **严重性**：用法（编译期明确报错，不静默，但每条浪费一轮编译迭代）。
- **事实**（cumsum P2.5 ①编译门，2026-09-17）：
  ① `A or B or C` 链式布尔 → `UnsupportedLanguageConstruct: chained boolean
  operators not supported`——用括号拆开或改写法；
  ② kernel 内 `if x_ptr.dtype.element_ty.is_int():` 这类 **dtype 本地分支不做
  constexpr 折叠**，两分支出不同 dtype 的同名变量即报 `Mismatched type for
  <var> between then block and else block`——**dtype 分支一律由 wrapper 判好后传
  `tl.constexpr` 参数**（如 `IS_INT=not x.is_floating_point()`）。
- **锚点**：`output/cumsum/validation_report.md` ①门第 4 条。

### C-025 0-d 标量张量算术编译失败：标量语义用 1 宽向量承载 [EXP-skeleton]
- **严重性**：用法（编译期明确报错，不静默）。
- **事实**（histc P2.5 ①门，2026-09-18）：kernel 内对 `tl.load(x_ptr + off)`
  得到的 0-d 张量做算术/比较，triton-adapter-opt 的 ConvertTritonIRToLinalgIR
  阶段报 `failed to legalize unresolved materialization from () to
  'tensor<1xf32>'`——后端 linalg 转化不消化 0-d 张量。
- **替代**：标量循环体一律用 **1 宽向量**承载（`offs = off + tl.arange(0, 1)`，
  load/compute/store/atomic 全在 [1] 形状上做）——顺带规避 C-024（1 宽向量无
  向量内重复地址）。
- **锚点**：`output/histc/validation_report.md` ①门第 2 条。

## preflight 检查清单（P4 生成后、验收前对终版代码执行）

| # | grep 模式 | 命中即违反 | 处置 |
|---|---|---|---|
| 1 | `%\s*[^=]`（kernel 体内取模） | C-001 | 改 `a-(a//b)*b` |
| 2 | `num_stages\|num_warps` | C-006 | 删除 |
| 3 | `@triton.autotune`（µs 级 kernel 终版） | C-007 | 固化 config 后重测 |
| 4 | `int64`（加法/mask 比较链路） | C-008/C-009 | 降 int32 / 改 fp32 比较 |
| 5 | `grid\s*=\s*(` 形态与算子类不匹配 | C-011 | 小块海量任务 → persistent-core；状态常驻递推核 → 两态并列并实测 |
| 6 | 属性参数未标 `tl.constexpr`（benchmark 稳定值） | C-013 | 实测两态取优 |
| 7 | mask 由 float 比较直接产生 | C-002/C-009 | 改整型/fp32 规范比较 |
| 8 | 尾轴 BLOCK 非 32B(VV)/512B(CV) 倍数 | C-010 | 对齐或转置方案 |
| 9 | `tl.zeros` 初值进入含 `tl.dot` 累加的 for 循环 | C-016 | 改全 False mask load 产零 / wrapper 传零 buffer |
| 10 | stride/步长参数标 `tl.constexpr` | C-017 | 改运行时传参（标量属性不受限，见 C-013） |
| 11 | `tl.load` 地址含运行时数据索引张量（gather） | C-018 | 判可否重组连续 load+片上选择；不可则卡片明示数量级差距预期 |
| 12 | `tl.cumsum`\|`tl.scan` 且 tile 各维乘积 >1024 | C-022 | 缩 tile 探包络，autotune 网格收敛进包络 |
| 13 | load/store 地址含 `R - 1 -`、`- offs` 等降序/非单调构造 | C-020 | 改代数两遍法等纯升序寻址 |
| 14 | for 循环内 `store(X)` 后下一迭代原地更新 `X`（人工核，grep 辅助：循环体同时含 `tl.store` 与同名变量 `+=`） | C-021 | 每迭代新 SSA 值再 store |
| 15 | `tl.atomic_add`/`tl.atomic_*` 地址为数据相关向量索引（scatter） | C-024 | 改无原子算法；或逐元素 1 宽向量循环发射 |
| 16 | `tl.load(x_ptr + off)`（off 为标量）后对结果做算术（0-d 张量） | C-025 | 改 `off + tl.arange(0, 1)` 1 宽向量承载 |
| 17 | 标量 for 循环内行式 load（地址含 `循环变量 * 行宽` 逐迭代换 base）+ 整型向量累加（人工核，grep 辅助：循环体含 `tl.load` 且地址表达式含裸循环变量乘法） | C-026 | 改 2D tile (BLOCK_R, BLOCK_N) 块状 load + `tl.sum(acc, axis=0)` |
| 18 | `tl.sort`/`tl.topk` 键为 int64 packed（值+索引联合排序） | C-027 | 废弃 packed-key 排序；改迭代 max+argmax 选择（C-028）或两行段候选归并 |
| 19 | `tl.argmax` 键为 int64（人工核：argmax 实参键向量 dtype） | C-028 | 键降 int32；64-bit 值域走「int64 max-reduce + int32 比较标志 argmax」两步 |
| 20 | 注意力 kernel 的 `tl.dot` 写 `input_precision="tf32"`；或 `p@V` 的 V 加载沿用 Kᵀ 的 (D,N) 方向；或 key 分块带掩码未对越界列 `-inf`（人工核：flash 注意力） | C-029 / C-030 / C-031 | fp32 用 `ieee`、f16/bf16 不传 precision；V 按 (N,D) 加载；softmax 前 `tl.where(n_ok, s, -inf)` |
| 21 | flash/注意力 kernel 的 BLOCK_D 硬编码过小、留 `d<HEAD_DIM` mask；或 BLOCK_M=64 未按 head_dim>128 降 32（fp32 O 累加器撑爆 UB，192KB）| C-032 | BLOCK_D=next_pow2(head_dim)；head_dim≤128 用 M64/N128，128~256 用 **M32/N64**，>256 走 D-split |

> preflight 全部通过才可进入 P4 性能测量；命中修复后须重跑正确性。
>
> **执行实现要点**（2026-09-14 apply_rotary_pos_emb 重提取回流）：
> ① 扫描前剥离注释与 docstring——头注中的禁令文字（如「禁 %」「num_stages」）会假阳性；
> ② 模式 1/4 的适用范围是 `@triton.jit` kernel 体，wrapper（host Python）里的 `%`、`int64`
> 不违禁——按函数边界切分后再 grep；
> ③ 推荐实现：先按 ast/行扫描剥离注释，再提取 `@triton.jit` 函数体做模式匹配。

## 维护记录

- 2026-09-18 **新增 C-027/C-028**（topk P2.5 ①编译门探针实测，README v1.4 第 5 条回流）：
  ① 排序硬件指令 `hivm.hir.vsort` 仅 f16/f32 元素类型——`tl.sort` int64 键编译失败
  （验证器报错原文入档）、`tl.topk` int64 键静默错值；int32 packed 键装不下
  「32bit 值键+索引」，packed-key 联合排序路线整体废弃。② `tl.argmax` int64 键
  编译 >7min CPU 且错值、int32 键正确快速；迭代「max-reduce + int32 flag-argmax
  （左most）+ 掩活标志」选择环双路线（int32 / int64 max+int32 flag）端到端
  exact。preflight 增 #18/#19。同日 ①门附带确认 `tl.rsqrt` 后端可用（见
  tl_op_lowering.md 维护记录 2026-09-18，layer_norm 卡）。
- 2026-09-18 **新增 C-026**（reduce_sum P2.5 ①编译门实测，README v1.4 第 5 条回流）：
  标量行循环「逐迭代换 base 的行 load + i32 向量累加」非确定错值（单 program 复现，
  f16/i64 同构未触发，C-021 同族）→ 2D tile 块状加载替代。preflight 增 #17。同日
  ②门顺带确认：i64 规约（vreduce i64）在本栈编译/正确性通过——与 dtype_support.md
  「A2/A3 标量化」[SRC] 一致（性能标量化、正确性不受限），不新增条目。
- 2026-09-18 **新增 C-024/C-025**（histc P2.5 ①编译门实测，README v1.4 第 5 条回流）：
  `tl.atomic_add` 向量 scatter 向量内重复地址丢更新（全同地址 1024→128=1/8，静默错值）；
  0-d 标量张量算术编译失败（`() → tensor<1xf32>` materialization），标量语义用 1 宽
  向量承载。preflight 增 #15/#16。另：histc 矩阵计数路线实测 BLOCK_X×BLOCK_BINS
  硬上限 128×64（128×128 → ub overflow 1606400>1572864 bits，自动降级重编译仍失败），
  属 C-005 常规 UB 预算的实例数据点，不新增条目。
- 2026-10-10 **新增 C-032**（crossformer_attention_v2 P2.5 ①编译门，README v1.4 第 5 条回流，
  v1.5 排除法≠穷举）：大 D head_dim>128 flash kernel fp32 O 累加器 UB 溢出（1587200 vs 1572864 bits，
  自动多缓冲/code-motion 关闭仍溢出）；BLOCK_N 无效、**BLOCK_M 是主杠杆**（32 逃逸 rmse=5e-8）；
  BLOCK_D 必须 ≥head_dim。与 C-005/histc UB 数据点同类（D 轴实例）。
- 2026-09-21 **新增 C-029/C-030/C-031**（crossformer_attention P2.5 ①②门实测，
  README v1.4 第 5 条回流）：f16/bf16 `tl.dot` 写 `input_precision="tf32"` 被拒、
  缺省走 hf32；`p@V` 的 V 加载须 `(BLOCK_N,BLOCK_D)`（key×feature）勿沿用 Kᵀ 方向；
  flash 在线 softmax 的 key 尾列须显式 `-inf` 防假质量。preflight 增 #17（flash 注意力
  相关）。
- 2026-09-18 **新增 C-020/C-021/C-022/C-023**（cumsum P2.5 ①编译门实测，README v1.4
  「P2.5 实测回流」立法首批实例）：降序/非单调地址 tile 只写首行；loop-carried 值
  「先 store 后原地更新」WAR 失同步（逐 run 非确定）；`tl.cumsum` CallOp UB 放大
  ~48×、tile 元素数硬上限 1024（本栈单点值）；前端 `or` 链不支持 + dtype 本地分支
  不折叠。preflight 增 #12~#14。

- 2026-09-11 初版：整合 add / fused_bias_leaky_relu / apply_rotary_pos_emb 三轮 P4 教训 + zh 文档事实 + AscendNPU-IR pass 锚点。待办：C-006 待 triton-ascend Python backend 源码确认；A3/950 的 UB/L1 常数待从 target spec 抽取。
- 2026-09-15 **C-011 适用范围收窄**：chunk_gated_delta_rule_fwd_h 生成实验（B 组卡包，用户执行）+ MCTS 0901 两个独立 [EXP] 数据点表明，「状态常驻递推核」（每 task 大块常驻 tile + task 内长串行循环）一块一 program 胜出，persistent-core 版出现 UB 溢出。原 0.69x 反模式证据限 elementwise 小块海量任务家族，不可外推。新增规则「同算子实测 > 家族默认 > 跨家族外推」；preflight #5 同步修订。复盘全文见 `output/chunk_gated_delta_rule_fwd_h/generation_retrospective_20260915.md`。
- 2026-09-15 **新增 C-016**：`tl.zeros` 作 loop-carried `tl.dot` 累加初值在 A1/A2 两组实验上下文触发 bishengir 段错误（B3 反例存在，判上下文相关）；替代写法（全 False mask load 产零）随 v2 骨架入库；preflight 增 #9。
- 2026-09-17 **C-018 突围路径端到端实证**：grid_sample 回放——段连续 gather
  主路线 ×34.6（0.0060→0.2211 vs 原生）；片上 tl.gather 全速；段 gather 发射
  地板定量（98ns/64B 段）；gather_out_to_ub 编译失败待 IR 取证。
- 2026-09-17 **新增 C-019**：verify/benchmark 环境红线 TORCH_DEVICE_BACKEND_AUTOLOAD=0
  ——grid_sample iter_0 的 88% 伪失败实证（44/50 假失败，孤立复跑全对）；立法「先孤立
  复跑再改代码」排查顺序 + 红线应内建于 verifier skill 而非 agent 笔记。
- 2026-09-17 **新增 C-018**：离散 gather load 标量发射（SIMT 模板，[SRC] HIVMVectorOps.td:1828-1859）——恒等索引即慢 24x、全随机再 ×4.6（grid_sample P2.5 微基准 [EXP-skeleton]）；preflight 增 #11。同日 grid_sample 骨架三连修（反射早退控制流 / nearest zeros int 判界 / bicubic reflection 逐 tap 浮点折叠）——P2.5 正确性门首批战果。
- 2026-09-15 **C-016 补充**：Gdn2Fwd 生成实验 iter_0 实证产零 load 的 dtype 跟随
  占位指针（bf16 指针 → loop-carried 类型不一致 CompilationError，42/50）——各分支
  必须显式 `.to(目标 dtype)`。
- 2026-09-15 **新增 C-017**：stride 参数全量 constexpr 化触发非确定性 NaN 误编译 + UB 需求爆炸——B4 复跑 opt_iter_0 三变体 bisect 实证（标量 constexpr 安全、stride constexpr 触发）；preflight 增 #10。同日 B4 复跑（v2 卡+KB）验证 v2 修订：2.3456x（≈A1 的 2.3535x），fp32 组 0.80→1.75 全恢复。
