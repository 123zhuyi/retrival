---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: "4.2"
title: Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）
description: 四条 Triton-Ascend 后端契约：mask 保持 int32 比较（fp32 导出的 mask 触发非确定 store 污染）；cos_row 类常量折叠产生 stride-0 memref 会触发 MLIR 断言崩溃（用 constexpr 分支出 1D 路径规避）；kernel 内禁用 % 运算符（写 a-(a//b)*b）；masked 2D state store 静默丢单元格（状态写回用 flat 1D / 逐行 / padded 全量三选一）。
tags: [mask_discipline, int32_mask, store_pollution, stride0_memref, mlir_crash, modulo_operator, masked_store, compile_trap, backend_bug]
confidence: verified
status: verified
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

## Triton-Ascend mask 纪律与编译期陷阱（来源：ApplyRotaryPosEmb P4 三组 + bisect 定位，2026-09-08）

【优化维度】：移植｜后端契约
【泛化层级】：普遍性（Triton-Ascend 当前版本；失效边界：后端修复后复测）

### 1. mask 保持 int32 比较——fp32 mask 触发非确定 store 污染

<原则：所有 mask 由整数比较直接产生（`i < N`、`nm[:, None] & mi[None, :]`），**不要从 fp32 运算结果导出 mask**（如 `(x != 0.0)` 再当 mask 用）。bisect 实证：fp32 导出的 mask 在并发下触发非确定性 store 污染（相邻行错值），grid=1 不复现。>

**已知实例**：
- ApplyRotaryPosEmb P4（2026-09-08，bisect 定位，出处即本卡来源）
- Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1）独立复现：fp32 导出 store mask 触发 o 输出随机丢 store、重跑不一致，改 int32 比较（`offs_k < K`）后 50/50 可复现（2026-09-14；证据：op_94_94_Gdn2Fwd_20260914_1219_8124/report.md + `.claude/template/linear-recurrent.md` L1.4）

### 2. 常量折叠产生 stride-0 memref → MLIR 断言崩溃

<原则：当某运行参数在特定 case 下被常量折叠为 0 并参与地址 stride（如 cos_row 恒 0 → `0*RD` 的 memref），MLIR 报 `strides must not be zero` 断言崩溃（R=1 边界 case 触发）。规避：用 `tl.constexpr` 分支（如 SINGLE_COS）把该情形拆成 1D 无零 stride 的独立加载路径，让 0 不出现在 stride 位置。>

```python
if SINGLE_COS:  # tl.constexpr；cos_row 恒 0 的情形
    cs = tl.load(cos_ptr + cols, mask=rope_c, other=0.0)[None, :]
else:
    crow = idx0 * CS0 + idx1 * CS1
    cs = tl.load(cos_ptr + crow[:, None] * RD + cols[None, :], ...)
```

### 3. kernel 内禁用 % 运算符

<原则：Triton-Ascend kernel 内 `a % b` 有编译陷阱（与常量折叠/强度削减交互），一律改写为 `a - (a // b) * b`（非负值域）。这也与「行解码标量化」纪律协同：行号解码放标量循环（见 [strided-2/interleave 访存契约](../memory-sram/triton-ascend-strided2-interleave-contract.md) 缺陷①），向量 div/mod 只用于纯地址计算且避开缺陷组合。>

### 4. masked 2D state store 静默丢单元格——状态写回不用 2D mask store

<原则：`tl.store` 对 [BK,BV] 类 2D 状态块带 mask 写时，被 mask 掉的单元格**不写且不报错**，读回是非确定脏值——与 load 的 `other=` 语义不同，store 侧没有"保持原值"保证。多 case / 并发场景下表现为随机精度失败且重跑不一致，极易误判为算法 bug。状态写回三选一：(1) 1D 连续 flat 布局写回（多维拆分时 flat 偏移必须含子块维项，如 `[n,h,k,vb*BV+vl]`）；(2) 计算行指针逐行 store；(3) padded 无 mask 全量 store + host 侧 slice 还原。>

```python
# 坏实践：2D mask store 写状态，被 mask 单元格静默丢
tl.store(STATE + offs_k[:, None] * V + offs_v[None, :], s, mask=mask2d)
# 正解：flat 1D 布局写回（NVB>1 时偏移含 vb*BV 项）
tl.store(STATE + n * stride_n + h * stride_h + k * V + vb * BV + vl, s, mask=mask1d)
```

**已知实例**：Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1）——状态 [BK,BV] 写回的「头号精度陷阱」，50 case 场景下非确定复现（2026-09-14；证据：op_94_94_Gdn2Fwd_20260914_1219_8124/report.md + `.claude/template/linear-recurrent.md` L1.3）

**已知实例（边界细化）**：Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1）——final_state 写回采用 `tl.reshape` 版 flat 1D store 触发后端误编译（fs 丢格 53344/81696，非确定复现，verify 0/50）；A/B 实验证明 **2D masked store（int32 比较 mask）与逐行 store 均确定性正确**（diff 9e-8），回退 2D masked store 后 50/50 全过。边界细化：§4「2D mask store 静默丢格」陷阱与「正解：flat 1D」在本后端（triton 3.2.0 + 910b1）需附加限定——**int32 比较 mask 的 2D store 安全，而 `tl.reshape` 版 flat 1D store 才是实际误编译路径**（2026-09-16；证据：op_94_Gdn2Fwd_20260916_1619_2077/output/iter_0/log.md + iter_0/verify/verify_result.json（0/50）→ iter_1/verify/verify_result.json（50/50））。

### 5. 段粒度 gather load 忽略逐 lane mask——masked 行 OOB 地址仍取址崩溃

<原则：段连续 gather load（`addr = seg[:, None] + cc[None, :]`，64B 段）按**段粒度**取址：`mask=false` 的行仍发射段 load，地址越出 tensor 分配即 vector core exception（retCode=0x31，SUSPECT REMOTE ERROR 级联全 case 崩溃）。标量 SIMT gather 按 lane predicate 豁免（同算子 NCHW 标量路径从不崩即因此）。正解：地址**无条件夹取**界内，zeros 语义改由权重清零（inb→f32 乘权重）承载，不依赖 load mask。>

**已知实例**：grid_sample（gather_scatter，ascend910b）——zeros 模式 OOB tap（cy=600 于 600 行 tensor，段越界 76KB）致 case27 起 24 个 RuntimeError 级联（verify 3/50）；地址无条件夹取后 case27 maxdiff=0.0、恢复 7/50 预期签名（2026-09-17；证据：op_12_12_grid_sample_20260917_1453_7647/output/opt_iter_0/log.md）

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 同属优化知识，主题相关
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — mask dtype 契约交点：int32 mask 纪律与 fp32 比较转换互补
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 向量化 mask + 手动指针实践
- 相关主题: [Triton-Ascend strided-2/interleave 访存契约（三个正确性缺陷 + 寄存器重排正解）](../memory-sram/triton-ascend-strided2-interleave-contract.md) — 同为后端正确性缺陷契约卡（masked store 污染同症状）
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — Part1 mask 检查与陷阱契约互补
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — mask 检查项实例
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — boundary_check mask 标量降级
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 代数掩码写法与 mask 纪律相关

<!-- okf:related:end -->
