# 双流水判定参考（判定逻辑固化）

> 本文档固化双流水（多缓冲/软件流水）判定的全部口径，来源：bishengir 源码
> （`AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/MarkMultiBuffer.cpp`）、
> 官方文档（`AscendNPU-IR/docs .../CVPipeline/CVPipelining.md`、`PlanMemory.md`）与
> GDN r39 实测验证。分析时**无需再查文档/源码**，按本文档口径执行即可。

## 1. 三个判定维度（运行时，裁决性证据）

| 维度 | 定义 | 指标 | 成功线 |
|---|---|---|---|
| **c 内双流水** | cube 子核内 搬运×CUBE 计算重叠 | max(C_MTE1_MAC, C_MTE2_MAC) | 重叠 ≥30% |
| **v 内双流水** | vector 子核内 搬运×VECTOR 计算重叠 | V_MTE2_VEC | 重叠 ≥30% |
| **cv 双流水** | Cube 与 Vector 两子核并行（CVPipelining 目标） | CV_CUBE_VEC（跨子核活跃交集/min） | 重叠 ≥30% |

- 重叠 = 活跃区间交集 / min(活跃A, 活跃B)；活跃区间 = X 事件**剔除 WAIT_FLAG\*/BAR** 后归并
- 纯 cube / 纯 vector kernel 对应维度记"—"（不适用）
- 未成功位置 = 该侧子核等待驻留 Top1 行（断流点）+ 主导单缓冲变量（结构佐证）

## 2. IR 证据链（编译器侧）

### 2.1 mark 决策树（MarkMultiBuffer.cpp，五条件全过才打标）

```
0. 编译旋钮 enable-auto-multibuffer 关 → 全不标
1. 挂钩类型：必须是 ND2NZ / Fixpipe / Load / Store 四类搬运 op 的片上侧
2. 回溯根必须 isAllocLikeOp（独立 alloc）——原地物化(materialize_in_destination
   借用视图)在此出局
3. alloc 必须在循环内（getParentLoop）
4. 祖先循环必须全是 scf.for / scf.while（scf.forall/parallel 破坏
   per-iteration slot rotation 不变量，源码注释明确）
5. 该缓冲的地址空间未被溢出回退整组关闭（disableMultiBufferOnL1/L0C/UB）
全过 → mark(深度默认 2)
```

### 2.2 双地址形态（结构落地的直接证据）

`bishengir-compile --mlir-print-debuginfo` 重放后，多缓冲缓冲表达为：

```mlir
pointer_cast(%cA_i64, %cB_i64) : memref<shape, #hivm.address_space<space>>
    loc("file.py":LINE:col)          ← 同一 op 两个偏移 + 源码行
```

- 双参数且 offA≠offB = 编译器乒乓对（仅在 multibuffer=True 重放出现，False 口径 0 个）
- 变量级判据：**≥2 个互不相同地址**才算双缓冲成立（同址双参数/单地址不算）
- 源码行 → 变量名：取该行左值（`var = ...`）

### 2.3 枝 5 识别（空间组关闭，实测最常见的"应双未双"根因）

**判据**：本 kernel 有标记但全落其它空间，而未标大缓冲在本空间 →
该空间被溢出回退整组关闭。

实测案例（GDN r39）：state/prep_is/prep_g（UB 紧张）标记 100% 落 cbuf、
UB 零标记——⚠ 名单（S 32KB/usol 16KB/k_raw/g2_t/v_t）全是 ub 空间；
对照组 gate（UB 宽松）ub 标记 ×140 正常。⇒ UB 组被
`disableMultiBufferOnUB` 关闭（源码注释："surgically turn off the
multi-buffer of just the overflowing address space"）。
**含义**：这类未双缓冲是容量物理约束，需先腾出该空间，改写法绕不过。

### 2.4 未双缓冲变量的原因分类（dual_buffer_vars 固化规则）

| 分类 | 判据 | 呈现 |
|---|---|---|
| **⚠ 应双未双** | 搬运/物化上下文 + ≥8KB + 访问≥5 次，未获标（枝 5 或循环类型） | 重点节 |
| 无需：小缓冲 | ≤4KB | 紧凑+原因 |
| 无需：循环外不变量 | 小搬运缓冲（归一系数类） | 紧凑+原因 |
| 无需：计算型/标量 | 无搬运 op 挂钩（枝 1），值不可预取 | 紧凑+原因 |
| 无需：原地物化（小） | 根非独立 alloc（枝 2）且不热不大 | 紧凑+原因 |

## 3. pass 级验证（diff_pass.sh）

- 判定：pass 执行前后 IR **字节级**比较（滤警告行后）——相同=未成功，有改写=成功
- **重要 pass 名**：`hivm-mark-multi-buffer`（真正写 `hivm.multi_buffer=N` 注解的
  pass，信息量大）；`hivm-enable-multi-buffer` 不改 IR、无信息量，勿再测
- **已知坑**：编译器 warning 行（deprecated flag/target triple 提示）会混入
  dump 造成假阳性——diff_pass.sh 已内置过滤（曾致 gate/out 的 cv-pipelining
  假"成功"）；注解级小区块（6 行）pass 判定仍有非确定性，多缓冲结论以
  2.2 双地址证据为准
- cv-pipelining 对 mix kernel 的真实改写：循环/work-item 重排（实测 4~363 行）

## 4. 运行时乒乓签名（trace）

- **pc 级**（严格口径）：同一 pc 的同一 WAIT_FLAG 指令跨迭代交替使用 ≥2 个
  FLAG id = 一个同步点多份缓冲。注意：仅"同一依赖边上有多个 flag"不成立
  （可能是多条数据流各自的**单缓冲**同步点，实测 gate 边级口径全为假阳性）
- 在途并发：pipe 原始忙碌/归并占用 >1 = 传输自身多笔在飞

## 5. 证据优先级与已知局限

1. 结构是否落地：2.2 双地址（硬证据）> 2.1/2.3 mark 分析 > pass 字节 diff
2. 是否用上：运行时重叠（第 1 节）> pc 级乒乓
3. 局限：行级归因会把同一源码行多个张量合并（shape 取主导缓冲）；
   编译器 loc 偶落 docstring 区（如 L29），出现时人工识别
4. 相关旋钮（cache `<kernel>.json` 真值）：`multibuffer` / `num_stages` /
   `set_workspace_multibuffer`（文档写默认 2，**安装版实测默认 4，以
   `bishengir-compile --help` 为准**）

## 5.5 本机实测硬件参数锚点（Ascend910_9372）

> roofline / 带宽判定一律用实测值，不用推算。实测环境：本文档所附 GDN 分析机。

| 参数 | 实测值 | 来源 |
|---|---|---|
| SOC 型号 | Ascend910_9372 | `detect_soc.py`（torch_npu get_device_name） |
| HBM 可持续带宽 | **1.11–1.17 TB/s** | copy/add 微基准（大张量 copy_ 读+写 / add 读2+写1） |
| HBM 容量 / 频率 | 64 GB / 1600 MHz | npu-smi |
| cube / vector 核数 | 20 / 40 | CANN platform_config + 实测 block_dim 互证 |
| cube 频率 | 1800 MHz | msprof OpBasicInfo |
| UB / L2 | 192 KB/核 / 192 MB | CANN platform_config |

（HBM 理论峰值 ≈ 1.6 TB/s 由 频率×位宽 推算，仅参考；实测可持续 ≈ 70% 理论。）

## 5.6 各搬运路径理论带宽推导（单位 B/Cycle，9372）

口径：`[AICoreMemoryRates]` 各字段单位 = **B/Cycle**（用户口径确认，与实测对拍一致）。
换算：**带宽 = rate × AIC频率(1800MHz) × 核数聚合**（cube 侧路径 ×20 cube 核，UB 侧 ×40 vector 核）。

| 路径 | B/cycle | 每核 GB/s | 聚合 TB/s |
|---|---|---|---|
| L1→L0A | 512 | 921.6 | 18.43 |
| L1→L0B | 256 | 460.8 | 9.22 |
| L0C→UB（fixpipe 出） | 256 | 460.8 | 9.22 |
| L1→UB | 128 | 230.4 | 9.22 |
| UB→L1 | 128 | 230.4 | 9.22 |
| UB→L2 | 64 | 115.2 | 4.61 |
| UB→DDR | 64 | 115.2 | 4.61 |
| L2 读 | 110 | 198.0 | 3.96 |
| L2 写 | 86 | 154.8 | 3.10 |
| **DDR(HBM)** | **32** | **57.6** | **1.15** |

**换算公式**：带宽(GB/s) = 速率(B/cycle) × AIC频率(1.8e9) × 核数（cube 侧×20 / ub 侧×40）。
例 DDR：32×1.8×20=1152 GB/s。频率域取 AIC 1800MHz 是由 DDR 行对实测对拍反推验证的；
L2 读/写两行作用域（是否共享聚合）最不确定；vector 18.4 TFLOPS 的 lane 口径无实测对拍（标假设）。

**交叉验证**：DDR 推导 1.15 TB/s ≈ 微基准实测 1.11–1.17 TB/s（偏差 <4%）——证明
"每核 @AIC 1800MHz"读法成立，DDR 行可直接当 roofline 的 HBM 带宽上限用。

**算力峰值（推导）**：cube 稠密 fp16/bf16 = 2×16×16×16×20×1.8G = **294.9 TFLOPS**；
cube INT8 = 2×16×32×16×20×1.8G = **589.8 TOPS**（DtypeMKN）；vector fp32 ≈ 18.4 TFLOPS（假设 vec_calc_size=128 lane × FMA）。

## 6. 落地脚本

| 脚本 | 职责 |
|---|---|
| `dual_pipeline_check.py` | 三维度重叠 + cv 指标 + pc 级乒乓 + 旋钮/pass/地址层汇合 |
| `dual_buffer_vars.py` | 变量级双地址判据 + 原因分类（含枝 5 自动识别） |
| `diff_pass.sh` | pass 前后字节 diff（滤警告） |
| `run_full_analysis.py` | 端到端编排，报告 §4 = 汇总表 + 每 kernel 三层明细 |
