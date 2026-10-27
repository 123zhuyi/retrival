---
name: linear-recurrent
description: 四 线性类·状态递推算子（Gated Delta Rule / fused recurrent linear attention）的 Triton Ascend 优化经验
metadata:
  type: reference
---

# 线性类·状态递推算子优化经验（四 线性类）

> 锚定算子: `92_FusedRecurrentGatedDeltaRule`（naive/default，fla 系 fused_recurrent gated delta rule）
> 探索期: 2026-08-21 ~ 2026-08-24
> 硬件: Ascend910_9372（DAV_2201，910B 家族；与 `linear.md` 同 SKU）
> 结果: **50/50 cases 精度通过（全部 dtype × flag 组合；2026-08-24 独立复验确认）**。~~geomean 5.7527× vs torch~~ **加速比声明已撤回**：同环境配方（卡 1 + `TORCH_DEVICE_BACKEND_AUTOLOAD=0`）独立重测仅 **0.143×**——原测量中 torch 基线比所有其他口径慢 ~12 倍（case 5: 0.49ms vs 0.038~0.040ms），加速比来自被拉慢的基线而非更快的实现（详见 §5.1）
> 任务约束（本算子用户指定，同类任务可沿用）: forward() 计算路径禁止调用 torch_npu 专有接口，全部计算由 Triton kernel 完成（torch 仅限布局/视图/空分配等宿主操作）
> 头号教训: **masked 2D state store 在 Triton-Ascend 上会静默丢弃单元格**——output 全对（~9.5e-7）而 final_state 错，20/25 失败 case 源于此（详见 L1.3）

---

## §0 适用范围与算子分类

attention_index 行 12：**无 softmax**，改为状态递推或结合律重排。本卡先落**状态递推**细分；**结合律重排**细分尚无实证，待归档。

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| FusedRecurrentGatedDeltaRule | `linear-recurrent` | 门控衰减（g/gk/gv 三种 gate 组合）+ delta rule rank-1 状态修正 `S = decay∘S + k⊗((v−Sᵀk)·β)`，每 `(b, vh)` 一套 `[K,V]` fp32 状态，T 步串行递推 | 单融合 persistent kernel：状态驻留 UB + flag constexpr 特化 + device 内串行 t 循环 |
| Gdn2Fwd（94，Gated DeltaNet 2） | `linear-recurrent`（已验证 2026-09-14） | varlen（cu_seqlens）+ 可选 initial_state + kernel 内 l2norm；`S=S·exp(g_t); erase=(b·k)ᵀS; v_new=w·v−erase; S+=k⊗v_new; o=(scale·q)ᵀS`；grid=(n_seg·H·NVB,)，V>64 拆 NVB 条带 | 同构复用 §1 全部约束；**50/50 精度 + geomean 19.9138× vs torch eager 实测**（framework 绝对延迟 0.34~158ms 随规模单调，口径可信）；BV cap=64 为 UB/向量粒度平衡点（§5.2） |
| Mamba/SSM、RetNet 等状态递推变体 | `linear-recurrent`（待验证） | 状态递推同构（`S = a·S + k⊗v` 族） | 可复用 §1；首验先小规模 |
| 结合律重排（无递推） | 待归档 | `(QKᵀ)V → Q(KᵀV)` 型重排 | 本卡递推约束不适用，勿混用 |

> ⚠️ **与 `recurrent-neural-network.md`（LSTM/GRU/RNN）的分工**：那张卡的哲学是 **host-loop 单步 kernel**——其状态是 `[B,H]` 向量、无大 tile 可驻留；本卡状态是 `[BK,BV]` 矩阵（fp32 ≤32KB）可整块驻留 UB，因此**单融合 persistent kernel + device 内 t 循环**更优（5.75× 实证）。两卡「时间维串行、program 持完整状态」结论一致，分歧仅在发射粒度。
> ⚠️ **禁止套用 `flash_attention.md` 的 online softmax / KV 分块**——本类无 softmax，状态递推是唯一主链，FA 的滚动 `m/l` 约束整套不适用。

---

## Layer 1: 设计约束（硬性规则）

### L1.1 状态矩阵驻留 UB，单融合 persistent kernel（默认骨架，2026-09-14 修订）
- **默认骨架**: 1D `grid = (B*VH,)`，每 program 负责一个 `(batch, value_head)`（GQA: `h = vh // GQA_R`，q/k 用 h、v 用 vh），状态 `[BK, BV]` fp32 驻留寄存器/UB，`for t in range(T)` 在 kernel 内串行递推，一次发射完成全部计算。
- **Why**: 状态 ≤ `128*64*4B = 32KB ≪ 192KB UB`，驻留免每步读写 GM；host-loop 逐步发射被 launch 开销淹没。host-loop 仅当状态是向量、无 tile 可驻留时才是正解（RNN 卡场景）。
- **禁止**: 把时间维展平进 grid 跨 program 并行（递推真数据依赖，RAW）；循环内 `break`（与 trip-count 相关的编译约束）。
- **⚠️ 架构选择约束（非正确性约束）——大 T case 必须并列评估 chunked/WY 骨架**：serial 骨架的 per-token pipe 同步（BAR+SET/WAIT ≈56%，msprof 指令级实测）是结构性性能地板，serial 架构内调优在 94_Gdn2Fwd 上已全部证伪（§5.2 五方向 + §5.3 serial 循环重构族）。当任务 case 中存在**大 T 且小并行度**（实测交叉点：`n_seg·H ≲ 15 且 T ≳ 500`；成本模型 `chunked 2.85us·C·H + 5.5us·T/32·waves + 150us host` vs `serial 1.7us·T`）时，草图**必须并列评估 Layer 2-B chunked WY 骨架**（94_Gdn2Fwd 对照实验：chunked+dispatch 在大 case 比 serial 快 38%，全 50 case 几何平均快 ~26%）。本条不再禁止 chunk 内时间维并行——WY 表示把片内递推改写为矩阵运算，不违反 RAW 约束。
- **正确性红线（不受上述松动影响）**：逐 token serial kernel 内的时间维仍禁止跨 program 展平；L1.2-L1.9 全部为正确性/环境约束，两种骨架均强制。

### L1.2 运行时 flag 全部 `tl.constexpr` 特化（强制）
- **必须**: 所有运行时开关（gate_mode 位 1/2/4、use_gate_in_kernel、use_bias、use_initial_state、output_final_state、use_qk_l2norm、use_beta_sigmoid、allow_neg_eigval、state_v_first 等）以 constexpr 参数传入，编译期消歧。Triton 按 flag 组合各自编译缓存；单 case 内组合固定，无重复编译开销。
- **禁止**: kernel 内用运行时标量 `if` 处理 flag —— 标量分支阻碍向量化，且不同 flag 组合各有最优代码路径。

### L1.3 ⚠️ masked 2D state store 静默丢单元格（头号精度陷阱，强制）
- **症状**: `tl.store(ptr + 2D偏移, tile, mask=2D_mask)` 写 `[BK, BV]` 状态时，**被 mask 掉的单元格不写且不报错**，读回呈**非确定性脏值**（用 `torch.zeros` 预填充后错误单元格精确为 0，可据此确诊"未写"）。典型表现：output 全对（~9.5e-7）而 `final_state` 错——实测 20/25 失败 case 与 `output_final_state=True` 相关，极隐蔽。
- **必须（修复，三案均已验证）**:
  1. **1D 连续 flat 布局**（终版采用）：kernel 内按 `[BK*BV]` 写（flat index = `k*BV + v`），host 侧 `view(B,VH,BK,BV)` → 裁剪 `[:, :, :K, :V]` → 按需 `transpose`；⚠️ **V 维多条带（NVB>1）时 flat 偏移必须含 `vb*BV` 项**（94_Gdn2Fwd 实测：缺 vb 偏移则第二条带覆盖第一条带且宽度减半 shape mismatch），flat 布局 `[n, h, k, vb*BV+vl]`；
  2. **计算出的行指针逐行 store**（Variant B2）：row-wise store with computed row pointers；
  3. **padded 无 mask 全量 store**（94_Gdn2Fwd o 输出采用）：kernel 对 padded buffer（宽度 `BV*NVB`）做**无 mask 全量 store**（pad 列由状态 0 pad 列计算恒为 0，写 0 无害），host 侧 slice `[..., :V]` 还原——彻底绕开 masked store 丢单元格风险，与方案 1 同一哲学。
- **确诊手法**: 输出 buffer 预填 `torch.zeros`，错误格精确为 0 ⇒ 未写（本陷阱）；非 0 ⇒ 写错值（另有根因）。

### L1.4 边界 mask 比较 dtype（强制，2026-09-14 修订）
- **store mask 必须 int32 比较**（94_Gdn2Fwd iter_0 实测证伪原 fp32 建议）：`m_k = offs_k < K`（int32 索引张量 < int32 标量）。
  fp32 导出的 mask 在并发下触发**非确定性 store 污染**——实测 o 输出部分单元格随机丢 store、同代码重跑结果不一致，改 int32 后 50/50 全过且可复现。
- 原建议（`m_k = offs_k.to(tl.float32) < K.to(tl.float32)` 防标量降级）**仅在纯计算 mask（不进 store）且无并发污染症状时可考虑**；一旦 mask 进 `tl.store`，正确性优先用 int32。两案冲突时以本条为准。

### L1.5 数值细节与 fp32 累加（强制）
- 状态、门控衰减、归约全程 fp32；输入读入即 `.to(tl.float32)`，仅最终 store cast 回目标 dtype。
- softplus 对齐 `F.softplus(threshold=20)` 的稳定形式：`tl.where(x > 20.0, x, tl.log(1 + tl.exp(x)))`。
- `F.normalize(p=2, eps=1e-6)` 对齐：`x / max(||x||₂, 1e-6)`（`tl.maximum` 防 0 除）。

### L1.6 kernel 返回前 host 侧同步（多 case 场景强制）
- host 侧 `torch.npu.synchronize()`：verify/benchmark 连续跑多 case 时，防止上一 case 的 kernel 异步写回与下一 case 的内存分配/复用竞争。单 case 无害，可常开。

### L1.7 可选输出指针不可传 None（强制）
- `output_final_state=False` 时 `ht_ptr` 仍需实参：传 1 元素占位张量（Triton 不接受 None 作指针），并用 constexpr 分支（L1.2）保证 kernel 不访问该指针。

### L1.8 BK=64 触发 PlanMemory 编译失败（UB 红线）
- 状态 + 多输入 tile 的 UB 需求超限时 bishengir 直接 `PlanMemory Failed`；需对 K 维分块（子 tile 递推）或降 BK。`[BK,BV]` 驻留与 BK 上限的边界本探索未完整扫描——遇编译失败先做 UB 记账再选 tile，不要盲目重试。
- **实测边界数据点（94_Gdn2Fwd，910b1，UB=192KB=1572864 bits）**：`[128,128]` fp32 状态（64KB）+ 每 token 6 个 `[128]` 输入向量 + broadcast 中间量 `[128,128]`，编译报 `ub overflow, requires 2123776 bits`——多 buffer 与中间量把名义 64KB 放大到 265KB。**`[128,64]` fp32（32KB）编译通过且实测为性能最优点**（BV cap=64 是 UB 容量与向量粒度的平衡点；cap=32 向量宽度不足劣化 24%）。

### L1.9 ⚠️ verify/benchmark 必须设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（环境红线，强制）
- **症状**: 未设该变量时 torch 双 npu 后端注册冲突，报 `RuntimeError: Two accelerators cannot be used at the same time in PyTorch: npu and npu`。若发生在 impl/框架加载或运行期，会产生**大面积伪失败**——包括看似真实的数值错位（3/616 违例、1e3 量级脏值）、NaN、乃至整组 case 报错，极易被误判为代码 bug（2026-08-24 实例：同一份 50/50 正确的代码被误判为 41/50 失败）。
- **必须**: 所有 verify/benchmark 运行环境显式 `export TORCH_DEVICE_BACKEND_AUTOLOAD=0`（本模板 L1.6 的 `torch.npu.synchronize()` 路径同样依赖它）；配套 `LD_LIBRARY_PATH` 指向实际使用的 torch/torch_npu 构建。
- **教训**: 排查"数值失败"前先确认环境变量，再做代码归因；跨会话采纳/复核代码时双方必须对齐环境配方。

---

## Layer 2: 算法骨架

### 骨架 A（默认）：单融合 persistent 递推 kernel

```
grid = (B * VH,)                         # 每 program: (b, vh)；GQA: h = vh // GQA_R
状态 S[BK, BV] fp32 = zeros / load(h0)   # 驻留 UB；h0 按 state_v_first 决定加载索引交换
for t in range(T):                       # 串行数据依赖，禁 break
    1) 门控衰减（flag 组合）:
         S *= exp(-exp(A_log)·softplus(g + dt_bias))     # gate_mode & 1 且 use_gate_in_kernel
       或 S *= exp(g)                                     # 标量 gate 路径
       或 S *= exp(gk)[:,None] · exp(gv)[None,:]          # 逐维 gate（&2 / &4）
    2) 读 q/k/v/beta（GQA: q/k 用 h，v/beta 用 vh）
       可选: l2norm(q), l2norm(k), sigmoid(beta)（allow_neg_eigval 时 ×2）
    3) delta rule rank-1 修正:
         kv    = Sᵀ k                    # [BV] = sum_k k[k]·S[k,v]
         resid = (v − kv) · β
         S    += k ⊗ resid
    4) 读出: o_t = (q·scale)ᵀ S          # [BV]，store 时 cast 回目标 dtype
末状态: S 按 1D flat [BK*BV] 连续写回（L1.3），host 侧 reshape/transpose 还原
```

host 侧要点：非 contiguous 输入防御性 `.contiguous()`；flag 全部转 bool 以 constexpr 传入（L1.2）；可选输出用占位张量（L1.7）；返回前 `synchronize()`（L1.6）；每 program 常量（如 `A_log[vh]`、`dt_bias[vh]`）在 t 循环外预取（LICM）。

### 骨架 B（大 T 备选，94_Gdn2Fwd 对照实验验证 2026-09-13）：fla 风格 chunked WY 管线 + cost-model dispatch

> 来源：94_Gdn2Fwd-naive-default 10 轮探索（crossover 实测：大 case 比骨架 A 快 38%，50 case 几何平均快 ~26%）。
> 触发条件（L1.1 交叉点）：`n_seg·H ≲ 15 且 T ≳ 500`；成本模型 `chunked ≈ 2.85us·C·H + 5.5us·(T/32)·waves + 150us host` vs `serial ≈ 1.7us·T`。

```
host: 预转置 k,g 到 [B,H,K,T] 连续布局（消除 kernel 内转置寻址 gather——列步长 H·K 的
      [BK,16] load 实测 ~22us/次, 50× 行 load, 病态）; cost-model 或实测覆写表按 case 选路
K1 (并行, grid=(C·H,)):  每 chunk BT=32
    log 域 cumsum（⚠️ dot-cumsum 前缀和必须 gT @ tl.trans(LT)，右乘下三角是后缀和 → exp 溢出 NaN）
    → Gram 矩阵 → Neumann 求逆 [16,16] 严格下三角 (I+L)^{-1} = (I−L)(I+L²)(I+L⁴)(I+L⁸)（4 dot 精确）
    → 写 fp32 工作区（⚠️ dot 馈入的中间量必须 fp32，bf16 工作区 MERE 8e-3~1.2e-2 过不了官方 verify）
K2 (串行, grid=(n_seg·H·NVB,)):  每 chunk 4 大 dot（状态传播 + 片内 WY 读出），
    fp32 状态 [BK,BV] 驻留 UB 跨 chunk 传递；融合 K2 优于分裂（state-pass/o-apply 分裂实测回退 8-13%）
dispatch: serial / chunked 按成本模型 host 选路；一刀切 chunked 在大 H/多 segment case 回退 0.45-0.70×
```

骨架 B 已验证约束（94_Gdn2Fwd R1-R10 实证）：
- **BT=32**：BT=64 全展开 K1（60+ dot）编译 >400s + aicore trap；单 kernel dot 总数控制在 ~20
- **单 kernel 内嵌 WY 不可行**：Gram+Neumann 全在串行 program 内时 70 个小 dot 依赖链 ~85us/chunk，比 serial 慢 3×——WY 必须配 kernel 分裂
- **mid-row 参考点指数分解**：防 e^{±320} 溢出（指数限制 e^{±40}）
- 已证伪勿重复：K2 分裂、仿射/M 折叠（减少 dot 数 ≠ 缩短关键路径，同步主导下收益为零）、head-pair 交错（编译器已流水化）、serial 双 token 复合递推（+5.9% 回退）
- simulator 实证：chunked 路径同步仍 55-70%（后端固有）、MMAD ≤19.6%（非 dot 硬件极限）

---

## Layer 3: 关键代码片段（技巧可参考，变量名/结构须重设计）

**L1.3 修复：状态 1D flat 写回 + host 侧还原**
```python
# kernel 内：[BK,BV] 状态按 1D 连续布局写，元素 [k,v] 的 flat index = k*BV + v
flat_offs = offs_k[:, None] * BV + offs_v[None, :]
tl.store(ht_ptr + (b * VH + vh) * BK * BV + flat_offs, s_tile, mask=bound_mask)
```
```python
# host 侧：view -> 裁剪 -> 按目标布局转置
final = buf.view(B, VH, BK, BV)[:, :, :K, :V]
if state_v_first:
    final = final.transpose(-2, -1).contiguous()   # 目标 [B, VH, V, K]
```

**L1.2 constexpr flag 特化（调用侧）**
```python
kernel[(B * VH,)](
    q, k, v, ..., output, final_state,
    T, K, V, VH, H, GQA_R, scale,
    BK=next_pow2(K), BV=next_pow2(V),
    HAS_G=bool(gate_mode & 1), HAS_GK=bool(gate_mode & 2), HAS_GV=bool(gate_mode & 4),
    USE_INITIAL_STATE=bool(...), OUTPUT_FINAL_STATE=bool(...), ...,
)
```

**L1.4 fp32 边界 mask / L1.5 稳定 softplus**
```python
m_k = offs_k.to(tl.float32) < K.to(tl.float32)
sp  = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))   # F.softplus(threshold=20)
```

---

## 性能可达性

- ~~5.7527× vs torch（目标 5.0×，已达标）~~ **撤回**。2026-08-24 独立复测（同 impl md5 `5454f179`、同参考 sha `321a37eb`、同环境配方：卡 1 + `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + 系统 torch 栈，warmup=5/repeats=50）：**geomean 0.143×**（50/50 精度通过）——比 torch 基线慢约 7 倍，未达任何加速目标。
- 原声明不可复现的根因：其测量的 framework 基线逐 case 慢 ~3-12×（case 5 报 0.4897ms，其余所有口径均为 0.038~0.040ms）；impl 耗时复测一致（case 1: 0.0079 vs 0.0070ms）。**加速比数字必须用 framework 绝对延迟做口径交叉校验**（见 §5.1）。
- 单融合 persistent kernel（状态驻留 UB + flag constexpr 特化 + device 内串行 t 循环）仍是本类形态的正确**架构方向**——精度 50/50 可达、结构合理；但架构正确 ≠ 性能达标：本实现 kernel 体（逐 token 标量 load + rank-1 更新）在大多数 case 上仍慢于优化过的 torch eager 参考实现，性能优化空间仍大（对照：同一 benchmark 上另一会话 13 轮迭代的最优实现为 1.3231×）。
- host 侧清理类微调（输入 contiguous 化精简等）无额外收益（见 §5），不值得做。
- **当前最佳（2026-09-15，94_Gdn2Fwd 50 case，910b1，vs torch eager）**：geomean **38.2648×**（双骨架 + double-buffer + dispatch 放宽 + 精确 buffer + chunk 朝向直读消 host 转置 + BV=128 精准复活，见 §5.7），反超并行窗口 hybrid 实现 ×1.1332（50 case 中 49 个 ≥1.0）；chunk 组（13 case）×1.33+，serial 组持平偏优。长 T chunk（T≥1500，BT=32 K2 串行）为本骨架固有优势区（vs hybrid ×1.42-1.49）。

---

## §4 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| masked 2D store 写状态后 final_state 错、output 却对 | mask 掉的单元格静默不写，读回为脏值 | L1.3 flat 1D store + host reshape；zeros 预填确诊 |
| BK=64 编译失败 | UB 需求超限 → PlanMemory Failed | L1.8 先 UB 记账，K 维分块或降 BK |
| fp32 比较导出 store mask | 并发下非确定性 store 污染（丢 store、重跑不一致） | L1.4（2026-09-14 修订）store mask 必须 int32 比较；fp32 比较仅限纯计算 mask |
| softplus 大输入溢出 / 与参考不齐 | 未对齐 F.softplus threshold | L1.5 稳定形式 |
| 多 case 连续 verify 结果随机错 | kernel 异步写回与下 case 分配竞争 | L1.6 synchronize |
| 可选输出传 None | Triton 不接受 None 指针 | L1.7 占位张量 + constexpr 分支 |
| 时间维进 grid 并行 | 递推 RAW | L1.1 串行 t 循环 |
| forward 调 torch_npu 融合接口 | 本类任务用户约束 + AST 门禁 | 全部计算入 kernel |
| state_v_first 初态/末态错位 | 布局 `[V,K]` vs `[K,V]` 索引交换遗漏 | 加载与写回两侧都按 flag 交换（Layer 2 骨架注） |
| "Two accelerators: npu and npu" 大面积伪失败（含数值脏值/NaN） | 未设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，双 npu 后端注册冲突 | L1.9 环境红线；归因代码前先查环境 |
| speedup 声明虚高（如 5.75×） | framework 基线测量被拉慢（频率锁定异常/卡忙/计时段污染），impl 实际更慢 | framework 绝对延迟与历史口径交叉校验（§5.1）；同口径复测后再声明 |

---

## §5 已验证无效/排除的尝试（别重复）

### §5.1 ⚠️ 5.7527× 加速比声明撤回（2026-08-24，测量口径事故）
- **现象**: 本卡曾声明 geomean 5.7527×（复验 5.8651×）达标。独立复测（同代码/同参考/同环境配方/warmup=5/repeats=50）仅 **0.143×**，且 impl 耗时与原测量一致——分歧全在 framework 基线（原测量慢 ~3-12×）。
- **疑似根因**: 复验所用新版 benchmark.py 的 NPU 频率锁定（lock_npu_frequency）在 framework 计时段失效或降频；或 framework 计时段卡忙/环境残留（如 `ASCEND_LAUNCH_BLOCKING=1` 调试变量泄漏会把逐算子 launch 变同步，对多算子 torch 参考惩罚最大）。
- **规避**: ① 任何 speedup 声明前，将 per-case framework 绝对延迟与历史/跨会话口径对照（本算子 case 1/5 应为 ~0.012/~0.040ms 量级）；② benchmark 环境与调试环境严格分离，`unset ASCEND_LAUNCH_BLOCKING`；③ 频率锁定失败时按 error 处理而非 warn。

- **标量头基准指针的状态写回**（Variant A）：无效。
- **指针行索引写法的 row-wise store 首版**（Variant B 原始版）：编译错误；修正为"计算出的行指针"后可用（即 L1.3 方案 2）。
- **host 侧清理迭代**（输入布局预处理精简，迭代 #17）：无 speedup 提升，保留基线。
- **调试提示**: kernel launch 卡顿时用 `ASCEND_LAUNCH_BLOCKING=1` 同步复现定位；torch inductor compile_worker 的 traceback 属探测噪音，可忽略。

### §5.2 94_Gdn2Fwd Phase 4 全扫描证伪方向（2026-09-14，50 case，基线 geomean 19.9138× vs torch eager）
以下五个方向在 94_Gdn2Fwd（varlen 逐 token 递推，T 普遍 4~285）上实测**均无提升或劣化**，同类算子别重复：
- **手动软件流水化预取**（循环内预取 t+1 的 load 再算本拍）：19.2498 vs 19.9138——load 数翻倍、寄存器压力上升，Triton-Ascend 后端已有自己的 load 调度。**【失效边界（2026-09-15 按三要素补注）】证伪的写法 = 每迭代同时 load t 与 t+1 两套、load 数翻倍；验证环境 = A 侧纯 serial 骨架、基线 19.9138。未证伪变体：cur/nxt 寄存器轮换式（load 数不变）——§5.5 已实测其快 1.54×，本条不得作为方向级禁令引用。**
- **persistent kernel 单波次 dispatch**（grid=核数，核内任务循环）：19.8046 vs 19.9138——短 T 递推的多波次发射开销占比极小，任务循环反增标量分解开销；grid/核数 ≤10 波次不在 KKB grid-sizing 收益区间。
- **l2norm `tl.rsqrt` 替换 `1.0/tl.sqrt(sum)`**（IR 分析指导）：19.4754 vs 19.9138，公共 case impl 延迟比 1.003——IR 上 vdiv+S↔V 同步链看似很重，实测主链 `[BK,BV]` tile 的 vmul_2d/reduce_sum_ra 才是耗时主体，标量链收益被淹没。**IR 推断的候选必须实测验证，不能凭 IR 指令数下关键路径结论**。
- **Block Size Scaling**：BV cap=128 → 编译期 UB overflow（L1.8）；cap=32 → 15.1109 劣化 24%。B0=64 即最优点。
- **Pass 合并 / 净指令数不变的代数改写**（simulator 诊断指导，decay 吸入 erase 权重 + decay/update 合并为 `S=S*e^g+k⊗v_new` 单 FMA 遍历）：19.6790 vs 19.9138，公共 48 case impl 延迟比 0.9838——msprof 指令级采集揭示 AIV 流水下**每条向量指令附 ~712cyc pipe_barrier（普通指令 ~150cyc），BAR+SET/WAIT 占 56% 为结构性瓶颈**；编译器未把合并表达式融合为更少指令，而权重吸收引入的 +1 条 1D 乘及其 BAR 反超省下的 2D 指令。**per-token 递推的 pipe 同步是性能地板（~16 条向量指令/token × 712cyc ≈ 总量 53%），只有真正减少循环内向量指令条数的改写才可能有收益，净条数不变或微增的代数等价改写一律无效**（与 rsqrt 失败同一根因）。

### §5.3 94_Gdn2Fwd 双骨架任务 Phase 4 证伪补充（2026-09-14，基线 geomean 22.2468×，双骨架 serial+chunked WY+dispatch）
以下结论来自 template 修订后第三次 94_Gdn2Fwd 任务（iter_0 一次通过 50/50，Phase 4 四轮扫描 scan_complete）：
- **点 28 CV 融合（al. 手写 scope / Batch 流水线 / T0-T5）在 910b1 + triton 3.2.0 平台不可用**：门槛测试证实 `triton.language.extra.ascend` 下**仅有 libdevice**，无 `language`(al)/`bl` 子模块——cv-fusion-pingpong.md 警示的 910B 系编译崩溃风险在本环境更早暴露（import 即 ModuleNotFoundError）。同平台任务对点 28 直接判不适用，无需重复门槛测试。
- **点 11 Load 重排序对 chunked K2 无效**：将 K2 chunk 循环内仅依赖 cc 的 wDK/k_t/wED load 提前至循环开头发射，verify 50/50 但 geomean 22.0006 vs 基线 22.2468（impl 0.3533 vs 0.3537ms 持平，差值为 fw 侧 profiling 噪声）——K2 关键路径在 Cube dot 链（wW@st→uT→{wM@uT, kd@uT}），编译器本就有 load 调度空间。
- **K1 三 exp 分解（e_hi/e_lo/eG）数值必需，禁合并**：e_hi = eG/exp(g_ref) 的代数化简在强衰减下 G≈-640 时 eG 与 exp(g_ref) 同时下溢为 0 → 0/0=NaN。IR 上看似冗余的 3 个 [64,32] exp 是 mid-row 参考点防溢出的固有成本。
- **IR 手工提取 artifact 认知**：TRITON_DEBUG 环境下 chunked 路径 kernel 执行崩溃（SUSPECT REMOTE ERROR 507057，普通环境正常）；手动 bishengir-compile ttadapter.mlir 时 K1 停在 ConvertHFusionToHIVM（arange 的 linalg.generic 未合法化）、K2 停在 MarkRealCoreType 后 core dump——均为手工流程缺前置 lowering pass 的 artifact，非代码问题；dump 出的中后段 IR 结构信息仍可分析。
- **双骨架代码的 Autotune/BlockSizeScaling 不适用**：BK=next_pow2(K)/BV=min(BVP,64) 系 host 按 shape 公式分档（按规则不适用 autotune）；BT=32 上限实证（BT=64 编译 trap）；BK/BV/BT/BVP/NVB 多维分块参数并存使 block_size_scaling 触发条件不满足。
- **结论性认知**：双骨架 22.2468× 下 Phase 4 全扫描（点 11/28/31 IR/32/33）无提升——骨架 A 触 serial 结构性地板、骨架 B 的 dot 链与工作区设计已在生成时吸收了 KB 经验（layout 预转置/Neumann/物化），指令级优化空间在架构设计阶段已被前置消化。**当 template+KB 经验完整时，Phase 4 的主要价值是验证收敛而非提升**。

### §5.4 94_Gdn2Fwd 定点迭代补充（2026-09-14 第三轮，opt_iter_4 点 20 simulator 驱动）
- **点 20 深化（K1 槽位原位替换 wEG/wDK→wQT/wKD，K2 循环体纯点化）**：verify 50/50 一次过（数值逐位一致论证成立），**chunked 5 case impl 全部真实下降 6.8~10.2%**（case43 -9.5%、case48 -10.2%），但 geomean 22.1725 < 基线 22.2468 被 4.4 闸门判无提升回退。
- **闸门口径认知（重要）**：基线复用规则下 geomean 判定 = 优化侧当次 fw 重测 vs Phase 3 历史 fw 测量。当优化只改动少数 case 路径（本例 5/50）时，45 个代码不变 case 的 s_i 差值全是 fw 跨 session 噪声（实测 ±1.3%），与期望增益同量级——**此类"局部路径优化"在 geomean 闸门下天然难过**，不代表机制无效。处置：机制有效性以 impl 延迟口径单独记录（产出代码保留在 opt_iter_4/ 供人工候选），流程判定仍严格按 geomean 闸门。
- **方法论闭环验证**：并行窗口 simulator 诊断（Cube 空等 Vector 26.9%）落到本架构 = K2 循环内残留 Vector 活（q/k 重 load + 双 L2N + 逐元素乘 + trans），点 20 物化解耦后 chunked impl 实降——证实"静态扫描判不出动态瓶颈、simulator 强制命中有效"的路径成立；同时证实本架构（K1/K2 预解耦）与并行窗口 B 线（fused 单核）的瓶颈构成不同（其"v-tile 重复 A/Tinv"在本架构不存在）。

### §5.5 94_Gdn2Fwd 同口径对比实验（2026-09-14，hybrid WY 两核版 vs 本仓双骨架基线）
并行窗口 hybrid 实现（`_gdn2_wy_kernel` 全并行预处理 + `_gdn2_state_kernel` 串行纯 dot + `_gdn2_full_kernel` token 路径 + `avg_seg>=192 && N*H*nvt<=20` 分派）原样进本仓 verify/benchmark 流程（同机 device4、同 50 case、warmup5/repeats10）：

- **结论**：hybrid verify 50/50 一次通过，geomean **33.0782 vs 本仓基线 22.2468（×1.487）**。对方 "~32" 可复现，差距为真实架构差距；fw 跨 session 噪声 ×0.953 反而压低对方，纯 impl 侧差距 ×1.561。**跨会话实现的性能对比必须同机同流程复跑，凭各自报告数字对比无意义。**
- **归因**（impl 几何比 对方/我们）：① 双方 serial/token 35 case = 0.6486（对方快 1.54×，geomean 贡献 ×1.354，**最大杠杆**）；② 路由差 10 case（本仓 serial → 对方 chunk）= 0.4788（2.09×，贡献 ×1.159）；③ 共同 chunk 5 case = 1.0549（本仓快 5%，贡献 ×0.995）。
- **⚠️ 修正 §5.2「手动流水化证伪」**：证伪的是 **load 数翻倍版**（循环内同时 load t 与 t+1 两套）。hybrid `_gdn2_full_kernel` 的 **cur/nxt 寄存器轮换 double-buffer**（prologue 载 t=0，循环内只 load t+1 并用上一拍寄存器值计算，每迭代 load 数不变）在 35 个 token case 上一致快 1.54×（impl 比 0.52~0.87 紧致分布），机制 = 隐藏逐 token 递推暴露的 MTE2 load 延迟。同类递推算子 token 路径应采用轮换式预取。
- **Dispatch 修订候选**：`avg_seg>=192 && n_seg*H*num_v_tiles<=20` 比本仓 `n_seg*H<=15 && T>=500` 多覆盖 10 个 case（7/17/20/27/32/35/40/41/45/50），chunk 路径在这 10 个 case 全部更快（2.09× 几何）；含 varlen 时用 avg_seg=T/n_seg 而非 T。
- **BT 按 T 分档候选**：共同 chunk case 上本仓 BT=32 fused 在 T≥1500 快 16~17%（case30/38/43），对方 BT=16 两核在 T~600-800 快 13%（case15/48）——BT=16 适合中 T、BT=32 适合长 T，可按 T≈1000 分档。
- **数值前提校正**：hybrid 的 `exp(lL-gc)`/`exp(gc)`/`exp(lL)` 在 g≤0（logsigmoid 门控）前提下全部 ≤1，本 50 case 集（含 T=3388）实证无溢出、精度 50/50——此前"正指数物化悬崖"的表述仅在 g 可正时不成立，g≤0 数据下 chunk-end 参考点与 mid-row 参考点等价安全。
- **合成预测**：token 轮换预取（×1.354）+ 分派放宽（×1.159）同时吸收进本仓双骨架，预测 geomean ≈ 34.9，且长 T chunk（BT=32）本仓已优于对方。

### §5.6 外来实现分析：TileLang/AscendC 纯 token 版（2026-09-15，ascendc-kernelgen-data br_perf_expert 31_Gdn2Fwd）
- **结论**：纯 token 递推单骨架（TileLang 主实现 + AscendC 转译，无 chunk/无 dispatch/无预取），8 个重叠 case 全部慢于本仓 Triton 基线 1.2~3.9×（口径差异：msprof.quick warmup3/repeats1 vs 本仓 warmup5/repeats10，差距量级远超口径噪声）。**无可移植性能技术**；其 final_report 自认"vs chunked WY 有 2-4× 差距"。
- **数据点 1（排除方向）**：l2norm 硬件 rsqrt 精度在 bf16 atol/rtol=1e-2 下实测 maxdiff≤0.002，**无需 Newton refine**（验证环境：910B2C + TileLang/AscendC 链，迁移到 Triton tl.rsqrt 时同为硬件指令，结论可参考）。
- **数据点 2（双向印证）**：其 V1 fixed-core（24 核 persistent）判不适用，与本仓 §5.2 persistent 实测无效一致。
- **方向确认（第 3 个独立来源）**：其 PERF_DESIGN V2 假设"逐 token CopyIn 与 Compute 串行、预取可重叠"——与 hybrid 轮换预取实测 ×1.354 同向（该实现自身未实施 V2）。
- **平台特有陷阱（TileLang/AscendC 链，Triton 层不存在，勿回流为本仓约束）**：GM 标量读落 AIC 核、2D BufferRegion 507015 崩溃、ReduceSum RA 内层轴 32B 对齐需 host V_pad、count=1 向量指令链被静默丢弃、CopyOut 需 PIPE_ALL。

### §5.7 94_Gdn2Fwd 合成迭代实测（2026-09-15，opt5-opt10 吸收 hybrid 技术，最终 geomean 38.2648 反超 hybrid ×1.1332）
验证环境：910b1、本仓双骨架（serial persistent + chunked WY K1/K2 + dispatch）、基线 Phase 3 geomean 22.2468、50 case（37 serial + 13 chunk）、device 4 warmup5/repeats10。五轮迭代全部 verify 50/50 + benchmark 实测判定，hybrid 四项技术判定完毕：

- **✅ serial 六张量寄存器轮换 double-buffer（opt5，×1.363 全量 / ×1.42 serial 组）**：q/k/g/b/v/w cur/nxt 轮换，prologue 载 t=0，循环内只 load t+1；**原始 dtype load（MTE2 减半）+ 用前转 fp32**；尾迭代 `tl.minimum(t+1, tlen-1)` clamp。机制：隐藏逐 token 递推暴露的 MTE2 load 延迟（§5.5 修正后首次落地本仓骨架）。
- **✅ dispatch 并集放宽（opt5 同轮）**：`((n_seg*H<=15) and (T>=500)) or ((avg_seg>=192) and (n_seg*H*NVB<=20))`，新转 chunk 组 ×1.88；并集（非替换）保证原路由不回退。
- **❌→✅ BV=128 + BT=16 捆绑移植（opt6 判负 27.9754 → opt10 解包复活 38.2648）**：opt6 三要素——失败写法 = hybrid 的 V_TILE=128/BT=16 启发式原样搬入本仓 K1/K2；失效边界 = 本仓 grid=n_seg·H·NVB 骨架下 BV 翻倍 → NVB 2→1 → grid 减半（hybrid grid 不依赖 V_TILE，**结构耦合参数禁止跨骨架盲目移植**）；未证伪变体 = hybrid 自身 3-kernel 骨架下的 BV=128。**opt10 逐 case 解包发现判负结论掩盖子组全胜**：BV=128 在 serial 路由 grid≥24 组（6/29/34/44/46）实际 ×1.15-1.30，grid 8/12（case4/12）退 0.70-0.74（starvation），27/40 的回落实为 BT=16 所致（NVB=1 触发 chunk 重路由 × BT=16 回退，两变量互相污染）。**精准复活公式**：`BV = 128 if (BVP>=128 and BK<=64) else min(BVP,64)` + serial 路由 grid<20 守卫回落 BV=64（BT=32 不动）——目标组 ×1.1561，27/40 chunk 重路由 ×1.39/×1.49，原 6 落后 case 全反超。**判负轮必须逐 case 解包：整体 geomean 判负可能掩盖子组全胜；两变量捆绑时各自作用需单变量隔离归因。**
- **✅ 精确 buffer + masked store（opt7，×1.0707 全量 / ×1.10 serial 组）**：o 精确 [B,T,H,V] + masked 1D store、fs 精确 [n,H,K,V] + masked 2D store，消 2 个 host contiguous copy kernel。**L1.3 丢格陷阱只约束 padded buffer；精确 buffer 全格落在 mask=true 区（mask 仅裁 tile 越界），50/50 实证安全**。小 T case 收益最大（case3 ×1.467）。
- **❌ BT=16 单变量隔离（opt8，判负 31.1067）三要素**：失败写法 = 本仓 chunked WY 双 kernel（K1 C_max 并行 + K2 chunk 串行 4-dot）下 BT=16；失效边界 = 全部 8 个 T<1024 chunk case 回退 0.59-0.72，T≥1024 case 保持 BT=32 纹丝不动（隔离干净）；机制 = K1 Neumann 五因子 dot 链工作量 ∝ BT²（BT=16 使 C_max/grid 翻倍、每 program 工作量降 4 倍、launch 开销翻倍）+ K2 串行步数翻倍依赖链拉长；未证伪变体 = hybrid 3-kernel 骨架下 BT=16。**BT=32 对本骨架最优，BT=16 方向封死（除非换骨架）**；附带结论：opt6 的 chunk 回退主因是 BT=16 而非 BV=128。
- **✅ chunk 消 host 转置 + K1 朝向直读 + tl.cumsum（opt9，×1.1374 全量 / ×1.4549 chunk 组，chunk 组反超 hybrid ×1.3371）**：瓶颈定位（SpyKernel Skipper 墙钟分解）= host 侧 6 个 permute+contiguous 预转置 + 6 个 workspace alloc 占 chunk 路径墙钟 **52-57%**（330-350us，case 7/17/20），K2 本来就赢 hybrid（75 vs 85-97us）。改法 = K1/K2 改 [BT,BK] 朝向直读原布局 [B,T,H,K]/[B,T,H,V]（K/V 连续合并访存，hybrid wy 同构）整体删 host 转置；`tl.cumsum(g, axis=0)` 替代右乘上三角 dot 做前缀和（消 [BK,BT]@[BT,BT] dot）；workspace 布局/BT=32/K2 串行结构不变，store 前 `tl.trans()` 对齐。机制认知：**host 侧"预转置消 kernel gather"对大 kernel 路径是伪优化——转置本身是 copy kernel 且 ∝ B·T·H·(4K+2V)，kernel 改朝向直读（连续维合并访存）即可整体消除**；小 T chunk case 相对收益最大（case7 ×1.9622）。
- **✅ BV=128 精准复活（opt10，×1.0196 全量 / ×1.1561 目标组 13 case，geomean 38.2648 对 hybrid ×1.1332）**：opt6 判负数据逐 case 解包（见上条 ❌→✅）后，以 `BV = 128 if (BVP>=128 and BK<=64)` + serial 路由 `n_seg·H<20` 守卫回落 + BT=32 基座重测。三重收益：serial 保持组（6/29/34/44/46）×1.18-1.30；27/40 因 NVB=1 命中 dispatch 第二条款重路由进 chunk 优势路径 ×1.39/×1.49；守卫组（4/12）零回退。小回退 case10 ×0.95（grid 60 波次量化）净收益远覆盖。**dispatch 公式中的 NVB 项是路由敏感参数：BV 选择会改变 case 的路由归属，评估 BV 必须联动检查 dispatch 重路由效应。**
- **逐 case 参照（opt10/hybrid）**：50 case 中 49 个 ≥1.0，唯一落后 case10 = 0.965（T=21,H=60，波次量化）；chunk 13 case ×1.17-1.49，原 BV=128 命中组 6 case 全部反超（27→1.267 / 40→1.311 最优）。
- **方法论（吸收外来实现的完整闭环）**：同机同流程复跑建立逐 case 性能地图（§5.5）→ 静态阅读只能列候选不能排序（实测优先级倒挂：token 路径 ×1.354 > 分派 ×1.159 > chunk 解耦 ×1.01）→ 逐项隔离实测（opt6 捆绑判负 → opt8 单变量隔离定位主因）→ 结构耦合参数识别（grid 依赖 V_TILE 与否决定可移植性）→ 墙钟分解定位剩余差距（host 转置 52-57%）→ 同构改造吸收（朝向直读 + cumsum）。

