# 15_LinformerAttention Triton-Ascend 性能优化报告

> 状态：**已收官**（生产版 v15b 已定型；2026-08-14 起工件部分丢失，见 §6）。
> 本文记录经 `verify` 50/50 + `benchmark` 50/50 双闸门的数据。

## 1. 达成值

| 项 | 值 |
|---|---|
| 基线 speedup_vs_torch（R1 Phase 3 初版） | **0.089**（50/50，fw 0.0757 ms / impl 0.8512 ms） |
| 编排器线终点（R9，round-9 代码） | **1.4800**（50/50，impl 0.0526 ms） |
| **生产版 v15b（历史口径）** | **2.0119–2.0315**（50/50，逐 case 几何平均；30/50 case >2.0×） |
| **生产版 v15b（2026-08-14 更新后 benchmark 口径）** | **2.3747**（50/50，32/50 case >2.0×） |
| 目标 2.2x | **未达成**（2.0 硬关已破，2.2 差 ~9%） |
| 测量口径 | `triton-op-verifier` 的 `verify.py`（50/50 强制）+ `benchmark.py`（warmup=5 / repeats=50；几何平均）；阈值未改，未用 `--verify_not_required` |
| 硬件 | **Ascend 910B3**（Ascend910_9372，**20 AIC**；逻辑 dev0 = 物理 NPU2；人工线 verify 用 NPU4） |
| 工具链 | triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0 / bishengir-compile（pip 自带） |
| 交付版本 | `0810/15/15_LinformerAttention.py`（v15b，md5 5bf252db85d80d6551a06c0f17f4369f）⚠️ 8/14 后盘上已丢失，代码存于 `v15b_work/verify/__pycache__/*.pyc` |

### 口径说明（两套数字不要混用）

| 口径 | framework 基线 | v15b speedup |
|---|---|---|
| 历史（原 benchmark.py） | ~0.078 ms | **2.0119–2.0315** |
| 2026-08-14（更新后 benchmark.py） | ~0.23 ms（torch_npu 侧慢 ~3x） | **2.3747** |

kernel 一字未改、50/50 逐位一致；framework 基线变慢系统性抬高 speedup。
已排除频率因素（warmup=50 仅 2 次漂移事件，NPU 稳定 1800MHz，framework 仍 ~0.23ms）。
频率锁在本机不可用（uid 1008 非root，dsmi rc=46），以高 warmup + 漂移计数替代。

## 2. 生产版架构（v15b，5 个 kernel）

```
q2d = x2 @ W_q^T                                  # _proj_matmul_kernel（通用 GEMM）
q   = q2d.view(b,s,H,hd).permute(0,2,1,3)          # strided view，不拷贝
if dtype == fp32:                                   # 低秩重排路径（结合律，仅 fp32 安全）
    x_red = x^T @ W_kr^T  → [d, P]（TINY）
    kr    = W_k @ x_red   → [H*hd, P]
else:                                               # 低精度走逐位对齐路径
    k2d  = x2 @ W_k^T ; kr = _kv_reduce_strided(k2d, W_kr)
    # ^ 专用 kernel 直读 k2d[B*S,H*hd]，转置折进寻址（BLOCK_P≤64, BLOCK_HD≤128, BLOCK_S≤256）
attn[b,s,H,hd] = softmax(q@kr^T·√hd^-1)@vr          # _linformer_attn_kernel_{hoist,strided}
                                                     # NUM_S>1: (b,H)-major 连续分区 + Kr/Vr 外提
                                                     # 输出直接按 [B,S,H,hd] stride 落盘
out = attn.reshape(M,d) @ W_o^T                     # reshape 免费（内存连续）
```

要点：
- forward 内零 PyTorch 计算算子（AST 退化预检 PASS），仅 empty/reshape/permute + kernel 发射；
- 权重以 `_make_linear_weight` 复现 nn.Linear kaiming 初始化（seed 42，逐位一致），
  构造在 `__init__`/cache 路径，不进计时；
- `_NUM_CORES` 运行期读 `num_aicore`=20。

## 3. 瓶颈画像（终态）

impl 时间构成（profiler op-share）：

| 段 | 占比 | 状态 |
|---|---|---|
| attention kernel（hoist+strided） | ~48% | cube-bound；R9 后 task 分区已优 |
| 投影 GEMM（_proj_matmul） | ~38% | **triton vs CANN aclnnMatmul 2.2x 库级 gap**（探针 0.059 vs 0.027 ms），AST 禁 torch.matmul ⇒ 钉死 |
| kv_reduce（strided） | ~14% | v15b 已 −22.5%（BLOCK_S 256） |

framework 侧（torch 参考）约一半算子时间是纯搬运（Transpose 77.0+46.4+5.1 µs）——
impl 的主要赢面来自消除同型拷贝（v4 输出直写、v6 转置折进寻址）。
逐 case 看：大 D fp32 case（对 aclnnMatmul 无还手之力）结构性 0.43–0.95×，
是 2.2× 目标剩余缺口的主要来源；小/中 case 普遍 >2×。

## 4. 有效改动（按贡献排序，全量）

| 改动 | 增益 | 依据 |
|---|---|---|
| attention tile 化 + `tl.dot`（R2） | 0.089 → 0.96（**10.8x**） | 基线单 token 标量点积，画像 attention 占绝对主导 |
| 架构重写两 kernel（R5 Phase3） | 1.24 → 1.3673 | 通用 GEMM + 融合 attention 分离 |
| 低秩重排 k/v（fp32 档，v3） | 并入 1.494 | 中间量 [M,d]→[d,P]，砍 2 大 GEMM + 2 拷贝；34/50 失败教训 → dtype 分派 |
| attn 输出 [B,S,H,hd] 直写（v4） | +16% → 1.7393 | 输出侧 `.contiguous()` 消除 |
| k/v 转置折进 strided-reduce（v6） | +14.5% → 1.9921 | 低精度路径 2 次 InplaceCopy 消除 |
| kv_reduce BLOCK_S 64→256（v15b） | +1% → **2.03 破关**；kernel −22.5% | UB 账 128KB<192KB；K 迭代减半 + MAC 加深 |
| attention (b,H)-major 分区 + Kr/Vr 外提（R8/R9） | 1.4061 → 1.4800 | 大 S case Kr/Vr 流量 O(B·H·NUM_S)→O(B·H) |
| GEMM Swizzle2D + dtype 分档 BK + num_cores 运行期（R5） | 1.3673 → 1.409 | L2 复用 + 与物理核数对齐 |
| attention block 自适应（R2 iter1） | 0.96 → 1.05 | hd/P/S 分布 20–80/2–53/3–897 |

## 5. 已排除方向（证据见迭代日志，此处列主条目）

host-concat / launch-fused 投影（0.973 / 0.9387，精度漂移）；
通用 GEMM 吃 4-D strided 布局（大 hd case 慢 1.86–11.78x）；
reduce 融进 attention（0.8882）；W^T swapped-stride（0.0525，预转置回升 26x）；
多输出 GEMM 融合（v8 −30% / v12 −24%，UB 逼 tile 减半）；
attention fp16 dot（v13 −5%）；online softmax 在小 P 形态（R3 0.5865）；
BK 守卫收紧（1.3541）；GROUP_M≠4（噪声）；num_stages（无效）；
host wall-clock 计时定位瓶颈（方法论错误，结论全错）。

## 6. ⚠️ 工件状态（2026-08-14 之后）

- **丢失**：v3–v16 全部 `.py` / `perf_result.json` / `report.md`、生产
  `15_LinformerAttention.py`、`repro_v15b` 复现数据（仅存 anchor + .pyc）。
- **幸存**：
  - 编排器线全部工件 `triton_ascend_output/15_LinformerAttention/opt-round-1..10/`
    （161 个 JSON，R 系列数字可验证）；
  - `v15b_work/verify/__pycache__/15_LinformerAttention_triton_optimized.cpython-311.pyc`
    （v15b 代码；BLOCK_S=256 已用字节码反汇编核实，dispatch 结构核实为
    fp32→restructured / else→original 双路径）；
  - `.baseline_anchor.json` 冻结的 golden 哈希 `93a543c…`（torch 参考从未被改动，成绩可信）；
  - 速度数字（2.0119–2.0315 / 2.3747）存于会话 transcript。
- ⚠️ 盘上 `triton_ascend_output` 的最高分 1.48× 是**编排器线**，**不是生产成绩**；
  扫描磁盘时勿以它下结论。

## 7. 当前判断

2.2× 未达的缺口在**投影 GEMM 的库级 gap**（triton GEMM vs aclnnMatmul 2.2x）+
大 D fp32 case 的结构性劣势。在 AST 禁 torch.matmul、UB 192KB、fp32 累加逐位约束下，
三条 cube 段（投影 38% / attention 48% / kv_reduce 14%）均已到达各自的 triton 级上限；
进一步收益需要更快的 cube 路径，当前工具链不存在。
破 2.0× 的全部 4 刀都是**非计算开销消除**（重排/直写/折寻址/tile 上限），
这一条对同族算子（"多投影 + 小 P attention"形态）可直接迁移。
