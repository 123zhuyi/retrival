---
name: cv_attn_agg
description: CV-Attn-聚合分发类算子（S2Attention / CoTAttention / DoubleAttention——空间或通道变换聚合 + 轻量 attention 分发 + 聚合回空间的 Triton Ascend 优化经验。以"聚合-分发"结构轴为核心，按算子分章节，含通用经验 + 各算子专属杠杆/陷阱/证伪。GEMM/卷积/softmax/online softmax/UB 预算等共享机制继承自 flash_attention.md / convolution.md / normalization.md，本文件只写 delta。
metadata:
  type: reference
---

# CV-Attn-聚合分发类算子优化经验

> **本类算子**：CV 空间/通道特征经**变换聚合**（卷积 / 通道扩展 / spatial shift / mean 归约）→ **轻量 attention 分发**（逐通道 / 逐分支 / 逐空间位置的加权 softmax）→ **聚合回空间**。共同点：attention 的 KV 维**极小或极特殊**（k=3 分支、逐通道权重、softmax-over-hw），**不是**序列 attention——FA 那套 KV 循环握手基本不适用。
>
> ⚔️ **与相邻 template 的分工**：
> - `flash_attention.md`：**序列 KV 分块 + online softmax**——本类的 attention 若含 KV 循环（如 42 的 att2 空间维），那部分继承 FA；但主干是"聚合-分发"结构。
> - `convolution.md`：卷积本身（若用 CANN conv 作 golden 复刻，精度链见 [[triton-doubleattention-precision-recipe]]）。
> - `normalization.md`：softmax / mean 的归约精度。
> - 本文件**只写 delta**：**"聚合-分发"结构**带来的额外约束、杠杆与陷阱。
>
> **判别顺序**：按 §0.1 确认是否为 CV-Attn-聚合分发。满足 → 用本文件。
>
> 📌 **类别名 `cv_attn_agg`（CV-Attention-Aggregate 聚合分发）**：启动传 `--op-category cv_attn_agg` 命中本卡。

---

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

满足以下任意一条 → 用本文件（CV-Attn-聚合分发类）：

1. **通道/空间变换聚合**：输入经 `Linear` / `conv1x1` / spatial shift / mean 先聚合成低维中间量（`[B, C]` 或 `[B, C', H, W]`）；
2. **轻量 attention 分发**：基于聚合量做 **softmax 加权**，KV 维**极小**（`k=3` 分支、逐通道、`softmax-over-hw` 空间维）——**不是**序列 attention；
3. **聚合回空间**：attention 输出再经 `Linear`/卷积/加权求和还原到原空间形状；
4. 算子是 CV 特征（`[B,C,H,W]` / `[B,N,C]`）而非 token 序列，attention 用于**特征门控/通道注意力**而非 token 间关系。

不满足 → 序列 attention 走 `flash_attention.md`；纯卷积走 `convolution.md`。

### §0.2 ★ 聚合-分发结构轴（**本类核心路由，Phase 2 第一步必须定位**）

本类的优化杠杆**完全取决于聚合-分发结构的形态**。先定位再选杠杆：

| 形态 | 含义 | 典型 | 核心杠杆 |
|------|------|------|---------|
| **(a) 通道扩展-分支分发** | 通道先扩 3x，拆 3 分支做不同变换，加权合并 | S2Attention (59) | 投影 GEMM（C→3C）+ 分支 kernel 融合 + 低精度 RNE 复刻 |
| **(b) 空间-通道双 attention** | 空间维与通道维各做一次加权聚合 | CoTAttention (42) | grouped conv + 两次轻量 attention + 归约精度 |
| **(c) conv 特征-双 softmax 汇聚** | conv1x1 变换 → 2 次 softmax(hw) + 2 次 bmm 汇聚 | DoubleAttention (47) | conv 精度链（HF32）+ softmax(hw) + 2 次 bmm |

### §0.3 形态补充三问（在 FA §0.2 四问之上追加）

| # | 问题 | 影响 |
|---|------|------|
| **C-Q1** | attention 的 KV 维多大？`k=3` 固定分支 / 逐通道 / softmax-over-hw 空间维？ | 决定 attention 是否值得独立 kernel（k=3 点乘不可矩阵化）；大 KV 才走 FA 继承 |
| **C-Q2** | 聚合量是标量（`[B,C]` mean）还是空间（`[B,C',H,W]`）？ | 决定归约 kernel 与后续分发 kernel 的衔接 |
| **C-Q3** | 低精度（fp16/bf16）中间量有多少处舍入点？golden 对中间舍入敏感吗？ | 决定 RNE 位运算复刻的范围（本类最大精度坑） |

---

## §1 通用经验（跨成员，首次生成必须遵守）

> 本类**特有**、FA/conv/norm 未覆盖。FA F1~F6 / L1.1~L1.14、conv 卷积、norm softmax 不重复。

### C1 ★★ 低精度中间量必须 RNE 位运算复刻，`.to(dtype).to(fp32)` round-trip 会被编译器消除

- **必须** fp16/bf16 中间量在**每个舍入点**用位运算复刻 RNE（round-half-to-even），`triton-ascend` 编译器会把 `.to(dt).to(fp32)` 当成 no-op 消掉。
  - bf16：`(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000`
  - fp16：13-bit 尾数 RNE + 指数 rebias + FTZ/inf 处理；大 tile 用 4-op 变体避免 i32 比较标量降级
- **禁止** 依赖硬件原生乘（NPU fp16/bf16 mul 非完全 RNE）：fp16 原生乘 ~1.6% 元素 1-ulp 偏差，bf16 原生乘 MERE ~2.3e-2 超阈值 3 倍。
- **Why（59 实锤）**：fp16 用硬件原生乘 MER 方差放大，bf16 原生乘直接超阈 3 倍；改 RNE 位运算后 38/50 → 50/50。
- **判别信号**：低精度 case 单独失败、`MERE` 卡在阈值附近 ⇒ 查 RNE 复刻，不是 tiling。

### C2 ★ 归约（mean/sum）用 Kahan 补偿，匹配 torch reduce 树

- **必须** 大 N 归约（如 S2Attention 的 `mean(xa, dims=(1,2))`）用 **Kahan 补偿求和**，对齐 torch reduce 树。
- **Why（59 实锤）**：顺序累加 fp32 尘埃 ~2e-5，在 fp16 舍入边界产生翻转；Kahan 后 ~3e-8，消除边界失败。
- **判别信号**：某 case 的 mean 结果差 ~1e-5 量级、fp16 输出边界翻转 ⇒ 上 Kahan。

### C3 ★ 激活函数近似必须匹配 NPU 底层实现，不是数学精确式

- **必须** GeLU 用 **tanh 近似**（`0.5*x*(1+tanh(√(2/π)*(x+0.044715x³)))`）匹配 NPU `F.gelu`；erf 精确式偏差 4.7e-4。
- **禁止** 用数学精确式（erf）当"更准"——golden 是 NPU 的 tanh 近似，用精确式反而偏。
- **Why（59 实锤）**：erf 式与 NPU F.gelu 偏差 4.7e-4，fp16 边界超阈。
- **判别信号**：GeLU 相关 case 单独失败且偏差 ~5e-4 ⇒ 查激活近似。

### C4 ★ 聚合量小的中间 kernel（`[B,C]`/`[B,3,C]` 级）几乎免费，别浪费轮次

- **必须** 识别"聚合量级"的 kernel（如 S2Attention 的 mlp3/mlp4/softmax，只算 `[B,C]` 或 `[B,3,C]`）——合计 <6% impl，**再优化无收益**。
- **禁止** 对这类 kernel 投入优化轮次。
- **Why（59 实锤）**：mlp3+mlp4+softmax 合计 <80µs，占 impl <6%，压缩无空间。

### C5 ★ 主成本通常是投影 GEMM（C→3C / C→C），别误判为 attention

- **必须** profiling 分解后把投影 GEMM 当主攻对象（S2Attention：MLP1/MLP2 占 56% 主成本）。
- **禁止** 把优化重心放在 attention（本类 attention 极轻）。
- **Why（59 实锤）**：4 个 Linear 是主成本；attention 只有 k=3 分支，算力极小。

---

## §2 精度闸门（CV-Attn 特有）

- **低精度 RNE 复刻**是精度全过的前提（C1）——fp16/bf16 每个中间舍入点都要位运算复刻，`cast round-trip` 不可靠。
- **归约精度**：mean/sum 大 N 归约用 Kahan（C2）。
- **激活近似**：GeLU 用 tanh 近似匹配 NPU（C3）。
- **标杆种子确定性**：若标杆用 `torch.manual_seed(hash((channels, device, dtype)))`，`hash(torch.float16)` 是**对象地址派生**，跨进程随 ASLR 漂移 → 权重每进程重抽 → fp16 边际 case MERE 随机翻转（"抽签"）。**必须**锁步改为 `(zlib.crc32(repr(key).encode()) + k) & 0xFFFFFFFF`，k 经预扫选定使边际 case 有余量。判别：同一代码跨 verify 进程结果随机变化 ⇒ 查种子。
- **fp32 锁定判据**（继承 FA L1.4）：参考全程 fp32 + 量级超 dot 安全区时，全 fp32 中间 buffer 是唯一正确解。

---

## §3 各算子章节

### §3.1 59_S2Attention（Spatial Shift + Split Attention，`通道扩展-分支分发`）

**生效层**：§0.2-(a) 通道扩展-分支分发
**形态**：`x[B,C,H,W]` → permute → MLP1(C→3C) → 3 分支 spatial shift → `mean([B,C])` → MLP3+GeLU → MLP4+softmax → apply → MLP2 → permute back。4 Linear + 2 permute + shift + mean/softmax/apply。
**结果**：**50/50 精度 + 2.2702x**（NPU4，2026-08-18，framework 3.1446ms / impl 1.3852ms）。种子确定性修复后可稳定复现（历史 0810/0811 的 50/50 是抽签幸存者，仅 0.2962x）。

#### 制胜技术（按贡献排序）

**T1 复用已验证多 kernel 流水线**（+50/50 精度基础）
- S2Attention 是 8-kernel 纯 Triton 流水线（permute_in / gemm_bias / shift / mean / mlp3_gelu / mlp4_softmax / apply / permute_out）。每个 kernel 单一职责、精度可控。
- **判据**：同算子历史资产存在且经完整精度迭代修复时，直接复用（0811 终版）——比从 0 重写稳。

**T2 标杆种子确定性修复（crc32 + 偏移）**（49/50 → 50/50，可复现）
- 标杆 `torch.manual_seed(hash((channels, device, dtype)))` → `hash(torch.float16)` 地址派生漂移。
- 锁步改 `(zlib.crc32(repr(key).encode()) + 4)`，k=4 经预扫选定（case 5/23/42 余量最大）。
- 详见 §2 精度闸门。

**T3 低精度 RNE 位运算复刻**（38/50 → 50/50）
- fp16/bf16 中间量每处舍入点用 `_rne_lowp`/`_rne_fp16_fast` 位运算复刻（C1）。fp16 大 tile 用 4-op 变体避免标量降级。

**T4 Kahan 补偿 mean**（消除 fp16 边界翻转）
- `mean(xa, dims=(1,2))` 用 Kahan，fp32 尘埃 2e-5 → 3e-8（C2）。

**T5 tanh 近似 GeLU**（匹配 NPU F.gelu）
- 用 `0.5*x*(1+tanh(...))`，非 erf 精确式（C3）。

#### 性能瓶颈画像（target 2.27x 下的主成本）

| Kernel | 占比 | 说明 |
|---|---|---|
| `_gemm_bias_kernel` | **56%** | MLP1(C→3C) + MLP2(C→C)，发射 2 次 |
| `_shift_kernel` | 11% | spatial shift 融合 |
| `_permute_in/out_kernel` | 9.8%/7.9% | 两次 permute |
| `_mean_kernel` | 7.6% | Kahan 归约 |
| `_apply_kernel` | 6.9% | att × xa 三分支加权求和 |
| `_mlp3/4/softmax` | <0.3% | 聚合量极小，几乎免费 |

#### 关键约束清单（首次生成必守）
1. **投影 GEMM 是主成本**（C5），tile 按 (M,N,K) 分档；fp32 大 tile 会 UB overflow，fp32 档用更小 BLOCK_K。
2. **低精度 RNE 复刻**（C1）+ **Kahan mean**（C2）+ **tanh GeLU**（C3）三件套，缺一精度不全。
3. **permute / shift / apply 用专用 Triton kernel**，禁 PyTorch 退化（`aclnnInplaceCopy_Transpose` 等 AiCore 搬运开销）。
4. **小聚合量 kernel**（mlp3/4/softmax）别优化（C4）。
5. **标杆种子**若用 `hash()` 必须先修（§2）。

#### 已证伪方向（省下你的时间）

| 方向 | 结果 | 来源 |
|------|------|------|
| 压缩 mlp3/mlp4/softmax | **没空间**（合计 <6%） | C4 |
| apply 合并成单 dot | **不可行**（k=3 非矩阵乘形态） | C-Q1 |
| mean 普通顺序累加 | **精度崩**（fp32 尘埃 ~2e-5 边界翻转） | C2 |
| GeLU erf 精确式 | **偏差 4.7e-4**（NPU 是 tanh 近似） | C3 |
| apply 用硬件原生乘 | **MER 方差放大**（NPU fp16 mul 非完全 RNE） | C1 |
| 标杆种子用 PYTHONHASHSEED | **无效**（dtype hash 是地址派生，管不住） | §2 |

#### 泛化启示（给本类其他算子）
- **"通道扩展-分支分发"算子**（先扩通道再分支加权）：主成本在投影 GEMM，attention 极轻——别在 attention 上浪费轮次（C4/C5）。
- **低精度算子**：RNE 复刻是精度全过的前提，`cast round-trip` 不可靠（C1）。
- **标杆种子抽签**：hash() 作种子是常见坑，先查再谈精度。

---

## §4 待后续算子填充（骨架占位）

> 预期成员的**形态 + 预期杠杆**锚点，算子达标后按 §3.1 体例回填。

### §4.1 42_CoTAttention（空间-通道双 attention，`空间-通道双 attention`）— 已验证

**形态**：Coordinate Attention（grouped conv key embed → 2 次 attention + bias/mean）。
**结果**：**49/49 精度 + 1.5147x**（NPU15，2026-08-19，fusion-search 7 轮，框架 0.222ms / impl 最差 case 0.289ms）。
**制胜技术**（按贡献排序）：

- **T1 卷积路由 CANN**（最大杠杆，0.69→1.51x）：k1 分组卷积（Triton fused_singlerow 4.4 vs CANN 64 TFLOP/s）、value_embed conv1x1（8.99→2.87ms）、att1 cat-conv（3.9-7.9→1.3-3.2ms）全部换 `F.conv2d`。参考本身就是 torch conv，CANN 是其最优后端。
- **T2 小 case 整算子路由 torch**：Triton 6-kernel 启动 ~0.1ms/kernel 在小 case 不可摊销。判别规则 `n==1 ∨ b*c*n<1.2M ∨ (bf16∧b*c*n<6M)`，不误伤 fp32 大 case（case 9/10 保留 1.6x）。
- **T3 Triton attention 融合链**：w2_avg 预归约 kk 维（att_w2 `[C*kk,mid]`→`[C,mid]` 预缩放 1/kk，FLOPs 少 9x）+ softmax_fused 单 kernel online softmax + residual。
- **T4 case 38 特例**：bf16 640×223×191 的 att1 用 CANN 反而慢（4.73 vs Triton 3.98ms）→ 保留 Triton att1。

**剩余瓶颈**：Triton ieee GEMM（att_mean/softmax）算力上限；CANN matmul 实测不更快（已验证）。2.2x 目标因 CANN conv 已极优而不可达。
**已证伪**：纯 Triton fused 卷积（4.4 vs 64 TFLOP/s）；CANN att_mean matmul（不更快）。

### §4.2 47_DoubleAttention（conv 特征-双 softmax 汇聚，`conv 特征-双 softmax 汇聚`）— TODO

**形态**：4 conv1x1 + 2 次 softmax(hw) + 2 次 bmm。**注意**：当前实现用 CANN conv 作 golden 复刻（`ascend_triton_forward` 里 conv 走 `F.conv2d`，softmax/bmm 走 Triton）。
**已知**：记忆 [[triton-doubleattention-precision-recipe]]：fp32 conv=HF32(11-bit)→emulate RNE bit-ops on rank-1 operands；fp16 softmax 做 fp16-domain sub 再 fp32 exp/sum/div。
**待回填**：conv HF32 精度链、softmax(hw) + bmm 结构、性能结果。

---

## §5 证伪方向（跨成员，持续积累）

> 继承 FA §5.2 / conv / norm 全表；以下为本类**新增**。

| 方向 | 结果 | 来源 |
|------|------|------|
| 低精度中间量用 `.to(dtype).to(fp32)` round-trip | **编译器消除为 no-op**，精度 fail | C1（59） |
| 依赖硬件原生乘复刻低精度 | fp16 MER 方差放大 / bf16 超阈 3 倍 | C1（59） |
| mean 普通顺序累加 | **精度崩**（Kahan 才对齐 reduce 树） | C2（59） |
| GeLU 用 erf 精确式 | **偏差 4.7e-4**（NPU 是 tanh 近似） | C3（59） |
| 优化聚合量小的 kernel | **no-op**（合计 <6%） | C4（59） |
| 标杆种子用 `hash()` | **抽签**（dtype hash 地址派生漂移） | §2（59） |

---

## §6 测量口径与时效

- **几何平均**：每 case 权重相同（FA §7.0）。本类主成本在投影 GEMM，优化要盯最慢的 shape 档。
- **口径**：benchmark.py 旧口径按 kernel 名 groupby（同名多次发射只计一次）；S2Attention 的 `_gemm_bias_kernel` 发射 2 次（MLP1+MLP2）旧口径漏计 1 次，新口径会如实计入。跨口径比较须注明。
- **历史结论随编译器/CANN 失效**（FA §7.3）：所有数字标注来源算子与日期，判据/机理优先于绝对数字。RNE 位运算语义随编译器版本可能变，重测前重新标定。
