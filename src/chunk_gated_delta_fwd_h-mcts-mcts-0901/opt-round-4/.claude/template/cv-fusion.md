---
name: cv-fusion
description: CV 融合类算子（QuantMatmul / GroupedMatmulSwigluQuant 等低精度 GEMM + 量化/激活后处理融合；5_LightningIndexer 等检索/select 融合）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# CV 融合类算子优化经验

本文档合并该类别算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与其他类别共用的通用约束见 `tensor-transform.md` G1-G8 / `quantization.md` Q1-Q8 等，此处只写本类别特有的）
- **§2 QuantMatmul**（低精度量化 GEMM + 多后处理融合）
- **§3 GroupedMatmulSwigluQuant**（MoE 分组 GEMM + SwiGLU + 动态量化融合）
- **§4 5_LightningIndexer**（检索/select 融合：轻打分 + 重 top-K 选位，sort-compaction 选位）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**）
- **§6 精度闸门** · **§7 测量口径** · **§8 常见陷阱** · **§9 与其他 template 的差异声明**

> ⚠️ **关键区分**：本类别的核心优化哲学是 **「低精度 Cube GEMM 做主干，量化/激活/归约后处理留在 `tl.dot` epilogue 单 kernel 完成」**（GEMM 量化类）与 **「轻打分 + 重 select，选位结构优先于打分调参」**（检索/select 类，见 §4）——融合的目的是消除 HBM 往返与 kernel launch 开销，瓶颈要么落在 **Cube↔Vector 跨核协同 + 量化精度链对齐**，要么落在 **select 选位的算法结构**。生成时**禁止混用**其他类别的经验：纯 GEMM 调参去看 `matrix-compute-merged.md`；标准 `npu_weight_quant_batchmatmul` 形态（fp16 x @ int8 weight + antiquant）去看 `quantization.md §4`；本文件管的是 **「双量化操作数的通用量化 matmul」**、**「GEMM/分组GEMM + 激活 + 量化的多算子融合」** 与 **「检索/select 融合（score + top-K 选位）」**（差异判据见 §9）。

> 📌 **证据基础**：本卡当前由两个**已达标、已优化完成**的算子的完整优化轨迹支撑——
> - **QuantMatmul**（57 case，几何平均 **1.2238x**，全量精度通过；Phase 3 生成 1.0902x → Phase 4 打磨至 1.2238x）
> - **GroupedMatmulSwigluQuant**（60 case，几何平均 **1.6967x**，全量精度通过，超 target 0.8x 达 112%；Phase 3 一次成型 1.6060x → Phase 4 微调至 1.6967x）
>
> 两者算法结构不同（通用量化 matmul vs MoE FFN 融合）却**收敛到同一套瓶颈结论**（int32 累加 + 后处理留 epilogue + tile 按 shape/UB 分档 + grid 收缩核数），这是本卡可外推的根据。
>
> ⛔ **反例（0812 批跑，重生成教训）**：GroupedMatmulSwigluQuant 的骨架按本卡 §3 逐字复刻（两 kernel 链 / tile 分档 / fp16 双舍入 / 标量 store trick 全对），但 K 归约被改写成 `acc += tl.dot(out_dtype=...)` + `tl.arange` 基向量内联两处"等价改写"后，**60/60 精度全过、几何平均跌到 0.5076x**（同 case impl 0.076→13.8ms，182x），Phase 4 仅扫 BLOCK_K（→0.6596x）未能挽回。⇒ 骨架正确≠细节正确：CV1 的累加器融合形式与 CV7b 的基向量外提是**写法级硬约束**，已列 §8 静态预检指纹。
>
> ✅ **检索/select 融合类已于 0817 优化达标入卡**：5_LightningIndexer（54/54 精度，**3.9558x** vs torch / **12.02x** vs baseline），见 **§4**。该类与 §2/§3 的 GEMM 量化路径**形态不同**（"轻打分 + 重 top-K 选位"，无量化链）——两类经验**禁止互套**，分界判据见 §9。

---

> ⚠️ **启动前置检查（Agent 首次读到本文件时必须先执行，再谈任何设计/生成）**：
> 回答下面 §0.1 判别特征——本次生成算子的计算图是否命中任一条（双量化 GEMM+反量化后处理 / 分组 GEMM+激活+量化 / 多子算子融合且主干低精度 GEMM / 检索 select 融合）？
> - **命中** → 判定 `category`（`quant-matmul` / `gmm-swiglu-quant` / `topk-select` 等，见 §0 表）与 `template_path = cv-fusion.md`；
>   **必须**产出 `{工作目录}/precheck.json`：`template_exists = true`、`layer1_constraints_loaded` **逐条原文摘录**（§2 或 §3 的 L1.x，不得为空）、`loaded_via = explicit_path`；
>   §2/§3 的 Layer 1 为本次设计**硬性边界**，草图与其冲突 → 退回 designer 重设计（≤2 次），不得带冲突进入代码生成。
> - **不命中** → 说明被误触发：按 AGENTS.md 判别表改判 `category` 并转读对应文件（`quantization.md` / `matrix-compute-merged.md` / `flash_attention.md` 等），不沿用本文件。
> - **改判也拿不到 `category` 时**（`new_category`）：同样必须产出 `precheck.json` 并标记 `new_category`，本文件不适用，走 AGENTS.md 新建模板流程。
>
> **为什么必须先做这一步**：本文件是 `cv-fusion` 类别的经验合集，仅当判别特征命中时 §2/§3/§4 的 Layer 1 才是你的硬性边界。不产 precheck.json 就进入设计/生成，等于放弃本文件已验证的架构（int32 累加 / 两 kernel 链 / tile 分档 / fp16 双舍入 / grid 收缩），历史上已反复出现"模板存在却凭直觉重写 → 精度/性能双双退化"的教训（见上文反例与 §9）。

## §0 适用范围与算子分类

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| QuantMatmul | `quant-matmul` | int8/int32 量化 GEMM（双量化操作数）+ per-channel scale / per-token scale / bias / offset / clamp 多后处理，输出 int8/int32/fp16/bf16 | int32 累加 + 后处理按"是否需要 clamp"分两条 tile 路径；**clamp 是 UB 溢出唯一根因**，无 clamp 输出开大 tile |
| GroupedMatmulSwigluQuant | `gmm-swiglu-quant` | MoE per-expert 分组 GEMM（int8 weight）+ 两级 dequant + SwiGLU 激活 + 逐行动态 INT8 量化；两 kernel 链 | K1（GMM+dequant，Cube）/ K2（SwiGLU+quant，Vector）职责分离；per-expert 独立 launch + tile 按专家行数分档；fp16 双舍入 dequant 逐位对齐参考 |
| 5_LightningIndexer | `topk-select`（🔄 归位于 index-computation.md §5.5，本卡仅留指针） | score 打分（Q@Kᵀ + relu + w head-sum）极轻 + top-K 选位极重（K=2048，S2≤32768），输出升序 top-K 索引/值 | sort-compaction 选位：`npu_sort_v2` 阈值 oracle + 二次 sort 压缩替代 `tl.cumsum` 前缀链；`tl.dot` 只承担打分，select 走 sort 库 |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. 计算图含**双量化操作数**的 GEMM（int8@int8、int8@int32-weight 等），且带 per-channel/per-token scale、bias、offset、clamp 等**反量化/再量化后处理**（→ §2 `quant-matmul`）；
2. 计算图含 **MoE / 分组矩阵乘** + 激活（SwiGLU 等）+ **动态/静态量化**的融合（→ §3 `gmm-swiglu-quant`）；
3. 计算图把 **≥2 类子算子**（GEMM/Cube · elementwise · reduce · quant/dequant · 激活）融合进**少量 kernel**，目的是消除 HBM 往返，且主干是低精度 GEMM。
4. 计算图是**检索/select 融合**：轻量打分（GEMM/elementwise 归约）+ 重 top-K / argmax 选位 + gather，输出有序索引/值，**select 成本 ≥ 打分成本**（→ index-computation.md §5.5 `topk-select`）。

不满足 → 按以下优先级去别的文件：
- 纯 GEMM 调参（无量化/激活融合）→ `matrix-compute-merged.md`
- 标准 `npu_weight_quant_batchmatmul`（fp16 x @ int8 weight + antiquant_scale/offset）→ `quantization.md §4`
- 单算子量化（DynamicQuant / SwigluQuant 不含 GEMM）→ `quantization.md §2/§3`
- attention 主链 → `flash_attention.md`

### §0.2 ★ 形态识别问（Phase 2 第一步必须回答，答案决定后续哪些章节适用）

| # | 问题 | 影响 |
|---|------|------|
| **Q1** | 操作数 dtype 是什么？两个操作数都量化，还是只有 weight 量化？ | 双量化（int8@int8）→ §2 + CV1 用 int32 累加器；仅 weight 量化（fp16 x @ int8 w）→ 多半是 `quantization.md §4`，不是本文件 |
| **Q2** | 后处理含 **clamp**（值域截断）吗？输出 dtype 是 fp16 还是 bf16/int32？ | 决定 §2 L1.5 的 tile 分档：fp16 输出需 clamp 走小 tile，bf16/int32 值域有界可关 clamp 开 128×128 大 tile（本类最大的一笔之一） |
| **Q3** | 参考实现的反量化/量化在什么精度链上算？有没有 fp16 中间舍入？ | 决定 CV2 精度契约：必须逐位复刻参考的舍入点（"算得更准"反而 fail） |
| **Q4** | 有 MoE / 分组结构吗（per-expert weight、group_list）？ | 有 → §3 走两 kernel 链 + per-expert 分组 launch；无 → §2 单 GEMM |
| **Q5** | shape 离散度大吗（M/N/K 跨度大、专家行数 m_e 忽大忽小）？ | 大 → CV4 必须 host 侧动态分档 tile，固定 tile 必有 case 空转或溢出 |
| **Q6** | 有重 top-K / argmax / select 选位吗？输出要**有序**索引吗？ | 有 → index-computation.md §5：select 是主墙，先定选位结构（禁 `tl.cumsum` 前缀链，index-computation.md §5.4 S1），再谈打分优化；K 大 + S2 大时 select 成本超过打分一个量级 |

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束在 QuantMatmul、GroupedMatmulSwigluQuant 两类算子中均适用；各算子专属约束见 §2/§3。其他文件已提取的通用约束（`quantization.md` Q1 动态核数 / Q2 1D grid+核内循环 / Q3 contiguous / Q5 flatten / Q6 forward 禁 torch / Q7 禁 continue-break / Q8 BLOCK pow2）此处不再重复，各算子章节引用时标注。

### CV1 ★ 低精度 GEMM 必须用 `tl.dot` + 整型/浮点累加器，禁止逐维外积

- **必须** `tl.dot(a, b, acc)`，累加器 dtype 跟随操作数：int8@int8 用 **`tl.int32` 累加器**（`tl.dot` int8 → int32 是硬件原生路径），fp16/bf16 操作数用 fp32 累加器。
- **必须** 用**累加器融合形式** `acc = tl.dot(a, b, acc)`——acc 作为第三个实参传进 dot，让 K 归约滚在 MMAD 链上。
- **禁止** `acc += tl.dot(a, b, out_dtype=...)`——"数学等价"但每次 K 迭代**物化一份 `[BM,BN]` 全尺寸 dot 输出再相加**，UB 占用翻倍、MMAD 链断开。
- **禁止** 把操作数升 fp32 再 dot（走异构路径，慢）；**禁止** 逐维标量循环 `for k in range(K): acc += a[:,None]*b[None,:]` 代替矩阵乘。
- **Why（0812 批跑实测）**：同一份 §3 骨架的重生成版把 `tl.dot(a,b,acc)` 改写成 `acc += tl.dot(a,b,out_dtype=int32)`，60/60 精度全过却整体慢 **4x**、大 tile 档 case 慢 **22~182x**（0.5076x vs 原版 1.6060x）；原版逐 case 0.076~0.16ms，重生成版同 case 3.6~13.8ms。精度不报错 + Phase 4 只扫 BLOCK_K ⇒ 这条退化一路漏到终局。
- **Why**：逐维外积让 Cube 完全空转（`aic_mac_ratio` 个位数），实测直接超时无有效数据。两算子的主 GEMM 都靠 `tl.dot` + int32 累加器拿到基础加速（QuantMatmul int8@int8、GroupedMatmul 的 `gmm_dequant_kernel_i8`）。
- **判别信号**：`aic_mac_ratio < 10%` 且代码里有 `for k in range(K)` 逐维累加 ⇒ 立即重写为 `tl.dot`，不用扫参。
- **判别信号（代码指纹，精度全过也命中）**：K 循环体内出现 `acc += tl.dot(` ⇒ 立即改 `acc = tl.dot(a, b, acc)`；与 offsets 未外提（见 CV7b）并列为"精度全过但比历史慢 4x+"的两大指纹。
- **与 `flash_attention.md` L1.1 的差异**：FA 类要求"原生 dtype 操作数 + fp32 累加器"；本类多一种 **int32 累加器**（int8 乘积在 int32 上累加），是量化 GEMM 的常态，不是错误。

### CV2 ★★ 量化精度链必须**逐位复刻参考实现**，禁止"算得更准"

- **必须** 先确认参考实现的算术路径（反量化在 fp32 还是 fp16 上做、scale 是否被 crop、sigmoid 怎么算、rounding 语义），再逐位复刻；**"比参考更准"是错的**。
- **必须** 在参考会舍入的每个位置都 cast 回对应 dtype（典型：两级 dequant 的 fp16 双舍入，见 §3 L1.3）。
- **禁止** 用 `tl.sigmoid`（Ascend 后端系统性偏差）→ 手写 `x / (1.0 + tl.exp(-x))`。
- **禁止** 用 `tl.round`（默认 round-half-to-even）→ 用 `tl.where(q >= 0, q + 0.5, q - 0.5)` 实现 **round-half-away-from-zero**，与 `npu_*_quant` 语义一致。
- **禁止** 对 int32 用 `tl.clamp`（仅支持浮点）→ int32 阶段用 `tl.maximum/tl.minimum`。
- **Why**：量化类算子的 golden 是参考在**特定精度链**上算出来的，实现若全程 fp32 会"比参考更准"反而不达标；某 SwigluQuant 形态全程 fp32 稳定落在 matched_ratio 0.76（判据 0.9），与 shape 无关。QuantMatmul 的 scale crop（19-bit）同理——scale 的 1 ulp 偏差经大 K 累加会顶穿阈值。
- **判别信号**：失败集合**严格与 dtype/精度路径相关**（如"全部且仅仅是 fp16/bf16 case"，或"全部且仅仅是带 pertoken 的 case"）⇒ 直接查算术路径，不要去搜 tiling。

### CV3 ★ grid 收缩到 `min(核数, tiles)` + 核内步长循环，核数必须动态读取

- **必须** host 侧算 `total_tiles`，`grid = (min(NUM_CORES, total_tiles),)`；kernel 内 `for tile in range(pid, total_tiles, NUM_PROGS): ...`。
- **必须** 动态读取核数：Cube kernel 读 `cube_core_num`，Vector kernel 读 `vector_core_num`（`torch_npu.npu.npu_config.get_device_limit(0)`），带合理默认值兜底。
- **禁止** `grid = (total_tiles,)`（total > 核数时任务被重复计算）；**禁止** 硬编码 `num_cores=8/24/48`（不同 NPU 核数不同）。
- **Why**：量化融合算子 shape 离散度大，total_tiles 跨度从几个到上千；只有 `min()` 两头都对。QuantMatmul 实测 grid 收缩 + 核内步长是脱离 1x 区间的基础写法。
- ⚠️ `NUM_CORES` 必须作为 `tl.constexpr` 传入（随 case 变化的值会触发编译爆炸）；`grid` 是启动参数，随 case 变化无妨。任何改变任务空间的改动（压平循环维等）**必须同步改 grid**，否则表现为静默算少 → AccuracyError 而非报错。

### CV4 ★ tile 必须 host 侧动态分档，禁止全局固定 tile

- **必须** 按 (M, N, K, dtype, 后处理组合) 在 host 侧选 `(BLOCK_M, BLOCK_N, BLOCK_K)`，典型分 3~5 档（大/中/小 + 特殊边界）。
- **禁止** 全局一个固定 tile（大 shape 溢出 UB、小 shape 空转）；**禁止** 只拿大 case 扫参得到的最优直接全局套用（几何平均口径下中等 case 会塌陷，见 §7.0）。
- **Why**：QuantMatmul 的 `_select_tile` 按 `total_mn = M*N`、`do_clamp`、M/N 量级分 5 档；GroupedMatmul 按 per-expert `m_e`、N、`tiles vs cube_cores` 分 4 档。两算子都靠分档把最慢的 case 拉起来。
- **配套**：分档判定里**必须含一档小 shape 收缩**（`min(..., ceil(...))`），否则固定大 tile 在 `M/BLOCK_M < 1` 时纯空转。

### CV5 ★★★ UB 预算：溢出根因常是某个后处理算子（clamp），不是 tile 本身

- **必须** 把 tile 面积上限**算出来**（不是拍脑袋）：`scores/acc` tile 是 `[BM, BN]` fp32 ⇒ 元素数有明确上界；再叠加后处理物化的临时缓冲。
- **关键洞察**：`tl.minimum(tl.maximum(acc, lo), hi)` 这类 **clamp 会在 `[BM,BN]` 上物化 2 个全尺寸 fp32 临时缓冲**，是 UB 溢出的**真正根因**；而 `*scale` / `*pertoken` / `+bias` / `+offset` 这些逐元素后处理（无 clamp）不占额外全尺寸缓冲。
- **必须** 因此**按"是否需要 clamp"分两条 tile 路径**（见 §2 L1.5 的实测破局：128×128+clamp 溢出 258KB，去掉 clamp 后 128×128 稳稳放下 192KB UB）。
- **禁止** 看到 `BLOCK = min(..., 64)  # avoid UB overflow` 这类预防性封顶就直接删（删了确实编译失败）；要先定位**是哪个后处理算子占了 UB**，要么移走、要么（如 clamp）通过分档路径绕开，再重扫 BLOCK。
- **Why**：QuantMatmul opt_iter_2 仅靠"无 clamp 输出开 128×128"一项，geomean 1.177 → 1.2238，是该算子 Phase 4 最大的单笔。UB 边界**不可用简单阈值预测**（同为 128×128，带 clamp 溢出、不带 clamp 放得下）——只能实测撞出来（编译器报 `requires N bits while 1572864 bits available`）。
- **判别信号**：某档 tile 编译失败 / 大批 case 连续 fail，且失败集合与"是否带 clamp / 是否 fp16"严格相关 ⇒ 查后处理缓冲，不要去缩小 tile。

### CV6 多后处理必须留在 `tl.dot` epilogue 单 kernel 内，顺序按精度链

- **必须** 把 scale / pertoken / bias / offset / clamp 等后处理**接在 `tl.dot` 累加器之后、同一 kernel 内**完成，按参考精度链的顺序排（典型：`acc.to(fp32) * scale → * pertoken → + bias → + offset → clamp → cast(out_dtype)`）。
- **禁止** 把后处理拆成独立 kernel（除非有 msprof 证据证明解耦收益，参考 `quantization.md §4 L1.10/L1.11` 的解耦判据）——每拆一个 MIX kernel 多 ~4.5us 固定 launch 开销，小 case 直接被吃光。
- **int32 bias 特例**（QuantMatmul §2 L1.4）：当 bias 是 int32 且无 pertoken 时，参考链是 `(acc + bias) * scale`（先加后乘），**不是** `acc*scale + bias`——顺序错了精度 fail。
- **Why**：融合的目的就是消除 HBM 往返；后处理留在 epilogue 既省 launch 又省一次 round-trip。两算子的后处理全部内联在主 kernel 尾部。

### CV7 host 侧 flatten batch → 2D/3D，kernel 内禁用 `%` 模运算

- **必须** 多维/批量输入在 `forward()` 里 `reshape(B, M, K)` / flatten 成 2D，kernel 只处理 `(batch, row, col)` 三维或二维逻辑。
- **禁止** kernel 内用 `%`（取模）拆 tile 坐标 → 用 `bb = tile // TILES_PER_BATCH; rem = tile - bb*TILES_PER_BATCH`（减法替代），实测 `%` 在 Ascend 上慢。
- **Why**：QuantMatmul 把任意前缀维 flatten 成 `B*M` 行三维处理；tile 坐标全部用除法+减法还原，不用 `%`。

### CV7b ★ 持久化 tile 循环的 `tl.arange` 基向量必须外提，禁止迭代内重算

- **必须** 把 `offs_m_base = tl.arange(0, BLOCK_M)` / `offs_n_base` / `offs_k_base` 三条基向量**声明在 tile 循环体之外**（与 `num_tiles_m/n` 同层），循环体内只做 `offs_m = m_start + offs_m_base` 的加法。
- **禁止** 在 `for tile_idx in range(pid, num_tiles, NUM_CORES)` 体内重新 `tl.arange(...)`——每 tile 每维重算一次序列生成，E 个 expert × 上千 tile 的累计发射开销可观，且阻断编译器对 tile 循环的归纳优化。
- **Why（0812 批跑实测）**：与 CV1 的累加器形式**同批发生**的两处"等价改写"之一（基向量内联 + `acc +=`），合并把 1.6060x 的架构拖到 0.5076x；精度 60/60 全过，纯性能退化零报错。
- **判别信号（代码指纹）**：tile 循环体内出现 `tl.arange` ⇒ 外提；与 `acc += tl.dot(` 并列为本卡的静态预检指纹。

### CV8 后处理开关必须 `tl.constexpr` 特化，禁止运行时标量分支

- **必须** `HAS_PERTOKEN` / `HAS_BIAS` / `BIAS_INT32` / `BIAS_BATCHED` / `HAS_OFFSET` / `DO_CLAMP` 等布尔以 `tl.constexpr` 传入，让编译器把不适用的整段后处理 DCE 掉。
- **禁止** 把它们作为运行时标量在 kernel 内 `if` 分支（会保留全部分支的指令与 UB 占用）。
- **Why**：不同 case 的后处理组合不同（有的带 bias 有的不带、有的需 clamp 有的不需），constexpr 特化让每档组合编译出最小化的 kernel。

### CV9 ★ 中间标量必须 `.to(tl.int32)`，避免 int64 地址降级

- **必须** `tl.program_id` / `tl.arange` 结果 `.to(tl.int32)`；地址计算全程 int32。
- **Why**：int64 标量触发地址计算降级；NPU 上 int32 索引更高效（继承自 `quantization.md` Q4，本类因 tile 坐标计算多而尤其重要）。

### CV10 ★ `tl.cumsum` 在 Ascend 上标量化（aiv_scalar≈0.99），select/前缀定位必须绕开

> 🔄 已迁移至 `index-computation.md` §5.4 S1（选位家族通用经验）。本小节保留备份，不再作为权威来源。

- **必须** 凡 forward 出现"选 K 位之前的计数→定位"（`tl.cumsum` 前缀链）且 profiling `aiv_scalar_ratio≈0.99`，立即**换结构**：排序压缩（见 index-computation.md §5.5 L1.1）或矩阵乘前缀和（Sparse-块跳过型 100/101 用 matmul 绕前缀，56x）；**不要扫参，参数救不了标量链**。
- **禁止** 用 `torch.cumsum` host 前缀表（validator AST 白名单禁）。
- **Why（0817/5 实测）**：LightningIndexer 串行 chunk-scan + `tl.cumsum` 前缀在 S2=32768 时 **10.1ms**（aiv_scalar=0.999）；并行 scatter + 两次 int32 cumsum 仍 **4.2ms**（int32 走 vector 但跨 chunk 前缀仍串行，aiv_scalar=0.997）；改 sort-compaction 后 **0.112ms**——**同一 select 问题，结构选择差 90x+**。与 Sparse-块跳过型 100/101 同族结论。
- **判别信号**：代码里 `tl.cumsum` 用于选位 / 紧凑化 ⇒ 预检直接标红，换结构而不是调 BLOCK。

### CV11 ★ `npu_sort_v2` 走 NPU 侧，是 `aclnnSort`（AiCpu）的 40x+ 替代，且可复用双角色

> 🔄 已迁移至 `index-computation.md` §5.4 S2（选位家族通用经验）。本小节保留备份，不再作为权威来源。

- **必须** 需要"第 K 大的值" / "前 K 有序"时优先 `torch_npu.npu_sort_v2`（合法 `torch_npu.*` 命名空间，validator 放行），**喂 2D `[rows, N]` view**（1D 布局不可靠，0817/5 踩坑）。
- **必须** 复用同一个 sort 调两用的套路：① 升序取 `tau=sorted[N-K]` 当阈值 oracle（省全量 top-k）；② 对 masked 数组升序压缩（把"选中的 rank 位 + 未选中 LARGE2"紧凑到前 K 位）——**排序同时完成"选 K + 排序 + 去重"**。
- **禁止** 用 `torch.topk`（validator 禁，且是唯一已知快路径 ~17µs 但不可用）；**禁止** 纯 Triton 排序网络做 top-k（tl.sort 编译崩 / gather-bitonic ~1ms 地板）。
- **Why**：torch 参考的 `aclnnSort_SortAiCpu_Sort` 单次 **1466µs**（CPU sort）是参考主墙；实现侧 `npu_sort_v2` 两次仅 **38µs**——结构差 40x。

### CV12 ★ 检索/select 类算子：主墙在 select 不在打分；天花板估算先归零"可消除结构"

> 🔄 已迁移至 `index-computation.md` §5.4 S3（选位家族通用经验）。本小节保留备份，不再作为权威来源。

- **必须** 先做 profiling `operators` 分解，确定打分 vs select 各自占比，再定优化方向——本类打分核常已 cube-bound 很轻，**select 才是墙**（LightningIndexer：打分 38µs 占 impl 22%，select 相关两次 sort 38µs + build_masked 68µs 占 61%）。
- **必须** 天花板估算把**可消除的算法结构**（串行前缀和 / 逐元素定位）**先归零再算**——R1 曾推算 select ~1ms 上限，被 sort-compaction 证伪（select 侧几乎免费，只剩 sort 是真墙）。
- **禁止** 用打分核的优化套路（tile 放大、流水）去攻 select——select 是**结构问题**，不是参数问题。

---

## §2 QuantMatmul 算子（`quant-matmul`，低精度量化 GEMM + 多后处理）

**算子类别**: `quant-matmul`（双量化操作数 GEMM + 反量化后处理）
**典型特征**: `x1` int8 `[B,M,K]` @ `x2` int8 `[B,K,N]`（或 int32-weight），per-channel `scale` `[N]`、可选 `pertoken_scale` `[M]`、`bias`、`offset`、`clamp`；输出 int8/int32/fp16/bf16
**性能基准**: 57/57 verify pass，几何平均 **1.2238x** vs torch（Phase 3 生成 1.0902x → Phase 4 opt_iter_2 1.2238x），per-shape 区间 [0.473, 3.283]

### §2.0 首次生成必读：tile 分档与 clamp 路径必须在 Phase 3 就写对

本算子最大的性能杠杆（**clamp 分档**，见 L1.5）**不是 Phase 4 的可选优化，而是首次生成就要做对的结构**——Phase 4 opt_iter_0 曾因"无脑给所有大 shape 开 128×128（含需 clamp 的 fp16）"导致 UB 溢出、精度 37/57 崩盘（见 §5 证伪表）。正确做法是首次生成就按 L1.5 的两条路径分档。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 int8 GEMM 用 `tl.dot` + int32 累加器（见 CV1）
- int8@int8 → `acc = tl.zeros((BM,BN), tl.int32); acc = tl.dot(a, b, acc)`。
- **禁止** 逐维累加；**禁止** 操作数升 fp32。

#### L1.2 grid 收缩 + 核内步长，cube 核数动态读取（见 CV3）
```python
NUM_BLOCKS_M = (M + BLOCK_M - 1) // BLOCK_M
NUM_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
TOTAL_TILES = B * NUM_BLOCKS_M * NUM_BLOCKS_N
grid = (min(CUBE_CORE_NUM, TOTAL_TILES),)
# kernel: for tile in range(pid, NUM_BLOCKS, num_cores): ...
```
`CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)`，作为 `num_cores: tl.constexpr` 传入。

#### L1.3 量化精度链：scale 由 host 做 19-bit crop 后喂 kernel
- **必须** 认 scale 的精度处理在 host 侧（`get_input_groups` 已对 fp16输出/fp32scale/无pertoken 的 scale 做 19-bit crop 预处理，参考与本实现共用同一 scale）→ kernel 内直接 `acc_f * scale`（fp32 乘），**不再** 叠加旧的 fp16 向零截断。
- **Why**：scale 的 1 ulp 偏差经大 K 累加会顶穿阈值；crop 预处理让 kernel 内的乘法与参考逐位一致。
- 配套：scale 在 host `reshape(-1).to(torch.float32).contiguous()` 成 1D `[N]`。

#### L1.4 后处理顺序与 int32 bias 特例（见 CV6）
- 通用顺序：`acc.to(fp32) → * scale → * pertoken → + bias → + offset → clamp → cast(out_dtype)`。
- **int32 bias + 无 pertoken 特例**：参考链是 `(acc + bias) * scale`（先加后乘）：
```python
if BIAS_INT32 and not HAS_PERTOKEN:
    acc = acc + bias_i[None, :]      # int32 加法
    acc_f = acc.to(tl.float32) * scale_t[None, :]
else:
    acc_f = acc.to(tl.float32) * scale_t[None, :]
    if HAS_PERTOKEN: acc_f = acc_f * pt[:, None]
    if HAS_BIAS:     acc_f = acc_f + bias_t.to(fp32)[None, :]
    if HAS_OFFSET:   acc_f = acc_f + off_t[None, :]
```
- **顺序错了直接精度 fail**（int32 bias 必须先加再乘 scale）。

#### L1.5 ★★★ tile 按"是否需要 clamp"分两条路径（本算子最大的一笔，首次生成就要做对）

**UB 溢出的唯一根因是 clamp，不是 tile 面积。** `tl.minimum(tl.maximum(acc_f, lo), hi)` 在 `[BM,BN]` 上物化 **2 个全尺寸 fp32 临时缓冲**，128×128 + clamp = 258KB 溢出 192KB UB；而无 clamp 的后处理（`*scale/*pertoken/+bias/+offset`）不占额外全尺寸缓冲，128×128 稳稳放下。

- **必须** 在 host 侧先判定 `do_clamp`，按它分档：
```python
def _select_tile(M, N, K, simple, do_clamp):
    block_k = 256      # BK=512 时 multi-buffer 的 a+b 缓冲超 UB
    total_mn = M * N
    # 无 clamp 且大 shape：开 128×128（含 pertoken/bias case，这些恰是最慢的）
    if (not do_clamp) and M >= 128 and N >= 128 and total_mn > 32*1024:
        return 128, 128, block_k
    if total_mn <= 32*1024:                 return 64, 64, block_k
    elif M <= 64:                           return 64, (128 if N>128 else 64), block_k
    elif N <= 64:                           return (128 if M>128 else 64), 64, block_k
    elif M <= 128 and N <= 128:             return 64, 64, block_k
    else:                                   return 64, 128, block_k     # fp16 需 clamp → 小 tile
```
- **bf16/int32 关 clamp**（值域有界：`acc ≤ 127*127*K*scale < 1.4e8 << bf16.max`，clamp 是 no-op），既安全又让大 shape 用上 128×128；**fp16 需 clamp 走 64×128**。
- **Why（实测）**：opt_iter_2 仅这一项改动，geomean **1.177 → 1.2238**，最慢的 bf16 pertoken/bias 大 shape（case42 0.33→0.47、case44 0.33→0.53）显著提升。
- **判别信号**：某档大 tile 编译失败 / 精度崩，且失败集合严格按"是否 fp16（需 clamp）"划分 ⇒ 查 clamp 缓冲，不要缩小无 clamp 路径的 tile。

#### L1.6 clamp 范围按 dtype 显式映射，不用 `torch.finfo`
```python
if out_dtype == torch.float16:   out_min, out_max, do_clamp = -65504.0, 65504.0, True
elif out_dtype == torch.int8:    out_min, out_max, do_clamp = -128.0, 127.0, True
else:                            out_min, out_max, do_clamp = 0.0, 0.0, False   # bf16/int32 无需 clamp
```

#### L1.7 host 侧 flatten batch → 3D + 禁用 `%` 模运算（见 CV7）
- `x1.reshape(B, M, K).contiguous()`；kernel 内 tile 坐标用 `bb = tile // TILES_PER_BATCH; rem = tile - bb*TILES_PER_BATCH`，禁止 `%`。

#### L1.8 后处理开关 constexpr 特化（见 CV8）
- `HAS_PERTOKEN/HAS_BIAS/BIAS_INT32/BIAS_BATCHED/HAS_OFFSET/DO_CLAMP` 全部 `tl.constexpr`。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧决策树
```python
def forward(self, x1, x2, scale, offset=None, pertoken_scale=None, bias=None, output_dtype=None, group_sizes=None):
    out_dtype = parse_output_dtype(output_dtype)               # 默认 fp16
    M, K = x1.shape[-2], x1.shape[-1]; N = x2.shape[-1]
    B = x1.numel() // (M*K)
    x1 = x1.reshape(B, M, K).contiguous(); x2 = x2.reshape(B, K, N).contiguous()
    scale_1d = scale.reshape(-1).to(torch.float32).contiguous()    # host 19-bit crop 已做
    # bias/pertoken/offset → 1D fp32（has_* 判定 + dummy 占位）
    out_min, out_max, do_clamp = dtype_clamp_map(out_dtype)        # L1.6
    simple = (not has_pertoken) and (not has_bias) and (not has_offset)
    BLOCK_M, BLOCK_N, BLOCK_K = _select_tile(M, N, K, simple, do_clamp)   # L1.5
    TOTAL_TILES = B * cdiv(M,BLOCK_M) * cdiv(N,BLOCK_N)
    grid = (min(CUBE_CORE_NUM, TOTAL_TILES),)
    quant_matmul_kernel[grid](x1, x2, scale_1d, pt, bias_t, off_t, out,
        B, M, N, K, out_min, out_max, num_cores=grid[0],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_PERTOKEN=has_pertoken, HAS_BIAS=has_bias, BIAS_INT32=bias_int32,
        BIAS_BATCHED=bias_batched, HAS_OFFSET=has_offset, DO_CLAMP=do_clamp)
    return out.reshape(lead + (M, N))
```

#### L2.2 Kernel 主循环骨架
```python
@triton.jit
def quant_matmul_kernel(x1_ptr, x2_ptr, scale_ptr, pertoken_ptr, bias_ptr, offset_ptr, out_ptr,
                        B, M, N, K, out_min, out_max,
                        num_cores: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        HAS_PERTOKEN: tl.constexpr, HAS_BIAS: tl.constexpr, BIAS_INT32: tl.constexpr,
                        BIAS_BATCHED: tl.constexpr, HAS_OFFSET: tl.constexpr, DO_CLAMP: tl.constexpr):
    pid = tl.program_id(0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M); NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    TILES_PER_BATCH = NUM_BLOCKS_M * NUM_BLOCKS_N; NUM_BLOCKS = B * TILES_PER_BATCH
    rm = tl.arange(0, BLOCK_M); rn = tl.arange(0, BLOCK_N); rk = tl.arange(0, BLOCK_K)
    for tile in range(pid, NUM_BLOCKS, num_cores):
        bb = tile // TILES_PER_BATCH; rem = tile - bb * TILES_PER_BATCH
        bm = rem // NUM_BLOCKS_N;     bn = rem - bm * NUM_BLOCKS_N
        m0 = bm * BLOCK_M; n0 = bn * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)            # L1.1
        for k in range(0, K, BLOCK_K):
            a = tl.load(...); b = tl.load(...)
            acc = tl.dot(a, b, acc)
        # 后处理 L1.4（顺序 + int32 bias 特例）→ L1.5（DO_CLAMP 分支）→ store
```

### §2.3 Layer 3: 关键技巧（Agent 可参考但不可复制结构）

#### L3.1 ★ clamp 分档路径（本算子核心，见 L1.5）
单机 UB 实验（`/tmp/ub_post.py`）实测：bf16 输出 + 无 clamp 时，bias-only / pertoken-only / pt+bias 的 128×128×256 全部放得下 UB；**clamp 是 128×128 溢出的唯一根因**。据此把 128×128 条件从 `simple and not do_clamp` 放宽为 `not do_clamp`，覆盖所有 bf16/int32 无 clamp 大 shape。

#### L3.2 dummy 指针复用 scale 占位
未启用的后处理 tensor（无 pertoken/bias/offset）用 `scale_1d` 作 dummy 指针传入，靠 `HAS_*: tl.constexpr = False` 让对应 load 被 DCE，避免分支与额外 buffer。

#### L3.3 int32 bias 先加后乘
见 L1.4。坏味道：kernel 注释写 `acc*scale + bias` 而 bias 实为 int32 ⇒ 顺序错，精度 fail。

### §2.4 QuantMatmul 性能基准与优化记录

| 版本 | cases | geomean | 说明 |
|------|-------|---------|------|
| Phase 3 iter_2（生成） | 57/57 | 1.0902x | int32 累加 + 动态 tile + grid 收缩，首次成型 |
| opt_iter_0 | 37/57 ❌ | — | **无脑给所有大 shape 开 128×128（含 fp16 需 clamp）→ UB 溢出精度崩**（见 §5 证伪） |
| opt_iter_1 | 57/57 | 1.177x | 收回：128×128 仅给 simple 路径 |
| **opt_iter_2（最终）** | **57/57** | **1.2238x** | **128×128 条件放宽到 `not do_clamp`（bf16/int32 无 clamp 全覆盖），fp16 仍 64×128** |

**有效优化点**: ① clamp 分档（L1.5，最大的一笔 1.177→1.2238）② 动态 tile 分档 ③ grid 收缩+核内步长 ④ int32 累加 ⑤ 后处理 epilogue 内联。
**关键结论**: 本算子的性能天花板由 **"clamp 是否占 UB"** 决定；把无 clamp 输出开到大 tile 是脱离 1.2x 区间的唯一杠杆。

---

## §3 GroupedMatmulSwigluQuant 算子（`gmm-swiglu-quant`，MoE FFN 融合）

**算子类别**: `gmm-swiglu-quant`（per-expert 分组 GEMM + SwiGLU + 动态量化）
**典型特征**: `x` fp16 `[M,K]` + per-expert int8 `weight[E,K,N]` + `weight_scale[E,1,N]` + `x_scale[M]` + `group_list`（专家行边界）；输出 `out_q` int8 `[M, N//2]` + `out_s` fp32 `[M]`（逐行动态量化 scale）
**性能基准**: 60/60 verify pass，几何平均 **1.6967x** vs torch（超 target 0.8x 达 112%），per-shape 区间 [1.106, 3.504]；Phase 3 一次成型 1.6060x → Phase 4 opt_iter_0 微调至 1.6967x

### §3.0 首次生成必读：两 kernel 链 + per-expert 分组是"一次成型"的关键

本算子 **Phase 3 一轮就达 1.6060x**（已超 target 0.8x 两倍），靠的是首次生成就把架构写对：**K1（GMM+dequant，Cube）/ K2（SwiGLU+quant，Vector）职责分离 + 每个 expert 独立 launch + tile 按专家行数 m_e 分档**。后续 Phase 4 只做微调（1.6060→1.6967）。Designer 必须在草图阶段就定下这个两 kernel 结构，不要试图单 kernel 承担全部。

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须拆成 K1（GMM+dequant）/ K2（SwiGLU+dynamic quant）两 kernel 链
- **必须** K1 在 Cube 上做 per-expert `gmm_dequant_kernel`（int8@int8 → int32 累加 → 两级 dequant → fp16 workspace `y[M,N]`）；K2 在 Vector 上做 `swiglu_dquant_kernel`（读 y → SwiGLU → 逐行 max_abs → 动态量化 → int8 out_q + fp32 out_s）。
- **禁止** 把 SwiGLU+量化塞进 GEMM kernel（Cube↔Vector 混在一个 kernel 里同步代价大）；**禁止** 把 dequant 拆成第三个 kernel（多一次 HBM 往返）。
- **Why**：MoE FFN 的两部分计算特征完全不同（K1 是 Cube-bound GEMM，K2 是 Vector-bound reduce+量化），分两个 kernel 让各自走对硬件路径；中间 fp16 workspace 是必要的解耦点。
- 配套：K1 按 expert 逐个 launch（`_launch_gmm_dequant_all_experts`），K2 一次性处理全部 M 行。

#### L1.2 K1 用 int32 累加 + 按 weight dtype 分 i8/i32w 两个 kernel（见 CV1）
- weight 是 int8 → `gmm_dequant_kernel_i8`（int32 累加器，操作数保持 int8）；weight 是 int32 → `gmm_dequant_kernel_i32w`（操作数 cast fp32 后 fp32 累加）。
- host 侧按 `w_dtype = weight[0].dtype` 分派。
- **必须** K1 的 K 归约写成累加器融合形式 `acc = tl.dot(a, b, acc)`、`tl.arange` 基向量外提到 tile 循环外（见 CV1 / CV7b）——本算子 0812 重生成版仅这两处"等价改写"（`acc += tl.dot(out_dtype=...)` + 基向量内联）就把 1.6060x 拖到 **0.5076x**（同 case impl 0.076ms → 13.8ms，182x），且 60/60 精度全过零报错。

#### L1.3 ★ 两级 dequant 必须 fp16 双舍入，逐位对齐 Ascend 参考链
- **必须** dequant 走 `int32 acc → fp32 → *ws → fp16 → fp32 → *xs → fp16`（两级，每级 cast 回 fp16 舍入），**不是** 一次性 fp32 乘完。
```python
f = acc.to(tl.float32)
f = f * ws[None, :]        # weight_scale [N]
f = f.to(tl.float16)       # ← 第一级 fp16 舍入（参考链在此舍入）
f = f.to(tl.float32)
f = f * xs[:, None]        # x_scale [M]
f = f.to(tl.float16)       # ← 第二级 fp16 舍入
```
- **禁止** 跳过中间 fp16 cast（"算得更准"反而 fail，见 CV2）。

#### L1.4 K2 手写 sigmoid（fp32）+ round-half-away-from-zero + int32 clamp + 寄存器内 per-row scale（见 CV2）
- **必须** `sig = gate_f32 / (1.0 + tl.exp(-gate_f32))`（fp32，禁止 `tl.sigmoid`）；`sw = sig * up_f32 → fp16 → fp32`（对齐参考的 fp16 舍入点）。
- **必须** per-row `max_abs = tl.max(tl.abs(sw), axis=0)` → `scale = 127.0 / max(max_abs, 1e-10)` 在寄存器内算（K2 是 Vector kernel，整行常驻）。
- **必须** round-half-away-from-zero：`q = sw*scale; q = tl.where(q>=0, q+0.5, q-0.5); q = q.to(int32); q = tl.maximum(tl.minimum(q, 127), -128); q = q.to(int8)`。

#### L1.5 per-expert 分组 launch + tile 按专家行数 m_e 分档（见 CV4）
- **必须** host 侧按 `group_list` 遍历 expert，每个 expert 取 `m_e = curr - prev` 行子张量独立 launch K1。
- **必须** tile 按 `(m_e, N)` 分档：
```python
if (N % 512 != 0) and (896 <= N <= 1536):           BLOCK_M=64,  BLOCK_N=128
elif (m_e < 128) and (512 <= N <= 1024):            BLOCK_M=16,  BLOCK_N=128
elif (M >= 500) and (N >= 2048):
    tiles_512 = cdiv(m_e,64)*cdiv(N,512)
    if 2*tiles_512 <= cube_cores:                   BLOCK_M=64,  BLOCK_N=256
    else:                                            BLOCK_M=64,  BLOCK_N=512
else:                                                BLOCK_M=64,  BLOCK_N=256
BLOCK_K = 128
```
- **Why**：专家行数 m_e 忽大忽小（有的专家几百行、有的几千行），固定 tile 会让小专家空转、大专家溢出；按 m_e 分档是 per-expert 结构的必然要求。

#### L1.6 K1/K2 grid 收缩 + 核内步长，cube/vec 核数动态读取（见 CV3）
- K1：`total_tiles = cdiv(m_e,BM)*cdiv(N,BN)`，`grid = min(cube_cores, total_tiles)`，kernel 内 `for tile_idx in range(pid, num_tiles, NUM_CORES)`。
- K2：`grid = M if M < 4*vec_cores else 4*vec_cores`，kernel 内 `for row_idx in range(pid, M, num_programs)`。
- `cube_core_num` / `vector_core_num` 从 `torch_npu.npu.npu_config.get_device_limit(0)` 动态读取。

#### L1.7 标量 scale store 的 broadcast trick
- per-row scale `out_s[M]` 是标量写回，用 `r1 = tl.arange(0, 1); tl.store(out_s_ptr + row_idx + r1, (max_abs/127.0) + r1*0.0)` 借 1 元素张量完成标量 store（避免标量直接 store 的限制）。

#### L1.8 host 侧 contiguous + 无 `%`（见 CV7）
- `x/weight/weight_scale/x_scale` 全部 `.contiguous()`；expert 行子张量用 `narrow(0, curr-m_e, m_e)` 取连续段。

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧两段分派
```python
def forward(self, x, weight, weight_scale, x_scale, group_list):
    M, K = x.shape; E = len(weight); N = weight[0].shape[1]; HALF_N = N // 2
    # contiguous；分配 y[M,N] fp16 workspace、out_q[M,HALF_N] int8、out_s[M] fp32
    cube_cores = get_device_limit('cube_core_num', 24); vec_cores = get_device_limit('vector_core_num', 48)
    # 路由 gate（v4s / v4-wide / fallback）；首版 fallback 走两 kernel 链
    _launch_gmm_dequant_all_experts(x, weight, weight_scale, x_scale, y, group_list,
                                    K, N, HALF_N, M, E, cube_cores, w_dtype)   # K1
    grid_k2 = M if M < 4*vec_cores else 4*vec_cores
    swiglu_dquant_kernel[(grid_k2,)](y, out_q, out_s, M, N, HALF_N,
                                     y.stride(0), out_q.stride(0), out_q.stride(1),
                                     BLOCK_N=triton.next_power_of_2(HALF_N), NUM_VEC_CORES=grid_k2)  # K2
    return out_q, out_s
```

#### L2.2 K1 per-expert 循环骨架
```python
def _launch_gmm_dequant_all_experts(x, weight, weight_scale, x_scale, y, group_list, K, N, HALF_N, M, E, cube_cores, w_dtype):
    prev = 0
    for e in range(E):
        curr = int(group_list[e].item()); m_e = curr - prev; prev = curr
        if m_e == 0: continue
        x_e = x.narrow(0, curr-m_e, m_e); w_e = weight[e]; ws_e = weight_scale[e]
        xs_e = x_scale.narrow(0, curr-m_e, m_e); y_e = y.narrow(0, curr-m_e, m_e)
        BLOCK_M, BLOCK_N = _select_gmm_tile(m_e, N, M, cube_cores)        # L1.5
        total_tiles = cdiv(m_e,BLOCK_M)*cdiv(N,BLOCK_N)
        grid = cube_cores if cube_cores < total_tiles else total_tiles
        (gmm_dequant_kernel_i8 if w_dtype==torch.int8 else gmm_dequant_kernel_i32w)[(grid,)](
            x_e, w_e, ws_e, xs_e, y_e, m_e, K, N, HALF_N, ...,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=128, NUM_CORES=grid)
```

kernel 侧 K 归约与基向量的**强制形态**（CV1 / CV7b，缺一即退化）：
```python
    offs_m_base = tl.arange(0, BLOCK_M)          # ← 基向量外提，tile 循环外（CV7b）
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    for tile_idx in range(pid, num_tiles, NUM_CORES):
        ...
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(...); b = tl.load(...)
            acc = tl.dot(a, b, acc)              # ← 累加器融合形式，禁止 acc += tl.dot(...)（CV1）
```

#### L2.3 K2 SwiGLU+quant 骨架
```python
@triton.jit
def swiglu_dquant_kernel(y_ptr, out_q_ptr, out_s_ptr, M, N, HALF_N, stride_ym, stride_qm, stride_qn,
                         BLOCK_N: tl.constexpr, NUM_VEC_CORES: tl.constexpr):
    pid = tl.program_id(0); num_programs = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_N)
    for row_idx in range(pid, M, num_programs):
        gate = tl.load(y_ptr + row_idx*stride_ym + offs_n,         mask=(offs_n<HALF_N), other=0.0)
        up   = tl.load(y_ptr + row_idx*stride_ym + HALF_N + offs_n, mask=(HALF_N+offs_n<N), other=0.0)
        sig  = gate.to(fp32) / (1.0 + tl.exp(-gate.to(fp32)))      # L1.4 手写 sigmoid
        sw   = (sig * up.to(fp32)).to(fp16).to(fp32)               # 对齐参考 fp16 舍入
        max_abs = tl.max(tl.abs(sw), axis=0)
        scale = 127.0 / tl.maximum(max_abs, 1e-10)
        q = tl.where(sw*scale >= 0, sw*scale+0.5, sw*scale-0.5).to(int32)
        q = tl.maximum(tl.minimum(q, 127), -128).to(int8)
        tl.store(out_q_ptr + row_idx*stride_qm + offs_n, q, mask=(offs_n<HALF_N))
        r1 = tl.arange(0,1); tl.store(out_s_ptr + row_idx + r1, (max_abs/127.0) + r1*0.0)   # L1.7
```

### §3.3 Layer 3: 关键技巧（Agent 可参考但不可复制结构）

#### L3.1 两 kernel 链 + fp16 workspace 解耦
K1（Cube）与 K2（Vector）经 fp16 `y[M,N]` workspace 解耦，各自走对硬件路径；workspace 是必要的中间落地点，不要试图消除（消除=单 kernel=Cube↔Vector 强耦合，更慢）。

#### L3.2 两级 dequant 的 fp16 双舍入
见 L1.3。坏味道：dequant 注释写"int32 → fp32 → *ws*xs → fp16"（一次舍入）⇒ 与参考两级舍入不符，精度 fail。

#### L3.3 per-expert tile 分档 + grid 按 tiles/cores 自适应
见 L1.5/L1.6。大专家走 BN=512、小专家走 BM=16，避免空转与溢出。

#### L3.4 路由 gate 留扩展位
`_v4s_eligible` / `_v4wide_eligible` / fallback 三类分支首版都落 fallback（两 kernel 链），为后续把"小 HALF_N 单 kernel 直出"等更激进融合留扩展位——**首版不要急着上 v4s/v4-wide**，先把两 kernel 链跑稳。

### §3.4 GroupedMatmulSwigluQuant 性能基准与优化记录

| 版本 | cases | geomean | 相对 Phase3 | 说明 |
|------|-------|---------|------------|------|
| Phase 3（生成） | 60/60 | 1.6060x | 1.0x | 两 kernel 链 + per-expert 分档 + int32 累加 + fp16 双舍入，**一次成型**已超 target 0.8x 两倍 |
| **opt_iter_0（最终）** | **60/60** | **1.6967x** | **1.0561x** | 微调 tile 分档（per-shape max 3.504x @ 256×8192） |

**有效优化点**: ① 两 kernel 链职责分离（架构基础）② per-expert tile 分档 ③ int32 累加 ④ 两级 fp16 双舍入 dequant ⑤ 手写 sigmoid + round-half-away ⑥ grid 按 tiles/cores 自适应。
**关键结论**: 本算子是 **"设计对了一次成型"** 的典型——架构（两 kernel + per-expert）写对就 1.6x，Phase 4 只剩微调空间；Designer 应在草图阶段就定下两 kernel 结构，不要在 Phase 3/4 反复改架构。

---

## §4 5_LightningIndexer 算子（`topk-select`，检索/select 融合）— 已迁移

> 🔄 **已迁移至 `index-computation.md` §5.5（LightningIndexer 完整经验）与 §5.4（选位家族通用经验 S1/S2/S3，原 CV10/CV11/CV12）**。
> 本算子运算形态 = 「score 打分 + top-K 选位 + 索引输出」，属索引计算家族的 `index-sort` / `topk-select`，
> 归位 `index-computation.md` 更准确。`sparse-index-topk` 子类标签不再使用，改以 `topk-select` 为准。
>
> cv-fusion.md 回归**纯 GEMM 量化融合卡**（QuantMatmul §2 / GroupedMatmulSwigluQuant §3）。
> 检索/select 类算子生成时读 `index-computation.md` §5，不要再读本文件 §4。

---
## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|------|---------|----------|
| 1 | ★★★ **clamp 分档**：无 clamp 输出（bf16/int32）开 128×128 大 tile，fp16 需 clamp 走 64×128 | QuantMatmul **1.177 → 1.2238** | 输出 dtype 决定 do_clamp；值域有界的 bf16/int32 关 clamp |
| 2 | ★★ 动态 tile 分档（host 侧按 shape/dtype/后处理分 3~5 档） | 两算子基础收益 | shape 离散度大 |
| 3 | ★★ int32 累加器（int8 GEMM） | Cube 不空转的前提 | int8 操作数 |
| 4 | ★★ 两级 fp16 双舍入 dequant | 精度前提（非性能，但少了直接 fail） | 反量化链有 fp16 舍入点 |
| 5 | ★ grid 收缩 `min(核数,tiles)` + 核内步长 | 脱离 1x 区间的基础写法 | 所有 kernel |
| 6 | ★ 后处理留 `tl.dot` epilogue（不拆 kernel） | 省 ~4.5us/次 launch | 小 case 尤其敏感 |
| 7 | ★★★ **top-K 选位用 sort-compaction 绕开 `tl.cumsum`**（count+build_masked+二次 `npu_sort_v2`+finalize） | 5_LightningIndexer **0.3178 → 3.9558（约 12.4x）** | select 尾处理出现串行 NCHUNK 循环 + `tl.cumsum` 前缀且 aiv_scalar≈0.99（index-computation.md §5.4 S1 / §5.5 L1.1） |
| 8 | ★★ `npu_sort_v2` 双角色复用（阈值 oracle + 压缩排序），必须 2D view | 5_LightningIndexer 两次 sort 仅 38µs vs 参考 AiCpu sort 1466µs | 需要"第 K 大的值" / "前 K 有序"时（index-computation.md §5.4 S2 / §5.5 L1.2） |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

| 方向 | 结果 | 原因 |
|------|------|------|
| **无脑给所有大 shape 开 128×128（含 fp16 需 clamp）** | **QuantMatmul opt_iter_0：57 → 37/57 精度崩**（UB 溢出 258KB） | clamp 物化 2 个全尺寸 fp32 缓冲；必须按 do_clamp 分档（§2 L1.5），不是按 shape 一刀切 |
| 把 BK 从 256 放到 512 求大 tile | UB 溢出 | multi-buffer 的 a+b 缓冲超 192KB；BK 固定 256 |
| int32 bias 路径写成 `acc*scale + bias`（先乘后加） | 精度 fail | 参考链是 `(acc+bias)*scale`（先加后乘），见 §2 L1.4 |
| dequant 一次性 fp32 乘完（跳过中间 fp16 cast） | 精度 fail（matched_ratio ~0.76） | 参考链是两级 fp16 双舍入，见 §3 L1.3 |
| 用 `tl.sigmoid` / `tl.round` / int32 `tl.clamp` | 精度 fail 或编译失败 | 见 CV2：手写 sigmoid、round-half-away、int32 用 max/min |
| 把 SwiGLU+量化塞进 GEMM kernel（单 kernel 承担全部） | 更慢 | Cube↔Vector 单 kernel 强耦合；必须两 kernel 链 |
| 把 dequant 拆成独立第三个 kernel | 更慢 | 多一次 HBM 往返；dequant 必须留在 GEMM kernel epilogue |
| `grid = (total_tiles,)` 不收缩 | task 被重复计算 | 必须 `min(核数, total_tiles)` |
| 硬编码 `num_cores` | 跨 NPU 退化 | 必须动态读 cube/vec 核数 |
| kernel 内用 `%` 拆 tile 坐标 | 慢 | 用 `//` + 减法替代，见 CV7 |
| 串行 chunk-select epilogue（grid=(rows,) 扫 NCHUNK + `tl.cumsum` 前缀） | **~120ms / 10.1ms**（aiv_scalar=0.999） | LightningIndexer R1：纯标量链，参数救不了，必须换结构（index-computation.md §5.5 L1.1） |
| 并行 chunk-scatter select（两次 int32 `tl.cumsum` 定位） | **4.2ms，抛弃** | int32 cumsum 跨 chunk 前缀仍串行（aiv_scalar=0.997）；vs sort-compaction 0.112ms 差 37 倍（index-computation.md §5.5 L1.1） |
| 纯 Triton 排序网络做 top-k（bitonic / tl.sort） | **~1.35ms 地板** | tl.sort 编译崩 / gather-bitonic ~1ms 地板；`torch.topk`（唯一快路径 ~17µs）被 validator 禁 |
| 检索打分核 BLOCK_J=1024 宽块 | **编译失败**（UB hivm-plan-memory） | 与 §2/§3 的 tile 放大结论不同——检索打分核的 UB 预算也顶不住宽块；256 是上限（index-computation.md §5.5 L1.3） |
| `npu_sort_v2` 喂 1D 布局 | 结果不确定 | 必须 2D `[rows,S2]` view（index-computation.md §5.5 L1.2） |

> ⚠️ 以上证伪项来自 QuantMatmul / GroupedMatmulSwigluQuant / 5_LightningIndexer 的**实测**失败。注意 §7.3：换编译器 / CANN 版本后，部分结构性结论（尤其是 UB 边界）可能变化，重测前先重新标定基线。

---

## §6 精度闸门（先过闸门，再谈性能）

- **判据来源**：一律走 `verify.py` 的 `matched_ratio`（逐元素相对阈值），不改阈值、不用 `--verify_not_required`。扫参阶段可用单 case 脚本秒级淘汰明显错误的变体。
- **判别推论**：若失败集合**严格与 dtype / 后处理组合相关**（如"全部且仅仅是 fp16 case"、"全部且仅仅是带 pertoken 的 case"、"全部且仅仅是 int32 bias 的 case"）⇒ 基本可直接判为**算术路径/精度链不匹配**（CV2），不用去搜 tiling。
  - QuantMatmul int32 bias 顺序错 → 失败集 = 全部 int32 bias case。
  - GroupedMatmulSwigluQuant 两级 dequant 漏舍入 → 失败集 = 全部 fp16 case。
- **精度契约会倒逼结构**：参考在 fp16 上做 dequant ⇒ 必须两级 cast（无法靠"全程 fp32 算得更准"绕过）。
- **检索/select 类（index-computation.md §5.5）**：精度风险在 **bf16 value 的舍入语义**（RNE 位模拟，index-computation.md §5.5 L1.5）与 **sparse_mode=3 mask 语义**（`j <= i + (as2-as1)`，index-computation.md §5.5 L1.4）。索引本身是整数位序，只要 count/build_masked 的 rank 编码对齐参考的 tie-breaking（eq 按 chunk 前缀补位），索引就精确。失败集若严格对应"fp16/bf16 case"或"某 mask 语义 case" ⇒ 查舍入/mask，不是 tiling。

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

### §7.0 ★★★ 官方 `speedup_vs_torch` 是**几何平均**，优化要盯最慢的用例

`benchmark.py` 的 `speedup_vs_torch = (∏ s_i)^(1/n)`（仅对 pass 且有限正数的 shape），不是总时间比、不是算术平均。**几何平均下把一个 0.5x 修到 1.0x，收益等于把 2x 提到 4x。**

- QuantMatmul opt_iter_2 的收益正来自把最慢的 bf16 pertoken/bias 大 shape（0.33x）拉到 0.47~0.53x，而不是去优化已经 3x 的大 case。
- **操作步骤**：从 `perf_result.json` 的 `per_shape_results` 按 `speedup_vs_torch` 升序排；只看 `<1.0x` 那批找共同特征（典型：中小 shape、带 pertoken/bias、需 clamp）；为这批单独开分派路径，**不要改全局参数**（全局改会同时动到已快的 case，往往净负收益）。

### §7.1 环境闸门与噪声

- torch 参考时延是天然探针，**必须双侧闸门**（异常变快和异常变慢一样会伪造结论）。
- 噪声水平：单次 ±2%；**≤3% 的改动必须用包夹协议（base→A→base）复测**，否则会把噪声当收益。
- profiler 的 `operators` 聚合 ≠ 几何平均（前者大 case 占大头）——"别让小 case 变慢"与"让大 case 变快"同等重要。

### §7.2 benchmark 时延口径（写报告时必须注明）

`benchmark.py` 按 kernel Name 分组，同名 kernel 一次 forward 内发射 L 次只计 1 次（低估 L 倍）。**禁止** 为迎合口径把多次发射合并成同一 Name（纯利用口径缺陷，不改真实性能）。

### §7.3 ⚠️ 历史结论会随编译器 / CANN 版本失效——成功与失败都会

UB 边界（尤其 §2 L1.5 的 128×128+clamp 溢出量）依赖编译器版本；换版本后**全部结构性结论作废，必须重新标定基线并重扫 tile**。精度链结论（CV2 的舍入点）相对稳定，但参考实现算术路径一变也全废。

### §7.4 编译器已经做了的事，不要手工做

手工循环不变量外提、手工 buffer 复用——基本 no-op（甚至有 UB 副作用）。**UB 手算模型预测不了真实边界**（§2 L1.5 的"无 clamp 放得下、有 clamp 溢出"就是手算模型预测反了的例子），边界只能实测撞出来。

### §7.5 检索/select 类：impl 的 `npu_sort_v2` 与 framework 的 `aclnnSort` 是两个不同算子

impl 侧的 `SortV21/SortV22`（`npu_sort_v2` 的 NPU 段）与 framework 侧的 `aclnnSort_SortAiCpu_Sort`（CPU sort 1466µs）**是两个不同的算子**——**不要拿 framework 的 1466µs 当实现侧的墙**。LightningIndexer 的 3.9558x 恰来自这个 40x 结构差。逐 case：大 S2（16384/32768）仍 0.3–1.8ms（第二次 sort 拖累），小/中 case 普遍 2–40x（多头大 shape case34=16.8x、case31=41.4x）。

---

## §8 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 无脑开大 tile 后大批 case 精度崩 / 编译失败 | clamp 物化 2 个全尺寸 fp32 缓冲占爆 UB | §2 L1.5：按 do_clamp 分两条 tile 路径 |
| 失败集 = 全部 fp16 / 全部带 pertoken / 全部 int32 bias case | 算术路径或精度链不匹配 | CV2：逐位复刻参考（手写 sigmoid、round-half-away、fp16 双舍入、int32 bias 先加后乘） |
| `tl.sigmoid` / `tl.round` / int32 `tl.clamp` | 后端偏差或类型不支持 | 手写 sigmoid、round-half-away、int32 用 max/min |
| grid 不收缩导致 task 重复计算 / 超时 | `grid=(total_tiles,)` 在 total>核数时重复 | CV3：`min(核数, total_tiles)` + 核内步长 |
| 跨 NPU 性能跳水 | 硬编码 num_cores | 动态读 cube/vec 核数 |
| 小 case 性能上不去 | 每拆一个 MIX kernel 多 ~4.5us launch | CV6：后处理留 dot epilogue，别拆 kernel |
| 诊断子集 +10%、正式 −3.7% | 扫参子集只有大 case | §7.0：盯最慢的 <1.0x case，分档而非全局改参 |
| per-expert 小专家空转 / 大专家溢出 | 固定 tile 不适应 m_e 跨度 | §3 L1.5：tile 按 m_e 分档 |
| 改了 tile 没同步改 grid | 静默算少 → AccuracyError | CV3：改任务空间必须同步改 grid |
| QuantMatmul 历史代码注释引用 `template/quantmatmul.md` | 该文件曾计划但未建 | 本文件 §2 即其正式归位（见 §9） |
| **精度全过但比历史同架构慢 4x+（重生成/移植后）** | K 归约被"等价改写"成 `acc += tl.dot(out_dtype=...)`：每次迭代物化 `[BM,BN]` 全尺寸 dot 输出，MMAD 链断开；大 tile 档叠加 UB 压力，放大到 22~182x | **CV1**：`acc = tl.dot(a, b, acc)` 累加器融合形式，配合 CV7b 基向量外提，列为生成时静态预检指纹 |
| tile 循环体内出现 `tl.arange`（持久化 kernel） | 基向量未外提，每 tile 每维重算序列生成，累计发射开销可观 | **CV7b**：`offs_*_base` 声明在 tile 循环外，体内只做加法 |
| Phase 4 只扫 BLOCK_K/BLOCK_N 一类参数、性能仍差一个量级 | 根因在**代码形态**（累加器形式/基向量外提）不在 tile 参数，扫参闭合不了 | 先跑 §8 上面的静态指纹预检，再进参数扫描 |
| top-K 选位用 `tl.cumsum` 前缀链（S2 大时） | 标量链（aiv_scalar=0.999），参数扫描救不了 | index-computation.md §5.4 S1 + §5.5 L1.1：sort-compaction 结构 |
| `npu_sort_v2` 喂 1D 布局 | 行为不确定，结果错 | index-computation.md §5.5 L1.2 / §5.4 S2：2D `[rows,S2]` view |
| 用 `torch.topk` / `torch.cumsum` 加速 select | validator AST 禁用 | 用 `npu_sort_v2` + count/build_masked 并行 |
| 打分核调参（tile/流水）攻 select 墙 | select 是结构问题不是参数问题 | index-computation.md §5.4 S3 + §5.5 L1.1：先归零"可消除结构" |

---

## §9 与其他 template 的差异声明

| 场景 | 用哪个文件 | 判据 |
|------|-----------|------|
| 双量化操作数 GEMM（int8@int8）+ scale/pertoken/bias/offset/clamp | **本文件 §2**（`quant-matmul`） | 两个操作数都量化；通用反量化 matmul |
| fp16 x @ int8 weight + antiquant_scale/offset（标准 `npu_weight_quant_batchmatmul`） | `quantization.md §4` | 只有 weight 量化；antiquant 形态 |
| MoE 分组 GEMM + SwiGLU + 动态量化 | **本文件 §3**（`gmm-swiglu-quant`） | per-expert weight + 激活 + 量化三段融合 |
| 单算子量化（DynamicQuant / SwigluQuant 无 GEMM） | `quantization.md §2/§3` | 无 GEMM 主干 |
| 纯 GEMM 调参（无量化/激活） | `matrix-compute-merged.md` | 无量化后处理 |
| attention 主链 | `flash_attention.md` | Q@Kᵀ→softmax→@V |
| 轻量打分 + 重 top-K 选位（score + topk/argmax + gather，无量化链） | **index-computation.md §5.5**（`topk-select`） | 检索/select 融合；select 成本 ≥ 打分成本 |

> 📌 **quantmatmul.md 归位说明**：QuantMatmul 的历史生成代码（`optimized_code.py`）注释中曾引用 `template/quantmatmul.md` 的 L1.1-L1.8，但该文件此前并未实际建立。本卡 §2 即为这些约束的**正式归位**——后续生成 QuantMatmul 类算子时，Phase 2 应定 `category=cv-fusion` 并读取本文件 §2（而非不存在的 quantmatmul.md）。
>
> ✅ **检索/reduce 融合类已入卡**：5_LightningIndexer（`topk-select`）0817 优化达标（54/54，3.9558x vs torch / 12.02x vs baseline），见 **index-computation.md §5.5**。其跨算子共性（`tl.cumsum` 标量化 S1 / `npu_sort_v2` 双角色 S2 / select 结构优先 S3）已提炼进 index-computation.md §5.4； build_masked 向量化、split-sort 供后续算子回填。
