# Attention 家族子类索引（按**写法**分类）

> **本文件是 attention 家族定 `category` 的唯一入口。**
> Phase 2 Step 1 判定为 attention 家族后，**必须以显式路径读取本文件**（`loaded_via: explicit_path`，
> 禁止自动发现），据此定出子类与 `template_path`，再去读该 template 并逐条摘录其 Layer 1。
> `precheck.json` 必须同时记录 `index_path`（本文件）、`writing_class`（下表的大类·细分编号）
> 与最终的 `category` / `template_path`。
> 跳过本文件就拿不到 `category` 与 `template_path`，Step 1 门禁不通过，**无法进入 Step 2**。

## 分类口径：看写法，不看降本手段

判据是**打开 kernel 看代码骨架**——写一个新算子时，相对标准注意力要多写什么、改写什么、还是整个换掉。
**写法相同的算子共用同一套模板与优化知识**，这正是模板复用要的口径。

同一个算法按"降本手段"和按"写法"可能落在不同大类。典型：**一个只在 scores 上加
`masked_fill` 的算子，按手段是稀疏，按写法只是标准三段式**——GEMM 规模一点没变，
与最朴素的三段式共用同一套模板。

基准写法（所有变体的参照系）：

```python
S = Q @ K.transpose(-2, -1) * scale     # GEMM-1
P = softmax(S, dim=-1)                  # 行归约
O = P @ V                               # GEMM-2
```

| 大类 | 相对基准的改动 | 模板可复用性 |
|---|---|---|
| **一 标准类** | 不变，或只加 mask / 改 head 索引 / 改成分块流式 | 同一套模板，参数化即可 |
| **二 压缩类** | K/V **取数前**插一段：head 映射、latent 投影、页表寻址、量化 | 标准模板 + 取数层替换 |
| **三 稀疏类** | KV 循环**之前**插「选哪些块」，循环范围变成子集 | 标准模板 + 选择层 + 循环改写 |
| **四 线性类** | **整个换掉**：无 softmax，改状态递推或结合律重排 | 独立模板，与标准无共用 |
| **五 专用类** | 换目标函数（去归一化）或换数据组织（特征图门控） | 各自独立模板 |
| **六 非注意力** | 不是注意力 | 不走本索引 |

## 判别表

按**从上到下**匹配，**命中即停**。同时命中多行时以先命中者为准（见下方"叠加与优先级"）。

> ⛔⛔ **`category` 只能取下表 `category` 列里出现过的字面值**，即
> `sparse_gather_attention` / `gqa` / `block_sparse_attention` / `flash_attention` / `new_category` 五者之一。
> **严禁自造**——尤其严禁从算子名派生（如看到算子叫 `XxxGqaSparseYyy` 就写成
> `gqa_sparse`、`template_path` 写成 `.claude/template/gqa_sparse.md`）。
> 实测出现过这种自造：`category` 写成一个不存在的值、`template_path` 指向一个不存在的文件，
> 结果 `layer1_constraints_loaded` **一条都没有**，整套 Layer 1 硬约束全部落空，
> 而 Phase 2 Step 3 的合规门无从比对——**看起来流程走完了，实际什么约束都没生效**。
>
> ⛔ **自检（三条都要过，任一不过就是 Step 1 未通过，必须回到本表重新取值）**：
> 1. `category` 的值在下表中**逐字符**出现过；
> 2. `template_path` 指向的文件**真实存在**（拿不准就 `ls` 一下 `.claude/template/`）；
> 3. `layer1_constraints_loaded` **非空**；交叉行（行 1）必须**两份都有**。

| # | 写法特征（打开 kernel 看骨架） | 写法类 | `category` | `template_path` |
|---|---|---|---|---|
| 0 | 选择集合**由输入张量直接给出**（`sparse_indices` / `topk` 张量，无需建表），且索引是**逐 token 的任意行/块 id、不可解析为区间** ⇒ 骨架里必然有一段把选中行**物化成 gather workspace** 再算 attention | 三·4 索引表 gather | **`sparse_gather_attention`** | `.claude/template/sparse_gather_attention.md` |
| 1 | 有块级选择结构（块掩码 / 块索引 / CSR / topK），**且主链有 `kv_head = q_head // group`**（交叉写法） | 三·2 块跳过 ＋ 二·1 ＋ 二·3（叠加） | **`block_sparse_attention`** | `.claude/template/block_sparse_attention.md`（选择表构建与跳块）**＋** `.claude/template/gqa.md`（§3.1 分页 decode 骨架 / §3.4 页内连续 / tile·UB 预算）——⚠️ **交叉行：两份的 Layer 1 都必须摘录**，见下 |
| 1b | 有块级选择结构（块掩码 / CSR / pooled-topk / 行图），**主链不含 GQA 分组**；或参考实现里有 `nonzero()` + 按选中块拼 K/V、`torch.topk` + `scatter_` 成 bool 掩码 | 三·2 块跳过/top-k | **`block_sparse_attention`** | `.claude/template/block_sparse_attention.md` |
| 2 | `phys = block_table[seq][logical]` → `K = cache[phys]`，页与页在显存中不连续，但**不跳块**（整段 KV 都要算） | 二·3 页表寻址 | `gqa` | `.claude/template/gqa.md`（§3.1 / §3.4） |
| 3 | attention 主链被**路由/门控**选择性执行（`argmax` + `one_hot` 或等价），多路权重按样本选 | 二·1 head 共享（门控变体） | **`gqa`** | `.claude/template/gqa.md`（§3.3 gqa-router） |
| 4 | 取数层只有 `kv_head = q_head // group` 这一处改动，KV 是连续区间 | 二·1 head 共享 | **`gqa`** | `.claude/template/gqa.md`（§3.2 稠密骨架）**＋** `flash_attention.md` |
| 5 | `kv_c = W_dkv @ h` 压缩存储、用时 `K = W_uk @ kv_c` 展开，rope 维单独一路 | 二·2 latent 投影 | `flash_attention` | `.claude/template/flash_attention.md`（**该细分暂无专属 template**） |
| 5b | K/V 取数前插一段**可学习 `[P,S]` 低秩投影**把序列维 `S→P` 降维（Linformer），主链仍是 `softmax(Q@Kᵣᵀ/√d)@Vᵣ` 三段式，但 attention 在 `[B,H,P,D]`（P≪S）上做，不对全 S 滚动 online softmax | 二·2 低秩投影（Linear 类） | `new_category` | `.claude/template/linear.md`（本类专属 template，含 L1.1~L1.8 硬约束；**禁止**回落到 `flash_attention.md`——无 softmax 滚动，FA 的 online softmax 约束整套不适用） |
| 6 | `q8, sq = quant(Q)`；`s = int8_gemm(q8,k8) * sq * sk` | 二·4 量化 | `flash_attention` | `.claude/template/flash_attention.md`（**暂无量化专属**） |
| 6a | 参考实现含 **4 个 `nn.Linear` 投影**（Q/K/V/O），主链是三段式或分块流式 `softmax(Q@Kᵀ/√d)@V`（self/cross/MHA 同构） | 一·1 标准类·含投影段 | **`mha`** | `.claude/template/mha.md`（投影段终态架构 + 实证）**＋** `.claude/template/flash_attention.md`（attention 主链）——⚠️ **交叉行：两份的 Layer 1 都必须摘录** |
| 7 | kernel 内对 KV 分块循环、跨迭代滚动 `m`/`l`/`acc`；或带 `causal`/`window`/`softcap` mask；或 `S` 大到不宜物化 `[S_Q,S_K]` | 一·2 分块流式（FA 系） | **`flash_attention`** | `.claude/template/flash_attention.md` |
| 8 | `S = S.masked_fill(mask, -inf)` 是唯一改动，**GEMM 规模不变** | 三·1 掩码填充 | **`block_sparse_attention`** | `.claude/template/block_sparse_attention.md`（⚠️ 见下节：标杆是掩码填充，生成目标必须是块跳过） |
| 9 | `kv = F.unfold(x, kernel=blk+2*halo, stride=blk)` 构造 KV | 三·3 邻域展开 | `flash_attention` | `.claude/template/flash_attention.md` |
| 10 | `x.flatten(2).transpose(1,2)` 把 `(B,C,H,W)` 拉成 token 再走三段式 | 一·3 空间 token 版 | `flash_attention` | `.claude/template/flash_attention.md` |
| 11 | 纯基准三段式，不含 KV 分块 | 一·1 基础三段式 | `flash_attention` | `.claude/template/flash_attention.md` |
| 12 | **无 softmax**，改为状态递推（`state = a*state + k⊗v`）或结合律重排 | 四 线性类 | `new_category` | **无可复用 template**，见下 |
| 13 | 去归一化（无 softmax 但保留 QK/PV 结构）、或特征图门控（CV 通道/空间/双分支） | 五 专用类 | `new_category` | **无可复用 template**，见下 |

## ⚠️ 最重要的一条警示：掩码填充是 golden，不是生成目标

**三·1 掩码填充型的 torch 标杆用 `masked_fill` 表达，最直白、便于校验——
但生成的 kernel 必须改写成三·2 的块跳过写法，否则等于白做稀疏。**

两种写法**数学等价、性能天差地别**。这类算子的头号问题不是稀疏本身，而是
**「稀疏被写成了逐 token 的 program」**——掩码是块粒度的（通常 128×128），program 却按 token 发，
同一掩码块里的 128 个 query 每人把整块 K/V 搬一遍，访存放大 128 倍，同时 `M=1` 让 Cube 完全闲置。
**第一优化动作永远是把 program 粒度提到掩码块粒度**，实测 **26.25×**（几何平均，
见 `block_sparse_attention.md`）。

## 叠加与优先级

- **行 0 排在行 1 之前**是刻意的。行 0 同时满足行 1 的两个条件（有块级选择结构 ＋
  主链有 `kv_head = q_head // group`），但行 1 的两条立论依据在行 0 上**都不成立**：
  1. 行 1 的头号红线是「选择表必须 device 侧建」（实测 host 建表虚高 **2.09×**）。
     行 0 的选择集合是**入参张量直接给的**，压根没有表要建，这条红线空转。
  2. 行 1 的成本模型是「`tl.dot` 的**个数** × KV 页迭代数」。行 0 实测
     **gather 阶段 0 FLOPs 却占 50.5% 时延、整算子 MFU 仅 4.29%**
     （`7_SparseFlashAttention` case 9 逐 kernel 分解，见该 template §1 G3），
     目标函数是「**GM 搬运字节数**」；按 dot 个数优化收益是个位数百分比。

  行 0 独有、另两份都不覆盖的硬约束：**同地址 gather 的硬件串行化**——
  索引退化成常量时必须检测并退化为单行广播，实测关掉后 gather **0.02ms→0.53ms（24×）**。
  且本类测试数据有 **30/72 用例索引恒为 0** 的退化陷阱（生成器按 BSND 取 `shape[1]`，
  遇 TND 变成 head 数），不知道这点会把退化用例的数字误当成布局差异。

  判别抓手：**选择集合是"给定的索引张量"还是"要自己从掩码/打分推出来的表"**——
  前者走行 0，后者走行 1/1b。

- **行 1 与行 1b 的分界是"主链有没有 `kv_head = q_head // group`"**。两者都属三·2，
  但行 1 是交叉写法：选择层与取数层替换同时存在，**主 `category` 取块跳过**（循环范围改写会
  重塑整个 kernel 骨架，取数层替换只是几行索引），GQA 侧的任务划分、页内布局与 tile/UB 预算
  由 `gqa.md` 补齐。
- ⚠️ **交叉行（行 1）的 `layer1_constraints_loaded` 必须同时摘录两份卡片的 Layer 1**，
  `template_path` 记两条。理由是实测出来的机制：**只有被摘录进 `precheck.json` 的 Layer 1
  才受 Phase 2 Step 3 的合规检查门管辖**——同一条规则写在 Layer 2 连续 5 轮无效、
  原样移进 Layer 1 后一轮就生效。只摘一份的话，另一份的硬约束等于没写。
  两份约束若有冲突，以 `category` 对应那份（`block_sparse_attention.md`）为准，
  并把冲突点连同实测差异补进本索引。
- **压缩类的四种插法可叠加**：latent 投影 + 页表寻址是常见组合（先压成 latent 再分页存）；
  原则上还能再叠加量化。唯一的组合约束是 **head 共享与 latent 投影二选一**——
  两者都在改 KV 的组织方式，冲突。
- **行 1/1b/2 排在行 7 之前**是刻意的。分页/块稀疏同时满足两行条件，但
  `flash_attention.md` 的主力手段「放大 tile 摊薄 CV 同步」在分页形态上**会直接撞 UB 墙**
  （实测多页合并三种写法全部 `PlanMemory Failed`），且「`p` 二段拆分」在 FA 上是 **+2.7%**、
  在分页 GQA 上是 **−49%**。两类的目标函数不同：FA 是「互有 UB 依赖的向量算子数 × KV 迭代数」，
  分页 GQA 是「`tl.dot` 的**个数** × KV 页迭代数」。
- **行 1 / 1b 是「三·2 块跳过」小类，走 `block_sparse_attention.md`；`gqa.md` 负责的是
  「二 压缩类·head 共享 / 页表寻址 / 路由」小类。** 分工依据见 `level4写法分类体系.md`：
  块跳过与 head 共享是两个并列小类；**同时**具备选择层与取数层替换的交叉写法归**稀疏·块跳过**
  （循环范围改写会重塑整个 kernel 骨架，取数层替换只是几行索引），所以行 1 以前者为主。
  按本文件"新增子类必须给出实测差异"的规矩，块跳过**不能**直接并进 `gqa.md` 的 Layer 1，两条依据：
  1. **本小类的选择集合是从「掩码语义」推导的，不是从数据打分 top-k 得来**——
     入参是块级掩码 / 每头类型码 / streaming 参数 / `cu_seqlens` + `causal`，
     参考实现用 CPU 上可展开的稠密掩码表达它，于是最容易犯的错是**把建表搬到 host**。
     实测同一算子两种写法的口径差 **2.09×**（host 建表 7842 vs device 建表 3758），
     前者是不合规的虚高。这条红线（`L1.2`）与配套禁令（`L1.7` 禁止复制参考实现的 mask helper）
     `gqa.md` 不覆盖。
  2. **本小类含反向写法**（入参给 `softmax_max` / `softmax_sum` / 前向输出，出 dq/dk/dv）：
     需要 q 主序与 k 主序**两张**选择表、三 kernel（delta / dq / dkdv）结构，两张表都必须 device 侧建。
     `gqa.md` 的骨架是前向 / decode 形态，没有对应 Layer 2。
- **行 4（稠密 GQA）两个文件都读**，`gqa.md` 的 §2/§5 优先——它的 §3.2 是稠密骨架，
  而 FA 的分块流式经验同样适用。行 1 同理，两份都读：`gqa.md` 给 GQA 主链与 tile/UB 预算，`block_sparse_attention.md` 给选择表构建。
- **行 6a（含投影段）排在行 7 之前**是刻意的：含 4 个投影的 MHA 形态同时满足行 7 的
  分块流式特征，但实测**最大的一笔收益在投影侧**（CrossAttention 权重预转置 -15%，
  见 `mha.md` §4.2）——若先命中行 7，首次生成只拿 FA 卡的 attention 侧约束，
  投影段头号杠杆（`flash_attention.md` L1.12）要靠 Phase 4 十几轮 rediscover
  （误分类事故见 `mha.md` §4.5）。行 6a 为交叉行：投影段看 `mha.md`，
  attention 主链 Layer 1 仍由 `flash_attention.md` 提供，两份都摘录。
- **一 标准类的三个细分（基础三段式 / 分块流式 / 空间 token 版）统一走 `flash_attention.md`；**
  **含 4 个 `nn.Linear` 投影的变体（MHA / SelfAttention / CrossAttention）走行 6a 的
  `mha.md` ＋ `flash_attention.md` 交叉行。**
  依据是分类体系里这一大类的判断：「同一套模板，参数化即可」。原先另有一份 `attention.md`
  承接朴素三段式，已退役——它的 L1.1（禁用 fp32 `tl.dot`）/ L1.3（`next_pow2`）/
  L1.6（全程 fp32）三条在 FA 路径上已被实测推翻，两份并存时最常见的故障就是
  朴素卡片的约束被套到 FA 算子上，让 Step 3 门禁把正确草图判为 A 类错误。

## 无 template 的子类怎么办

行 5/6（latent 投影、量化）与行 12/13（线性类、专用类）尚无专属 template。
按 Step 1 的既有机制处理：
标 `new_category`，先回落到表中给出的 `template_path`（没有则不加载 Layer 1 约束），
**草图通过后新建 `.claude/template/{category}.md` 并回填 Layer 1**。

⚠️ 线性类（四）与专用类（五）**不要**回落到 `flash_attention.md`：它们没有 softmax，
FA 的 online softmax 约束整套不适用，误用会让 Step 3 门禁把正确草图判为 A 类错误。

## 新增子类的规矩

再出现新的 attention 变体时，**在本表加一行，不要回头去改 `AGENTS.md`**——
`AGENTS.md` 只保留"attention 家族 → 读本文件"这一跳。

新增行必须同时给出：可模式匹配的**写法**特征（打开 kernel 能对上的骨架，不是降本手段的名字）、
所属大类·细分、`category`、`template_path`，以及**它为什么不能落到已有的某一行**（附实测差异，
像上面行 1/1b vs 行 7 那样）。拿不出实测差异就不要新开子类——加进已有 template 的 Layer 1 更可靠。
