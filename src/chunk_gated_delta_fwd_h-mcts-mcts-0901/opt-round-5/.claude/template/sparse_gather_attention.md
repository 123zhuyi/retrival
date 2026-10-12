---
schema_version: okf.v1
kind: template
type: operator-optimization
source_family: attention
category: sparse_gather_attention
tags: [attention, sparse, gather, mla, rope, ascend910b, triton]
---

# 三·4 · 索引表 gather 型稀疏 attention 优化经验

> **适用判据（一句话）**：KV 侧不是连续区间、也不是"跳过整块"，而是**按一张逐 token 的索引表
> `sparse_indices` 把任意行/块 gather 出来**，再在这份 gather 结果上做标准三段式。
> 骨架里必然出现一个**"取数物化"阶段**——这是本类与行 1/1b（块跳过/top-k）最大的写法差异。

代表算子：`7_SparseFlashAttention`（NSA/DSA 式，MLA 维度 + 逐 token top-k 索引）。

---

## §0 适用范围与形态识别

### §0.1 判别特征（决定用不用本文件）

打开参考实现，命中**全部三条**才用本文件：

1. KV 取数前有一段 **`for sparse_id in topk: gather rows`**（或等价的 `index_select` /
   高级索引），索引**来自输入张量**而非从掩码/打分推出来的表；
2. 索引是**逐 (token, kv_head) 一份**，不同 token 的选择集互不相同 ⇒ 无法提到循环外复用；
3. 选中集合**不小**（本类典型 `sel_rows / S2 ≈ 25%`），所以"稀疏"省下的是 GEMM 规模，
   但**新引入的 gather 搬运可能比省下的 GEMM 还贵**。

⚠️ 若选择集合需要自己从块掩码 / 打分 top-k **推导建表** ⇒ 回行 1/1b
（`block_sparse_attention.md`），不要用本文件——那边的头号红线「选择表必须 device 侧建」
在本类是空转的（本类没有表要建）。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答）

| # | 问题 | 为什么决定后续 |
|---|---|---|
| 1 | `G = n1 // n2` 是多少？ | G 大（本类实测 **G=128**）⇒ 一份 gather 结果被 G 个 q head 共用，**gather 成本按 G 摊薄**，任务划分必须整组 G 一起做 |
| 2 | `D_main` / `D_rope` / `D_v` 各是多少？rope 是**拼接**还是**旋转**？ | 本类是 **MLA 式拼接**（`q_fp = cat([q, q_rope], -1)`），不是 RoPE 旋转。拼接维要**分离两路**传，禁止 host 侧 `cat`（见 L1.4） |
| 3 | `sparse_block_size` 取值集合？ | `bs==1` 是"逐行 gather"，`bs>1` 是"逐块 gather"。两者的 span/dedup 策略不同（见 L1.3） |
| 4 | layout 有几种？（`BSND` / `TND` / `PA_BSND`） | 决定 host 侧的展平方式；**`n2==1` 时 BSND 可退化为纯 view，禁止真拷贝**（见 L1.4） |
| 5 | `sparse_indices` 的取值范围是多少？**有没有退化成常量？** | ★★★ 见 §1 G2——本类实测有 **30/72 用例索引恒为 0**，是完全不同的性能形态 |

---

## §1 通用经验

### G1 ★ 环境闸门必须前置

非交互 shell 读不到 CANN 的 `set_env.sh`，会静默落回旧 CANN + 旧 bishengir，
**编译相关结论整片作废**（实测：同一份代码在 bishengir 0.1.0 下只有 32×32 能编过，
1.2.0 下 128×128 全通）。所有 verify / benchmark / probe 之前先 source 统一 env。

### G2 ★★★ 索引表可能退化成常量，必须先查再优化

本类算子的测试数据由 `get_input_groups()` 生成，索引范围来自
`max_idx = key_info["shape"][1] // sparse_block_size`。
**这一行按 `BSND` 的 `[B,S2,N,D]` 写死取 `shape[1]`**，遇到 `TND` 的 `[T,N,D]` 时
`shape[1]` 是 head 数而不是序列长 ⇒ `max_idx=1` ⇒ **索引恒为 0**。

`7_SparseFlashAttention` 实测分布：

| `layout_kv` | 用例数 | 索引 |
|---|---|---|
| BSND | 40 | 真实随机 |
| PA_BSND | 2 | 真实随机 |
| **TND** | **30** | **恒为 0（退化）** |

后果有两个方向，都必须知道：

- **不要拿退化用例评估 gather 优化**。实测 case 9(BSND) 与 case 50(TND)
  **参数完全相同**（t1=128, n1=128, topk=2048, bs=1, d=512, rope=64），
  gather 耗时却是 **2429.7us vs 466.0us（5.2×）**——差异全部来自索引退化，
  与 layout 无关。曾据此误判为"BSND 布局慢 2.33 倍"，是错的。
- **必须为退化形态留快速路径**。同地址 gather 会被硬件串行化，见 L1.3。

### G3 ★★★ 本类的成本单位是 **GM 搬运字节数**，不是 `tl.dot` 的个数

这是本类与行 1/1b 最重要的分歧。那边的成本模型是「`tl.dot` 的个数 × KV 页迭代数」；
本类实测**完全不适用**：

`7_SparseFlashAttention` case 9（`[1,128,128,512]` bf16，t1=128，sel_rows=2048）逐 kernel 分解：

| kernel | 耗时 | 占比 | FLOPs |
|---|---|---|---|
| `gather_kv`（物化选中 K/KR/V 到 workspace） | **2429.7us** | **50.5%** | **0** |
| `spec_attn`（真正的 QK^T + PV） | 1948.2us | 40.5% | 73.0 GFLOP |
| `split_combine`（split-KV 合并） | 351.2us | 7.3% | **0** |
| `aclnnInplaceCopy_Transpose`（host 侧布局拷贝） | 78.0us | 1.6% | **0** |

**整算子 MFU 仅 4.29%**（72 例平均 7.16%，最高 9.99%，cube 峰值 353.89 TFLOPS 口径）。
即**约六成时间花在 0 FLOPs 的搬运上**，cube 基本闲着。

⇒ **优化顺序必须是：先砍搬运字节，再谈 tile 与流水。**
反过来（先调 tile 大小、加 multibuffer）在本类上收益都是个位数百分比。

按带宽记账（case 9）：每 token 需搬 K `2048×576×2B` + V `2048×512×2B` ≈ 4.46 MB，
×t1=128 ≈ 571 MB；写 workspace 再读回 ⇒ 约 1.14 GB。
1.4 TB/s 下理论 ~815us，实测 2430us ⇒ **仅达带宽上限的 1/3**，这是首要缺口。

### G4 测量口径

**标定点**：`7_SparseFlashAttention` 的人工优化合规版在官方口径下实测
**`speedup_vs_torch = 6.4699`，72/72 全过，无一例 < 1.0x**（最低 1.156、最高 19.186；
CANN 9.1.0 / bishengir 1.2.0 / triton_ascend 3.2.2，NPU 独占，warmup=5 repeats=5）。
该实现的逐 kernel 分解就是 §1 G3 那张表——**拿到 6.47x 的实现，MFU 依然只有 4.29%**，
说明本类的天花板远没到，瓶颈始终在搬运侧。

**卡片驱动生成的实测标定点**（用于判断自己这一轮是否正常）：

**⚠️ 比较前先核对 `warmup` / `repeats`**（`perf_result.json` 里有这两个字段）。
下表**统一为 `warmup=5 / repeats=5`**；跨口径比较会得出错误倍数（曾据此误报过一次）。

| 版本 | 具备的优化 | kernel 数 | speedup |
|---|---|---|---|
| 早期（无 L1.3 / 无 L1.5） | 仅行维向量化 | 2~3 | **0.302** |
| **本卡片完整版** | span-select + 输入 dtype + 宽载 + 布局对齐 + 形态路由 | **3** | **1.4379 / 1.4187 / 1.4166**<br>（三次独立生成，稳定 ≈**1.42**） |
| 人工优化达标版 | 上述全部 **＋ spec 特化 ＋ split-KV** | **5** | **6.4699** |

⇒ **卡片的局部优化合计约 4.7 倍**（0.302 → 1.42，三次交叉验证）。
⇒ **从 1.42 到 6.47 的差距在 §3.2 的结构性分派**（spec 特化 + split-KV），不在 tile 调参。
   **只写出 3 个 kernel 就到不了 2.0x**——九轮生成无例外。

⇒ **首版性能基本即最终性能**：Phase 4 自主优化的净收益实测为零
（两轮独立样本：`0.9938→0.9915`、`1.4187→1.4166`，均 −0.2% 上下）。
**不要指望 Phase 4 补救首版的不足。**

- 一律用 triton-op-generator 内部的官方 `verify.py` / `benchmark.py` 判定，
  自建脚本只能用于调试，**不能用于出结论**。
- **看 speedup 前先看 `passed_cases == total_cases`**。编译失败的 case 不进几何平均，
  会让数字**变好看而不是变差**（同类算子实测：12 个 case 编译失败、只剩 38 个偏小 case
  进平均，得出虚高的 10.13x）。
- 总分是**逐 case speedup 的几何平均**，最慢的用例杠杆最大。
- 参考实现是重 Python 循环，单轮 verify 要跑满全部用例，**小时级**，超时要放到 2 万秒以上。
- framework 侧跑的是一字未改的参考实现，**它的时延就是天然环境探针**：
  同一形态三次测量漂出 ±25% 就该怀疑测量环境（机器 load / NPU 未独占），而不是改代码。

### G5 ⛔⛔ `validate_triton_impl.py` 静态闸门：**过不了它连编译都进不去**

verify 之前有一道 AST 静态校验，**规则是白名单制**。实测两轮生成各在这里损失约 15 分钟，
且它会逼你改结构（不是改几行）。**写草图时就要按它的规则写。**

| 类别 | 规则 |
|---|---|
| **Python 控制流** | **仅 `forward()` 函数体自身**禁 `for` / `while` 与 `list.append`；**被 `forward()` 调用的 helper 里可以用**（实测：达标实现在 `_build_meta` / `_sfa_run` 里有 For/While，`valid=True`）。`if`/`elif`/`else` 处处允许 |
| **可达性** | **跨函数分析只作用于「借 PyTorch 算主链」这一类**（Type 3）：把 `torch.matmul` 之类挪进 helper 绕不过去。**循环限制不跨函数**——见上一行 |
| **torch 函数白名单** | 仅 `empty / empty_like / empty_strided / zeros(_like) / ones(_like) / full(_like) / tensor / arange / linspace / as_tensor`。**不在名单上的一律拦**——`torch.cumsum` 就是这样被拦的 |
| **tensor 方法白名单** | 形状/布局/dtype 类（`view/reshape/permute/transpose/expand/contiguous/to/...`）+ `item` / `tolist` / `copy_` / `fill_` / `zero_` |
| **tensor 方法黑名单** | `sum/mean/max/min/matmul/mm/bmm/exp/log/sqrt/softmax/...` 等一切计算 |
| **triton 属性** | 仅 `triton.cdiv`、`triton.next_power_of_2` |
| **内置 / math** | `range/len/min/max/int/float/str/enumerate/zip` + `math.prod/ceil/floor/log2/pow` |

⇒ **host 侧 meta 只能用纯 Python 标量运算 + 白名单构造函数**。
需要前缀和之类的量，用 Python 列表算好再 `torch.tensor(..., device=..., dtype=...)` 传进去，
**不要用 `torch.cumsum`**。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

> **本节共 15 条（L1.1、L1.1b、L1.2 ~ L1.14），全部是 Phase 2 的硬性边界。**
> `precheck.json` 必须逐条载入全部 15 条，不得截断。
>
> ⚠️ **先读 L1.14（kernel 签名规格）再写代码**——它定义了其它条款里
> `NUM_DM` / `NEG` / `BLOCK_D_*` 这些标识符的出处。跳过它会连撞两类编译错误。
>
> ⚠️ **L1.13 是当前失败率最高的一条**——三轮真实生成各有 62~72 个用例死在它上面，
> 且它此前只写在 §3.1 的示例里、没进本清单，结果三轮都没被遵守。**不要跳过。**
>
> ⚠️⚠️ **L1.3 / L1.5 是性能的生死线，L1.8 / L1.9 是编译的生死线——四条最容易漏**——实测一次真实生成里，
> `precheck.json` 记录了 8/8 条已加载，但首版代码里 L1.3（单行广播）、L1.5（workspace 输入 dtype）
> 一处都没有，且 L1.8/L1.9 两个语法陷阱各吃掉一轮完整 verify。**写完草图后逐条自查一遍。**

### L1.1 ★★★ 任务单元 = (token 组, kv_head)，**整组 G 个 q head 一起做**

一份 gather 出来的 KV 被该 (token, kv_head) 下的 **全部 G 个 q head 共用**。
若按单个 q head 划分任务，同一份 KV 会被重复 gather G 次。

```python
# ❌ G 个 q head 各自 gather 一遍同一份 KV
grid = (t1 * n1, ...)
# ✅ 一个 program 覆盖整组 G，q tile 是 [BLOCK_G, D]
grid = (t1 * n2, cdiv(G, BLOCK_G), ...)          # t1 全量进 grid（L1.2：无 host 循环）
```

G=128 时这是 **128 倍**的搬运差异，是本类第一位的结构性约束。

### L1.1b ★★★★ gather kernel 的**行维必须向量化**，禁止逐索引标量循环

这是本类最容易写错、代价最大的一条。**必须**让一个 program 一次处理一整块行：

```python
# host
grid = (t1 * n2, triton.cdiv(sel_rows, BLOCK_N))       # 行维进 grid，BLOCK_N=64/128
# kernel
pid_t = tl.program_id(0); pid_n = tl.program_id(1)
n_offs   = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # ✅ 一次 BLOCK_N 行
row_in   = n_offs < SEL_ROWS
block_idx = n_offs // BS
offset    = n_offs % BS
idx = tl.load(IDX + (t1g * N2 + n2) * TOPK + block_idx, mask=row_in, other=-1)  # 向量 load
src = kv_start + idx * BS + offset
kc  = tl.load(K + src[:, None] * (N2 * D_MAIN) + n2 * D_MAIN + d_cols[None, :], ...)
```

⛔ **绝对禁止**这种写法：

```python
for k_idx in range(0, TOPK_LEN):                       # ❌ TOPK_LEN=2048 ⇒ 串行 2048 轮
    idx = tl.load(base_idx + k_idx * stride_idx_k)     # ❌ 标量 load，一次只取 1 个索引
    src_rows = kv_start + idx * BS + offs_b            # ❌ 一次只处理 BS 行（bs=1 时 1 行）
    for d0 in range(0, D_MAIN, BLOCK_D_MAIN):
        ...
```

**实测代价**（同算子、同模型、同流程，唯一变量是 gather 的行维写法）：

| 版本 | gather 写法 | 精度 | `speedup_vs_torch` |
|---|---|---|---|
| A | `for k_idx in range(2048)` 逐索引标量循环 | 72/72 | **0.0876** |
| B | 行维向量化（`BLOCK_N` + 行维进 grid） | 72/72 | **0.2059** |
| C | 向量化 ＋ §4.2 全部基因（人工优化版） | 72/72 | **6.4699** |

⇒ **本条本身值 2.35×**（A→B，0.0876→0.2059）。
⚠️ 不要把 A 与 C 的 74 倍差距全记在本条头上——**剩下的约 31 倍来自 L1.3 / L1.5
与 §4.2 的基因组合**（`G_BF16DOT`、`BDG128`、`SPEC_ROW/DV`、multibuffer、
split-KV combine、定向 `BLOCK_N` 等）。本条是**必要前提而非充分条件**：
不做它，后面那些优化无处安放；只做它，也只能到 0.2 量级。

⚠️ **顺序很重要**：L1.3（同地址单行广播）是在**已经向量化的 tile 内**做去重，
行维都没向量化的话，L1.3 与 L1.5 都无从谈起，写再详细的配方也落不了地。
**先满足本条，再谈 L1.3 / L1.5。**

### L1.2 ★★★ `forward()` **函数体自身**不许有 Python `for`/`while`——但 **helper 里可以**

**闸门只查 `forward()` 本体**（见 G5）。⚠️ **本条曾被写成"helper 也禁"，是错的**：
达标实现（6.4699x）在 `_build_meta` / `_sfa_run` 两个 forward 可达的 helper 里
都有 `for`/`while`，静态闸门判 `valid=True`。

⇒ **正确做法：把 host 侧的 chunk 循环放进 helper**，`forward()` 只负责调用：

```python
def forward(self, query, key, value, sparse_indices, ...):
    ...                                     # 只有直线代码 + if 分支
    return _sfa_run(q, k, v, idx, ...)      # ← 循环写在 _sfa_run 里

def _sfa_run(...):                          # helper：可以用 for/while
    BLOCK_T1 = 16
    for t1_start in range(0, t1, BLOCK_T1):         # ✅ 合规
        t1c  = min(BLOCK_T1, t1 - t1_start)
        k_ws = torch.empty((t1c * n2 * sel_rows, d_main), dtype=k.dtype, device=device)
        ...                                          # workspace 按 chunk 复用
```

**这条很重要**：分 chunk 能把 gather workspace 压到 **1/8**（`BLOCK_T1=16` vs `t1=128`），
达标实现正是这么做的。一次性开全量虽然也能跑（实测不 OOM），但放弃了这个手段。

```python
# ❌ 被闸门拦下：forward() 及其可达 helper 内禁止 Python for/while
BLOCK_T1 = 16
for t1_start in range(0, t1, BLOCK_T1):
    t1c = min(BLOCK_T1, t1 - t1_start)
    k_ws = torch.empty((t1c * n2 * sel_rows, d_main), dtype=k.dtype, device=device)

# ✅ 一次性开全量，grid 覆盖所有 token，chunk 语义交给 kernel 内的 program id
rows_total = t1 * n2 * sel_rows
k_ws  = torch.empty((rows_total, d_main), dtype=k.dtype, device=device)   # L1.5 输入 dtype
kr_ws = torch.empty((rows_total, dr_eff), dtype=k.dtype, device=device)
v_ws  = torch.empty((rows_total, d_v),    dtype=v.dtype, device=device)
gather_kv_kernel[(t1 * n2, triton.cdiv(sel_rows, BLOCK_N))](...)          # 只 launch 一次
```

**显存不是问题**：全量 workspace 最大约 571 MB（t1=128、sel_rows=2048、D≈1088、bf16），
910B 上实测可分配，**不会 OOM**——早期版本写"必须分 chunk 否则爆显存"是按字节数推断的，
已被实测推翻，不要再据此加 host 循环。

⚠️ 真正要防的是 **grid 并行度饥饿**：`t1==1` 的小形态把 `BLOCK_N` 从 64 提到 128
会退化约 **+20%**（gather program 数 32→16）。这靠 §3.2 的分档判据解决，不靠 host 分块。

#### 一致性要求：token 维进 grid 后，gather 写侧与 attn 读侧必须同一条行映射

```python
ws_row = (t * n2 + n) * sel_rows + row      # 两个 kernel 用同一条公式
```
写侧若用扁平的 `program_id(0)`（grid dim0 = `t1*n2`）、读侧用分开的 `pid_t / pid_n2`，
两者只在 `n2 == 1` 时等价 —— `n2 > 1` 的用例会整片错位。

### L1.3 ★★★ 同地址 gather 会被硬件串行化，必须检测并退化为**单行广播**

一个 tile 内若多行的源地址相同（G2 的退化形态就是全部相同），逐行 gather 会被硬件串行化。
必须在 kernel 内检测 tile 内地址跨度，跨度足够小时**只加载窗口内的几行再等值选择广播**：

```python
# ⚠️ 变量名带路径后缀（_main / _rope / _v）：三条路径 tile 宽不同，
#    共用一个名字 ⇒ 72/72 编译失败，见 §3.1 末尾的 ⛔⛔
src_lo = tl.min(src_safe)
src_hi = tl.max(src_safe)
span_ok = (src_hi - src_lo) < (BS if BS <= 4 else 4)   # 窗口钳制，见下方 ⚠️
if span_ok:
    if BS == 1:                      # constexpr 守卫：窗口必坍缩为单行
        r0 = tl.load(K + src_lo * stride + d_cols_m, mask=dm_mask, other=0.0)
        kc_main = tl.broadcast_to(r0[None, :], (BLOCK_N, BLOCK_D_MAIN))  # 纯广播，无 select 链
    else:
        # 逐个候选行 load + tl.where 等值选择；每个分支都必须产出
        # 形状同为 (BLOCK_N, BLOCK_D_MAIN) 的 kc_main，否则类型不一致
        acc_row = tl.load(K + src_lo * stride + d_cols_m, mask=dm_mask, other=0.0)
        kc_main = tl.broadcast_to(acc_row[None, :], (BLOCK_N, BLOCK_D_MAIN))
        for w in tl.static_range(1, 4):        # 窗口上界是 constexpr（L1.11）
            rw  = tl.load(K + (src_lo + w) * stride + d_cols_m, mask=dm_mask, other=0.0)
            hit = (src_safe == (src_lo + w))[:, None]
            kc_main = tl.where(hit, tl.broadcast_to(rw[None, :], (BLOCK_N, BLOCK_D_MAIN)), kc_main)
else:
    kc_main = tl.load(K + src_safe[:, None] * stride + d_cols_m[None, :],
                      mask=row_in[:, None] & dm_mask[None, :], other=0.0)   # 回退逐行 gather
```

⚠️ **`if` 的每个分支都必须给 `kc_main` 赋同样形状的值**——这是上面那条类型一致性要求的直接推论。
原先此处写 `...  # 逐个候选行 load + tl.where 等值选择` 留白，实测被补成了形状不一致的版本。

实测代价（关掉 span-select 的消融）：退化形态 gather **0.02ms → 0.53ms，24 倍串行化退化**。

**完整可粘贴实现**（放在 gather kernel 里，K/KR/V 三段同改）：

```python
idx   = tl.load(IDX + (t1g * N2 + n2) * TOPK + block_idx, mask=row_in, other=-1)
src   = kv_start + idx * BS + offset
valid = row_in & (idx != -1) & (src >= kv_start) & (src < kv_end)
src_safe = tl.where(valid, src, kv_start)          # 地址钳到界内，禁止越界

src_lo = tl.min(src_safe)
src_hi = tl.max(src_safe)
span_ok = (src_hi - src_lo) < (BS if BS <= 4 else 4)

offs_d = tl.arange(0, BLOCK_D)
for dk in range(NUM_DM):
    d_cols = dk * BLOCK_D + offs_d
    d_mask = d_cols < D_MAIN
    if span_ok:
        if BS == 1:                                 # constexpr 守卫：窗口必坍缩为单行
            r0 = tl.load(K + src_lo * (N2 * D_MAIN) + n2 * D_MAIN + d_cols,
                         mask=d_mask, other=0.0)
            kc = tl.broadcast_to(r0[None, :], (BLOCK_N, BLOCK_D))   # 纯广播，无 select 链
        else:
            kc = tl.zeros((BLOCK_N, BLOCK_D), dtype=K_WS.dtype.element_ty)
            for i in tl.static_range(BS):           # 逐候选行等值选择
                ai = tl.minimum(src_lo + i, src_hi)
                ri = tl.load(K + ai * (N2 * D_MAIN) + n2 * D_MAIN + d_cols,
                             mask=d_mask, other=0.0)
                kc = tl.where(((src_safe == src_lo + i) & valid)[:, None],
                              tl.broadcast_to(ri[None, :], (BLOCK_N, BLOCK_D)), kc)
    else:                                           # 随机索引 ⇒ 回退逐行 gather
        kc = tl.load(K + src_safe[:, None] * (N2 * D_MAIN) + n2 * D_MAIN + d_cols[None, :],
                     mask=valid[:, None] & d_mask[None, :], other=0.0)
    tl.store(K_WS + ws_row[:, None] * D_MAIN + d_cols[None, :], kc,
             mask=row_in[:, None] & d_mask[None, :])
```

⛔ **修精度问题时不要顺手把这段删掉**。实测出现过：为修 NaN 重写 gather，把 span-select 一并移除，
精度确实修好了（72/72），但性能塌到 **0.0876x**。**精度与这段逻辑无关**——它只改取数方式，
数值逐位一致。

⚠️ 三个必须遵守的细节：

1. **窗口钳制到 `min(BS, 4)`**：`BS>4` 时若不钳制会出精度 bug；钳制后 `BS<=4` 与原条件逐位等价。
2. **`BS==1` 走 constexpr 单行坍缩快速路径**，不要写成运行时 `all_same` 的嵌套 `if`——
   实测运行时嵌套分支会干扰 scattered 路径的调度，使 `bs∈{2,4}` 用例**回退 10~14%**。
3. `span_ok` 对随机索引恒为 False，会自动回退 scattered 路径，两种形态兼得；
   **不要为了简化而直接关掉 span-select**（该假设已被退化形态实测证伪）。

### L1.4 ★★ host 侧禁止真拷贝：rope 分离两路传，`n2==1` 的 BSND 展平必须是纯 view

参考实现写的是 `q_fp = torch.cat([q_fp, q_rope], dim=-1)`，**照抄就是一次全量物化**。

```python
# ❌ host 侧拼接主维与 rope 维
q_all = torch.cat([q, q_rope], dim=-1)
# ✅ 分两路指针传进 kernel，QK^T 拆成 main 与 rope 两段 dot 相加
scores  = tl.dot(q0, tl.trans(k0)) + tl.dot(q1, tl.trans(k1))   # main，按 BLOCK_DQ 分块
scores += tl.dot(q_rope, tl.trans(kr))                          # rope 一段
```

同理，KV 展平：`n2 == 1` 时 `[B,S2,1,D] → [B*S2,1,D]` 是**纯 view**，
必须写成 `key.reshape(t2, n2, d)`；写成 `key.permute(0,2,1,3).reshape(...)`
会因非连续而触发真拷贝（实测 host 侧多出 `aclnnInplaceCopy_Transpose` 78us）。

⚠️⚠️ **本条只适用于 KV，绝不能套到 query 上**——两者被 permute 的那一维大小不同：

| 张量 | 形状 | 被 permute 的维 | `permute(0,2,1,3)` 的效果 |
|---|---|---|---|
| `key` / `value` | `[B, S2, **n2**, D]`，`n2 == 1` | **size-1** | 元素顺序**不变** ⇒ 可省，等价 `reshape` |
| `query` | `[B, S1, **n1**, D]`，`n1 = n2*g`（实测 128） | **≫1** | 元素顺序**真的改变** ⇒ **必须保留**，见 **L1.12** |

⇒ 看到"展平是纯 view"这句时先问：**被移动的那一维是不是 1**。是 1 才成立。
query 上照搬本条 = 直接触发 L1.12 那 15 例整片错。

### L1.5 ★★★ gather workspace 用**输入 dtype**（bf16/fp16）存储，不要升 fp32

workspace 是纯搬运，升 fp32 会让 GM 读写带宽**翻倍**，而本类恰恰是带宽瓶颈（G3）。
`load → store` 保持输入 dtype 是**位级无损拷贝**，精度无损失。

**host 侧照抄这段**（`k.dtype` 是关键，不要写 `torch.float32`）：

```python
# ✅ workspace 与源张量同 dtype —— GM 读写带宽减半
k_ws  = torch.empty((rows_total, d_main), dtype=k.dtype, device=device)
kr_ws = torch.empty((rows_total, dr_eff), dtype=k.dtype, device=device)
v_ws  = torch.empty((rows_total, d_v),    dtype=v.dtype, device=device)
pos_ws = torch.empty((rows_total,), dtype=torch.float32, device=device)  # pos 例外，用 fp32

# ❌ 这样写 GM 流量翻倍，本类会直接把性能压到 1/10 量级
# k_ws = torch.empty((rows_total, d_main), dtype=torch.float32, device=device)
```

gather kernel 里 `load → store` 不做任何精度提升；`tl.dot` 侧按 L1.6 用原生 dtype 操作数 + fp32 累加器。

⚠️ **实测代价**：一次真实生成里 L1.3 与 L1.5 双双缺失，精度 72/72 全过但
**`speedup_vs_torch` 只有 0.0876（比参考实现慢 11 倍，72 个用例无一 ≥1.0x）**。
本类"精度过了性能极差"的头号成因就是这两条。

### L1.6 ★ `tl.dot` 契约：原生 dtype 操作数 + fp32 累加

`QK^T` 用 bf16/fp16 输入喂 cube、fp32 累加。
`PV` 的 `p`（softmax 输出）**保持 fp32**，V 加载后 in-register 升 fp32——
`p` 直接降到输入 dtype 会放大误差（这是 attention 家族的共性结论）。

### L1.6b ★★★ D≥256/512 的 QK 累加精度：CANN 内部按 K-chunk 累加，triton 复刻只能三段式（与 flash_attention.md §3.3 同根）

**问题**：`D_main ≥ 256/512` 时，CANN golden 的 QK 是 **K-chunk 分块累加**（如 2×256 或 4×128），
与整段 K=512 单 `tl.dot` 的累加顺序不同 → 差 ~5 ULP。bf16 大 D 时 exp 放大进 softmax 分母，顶穿 MERE 阈值。

- **必须** 对 `D_main ≥ 256` 的 QK 做 **K 分块累加**，且 **triton-ascend 编译器会把单 kernel 内多个 `tl.dot` 融合成单个 K 归约**——runtime 循环 / debug_barrier / 分离变量 / 显式两半**全部无效**（已实证）。
- **必须** 用 **显式 unroll + Q 寄存器驻留**（0811/5.94x 实证，精度全过的唯一写法）：
  ```python
  # Q 一次性加载到寄存器（编译期展开，非循环）：
  offs_d = tl.arange(0, BLOCK_D)               # BLOCK_D = 128
  d0, d1, d2, d3, d4 = offs_d, BLOCK_D+offs_d, 2*BLOCK_D+offs_d, 3*BLOCK_D+offs_d, 4*BLOCK_D+offs_d
  q0 = tl.load(q_base + d0[None, :], mask=gm & (d0 < D_MAIN)[None, :], other=0.0)
  q1 = q0
  if NUM_DM > 1: q1 = tl.load(q_base + d1[None, :], mask=gm & (d1 < D_MAIN)[None, :], other=0.0)
  q2 = q0
  if NUM_DM > 2: q2 = tl.load(q_base + d2[None, :], mask=gm & (d2 < D_MAIN)[None, :], other=0.0)
  # K 循环内显式累加（禁止 for/static_range 循环 `scores += tl.dot(...)`）：
  k_base = K_WS + (ws_base + rows)[:, None] * D_MAIN
  scores = tl.dot(q0, tl.trans(tl.load(k_base + d0[None, :], mask=rm & (d0 < D_MAIN)[None, :], other=0.0)))
  if NUM_DM > 1: scores = scores + tl.dot(q1, tl.trans(tl.load(k_base + d1[None, :], mask=rm & (d1 < D_MAIN)[None, :], other=0.0)))
  if NUM_DM > 2: scores = scores + tl.dot(q2, tl.trans(tl.load(k_base + d2[None, :], mask=rm & (d2 < D_MAIN)[None, :], other=0.0)))
  if HAS_ROPE: scores = scores + tl.dot(q_rope, tl.trans(tl.load(kr_base + offs_r[None, :], mask=rm & (offs_r < D_ROPE)[None, :], other=0.0)))
  ```
  ⚠️ **关键**：`scores = scores + tl.dot(...)`（显式 add）能绕开编译器融合；**`scores += tl.dot(...)` 在 static_range 循环里仍可能被融合**（0818 实锤：用循环累加后 case 10 D=512 仍精度失败）。
- **PV 侧**（0811/5.0x 实证，分路径）：
  - **通用 gathered_attn_kernel**：`tl.dot(p, v.to(tl.float32))`——**p 保持 fp32，V 加载后升 fp32**（精度优先，G_BF16DOT 的精度方案）。
  - **split_attn_kernel**：`p_c = p.to(V_WS.dtype)` 后 `tl.dot(p_c, v)`（bf16 cube dot，V 不升 fp32）——分段路径 PV 走 bf16 cube（+3.6%）。
  - acc 用显式 unroll 分块（acc0-acc3，`if NUM_DVW>N` 展开），V 也按 DVW 分块 load。
- **首次生成必守**：看到 `D_MAIN ≥ 256` 直接写显式 unroll（Q 寄存器驻留 + `scores = scores + dot`），不要等失败再改。
- **判别信号**：bf16/fp16 大 D case（D=512）单独失败、`ssum`/`max_abs` 差 ~1e-3 量级、MERE 卡阈值附近 ⇒ 查 QK 累加顺序 + 是否用了循环累加（改显式 unroll），不是 tiling。

### L1.7 ★ 因果掩码用「原始 KV 位置」比阈值，**gather 时把 pos 一并物化**

参考实现的 `_cal_attenmask` 是按块边界推的 Python 循环，**禁止照抄**。
等价且可向量化的写法：每个选中行记下它在**原始 KV 序列**中的位置 `pos`，
与逐 token 的阈值 `thresh = (cur_s2 - cur_s1) + s1_idx + 1` 比较：

```python
# gather kernel：与 K/V 一起把 pos 写进 workspace，无效行写 -1
pos = tl.where(valid, (idx * BS + offset).to(tl.float32), -1.0)
tl.store(POS_WS + ws_row, pos, mask=row_in)
# attention kernel：
row_ok = row_in & (pos >= 0.0) & (pos < thresh)
```

⚠️ 掩码必须**加在 scores 上**、用**有限极小值**（如 `-2e35`）而不是 `-inf`，
也不要靠"把 K 清零"来实现——清零解决不了、反而正是错因。
归约维 padding 同理：`tl.load(..., other=0.0)` **不是掩码**，
`max` 前要 `tl.where(valid, scores, NEG)`、`sum` 前要 `tl.where(valid, p, 0.0)`。

#### ⛔⛔ 掩码之后**必须护住 softmax 的除法**，否则全掩码行输出 NaN

**这是上面那条掩码要求的必然推论，两者必须成对出现。**
把 `p` 整行清零之后，若某行的 KV **全部无效**，`l_i` 就恒为 0：

```python
# ❌ 实测：输出 196608/196608 全部为 NaN，并触发
#    RuntimeError: ACL stream synchronize failed, error code:507035（70/72 用例）
l_i = tl.full((BLOCK_G,), 0.0, dtype=tl.float32)
...
l_i = alpha * l_i + tl.sum(p, axis=1)      # 全掩码行 ⇒ 恒为 0
acc = acc / l_i[:, None]                   # ⇒ 0/0 = NaN，沿后续计算扩散

# ✅ 用 1.0 兜底，再把全掩码行显式置 0 —— 不引入任何小常量
l_safe = tl.where(l_i > 0.0, l_i, 1.0)            # 分母恒 > 0
acc    = acc / l_safe[:, None]
acc    = tl.where(l_i[:, None] > 0.0, acc, 0.0)   # 全掩码行显式给 0，不留 NaN
```

⛔⛔ **不要用 `tl.maximum(l_i, 1e-38)` 这类"极小常量钳制"**——实测 **65/72 编译失败**：

```
LLVM ERROR: unsupported datatype for arith::ExtFOp to hfusion
```

**原因**：`1e-38` 比 fp32 最小正规数 `2^-126 ≈ 1.1755e-38` 还小，是**次正规数**，
Triton 为精确表示它会提升到 **fp64**，于是要把 `l_i` 从 fp32 扩展到 fp64（`ExtFOp`），
而 **Ascend 后端不支持 fp64**（同族报错：fp64 截断会报 `arith::TruncFOp`）。

⇒ **通用判据**：kernel 里的浮点字面量必须落在 **fp32 正规范围**内
（绝对值 ≳ `1.2e-38`，且不要写 `1e-40` / `1e-45` 这类）。
需要"很小的正数"时，改用 `tl.where(x > 0, x, 1.0)` 这种**只含 0.0 / 1.0 的写法**，
两者都是 fp32 精确值，不会触发任何类型提升。

⚠️ **参考实现里没有这个保护，不要以为它不需要**：PyTorch 的 `_tsoftmax` 做的是
`y / x_sum`，而 `exp(x - x_max)` 至少有一个元素等于 1 ⇒ 分母恒 ≥ 1，**永远不会为 0**。
只有 Triton 侧因为要处理 **padding 行**与**全掩码行**才会出现 0 分母。
⇒ **凡是参考实现靠"数据天然非空"隐式成立的假设，在带掩码的 Triton 实现里都要显式补上。**

⚠️ **NaN 的报错形态会伪装成硬件故障**：实测 70/72 报
`ACL stream synchronize failed, error code:507035`，看着像 aicore 异常或坏卡。
**先查 `npu-smi info -t health / -t ecc`**：若 Health=OK 且 ECC 全零，就是代码里的 NaN，
不是设备问题（本轮已按此排除）。

### L1.8 ⛔ 禁止照抄参考实现的 Python 控制流：`break` / 提前退出在 Triton 里不存在

参考实现的 `_gather_kv` 写的是：

```python
for sparse_id in topk.view(-1).tolist():
    if sparse_id == -1:
        break                      # ← 照抄进 kernel 直接编译失败
```

实测报错：`UnsupportedLanguageConstruct: unsupported AST node type: Break`，
**72 个用例全灭**。Triton kernel 里没有 `break` / `continue` / 提前 `return`。

✅ 正确写法是**转成掩码**，让无效行照常参与计算但被遮蔽（与 L1.7 的 `pos` 机制同源）：

```python
idx   = tl.load(IDX + ..., mask=row_in, other=-1)
valid = row_in & (idx != -1)
src   = tl.where(valid, kv_start + idx * BS + offset, kv_start)   # 地址钳到界内
pos   = tl.where(valid, (idx * BS + offset).to(tl.float32), -1.0) # 无效行 pos=-1
```

⚠️ 同类陷阱：`.tolist()` / `for x in tensor` / 运行时 `if` 依赖张量值——
参考实现里凡是**依赖数据取值改变控制流**的写法，一律要改写成掩码或 `tl.where`。

#### ⛔⛔ kernel 内**没有 Python 容器**：不能用 list 收集 tile 再拼接

实测 **65/72 编译失败**，签名 `Error('q_main_chunks is not defined')`：

```python
# ❌ kernel 内不存在 Python list，append 无法解析
q_main_chunks = []
for dk in tl.static_range(0, NUM_DM):
    q = tl.load(Q_ptr + base_q[:, None] * D_MAIN + d_cols[None, :], ...)
    q_main_chunks.append(q)                    # ← NameError
q_all = tl.cat(q_main_chunks)                  # ← 也没有这种用法

# ✅ 两种正确写法，按用途二选一
# (a) 每块当场用完就写回，不跨块保留（gather / 逐块搬运走这条）
for dk in tl.static_range(0, NUM_DM):
    d_cols_m = dk * BLOCK_D_MAIN + tl.arange(0, BLOCK_D_MAIN)
    chunk = tl.load(SRC + ... + d_cols_m[None, :], mask=..., other=0.0)
    tl.store(DST + ... + d_cols_m[None, :], chunk, mask=...)

# (b) 需要跨块累加时，用固定形状的累加器，不用容器（dot / softmax 走这条）
acc = tl.zeros([BLOCK_G, BLOCK_DV], dtype=tl.float32)
for dk in tl.static_range(0, NUM_DM):
    acc += tl.dot(q_chunk, k_chunk)            # 形状恒定，见上方类型一致性要求
```

⇒ **判据**：kernel 里能跨迭代存活的只有**形状固定的张量变量**。
`list` / `dict` / `append` / 变长拼接一律不可用；`ListComp` 同样报
`UnsupportedLanguageConstruct`（实测另一轮 72/72 全灭）。
⇒ **UB 视角**：想"先收集全部 D 块再一起处理"本身就违反分块初衷——
实测这样写会触发 `ub overflow, requires 2377728 bits while 1572864 bits available`
（290KB > 192KB，同一轮 7/72）。**分块的意义就是不让它们同时存在。**

### L1.9 ⛔ `ptr.dtype` 是**指针类型**，取元素类型必须用 `.dtype.element_ty`

```python
out = (acc / l_i[:, None]).to(q_ptr.dtype)          # ❌
# AssertionError: cannot cast fp32[32,64] to <[32,64], pointer<bf16>>
out = (acc / l_i[:, None]).to(q_ptr.dtype.element_ty)   # ✅
out = (acc / l_i[:, None]).to(OUT_DTYPE)                # ✅ 或用 constexpr 显式传入
```

实测同样是 **72/72 全部编译失败**。这两条（L1.8/L1.9）加起来在一次真实生成里
吃掉了两轮完整 verify（约 40 分钟）。

#### ⛔ 同族陷阱：`tl.constexpr` 参数**没有 `.to()`**，别把它当张量

```python
# ❌ 72/72 全部编译失败：Error("'constexpr' object has no attribute 'to'")
pos = tl.where(valid, idx.to(tl.float32) * BS.to(tl.float32) + offset.to(tl.float32), -1.0)
#                                          ^^^^^^^^^^^^^^^  BS 是 tl.constexpr

# ✅ constexpr 直接参与算式即可，会自动提升；要显式转就用 Python 的 float()/int()
pos = tl.where(valid, idx.to(tl.float32) * BS + offset.to(tl.float32), -1.0)
pos = tl.where(valid, idx.to(tl.float32) * float(BS) + offset.to(tl.float32), -1.0)   # 也可
```

⇒ **判据**：`.to()` / `.dtype` / `tl.where` 的张量语义只对**张量**成立。
`tl.constexpr` 参数是编译期常量，`tl.program_id()` 的返回是标量——
对它们只能用 Python 运算符和 `float()`/`int()`。
⇒ 这条与 L1.9 同源：**在非张量对象上套张量的类型方法**，是本类算子第三高频的编译死因。

### L1.12 ⛔⛔ BSND→TND 的布局变换：**参考实现的注释和它的代码不一致，以代码为准**

参考实现里这一行是本类算子**最大的单点陷阱**：

```python
b, s1, n1, d = query.shape
# 注释写的是：转 TND 格式处理: [B, S1, N2, G, D] -> [B*S1, N2, G, D]
q_fp = query.permute(0, 2, 1, 3).reshape(t1, n2, g, d).float()      # ← 以这行为准
```

注释描述的语义（把 `n1` 拆成 `n2*g`、`(b,s1)` 合成 `t1`）**根本不需要 permute**。
而实际代码带了 `permute(0,2,1,3)`，元素顺序是 **`(b, n1, s1, d)` 重解释**成 `(t1, n2, g, d)`。

**两者只在 `s1 == 1` 时恒等**（CPU 上 `torch.equal` 逐形态验证）：

| b | s1 | n2 | 两种写法一致 |
|---|---|---|---|
| 1 | **1** | 1 | ✅ True |
| 1 | 3 | 1 | ❌ False |
| 1 | 5 | 1 | ❌ False |
| 3 | 7 | 1 | ❌ False |
| 1 | 4 | 2 | ❌ False |

**实测代价（r4，57/72）**：按注释语义实现 ⇒ `s1 > 1` 的用例**全错**，
`matched_ratio` 仅 **0.003 ~ 0.34**（判据 ≥0.9），输出**连符号都反**
（`framework=-2.343750 / impl=+2.078125`）。15 个失败用例的 `s1` 分别是
3,5,8,13,21,34,55,89,128,7,3,7,13,37,99 —— **无一例外全部 `s1 > 1`**。

⇒ **`matched_ratio < 0.01` 是布局/索引错，不是精度不够**：
不要往容差 / dtype / 累加顺序 / 低精度舍入方向查，直接对布局变换。
⇒ **`s1 = 1` 的用例掩盖这个 bug**，自测必须挑 `s1 ≥ 3` 的形态。
⇒ 通用教训：**参考实现的注释不可信，只有代码是规范**。写 host 侧布局变换前，
先用 `torch.arange` 填一个小张量、把参考写法和你的写法 `torch.equal` 对一遍，
5 分钟就能排掉这类整片失败。

### L1.13 ⛔⛔ 每条 D 路径**独立一套变量**；同一表达式的张量必须用**同一个** `BLOCK_*`

**实测三轮全灭在这一条上**（r5 iter_1、r5 iter_3、r6 iter_0，各 62~72 例）：

```
Error('initial value for `kc` is of type <[64, 128], bf16>, but the then block redefines it as <[64, 64], bf16>')
Error('Cannot make_shape_compatible: incompatible dimensions at index 1: 16 and 64')
```

起因固定：K main 宽 `BLOCK_D_MAIN=128`、K rope 宽 `BLOCK_D_ROPE=64`、V 宽 `BLOCK_D_V`，
**三条路径共用一个 `kc` / `d_cols` / `d_mask`** ⇒ 第二次赋值形状变了，Triton 直接拒绝。

| 路径 | 宽度常量 | tile | 下标 | 掩码 |
|---|---|---|---|---|
| K main | `BLOCK_D_MAIN` | `kc_main` | `d_cols_m` | `dm_mask` |
| K rope | `BLOCK_D_ROPE` | `kc_rope` | `d_cols_r` | `dr_mask` |
| V | `BLOCK_D_V` | `vc` | `d_cols_v` | `dv_mask` |

**两条硬性检查，写完 kernel 后逐条过一遍：**

1. **搜一遍 `kc` / `d_cols` / `d_mask` 这三个裸名，出现即错**——必须带 `_main` / `_r` / `_v` 后缀。
2. **把所有 `tl.zeros([...])` 与 `tl.arange(0, X)` 的常量列出来**，
   确认同一个表达式里的两者用的是同一个 `BLOCK_*`
   （典型错法：累加器按 `BLOCK_DV` 开、载入的 tile 按 `BLOCK_D_MAIN` 开）。

⚠️ 宽度常量本身也不要起两个名字：实测出现过 `BLOCK_DV` 与 `BLOCK_D_V` 并存。
**每个宽度只允许一个常量名。**

### L1.14 ⛔⛔ kernel 签名规格：所有循环上界、宽度、开关**必须声明为 `tl.constexpr` 参数**

**实测三轮各 10/72 编译失败**（`tl.static_range end value is not a constexpr`）与
**一轮 60/72 编译失败**（`Cannot access global variable NEG_MASK from within @jit'ed function`），
根因相同：**卡片其它条款引用了 `NUM_DM` / `NEG` 这些标识符，但从没说它们从哪来**。
照此签名写，两类错误一起消失：

**以下签名摘自本算子已验证达标（72/72、6.4699x）的实现，逐字照此划分：**

```python
@triton.jit
def gather_kv_kernel(
    IDX, K, KR, V, POS_WS, K_WS, KR_WS, V_WS, KV_START, KV_END,   # 指针
    t1_start, N2, TOPK, SEL_ROWS,          # ★ 运行期参数，**不加 tl.constexpr**
    BS: tl.constexpr,                      # ★ 必须 constexpr：BS==1 时 //BS、%BS 编译期折叠
    D_MAIN: tl.constexpr, D_ROPE: tl.constexpr, D_V: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DR: tl.constexpr,
    NUM_DM: tl.constexpr, NUM_DV: tl.constexpr,      # 循环上界
    USE_SPAN_SELECT: tl.constexpr,                    # 分档开关
):
    NEG = -2e35            # ★ 掩码常量写在 kernel 内部，**不要**放模块级全局
    pid_t = tl.program_id(0)        # [T1 * N2]
    pid_n = tl.program_id(1)        # [cdiv(SEL_ROWS, BLOCK_N)]
    n2  = pid_t % N2
    t1g = t1_start + pid_t // N2
    kv_start = tl.load(KV_START + t1g)
    kv_end   = tl.load(KV_END + t1g)

    n_offs    = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_in    = n_offs < SEL_ROWS
    block_idx = n_offs // BS
    offset    = n_offs % BS
    idx       = tl.load(IDX + (t1g * N2 + n2) * TOPK + block_idx, mask=row_in, other=-1)
    src       = kv_start + idx * BS + offset
    valid     = row_in & (idx != -1) & (src >= kv_start) & (src < kv_end)
    src_safe  = tl.where(valid, src, kv_start)
    pos       = tl.where(valid, (idx * BS + offset).to(tl.float32), -1.0)
    ws_row    = pid_t * SEL_ROWS + n_offs           # ← 行映射公式，两个 kernel 必须一致
    tl.store(POS_WS + ws_row, pos, mask=row_in)
```

⚠️ **注意 `SEL_ROWS` / `N2` / `TOPK` / `t1_start` 不是 constexpr**——
它们随 shape 变化，若声明成 constexpr 会导致**每种 shape 重新编译一次 kernel**。
**只有"决定代码形状"的量才 constexpr**：tile 宽度、循环上界、分档开关、以及 `BS`
（`BS` 特殊：它进了 `//` 和 `%`，constexpr 化才能在 `BS==1` 时折叠掉整条除法链）。

host 侧把 constexpr 值**按 python int 传入**：

```python
NUM_DM = (d_main + BLOCK_D - 1) // BLOCK_D          # 纯 python 整数除法
gather_kv_kernel[grid](..., BS=bs, D_MAIN=d_main, BLOCK_D=128, NUM_DM=NUM_DM, ...)
```

⛔ **三条硬性规则**：

1. **凡是出现在 `tl.static_range(...)` 边界、`tl.arange(0, X)`、tile 形状里的量，
   一律声明为 `tl.constexpr` 参数**——它们必须在编译期已知。
2. **常量（如掩码用的 `-2e35`）不要在 kernel 内部直接引用模块级全局**。
   受限的是**访问**，不是定义——三种写法都合法：

   ```python
   # ✅ a) 写在 kernel 函数体内（最简单）
   @triton.jit
   def attn_kernel(...):
       NEG = -2e35

   # ✅ b) 模块级定义，host 侧引用后按参数传入
   NEG_MASK = -2.0e35                      # 模块级定义本身没问题
   attn_kernel[grid](..., NEG=NEG_MASK)    # host 侧传值

   # ✅ c) 模块级但标注 constexpr
   NEG_MASK: tl.constexpr = -2.0e35

   # ❌ kernel 内部直接读未标注的模块级全局
   @triton.jit
   def attn_kernel(...):
       NEG = NEG_MASK      # NameError: Cannot access global variable NEG_MASK
   ```

   实测 **60/72 编译失败**，签名
   `Cannot access global variable NEG_MASK from within @jit'ed function`。
3. **运行期量（`kv_start` / `thresh` / stride）不加 `tl.constexpr`**，
   它们只能进算式与掩码，不能进循环边界或形状。

### L1.11 ⛔ `tl.static_range` 的**起止两端**都必须是 `tl.constexpr`，不能传运行期值

实测 **10/72 用例编译失败**，签名唯一：

```
Error('int32[] used as tl.static_range start value is not a constexpr')
```

起因是把 `kv_start` / `sel_rows` 这类**运行期张量或运行期标量**当成了循环边界：

```python
# ❌ 编译失败：start 是运行期 int32
for i in tl.static_range(kv_start, kv_end):
    ...

# ✅ 边界用 constexpr，运行期部分转成偏移量或掩码
for i in tl.static_range(0, NUM_CHUNKS):          # NUM_CHUNKS: tl.constexpr
    off = base + i * BLOCK                         # 运行期量只进算式，不进循环边界
    ...
```

⚠️ 区分两个循环：
- `tl.static_range(...)` —— 编译期完全展开，**边界只能是 `tl.constexpr`**
- `range(...)` —— Triton 里是运行期循环，边界可以是运行期标量

拿不准就用 `range`；只有在确实需要展开、且边界是 `tl.constexpr` 时才用 `tl.static_range`。

### L1.10 Grid 收缩到 `min(核数, tasks)` + 核内步长循环，且 grid 与步长必须一致

⚠️ UB 只有 192KB：**绝对禁止** `BLOCK_* = triton.next_power_of_2(<运行期变量>)` 且不加
`min()` 封顶。中间张量元素数要有**与输入无关的上界**（实测 2D tile 16384 元素 OK、
32768 元素触发 `ub overflow`）。一次 `ub overflow` 会触发 vector core exception
**污染设备、使后续用例连带失败**——看到大批连续失败时先定位**第一个**失败用例单独复跑，
不要按失败总数估计缺陷数量。

---

## §3 Layer 2: 算法骨架

⚠️ **先分清什么该抄、什么不该抄**——这两件事常被混为一谈：

| | 要求 |
|---|---|
| **参考实现的代码** | ⛔ **不得照抄**（Python 双循环、`_gather_kv` 等 helper 都不能搬） |
| **本节的 kernel 分解与 host 分派结构** | ✅ **必须照做**——这是达标实现的骨架，不是"参考方向" |
| 具体每行 Triton 代码 | 自己写，但要落在上面那个结构里 |

⇒ 「输出必须是全新草图」指的是**代码要自己写**，**不是**"结构可以自行简化"。
实测：把 ②a/②b/②c 合并成一个通用 kernel、省掉 ③，精度能过但
`speedup` 只有 **0.2 ~ 0.3x**（目标 2.2）——**结构一简化，性能上限就锁死了**。

三段式：

```
host: 布局统一为 TND 语义（**按 L1.12 逐字对齐参考实现的 permute+reshape，不是纯 view**）
      → 逐 token 阈值 thresh / kv_start / kv_end 预算
  └ 形态分派器（§3.2）：算 meta → 定 spec / bs>4 / 一般 三条路径的 tile 与开关
      └ rows_total = t1 * n2 * sel_rows；workspace 一次性开全量（L1.2，**禁止 host 循环**）
          ① gather_kv_kernel  : 按 idx 把 K/KR/V 选中行 + pos 物化进 workspace
                                grid = (t1 * n2, cdiv(sel_rows, BLOCK_N))   ← token 维进 grid
                                （L1.1b 行维向量化；L1.3 span-select；L1.5 输入 dtype）
          ② compute kernel，按 §3.2 的路由三选一：
             ②a attn_kernel        通用：整段 workspace 上 online softmax
             ②b split_attn_kernel  并行度不足时按行分段
             ②c spec_attn_kernel   spec 形态专用 tiling（BLOCK_DQ=256 / ROW=512 / DV=256）
          ③ split_combine_kernel : 仅 ②b 之后，用 logsumexp 合并分段结果
```

⚠️ **kernel 数量是结构信号**：达标实现是 **4~5 个 `@triton.jit`**（gather / 通用 attn /
split attn / combine / spec attn）。只写出 2~3 个，说明分派没做，性能会卡在 0.2~0.3x。

### §3.1 ⛔⛔ 首版就必须这么写的三处，**不是"优化项"，不许推给 Phase 4**

实测教训：L1.3 / L1.5 因为**不影响正确性**，被连续多轮推迟到 Phase 4，
而 Phase 4 又不会照 §4 的清单逐条执行 ⇒ 最终一条都没落地，
精度 72/72 但 `speedup` 只有 **0.2059**（目标 2.2）。
**所以这三处写进骨架，首版代码就要长这样。**

```python
# ① host：workspace 与源张量同 dtype —— 少写一个 dtype 就等于 GM 流量翻倍
k_ws  = torch.empty((rows_total, d_main), dtype=k.dtype, device=device)   # 不是 torch.float32
kr_ws = torch.empty((rows_total, dr_eff), dtype=k.dtype, device=device)
v_ws  = torch.empty((rows_total, d_v),    dtype=v.dtype, device=device)
pos_ws = torch.empty((rows_total,), dtype=torch.float32, device=device)   # 仅 pos 用 fp32

# ② gather kernel：行维向量化（L1.1b）+ D 维分块（避免 UB 超限）
grid = (t1 * n2, triton.cdiv(sel_rows, BLOCK_N))      # BLOCK_N = 64 或 128
n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # 一次一整块行
# ⛔ 每个 D 块必须**当场 load → 当场 store**，禁止先收集再拼接（见 §3.1 末尾 ⛔⛔）
for dk in tl.static_range(0, NUM_DM):                 # NUM_DM: tl.constexpr（L1.11）
    d_cols_m = dk * BLOCK_D_MAIN + tl.arange(0, BLOCK_D_MAIN)
    dm_mask  = d_cols_m < D_MAIN
    kc_main  = tl.load(K + src_safe[:, None] * stride_k + d_cols_m[None, :],
                       mask=row_in[:, None] & dm_mask[None, :], other=0.0)
    tl.store(K_WS + ws_row[:, None] * D_MAIN + d_cols_m[None, :], kc_main,
             mask=row_in[:, None] & dm_mask[None, :])   # ← 立即写回，不在 UB 里累积

# ③ gather kernel：同地址退化的单行广播（L1.3），与 ② 写在同一个循环里
#    ⚠️ 变量名必须按路径区分：main / rope / v 三条路径的 tile 宽不同，见下方 ⛔⛔
src_lo, src_hi = tl.min(src_safe), tl.max(src_safe)
span_ok = (src_hi - src_lo) < (BS if BS <= 4 else 4)
if span_ok:
    if BS == 1:
        r0 = tl.load(K + src_lo * stride + d_cols_m, mask=dm_mask, other=0.0)
        kc_main = tl.broadcast_to(r0[None, :], (BLOCK_N, BLOCK_D_MAIN))
    else:
        ...
else:
    kc_main = tl.load(K + src_safe[:, None] * stride + d_cols_m[None, :], ...)

# rope 路径：**独立一套变量**，因为 BLOCK_D_ROPE ≠ BLOCK_D_MAIN
# ⛔ 不能复用上面的 kc_main / d_cols_m / dm_mask —— 宽度不同，复用必然编译失败
if D_ROPE > 0:
    for dr in tl.static_range(0, NUM_DR):          # NUM_DR: tl.constexpr
        d_cols_r = dr * BLOCK_D_ROPE + tl.arange(0, BLOCK_D_ROPE)
        dr_mask  = d_cols_r < D_ROPE
        if span_ok:
            r0r = tl.load(KR + src_lo * stride_r + d_cols_r, mask=dr_mask, other=0.0)
            kc_rope = tl.broadcast_to(r0r[None, :], (BLOCK_N, BLOCK_D_ROPE))
        else:
            kc_rope = tl.load(KR + src_safe[:, None] * stride_r + d_cols_r[None, :],
                              mask=row_in[:, None] & dr_mask[None, :], other=0.0)
        tl.store(KR_WS + ws_row[:, None] * D_ROPE + d_cols_r[None, :], kc_rope,
                 mask=row_in[:, None] & dr_mask[None, :])
```

**三条路径的变量表**（照此命名，不要合并）：

| 路径 | 宽度常量 | tile | 下标 | 掩码 |
|---|---|---|---|---|
| K main | `BLOCK_D_MAIN` | `kc_main` | `d_cols_m` | `dm_mask` |
| K rope | `BLOCK_D_ROPE` | `kc_rope` | `d_cols_r` | `dr_mask` |
| V | `BLOCK_D_V` | `vc` | `d_cols_v` | `dv_mask` |

#### ⛔⛔ Triton 变量在 `if` 前后必须**保持同一类型**——三条路径不能共用一个名字

实测 **72/72 全部编译失败**，签名：

```
Error('initial value for `kc` is of type <[64, 128], bf16>,
       but the then block redefines it as <[64, 64], bf16>')
```

起因：main 维 tile 是 `BLOCK_D_MAIN = 128`、rope 维是 `BLOCK_D_ROPE = 64`，
两条路径复用同一个变量名 `kc` ⇒ 第二次赋值形状变了，Triton 直接拒绝。

| 路径 | tile 宽 | 变量名 |
|---|---|---|
| K main | `BLOCK_D_MAIN`（128） | `kc_main` |
| K rope | `BLOCK_D_ROPE`（64） | `kc_rope` |
| V | `BLOCK_D_V`（128） | `vc` |

⇒ **判据**：只要两处赋值的 tile 形状可能不同，就必须用不同变量名，
`d_cols` / `d_mask` 这类下标与掩码同理（`d_cols_m` / `d_cols_r`）。
⇒ 这条同样适用于**循环体内**：`for` 各轮之间变量类型也必须一致。

**同族的第二种表现——多操作数运算的形状必须逐维相等**（实测 72/72 全灭）：

```
Error('Cannot make_shape_compatible: incompatible dimensions at index 1: 16 and 64')
    acc0 = tl.where(cond, v_tile, acc0)      # acc0 是 [*,16]，v_tile 是 [*,64]
```

`tl.where` / `+ - * /` / `tl.maximum` 的所有操作数**必须逐维相等**（Triton 不做隐式广播补齐）。
最常见的错法：**累加器按 `BLOCK_DV` 开、载入的 tile 按 `BLOCK_D_MAIN` 开**，两个常量不同就崩。

```python
# ✅ 累加器与参与运算的 tile 用同一个宽度常量
acc = tl.zeros([BLOCK_G, BLOCK_DV], dtype=tl.float32)
v_tile = tl.load(V_WS + ws_row[:, None] * D_V + d_cols_v[None, :], ...)  # d_cols_v 用 BLOCK_DV
acc = acc + tl.dot(p, v_tile)
```

⇒ **自查动作**：写完 kernel 后把所有 `tl.zeros([...])` 与 `tl.arange(0, X)` 的常量列出来，
确认参与同一个表达式的两者用的是**同一个** `BLOCK_*`。四轮真实生成里有两轮死在这里。

⚠️ **② 的两半必须同时写**：只做行维向量化、D 维不分块 ⇒ 实测
`MLIRCompilationError: need extra local buffer`（UB 192KB 装不下 `BLOCK_N × 576`）。

### §3.2 ⛔⛔ host 侧必须是**形态分派器**，不是一套常量走到底

> ⚠️ **这是 Layer 1 硬约束，首次生成必守，不是 Phase 4 可选项**：
> **必须** 首版就写 **5 个指定职责的 `@triton.jit` kernel**：`gather_kv_kernel` + `attn_kernel`（通用直算）+ `spec_attn_kernel` + `split_attn_kernel` + `split_combine_kernel`，
> host 侧按 §3.2 形态分派器选路径。**禁止** 用任意 5 个"打散流程" kernel 冒充三路径分派——0818/7 二测实锤：agent 写了 `_prep_tables/_src_row/_q_prep/_gather_kv/_attention` 5 个 kernel，`grep -c "^@triton.jit"` = 5 **通过了数量自查**，但 **没有 spec_attn/split_attn/split_combine**（三路径分派完全缺失）→ 0.54x，Phase 4 没跑。
>
> **自查（比数量更严格，必须全过）**：
> 1. `grep -cE "spec_attn_kernel|split_attn_kernel|split_combine_kernel" generated.py` 必须 **≥3**（三个分派 kernel 名都在）；
> 2. `grep -c "^@triton.jit"` ≥5；
> 3. **sketch 的 `@architecture_decision` 必须包含** "spec-attn-kernel" 与 "split-kv-kernel" 两条决策（agent 二测的 sketch 只有 "gather-workspace-4kernel"，没有分派决策——这等于根本没把三路径当架构）。
> 任一条不过 → **立即补全，不要进 Phase 4**。
>
> **Why**：72 个用例横跨 TND 退化索引 / BSND 随机索引、`bs` 1~64、`d_main` 64~512 等多种形态，
> 单一通用 kernel 配单套 tile 常量，无论 Phase 4 怎么调都到不了目标
> （实测三条基因全落地、单路径版本仍只有 0.2~0.3x；达标版 6.4699x 需三路径）。
>
> ⚠️ **架构只是第一步，tile 推满才是 5.0x**：5 kernel 正确但 tile 保守 = 0.87x（0818/7）；
> 5 kernel + spec tile 推到 UB 上限组合（G=32/ROW=512/DQ=256/DV=256）= 5.94x（0811/7）。
> **Phase 4 必须按 §4.0c 的 5 轮路径把 spec tile 一次推满**，不要停在 0.87x。
>
> ⚠️ **Phase 4 是完成判定的必要部分，不是可选**：0818/7 二测 agent 停在 Phase 3（只有 iter_0，无 opt_iter）就结束，report 说"后续优化方向见 sketch"——这是**未完成**。必须跑完 Phase 4 的 §4.0c 5 轮路径（推 spec tile）才算达标，不能停在首版 0.54x。

首版就要写成"host 先算 meta → 选路径 → 传不同 constexpr"：

```python
# ① 形态判据（host 侧，纯 python，不进 kernel）
spec = (bs <= 4 and d_main == 512 and d_v == 512 and n2 == 1
        and (not has_rope or d_rope <= 64))

# ② 按形态给 gather 的 tile 与开关
if spec:
    BLOCK_N, USE_SPAN_SELECT, BLOCK_D_GATHER = 64, True, 128
    SPEC_BLOCK_ROW, SPEC_BLOCK_DQ, SPEC_BLOCK_DV = 512, 256, 256   # 专用 compute tiling
elif bs > 4:
    BLOCK_N, USE_SPAN_SELECT, BLOCK_D_GATHER = 128, False, 128     # bs>4 时 span 窗口失效
else:
    BLOCK_N, USE_SPAN_SELECT, BLOCK_D_GATHER = 64, True, 128

# ③ split-KV 路由（0811/5.0x 完整 4 级分层，逐字照抄）
g_blocks = cdiv(G, BLOCK_G)
base_par = t1 * n2 * g_blocks
bg_dir, br_dir = BLOCK_G, BLOCK_ROW
if base_par < 32:
    num_splits = min(cdiv(sel_rows, BLOCK_ROW), cdiv(64, base_par))
    num_splits = max(1, num_splits)                        # 极低并行 → 尽量分段
elif (t1 <= BLOCK_T1 and t1 * n2 * cdiv(G, bg_dir) < 64
      and sel_rows >= 4 * br_dir):
    num_splits = min(cdiv(sel_rows, br_sp), cdiv(64, base_par_d), 2)
    num_splits = max(1, num_splits)                        # 单 chunk mid-parallel 2 段温和 split
elif (base_par < 160 and sel_rows >= 4 * br_dir):
    if base_par < 48 and sel_rows >= 8 * br_dir: target_split = 6
    elif base_par < 96 and sel_rows >= 4 * br_dir: target_split = 4
    else: target_split = 3
    num_splits = min(cdiv(sel_rows, br_sp), target_split)  # SuperFusion 6/4/3 分层
    num_splits = max(1, num_splits)
else:
    num_splits = 1                                         # 并行度足够，不分段
use_split = num_splits > 1
# spec 形态：rows_per_split 按 SPEC_BLOCK_ROW 对齐
#   rows_per_split = cdiv(cdiv(sel_rows, num_splits), SPEC_BLOCK_ROW) * SPEC_BLOCK_ROW
#   num_splits = cdiv(sel_rows, rows_per_split)
```

**每条判据背后的实测依据**（不要当成魔法常量）：

| 判据 | 依据 |
|---|---|
| `USE_SPAN_SELECT` 随 `bs` 开关 | `bs=1` 时 span 窗口=1 行，对 BSND 随机索引恒 `False` 自动回退散射路径；对 TND 退化索引（30/72 例，`sparse_indices` 全 0）命中单行广播。**关掉它 gather 从 0.02ms 涨到 0.53ms，24x 退化** |
| `bs > 4` 关掉 span | 窗口内行不再同址，判据恒假，白算一遍 min/max |
| `BLOCK_D_GATHER = 128` | 每 program chunk 搬运次数 17→9，单次 `[64,128]` bf16 = 256B 连续宽载 |
| `BLOCK_N` 只对大形态放 128 | 全量 64→128 时 `t1==1` 的小形态因 gather program 数 32→16 **并行度饥饿退化 ~+20%**；只在 `bs==1 且 sel_rows>=1024 且 gather grid 并行度>=160` 时放开 |
| split 阈值 160 | 与 gather 并行度同口径；不够并行才分段，否则 combine 是纯开销（实测占 7.3%、0 FLOPs） |

⚠️ **不要把 spec 路径写成"先做通用版，Phase 4 再加"**——专用 compute kernel 的 tiling
（`BLOCK_DQ=256` 两 chunk + `BLOCK_ROW=512` + `BLOCK_DV=256`）与通用版差异过大，
事后改等于重写；而 Phase 4 不会做这种规模的重构。

#### ⛔⛔ 三个 compute kernel 的签名规格（摘自已验证达标实现，逐字照此写）

**这是 0.99x → 2.0x 以上的唯一通路。** 实测：只写通用 attn kernel 一条路径、
精度 72/72 全过，`speedup` 也只有 **0.9938**；加上下面两条路径的达标版是 **6.4699**。
**七轮真实生成全部只写出 2~3 个 kernel，无一例外卡在 1.0x 以下。**

```python
# ① 通用直算：gathered_attn_kernel —— 已有，略

# ② spec 形态专用（用 IS_SPLIT 一份代码覆盖直算与分段两条路径）
@triton.jit
def spec_attn_kernel(
    Q, QR, POS_WS, K_WS, KR_WS, V_WS, THRESH,
    OUT, XMAX, XSUM, P_M, P_L, P_ACC,            # ★ 后三个是分段用的部分结果缓冲
    t1_start, N2, G, SEL_ROWS, ROWS_PER_SPLIT, SCALE,     # 运行期参数
    D_MAIN: tl.constexpr, D_ROPE: tl.constexpr, D_V: tl.constexpr,
    HAS_ROPE: tl.constexpr, IS_SPLIT: tl.constexpr,       # ★ 分段开关
    BLOCK_G: tl.constexpr, BLOCK_ROW: tl.constexpr,
    BLOCK_DQ: tl.constexpr, BLOCK_DR: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    pid_t = tl.program_id(0)      # [T1 * N2]
    pid_g = tl.program_id(1)      # [cdiv(G, BLOCK_G)]

# ③ 通用分段：split_attn_kernel（写部分结果，不写最终输出）
@triton.jit
def split_attn_kernel(
    Q, QR, POS_WS, K_WS, KR_WS, V_WS, THRESH, P_M, P_L, P_ACC,
    t1_start, N2, G, SEL_ROWS, ROWS_PER_SPLIT, SCALE,
    D_MAIN: tl.constexpr, D_ROPE: tl.constexpr, D_V: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_ROW: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_DR: tl.constexpr, BLOCK_DV: tl.constexpr,
    NUM_DM: tl.constexpr, NUM_DV: tl.constexpr, NUM_DVW: tl.constexpr,
):
    pid_t = tl.program_id(0)      # [T1 * N2]
    pid_g = tl.program_id(1)      # [cdiv(G, BLOCK_G)]
    pid_s = tl.program_id(2)      # ★ [NUM_SPLITS] —— 分段维进 grid 第三维

# ④ 分段合并：split_combine_kernel（仅 ③ 之后调用）
@triton.jit
def split_combine_kernel(
    P_M, P_L, P_ACC, OUT, XMAX, XSUM,
    t1_start, N2, G,
    D_V: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_DV: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_g = tl.program_id(1)
```

**部分结果缓冲的契约**（③ 写、④ 读，形状必须对齐）：

```python
P_M   = torch.empty((t1 * n2, cdiv(G,BLOCK_G)*BLOCK_G, NUM_SPLITS), dtype=torch.float32, ...)  # 每段的 rowmax
P_L   = torch.empty_like(P_M)                                                                   # 每段的 sum
P_ACC = torch.empty((..., NUM_SPLITS, d_v), dtype=torch.float32, ...)                           # 每段的 acc
# ④ 里按 logsumexp 归并：m = max(m_s)；l = Σ l_s * exp(m_s - m)；acc = Σ acc_s * exp(m_s - m)
```

⇒ **`IS_SPLIT` 是 `tl.constexpr`**：同一份 spec kernel 代码，`IS_SPLIT=False` 时写 `OUT`，
`IS_SPLIT=True` 时写 `P_*`，编译期择一，**不要写成两个 kernel**。
⇒ **`pid_s = tl.program_id(2)`**：分段维进 grid 第三维，不是 kernel 内循环。

---

## §4 Phase 4 优化点清单

按**收益期望**排序。G3 已经说明：**先砍搬运，再谈 tile 与流水**。

### §4.0 ⛔⛔ Phase 4 必踩的第一个坑：**Q tile 预载入外提会撑爆 UB**

**两轮独立生成的 Phase 4，第一个优化动作完全相同、失败字节数逐位相同**——
把 Q 的加载提到 KV 循环外（"避免每个行块重复加载 Q"）：

```python
# ❌ 实测两轮各 72/72 编译失败，签名逐位相同：
#    ub overflow, requires 2116608 bits while 1572864 bits available   （258KB > 192KB）
# Preload Q main tiles (independent of KV row chunk)
if NUM_DM > 0:
    q_m_0 = tl.load(Q + ... + d_cols_m0[None, :] * stride_qd, ...)
if NUM_DM > 1:
    q_m_1 = tl.load(Q + ... + d_cols_m1[None, :] * stride_qd, ...)   # ← 全部常驻 UB
```

**UB 账**：`NUM_DM` 个 `[BLOCK_G, BLOCK_D_MAIN]` fp32 常驻，
加上 `acc`（`NUM_DV × [BLOCK_G, BLOCK_DV]` fp32）与 `scores`、K/V tile，
实测需 **258KB**，而 UB 只有 **192KB**，差 66KB。

✅ **想做这个优化，必须同时缩小别的东西**，三选一：
- 只预载 `NUM_DM` 中的**第一块**，其余仍在循环内加载
- `BLOCK_G` 减半（32 → 16），Q 常驻量随之减半
- 先做 §4.1 削减 workspace 往返（收益更大且不吃 UB），别动 Q 预载

⚠️ **这一步失败会白白消耗一整轮 Phase 4（实测约 30~40 分钟）**。
先算 UB 账再改，不要试了才知道。

#### ⛔ 已证伪：`compile_hint("mayDiscretememaccess")` 与消除取模，**均无收益**

Phase 4 曾自主尝试以下两项，实测 speedup **1.4187 → 1.4166（−0.15%）**，精度仍 72/72：

```python
# ❌ 无收益：给 gather 的 load 加离散访问编译提示
tl.extra.cann.extension.compile_hint(kc_main, "mayDiscretememaccess")
tl.extra.cann.extension.compile_hint(kc_rope, "mayDiscretememaccess")
tl.extra.cann.extension.compile_hint(vc,      "mayDiscretememaccess")

# ❌ 无收益：把取模改写成减法（本以为整数除法在 Ascend 上昂贵）
n2_idx = pid_t - t * N2          # 原 pid_t % N2
offset = n_offs - block_idx * BS  # 原 n_offs % BS
```

⇒ 两者看似都对症（gather 确实是离散访问、取模确实较贵），但**在本类算子上被搬运带宽完全掩盖**。
与 G3 一致：**瓶颈在 GM 搬运字节，不在指令级开销**。不要重试这两项。

⇒ **Phase 4 自主优化的实测收益（三轮样本，差异极大）**：
`0.9938→0.9915`（−0.2%）、`1.4187→1.4166`（−0.2%）、**`1.4166→1.7698`（+24.9%）**。
**不是恒为零**——方向对了就有大收益，方向错了就是白跑一轮。有效方向见 §4.0b。

### §4.0b ★★★ 已验证有效：**V tile 的 load 提到 softmax 之前**（实测 +24.9%）

`V` 的加载只依赖 `row_mask` 与 `D_V` 边界，**与 `scores` / softmax 无关**，
因此可以提前发射，让 **MTE2（搬运）与 Vector（softmax）并行**：

```python
# ❌ 原写法：softmax 算完才载入 V，MTE2 与 Vector 串行
scores = scores * scale
scores = scores + tl.where(valid_row[None, :], 0.0, NEG)
m_new  = tl.maximum(m, tl.max(scores, axis=1))
p      = tl.exp(scores - m_new[:, None]) * valid_row[None, :].to(tl.float32)
l      = l * alpha + tl.sum(p, axis=1)
v0 = tl.load(V_WS + ...)                      # ← 载入在这里
acc0 = acc0 * alpha[:, None] + tl.dot(p, v0.to(tl.float32), out_dtype=tl.float32)

# ✅ 把 V 的 load 提到 softmax 之前：MTE2 与 Vector 重叠
v0 = tl.load(V_WS + ...)                      # ← 提前发射，只依赖 row_mask / D_V
v1 = tl.load(V_WS + ...)
scores = scores * scale
scores = scores + tl.where(valid_row[None, :], 0.0, NEG)
m_new  = tl.maximum(m, tl.max(scores, axis=1))
p      = tl.exp(scores - m_new[:, None]) * valid_row[None, :].to(tl.float32)
l      = l * alpha + tl.sum(p, axis=1)
acc0 = acc0 * alpha[:, None] + tl.dot(p, v0.to(tl.float32), out_dtype=tl.float32)
```

**实测 `1.4166 → 1.7698`（+24.9%，72/72，warmup=5 repeats=5）。**
⇒ 与 G3 一致：本类瓶颈是搬运，**让搬运与计算重叠比削减指令更有效**。
⇒ ⚠️ 注意 UB：提前载入的 V tile 会多占驻留空间，与 §4.0 的 Q 预载入**不要同时做**
（Q 预载入实测直接撑爆 258KB）。

### §4.0c ★★★ 收敛目标：spec kernel 的 tile 必须推到 UB 上限组合（0811/5.94x 的核心，首版就朝这个写，Phase 4 5 轮内必达）

**这是 0.54x/0.87x → 5.0x+（0811/7）的全部差距，两种失败都要避开**：
- **0818 首测（0.87x）**：架构对了（5 kernel + 三路径 + 显式 unroll），但 Phase 4 只把 `BLOCK_ROW` 推到 256 就停，`BLOCK_G=16/DQ=128/DV=128` 太保守。
- **0818 二测（0.54x）**：架构就错了（5 个打散流程 kernel，无 spec/split/combine 三路径）+ **Phase 4 完全没跑**（只有 iter_0，report 说"后续优化方向见 sketch"）。

0811 把 **spec kernel 的 tile 一起推到 UB 上限组合** → 5.94x。**两者都是未达标：三路径缺失 → 先按 §3.2 自查补 kernel；tile 保守/Phase 4 没跑 → 必须跑完下方 5 轮路径。**

#### ★ 最终必须达到的 spec tile 组合（UB 核算验证过，可直接用）

```python
# spec 形态（bs<=4 && d_main==512 && d_v==512 && n2==1）的 compute kernel：
BLOCK_G        = 32          # 0818 只到 16 → 差一倍并行度
SPEC_BLOCK_ROW = 512         # 0818 只到 256 → 差一倍行吞吐
SPEC_BLOCK_DQ  = 256         # 0818 只到 128 → 差一倍 D 吞吐
SPEC_BLOCK_DV  = 256         # 0818 只到 128 → 差一倍 V 吞吐
# 通用路径：BLOCK_ROW=128, BLOCK_D=128, BLOCK_DR=64（不变）
# gather：BLOCK_N=64/128(定向), BLOCK_D_GATHER=128, USE_SPAN_SELECT 随 bs
```

**UB 核算（spec，ascend910b，192KB）**：acc `2×[32,256] fp32` = 64KB（DV=256 时 acc0+acc1）+ K/V tile 与行块循环内复用。
⚠️ **以 0811 已验证为准，不要手算否定**：0811 注释明确"V tile [512,256]bf16 与已验证 K tile [512,256]bf16 同尺寸，UB 无新增风险"——BLOCK_ROW=512+DV=256 在 UB 内可行（配合 multibuffer 流水）。**直接抄 0811 已验证组合，不要自己重新核算而缩水。**

#### ★ Phase 4 达标路径（5 轮内，按序执行）

| 轮 | 动作 | 预期 |
|---|---|---|
| 1 | 首版写 5 kernel 三路径 + 显式 unroll（§3.2/L1.6b），tile **先保守**（G=16/ROW=256/DQ=128）保证编译通过 | 精度 72/72，0.18x |
| 2 | **spec tile 一次推到 UB 上限组合**：G=32/ROW=512/DQ=256/DV=256（**四参数一起动，禁止单点**） | 0.18 → ~0.9x |
| 3 | gather `BDG128` + 定向 `BLOCK_N=128`（bs==1 大形态） | +17% |
| 4 | PV bf16 cube dot + `combine BLOCK_D 256→512` | +5% |
| 5 | `multibuffer` 流水 + V 提前加载（§4.0b） | +2% |

> ⚠️ **为什么这次 BLOCK_G=32 回退了**：0818 opt_iter_2 把 BLOCK_G 16→32 时**没同步放大 DQ/DV**，单独放大 G 导致 UB 压力。**必须 G/DQ/DV/ROW 一起放大**（0811 的组合），不能单点动。
> ⚠️ **tile 收益不可手算**：0811 注释明确"V tile [512,256] 与 K 同尺寸无风险"——按 0811 已验证组合直接抄，不要自己重新核算而缩水。
> ⚠️ **首版不必推满**：0811 也是 Phase 4 逐轮加宽的（Opt-2-r3 等）。首版保守保证编译过，**Phase 4 第 2 轮必须一次推满四参数**——这是 5.0x 的破局点，不要在第 2 轮又只放大 ROW 一个参数。

### §4.1 首要方向：削减 workspace 往返

当前骨架的 K/V 每字节要走：`gather 读源 1 次 + 写 workspace 1 次 + attention 从 workspace 读 cdiv(G,BLOCK_G) 次`。
G=128、BLOCK_G=32 时是 **1+1+4 = 6 趟**。可考虑的方向（按风险从低到高）：

1. **加大 `BLOCK_G`**，直接减少 workspace 重读次数（4→2 即省 2 趟）。
   受 UB 限制：`acc` 是 `[BLOCK_G, BLOCK_DV] fp32`，先按 §1 G3 的方式记账再定。
2. **把 gather 融进 attention kernel**（去掉 workspace 的一写一读）。
   ⚠️ 融合后**必须把 L1.3 的 span-select 逻辑一并搬过去**，否则退化形态会吃 24 倍回退——
   这是本方向的主要复杂度来源，也是前一轮把两者拆成独立 kernel 的原因。
3. **V 的 workspace 可否省掉**：PV 阶段 V 的重读放大比 K 小。

> 本节 1/2/3 是**方向**，不是已验证结论。落地前必须按 §1 G4 的口径实测，
> 且实测必须排除 G2 的退化用例（用 BSND 组对照）。

### §4.2 已验证有效的基因（可直接采用）

| 基因 | 内容 | 实测收益 |
|---|---|---|
| `G_BF16DOT` | workspace 改输入 dtype（GM 带宽减半）+ QK^T bf16 输入 fp32 累加 | 基座级 |
| `BDG128` | `BLOCK_D_GATHER` 64→128（每 program chunk 搬运 17→9 次，256B 连续宽载） | 净 +0.172 |
| `SPEC_ROW` | 专用 compute kernel 的 `BLOCK_ROW` 256→512 | 净 +0.10 |
| `SPEC_DV` | 专用 compute kernel `BLOCK_DV` 128→256（acc 总量不变） | +5.7% |
| `combine BLOCK_D` | combine kernel `BLOCK_D` 256→512 | +1.6% |
| `multibuffer` | `tl.extra.cann.extension.multibuffer(t, 2)` 语句式标注，作用于 gather/attn 的全部流式 load tile | +1.6% |
| `PV bf16 cube dot` | split 路径 PV 走 bf16 cube dot | +3.6% |
| 定向 `BLOCK_N=128` | 仅 `bs==1 && sel_rows>=1024 && t1*n2*cdiv(sel_rows,128)>=160` 时启用 | 大形态 −1~−11% 耗时 |
| **独立 split kernel + 3D grid**（0811/5.94x 必备，0818 丢分主因） | split-KV 必须是**独立 `split_attn_kernel`**（`pid_s=program_id(2)` 行段）+ **独立 `split_combine_kernel`**（logsumexp 合并），3D grid `grid_split=(t1c*n2, cdiv(g,BLOCK_G), num_splits)` 启动；`base_par<160 && sel_rows>=512` 必走。**禁止**塞进 compute kernel 用 2D grid（"挂死"后整体禁用的做法 = 丢 3.3x） | 并行度不足形态（`base_par<160, sel_rows>=512`） |

### §4.3 ⛔ 证伪方向全表（不要重跑这些死路）

| 方向 | 实测结果 |
|---|---|
| 全量把 `BLOCK_N` 64→128 | 大形态改善，但 **t1==1 小形态并行度饥饿退化 ~+20%**，整体被拒；必须做成定向条件 |
| 关闭 span-select 以简化 gather | 退化形态 **0.02ms→0.53ms（24×）**，见 L1.3 |
| 运行时 `all_same` 嵌套 `if` 做单行坍缩 | 干扰 scattered 路径调度，`bs∈{2,4}` 用例**回退 10~14%**；必须用 constexpr 守卫 |
| 多页/多块合并加大 tile 摊薄 CV 同步 | 本类**直接撞 UB 墙**（同族分页形态实测三种写法全部 `PlanMemory Failed`） |
| gathered K 加 multibuffer 的某些变体 | 消融未复现 / UB overflow，已拒 |
| **把 split 塞进 spec/compute kernel 用 `IS_SPLIT` 分支 + 2D grid，遇到挂死就整体禁用 split（0818/7 批跑实锤）** | **1.79x vs 0811 5.94x（丢 3.3x）**。0818 把 split 并入 spec_attn_kernel（`IS_SPLIT` + `pid_s=program_id(2)`），"3D grid 挂死"后直接 `use_split=False` 禁用，丢掉 split-KV 并行度杠杆（`base_par<160 且 sel_rows>=512` 形态全靠它）。**正解（0811/5.94x）**：split 必须是**独立 `split_attn_kernel` + `split_combine_kernel`（logsumexp 合并）**，用 **3D grid** `grid_split=(t1c*n2, cdiv(g,BLOCK_G), num_splits)` 启动，IS_SPLIT constexpr 只在一个 kernel 内做直算/分段切换、不承载"整个 split 子流程"。split 挂死要修 grid/内存布局，不是整条路径禁用 |

---

## §5 精度闸门

1. **先过闸门再谈性能**：`verify.py` 全部用例通过之后，性能数字才有意义。
2. 掩码用有限极小值加在 scores 上（L1.7）；无效行写入一个**有限值行**（如窗口首行）
   而不是 0.0，配合 `p = exp(score - m)` 精确遮蔽，输出位级不变。
3. 每一次"纯调度 hint"（multibuffer / tile 尺寸切换）都要论证**数值位级不变**，
   并在改动后重跑全量 verify——本类用例多、形态杂，位级论证比抽查可靠。

---

## §6 与相邻 template 的差异声明

| 相邻 | 为什么本类不能落到它 |
|---|---|
| `block_sparse_attention.md`（行 1/1b 块跳过） | 该文件的头号红线是「选择表必须 device 侧建」（实测 host 建表虚高 2.09×）；本类的选择集合是**入参张量直接给的**，没有表要建，这条红线空转。且成本模型不同（见下）。 |
| `gqa.md`（二 压缩类·head 共享） | 成本模型是「`tl.dot` 个数 × KV 页迭代数」，本类实测 **gather(0 FLOPs) 占 50.5%、MFU 4.29%**，目标函数是 **GM 搬运字节数**，按 dot 个数优化收益是个位数百分比。 |
| `flash_attention.md` | FA 的主力手段「放大 tile 摊薄 CV 同步」在本类会撞 UB 墙；且 FA 假设 KV 是连续区间，本类 KV 是按索引表 gather 出来的任意行集合，没有"KV 迭代数"这个可压缩量。 |
