# 93_MsaSparseAttnFwd 迭代日志

环境：CANN 9.1.0 runtime + triton 3.2.0 / **triton_ascend 3.2.2**
（自带 `backends/ascend/bishengir/bin/bishengir-compile 1.2.0`，优先级高于 PATH
上 CANN 的那份 —— 见 `_get_npucompiler_path`）。910B2C，24 AIC，UB 192KB。

## 基线状况

交付基线 `93_MsaSparseAttnFwd_generated.py` 历史 speedup **0.1868**（几何平均），
而且**过不了 AST 退化预检**：`_build_q2k` 里有 `torch.cumsum`，命中 Type 3。
所以它本身就不是合法交付物，必须重写。

基线的两个结构性问题：
1. forward 里用 torch 反推 q2k（`repeat_interleave` / `argsort` / `cumsum` /
   `index_put`）。profiling 里 `aclnnSort_SortAiCpu_Sort` 3.59ms +
   `aclnnRepeatInterleave` 4.15ms（全 case 累计），小 case 上这就是全部开销。
2. `grid=(total_q, Hq)`，每个 program 只算 1 个 query token、topK 个 128×128 KV 块，
   且用 `tl.sum(q*k, 1)` 手写 fp32 点积（Cube 闲置）。最大 case
   `total_q=32768, Hq=32` → 10^6 个 program、约 134GB 的 KV 访存，实测 505ms。

## 轮 v1_kcentric：k-centric + atomic 归约 —— **证伪，放弃**

思路：沿 CSR 行遍历 `(kv_head, kv_block)`，内层是"选中该块的 query 列表"，
KV 块每个 (kv_head, block) 只搬一次，访存从 O(total_q·Hkv·topK) 降到
O(Hkv·nblocks)。代价是单个 query 的 softmax 被拆到 topK 个 program，
需要两趟 + 原子归约（pass0 `atomic_max` 求 m，pass1 `atomic_add` 累加 l 与 acc）。

**结果：m 完全正确（maxdiff 0.0）、l 完全正确（rel 2.4e-7）、acc 恒为 0。**

逐个对照实验（全部在 case 1 上，用 `dbg93.py` 直接比对 kernel 的中间量 m/l/acc）：

| 变体 | 改动 | acc 结果 | 结论 |
|---|---|---|---|
| 原版 | `o = dot(ph,vt) + dot(pl,vt)` | **0** | — |
| dbg_a | 只留单段 `dot(ph,vt)` | **0** | 与二段拆分无关 |
| dbg_c | `o` 换成常量张量 1.0 | 非 0（正常写入） | **atomic_add 本身没问题** |
| dbg_d | A 操作数换成 `tl.full(...)` 常量 | 非 0 | dot→atomic 通路没问题 |
| dbg_f | A 换成 `(p*0.0+1.0)`（常量折叠掉 p 依赖） | 非 0 | 不是"provenance"问题 |
| dbg_q | A 换成**从 GM load 的** `qt` | 非 0 | **两个操作数都来自 load 就正常** |
| dbg_e | vt 移进内层 + dot 提前到 atomic 之前 | 0 | 与顺序无关 |
| dbg_h | dot 前插 `tl.debug_barrier()` | 0 | 挡不住 |
| dbg_k | `has_auto_blockify_blacklist_op=False` | 0 | 编译器自己的建议无效 |
| dbg_v | 关掉 multibuffer / ubuf_saving | **连 m、l 也错了** | 关开关只会更糟 |
| dbg_n / dbg_r | 把 atomic 换成 tl.store | bishengir **SIGABRT** | 无 atomic 时该 kernel 形状直接编译崩溃 |
| dbg_s | 转置形式 `dot(vT, trans(pT))` | **AI Core aivec timeout/trap** | 运行期崩 |
| dbg_u | ph/pl 经 per-program 小 scratch 走一遍 GM 再 load | 编译报错 | 绕不过去 |
| case 2 (G=1) | 原版 | 非 0 但仍错 | 与 G 展开无关 |

**结论（可复用）：triton_ascend 3.2.2 + bishengir 1.2.0 上，
kernel 内只要出现 `tl.atomic_*`，凡是操作数由本 kernel 内 Vector 运算
（`tl.exp` / `.to()` 等）产生的 `tl.dot` 就静默返回全 0；
两个操作数都来自 `tl.load` 时正常。**
`AutoBlockify disabled ... Unsafe ops: atomic operations` 这条 WARNING 是它的表征。
→ **Ascend Triton 上"attention 类 kernel + 跨 program 原子归约"这条路暂时不可用**，
需要设计成单 program 内闭合的归约。

## 轮 v2_qcentric：回到 q-centric，但把三件事做对

1. **不反推 q2k**：用一个无 dot、无 atomic 的 Triton kernel 把 CSR 直接展开成
   `rowsel[h_kv, block, q]` 的 int8 位图（布局把 q 放最内轴，于是 kernel 里
   取一个 tile 的选择列是一次连续 64B 读）。torch 侧算子全部消失。
2. **KV 预转置**：参考实现 `k[pages].reshape(nb*blk, Hkv, D)` 把页内 (Hkv, blk)
   两轴当成 (blk, Hkv) 重解释 —— 第 i 个 token 落在页内偏移 `(i*Hkv+h)*D`，
   行距 Hkv*D、每行只有 256B 连续。`k.view(P,BLK,Hkv,D).transpose(1,2).contiguous()`
   之后同一块 128 行变成连续 32KB。Hkv==1 时布局本来就连续，跳过。
3. **任务序 (kv_head, batch, q_tile, g)**：24 个核同一时刻工作在同一个
   (kv_head, batch) 上，该组合的 KV 工作集 = nblocks×64KB（最大 case 16MB），
   可整体驻留 L2；基线的 `grid=(total_q, Hq)` 相邻 program 落在不同 kv_head，
   是最差局部性。

精度：smoke（诊断用）case 1/3/4 的 MERE 5e-6 ~ 6e-6、max_abs 3e-2（bf16 ULP 量级）。
正式判定见 `iters/v2_qcentric/verify_result.json`。

---

## 关键口径澄清

`benchmark.py` 报的 `implementation.avg_latency_ms` **是几何平均，不是算术平均**
（v2 实测：报告值 3.0149，逐 case 算术平均 94.1179，几何平均 3.0149，中位数 2.4361）。
99 号那个 0.0868 ms 也是几何平均。所以"时延 < 0.1 ms"这个目标是对几何平均说的。

## 轮 v2_qcentric：正式精度通过

`verify.py` **50/50 全过**（`iters/v2_qcentric/verify_result.json`）。
impl 时延用官方 `benchmark.py --skip_framework --keep_res` 单独测
（**仅 impl 口径，不是正式 speedup 结论**）：**几何平均 3.0149 ms**。
分布极不均匀：case 1~10 是 0.02~0.17 ms，case 41~50 是 70~1362 ms。

## 轮 v4_nobranch：**假设被证伪**

假设：v2 慢是因为每个候选块一次 `if tl.max(sel) > 0` 的数据依赖分支
（Vector 算 tl.max → 同步给标量单元）把流水打断。
做法：把分支改成无条件执行（数值完全等价，非选中行本来就被 `sel` 掩掉）。
结果：case 49 **1472 ms vs v2 的 1386 ms —— 反而略慢**。
**证伪**：分支不是主要开销。

## 轮 v3_pertoken：**方向被证伪**

做法：改成"每个 (token, kv_head) 一个任务、只循环自己的 topK 个块"，
用无 dot 的 Triton kernel + int32 atomic 建 q2k 压缩表（q2k 正确性已单独校验：
`dbgq2k.py` 对 case 3 报 0/16 mismatch）。
结果（诊断口径）：case 20 **21.3 ms（v2 是 2.78）**、case 40 **954 ms（v2 是 54.5）**、
case 49 1327 ms（v2 是 1386）。**除最大 case 外全面变差。**
原因：任务数从 3072（case 40）暴涨到 98304，每个任务只算 GP=2 行 × topK 块，
tile 太小、每任务的标量前导开销占满。
**证伪**：逐 token 粒度即使把访存降到理论最小也赢不回 tile 变小的代价。

### v3 顺带修掉的一个真 bug（值得记）
`GP = max(16, next_power_of_2(G))` 会让 `hk*G + offs_g` 越过 `Hq`，
**masked load 仍然会算出越界地址**，实测在 case 3/4 上直接产生 NaN
（GP=16 崩、GP=4 正常）。正确做法是 `GP = next_power_of_2(G)`，
并且把行号 `tl.minimum(hk*G + offs_g, Hq-1)` 钳在界内、再靠 mask 丢弃无效行。

## 当前的量化判断（决定下一步）

以 case 40（q [4096,48,128]、Hkv=24、G=2、topK=16、L=4096、qlen=2048、B=2）为例：
- v2 的块迭代次数 ≈ 147456 次子迭代，实测 54.5 ms → **每次子迭代 8.9 µs**
- 但每次子迭代的理论量：Cube ~1.57M MAC（≈8 ns @200TFLOPS）、
  Vector ~41K element-op（≈4 ns）、访存 64KB 且 (b,kv_head) 工作集只有 2MB（L2 常驻）
→ **实测比理论慢约 18 倍，且与访存量无关 —— 是典型的"指令发射/同步开销主导"**，
不是带宽瓶颈也不是算力瓶颈。

所以下一步只有一条路：**把每次迭代做的事变大**（加大 tile、去掉 NSUB 拆分、
砍掉每次迭代的 Vector 趟数），而不是继续减少访存量。
已知约束：`BM=BN=128` 在本 kernel 上 **ub overflow 编不过**
（编译器已自动关掉 code-motion 和 multi-buffer 重试，仍然失败）。
