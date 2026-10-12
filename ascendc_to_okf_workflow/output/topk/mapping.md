# P2 语义转化 — topk（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-18｜更新：2026-09-20（v3 发射级迭代落地 + ②门实况回流）｜依据：`extraction.md`｜**非低价值 → P2.5 降级执行①编译门+②正确性门**（③③.5④ 跳过，BLOCK 选型/瓶颈定型两态）。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| `tl.max`/`tl.argmax` lowering | `linalg.reduce` T2 | tl_op_lowering.md:39-40 |
| `tl.sort`/`tl.topk` | 部分 bitonic top-k（triton_patch）→ 后端模式匹配 `hivm.hir.vsort`——**仅 f16/f32**，int64 键编译失败（预取证 2026-09-18，①门正式回流） | 探针 probe4 |
| `tl.argmax` int64 键 | 编译 >7min CPU 且错值；int32 正常（预取证，①门回流） | 探针 probe5/6 |
| **C-024 atomic** | 不命中（无 atomic） | — |
| `%` 禁用 / C-001 | 两行段 task 解码用 `//` + `a-(a//b)*b` | C-001 |
| stride/shape 运行传参 | k/N/numTiles 运行传参；LARGEST/IS64/SEGMENT constexpr（C-023） | C-017/C-023 |
| persistent-core | 行主语小块任务默认 persistent-core | C-011 |
| i64 vreduce | A2/A3 标量化——本路线 i64 max-reduce 沿用（sum 卡已证正确） | dtype_support.md:58 |

## 1. 算法路线判定树（P2 核心决策 → v3 落地修正）

| 候选 | 判定 | 理由 |
|---|---|---|
| ① packed-key `tl.topk`（值+索引打包单趟排序） | ❌ **废弃**（后端限制） | vsort 仅 f16/f32（C-027）；int64 键编译失败；int32 键装不下 32bit 值键+索引（f32 值键已占 32bit） |
| ② `tl.sort` f32 值域 + 二次寻址 | ❌ 不采用 | 值排序后索引需回查，重复值 tie 破歧困难；且 f32 sort 正确性未证（留 sort 卡） |
| ③ **迭代 max+argmax 选择（逐趟掩码）** | ✅ **主路线** | probe7 端到端 exact（int32/int64）；天然携带索引；tie 左most；NaN 键映射天然正确；无排序网络编译爆炸；官方 MULT_CORE_OPTIM 二段结构同构 |
| ④ 官方 radix 多趟直译 | ❌ 不译 | bin 计数+前缀和+按位重排为硬件向量化特化，无 Triton 对应；语义由 ③ 覆盖 |

**⛔ v3 落地关键修正（本轮 ②门实况回流）**：
- **迭代须在 wrapper 发射级（k 次 kernel 发射 + k 次 host 同步 + torch scatter 掩码），不能放 kernel 内动态循环**。v2 kernel 内 `scf.for` 动态界 × 归约 → (BLOCK=512,k=32) 设备侧挂死（vector core timeout 507034，后端 loop 代码生成缺陷类，见 mapping §2 注）。v3 无环单选 kernel（直列码）probe6/7 端到端 exact。
- 单发射粒度：一行一 program（grid=M），**k 次发射**；选中位经 `torch.scatter_` 改写 x_work 沉底（largest→type-min，即用 `tl.where` 掩码语义的发射级等价）。性能两态（~ms 级 launch+sync），正确性优先。
- **指标：单行段 N≤BLOCK_N 全 dtype × dim 三态 × 双 largest 已②门全绿**（N=16 也是 BLOCK_N=16，dim=0 等非尾轴经 movedim+contiguous 化后无步幅错读，探测 14 实证）。

## 2. 主 kernel 结构（v3 无环单选 kernel，发射级迭代）

```python
@triton.jit
def _pick_kernel(x_ptr, v_ptr, i_ptr, N, j, sv, si,
                 LARGEST: tl.constexpr, DT: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)                    # 一行一 program（grid=M，无动态循环）
    offs = tl.arange(0, BLOCK_N); msk = offs < N
    x = tl.load(x_ptr + pid.to(tl.int64) * N + offs, mask=msk, other=0)
    cur = sortable_key(x, DT)                 # 每 dtype 族映射（§3），int64 无符号保序键
    if not LARGEST: cur = ~cur                # constexpr 序反转；NaN 两向均正确
    sel = tl.where(msk, cur, tl.full((BLOCK_N,), -(2**63), tl.int64))
    mx = tl.max(sel, axis=0)                  # int64 max reduce（标量化，probe7 证正确）
    m = tl.argmax(tl.where(sel == mx, BLOCK_N - offs, 0), axis=0)   # 左most；返回块内 lane
    v = tl.sum(tl.where(offs == m, x, 0))     # 取 m 处值（x[m] 标量索引不支持，sum 全零归约取单）
    tl.store(v_ptr + pid*s_v + j, v); tl.store(i_ptr + pid*s_i + j, m)
# wrapper：for j in range(k_val):  _pick_kernel[(M,)](x_work,...);  x_work = scatter_(x_work, idx[:,j], type_min)
```

- **无 `for j` kernel 内循环**（v2 挂死类规避，见 §1 注）；k 趟 = k 次发射 + k 次 host 同步。
- `offs == m` 里 offs 是**块内 lane**（一行一 program 时恰=全局轴内下标，dim=-1 尾轴）；跨 tile 版本（两行段 §4）须用 `tl.arange(0,BLOCK_N)==m` 取局部值、`tb*BLOCK_N+m` 拼全局索引（smoke5 两行段抓获的 offs==m 全局比对事故）。
- **两行段 stage1 同理**：`_pick_tile_kernel` 一 (row,tile) 一 program，`_pick_merge_kernel` 归并候选。

- 值输出无计算域提升：v 原值直存（dtype=x）。
- sorted=False 亦输出有序（超集合法，golden 比较器先排序再比）。
- k=0 早退 / k=N 等价全排（走 sort 卡，登记）/ k 越界 assert。

## 3. 全序键映射（constexpr DTYPE 族分支，wrapper 传码）

| x dtype | 键构造（→signed-order，NaN/Inf 保序） |
|---|---|
| f32 | `b=x.bitcast(i32); u=(b&0xFFFFFFFF 零扩展后) ^ (b<0 ? 0x7FFFFFFF : 0x80000000)`；NaN→键最大；容器 int32（再 ^0x80000000 转 signed 序，或全程 int64 零扩展域——probe5/7 实证 int64 容器稳妥） |
| f16/bf16 | 同 f32，位宽 16（零扩展 int32） |
| i32 | 恒等（int32） |
| i8/i16 | 符号扩展 int32 |
| u8/u16/u32 | 零扩展 int32（^0x80000000 转 signed 序） |
| i64 | 恒等 int64（max int64） |
| u64 | `bits ^ INT64_MIN`（golden 同构） |

统一实现：把值位模式零扩展到 64bit 无符号序键 `u64`（浮点先 sign-flip 到无符号保序、NaN 置全 1），然后容器内 `us = u64 ^ SIGN_BIT64` 得 signed-order。≤32bit 值键高位全 0，int64 容器比较正确（probe5 首两选正确即证）；i64/u64 天然 64bit。**单一代码路径**：全部 int64 键 + int64 max-reduce + int32 flag-argmax（`where(sel==mx, BLOCK-offs, 0)` 左most，probe7 loop64 证）。代价：i32/f32 也走 64bit 容器（寄存器×2，性能两态）；优点：单 kernel 全 dtype、无 constexpr 键宽分支。BLOCK_N 上限相应下调（int64 cur + x + sel ≈ 3×8B×BN → BN≤2048 安全）。

## 4. 大 N 两行段（N > BLOCK_N，官方 MULT_CORE_OPTIM 同构）

```
stage1: grid 任务 = M × ceil(N/BLOCK_N)；task 解码 // + a-(a//b)*b（C-001）
        每 tile 同 §2 选择 min(k, BLOCK_N) 候选 → workspace[候选值(dtype=x), 全局索引(i64)]
stage2: 每行载入 T×k' 候选（pad 到 BLOCK_N2=next_pow2(T·k') ≤2048）→ 同 §2 再选 k 趟
        值取候选值；索引=候选携带的全局索引（gather from ws-index）
```
- 二段均复用同一选择环；stage2 多一个 idx-src gather（`tl.load(wi + m)`）。
- workspace 布局 [M, T, k'] 连续；T·k' 超 BLOCK_N2 时登记降级（N 超 ~4M 场景，sort_and_top_k 域）。

## 5. wrapper 归一化

```python
topk(x, k, dim=-1, largest=True, sorted=True, indices_dtype=torch.int32)
```
- k：tensor→`int(k.reshape(-1)[0])`；assert 0≤k≤N；k=0 空输出早退。
- dim：负轴归一 + movedim 到尾轴 + contiguous（官方 non_last_small 免转置=性能机制不译）；折叠外层 M。
- LARGEST/IS...：constexpr 传 LARGEST 即可（键路径统一 int64）。
- indices_dtype：i32/i64 输出分配，store 端 cast。
- sorted：忽略（恒有序输出，超集登记）。
- M=0 早退。

## 6. BLOCK / 预算

- BLOCK_N = min(2048, next_pow2(N))（int64 容器 UB 预算，C-005；①门实测上限）。
- 每 program 一行（BM=1）：cur/sel int64 2×8B + x 4B + 键中间 ≈ 24B×2048 ≈ 48KB，普通预算。
- 无 dot/scan/atomic——preflight 高危项天然不命中；% 仅 stage1 task 解码（已代数规避）。

## 7. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| radix 多趟（b8~b64×NUM_PASS） | 迭代 max+argmax 选择 | ⚠️ 算法不同、语义同构（exact 索引+tie+NaN，②门验） |
| SmallBitonicSort 32 槽网 | 同左选择环 | ⚠️ 网络不同、输出序同构（严格比较稳定→左most tie，golden 证） |
| MULT_CORE_OPTIM 二段（局选+归并） | stage1+stage2 两行段 | ✅ 结构同构 |
| SORT_AND_TOP_K（≥1e7 全排截断） | 不走（k≪N 场景）；k≈N 走 sort 卡 | ⚠️ 登记 |
| NON_LAST_SMALL_AXIS 免转置 | wrapper movedim+contiguous | ❌ 性能机制不译（语义等价） |
| sort_with_index 联合排序基座 | 候选 (value,idx) 对搬运 | ✅ 同构 |
| AscendC::Sort 库精排 | 选择环末段无需精排（直接有序输出） | ⚠️ 半壳，机制推断 |
| 五模式 tilingkey 分派 | 单 kernel + 两行段自适应 | ⚠️ 分派不译（shape 自适应覆盖） |
| k tensor 输入 / indices_dtype / largest / sorted | wrapper 全复刻 | ✅ |

## 8. 精度策略与验收

- 值：无精度变换，exact 相等（整型）/ 位级相等（浮点，同输入值直拷）。
- 索引：k≤32&&sorted → **exact 比对**（含 golden 五注入模式的 tie/NaN 期望）；k>32 → 语义比较（合法+gather+无重复）+ 唯一值 case exact。
- 覆盖：7+ dtype 族 × {largest×sorted} × k{1,2,7,8,16,32,33,N/2} × N{小64,中1536,非2幂700,大>2048 两行段} × 五注入模式 × dim{尾,中,首} × indices_dtype{i32,i64}。
- 基准：CPU torch.topk；uint 走 view+stable sort golden 路径；NaN case 对照 golden `_value_better`。

## 9. P2 质量门自查（v3 修订）

- [x] tl.max/argmax T2 [SRC]；vsort/argmax-int64 限制已预取证 → ①门正式回流 C-027/C-028
- [x] ❌ 项（radix 网络/五模式分派/免转置）均有降级论证；主路线 probe6/7 预验证 exact
- [x] 无 %（stage1 用 // 与 a-(a//b)*b）；stride 运行传参；LARGEST constexpr；无 atomic
- [x] **P2.5 降级①+②**：①编译门 = 全 dtype 键映射×LARGEST×N 三档真编译 + f32 tl.sort 附带验证（sort 卡伏笔）；②正确性门 = `__main__` smoke（对 CPU torch.topk）。**实况：单行段全 dtype × dim 三态 × 双 largest ②门全绿（含非 2 幂 N、k=N、注入模式）；两行段 N>BLOCK_N 与 u64 两处边界受限**（见 P4 报告，涉及旧缓存编译残留与 NPU scatter_ dtype 限制，均已在代码层修复、未重验）。
