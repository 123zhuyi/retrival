# P2 语义转化 — histc（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-18｜依据：`extraction.md`｜**非低价值 → P2.5 降级执行①编译门+②正确性门**（③③.5④ 跳过，BLOCK 选型/瓶颈定型保持两态）。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| **`tl.atomic_add` lowering** | `linalg.generic`，**T2 级**（官方支持） | tl_op_lowering.md:49 [SRC]；zh `triton_api/Atomic_Ops/` 有 atomic_add/max/or 等完整 API 文档 [DOC] |
| 离散向量索引访存 | `tl.load` gather 标量发射，恒等索引即慢 ~24x；**`tl.atomic_add` 向量 scatter 重复地址丢更新（实测 1/8 因子）** | backend_contracts.md C-018 [EXP-skeleton][SRC] + histc ①门实测（2026-09-18，已回流契约） |
| int64 向量算术 | 4a：vadd/vcmp 不支持 i64 标量化；4e：int64 仅标量基址 | dtype_support.md 4a/4e [SRC] |
| 浮点除 | vdiv f16/f32/i32 安全 | dtype_support.md:52 [SRC] |
| scan/UB 静态预算 | 本算子无 scan/CallOp；普通 UB 预算适用（C-005） | backend_contracts.md C-005/C-022 |
| mask / 对齐 / multibuffer / persistent-core | 同前批 | C-002/C-010/C-011/compiler_options |

## 1. 分核策略 → grid（两路线不同构）

Ascend C 事实：x 均分核（formerLength=ceil(total/coreNum)，每核连续段）；y 清零均分+SyncAll；三档 LoadMode。
Triton 决策——**三路线，constexpr `HIST_IMPL` 切换；①编译门实测定案（2026-09-18）**：

1. ~~**`HIST_IMPL=0` GM 原子散加**~~ → ❌ **实测错值，禁用**：`tl.atomic_add` 向量 scatter 编译通过，但**向量内重复地址的更新被静默丢弃**（全同地址 1024→128=恰好 1/8，约 8 lane 冲突组；异地址 63/63 正确；910_9372+CANN 9.1.0）。与 C-018 同族，已回流契约。kernel 保留作证据，wrapper impl=0 直接 raise。
2. **`HIST_IMPL=1` 向量化 bin 归属矩阵计数（无原子）——主路径，实测全过**：grid=cdiv(bins, BLOCK_BINS)，每 program **独占** BLOCK_BINS 个 bin，循环扫全量 x tile：`cnt += tl.sum(idx[:,None]==bin_ids[None,:], axis=0)`，末段连续 store（**无原子、无冲突——bin 所有权唯一**）。实测 UB 约束：**BLOCK_X×BLOCK_BINS ≤ 128×64**（128×128 → ub overflow 1606400>1572864 bits，自动降级重编译仍失败）。bins 大或 numel 巨大时比较代价退化 → 路线 2。
3. **`HIST_IMPL=2` 逐元素 1 宽向量 atomic（= 官方档 2 GM_ATOMIC 真同构）——正确性兜底，实测全过**：persistent-core 分连续段，元素循环 `tl.atomic_add(y_ptr+idx, 1, mask=valid)`，idx 为 1 宽向量（无向量内重复地址，规避路线 0 坑；跨核并发由 GM 硬件原子保证，与官方 SIMT 同机制）。**注意：0-d 标量张量算术编译不过**（`() → tensor<1xf32>` materialization 失败），必须用 1 宽向量承载标量语义——已回流契约。
- 选型判定树：bins ≤ ~2048（矩阵 UB 可容纳多 program 划分）且 numel 不巨大 → 路线 1；bins 大或 numel 巨大 → 路线 2；路线 0 任何场景禁用。
- y 清零：wrapper `torch.zeros`（流内序，替代官方清零+SyncAll ❌ 项——Triton 不需要核间屏障）。

## 2. bin 计算 → 双域 constexpr 分支（IS_INT）

- **浮点域**（x=f32/f16→fp32）：`idx = int32((clamp(v,min,max) − min) · invRange · bins)`——与 kernel/golden **同序**（先减、乘 invRange、乘 bins、截断）；f16 load 后 `.to(tl.float32)`（COMPUTE_TYPE=float 同构）。
- **整型域**（x=i32/i64→**int64**）：`idx = int32((clamp(v,min,max) − min) · bins // range)`——`(c−min)·bins` 可超 int32，**int64 中间必须保留**（4a 标量化预警但官方 SIMT 本就逐元素标量，**非回归**，extraction §9.4）；i32 输入先 `.to(tl.int64)`。
- IS_INT 走 wrapper 传 `tl.constexpr`（C-023：dtype 本地分支不折叠）。

## 3. 特殊值 → 向量化 tl.where 链（仅浮点域）

```
is_pinf = v == +inf；is_nan = v != v；is_ninf = v == -inf
valid = ~(is_pinf | is_nan | (is_ninf & ~min_ninf & ~max_pinf))   # min_ninf/max_pinf 为标量
idx = where(min_ninf, bins-1, where(max_pinf, 0, clamp_idx))
```
⚠️ valid 第三项的 `~max_pinf` 是**按 golden 对齐**（kernel DoBin:72-74 在 max=+inf 时仍丢 −inf，源内矛盾，验收以 golden 为准——extraction §1/§9）。min=−inf/max=+inf 是标量级分支 → `tl.where` 广播即可，不进 uniform 控制流（避免 SIMT 化风险）。整型域无特殊值（无 inf/NaN 可表达），IS_INT 分支跳过整链。

## 4. 属性/输入面 → wrapper 标量化

- nbins/range 是 tensor 输入 + host tiling 值依赖（TilingInputsDataDependency）→ wrapper 按 Python int/float 标量接（GEIR 图模式常量场景）；kernel 内 bins/min/max/invRange 为**运行时标量参数**（不 constexpr 化——C-017 教训，标量运行传参安全）。
- range dtype==x dtype 的校验、min<max 校验 → wrapper assert。
- torch.histc 对应：语义对齐（等宽+越界归边），返回 dtype 差异（torch=输入同型浮点，本算子=int32）登记。

## 5. 片上内存 → 预算

- 路线 1（主）：比较矩阵 BLOCK_X×BLOCK_BINS——落地 (128,64)，i1→i32 累加；实测 **128×128 触发 ub overflow**（1606400>1572864 bits，自动关 code-motion/multi-buffer 仍失败）→ 硬上限 128×64。无 C-022 风险（无 scan）。
- 路线 2（兜底）：1 宽向量，UB 占用可忽略；瓶颈在逐元素循环（量级=numel/核数 次标量发射）。

## 6. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| x 均分核（formerLength 连续段） | persistent-core 连续段（路线 1/2） | ✅ 同构（C-011 范式） |
| 档 2 GM_ATOMIC（逐元素 GM 原子加） | 路线 2：1 宽向量逐元素 `tl.atomic_add` | ✅ 实测正确（②门 36 case 全过）；⚠️ 0-d 标量算术编译不过，必须 1 宽向量 |
| 档 2 的「向量 scatter」直译（路线 0） | `tl.atomic_add` 向量 scatter | ❌ **实测错值**（向量内重复地址丢更新，1/8 因子）——已回流契约 |
| 档 0 UB_FULL（UB 私有直方图+UB 原子） | 无可寻址 UB scratch | ❌ 结构性；路线 1 的 program 独占 bin 是**所有权等价**（不同机制） |
| 档 1 UB_NOT_FULL（分段重扫 x） | 路线 1 的 bins 分 program 天然覆盖 | ⚠️ 思想同构（读放大换容量），机制不同 |
| SetAtomicAdd+DataCopyPad GM 原子写回 | 路线 2 由 atomic_add 替代；路线 1 连续 store | ❌ 不需要（机制替代） |
| 清零+SyncAll | wrapper torch.zeros + 流内序 | ❌→不需要（等价保证） |
| SIMT 512 线程 + asc_ldcg + UNROLL=6 | 无对应 | ❌ 线程模型，不翻译（无语义影响） |
| DoBin 双域计算式 + 特殊值链 | §2/§3 constexpr+where 链 | ✅ 逐式同构（golden 比对闭环；⚠️ −inf/max=+inf 边界按 golden 对齐，源内矛盾登记） |
| int64 bin 计算 | int64 向量算术 | ⚠️ 4a 标量化 [SRC]，非回归（官方 SIMT 亦标量） |

## 7. 精度策略 → 验收

- 计数 int32 **精确相等**（无容差概念）——验收标准 = y 全等。
- bin 归属边界：浮点式与整型式截断行为不同——**勿跨域统一**；golden 同式比对。
- 特殊值 case（NaN/±inf/min=−inf/max=+inf/越界）必须单列验收。

## 8. P2 质量门自查

- [x] tl.atomic_add 取证 [SRC][DOC]，向量 scatter 风险标 ⚠️ 两态（不写「可用」结论，等①门）
- [x] ❌=4（UB scratch/SetAtomicAdd 写回/SyncAll/SIMT 模型）均有降级或「不需要」论证
- [x] UB 预算校核（无 CallOp，C-005 适用）；无 %；mask 整型；int64 仅 bin 计算中间域（标量化预警登记）
- [x] **P2.5 降级①+② 计划**：①编译门 = 两 HIST_IMPL × 配置真编译（重点：路线 0 scatter atomic 可编译性+正确性）；②正确性门 = `__main__` smoke（4 dtype × 2 impl × 特殊值组合对 golden，y 精确相等）
