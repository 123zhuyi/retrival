# P2 语义转化 — sum（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-18｜依据：`extraction.md`｜**非低价值 → P2.5 降级执行①编译门+②正确性门**（③③.5④ 跳过，BLOCK 选型/瓶颈定型两态）。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| **`tl.sum` lowering** | `linalg.reduce`，**T2 级**（官方成熟 lowering） | tl_op_lowering.md:39 [SRC] |
| i64 规约 | vreduce i64：**A2/A3 标量化**；950 不走标量兜底（模板库路径，实测前勿外推） | dtype_support.md:58/68/122 [SRC] |
| `tl.argmax/argmin` | linalg.reduce T2（本算子不涉及，登记） | tl_op_lowering.md:40 [SRC] |
| `%` 禁用 / C-001 | 分核解码用 `//` + `a-(a//b)*b` | backend_contracts.md C-001 |
| stride/shape 运行传参 | 不 constexpr 化 | C-017 |
| persistent-core 范式 | 小块海量任务默认 persistent-core | C-011 |
| 尾轴对齐 | VV 32B / CV 512B | C-010 |
| scatter atomic | **本算子禁用原子归约写法**（确定性 + C-024 错值双保险） | C-024 + README 确定性声明 |

## 1. 分核策略 → grid（两路线 + wrapper 归一化）

Ascend C 事实：任意 axes → 去 1 维/合轴 → **AR 交错 pattern** 统一抽象；A 轴优先分核、R 轴次之；**groupR>1 → GM workspace 两阶段跨核归约 + ScheduleMode=1 核间同步**（❌ 项）。
Triton 决策——**单 kernel A 轴全载（groupR=1 形态）为主路径**，每 program 独占若干输出元素、串行扫整条规约轴：

- **路线 T（尾轴规约，(M,R)→(M,)）**：persistent-core 1D grid=min(M, nvc)，program 循环 m，内层 R 分块 `acc += tile`，末 `tl.sum(acc)`。= 官方 pattern A/尾 A 且 A 不分核形态的同构。
- **路线 M（中间轴规约，(M,R,N)→(M,N)）**：persistent-core 1D grid=min(M·cdiv(N,BLOCK_N), nvc)，task 解码 `m = t // nb_n`、`nb = t - m*nb_n`（**不用 %**，C-001），向量化 N 维 tile，R 维**分块 2D tile (BLOCK_R, BLOCK_N) 累加**（尾轴 kernel 同构）。⚠️ 初版「标量 r 循环 + 每迭代换 base 的行 load」在 i32/N=128 实测**非确定错值**（同输入两次运行结果不同，单 program 复现；f16/i64 同构未触发）——疑似 C-021 同族流水失同步，已改 2D tile 结构并回流契约。= 官方 ARA 形态（A 轴全载）的同构。
- **wrapper 归一化**：axes 排序后若成连续段 [a..b] → 折 (M=∏shape[:a], R=∏shape[a:b], N=∏shape[b+1:])，N==1 走路线 T 否则路线 M；**非连续段 axes（如 4D 的 {0,2}）→ permute(kept+reduced)+contiguous 后折 2D 走路线 T**（= 官方 Contiguous 化路径的代价形态，登记：官方另有 NDDMA 免拷路径 ❌ 不可译）。
- **跨核两阶段 ❌ 不翻译**：单 kernel A 全载保证确定性与简单性；R 轴极大（>~10⁶）时核间并行度损失两态——卡片给「wrapper 两段式（partial+final，GM 中转=workspace 同构）」为后续优化方向，本骨架不实现。
- 空 axes（noop_with_empty_axes=True 默认）→ wrapper 早退直返 x（identity）；axes=None → 全轴（= 路线 T 的 M=1 形态）。

## 2. 计算域 → ACC constexpr（DAG 同构）

- **f16/bf16/f32 → fp32 累加域**（DAG `Cast<PromoteT>→ReduceSumOp<PromoteT>→Cast<T>` 同构；golden 升 fp32 佐证）——load 后 `.to(tl.float32)`，store 前 `.to(y_ptr.dtype.element_ty)` 降回。
- **i32 → i32 域**（def y 同型）；⚠️ GetPromoteType 定义缺省，官方是否升 i64 域未知——验收以 golden 为准；大 R 轴 i32 溢出风险登记（np/torch 默认升 i64）。
- **i64/bool → i64 域**：i64 直算（**vreduce i64 A2/A3 标量化 [SRC]——①门附取证编译/正确性，性能两态**）；bool wrapper `.to(torch.int8)` 后 kernel 内 `.to(tl.int64)`，y=int64（def 表；kernel BoolDag int64 域同构——aclnn 2²⁴ FLOAT 限流是接口层另一套策略，不对齐，登记）。
- ACC 走 wrapper 传 `tl.constexpr` 码（0=fp32/1=i32/2=i64，C-023：dtype 分支 wrapper 判好传入）。

## 3. 属性/输入面 → wrapper

- axes：Python int/tuple/None 接；负轴归一+排序（MakeWrapDim 同构）；空→identity；None→全轴。
- keep_dims：wrapper reshape 处理，不进 kernel。
- dtype 输出参数：遵循 GEIR def 表（bool→i64，其余默认同 x）；显式 dtype 时 **store 端 cast**——⚠️ 与 aclnn「先 cast 输入再规约」（aclnn:245-249）中间域不同，本卡对齐 GEIR kernel 层，差异登记。
- shape/stride（M/R/N/nb_n）**运行传参不 constexpr 化**（C-017）；contiguous 前置（非连续走 permute+contiguous 归一化路径）。

## 4. 片上内存 → 预算

- 路线 T：tile=BLOCK_R 元素（fp32 域 ≤4B），BLOCK_R=256 → 1KB 级，普通 UB 预算（C-005），无 CallOp。
- 路线 M：tile=BLOCK_N=128 元素向量 + acc 同形，亚 KB 级。
- 无 scan/无 dot/无 gather——preflight 高危项天然不命中。

## 5. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| 任意 axes → AR 交错 pattern（去 1 维/合轴） | wrapper 连续段折叠 (M,R,N) / 非连续段 permute+contiguous | ✅ 语义同构（官方 NDDMA 免拷路径 ❌ 性能机制不译） |
| A 轴优先分核 + R 轴分核 | 输出元素级 persistent-core（路线 T/M） | ✅ 同构（C-011 范式） |
| **groupR>1 跨核归约**（GM workspace 两阶段 + ScheduleMode=1） | 单 kernel A 全载（groupR=1 形态） | ❌ 核间同步结构性不可译；降级=A 全载（确定性保证反而对齐 README）；两段式为后续优化方向 |
| basicBlock/resultBlock/DB 流水 | BLOCK_R/BLOCK_N 固定小 tile | ⚠️ 机制推断（调度器半壳）；UB 预算远离上限 |
| Cast→ReduceSumOp→Cast（fp32 中间域） | load→fp32→tl.sum→store cast | ✅ 逐式同构（golden 比对闭环） |
| BoolDag i8→half→float→int64 | wrapper int8 + kernel int64 域 | ✅ 语义同构（三级链的中间精度对计数无语义影响） |
| useNddma 非连续搬运 | permute+contiguous 拷贝 | ❌ 性能机制不译（语义等价） |
| 空 tensor / 空 axes noop | wrapper 早退 | ✅ |

## 6. 精度策略 → 验收

- f16/bf16：fp32 中间域必须保留（长 R 轴直算会丢精度）；容差对齐 ST（f16 1e-3、bf16 1e-2 级）。
- f32：tl.sum 树形 vs torch 顺序累加，位级不保证——rtol 1e-4。
- i32/i64/bool：精确相等（i32 溢出场景除外，登记）。
- ②门覆盖：6 dtype × {尾轴/中间轴/相邻多轴/非相邻多轴/负轴/全轴/空 axes noop/keep_dims 两态/大 R 轴}。

## 7. P2 质量门自查

- [x] tl.sum 取证 T2 [SRC]；i64 vreduce 标量化标 ⚠️ 两态（①门附取证）
- [x] ❌ 项（跨核两阶段/NDDMA/ScheduleMode）均有降级论证；禁用原子归约写法（C-024 + 确定性声明）
- [x] 无 %（task 解码用 // 与 a-(a//b)*b）；stride 运行传参；ACC constexpr 分支
- [x] **P2.5 降级①+② 计划**：①编译门 = 路线 T/M × 3 ACC 码真编译（重点 i64 域 + bool 路径）；②正确性门 = `__main__` smoke（§6 覆盖面对 golden 容差/精确比对）
