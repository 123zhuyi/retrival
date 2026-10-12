# P2 语义转化 — cumsum（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-17｜依据：`extraction.md`｜**非低价值 → P2.5 降级执行①编译门+②正确性门**（③ simulator/③.5 board/④ IR 跳过，BLOCK 选型与瓶颈定型保持两态）。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| **`tl.cumsum`/`tl.scan` lowering** | **`func::CallOp`——库调用，非向量化直译**；官方备注「手写循环 + 规约重写，或换算法（如分块前缀和）」 | tl_op_lowering.md:70 [SRC] |
| vcum* 指令 dtype | I1/I8/I16/I32/I64/F16/F32/BF16 可达；**i64 标量化**；A2/A3 非 i64 且 cum 维最内也标量化；950 1D dim0 i32→i64 有 SIMT 库例外 | dtype_support.md:165 [SRC] |
| 行序循环备选 | vadd f32/i8/i16/i32/u8/u16/u32/f16（bf16 升 fp32）——纯加法行序循环全 dtype 可行 | dtype_support.md:52 [SRC] |
| 跨核 SyncAll | Triton 无任何对应（单 kernel 无全局屏障） | ❌ 结构性 |
| int64 寻址/算术 | 4e：int64 仅标量基址；i64 向量算术标量化 | dtype_support 4a/4e [SRC] |
| mask / 对齐 / multibuffer | 同 gelu mapping §0 | C-002/C-010/compiler_options [SRC]/[DOC] |

## 1. 分核策略 → grid（跨核 ❌ 的降级设计）

Ascend C 事实：M×R×N 折叠；M（借 N）分核为主线；M 太小才借 R 跨核（CORE_SS，SyncAll 协议）。
Triton 决策：**扫描线级并行**——每个 program 负责 (m, n 条带) 的**整条 R**：
```
NUM_TASKS = M * cdiv(N, BLOCK_N)
grid = min(NUM_TASKS, num_vectorcore)   # persistent-core 连续段（mapping §1）
```
- 等价论证：官方主路径（1001/1011，M 分核 + 核内 R 全载或 UB_SS 分块进位）= 本设计逐字同构（核内 R 分块 carry 循环 = UB_SS 语义）；官方 CORE_SS 跨核档只在 M×N 并行度不足时启用——Triton 侧**放弃跨核并行**，登记为 ❌→降级：「M×cdiv(N,BLOCK_N) < 核数时核占不满，性能低于官方 CORE_SS；补救 = 两 kernel 分段前缀和（本卡不实现，§9 登记）」。
- ❌ 根因：Triton 单 kernel 无 `SyncAll()` 全局屏障（extraction §6.4）；两 kernel 方案需 host 两次 launch，超本骨架范围。

## 2. 扫描算法 → tl.cumsum 主路径（行序循环备选已实测废弃）

Ascend C 事实：Reg 级 Sklansky 扇出（log2(R) 趟，❌ 无对应）；twoway 折叠（布局工程，不翻译）。
Triton 决策：**块内 `tl.cumsum(axis=0)` + 块间 carry**——R 分 R_CHUNK 块，块内 scan、块首加 carry、carry += `tl.sum(tile, axis=0)`。语义 = UB_SS 的忠实映射。

**【P2.5 ①编译门实测，2026-09-17，Ascend910_9372 + bishengir】**
1. `tl.cumsum` → `func::CallOp` 库调用 UB 放大 ~48×：**tile 元素数 R_CHUNK×BLOCK_N 硬上限 1024**（16×64/32×32/8×128 过；16×128/32×64 起即 "ub overflow, requires 6324224 bits"——报错需求恒定 ~6.3Mbit，与 tile 尺寸解耦）。autotune 配置网格全部约束在 ≤1024。
2. **行序循环备选废弃**：逐行 `load→carry+=row→store(carry)` 在 UB 上构成 WAR（store 读 carry 与下一行 carry 更新），后端流水偶发同步不足 → 输出系统性前移一行、逐 run 非确定（小 case 三连跑 2/3 出错）。教训：**loop-carried UB 值在同一迭代内「先 store 后原地更新」的模式在本后端不可靠**；tl.cumsum 路径每迭代生成新 SSA 值 s，无此隐患。
3. Triton 前端：不支持 `or` 链（chained boolean）；`.is_int()` 本地 `if` 不做 constexpr 折叠（跨块 dtype 不一致报 Mismatched type）→ dtype 分支一律 wrapper 传 `tl.constexpr` 参数（IS_INT）。
4. **降序行地址 tile 后端存取错误**（reverse 初版 `rows = R−1−(c·R_CHUNK+rr)`：只首行写出且值为总和）→ reverse 改后缀和两遍法（§3）。

- Reg 级 Sklansky 本身 ❌ 不翻译（无寄存器布局控制）；其「log2(R) 趟树形求和」的舍入特征由 tl.cumsum 内部实现决定，与官方不同构但同量级（extraction §7 已登记）。

## 3. 属性面 → constexpr 两态

- **exclusive**：代数映射 `y = inclusive − x`（块内：`(carry + tl.cumsum(tile)) − tile`）。整型**精确**；fp 下与 golden 的精确移位存在 ≤1ulp 差（extraction §7）——卡片验收注记；实现零额外搬运，优于「移位+补 0」的显式 roll（Triton 无块内行移位原语）。
- **reverse**：初版降序行 tile（`r = R−1−(c·R_CHUNK+rr)`）**实测后端存取错误**（只首行写出、值为总和）→ **后缀和两遍法**（纯升序寻址）：pass1 累 `total=Σtile`；pass2 重算 inclusive 前缀 `s`，`out = total − s`（exclusive）/ `total − s + tile`（inclusive）。语义 = golden flip→cumsum→flip ✓；fp 下减法引入的抵消误差登记（长 R 大动态范围时 |total|≫|s_i| 场景精度两态）。
- 两属性均为 `tl.constexpr`（benchmark 属性编译期稳定场景，mapping §5 float 属性两态行类推）。

## 4. dtype 分派 → 单 kernel

- f16/bf16 → `.to(fp32)` 计算 → `.to(out)`（PromoteType=fp32 同构，cumsum_base.h:64-77；vcum*/CallOp 的 bf16 行为不透明，**fp32 域规避**）。
- fp32 直通；整型原生（i8/u8/i32 安全；**i64/u64：vcum* 标量化 + 4a 预警**，卡片性能两态）。
- axis tensor → wrapper 标量 int（aclnn V1 dim 语义；V2 的 axis tensor 取值后同）。
- 折叠 (M, R, N) = swiglu 同款 before/after 乘积；寻址 `m·R·N + r·N + n`。

## 5. 片上内存 → 预算（①编译门实测修正）

实测硬约束（见 §2）：**R_CHUNK×BLOCK_N ≤ 1024 元素**（tl.cumsum CallOp UB 放大 ~48×，恒 ~6.3Mbit 需求 vs 1.57Mbit 可用）。候选 (R_CHUNK, BLOCK_N) ∈ {(16,64), (32,32), (8,128), (4,256), (2,512)}，尾轴 BLOCK_N 32B 对齐（C-010：fp32→8 的倍数，全满足）。原静态预算（128×64 ≈ 64KB < 192/248KB）对普通算子有效、**对含 scan CallOp 的算子失效**——登记为后端取证新条目。

## 6. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| M 分核（借 N） | 扫描线级 persistent-core | ✅ 同构 |
| UB_SS 核内分块进位 | R_CHUNK 块 + carry 循环 | ✅ 同构 |
| 单向 Sklansky（log2 R 趟） | `tl.cumsum`（CallOp） | ⚠️ 语义等价，lowering/性能不透明 [SRC] |
| twoway 折叠 | 无对应 | ❌ 布局工程，不翻译（无语义影响） |
| **CORE_SS 跨核 SyncAll** | 无对应 | ❌ **降级：放弃跨核并行（§1）；两 kernel 方案登记不实现** |
| exclusive（错位+补 0） | `inclusive − x` 代数 | ✅（fp ≤1ulp 注记） |
| reverse（flag=−1 镜像） | 反向寻址 constexpr | ✅ |
| Cast NONE/RINT（fp32 域） | `.to(fp32)` / `.to(out)` | ✅ 必须保留 |
| 整型三 key 家族 | 单 kernel dtype 推导 | ✅（i64 标量化预警） |

## 7. 精度策略 → 验收

- fp32 中间域 + RTNE 回写（f16/bf16）；整型原生。
- 树形 vs 顺序求和的舍入路径差：容差按 ST acc=default；exclusive 代数差 ≤1ulp 单列。
- golden：np.cumsum + roll/flip（f16/bf16 升 fp32）。

## 8. P2 质量门自查

- [x] tl.cumsum lowering 取证 [SRC]，性能两态不写实测结论
- [x] ❌=3（Sklansky 寄存器级/twoway 折叠/跨核屏障）均有降级方案；⚠️ 项有出处
- [x] UB 预算校核；无 %；mask 整型；int64 仅标量基址
- [x] **P2.5 降级①+② 计划**：①编译门 = 两 SCAN_IMPL × 配置网格真编译；②正确性门 = `__main__` smoke（dtype × exclusive/reverse 全组合对 golden）
