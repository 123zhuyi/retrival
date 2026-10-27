# 逐层 IR 读法

> 拿到 IR 后怎么读。每层回答不同的问题，不要混。

## 层 1：ttir（前端语义层）

**回答**：用户 kernel 逻辑长什么样、constexpr 是什么值。

- 函数签名带 `!tt.ptr<bf16>` + `tt.divisibility`；计算是 `tt.load/tt.dot/tt.store`。
- **constexpr 值以 arith.constant 出现在函数体开头**：`%c4096_i32`、`%c128_i32` → 对应
  N_CTX/BLOCK 等。这是辨认缓存变体的主要依据。
- 文件头部 `#loc` 注释给出 MLIR 位置 → 源 .py 行号映射，全程有效。

## 层 2：ttadapter（适配层，送 bishengir 前）

**回答**：编译器"看到的意图"——tile 如何切、循环进位是什么。

- 指针变 `memref`；`memref.reinterpret_cast`（全局偏移切块）、`memref.alloc` + `memref.copy`（GM→本地）、
  `bufferization.to_tensor`、计算为 `linalg.matmul/linalg.reduce/linalg.broadcast` + `arith.*`。
- **循环进位（iter_args）直接可见**——判断 CV 流水能否生效的第一现场。FA 单流典型：
  ```mlir
  scf.for ... iter_args(%l = ..., %acc = ..., %m = ...) {
    %qk = linalg.matmul(q, kT)           // cube
    ... vector softmax（用 %m、%l）...
    %acc2 = arith.mulf(%acc, alpha)      // vector rescale（用 %acc）
    %acc3 = linalg.matmul(p, v, outs=%acc2)  // cube（定义新 %acc）
    scf.yield %l', %acc3, %m'
  }
  ```
- **CVPipelining 拒绝模式**（官方文档约束 2）：循环进位定义在尾部 vector work item、被头部
  vector 使用、中间隔着 cube → 文档称拒转。⚠️ **文档约束与实际行为可能不一致**（实测 FA 的
  m/l/acc 进位看似命中该模式，但 pass 实际完成了变换）——文档只作参考，**判定一律以
  `scripts/diff_pass.sh` 的 before/after diff 为准**。

## 层 3：HIVM / npuir（编译器决策层）

**回答**：编译器实际生成了什么——核拆分、buffer 换手路径、同步结构。

mix 内核拆成两个函数：`_mix_aic`（cube 核程序，`hivm.func_core_type = AIC`）和
`_mix_aiv`（vector 核程序，`AIV`）。两边独立阅读。

### 关键 op 速查

| op | 含义 | 关注点 |
|---|---|---|
| `hivm.hir.mmadL1` | L1 上的矩阵乘（cube） | `{b_transpose}`=K 转置在 cube 侧做；`sync_related_args` 挂事件 ID |
| `hivm.hir.fixpipe` | L0C→目标（UB 或 GM） | `outs` 的地址空间：**`<gm>` = dot 结果落 GM workspace（跨核换手走显存）**；nz2nd/nz2nz |
| `hivm.hir.nd2nz` | GM→L1 装载（带格式转换） | cube 输入路径；从 workspace 读回说明中间结果走了 GM |
| `hivm.hir.load/store` | GM↔UB（vector 侧） | vector 的访存 |
| `hivm.hir.set_flag / wait_flag` | pipe 间事件同步 | `[<PIPE_A>, <PIPE_B>, <EVENT_IDn>]`；成对出现 |
| `hivm.hir.sync_block_set/wait` | **跨核**（cube↔vector）同步 | mix 内核交接点；数量多=交接频繁 |
| `memref_ext.alloc_workspace` | workspace 分配 | `memref<NxMxKxf32, <gm>>`：N=多缓冲深度（multibuffer 级数实锤） |
| `annotation.mark {hivm.multi_buffer = N}` | 片上 buffer 多缓冲 | N=深度 |
| `scf.for ... {multibuffer_unroll_factor = N}` | 循环多缓冲展开 | N 来自 `--set-workspace-multibuffer`（安装版默认 4） |

### CV 流水生效的判定签名

对 `--mlir-print-ir-before/after=cv-pipelining` 做 diff（`scripts/diff_pass.sh`）：

- **未生效**：前后字节级相同（diff 0 行）。
- **生效**：单个计算循环被拆成多个按 core type 划分的子循环，例如 FA 双流版拆成 4 个：
  ```
  子循环1: mmadL1(QK 流A) + mmadL1(QK 流B)     ← 纯 cube
  子循环2: vmul/vreduce/vmax/vsub/vexp...       ← 纯 vector（携带 m/l 进位）
  子循环3: mmadL1(PV 流A) + mmadL1(PV 流B)      ← 纯 cube
  子循环4: rescale（携带 acc 进位）             ← 纯 vector
  ```
  即"每个子循环只含一种 core type 的 op，循环进位按 work item 分离"。

### 数据换手路径判定（UB 直通 vs GM 往返）

看 dot 结果的 `fixpipe` outs 与下一个 dot 输入的来源：

- **UB 直通**（快）：fixpipe → UB 地址空间，vector 处理后直接喂回 cube 输入 buffer。
- **GM 往返**（慢）：fixpipe → `alloc_workspace`（`<gm>`），vector 侧 GM load 处理后再 GM store，
  cube 侧 nd2nz 读回。**每多一次 GM 往返 = 多一次跨核 sync_block + 双倍显存流量**。

### 与 simulator 数据的对应关系

| IR 结构 | simulator 信号 |
|---|---|
| fixpipe → GM workspace | Cube FIXP pipe 占比高 + Vector MTE3 高 |
| sync_block / wait_flag 密集 | FLOWCTRL 占比高（WAIT_FLAG_DEVI） |
| multibuffer_unroll_factor=N | workspace `memref<Nx...>` |
| CV 流水生效（work item 拆分） | 总 cycles 下降、FLOWCTRL/MTE3 占比下降 |

## 读 IR 的顺序建议

1. 先看函数签名列表：`func.func` 有几个 → 是否 mix 拆分。
2. 看外层 persistent 循环与内层计算循环的层级（`scf.for` 嵌套）。
3. 内层循环体内按 op 顺序走一遍数据流：load → dot → 变换 → dot → store。
4. 数同步：set_flag/wait_flag/sync_block 的数量和位置 → 交接点密度。
5. 需要确认某 pass 是否起作用时，不要做静态阅读猜测——直接 before/after diff。
