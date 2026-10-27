# Pooling 类算子（MaxPool/AvgPool，2D/3D）优化经验

> 来源：27_MaxPool3d（ascend910b3，50 cases，2026-08-17 完成，geomean **1.1096x**，50/50）。
> 基础方法论：`.claude/skills/triton-latency-optimizer/references/ascend-pooling-optimization.md`
> （7-Phase 指南，Phase 3 一次生成即达 0.7249x 基线）。本文件记录指南之外的增量验证结果。

## §0 适用范围

- MaxPool3d / AvgPool3d / MaxPool2d / AvgPool2d，多 case（k/s/p/ceil/dtype 混合）
- 已验证场景：k∈{2,3,5}，s∈{1,2}，p∈{0,1,2}，dilation=1，return_indices=False，fp32/fp16/bf16

## §1 Layer 1：设计约束（硬性，无例外）

### L1.1 禁止 1D 扁平索引 + 向量 `%`/`//` 解码多维坐标（硬性）
- i32 向量除法在 Ascend 退化为标量循环。必须 2 级分块：标量 decode + 仅最内维向量化，
  用 `a - (a//b)*b` 替代 `%`（与 pooling 指南 Phase 1 一致）。

### L1.2 `tl.where` 归约累加必须显式 AND 全部 mask（硬性，2D tile kernel）
- 2D mask + `other=-inf` 隐式依赖 → 编译器产生**错误值**（指南 Phase 7 已文档化）。
- ✅ `acc = tl.where(wv_2d & cm_2d & (val > acc), val, acc)`；❌ `tl.where(val > acc, val, acc)`。

### L1.3 禁止 `tl.reshape` + `tl.max(axis=...)` 做相邻偶对归约（硬性，本任务实测）
- 实测一：masked load（任意 other）+ reshape + max(axis=1) → **全错值**。
- 实测二：循环累加张量仅被 reshape 读取即**损坏源张量寄存器**（isolate: /tmp/k2s2_dbg4.py 模式）。
- 实测三：循环内对新 load 立即 reshape 也错。**任何形式的循环上下文 reshape 归约均不可信**。
- 替代：2D offset load `(2, BLOCK_W)` + `tl.max(axis=0)`（功能正确，但性能极差，见 L1.4）。

### L1.4 k=2 s=2 p=0 禁止走 NCDHW 直读偶对归约路线（硬性，失败路径勿重试）
- 2D offset strided load + axis=0 reduce 退化为离散访存，实测 **5-40x 劣化**
  （case [1,64,8,224,224]: 0.19ms→7.88ms）。该组留在 NDHWC 泛用路径。
- 与历史记忆 "EVEN_C/FLAT_W 失败路径" 一致。

### L1.5 s==1 且 max(K)>=3 必须用 3-pass 逐轴可分离 max（硬性，本次最大杠杆 +53%）
- max 可分离：`max(kd,kh,kw)` = 逐轴复合。读放大 K³ → 3K（k=3: 27→9，k=5: 125→15）。
- 路由条件：`SD==SH==SW==1 and max(KD,KH,KW)>=3`；k=2 s=1 组无收益（8→6+2 次中间写），保持泛用。
- 中间张量保持 NDHWC，三次 pass 间无需 permute；max 无舍入，任意 dtype 逐位等价。

### L1.6 ceil_mode + 奇数维度时窗口末位越界必须处理（硬性）
- ceil 可使 `s*OD = D+s-1 > D`：D/H 维靠**标量分支**（`if di < D`）跳过越界 iter；
  W 维用 **clamp 代替 mask**（`min(pos, W-1)`）——W 为奇数时唯一越界位 pos=W 恰 clamp
  到同窗口首元素（W-1=2(OW-1)），重复值不改变 max，-inf 填充语义精确保持。

### L1.7 双 kernel/多 kernel 分发必须在类外函数（硬性，AST 闸门）
- forward() 内直接 if/else 多次 kernel 启动触发 AST "2次启动" FAIL。
- dispatch 放模块级 `_run_kernel()`（注意：`self._route` 方法形式在其它任务实测触发 AST FAIL，勿用）。

### L1.8 1D 流式归约 kernel 可用 `tl.maximum` 替代 `tl.where` 链（推荐）
- IR 证据：`tl.where(mask & (val>acc), val, acc)` 每迭代产生 vcast+vcmp+vand+vsel+**ub→ub copy 回写**。
- `tl.load(other=-inf)` 已保证无效 lane 为 -inf，`tl.maximum(acc, val)` 数学等价且单指令。
  （仅限 1D axis kernel；2D tile kernel 仍受 L1.2 约束。）
- ⚠️ bf16 的 `tl.maximum` 结果提升到 fp32 → acc 必须按 dtype 条件化
  （`tl.float32 if element_ty==tl.bfloat16 else element_ty`），否则 scf 类型重定义编译错。

## §2 Layer 2：算法骨架

### L2.1 三路分发骨架（host `_run_kernel`）
```
if s==1 且 max(K)>=3:  3-pass 可分离 axis_max（D→H→W，NDHWC 中间张量）
elif p==0 且非 ceil:   pool_kernel_nopad（NDHWC 2D-tile，无边界检查）
else:                  pool_kernel_pad（NDHWC 2D-tile，clamp+cmp W 边界 + 标量 D/H 分支）
```

### L2.2 axis_max 单轴 kernel 骨架
```
x 视图 [ROWS, I, TAIL] contiguous；out [ROWS, O, TAIL]
块 = (输出行 r, tail 块 tb)；向量维 = TAIL（BLOCK≤2048 足够，4096 无增益）
for k in range(K):            # K constexpr，range() 不展开（策略 C）
    ii = oi + k - P           # s=1；标量
    if 0 <= ii < I:           # 标量分支
        acc = tl.maximum(acc, tl.load(x + n*I*TAIL + ii*TAIL + t, mask=t<TAIL, other=-inf))
store(out + r*TAIL + t, acc, mask=t<TAIL)
```

### L2.3 BLOCK 选择（沿用指南 Phase 6，已验证）
- BLOCK_C: C≤128 → C（exact-fit）；否则 128。
- BLOCK_W: max_bw=32(fp32)/64(fp16,bf16)；OW≤4 保底 8；除数策略 ≥0.75·max_bw。
- axis_max BLOCK: min(2048, next_pow2(TAIL))；floor 8。

## §3 Layer 3：关键技巧

- 可分离 pass 视图链：pass1 (N, D, H*W*C) → pass2 (N*OD, H, W*C) → pass3 (N*OD*OH, W, C)，
  输出即最终 NDHWC，入口/出口各一次 permute。
- 负载均衡：`bp=ceil(total_blocks,nvc); sb=pid*bp; eb=min((pid+1)*bp,total)`，块区间连续。
- IR 提取陷阱：单进程跑多个 kernel 路径触发 507057 SUSPECT REMOTE ERROR；
  须拆成单 kernel/单 case 一次提取，且 dump 可能被 torch_npu 内部 kernel 抢占。
- benchmark 基建抖动：profiler 偶发 "无法提取有效时延数据" / vector core timeout（EE9999），
  重跑即恢复，勿误判为 kernel bug（先看失败点是否跨轮一致）。

## §4 性能基准（27_MaxPool3d，ascend910b3，50/50）

| 阶段 | geomean | 说明 |
|------|---------|------|
| Phase 3 基线（指南 7-Phase 全应用） | 0.7249 | NDHWC+2D tile+双 kernel+BLOCK 策略 |
| +opt_iter_0（L1.5 可分离 3-pass） | **1.1096** | s=1 组 17 cases，最差 case 0.144→4.3 |
| opt_iter_1（L1.4 k2s2 NCDHW 直读） | 0.6805 ❌拒绝 | 2D-offset load 40x 劣化 |
| opt_iter_2（axis_max BLOCK 4096） | 1.1068 ≈持平 | 无收益 |
| opt_iter_3_ir_1（tl.maximum 替代 where 链） | 1.0982 ≈持平 | 指令减半但纯带宽流无收益 |

## §5 陷阱表

| 问题 | 症状 | 正解 |
|------|------|------|
| masked load + reshape + max | 全 -inf/全错值 | L1.3：禁 reshape 归约 |
| 循环累加张量被 reshape 读取 | acc2 变垃圾浮点 | 同上，reshape 污染源寄存器 |
| bf16 tl.maximum | scf 类型重定义编译错 | L1.8：acc dtype 条件化 |
| k2s2 NCDHW 偶对归约 | 5-40x 劣化 | L1.4：留 NDHWC 泛用路径 |
| ceil+奇数 D/H 窗口越界 | 读到邻通道垃圾，精度 FAIL | L1.6：标量分支 + W clamp |
| `_route` 方法分发多 kernel | AST "2次启动" FAIL | L1.7：模块级 `_run_kernel` |
| IR 提取多 kernel 单进程 | 507057 设备错误 | §3：单 kernel 单次提取 |
