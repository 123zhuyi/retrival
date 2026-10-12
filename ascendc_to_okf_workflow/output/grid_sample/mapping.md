# P2 语义转化 — grid_sample（Ascend C → Triton）

> 日期：2026-09-17 ｜ 输入：extraction.md（P1）｜ 产物：`triton_skeleton.py`。
> **核心决策**：gather 类纯 Vector 算子（无 tl.dot）；程序粒度 = 输出 tile
> [BC, BHW]；**消解 CANN 侧两个结构性开销**（NCHW 标量 gather 的逐 c 循环、
> 4 角点 GM atomicAdd）——坐标每 hw 块算一次被 BC 个通道复用，4/16 角点片上
> 累加后单次 store，零 atomic。

## 0.5 检索记录（六任务，证据全标）

| # | 任务 | 命中层 | 结果 |
|---|---|---|---|
| 1 | 算子类归属/grid | backend_facts ✅ | operator_class_patterns §0：无 tl.dot → **Vector 类**；§1 复杂算子组织法（多重 mask 分别命名只在 load/store 组合） |
| 2 | tile/UB | backend_facts ✅ | 无常驻状态——UB 压力远低于递推族；§1 UB 经验式参考；起点 BC×BHW 走 P2.5 编译门+board 门定 |
| 3 | dtype 可行性 | backend_facts ✅ | fp16/fp32 套件；坐标/权重 fp32 域（dtype_support §3：浮点数学 F32 全支持）；**gather 索引禁 int64**（dtype_support §3 注 3——索引用 int32，基址 int64 标量） |
| 4 | 正确性陷阱 | backend_facts ✅ | C-001（禁 %：reflection 的 flips%2 用 `f−(f//2)*2`）；C-002/4c（load mask 用整型比较）；**C-009 分离写法：zeros 的越界权重清零是算术语境 → fp32 比较合规且更优** |
| 5 | 寻址纪律 | backend_facts ✅ | 4e：int64 仅标量块基址；C-017 stride 运行时传参；corner 地址 = n64·CHW + c·HW + y·W + x（块内 int32） |
| 6 | wrapper 约束 | backend_facts ✅ | AST checker 禁 max/min/while（B4 iter_0 教训）——纯三元守卫 |

六任务全命中首层。interpolate skill 文档（`ascend-interpolate-optimization.md`）
已核对：其 P0 级整数倍上下采样特化**本套件不适用**（H_out==H_in 恒成立），
可借的是「分派架构 + 行预载 next_pow2 constexpr」思想（本骨架 grid 行按
BHW pow2 块载，同型）。

## 1. grid 设计

```
grid = (N · cdiv(C, BC) · cdiv(H_out·W_out, BHW),)   # 一块一 program 领一个输出 tile
program = (n, c_blk, hw_blk)：tile [BC, BHW]
```

- **坐标计算每 program 一次**（gx/gy load + 反归一化 + padding + 角点/权重），
  被 BC 个通道复用——CANN general 版逐 channel-block 重复（其 channelLoop
  内每次重 gather），这里是第一个结构优势点；
- 5D 走独立 kernel（trilinear 8 角点 / nearest 1 角点；套件仅 3 个小 case）。

## 2. BLOCK 超参（起点假设，P2.5 两门实测标定）

| 参数 | 起点 | 依据 |
|---|---|---|
| BHW | 256（pow2，对齐 C-010 32B 的 fp32 倍数） | 输出点块；UB 无常驻压力 |
| BC | min(next_pow2(C), 32) | 角点地址 c·HW 共享坐标；[BC,BHW] fp32 acc ≤ 64KB |
| MODE/PAD/AC/IS_3D | constexpr 分派 | 运行期 flag 编译期消化（对应 template 臂 L1.2 同款纪律） |

## 3. 语义映射（kernel ↔ CANN/torch 锚点）

| 语义 | Triton 写法 | CANN 锚点 |
|---|---|---|
| 反归一化 ac=False | `x = (gx+1)·0.5·W − 0.5` | grid_sampler_2d.h:626-630 |
| ac=True | `x = (gx+1)·0.5·(W−1)` | :623-624 |
| zeros | 角点 int 越界 mask → load other=0（等价权重清零） | :154-157,:532-534 |
| border | 四角点各自 clamp [0,W−1]/[0,H−1] | :204-230 |
| reflection | `t=|x−min|`、`flips=floor(t/span)`、`extra=t−flips·span`、`where(flips 偶, extra+min, span−extra+min)`；ac 相关端点（ac=1: [0,2(W−1)]；ac=0: [−1,2W−1]） | :280-354,:242-249 |
| bilinear 权重 | nw=(x1−x)(y1−y) 等四式 | :668-676 |
| **nearest = rint(x')** | `libdevice.rint`（**银行家舍入**，非 floor(x+0.5)！） | grid_sampler_2d_nearest.h:1039-1040（CAST_RINT） |
| **bicubic A=−0.75** | cubic1: ((A+2)t+(A−3))t²+1（\|t\|≤1）；cubic2: ((At−5A)t+8A)t−4A（1<\|t\|≤2） | grid_sampler_2d_bicubic.h CubicConvolution1（alph=2−0.75, beta=0.75−3） |
| floor 取角 | `tl.floor`（负坐标安全，勿 truncate） | :646-647 CAST_FLOOR |
| 计算域 | 坐标/权重/累加全 fp32，store 前降回输入 dtype | CANN 同款（P1 精度节） |

## 4. 显式丢弃 / 不可达

| CANN 机制 | 处置 | 理由 |
|---|---|---|
| NCHW 标量 gather（GetValue/SetValue 逐元素） | 不复刻 | Triton 向量 gather load 天然替代；其标量化是 AscendC 侧的保守写法 |
| 4 角点 GM atomicAdd 累加 | **结构性消解** | 单 program 持完整 [BC,BHW] tile，片上累加单次 store |
| slide_window 滑窗复用输入行 | **本套件无效**（grid 全随机 U[-1,1]，无行局部性红利）——golden `_random_grid` 实证 | 不内建；列为 ⚠️ 观察项（真实业务 grid 若有平滑性可复活） |
| fullLoad 变体（全输入载 UB） | 不采用 | 仅 NHWC+小输入触发（tiling.cpp:182-197），套件 NCHW 不适用；且输入最大 1120²·112·4B=561MB 远超 UB |
| TransDataTo5HD 输出转置 | 不需要 | 那是 CANN 从 gather 布局到写出布局的修补；本骨架输出天然 [BC,BHW] 连续 |
| channelLast/NHWC、310P/310B/kirin/950 变体 | 丢弃 | 套件不涉及 |

## 5. 骨架已知风险（P2.5 验证目标）

1. **向量 gather 的 lowering 形态未知**：2D gather（[BC,BHW] 离散地址）在
   triton-ascend 是向量发射还是标量退化——P2.5 ③ simulator 门直接看
   MTE2/SCALAR 占比定性（若 SCALAR 高 = gather 退化，杠杆=按行重组或
   分派特化）；
2. **rint 的可用性**：`libdevice.rint` 在 ascend 后端的 lowering 未在
   tl_op_lowering 表内——编译门首验，fallback `tl.floor(x+0.5)`（仅在恰好
   ±x.5 时与银行家舍入不同，随机 grid 命中概率低但非零）；
3. reflection 的 fp32 floor/fmod 精度（大坐标时 fmod 误差）——smoke 覆盖；
4. bicubic 仅 2 case，正确性义务重、性能零投入。
