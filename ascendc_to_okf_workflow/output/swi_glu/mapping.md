# P2 语义转化 — swi_glu（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-17｜依据：`extraction.md`｜**P2.5 skipped: low-value**，证据 [SRC]/[DOC]。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| `tl.exp` dtype | vexp 仅 F16/F32 → fp32 域（与源码 calcType=float 同构） | dtype_support.md:157 [SRC] |
| 除法 | vdiv i32/f16/f32 支持；地址折算的整数 `//` 用 i32 安全 | dtype_support.md:52 [SRC] |
| `//`/`%` 在地址表达式 | 禁 `%`（C-001）；`//` 允许但勿进 mask 表达式（rewrite 风险）——本骨架 decode 仅用一次 `//`，余数用减法 | tl_op_lowering.md:79-81、C-001 |
| mask | load/store mask 整型比较 | dtype_support §3 规则 2 |
| 双缓冲 | TQue bufferNum=2 → multibuffer 默认 true，不显式编码 | compiler_options.yaml:21-25 [SRC]、mapping §2 |
| DataCopyPad | `other=0.0` 补零 + mask 非对齐 | mapping §2 DataCopyPad 行 ✅ |
| 对齐 | 32B(VV)/512B(CV) 与源码 tiling 常数互证（tiling.cpp:12-13） | C-010 [DOC] |

## 1. 分核策略 → grid

Ascend C 事实：2D base-tile 优选（baseRowLen×baseColLen，tile 数均分到核，tiling.cpp:56-64）。
Triton 决策：不按 2D tile 复刻——**输出展平 1D + persistent-core 连续段**（同 gelu/abs 范式）：`grid = min(cdiv(NUMEL_OUT, BLOCK), num_vectorcore)`，NUMEL_OUT = numel_x/2。等价论证：elementwise 算子的 2D tile 仅为凑 DataCopy 连续性，Triton 的 1D BLOCK + 编译器 MTE 拆分覆盖同语义；官方优选目标是核数占满，persistent-core grid 天然满足。

## 2. dim 折叠 → 地址折算（本算子唯一的寻址设计）

Ascend C 事实：SetTotalShape 折叠 (before, 2H, after)（tiling.cpp:172-190），A/B 各 half 长沿 splitDim 相邻。
Triton 决策：wrapper 折三段 `before/H/after`；kernel 内输出线性偏移 `offs` → **一次整数除法解码**：
```
HA = H * after              # 输出每个 before 段的元素数
i  = offs // HA             # before 段号（向量 i32 除法，vdiv-i32 [SRC] 安全）
a_off = offs + i * HA       # = (i·2H + j)·after + k，代数化简掉 j/k 解码
b_off = a_off + HA          # B = A 后半，偏移恰差 HA
```
- 无 `%`（C-001 合规）；j/k 解码被代数化简消除（a_off = offs + i·HA 恒等，证明：offs = i·HA + j·after + k，a_off = i·2HA + j·after + k = offs + i·HA）。
- dim=−1（after=1）时 HA=H，除法仍在但语义同形——不特设 constexpr 快路径（ST dim 分布广，统一路径换简单；整数除每元素一次，带宽-bound 下可接受，记 §9 优化项）。

## 3. dtype 分派 → 单 kernel 恒 fp32 计算域

Ascend C 事实：fp32 直通；fp16/bf16 升 fp32（calcType=float，两链数学等价）。
Triton 决策：**恒 fp32 计算域**——`load → .to(fp32) → y = a/(1+tl.exp(−a))·b → .to(out_dtype)`。fp32 时 cast 恒等。选 bf16 链的结合序（a/(1+e^{−a}) 一步，而非 sigmoid 先成再乘 a）——少一次乘法，与低精度 golden（fp16 升 fp32 的 silu）同路径。
dtype 码 tilingkey {0,1,27} 与 isDoubleBuffer 档 → 指针推导 + multibuffer 默认吸收，无需 constexpr。

## 4. 片上内存 → 预算

BLOCK=4096 bf16 最坏：a/b 载入 8KB×2 + fp32 中间 a32/b32/y（编译器复用按 3 块）48KB + out 8KB ≈ 72KB；multibuffer 输入翻倍 +32KB ≈ 104KB < 192/248KB [SRC] ✅。候选 {1024, 2048, 4096}（不上 8192，留 multibuffer 余量）。

## 5. 大 numel 寻址

骨架声明 int32 安全域：wrapper 断言 `numel_x < 2³¹`（a_off 上界 = numel_x−1）。超域输入（≥4G 元素，≥16GB bf16）在卡片 §9 登记已知限制 + 处置（wrapper 层按 before 维切块循环调用）——低价值算子不引入 int64 向量 decode（i64 div 标量化，dtype_support 4a）。

## 6. 显式丢弃 / 不涉及清单

- 2D base-tile 优选、DataCopy 硬件上限（4095/65535）、DataCopyPad 双路径 → 编译器/autotune/mask 承接，丢弃。
- PipeBarrier<PIPE_V> 逐条同步 → 编译器自动，丢弃。
- TILING_KEY_EMPTY 早退 → wrapper `numel==0` 早退等价。
- beta 的 GM runtime 属性痕迹（注释掉的代码）→ 不复活，硬编码 −1.0（与现源码一致）。

## 7. 精度策略 → 验收

- 恒 fp32 中间域 + RTNE 回写，写入验收标准（fp16 也升——**与 abs 卡相反**，生成指引单列）。
- golden = F.silu(chunk)·chunk；容差按 ST：fp32 1e-4、fp16/bf16 1e-3。
- ST 值域含 (-1000,-10) 大负数——sigmoid 形 e^{−a} 上溢支路已论证安全（extraction §7）。

## 8. P2 质量门自查

- [x] tl op lowering 依据 [SRC]；地址 decode 无 `%`、`//` 不进 mask
- [x] ❌=0；⚠️=3（VL/同步/multibuffer 自动）有出处
- [x] UB 预算校核；BLOCK 候选对齐 32B
- [x] **P2.5 skipped: low-value**
