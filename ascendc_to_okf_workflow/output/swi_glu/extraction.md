# P1 语义抽取 — swi_glu

> 生成时间：2026-09-17｜依据：`output/swi_glu/manifest.md`（低价值，P2.5 skipped: low-value）
> 根目录 `src/ops-nn/activation/swi_glu/`；hpp 均指 `op_kernel/arch35/` 下文件。

## 1. 功能与公式

- `out = SiLU(A) · B`，A/B = x 沿属性 dim 对半切（README；docs/aclnnSwiGlu.md）。
- **kernel 实现形态（sigmoid 形）**：`y = a / (1 + e^{β·a}) · b`，**β = −1.0**——负号不在计算类成员初值里（impl.hpp:29 / bf16.hpp:30 `beta = 1.0` 是诱饵），而在 `SwigluSingle::InitGmBuffer` 覆写 `this->beta = -1.0f`（single.hpp:45；同行注释掉的 `* beta_gm[0]` 表明曾设计为 GM 传入的 runtime 属性）。**只读 impl 层会得出 a·sigmoid(−a) 的错误语义**——本算子第一语义陷阱。
- golden = `torch.chunk(x, 2, dim)` → `F.silu(a) * b`，fp16 升 fp32 计算（golden.py）。

## 2. 接口与原型

### 2.1 GEIR op（swi_glu_def.cpp）
- x/y：{F16, F32, BF16}，ND（:23-30）；**Attr dim OPTIONAL，默认 −1，取值 [-x.dim(), x.dim()-1]**（:32，README）。
- 平台分组（:33-55）：910b/910_93 全 dtype；310p/kirinx90/kirin9030 **无 bf16**；**950 = regbase cfg + `opFile.value=swi_glu_apt`**。
- DynamicRank/Shape=true。

### 2.2 infershape（swi_glu_infershape.cpp）
- 输出 = 输入在 splitDim 轴减半；dim 负值归一（tiling SetTotalShape 同样归一，swi_glu_tiling.cpp:182）。

### 2.3 aclnn 层
- 两段式直通（examples/test_aclnn_swi_glu.cpp）；aclnn golden 与 kernel golden 同语义。

## 3. 规格约束

| 项 | 值 | 锚点 |
|---|---|---|
| dtype | fp16/fp32/bf16（310p/kirin 无 bf16），in/out 同型 | def.cpp:25,30,39,44 |
| shape | 动态 rank/shape；splitDim 轴须可被 2 整除（对半切） | def.cpp:53-54；README |
| dim | [-rank, rank-1]，默认 −1 | def.cpp:32 |
| 平台 | 910b/910_93/310p/kirin/950 | def.cpp:33-55 |
| 双 kernel | 910b→swi_glu.cpp（bf16 受 `__CCE_AICORE__==220` 守卫）；950→swi_glu_apt.cpp（bf16 无守卫 + KEY_EMPTY 早退） | 两入口文件 |
| ST 值域 | 多为 (-1,1)，含 (-1000,-10)、(-0.01,0.01) | ttk csv |

## 4. 分核策略与 tiling（✅ 手写 tiling 在仓，非 base 库）

### 4.1 shape 折叠（SetTotalShape，swi_glu_tiling.cpp:172-190）
- splitDim 归一 → `shapeBefore = ∏ dims<splitDim`、`shapeAfter = ∏ dims>splitDim`；问题折叠为 2D：(before·after) 行 × half 列的 A/B 交错访问（A/B 沿 splitDim 相邻，各 half 长）。
- 输出 numel = 输入 numel / 2。

### 4.2 tilingkey 与双缓冲
- tilingkey = dtype 码 {0=fp32, 1=fp16, 27=bf16}（两入口）；`isDoubleBuffer`（tilingdata）→ bufferNum 2/1 两档实例。
- `TILING_KEY_EMPTY = 100`（tiling.cpp:32）→ apt 入口早退（空/退化保护，触发条件在 host 侧，疑似 numel=0/非法 shape 兜底）。
- host 常数：L2 cache line 512B、UB 32B 对齐、DEFAULT_BUFFER_NUM=2、**DataCopy 硬件上限：blockCount≤4095、blockLen≤65535×32B、stride≤65535**（tiling.cpp:11-21）、系统 workspace 16MB（:31）。

### 4.3 2D base-tile 优选（GluSingleTilingOptParam，tiling.cpp:56-64）
- 在 UB 上限内枚举 baseRowLen×baseColLen（alignPackLen = 512B 或 32B 对齐包长），选 tile 数/核数占用最优解；尾块单独处理。
- **Triton 转化视角**：这是 AscendC DataCopy 粒度工程（凑 2D 连续块最大化单条 DataCopy），Triton 的 tl.load 由编译器拆 MTE——**不需要复刻**，对应物是 BLOCK autotune。32B/512B 常数与 backend_facts C-010 互证。

## 5. 片上内存

- 3 个 TQue（inA/inB/outC，bufferNum=2 双缓冲）+ tmp buffer：fp32 路径 1 个 float tmp（预填 1.0，impl.hpp:33-35）；bf16/fp16 路径 2 个 calcType tmp（input/output 复用，bf16.hpp:35-37，注释 "a/b 复用 todo tiling 中 BUFFER 大小刷新"）。
- UB 预算（P2 校核）：BLOCK 元素的 a(in) + b(in) + fp32 中间×2 + out。

## 6. 流水编排（kernel 实现 —— 核心语义资产）

### 6.1 类结构
- `SwigluSingle<ParentClass, inType, outType>` 继承 `SwigluVector`（fp32）或 `SwigluVectorBF16`（fp16/bf16）：父类管 UB 计算，子类管 GM 寻址 + **beta 覆写**。
- CopyIn：A、B 两次独立 DataCopy（splitVecGmOffset1/2，blockCount/blockLen/stride 三元组，single.hpp:52-67）；is32BAligned==0 时走 DataCopyPad + DataCopyPadExtParams（:78-100）。

### 6.2 计算链（两 dtype 档，β=−1 后）

fp32（impl.hpp:59-83，SwigluVector<float,float>）：
```
c = a·(−1) → c = e^c → c = 1+c → c = 1.0/c      [= sigmoid(a)，tmp 预填 1.0]
c = c·a → c = c·b                               [全程 float]
```
fp16/bf16（bf16.hpp:62-88，calcType=float）：
```
a32 = Cast(a, NONE) → c = e^{−a32} → c = 1+c → c = a32/c   [= a·sigmoid(a) 一步]
b32 = Cast(b, NONE) → c = c·b32 → out = Cast(c, RINT)
```
- **fp16 与 bf16 同档升 fp32**（入口 key=1 → `SwigluVectorBF16<half,float,half>`）——与 golden 的 fp16 升 fp32 一致；与 abs 的 fp16 原生相反（第二语义陷阱）。
- 两链数学等价、舍入路径不同（sigmoid 先成 vs a/· 一步）；Triton 取 bf16 链（a/(1+e^{−a})·b，少一次乘法且与低精度标杆同路径）。

### 6.3 同步
- PipeBarrier<PIPE_V> 逐条分隔（impl 每步一道）——保守手写同步，Triton 无对应，编译器自动（⚠️）。

## 7. 数值精度

- 低精度档全程 fp32 中间域 + RINT 回写（同 gelu 家族）。
- sigmoid 形溢出行为：a 大正 → e^{−a}→0 → y=a·b ✓；a 大负 → e^{−a}→+inf → a/inf→−0·b ✓（ST 含 (-1000,-10) 值域 case 背书）。
- ST 容差：fp32 1e-4、fp16/bf16 1e-3（csv precision_tolerances 列）。
- beta 曾是 runtime GM 属性（注释掉的代码）——现硬编码 −1.0，无属性面。

## 8. 测试与标杆

- golden：chunk → silu·mul（fp16 升 fp32）；aclnn golden 同。
- ST：79 case fuzz，dim 覆盖正负/中间/尾轴、rank 2~7；**⚠️ 全部行 output_shapes==input_shapes，与 dim 轴减半语义矛盾——疑 fuzz 配置错误，验收以 golden 为准**。
- UT：kernel 级（test_swi_glu.cpp + gen_data/gen_tiling）。

## 9. 偏差与风险清单

1. **beta 符号陷阱**（§1）——本卡存在的第一理由。
2. **fp16 升 fp32**（§6.2）——与 abs 卡相反，防跨卡习惯错抄。
3. ST csv 形状矛盾（§8）——golden 为权威。
4. 910b bf16 守卫（`__CCE_AICORE__==220`）与 310p/kirin 无 bf16——平台边界。
5. ST 无 dim 非尾轴 + 大 after 的性能 case——地址折算路径（整数除法）的性能未在官方测试覆盖，P4 冒烟注意。
6. TILING_KEY_EMPTY=100 触发条件未从 host 代码定位（早退路径，不影响正常语义）。

## P1 质量门

- [x] 三层语义齐备（kernel 实现链 / aclnn / 测试层）
- [x] 两语义陷阱定位到行号；关键论断带锚点
- [x] ❌=0（DataCopyPad→mask、strided copy→stride 折算、双缓冲→multibuffer、2D tile 优选→autotune）
- [x] 低价值复核通过 → P2，**P2.5 skipped: low-value**
