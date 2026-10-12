# P1 语义抽取 — grid_sample（cv_01，12_grid_sample）

> 日期：2026-09-17 ｜ 源丰富型（85 型）：CANN 官方 ops-cv `image/grid_sample` 完整源码。
> golden 特殊性：**融合版直接调 `F.grid_sample`**——NPU 上它分派到 CANN GridSampler
> kernel 本身，故 framework 参照系 = CANN 原生（perf.json 为 aclnn 实测）。语义仲裁
> 因此极简：golden 语义 ≡ CANN 语义 ≡ torch 语义，三者同源，无分叉表需要。

## 1.1 功能分析（算什么）

对每个输出位置 (n, c, ho, wo)：取 grid[n, ho, wo] = (gx, gy)（归一化 [-1,1]）→
反归一化到输入像素坐标 (x', y') → 按 padding 模式处理越界 → 按插值模式采样
input[n, c, y', x'] 邻域 → 得 y[n, c, ho, wo]。

**坐标约定**（docs/aclnnGridSampler2D.md + kernel 实证）：grid 末维 [2] = (x, y)，
x 沿 W、y 沿 H；(−1,−1) 左上。kernel 侧实证：GatherMask 分离交错 xy
（grid_sampler_2d.h:618-619，xPattern=1/yPattern=2 作用于 +1 后的 buffer）。

**反归一化两式**（docs + kernel :622-631 逐行对应）：

| align_corners | 公式 | kernel 实现 |
|---|---|---|
| true | x' = (gx+1)/2·(W−1) | `Muls(x, 0.5·(W−1))`（gx+1 已在前） |
| false | x' = ((gx+1)·W −1)/2 | `Muls(x, 0.5·W); Adds(x, −0.5)` |

**padding 三模式**（kernel `Clip` :133-140 分派）：
- **zeros**：坐标不夹取；越界角点**权重清零**——mask 由 fp32 范围比较产生
  （`CoordinatesGetMaskWithRange` :154-157，x∈[0,W−1] 且 y∈[0,H−1]），
  `PointBilinear` :532-534 把 mask=0 的权重 Select 成 0.0f；
- **border**：`BorderClip` :204-230——坐标 Mins/Maxs 夹取到 [0, W−1]/[0, H−1]；
- **reflection**：`ReflectClip` :232-278 + `ReflectCoordinatesGeneral` :280-354——
  minS/abs/fmod/flip 标准反射折叠；**边界依赖 align_corners**：ac=1 →
  twiceLow=0、twiceHigh=2(W−1)；ac=0 → twiceLow=−1、twiceHigh=2W−1
  （:242-249）——与 torch 的反射语义（ac 影响反射周期端点）一致。

**插值三模式**（docs mode 语义 + kernel）：
- bilinear（mode=0）：floor 取角（:646-647 CAST_FLOOR）、xE=xW+1、四权重
  nw=(xE−x)(yE−y) 等（:668-676 两次 Sub+Mul 合成）；
- nearest（mode=1）：`grid_sampler_2d_nearest.h`（1096 行，最近邻取整）；
- bicubic（mode=2）：`grid_sampler_2d_bicubic.h`（1299 行，16 点立方加权）；
- **3D（5D 输入）**：`grid_sampler_3d*.h`——trilinear 8 角点（3D 版坐标多一维 z）。

## 2 原型设计（接口）

def（grid_sample_def.cpp:108-120）：x/grid/y 均 **fp16/fp32/bf16**，ND。
平台特例 config：310P 仅 fp32（:20-47）、310B 仅 fp16（:49-76）、kirin
fp16/fp32 + PrecisionReduce（:78-103）——910b 主配置全 dtype。
属性（tiling.cpp:140-166）：interpolationMode（"bilinear"/"nearest"/"bicubic"
字符串）、paddingMode（"zeros"/"border"/"reflection"）、alignCorners（bool）、
channelLast（bool，**NHWC 布局开关，套件不涉及**）、schedulerMode（0/1，仅
channelLast 场景可为 1，:216-218）。

## 2.1 原型约束

- grid 末维：2D=2 / 3D=3，否则报错（tiling.cpp:71-81）；
- **inH·inW（3D 含 inD）≤ INT32_MAX**（:205-208, :241-244）——int32 寻址上限
  写死在原型校验里（Triton 侧 int64 标量基址纪律不受影响，套件最大
  1120²=1.25M 远低于限）；
- 每维 ≥1（:199-204）；
- bicubic 平台约束：部分平台不支持（docs 注记），910b 支持（kernel 有
  fp32/fp16/bf16 三实例，grid_sample.cpp:102-116）。

## 3 规格约束（50-case 套件，[GOLDEN] json 统计）

| 维度 | 分布 | P2 含义 |
|---|---|---|
| 维度 | 4D×47 / 5D×3（5D 均小 shape 24³-42³） | 2D 主线，3D 长尾正确性义务 |
| dtype | **fp16×43 / fp32×7**（无 bf16） | fp16 主路径；def 的 bf16 支持套件不覆盖 |
| mode | **bilinear×46** / nearest×2 / bicubic×2 | bilinear 压倒性主线；bicubic 仅正确性义务 |
| padding | zeros×35 / reflection×8 / border×7 | 三模式都要对；zeros 是主线 |
| align_corners | False×42 / True×8 | 两式都要实现 |
| **H_out==H_in 恒成立** | 比例恒 1.00 | **纯 warp：无上下采样，interpolate skill 的整数倍特化不适用** |
| shape | N 1-4，C 2-112，H/W 24-1120 | 大 shape 存在（1120²=1.25M 点） |

## 4.1.1 分核策略（CANN 侧，[SRC]）

- 任务块 = (n, hw-block)：HW 维按 **CAL_H_W_BLOCK=512** 输出点切块
  （grid_sampler_2d_base.h:119），gridHW = outH·outW 分 preNUbLoop_ 块
  （:39-44），core 均分（preCoreLoop_/lastCoreLoop_），**grid = needCoreNum_
  （persistent-core 形态，tiling 定核数）**；
- 每核内：HW 块串行循环，块内 channel 按 **CHANNEL_BLOCK=64** 循环（:46-48）。
- workspaceSize_：基础 16MB + fp16/bf16 时追加 needCoreNum·inC·hwFactor·4B
  （tiling.cpp:277-289）。

## 4.1.2 单核内切分与访存策略（核心性能知识，[SRC]）

**变体分派（tilingkey，910b NCHW 主线）**（grid_sample.cpp:82-146）：

| tilingkey | 变体 | 场景 |
|---|---|---|
| 1000220 | GridSampler2D\<float\>（general normal） | **fp32 bilinear**（套件 7 case） |
| 1000210/1001210 | GridSampler2DFP16SlideWindow\<half\> | **fp16 bilinear（套件 43 case 主路径！）** |
| 1000230/1001230 | 同左 bf16 | 套件无 |
| 1000221 系 | Nearest | nearest（2 case） |
| 1000222 系 | Bicubic | bicubic（2 case） |
| 2000xxx/2100xxx | FullLoad（全输入载 UB） | **仅 channelLast(NHWC)+bilinear+小输入**（tiling.cpp:182-197）——套件 NCHW 不触发 |

**访存策略（两变体的本质差异）**：
- general fp32 版：NCHW gather 用 **`GetValue/SetValue` 逐元素标量读**
  （`MTE2ForNCHW` :360-373——每角点每 channel 逐个 GM 标量访存！）；
  NHWC 才有 8 连 C 的 DataCopyPad 向量化（:376-422）；
- fp16 slide_window 版：`grid_sampler_2d_fp16_slide_window.h`（1450 行）——
  滑窗复用输入行，减少重复 gather（CANN 对 fp16 主路径的专门优化）；
- **写出用 GM atomicAdd 累加 4 角点**（PointBilinear：nw 直写 + ne/sw/se
  `SetAtomicAdd<float>` :491-516）——因为 channel 分块循环导致同一输出位置
  多次写。**Triton 化时此结构可消解**：单 program 持有 [C_blk, HW_blk] 完整
  tile，4 角点贡献在片上累加后**单次 store**，零 atomic（结构性消解，
  与 op-85 消解 CV 握手同型）。

## 4.1.3 tilingkey / tilingdata

tilingkey = `GET_TILINGKEY(interpolationMode, dtypeKey, dimValue, schedulerMode,
dimension, templateCNum, tempType)`（tiling.cpp:31-32，7 位十进制编码）。
**⚠️ 宏定义在 ops-cv `common/`**（本次 sparse checkout 未含）——位义破译
为 P2 前置项（或从 kernel 分派表反推：第 3 位区分 normal(0)/slide(1)/
fullLoad(2)，第 5-6 位区分 mode(0=bilinear,1=nearest,2=bicubic)，第 4 位
dtype(1=fp16,2=fp32,3=bf16)——[DERIVED] 自分派注释，待 common/ 证实）。
tilingdata 字段见 base.h:124-146（coreNum/shape 6 项/mode 3 项/loop 参数 8 项）。

## 5 片上内存（UB 预算，general 版 Init :65-83）

grid 4KB + x 32KB + 坐标/权重/中间量约 30KB + outValue 32KB + mask ~1.3KB
≈ **99KB / 192KB**（fp32 版）；fp16 slide_window 版预算见其文件（P2 参考）。
每次循环 CAL_H_W_BLOCK=512 输出点 × CHANNEL_BLOCK=64 通道。

## 6 计算流程（与 golden 逐行对照）

golden = `F.grid_sample(input, grid, mode, padding_mode, align_corners)`
（12_grid_sample.py forward 直接透传）——**无中间层，语义= torch 官方定义**。
kernel 流程与 torch 语义的对照：

| torch 语义 | kernel 锚点 | 一致性 |
|---|---|---|
| 归一化坐标 → 像素坐标（ac 两式） | :605-631 | ✅ 逐式对应（docs 公式同） |
| zeros 越界置零 | 权重清零法 :154-157,:532-534 | ✅（等价于邻域越界点置 0） |
| border 夹取 | :204-230 | ✅ |
| reflection 折叠（ac 相关端点） | :232-354 | ✅（twiceLow/High 与 torch 约定一致） |
| bilinear 4 点加权 | :646-676 | ✅ 标准 |
| 输出 dtype = 输入 dtype | def :118-120 | ✅（计算全程 fp32，store 前 cast） |

## 精度策略

- 坐标/权重**全程 fp32**（grid load 后即 Adds/GatherMask 于 fp32 域 :605-631；
  x 输入在 gather 后 Cast fp32）；输出 store 前降回输入 dtype（fp16 版同理）；
- **floor 用 CAST_FLOOR**（:646-647）——Triton 侧注意 floor 语义（负数坐标
  不能 truncate）；**zeros 的 mask 比较是 fp32 域**（:154-157——注意这与
  C-002「fp32 导出 mask 风险」的交集：CANN 侧合法，Triton 侧须按 dtype_support
  分离写法评估，P2 两态）；
- 容差：ATK json 无显式 rtol/atol（框架默认）——验收以 harness 口径为准
  （与 94 同形态）。

## 测试设计

- ST：`tests/st/aclnnGridSampler2D/3D`（ATK 框架 executor + json）；
- UT：`tests/ut/op_graph/fusion_pass`（图级融合 pass 单测存在——P2 前需扫
  一眼是否改变语义）；
- 套件 50 case 的 mode/padding/ac 长尾组合（nearest/bicubic/reflection/5D）
  是正确性验收的重点覆盖区。

## 架构差异（副线）

- 310P：fp32-only + 专用 slide_window/fullLoad 310P 变体（kernel .cpp:51-63）；
- 310B：fp16-only 专用变体（:64-80）；
- 950（arch35/regBase）：`IsCapable` :266-273——regBase 平台 bilinear+NCHW
  **不走本模板**，转 arch35 smit 变体（`arch35/grid_sampler_2d_bilinear_smit.h`）；
- 910b（本目标）：走 §4.1.2 表中的 normal/slide_window 变体群。
