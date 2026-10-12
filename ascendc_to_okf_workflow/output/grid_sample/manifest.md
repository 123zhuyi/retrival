# P0 manifest — grid_sample（cv_01，12_grid_sample）

> 日期：2026-09-17 ｜ **首个「源丰富 + P2.5 全流程正向」验证算子**（85 型提取主线 +
> 新立项的骨架实测门首次正向使用）。
> **盲纪律说明**：全仓无 grid_sample 既有生成实验归档（已查 triton_ascend_output
> 各 checkout），无污染源。skill 库 `ascend-interpolate-optimization.md`（interpolate
> 2D 优化体系）属共享优化参考（非本算子的对手答案），P2 阶段按族/范式引用。

## 0. 算子定位与复杂度预判

- **算子**：GridSample 2D/3D 采样（融合版）——对每个输出位置：grid 归一化坐标
  → 反归一化（align_corners 两式）→ 越界处理（zeros/border/reflection）→ 按
  interpolationMode 插值（bilinear 4 点 / nearest 1 点 / bicubic 16 点）。
  golden 头注明示对应 ops-cv `grid_sample` AscendC kernel（aclnnGridSampler2D/3D）。
- **算子类判定（预）**：**离散访存/gather-scatter 类**（operator_class_patterns §1
  复杂 Vector 组织法：批量搬 UB + UB 内选择 + 多重 mask）——无 tl.dot，非 Cube；
  平台「视觉类」标签是应用领域分档，不改变提取范式。
- **套件特征（50-case 统计）**：47 4D / 3 5D；fp16 43 / fp32 7（无 bf16——
  与 def 的 bf16 支持有差，卡片勿给 bf16 承诺）；**bilinear 46 / nearest 2 /
  bicubic 2**（bilinear 压倒性主导，nearest/bicubic 是长尾正确性义务）；
  padding zeros 35 / reflection 8 / border 7；align_corners False 42 / True 8；
  **H_out==H_in 恒成立（比例恒 1.00）——纯 warp 重采样，无分辨率变化**；
  N 1-4、C 2-112、H/W 24-1120；5D case 均为小 shape（24³-42³）。
- **复杂度预判：中高**。难点不在公式而在**访存模式**（任意流场 gather，
  坐标计算 + 越界分支 + 多点插值融合）+ 长尾模式组合（3 mode × 3 padding ×
  2 align_corners × 2D/3D）。结构简单但特化空间大（interpolate skill 文档的
  同类教训：通用路径与特化路径差 10x 量级——但注意该文档的上下采样特化在
  本套件不适用，H_out==H_in）。

## 1. 材料清单

| 类别 | 路径 | 存在 | 关键符号 / 行号 | 对应详设章节 |
|---|---|---|---|---|
| **评测 golden** | `1-spare/ascendc-kernelgen-data-perf-baseline/npu_benchmark/level3/12_grid_sample.py`（br_perf_baseline 分支） | ✅ | 融合版：forward 直接调 `F.grid_sample`（头注：与 ops-cv 实现同源）；Model 类 + 参数 mode/padding_mode/align_corners | 语义基准 |
| **用例套件** | 同目录 `12_grid_sample.json`（jsonl 50 case） | ✅ | 分布见 §0 | 输入域 |
| **⭐ 原生性能基线** | 同目录 `12_grid_sample_perf.json` | ✅ | **ops-cv grid_sample aclnn 实测**（warmup=25/repeats=10/msprof task_time 中值口径；2026-09-15 case 重设后全量重测）——**回放参照系 = CANN 原生 kernel，首次「Triton 生成 vs CANN 原生」直接对标**；generator Phase 1 的 ref_perf_json 注入机制可免测框架侧 | 验收口径 |
| **原型定义** | `ops-cv/image/grid_sample/op_host/grid_sample_def.cpp` | ✅ | `OP_ADD(GridSample)`；x/grid/y 均 fp16/fp32/bf16 ND（:108-120）；平台特例 config：310P 仅 fp32（:20-47）、310B 仅 fp16（:49-76）、kirin fp16/fp32+PrecisionReduce（:78-103） | 原型设计、规格约束 |
| **tiling host** | `op_host/grid_sample_tiling.cpp` + `_tiling.h` + `_tiling_arch35.{h,cpp}` | ✅ | **tilingkey 多位编码**：`GET_TILINGKEY(interpolationMode, dtypeKey, dimValue, schedulerMode, dimension, templateCNum, ...)`（tiling.cpp:31，宏定义在 ops-cv common——P1 破译）；% TILING_OFFSET 取模（:35） | 分核策略、tilingkey |
| **平台配置** | `op_host/config/ascend910b/` 等 7 平台（910b/910_93/950/310b/310p/kirin9030/kirinx90） | ✅ | simplified_key.ini + binary.json（85 缺此类材料，这里是**多平台配置完整样本**） | 平台配置 |
| **kernel 入口** | `op_kernel/grid_sample.cpp`（216 行）+ `grid_sample_apt.cpp`（306 行） | ✅ | tilingkey 分派到各变体（grid_sample.cpp:52-142：1000220 系 → …；1001210 系 → …） | 模板分支 |
| **kernel 实现（910 主线变体群）** | `op_kernel/grid_sampler_2d.h`(723) / `_base.h`(162) / `_bicubic.h`(1299) / `_nearest.h`(1096) / `_slide_window.h`(1507) / `_fullLoad.h`(1511) / `_fp16_slide_window.h`(1450) / `grid_sampler_3d.h`(1017) / `3d_nearest.h`(524) / `3d_portrait.h`(986) + `_common.h`×2 | ✅ | **变体群结构**：fullLoad（小输入全载 UB）/ slide_window（滑窗）/ fp16_slide_window / nearest / bicubic / 3d / 3d_portrait——**CANN 的特化分派策略本身是核心知识**（哪种 shape 走哪个变体 = tiling 的 schedulerMode 判定） | 计算流程、片上内存 |
| **kernel（arch35=950 副线）** | `op_kernel/arch35/grid_sampler_{2d,3d}_bilinear_smit.h` + `_common.h`（共 473 行） | ✅ | smit（单指令多数据扩展）实现，仅 bilinear——950 平台特化 | 架构差异 |
| **文档** | `docs/aclnnGridSampler2D.md`(568) / `3D.md`(513) | ✅ | 公式：归一化 [-1,1]、反归一化 align_corners 两式（aclnnGridSampler2D.md「反归一化」节）；mode 语义：0=bilinear 4 点加权、1=nearest、2=bicubic 16 点加权；**部分平台不支持 bicubic** 的平台约束注记 | 功能分析、规格约束 |
| **图级语义** | `op_graph/` + `tests/ut/op_graph/fusion_pass/` | ✅ | fusion_pass 存在——P1 须看（85 的图级 in-place 经验） | 图级语义 |
| **测试** | `tests/st/aclnnGridSampler2D/3D`（ATK 框架） | ✅ | atk json + executor py——容差值在 ATK 配置内（P1 提取） | 测试标杆 |
| **共享 skill 参考** | `1-spare/triton-ir-analysis/`、`1-spare/triton-simulator-optimizer/`、`ascend-interpolate-optimization.md`（latency-optimizer ref，在 mcts 快照内） | ✅ | interpolate 的 gather 特化/分派架构/行预载 next_pow2 等 P0 级教训——**注意其上下采样特化（整数倍）本套件不适用**（H_out==H_in） | P2 范式参考 |

## 2. 缺省项（进入 P1 前明示）

1. **GET_TILINGKEY 宏定义在 ops-cv `common/`**（本次 sparse checkout 只拉了
   image/grid_sample）——tilingkey 编码位义需 P1 破译时补拉 `common/` 目录或
   从调用点（tiling.cpp:31）反推位序。
2. **ST 容差未显式可见**：ATK 框架的容差在 atk_aclnnGridSampler2D.json 或
   框架默认里，P1 需读 atk json 提取；golden 本身无容差断言（与 94 同形态，
   harness 统一口径）。
3. **bf16 支持差**：def 支持 bf16 但套件无 bf16 case——卡片按套件（fp16 主
   /fp32 次）设计，bf16 路径标「def 支持但套件不覆盖」。
4. **5D（3D 采样）只有 3 个小 case**：3D 路径是长尾正确性义务，性能优化
   聚焦 2D bilinear 主线。

## 3. P1 抽取要点提示

- **坐标映射的两种 align_corners 式必须逐式锚定**（docs 公式 + kernel 实现
  对照）——align_corners=True 只有 8 case 但语义错误会直接挂掉该组。
- **三种 padding 的越界处理**（zeros 补零 / border 夹取 / reflection 反射）
  是正确性高发区：reflection 的坐标折叠公式（周期 2n 还是 2n−2）必须与
  golden（torch 语义）逐行对齐。
- **kernel 变体分派逻辑（schedulerMode → fullLoad/slide_window/…）是本的**
  CANN 侧最核心的性能知识**：它回答了「什么 shape 用什么访存策略」——提取时
  把分派条件（tiling 里的判定式）逐条锚定，这是 Triton 侧 dispatch 设计的
  直接参照（类似 94 的 serial/WY dispatch，但这里是访存模式分派）。
- **bicubic 的 16 点加权**只有 2 case：正确性必须对（长尾），性能不投入。
- **fp32 组 7 case**：注意 dtype_support 4a-4e 的 int32/fp32 纪律差异。
- 图级 fusion_pass 看一眼是否影响语义（如坐标计算被融合优化）。

## 4. 验证设计备忘（预注册，P4 回放时对照）

- **验收参照系 = CANN 原生**（perf.json）——本次生成实验的 speedup 分母是
  aclnn 原生 kernel，不是 torch 拼接；target_speedup 预期 ~1.0x 口径
  （追平原生即为目标，超越即「特化反超」——与 85/94 的 torch 口径完全不同，
  分析时勿混用）。
- 预注册判据（沿用 94 框架并校准）：iter_0 ≥45/50；gen ≤3；零勿重试违反；
  **geomean vs CANN 原生基线**（替代「胜 naive」——naive 不存在）；
  P2.5 四门全过且卡片 ⚠️ 项数 ≤3（新指标：静态层厚度）。
- 首次正向 P2.5：编译门（BLOCK×dtype 网格）、正确性门（3 mode×3 padding×
  2 align_corners×2D/3D 开关矩阵）、simulator 门（≥2 shape：24² 小 /
  1120² 大）、board 时序门（BLOCK 配置计时选型）。
