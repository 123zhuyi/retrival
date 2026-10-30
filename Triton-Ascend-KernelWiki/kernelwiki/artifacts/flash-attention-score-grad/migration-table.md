# FlashAttentionScoreGrad - CANN -> Triton-Ascend 迁移分类 (S3)

依据 [references/migration-rubric.md](../../../integrations/cannbot-skills/ops/triton-kb-cann-ingest/references/migration-rubric.md) 四桶分类。所有结论基于静态源码分析（confidence: inferred）。`needs_verification` 列：yes=需硬件/运行时验证，no=源码可确认。

| 知识项 | CANN 来源 (path Lxx) | Triton-Ascend 处理 | 桶 | Confidence | Needs verification |
|--------|------|------|------|------|------|
| 反向 6 步公式（重算 p、dp、ds、dq、dk、dv） | `docs/FAG算子设计介绍.md` L9-14 | 直接照搬为算法骨架；p 由 QK^T 重算，ds=P⊙(dp−FlashSoftmaxGrad(dy,Y)) | transferable | inferred | no |
| FlashSoftmaxGrad：ds = P⊙(dP − rowsum(dP⊙P))，dP=dY·V^T | `docs/FAG算子设计介绍.md` L11；`op_kernel/arch35/vector_api/vf_softmax_grad_front_cast*.h` | 在线 softmax 反向，FP32 累加；移植为 K 循环内 running 统计 | transferable | inferred | no |
| dV=P^T·dY、dQ=(ds·K)·scale、dK=(ds^T·Q)·scale | `README.md` L50-60；`docs/FAG算子设计介绍.md` L12-14 | matmul 语义直接移植；scale 作为常数乘 | transferable | inferred | no |
| mask/predicate 逻辑（atten_mask 1=不参与→负无穷；causal/prefix/band sparse_mode 0-8） | `op_host/arch32/flash_attention_score_grad_tiling_common.h` L187-197；`README.md` L127 | 移植为 `tl.where`/掩码加载；sparse_mode→causal window 偏移 | transferable | inferred | no |
| FP32 累加（BF16/FP8 输入，中间 FP32） | `docs/FAG算子设计介绍.md` L33,56；kernel 模板 `float` 累加类型 | `tl.dot(...).to(tl.float32)` 累加，输出再 cast 回 | transferable | inferred | no |
| 尾块/边界 mask（S1/S2 不整除基本块） | `op_kernel/arch32/flash_attention_score_grad_tiling.h` tail 字段 | `mask = offsets < n` 的 masked load/store | transferable | inferred | no |
| scale/keep_prob/pse_type 语义 | `op_host/flash_attention_score_grad_def.cpp` L222-248 | 直接作为标量参数 / `tl.constexpr` 分支 | transferable | inferred | no |
| 变体分类（forward/backward/inference/varlen/unpadding） | `docs/aclnnFlashAttentionScoreGrad*.md` V1-V4/Unpadding | 仅作语义说明，非实现 | transferable | inferred | no |
| arch32 多模板（B/N2/SameAB/S1S2/BN/Basic/Det）按 shape 路由 | `op_kernel/flash_attention_score_grad.cpp` L508-901；`docs/FAG算子设计介绍.md` L264-319 | 映射为 launcher 按 shape 选 `BLOCK_M/N` 与 grid；`if constexpr`/独立 `@triton.jit` | rewrite | inferred | yes（具体基本块数值） |
| arch32 CV tiling 分离（C 128×1024，V 8×1024，nRatio=8，1:16） | `docs/FAG算子设计介绍.md` L33-50 | 映射为 `BLOCK_M/N` constexpr；CV 配比→程序网格形状；UB 由编译器管理 | rewrite | inferred | yes（配比数值） |
| arch35 统一模板（C 128×128，V 64×128，L0C→UB/UB→L1） | `docs/FAG算子设计介绍.md` L52-72；`op_host/arch35/...regbase.cpp` L93-106 | 映射为 `BLOCK_M=128/N=128`；AIC→UB 通路抽象为 tile 内复用/融合存取 | rewrite | inferred | yes（基本块数值） |
| 核间切分轴（B/N2/G/S1，S2 不切多核）→ block_dim | `op_host/arch32/*tiling*.cpp`；`docs/FAG算子设计介绍.md` L126 | `grid` 形状 + `tl.program_id`；S2 保留为 K 循环轴 | rewrite | inferred | no |
| tiling key（20 模板参数 bit 编码）→ 模板选择 | `op_kernel/arch32/flash_attention_score_grad_template_tiling_key.h`；`op_kernel/flash_attention_score_grad.cpp` L488-501 | 启动器按 dtype/layout/mask/dropout/rope 选 kernel 与 constexpr；key 本身不移植 | rewrite | inferred | no |
| 空 tensor 路径（key=0/1，100MB workspace，dq/dk/dv 清零分核） | `op_host/flash_attention_score_grad_tiling.cpp` L49-51,68-234 | 启动器短路：全零输出，无计算 | rewrite | inferred | no |
| MTE/`DataCopy`/`CopyGMToUB`/`CopyUBToGM` 手动搬运 | `docs/FAG算子设计介绍.md` L40-45,64-68 | 抽象为 tile 级 fused load/store；不作为 Triton 步骤 | CANN-specific | inferred | no |
| BufferID/`SetFlag`/`WaitFlag`/TQue/TBuf 同步 | `op_kernel/arch35/*` queue/buffer 模块 | 抽象为编译器管 pipe；不移植 | CANN-specific | inferred | no |
| 手动 Cube/Vector 流水编排、`IterateBmm1`/`ProcessVec` 时序、nRatio CV 配比调优 | `docs/FAG算子设计介绍.md` L93-110；`op_kernel/arch32/basic_modules/` | 仅说明“提前发射/双缓冲”意图；Triton 由编译器排流 | CANN-specific | inferred | no |
| ND↔NZ 物理格式转换、`DataCopyPad` 偏移、`fixp_l0c_to_ub`/`copy_ub_to_l1` | `docs/FAG算子设计介绍.md` L64-68；`op_kernel/arch35/matmul_modules/` | 抽象为 tile 复用/布局 stride；不移植格式转换 | CANN-specific | inferred | no |
| TBuf `Get<TPosition>`（UB/L1/L0A/L0B/L0C）放置、`TID`/`PIPE_ID` | `op_kernel/arch35/*` | 不移植；Triton 无显式 buffer 位置 | CANN-specific | inferred | no |
| `ascendcPlatform` 运行时容量查询（coreNum/ubSize） | `op_host/flash_attention_score_grad_tiling.cpp` L411-420 | 启动器按 shape 选 `BLOCK_*`；skeleton 不需运行时查询 | CANN-specific | inferred | no |
| 二进制 hash/`*_binary.json`/`simplified_key.ini` 注册选择 | （CANN 编译机制） | 不移植（编译期注册，非算法） | CANN-specific | inferred | no |
| 具体 tiling 参数数值（S1.i=128 等）是否最优 | tiling cpp 中的基本块常量 | 作为 `BLOCK_*` 源码报告值，不主张最优 | needs-verification | inferred | yes |
| UB/L1/L0C 字节占用与占用率 | tiling_common.h `TensorSize` 字段 | 不主张具体占用 | needs-verification | inferred | yes |
| FP8/BF16 精度边界 | `op_host/flash_attention_score_grad_def.cpp` dtype；`op_kernel/arch35/vector_api/vf_anti_quant_*` | 未运行时测试，不主张精度 | needs-verification | inferred | yes |
| aclnn 各版本 API 契约细节 | `docs/aclnnFlashAttentionScoreGrad*.md` | 仅记录版本差异；运行时契约需头文件确认 | needs-verification | inferred | yes |
| TND/BSH 等布局偏移算术 | `op_host/.../tiling_common.h` `LAYOUT_TO_AXIS` L95-102 | 需张量实例确认偏移 | needs-verification | inferred | yes |
| 性能（任何加速比） | — | `static_ingest` 模式无 baseline，不主张 | needs-verification | inferred | yes |
