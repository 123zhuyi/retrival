# mla_preprocess - CANN -> Triton-Ascend 迁移分类

四分类。每条带 CANN 来源 path Lxx。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| RmsNorm 公式 γ·x/√(mean(x²)+eps) | README L24-30 | `tl.sum(x*x)/N` + `rsqrt` + 乘 γ | transferable | inferred | no |
| RmsNormQuant = (RmsNorm+bias)·deqScale | README L32-34 | cast + scale 融合 | transferable | inferred | no |
| 3 matmul（W^DQKV/W^UQ/W^UK） | README L12-19；tilingdata mm1/mm2/mm3 L51-53 | 3 个 `tl.dot`（FP32 acc） | transferable | inferred | yes（吸收与否） |
| RoPE 旋转位置编码（cos/sin） | README L43/53；def cos/sin L89-90 | 复数旋转乘 | transferable | inferred | no |
| MLA 权重吸收（q^N=RmsNormQuant·W^DQKV·W^UK） | README L38-40 | 2 matmul 等价（W^UQ·W^UK 折叠） | transferable | inferred | yes（吸收条件） |
| Cache scatter 写回（slot_mapping） | def L94；README L222-226 | `tl.store(cache + slot*stride)` | transferable | inferred | yes（cacheMode offset） |
| FP32 累加（BF16 输入） | tilingdata；bf16 模板 | `tl.dot(..., out_dtype=tl.float32)` | transferable | inferred | no |
| 尾核边界（perTaskNum/resTaskNum） | tilingdata L49-50 | `mask = offs < n` | transferable | inferred | no |
| quantMode（per_tensor/per_token/no_quant） | tiling.h L28-33 | cast + scale 分支 | transferable | inferred | yes（per_token 实现） |
| cacheMode（0-3 输出格式/量化） | README L306-311；def L110 | 输出 layout/量化分支 | transferable | inferred | yes（cacheMode=2 int8 量化） |
| tokenNum 核间切分（numCore/perTaskNum） | tilingdata L47-50 | `tl.program_id` + grid | rewrite | inferred | no |
| mm1/mm2/mm3 基本块 m0/k0/n0 | tilingdata L30-32 | `tl.constexpr BLOCK_M/K/N`（值 source-reported） | rewrite | inferred | yes（最优值） |
| tiling key（cacheMode×format ND/NZ） | mla_preprocess.cpp L46+ | `if constexpr`/launcher（key 不迁移） | rewrite | inferred | no |
| dtype dispatch（FP16/BF16/no_quant） | mla_preprocess.cpp L17-22 | 独立 kernel / `if constexpr` | rewrite | inferred | no |
| ND↔NZ 输出格式 | def nd_nz_Format L53-58；cacheMode=2/3 | Triton layout（编译器） | CANN-specific | inferred | no |
| `*_iterator.h` buffer 搬运（GM/L1/L0C/UB） | op_kernel/*_iterator.h | 不迁移；load/store（编译器） | CANN-specific | inferred | no |
| ProcessCube/ProcessVector CV 分离 | mla_preprocess.cpp L52-53 | 不迁移；fused 单 program | CANN-specific | inferred | no |
| swizzle/ping-pong/enShuffleK | tilingdata L37-42 | 不迁移（CANN 调度） | CANN-specific | inferred | no |
| `__DAV_C220_CUBE__` 平台 guard | mla_preprocess.cpp L36 | 不迁移；launcher 按 arch 选 | CANN-specific | inferred | no |
| tiling 参数数值（m0/k0/n0 最优） | tiling.cpp | constexpr 值 source-reported | needs verification | inferred | yes |
| workspace 字节预算（maxWorkspaceSize） | tiling.h SetMlapoWorkSpace | 需硬件实测 | needs verification | inferred | yes |
| MLA 吸收数值等价性 | README 公式 vs kernel 3 matmul | 需验证吸收是否在 kernel 内执行 | needs verification | inferred | yes |
| quantMode 1/2/3 实现状态 | README L315（1/2/3 未实现） | 仅 0 实现，其余 needs_verification | needs verification | source-reported | yes |
| 预留参数（wdq_dim/transpose_*等） | def L101-113（只支持单值） | 固定值 | needs verification | source-reported | no |
