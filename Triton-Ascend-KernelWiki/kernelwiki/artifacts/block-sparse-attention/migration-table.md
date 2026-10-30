# block_sparse_attention - CANN -> Triton-Ascend 迁移分类

四分类。块稀疏 mask 压缩 + gather 方法见 [[technique-block-sparse-mask-gather]]；QK->online softmax->PV 流水骨架见 [[technique-flash-attention-pipeline]]。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 块稀疏注意力公式 `Softmax(scale·Q·K_sparse^T)·V_sparse` | docs L18-22 | `tl.dot` + online softmax | directly transferable | inferred | no |
| 在线 FlashSoftmax rescale（running max/sum） | kernel.h L642-652,717-731；epilogue_online_softmax | `tl.max`/`tl.sum` + `exp` rescale + PV 累加 | directly transferable | inferred | no |
| scale_value 缩放（默认 1/√D） | def.cpp L68；tiling.cpp L492-495 | `s *= scale` | directly transferable | inferred | yes（null 指针默认语义） |
| FP32 累加（FLOATSM，BF16/FP16 输入） | epilogue_online_softmax（float 特化） | `tl.dot(..., out_dtype=tl.float32)` | directly transferable | inferred | no |
| GQA groupSize 共享 KV（N1≥N2, N1%N2==0） | tiling.cpp L572；kernel.h L419,506-507 | Q head 维循环复用同 KV | directly transferable | inferred | no |
| softmaxLse 输出（softmaxLseFlag） | def.cpp L73；kernel.h L312-313,535,542；rescale_o L295 | `tl.store(lse_ptr, m_i+log(l_i))` | directly transferable | inferred | yes（lse=ln(sum) 还是 max+ln(sum) 表达） |
| blockSparseMask 语义（1=算/0=跳，块级） | docs L148-163；README L123-146 | mask 决定选中 KV 块集合 | directly transferable | inferred | no |
| ceilDiv 非对齐分块 + tail | tiling.cpp L95-110；README L504 | `tl.arange` + `mask=` 边界 | directly transferable | inferred | no |
| blockShapeY 必须 128 倍数 / blockShape 默认 [128,128] | tiling.cpp L396-400,422-424 | `tl.constexpr` 对齐约束 | directly transferable | inferred | no |
| **blockSparseMask -> CSR(selectIdx,selectNumIdx) 压缩** | kernel.h L93-248（Mask2IdxAndCount） | 预处理：`tl.load(mask)` -> 压缩得选中 KV-block 索引数组 | rewrite | inferred | yes（压缩粒度/并行策略） |
| **稀疏 KV gather（沿 selectIdx load K/V）** | block_mmad_qk.hpp L152-227；block_mmad_pv.hpp L121-186 | `idx=tl.load(selectIdx); tl.load(k_ptr+idx*...)` 间接 load | rewrite | inferred | no |
| 全 0 行整 Q-block 跳过（curSelectNum==0 continue） | kernel.h L510-514 | `if valid_count==0: continue` | rewrite | inferred | no |
| core id / 任务切分（Q-block×head） | tiling.cpp L557-603；kernel.h L447-501 | `tl.program_id` + `grid` | rewrite | inferred | no |
| BASIC_BLOCK_SIZE=128 / tile shape <128,128,128> | kernel_common.hpp L40-84；interface.cpp L68-79 | `tl.constexpr BLOCK_M/N/K`（值 source-reported） | rewrite | inferred | yes（最优值） |
| QK/PV 块 mmad（L1/L0A/L0B/L0C staging） | block_mmad_qk/pv.hpp | `tl.dot`（编译器选 buffer） | rewrite | inferred | no |
| innerPrecise=1 FP16 softmax（HALFSM/low_prec） | tilingkey.h；epilogue_online_softmax_low_prec | `if constexpr` + `out_dtype=tl.float16` | rewrite | inferred | yes（精度/溢出权衡） |
| 12 tiling key 预处理分派 | block_sparse_attention.cpp L29-94 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| 4 阶段 cube↔vector 流水（QK->sm->PV->rescale） | kernel.h L596-731 | 抽象为 tile 循环 + fused op（不显式 CV 分离） | rewrite | inferred | no |
| MTE DataCopy GM↔L1/L0/UB（mask 压缩、KV 搬运） | Mask2Idx L162-171；copy_*.hpp | Triton load/store（编译器） | CANN-specific | inferred | no |
| CrossCoreSetFlag/WaitFlag（qkReady/softmaxReady/pvReady，3 event） | kernel.h L633,640,652,697,715；cross_core_sync.hpp | 不迁移；抽象为 tile 依赖 | CANN-specific | inferred | no |
| PreloadPipeline 3-slot（PRE_LAUNCH=2 循环 buffer） | kernel.h L572,596-605；kernel_common.hpp L77 | 不迁移；抽象为 tile 复用 | CANN-specific | inferred | no |
| TQue/LocalTensor/BufferID（UB/L1/L0A/L0B/L0C） | local_tensor_buffer.hpp；resource.hpp | 不迁移（CANN 缓冲区模型） | CANN-specific | inferred | no |
| ND↔NZ / DataCopyPad 格式转换 | copy_gm_to_l1.hpp；layout/matrix.hpp | 不迁移；Triton layout（编译器） | CANN-specific | inferred | no |
| KERNEL_TYPE_MIX_AIC_1_2（CV 1:2） | block_sparse_attention.cpp L23 | 不迁移（CANN 算力配比） | CANN-specific | inferred | no |
| GatherMask/CompareScalar（mask 压缩原语） | kernel.h L189-202 | `tl.where`/布尔索引（语义等价） | CANN-specific | inferred | no |
| tiling 参数具体数值（blockStackNum、sBase 等） | tiling.cpp | constexpr 值 source-reported | needs verification | inferred | yes |
| UB/L1 字节预算 / workspace 占用 | tiling.cpp L605-630 | 需硬件实测 | needs verification | inferred | yes |
| maskType=3 causal 分支（死代码） | tiling.cpp L730-732；README L163 | 文档/代码不一致，仅 maskType=0 支持 | needs verification | inferred | yes |
| innerPrecise=4 (LOW_HIGH_MIXED) 无 kernel 路径 | tiling.h L116；tilingkey.h | 仅 0/1 有分派，4 未实现 | needs verification | inferred | yes |
| PAGED_CACHE_FLAG=false 但 blockTable 仍传入 GEMM | kernel.h L619,681；L466-478 | paged cache 未支持，blockTable 是否消费待确认 | needs verification | inferred | yes |
| scaleValue attr null -> 1/√D 默认 | tiling.cpp L492-495 | 取决于框架 null 指针语义 | needs verification | inferred | yes |
