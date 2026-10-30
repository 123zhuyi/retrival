# incre_flash_attention - CANN -> Triton-Ascend 迁移分类

四分类（rubric：directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。每条带 CANN 来源（path Lxx）。IFA 为增量解码算子，招牌技术为 FlashDecode（split-KV + lse 规约）。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 注意力公式 Softmax(QK^T/√d)V | README L30-40 | `tl.dot`+online softmax | transferable | inferred | no |
| 在线 FlashSoftmax rescale | 设计介绍 L22-32 | running max/sum + `exp(m_old-m_new)` rescale PV 累加 | transferable | inferred | no |
| query S=1 增量解码 | 设计介绍 L9 | `BLOCK_M=1`（单 token）；循环沿 S2 轴 | transferable | inferred | no |
| mask select->-inf（atten_mask true 遮蔽） | 设计介绍 L20 | `tl.where(mask, -inf, s)` | transferable | inferred | no |
| pse 位置编码 add | def.cpp `pse_shift` L70-86；V4 doc L145-153 | `s = s + pse` | transferable | inferred | no |
| scale_value 缩放 | def.cpp L289；V4 doc L275-283 | `s *= sm_scale` | transferable | inferred | no |
| FP32 累加（BF16/INT8 输入） | 设计介绍（FP32 中间） | `tl.dot(..., out_dtype=tl.float32)` | transferable | inferred | no |
| KV cache（历史 state 累积，per-batch S 不同） | 设计介绍 L10；V4 doc L20 | actualSeqLengths 驱动 per-batch KV 长度 + mask | transferable | inferred | no |
| 变长 actualSeqLengths（per-batch 实际 KV 长度） | def.cpp L104-121；V4 doc L165-173 | per-batch offset + mask 边界 | transferable | inferred | yes（TND/非连续 packing offset） |
| **FlashDecode（split-KV 规约，lse 加权求和）** | 设计介绍 L73/119-138 | `tl.program_id` 沿 S2 切多核 + lse `exp(lse)/sum(exp(lse))` 加权 reduce | transferable | inferred | yes（lse 精确公式以 `vf_softmax_const.h` 为准） |
| KV 左 padding（Smax - kvPaddingSize - actualSeqLen） | V4 doc L255-263；def.cpp `kv_padding_size` L256-271 | KV 起始偏移 = `Smax - kvPadding - actualLen`，越界返回 0 | transferable | inferred | yes（与 atten_mask 联动的精度边界） |
| GQA（G=queryN/kvN，UB 驱动 G 轴切分） | 设计介绍 L160-173 | `G` 维循环切分，sub_block 由 UB 容量决定 | rewrite | inferred | yes（G 切分阈值依赖 UB） |
| AntiQuant-MSD（A 变换适配 B，int8×int8=int32 reduce） | 设计介绍 L140-155 | Q 展开成多行 int8 -> `tl.dot(int8,int8)->int32` -> reduce | rewrite | inferred | yes（MSD 精度边界） |
| 量化 dequant_scale1/2 + quant_scale1/2 | def.cpp L122-206；V4 doc L175-223 | cast + scale 融合（per-tensor/channel） | rewrite | inferred | yes（量化精度边界） |
| antiquant_scale/offset（per-tensor/channel） | def.cpp L207-239；V4 doc L224-243 | 反量化 cast + scale*offset 融合 | rewrite | inferred | yes（per-channel shape 约束） |
| PageAttention（blockTable 查表 + 回调拷贝 GM->L1） | 设计介绍 L156-158；V4 doc L244-253 | `tl.load(block_table)` 查物理 KV 偏移 -> 间接 load | rewrite | inferred | yes（PA offset 算术 + block_size 对齐） |
| 三模板选择（C+V / All-Vector / preload） | 设计介绍 L40-57 | 按 dtype/layout/shape 选 `@triton.jit` kernel 分支 | rewrite | inferred | no |
| TilingKey 十进制 16 位组装 | 设计介绍 L189-244；`tilingkey.h` L18-454 | `if constexpr`/launcher dispatch（key 本身不迁移） | rewrite | inferred | no |
| 五轴切分 B/N2/G/S1=1/S2 | 设计介绍 L71 | `tl.program_id` 轴映射 + `grid` | rewrite | inferred | no |
| 核间 BN2 分核 + SplitKv（FlashDecode） | 设计介绍 L73 | `grid` 沿 BN2 与 S2-split 两维 | rewrite | inferred | no |
| 核内 S2 切分（FlashAttention 循环） | 设计介绍 L75 | `for start_n in range(0, kv_len, BLOCK_N)` | rewrite | inferred | no |
| 分核负载均衡重排（blockid 数组） | 设计介绍 L180-188 | launcher 内静态分块（Triton 无显式重排需求） | rewrite | inferred | no |
| MLA 模式（IFA 支持范围） | `tiling_mla.h` L27-98；`tiling.cpp` L297 | D 维拼接 constexpr + 分段 dot | rewrite | inferred | **yes（IFA 是否走 MLA 路径需确认；设计介绍 L56 称 MLA 全量化模板仅 FIA 调用）** |
| MTE 手动 DataCopy GM↔UB | 设计介绍 L156-158 | Triton load/store（编译器调度） | CANN-specific | inferred | no |
| BufferID/queue/event/CV 手动同步 | 设计介绍 L49/94-117 | 不迁移；抽象为 tile 复用/fused load-store | CANN-specific | inferred | no |
| ND↔NZ 物理格式转换 | arch35 `matmul_modules/ifa_copy_cube_in.h` | Triton layout（编译器管理） | CANN-specific | inferred | no |
| preload N-Buffer 流水 / Cube 双发 | 设计介绍 L49 | 不迁移（CANN 算力配比）；抽象为 tile 复用 | CANN-specific | inferred | no |
| Cube/Vector 算力配比（nRatio） | 设计介绍 L65-69 | 不迁移（CANN 专有） | CANN-specific | inferred | no |
| TBuf 放置 UB/L1/L0 + 平台查询 | `tiling_register.cpp` L22 | 不迁移；launcher 按 shape 选 meta | CANN-specific | inferred | no |
| tiling 参数具体数值（基本块大小） | `tiling.cpp`/`tiling_base.h` | constexpr 值 source-reported，不声称最优 | needs verification | inferred | yes |
| UB/L1/L0C 字节预算/占用 | `tiling.cpp` workspace L3597-3625 | 不声称；需硬件实测 | needs verification | inferred | yes |
| INT8/量化精度边界 | def.cpp dtype；V4 doc 约束 L455 | 需运行时验证 | needs verification | inferred | yes |
| FlashDecode lse 精确公式 | `vf_softmax_const.h`；设计介绍 L119-138 | 以源码为准 | needs verification | inferred | yes |
| V4 在 950PR/950DT 的支持 | V4 doc L9（×）vs def.cpp L316（注册 950） | V4 接口在 950 支持范围待确认 | needs verification | inferred | yes |
| arch20（310P）路径与 arch32 入口关系 | `def.cpp` L318-408；`arch20/*` | 310P 经 arch32 入口转 arch20 解码器，关系待确认 | needs verification | inferred | yes |
