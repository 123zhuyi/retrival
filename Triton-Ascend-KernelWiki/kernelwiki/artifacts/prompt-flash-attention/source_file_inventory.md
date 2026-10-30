# prompt_flash_attention — 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/prompt_flash_attention/`。`fixture: true`（摘录用于分析，非整仓复制）。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 174 | doc | core | 产品支持 L5-12；功能/公式 L14-34；参数 L36-86；约束 L88-168 | `e5524e4e192fc86b...` | 算子总览/约束 |
| `docs/PFA算子设计介绍.md` | 137 | doc | core | 实现原理(在线FlashSoftmax) L9-22；模板设计 L24-49；CV tiling分离 L55-59；tilingkey规则 L61-92；V/CV流水 L94-138 | `6a5e2b076a942d33...` | 设计权威 |
| `docs/aclnnPromptFlashAttention.md` | 638 | doc/api | core | V1 aclnn API（输入/输出/属性/约束） | `18a6df2ae0500cd1...` | V1 接口语义 |
| `docs/aclnnPromptFlashAttentionV2.md` | 773 | doc/api | core | V2 API（V2 相对 V1 新增：MLA/PA/quant 扩展） | `7dea6ccc395ce3e3...` | V2 接口语义 |
| `docs/aclnnPromptFlashAttentionV3.md` | 960 | doc/api | core | V3 API（V3 相对 V2 新增：FP8/FP4/learnableSink/rope 等） | `b34723da9fbf6e69...` | V3 接口语义 |
| `op_host/prompt_flash_attention_def.cpp` | 211 | host/def | core | `PromptFlashAttention` OpDef：输入输出dtype L26-79；属性 L80-103；AICore config(ascend910b/910_93/950/310p) L104-206 | `e34dcb4534aa2886...` | 算子定义 |
| `op_host/prompt_flash_attention_infershape.cpp` | 219 | host | core | infershape 规则 | `67258a42a7bb6016...` | shape 推导 |
| `op_host/prompt_flash_attention_tiling.cpp` | 6520 | host/tiling | core | arch分发 L429/433/1469/3920；splitCoreMode L569/926-932；基本块常量 L93-106；workspace L465-490；MLA判定 L159/2848；空KV L73/350 | `7a991bdbf31a81e1...` | arch32/A2 tiling 主逻辑 |
| `op_host/prompt_flash_attention_tiling.h` | 862 | host/tiling | core | tiling 数据结构（PromptAttentionBaseParams/PFAtensorSizeParams/PFAcoreParams/PFAmultiCoreParams 等） | `2a471558f53b9ff0...` | tiling 结构体 |
| `op_host/prompt_flash_attention_tiling_const.h` | 65 | host/tiling | core | dtype size L19-25；HIGH_PRECISION/PERFORMANCE L53-54；MAX_BATCH=256 L59；MAX_VAR_LEN_SEQ_LEN=4096 L61；COF={256..1024} L63 | `f32ec91ba467337d...` | tiling 常量 |
| `op_host/prompt_flash_attention_tiling_register.cpp` | 58 | host/tiling | core | `TilingPrepareForPromptFlashAttention` L25-52（读 aivNum/aicNum/ubSize/l1Size/l0C/L0A/L0B/socShortName）；TilingInputsDataDependency(actual_seq_q=5, kv=6) L55 | `74f2f6bd35d6f1ad...` | tiling 注册/平台查询 |
| `op_host/prompt_flash_attention_tiling_v2.cpp` | 5234 | host/tiling | core | V2 tiling（arch35/950 路径，MLA/PA/quant 扩展） | `—`（未单独计算） | arch35 tiling |
| `op_kernel/prompt_flash_attention.cpp` | 58 | kernel/entry | core | `prompt_flash_attention_FIAS`（全量入参，V2/V3）L25-46；`prompt_flash_attention`（V1 兼容）L48-59；转发 arch32 | `74f2f6bd35d6f1ad...`→实为 `c5b7ffa59784b248...` | kernel 入口 |
| `op_kernel/prompt_flash_attention_tilingkey.h` | 153 | kernel/dispatch | core | arch32 tilingkey 宏（~130 个，十进制组装：Q/KV/OUT dtype×layout×precision×API×feature） | `c5b7ffa59784b248...` | arch32 模板选择 |
| `op_kernel/arch35/prompt_flash_attention_template_tiling_key.h` | 1519 | kernel/dispatch | core | arch35 模板分发（if constexpr on layout/S1/S2/D/implMode...） | `24c1b73a34e93cb1...` | arch35 模板分发 |
| `op_kernel/arch35/prompt_flash_attention_template_tiling_key_enum.h` | 232 | kernel/dispatch | core | 枚举：Layout L51-52；D L53-67；S1 L69-76；S2 L78-87；ImplMode L89-93；PseMode L189-194；QuantMode L197-206；Mask L209-211；MatMulType L214-219；SplitCoreMode L222-227；ConfigValue[27] L129-157 | `—` | arch35 模板参数化 |
| `op_kernel/arch35/prompt_flash_attention_entry_regbase.h` | 532 | kernel | core | arch35 entry/Init/Process | `a91cbe229e8ca599...` | arch35 kernel 入口 |
| `op_kernel/arch35/service/prompt_flash_attention_mm1.h` | 176 | kernel | core | Cube 阶段：Q·K^T | `e8eca5c364f44f68...` | arch35 Bmm1 |
| `op_kernel/arch35/service/prompt_flash_attention_vector1.h` | 491 | kernel | core | Vector 阶段：pse/scale/mask + FlashSoftmax | `5ce1375116f60c8d...` | arch35 Vec1 |
| `op_kernel/arch35/service/prompt_flash_attention_mm2.h` | 224 | kernel | core | Cube 阶段：P·V | `—` | arch35 Bmm2 |
| `op_kernel/arch35/service/prompt_flash_attention_vector2.h` | 316 | kernel | core | Vector 阶段：rescale + 写回 | `—` | arch35 Vec2 |
| `op_kernel/arch35/vf_softmaxflashv3.h` | 2703 | kernel | core | 在线 FlashSoftmax（running max/sum/exp rescale） | `b66741db757a2097...` | arch35 softmax 实现 |
| `op_kernel/arch35/vf_flashupdate.h` | 739 | kernel | core | KV cache update 路径 | `—` | KV-cache 写回 |
| `op_kernel/arch35/prompt_flash_attention_kvcache.h` | 762 | kernel | core | KV cache 处理 | `—` | KV-cache |
| `op_kernel/arch35/prompt_flash_attention_comm.h` | 1013 | kernel/common | supporting | arch35 公共 buffer/sync | `—` | arch35 公共 |
| `op_kernel/arch35/prompt_flash_attention_sparse.h` | 149 | kernel | core | sparse mask 处理 | `—` | sparse |
| `op_kernel/prompt_flash_attention_arch32.h` | 826 | kernel | core | arch32 kernel 主体（A2/A3） | `—` | arch32 kernel |
| `op_kernel/arch32/prompt_flash_attention_s1s2_bns1_x910.h` | 2718 | kernel | core | arch32 S1S2 BNS1 模板（x910） | `—` | arch32 模板 |
| `op_kernel/arch32/prompt_flash_attention_bnstilling_n_s_tail.h` | 553 | kernel | core | arch32 BNSD tilling 含尾块模板 | `—` | arch32 模板 |
| `op_kernel/arch32/prompt_flash_attention_empty_tensor.h` | 61 | kernel | core | 空 tensor 模板 | `—` | 边界 |
| `op_host/op_api/aclnn_prompt_flash_attention*.cpp` | 99-828 | api | core | aclnn V1/V2/V3 包装 | `—` | C++ API |
| `tests/ut/op_host/test_prompt_flash_attention_infershape.cpp` | 955 | test | indirect | infershape UT | `—` | 测试 |
| `tests/ut/op_host/arch35/test_prompt_flash_attention_tiling.cpp` | 535 | test | indirect | arch35 tiling UT | `—` | 测试 |

注：标 `—` 的 SHA-256 未在本分析中单独计算（非核心或体量过大），如需完整冻结可在固定提交下补算。arch 子目录 `.../` 在本表中已展开为 `arch32/`/`arch35/`。
