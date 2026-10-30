# incre_flash_attention - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/incre_flash_attention/`。`fixture: true`（摘录用于分析，非整仓复制）。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位（完整值可在固定提交下重算）。

## docs

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 143 | doc | core | 产品支持 L5-13；功能/公式 L14-40；约束 L99-138 | `91148eb6851da8e6…` | 算子总览（基础版） |
| `docs/IFA算子设计介绍.md` | 244 | doc | core | 实现原理(FlashSoftmax在线) L12-32；模板设计 L34-57；数据切分 L60-75；主流程伪码 L77-117；FlashDecode L119-138；AntiQuant-MSD L140-155；PageAttention L156-158；GQA L160-173；Tiling设计 L175-188；TilingKey L189-244 | `66948064d6b6914c…` | **设计权威** |
| `docs/aclnnIncreFlashAttention.md` | 514 | doc/api | indirect | V1 aclnn API | `dc85b2ac310a8b86…` | V1 接口语义 |
| `docs/aclnnIncreFlashAttentionV2.md` | 577 | doc/api | core | V2 API（V2 相对 V1 扩展） | `2af5acda05677d7d…` | V2 接口语义 |
| `docs/aclnnIncreFlashAttentionV3.md` | 652 | doc/api | core | V3 API（V3 相对 V2 扩展） | `b03f0c79886f48f7…` | V3 接口语义 |
| `docs/aclnnIncreFlashAttentionV4.md` | 791 | doc/api | core | V4 API（V4 相对 V3 新增 KV 左 Padding L18）；参数表 L88-354；约束 L436-472 | `44e4cc9ef651e844…` | **V4 接口语义（主）** |

## op_host

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/incre_flash_attention_def.cpp` | 412 | host/def | core | `IncreFlashAttention` OpDef L18-409；输入输出 dtype L22-287；属性 L288-293；AICore config（910b/910_93 L303-304、950 L306-316、310p L318-408） | `fe0be89bbe3fe2c4…` | 算子定义 |
| `op_host/incre_flash_attention_infershape.cpp` | 127 | host | core | `InferShapeIncreFlashAttention` L38；`InferDataTypeIncreFlashAttention` L104 | `6269bd7611803373…` | shape/dtype 推导 |
| `op_host/incre_flash_attention_tiling.cpp` | 4195 | host/tiling | core | arch32 tiling 主逻辑：arch/SocVersion 分发 L96-147；splitCoreMode L292-317/1936-1955；FlashDecode/SplitKv 判定 L2948-2993；MLA 判定 L297；PagedAttention 判定 L3847；workspace L3597-3625；`GenTilingKey` L3409-3470 | `c84dbe37632ed428…` | arch32 tiling |
| `op_host/incre_flash_attention_tiling_v2.cpp` | 4476 | host/tiling | core | arch35 tiling 主逻辑（arch 分发/分核/FlashDecode/MLA/PA/workspace/GenTilingKey，结构与 arch32 对应；精确行号以源码为准） | `a8590eddd3487f9a…` | arch35 tiling |
| `op_host/incre_flash_attention_tiling.h` | 195 | host/tiling | core | tiling 结构：`BaseParams` L44-88、`CoreParams` L90、`SplitCoreParams` L97、`SingleCoreParams` L114、`TilingData` L141、`TilingDataV2` L168 | `e68b2d10e40bc9dd…` | tiling 结构体 |
| `op_host/incre_flash_attention_tiling_v2.h` | 440 | host/tiling | core | arch35 tiling 结构体 | `87b79d9b8fe80b8c…` | arch35 tiling 结构体 |
| `op_host/incre_flash_attention_tiling_struct.h` | 161 | host/tiling | core | 辅助结构：`ActualSeqInfo` L29、`SeqTilingInfo` L37、`BalancedSplitTilingInfo` L47、`IfaWorkSpaceSizeParams` L61、`KvLayoutInfo` L72、`TilingIndexes` L77、枚举 L86-153 | `f7615e7a32aea167…` | tiling 辅助结构+枚举 |
| `op_host/incre_flash_attention_tiling_base.h` | 201 | host/tiling | core | 基础常量/数据类型映射（`DATATYPE_TO_STRING_MAP` L24、常量 L63-199） | `c563703f2ae91b00…` | tiling 基础常量 |
| `op_host/incre_flash_attention_tiling_impl.h` | 519 | host/tiling | core | `IFATiling` 类 L37-508（分块调度核心实现） | `09b4aa4116ad9596…` | tiling 实现类 |
| `op_host/incre_flash_attention_tiling_check.cpp` | 2032 | host/check | core | 约束检查：`CheckPABlockSize` L30、`CheckInputParameterFormat` L135、`CheckInputAntiquantFormat` L174、`CheckInputQKVTypeMatch` L323、`CheckQKOutShape` L538、`IsFlashDecode` L988、`CheckConstraints` L1097 | `3289a900e3da5f5e…` | 约束检查 |
| `op_host/incre_flash_attention_tiling_mla.h` | 102 | host/tiling | core | MLA tiling 结构：`BaseParamsMla` L27、`CoreParamsMla` L45、`SingleCoreParamsMla` L53、`SplitKVParamsMla` L67、`TNDSplitCoreParamsMla` L75、`TilingDataMla` L91 | `860d82ca9652a17a…` | MLA tiling 参数 |
| `op_host/incre_flash_attention_tiling_context.h` | 98 | host/context | core | tiling 上下文：`RequiredParaInfo` L25、`OptionalParaInfo` L30、`IncreFlashAttentionContext` L35-96 | `a74e5e2bab782d4f…` | tiling 上下文 |
| `op_host/incre_flash_attention_tiling_register.cpp` | 30 | host/register | core | `TilingPrepareForIncreFlashAttention` L22（平台查询 aivNum/aicNum/ub/l1）；`TilingIncreFlashAttention` L28 | `530bdb300b05de61…` | tiling 注册/平台查询 |
| `op_host/fallback_incre_flash_attention.cpp` | 129 | host/fallback | supporting | `IncreHostExecuteFunc` L45-122（host fallback 执行） | `616842685e63ab74…` | host fallback |
| `op_host/op_api/aclnn_incre_flash_attention*.cpp/.h` | 65-79 | api | core | aclnn V1/V2/V3/V4 包装 | `710a01f8…`/`310b191a…`/`2f8c193a…`/`79…` | aclnn C++ API |

## op_kernel（共享 + arch32 模板）

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/incre_flash_attention.cpp` | 55 | kernel/entry | core | `incre_flash_attention_FIAS` L27-42；`incre_flash_attention` L44-56（转发 arch32） | `e04dad525a025480…` | kernel 入口（arch32） |
| `op_kernel/incre_flash_attention_apt.cpp` | 72 | kernel/entry | core | arch35 入口：`incre_flash_attention_FIAS` L30-49；`incre_flash_attention` L50-73（模板分发） | `800c5be58a14b4ed…` | kernel 入口（arch35） |
| `op_kernel/incre_flash_attention_arch32.h` | 2095 | kernel | core | arch32 主体：`incre_flash_attention_FIAS_arch32` L203；`TILING_KEY_IS` 宏 L228-261/386-455；`INVOKE_IFA_*_OP_IMPL` L263-380（按 tiling key 分发模板） | `20cb2397d84cf5d5…` | arch32 kernel 主体 |
| `op_kernel/incre_flash_attention_tilingkey.h` | 454 | kernel/dispatch | core | arch32 tilingkey 宏定义 L18-454（十进制组装，见设计介绍 L189-244） | `3d832c813a18be5b…` | arch32 模板选择 |
| `op_kernel/ifa_public_define.h` | 846 | kernel/common | core | 公共定义：常量 L39-82/155-839；`CalcMode` L197；`LAYOUT` L203；`AMLAMODE` L213；`IFAType` L228；工具函数 L251-699 | `20e1ce3afce9c843…` | IFA 公共定义 |
| `op_kernel/arch32/incre_flash_attention_split_Bbn2s2_Us2.h` | 4218 | kernel/template | core | `IncreFlashAttentionAttenSplitBbn2s2Us2` L36；**C+V 基础模板**（Cube+Vector，matmul 高阶 API，设计介绍 L40） | `a5ef1c63be0303e4…` | C+V 基础模板 |
| `op_kernel/incre_flash_attention_allvec_new.h` | 1878 | kernel/template | core | `IncreFlashAttentionAttenAllVecNew` L68；**全 Vector 模板**（matmul 由 vector 实现，设计介绍 L41-45） | `1777ac030c0bba3b…` | 全 Vector 模板 |
| `op_kernel/arch32/incre_flash_attention_preload.h` | 4478 | kernel/template | core | `IncreFlashAttentionAttenPreload` L84；**matmul 基础 API 模板**（CUBE 视角 + N-Buffer 流水，设计介绍 L46-52） | `0fa3095cc72799a6…` | matmul 基础 API 模板 |
| `op_kernel/arch32/incre_flash_attention_preload_dd.h` | 4057 | kernel/template | indirect | 伪量化 MSD DD 模板（设计介绍 L54：**仅 FIA 调用，IFA 不调用**） | `5c6b7bd5119a39f2…` | DD 模板（IFA 不用） |
| `op_kernel/arch32/incre_flash_attention_preload_mla.h` | 4034 | kernel/template | indirect | MLA 全量化模板（设计介绍 L56：**仅 FIA 调用，IFA 不调用**） | `a77924451368857c…` | MLA 全量化模板（IFA 不用） |
| `op_kernel/arch32/service_vector_flashdecode_mla.h` | 827 | kernel/service | core | FlashDecode MLA 规约（设计介绍 L119-138） | `fdd64fa376b455d6…` | FlashDecode MLA 规约 |
| `op_kernel/arch32/paged_attention_antiquantkv.h` | 1984 | kernel/service | core | PA 场景 KV 反量化处理 | `b8015e5e1c8ef1fe…` | PA 反量化 KV |
| `op_kernel/arch32/ifa_service_matmul_full_quant.h` | 1210 | kernel/matmul | core | 全量化 matmul 服务（INT8/INT4，设计介绍 L142-155 MSD 算法） | `d0a15e853c856e19…` | 全量化 matmul |

## op_kernel（arch35）

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/arch35/incre_flash_attention_entry_regbase.h` | 333 | kernel/entry | core | arch35 entry/`INVOKE` 宏 L39-63 | `39223a41a750a952…` | arch35 kernel 入口 |
| `op_kernel/arch35/incre_flash_attention_template_tiling_key.h` | 2516 | kernel/dispatch | core | arch35 模板分发（模板参数 L35-119，`if constexpr` on layout/S/D/implMode/quant/mask） | `970ffe440799c4c8…` | arch35 模板分发 |
| `op_kernel/arch35/incre_flash_attention_tiling_regbase.h` | 383 | kernel/common | core | arch35 tiling 结构/注册 | `3006b27dba785164…` | arch35 tiling 注册 |
| `op_kernel/arch35/incre_flash_attention_pub.h` | 280 | kernel/common | core | arch35 公共定义 | `f692c409eec1a897…` | arch35 公共 |
| `op_kernel/arch35/kernel/incre_flash_attention_normal_Bbn2s2_Us2_regbase.h` | 1282 | kernel/template | core | arch35 normal 模式 split 实现 | `36cd9c5e2aeb4f0b…` | arch35 normal 模板 |
| `op_kernel/arch35/kernel/incre_flash_attention_antiquant_Bbn2s2_Us2_regbase.h` | 1789 | kernel/template | core | arch35 antiquant 模式 split 实现 | `67614b15dc3d113f…` | arch35 antiquant 模板 |
| `op_kernel/arch35/service/mm1_processor.h` | 117 | kernel/service | core | Cube 阶段 Q·K^T | `4462689de0c4caba…` | arch35 Bmm1 |
| `op_kernel/arch35/service/vec1_processor.h` | 366 | kernel/service | core | Vector 阶段 pse/scale/mask + FlashSoftmax | `45f2c46eefff7ce9…` | arch35 Vec1 |
| `op_kernel/arch35/service/mm2_processor.h` | 89 | kernel/service | core | Cube 阶段 P·V | `8c1b59a5b05bee87…` | arch35 Bmm2 |
| `op_kernel/arch35/service/vec2_processor.h` | 227 | kernel/service | core | Vector 阶段 rescale + 写回 | `977c1855a80300a7…` | arch35 Vec2 |
| `op_kernel/arch35/service/antiquant_processor.h` | 523 | kernel/service | core | 反量化处理 | `1fece20319f0e167…` | arch35 antiquant 处理 |
| `op_kernel/arch35/service/antiquant_preprocessor_q.h` | 119 | kernel/service | core | Query 反量化预处理 | `beabff0a1cce0e38…` | arch35 Q 反量化预处理 |
| `op_kernel/arch35/matmul_modules/ifa_copy_cube_in.h` | 112 | kernel/matmul | supporting | Cube 输入拷贝模块 | `5a274646cb189298…` | arch35 matmul 辅助 |
| `op_kernel/arch35/matmul_modules/ifa_cube_in_buffer.h` | 84 | kernel/matmul | supporting | Cube 输入缓冲模块 | `9e6e0c58ddf1c282…` | arch35 matmul 辅助 |
| `op_kernel/arch35/matmul_modules/ifa_flag_data.h` | 25 | kernel/matmul | supporting | 标志数据模块 | `bb9f1d5c86fca2c0…` | arch35 matmul 辅助 |
| `op_kernel/arch35/matmul_modules/ifa_matmul_policy.h` | 37 | kernel/matmul | supporting | Matmul 策略配置模块 | `4f45fe068bda8cdc…` | arch35 matmul 辅助 |
| `op_kernel/arch35/vector_api/vf_flashupdate.h` | 299 | kernel/vector-api | core | FlashUpdate（KV cache 更新路径） | `5132241dec798959…` | arch35 flashupdate |
| `op_kernel/arch35/vector_api/vf_softmax_const.h` | 917 | kernel/vector-api | core | 在线 FlashSoftmax（running max/sum/exp rescale） | `1356e612b02493ea…` | arch35 softmax 实现 |
| `op_kernel/arch35/vector_api/vf_row_invalid.h` | 91 | kernel/vector-api | supporting | 行无效标记 | `75707b410f1e9e0a…` | arch35 行无效处理 |
| `op_kernel/arch35/incre_flash_attention_dummy.h` | 54 | kernel | supporting | arch35 占位/桩 | `aadef035bb91dd85…` | arch35 桩 |

## op_kernel（arch20 / 310P，简述）

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/arch20/unpad_flash_attention_decoder.h` | 1504 | kernel/template | supporting | arch20（310P）非 PA 解码器 | `16eb606b1d7719d8…` | 310P 解码器（非 PA） |
| `op_kernel/arch20/unpad_paged_attention_decoder.h` | 1450 | kernel/template | supporting | arch20（310P）PA 解码器 | `ebd6eb1338dd8296…` | 310P 解码器（PA） |
| `op_kernel/arch20/incre_flash_attention_cube_310P_kvquant.h` | 1918 | kernel/template | supporting | arch20（310P）KV 量化 | `4a7083eac8b58d83…` | 310P KV 量化 |

## tests / examples / framework / op_graph

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `tests/ut/op_host/arch35/test_incre_flash_attention_infershape.cpp` | 569 | test | indirect | infershape UT | `905dbdbdbecd8f3e…` | 测试 |
| `tests/ut/op_host/arch35/test_incre_flash_attention_tiling.cpp` | 2424 | test | indirect | arch35 tiling UT | `d74ed9d55acad2d8…` | 测试 |
| `tests/st/aclnnIncreFlashAttentionV4/executor_aclnnIncreFlashAttentionV4.py` | 227 | test | indirect | V4 ST 执行器 | `21d990f563cbdc2e…` | 测试 |
| `examples/test_aclnn_incre_flash_attention.cpp` | 329 | example | indirect | aclnn 调用示例 | `4b5ba28d953cb56e…` | 示例 |
| `framework/npu_incre_flash_attention_onnx_plugin.cpp` | 165 | framework | supporting | ONNX 插件 | `6fac1ecc44a37ad3…` | ONNX 插件 |
| `op_graph/incre_flash_attention_proto.h` | 110 | op_graph | supporting | proto 定义 | `8d9b6732d108165b…` | op graph proto |

注：
- IFA 实际调用 3 种 arch32 模板（C+V `split_Bbn2s2_Us2` / 全 Vector `allvec_new` / matmul 基础 API `preload`）；`preload_dd`（伪量化 MSD DD）与 `preload_mla`（MLA 全量化）两种模板设计文档明确标注**仅 FIA 调用、IFA 不调用**（设计介绍 L54/L56）。
- `incre_flash_attention_allvec_new.h` 位于 `op_kernel/` 根目录（非 `arch32/` 子目录），与 `split_Bbn2s2_Us2.h`/`preload.h`（均在 `arch32/`）同为 arch32 三模板之一。
- arch35 `tiling_v2.cpp` 与 arch32 `tiling.cpp` 结构对应但为独立文件；本表对 v2 关键逻辑用概括描述，精确行号以源码为准。
