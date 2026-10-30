# ring_attention_update - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。支持 A2/A3 + 950 + Kirin。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 137 | doc | core | 功能 L14-15；rescale 公式 L18-28；参数 L30-125；约束 L127-131 | `6927ed510afb97bc...` |
| `docs/aclnnRingAttentionUpdate.md` | 582 | doc/api | core | V1 API 语义 | `3b2f537173a107b1...` |
| `docs/aclnnRingAttentionUpdateV2.md` | 596 | doc/api | core | V2 API 语义（新增 SBH） | `7913f29d59c4d1a1...` |
| `op_host/ring_attention_update_def.cpp` | 169 | host/def | core | `RingAttentionUpdate` OpDef：输入 L22-63；输出 L64-78；属性 L79-80；AICore config(910b/910_93 L81-82 + 950 L84-92 + Kirin L94-96) | `30ce24c1e3574808...` |
| `op_host/ring_attention_update_tiling.cpp` | 634 | host/tiling | core | tiling 主逻辑 | `b4bf217ead457ee6...` |
| `op_host/ring_attention_update_tiling.h` | 116 | host/tiling | core | `RingAttentionUpdateTiling` 类 | `a256381f5d82c2eb...` |
| `op_host/ring_attention_update_tiling_arch35.cpp` | 873 | host/tiling | core | arch35 专用 tiling | `a070f8419e189e85...` |
| `op_kernel/ring_attention_update.cpp` | 85 | kernel/entry | core | 6 tiling key 分支(0/1/2 SBH + 10/11/12 TND × FP16/BF16/FLOAT) L32-84；`KernelRingAttentionUpdate` / `KernelRingAttentionUpdateTND` | `52f7bc0669ed78b0...` |
| `op_kernel/ring_attention_update.h` | 385 | kernel | core | `KernelRingAttentionUpdate`：Process 循环 rescale | `8143e12addbd69f3...` |
| `op_kernel/ring_attention_update_tnd.h` | 445 | kernel | core | TND 布局 kernel | `45fb34934b6eb46f...` |
| `op_kernel/ring_attention_update_apt.cpp` | 60 | kernel/entry | core | 950 kernel 入口：3 tiling key(100/110/120 SBH/SoftmaxTND/TND) | `3d0b0480e733bbeb...` |
| `op_kernel/arch35/ring_attention_update_regbase_{common,sbh,softmax_tnd,tnd}.h` | 236-481 | kernel/arch35 | core | 950 4 模板 | 见前述 SHA-256 |

另含 `op_api/`(4)、`op_graph/`(2)、`tests/`(4) 等辅助文件。