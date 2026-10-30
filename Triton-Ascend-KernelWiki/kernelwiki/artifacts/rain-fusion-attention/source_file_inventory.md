# rain_fusion_attention - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录（基于 CATLASS 模板库）。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 193 | doc | core | 功能/特性 L3-14；接口定义 L16-41；参数 L53-81；稀疏模式 L138-163 | `e698daceeddf158e...` |
| `docs/aclnnRainFusionAttention.md` | 766 | doc/api | core | aclnn API 语义 | `d1f07c047c406569...` |
| `op_host/rain_fusion_attention_def.cpp` | 84 | host/def | core | `RainFusionAttention` OpDef：输入 L19-58；输出 L59-66；属性 L68-74；AICore config(910b/910_93) L76-77 | `d8e916c4b897ac13...` |
| `op_host/rain_fusion_attention_tiling.cpp` | 1014 | host/tiling | core | tiling 主逻辑 | `726711b4c745d9e5...` |
| `op_host/rain_fusion_attention_tiling.h` | 216 | host/tiling | core | tiling 类 | `961ee87783ac22d0...` |
| `op_kernel/rain_fusion_attention.cpp` | 78 | kernel/entry | core | kernel 入口 L26-77；6 tiling key 分支 L40-75；MIX_AIC_1_2 L34 | `e692caebee2877b0...` |
| `op_kernel/rain_fusion_attention_tilingkey.h` | 48 | kernel/dispatch | core | tiling key 位域编码 L17-26；6 个 key L31-46；RFA_BASE_TILING=9000000000000000 L28 | `cad683fd297a2bff...` |
| `op_kernel/rain_fusion_attention_interface.cpp` | 114 | kernel | core | `RainFusionAttentionInfer` 模板调用接口 | `3fd58ceff392f35d...` |
| `op_kernel/rain_fusion_attention_kernel.h` | 589 | kernel | core | `RainFusionAttentionInfer` kernel 主体（selectIdx 驱动 QK/PV + online softmax） | `be6c4e8c3a424912...` |

另含 `op_api/`(4)、`tests/`(2) 等辅助文件。