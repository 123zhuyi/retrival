# fused_floyd_attention_grad CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki` | source-confirmed |
| OPERATOR_NAME | `fused_floyd_attention_grad` | source-confirmed |
| OPERATOR_TYPE | `attention` | source-confirmed |
| VARIANTS | `backward,training` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `attention/fused_floyd_attention_grad/`，源码 ~5k 行。Floyd Attention（Moonshot Kimi 长上下文）的反向梯度算子，双流注意力（K1/V1+K2/V2），仅 A2/A3 训练。三段式 kernel（Pre→S1S2→Post），6 个 matmul 反向。

完整清单见 `artifacts/fused-floyd-attention-grad/source_file_inventory.md`。核心文件：`README.md`（公式）、`def.cpp`（AICore config）、`kernel.cpp`（三段式入口 + TilingKey 分发）、`s1s2_bn2gs1s2.h`（6 个 matmul 反向核心）。

## 3. Operator semantics

- 输入（7+3 可选）：query/key1/value1/key2/value2/dy + atten_mask/softmax_max/softmax_sum/attention_in。
- 输出（5）：dq/dk1/dv1/dk2/dv2。
- 公式：6 个 matmul 反向（dS = S*dY, dV1 = P^T·dY, dV2 = P^T·dY, dQ = (dS·K1+dS·K2)/√d, dK1 = dS^T·Q/√d, dK2 = dS^T·Q/√d）。
- 约束：D 仅 32/64/128，N%16==0，M%128==0，K%128==0。
- 详见 `wiki/kernels/fused-floyd-attention-grad.md` §1-2。

## 4. Stage / pipeline table

| Stage | Unit | Op |
|---|---|---|
| Pre | Vector | dS = S * dY |
| S1S2 Bmm1 | Cube | dV1 = P^T · dY |
| S1S2 Bmm2 | Cube | dV2 = P^T · dY |
| S1S2 MM1 | Cube | dQ = (dS·K1 + dS·K2) / √d |
| S1S2 MM2 | Cube | dK1 = (dS^T · Q) / √d |
| S1S2 MM3 | Cube | dK2 = (dS^T · Q) / √d |
| Post | Vector | 梯度搬运 + 格式转换 |

## 5. Tiling parameters

- 切分：S1S2 BN2GS1S2 模板。
- TilingKey：4 种（BF16/FP16 × mask=0/mask=1）。
- 三段式 SyncAll 核间同步（Pre→S1S2→Post）。

## 6. Core partition strategy

- 核间：沿 BN2GS1 轴切分。
- 核内：6 个 matmul 反向（dV1/dV2/dQ/dK1/dK2）。

## 7. Kernel templates & dispatch

- 4 种 TilingKey（BF16/FP16 × mask=0/1）。
- 架构：`ascend910b`/`ascend910_93`（arch32），无 950，无 310P。

## 8. Pipeline analysis

| Stage | CANN mechanism | Triton abstraction | Transferable? |
|---|---|---|---|
| 6 matmul 反向 | Cube mm | `tl.dot` × 6 | 是（数学） |
| dS = S * dY | Vector elemwise | `dS = S * dY` | 是（数学） |
| 三段式 SyncAll | CrossCoreSync | 不迁移（Triton kernel launch） | 否（机制） |
| ND↔NZ 格式转换 | DataCopyPad | Triton layout（编译器） | 否（机制） |

## 9. CANN -> Triton-Ascend migration classification

- **transferable**：6 个 matmul 反向公式、dS = S * dY、双流相加、scale_value 缩放、FP32 累加、mask。
- **rewrite**：三段式 -> 单/多 kernel、TilingKey -> `if constexpr`、核间切分。
- **CANN-specific**：三段式 SyncAll、ND↔NZ 格式转换。
- **needs verification**：tiling 数值、dS 精确公式（依赖前向 softmax_max/sum 恢复）。

## 10. Triton-Ascend skeleton

`artifacts/fused-floyd-attention-grad/fused-floyd-attention-grad-triton-skeleton.py`，`reproducibility: pseudocode`。体现 6 个 matmul 反向 + 双流注意力。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/fused-floyd-attention-grad.md` | source | `ops-transformer-fused-floyd-attention-grad` | - | - | - |
| `wiki/kernels/fused-floyd-attention-grad.md` | kernel | `kernel-fused-floyd-attention-grad` | `[ops-transformer-fused-floyd-attention-grad]` | inferred | pseudocode |
| `artifacts/fused-floyd-attention-grad/PROVENANCE.yaml` | provenance | - | - | - | - |
| 5 artifacts 分析文件 | approach-notes | - | - | inferred | - |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**。
- 检索待补跑（分类器暂时不可用）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数数值 | 依赖运行时 shape/平台 |
| dS 精确公式 | 依赖前向 softmax_max/sum 恢复 |
| ND↔NZ 格式转换细节 | 以源码为准 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/fused-floyd-attention-grad/PROVENANCE.yaml` | bundle 溯源 |
| `kernelwiki/artifacts/fused-floyd-attention-grad/source_file_inventory.md` | S1 |
| `kernelwiki/artifacts/fused-floyd-attention-grad/migration-table.md` | S3 |
| `kernelwiki/artifacts/fused-floyd-attention-grad/tiling_dispatch_tables.md` | S4/S5 |
| `kernelwiki/artifacts/fused-floyd-attention-grad/pipeline-diagram.md` | S6 |
| `kernelwiki/artifacts/fused-floyd-attention-grad/fused-floyd-attention-grad-triton-skeleton.py` | S7 |
| `kernelwiki/sources/ops-transformer/fused-floyd-attention-grad.md` | S8 source |
| `kernelwiki/wiki/kernels/fused-floyd-attention-grad.md` | S8 kernel |
| `reports/fused-floyd-attention-grad-ingest-report.md` | S10 |