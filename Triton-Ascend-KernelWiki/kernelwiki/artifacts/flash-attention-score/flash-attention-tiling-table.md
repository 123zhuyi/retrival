# FlashAttention Tiling Parameter Table

Derived from static analysis of CANN ops-transformer FlashAttentionScore (commit d28d8f99).
`confidence: inferred`, `reproducibility: snippet`.

## 1. arch32 (A2 训练系列) 多模板 tiling 参数

### 1.1 模板选择条件与 CV 基本块

| 模板 | 核内切分 | 核间切分 | 进入条件 | S1.i (CV) | S2.i (CV) | nRatio |
|------|---------|---------|---------|-----------|-----------|--------|
| TNDSameAB | UB切S1 | B,N2,G | TND & ((B≥4 & S1≥8192 & S2≥8192 & maxS1≥512 & maxS2≥512) ∥ (maxS2≥5120 & maxS1≥5120)) | — | — | — |
| TND | UB切S1S2 | B,N2,G,S1 | TND & 不满足TNDSameAB | — | — | — |
| SameAB | UB切S1 | B,N2,G,S1 | !TND & !FP32 & S2≥512 & (D%16≠0 ∥ D=96) ∥ S2>1024 & 128<D<196 | — | — | — |
| S1S2 | UB切S1S2 | B,N2,G,S1 | S2>1024 & 不满足上述 | 默认64; B*N2*G*S1.o>VecCores→128 | 1024 | 8 |
| S1 | UB切S1D | B,N2,G,S1 | 中等shape, 不满足上述 | 依据S2动态调整 | 不切分,=S2 | 8 |
| B | — | B | 小shape兜底 | — | — | — |

### 1.2 CompileInfo 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| aivNum | uint32_t | Vector 核数 |
| aicNum | uint32_t | Cube 核数 |
| ubSize | uint64_t | UB 容量 (Byte) |
| l1Size | uint64_t | L1 容量 (Byte) |
| l0cSize | uint64_t | L0C 容量 (Byte) |
| l2CacheSize | uint64_t | L2 Cache 容量 (Byte) |
| socVersion | SocVersion | SoC 版本枚举 |
| npuArch | NpuArch | NPU 架构枚举 |

### 1.3 CoreParams 参数（通用 tiling 数据结构）

| 参数 | 类型 | 说明 |
|------|------|------|
| s1BaseSize | int32_t | S1 核内基本块大小 |
| s1BaseTailSize | int32_t | S1 核内基本块尾块大小 |
| s1OuterSize | int64_t | S1 核内循环次数 |
| s2BaseSize | int32_t | S2 核内基本块大小 |
| s2BaseTailSize | int32_t | S2 核内基本块尾块大小 |
| s2OuterSize | int64_t | S2 核内循环次数 |
| dBaseSize | int32_t | D 核内基本块大小 |
| bBaseSize | int32_t | B 核间基本块大小 |
| n2BaseSize | int32_t | N2 核间基本块大小 |
| gBaseSize | int32_t | G 核间基本块大小 |
| nRatio | int32_t | CV 配比 (通常=8) |
| dRopeSize | int32_t | RoPE 维度 |
| d2Size | int32_t | D2 (Dv) 维度 |

### 1.4 TensorSizeParams（UB/L1 内存分配）

| 参数 | 类型 | 说明 |
|------|------|------|
| bmm1ResUbSize | int32_t | Bmm1 结果 UB 大小 |
| attenMaskUbSize | int32_t | attenMask UB 大小 |
| pseUbSize | int32_t | pse UB 大小 |
| dropMaskUbSize | int32_t | dropMask UB 大小 |
| softmaxMaxUbSize | int32_t | softmax max UB 大小 |
| softmaxSumUbSize | int32_t | softmax sum UB 大小 |
| softmaxExpUbSize | int32_t | softmax exp UB 大小 |
| bmm2ResUbSize | int32_t | Bmm2 结果 UB 大小 |
| apiTmpBufferBytes | int32_t | API 临时缓冲区大小 |
| inputQueBytes | int32_t | Cube 输入队列大小 |
| outputQueBytes | int32_t | Cube 输出队列大小 |

## 2. arch35 (950PR/950DT) 统一模板 tiling 参数

| 参数 | 值/范围 | 说明 |
|------|---------|------|
| S1.i | 128（少量 64） | 核内基本块 |
| S2.i | 128（少量 256） | 核内基本块 |
| D | 64-768 | head_dim 范围 |
| Dv | 64-768 | value head_dim（可 ≠ D） |
| 模板数 | 1 套 | 不区分 layout/TND |

## 3. CV tiling 分离参数

| 架构 | Cube 基本块 | Vector 基本块 | nRatio | Cube→Vector 通路 |
|------|-----------|-------------|--------|-----------------|
| A2 (arch32) | (128, 1024) | (8, 1024) | 8 | L2 + GM |
| 950PR (arch35) | (128, 128) | 同 Cube | 1 | UB (融合) / GM (Dv>128) |

## 4. Workspace 分配

| 场景 | 大小 | 说明 |
|------|------|------|
| 空 tensor | 100 MB | 固定预留 |
| 正常 (arch32) | wkspSection1 + wkspSection2 | mm1 临时 + mm2/bmm2 临时 |
| 正常 (arch35) | wkspSection1 + wkspSection2 | 同上，偏移由 tiling 计算 |

## 5. Sparse 模式编码

| Sparse 值 | 含义 |
|-----------|------|
| 0 | ALL |
| 1 | NONE |
| 2 | ANY |
| 3 | CAUSAL |
| 4 | BAND |
| 5 | PREFIX |
| 6 | BAND_COMPRESS |
| 7 | RIGHT_DOWN_CAUSAL |
| 8 | RIGHT_DOWN_CAUSAL_BAND |
| 9 | BAND_LEFT_UP_CAUSAL |
