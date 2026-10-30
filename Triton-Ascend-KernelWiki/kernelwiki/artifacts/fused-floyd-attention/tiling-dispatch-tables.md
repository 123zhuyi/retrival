# FusedFloydAttention — Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

FloydAttention 变体，支持 2 组 KV head group。公式：

```
weights = Softmax(attenMask + scale * (einsum(Q, K1^T) + einsum(Q, K2^T)))
attention_out = einsum(weights, V1) + einsum(weights, V2)
```

**约束**（README.md L112-124）：
- B: 1~2K；H(N): 1~256；N(S1): 16~1M, N%16==0；M(S2): 128~1M, M%128==0；K(D): 32/64/128
- query 与 key1 第 0/2/4 轴相同；key1/value1 shape 相同；key2/value2 shape 相同
- D 仅支持 32/64/128

## B. Tiling 参数

4 次 matmul：bmm1=Q·K1^T、bmm1k1=Q·K2^T、bmm2=P·V1、bmm2v2=P·V2

TilingData 结构体（`tiling.h`）：
```
inputParams (bSize/n2Size/gSize/s1Size/s2Size/dSize/scaleValue/alignedS2)
coreParams (s1BaseSize/s2BaseSize/s1OuterSize/d2Size)
multiCoreParams (coreNum/totalSize/splitFactorSize)
bmm1/bmm1k1/bmm2/bmm2v2 TilingData
softmaxTilingData
```

## C. 模板 dispatch（kernel L114-150）

- **单模板** `FusedFloydAttentionS1s2Bn2gs1`（`fused_floyd_attention_s1s2_bn2gs1.h`）
- 4 分支：FP16/BF16 × hasAttenMask（tiling key `10000000001022030943UL` mask / `10000000000022030943UL` 无 mask）
- 仅 BNSD layout（`LayOutTypeEnum::LAYOUT_BNSD`）
- 仅 ascend910b/910_93（不支持 ascend950）

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| 核间切分 (B,N,S1_block) | grid + tl.program_id |
| S1/S2 基本块 | BLOCK_M/BLOCK_N constexpr |
| 4 次 matmul 编排 | 2 次 QK + softmax + 2 次 PV |
| FP32 累加 | tl.dot(..).to(tl.float32) |
| 双 KV head group | 分别 tl.dot 后相加 |