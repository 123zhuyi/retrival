# mhc_pre Performance Analysis

## Comparison with mhc_post

| Metric | mhc_post | mhc_pre |
|--------|----------|--------|
| Read | 1× input [B,S,D] | N× input [B×N,S,D] |
| Compute | 1 Muls per element | N Muls + (N-1) Adds per element |
| Write | N× output [B×N,S,D] | 1× output [B,S,D] |
| blockDim | B × N | B |

## Memory Bandwidth Analysis

For typical N=4 streams:

- mhc_post: reads B×S×D, writes B×N×S×D → 5× data movement
- mhc_pre: reads B×N×S×D, writes B×S×D → 5× data movement

Both operators have similar memory bandwidth requirements.

## Optimization Notes

### Dynamic Tiling

- Tile size computed based on UB_SIZE (192KB)
- Formula for fp32/fp16: `tile = UB_SIZE / ((BUFFER_NUM * 2 + 1) * sizeof(T))`
  - Buffers: inQue (2) + outQue (2) + tmpBuf (1) = 5 buffers
- Formula for bf16: `tile = UB_SIZE / ((BUFFER_NUM * 2 + 2) * sizeof(float))`
  - Additional fp32 accBuf for intermediate accumulation
- Tile is aligned to 8 (fp32) or 16 (fp16/bf16) elements

### BF16 Handling

- BF16 lacks native Muls support on some AI cores
- Compute path: Cast(bf16→fp32) → Muls → Add → Cast(fp32→bf16)
- Accumulation done in fp32 for numerical stability

## Benchmark Results

*Run `./perf_test` and `./perf_sweep` to collect hardware-specific data.*
