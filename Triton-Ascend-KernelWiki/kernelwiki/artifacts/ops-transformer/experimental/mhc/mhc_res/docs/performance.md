# mhc_res Performance

## Optimization Strategy

- **Multi-core parallel**: Each AI Core handles one or more (batch, target_stream) pairs
- **Dynamic tiling**: Tile size computed based on UB size and dtype
- **Double buffering**: Overlapped data movement and computation

## Tile Size Calculation

```cpp
// 5 buffers: inQue(2) + outQue(2) + tmpBuf(1)
tile = UB_SIZE / (5 * sizeof(T))
```

For bf16: 6 buffers (extra fp32 accumulator)

## Benchmark vs einsum

| Config | einsum | AscendC | Speedup |
|--------|--------|---------|--------|
| batch=8, seq=1024, dim=512, ns=4 | 0.44ms | 0.18ms | **2.5x** |
| batch=16, seq=256, dim=1024, ns=8 | 1.11ms | 0.36ms | **3.1x** |

## Memory Bandwidth

- Read: batch × num_streams × seq × dim × sizeof(T)
- Read: num_streams × num_streams × sizeof(T) (weights)
- Write: batch × num_streams × seq × dim × sizeof(T)
