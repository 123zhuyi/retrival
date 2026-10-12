# Performance Data

## Test Environment

- **Hardware**: Ascend 910B2 (20 AI Cores)
- **CANN**: 8.5
- **Baseline**: torch.einsum on NPU

## Adaptive Strategy

The kernel automatically selects between two strategies based on shape:

- **Strategy A (per-stream)**: Parallelize over `(batch, stream)` pairs. Best for small/medium `seq×dim`.
- **Strategy B (read-once)**: Parallelize over `(batch, tile)` pairs, read input once and write N outputs. Best for large `seq×dim` with large batch.

Selection rule: Use B when `seq×dim×sizeof(T)×num_streams ≥ 4MB` AND `batch ≥ 16`.

## Performance vs einsum

| Shape | NPU (ms) | einsum (ms) | Speedup |
|-------|----------|-------------|--------|
| (4,512,256) ns=4 | 0.021 | 0.081 | **3.8x** |
| (8,1024,512) ns=4 | 0.031 | 0.081 | **2.6x** |
| (16,2048,1024) ns=4 | 0.510 | 0.480 | 0.94x |
| (32,2048,1024) ns=4 | 1.047 | 1.300 | **1.24x** |
| (16,64,4096) ns=4 | 0.040 | 0.079 | **2.0x** |
| (4,512,1024) ns=4 | 0.028 | 0.080 | **2.9x** |
| (8,256,512) ns=4 | 0.021 | 0.080 | **3.9x** |
| (16,1024,1024) ns=4 | 0.260 | 0.244 | 0.94x |

**Summary**: 6/8 shapes faster than einsum. The 6% slowdown on batch=16 + large elem is due to framework-level optimizations in aclnnMul that cannot be replicated.

## Data Types

| dtype | Precision Criterion |
|-------|--------------------|
| fp32  | bit-exact (ULP=0) |
| fp16  | allclose(atol=1e-4, rtol=1e-3) |
| bf16  | allclose(atol=1e-3, rtol=4e-3) |

## Implementation Notes

- **UB_SIZE**: 192KB
- **Double buffering**: BUFFER_NUM=2
- **Dynamic tiling**: Tile size computed based on `seq×dim` and dtype
- **BF16**: Uses fp32 compute path (Cast→Muls→Cast)

## Usage

```python
import mhc_post_ext

x = torch.randn(B, S, D, dtype=torch.float32, device='npu')
h = torch.randn(N, dtype=torch.float32, device='npu')
out = mhc_post_ext.forward(x, h)  # [B*N, S, D]
```
