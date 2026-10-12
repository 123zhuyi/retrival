# mhc_post Correctness Proof

## 1. Paper Formula

mHC Paper Equation 3:

```text
x_{l+1} = H_l^{res} · x_l + H_l^{post}^T · F(H_l^{pre} · x_l, W_l)
                            ^^^^^^^^^^^^^^^^^
                            mhc_post computes this
```

**mhc_post implements `H_l^{post}^T · F(...)`** — broadcast 1 input to N streams.

```text
output[b × N + s, seq, d] = branch_output[b, seq, d] × h_post[s]
```

Where:

- `branch_output`: [batch, seq_len, dim] — output from branch module F(...)
- `h_post`: [num_streams] — learned weights (normalization handled upstream)
- `output`: [batch × num_streams, seq_len, dim] — distributed to N streams

## 2. PyTorch Reference (tokenbender/mHC)

```python
# Source: hyper_connections_mhc.py depth_connection()

def depth_connection(self, branch_output, residuals, *, beta):
    # beta is h_post, shape [num_streams]
    # branch_output shape: [batch, seq, dim]
    
    # Step 1: "b ... d, s -> b ... s d"
    # Broadcast multiply: [B, S, D] × [N] -> [B, S, N, D]
    # out[b,seq,s,d] = branch_output[b,seq,d] × beta[s]
    output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
    
    # Step 2: "b ... s d -> (b s) ... d"
    # Reshape: [B, S, N, D] -> [B×N, S, D]
    output = rearrange(output, "b ... s d -> (b s) ... d")
    
    return output
```

## 3. Implementation Comparison

### CPU Reference

```cpp
void cpu_reference_fp32(
    const float* branch_output,  // [B, S, D]
    const float* h_post,         // [N]
    float* output,               // [B×N, S, D]
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
) {
    int64_t E = seq_len * dim;  // elements per batch
    
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < num_streams; ++s) {
            float weight = h_post[s];
            int64_t out_batch = b * num_streams + s;  // output[b×N+s, ...]
            
            for (int64_t i = 0; i < E; ++i) {
                // output[b×N+s, i] = branch_output[b, i] × h_post[s]
                output[out_batch * E + i] = branch_output[b * E + i] * weight;
            }
        }
    }
}
```

### NPU Kernel

```cpp
__aicore__ inline void ProcessOne(int64_t batch_idx, int64_t stream_idx) {
    // in_off = b × E  ← branch_output[b, ...]
    int64_t in_off = batch_idx * batch_elements;
    // out_off = (b × N + s) × E  ← output[b×N+s, ...]
    int64_t out_off = (batch_idx * num_streams + stream_idx) * batch_elements;
    
    gm_in.SetGlobalBuffer(gm_branch + in_off, batch_elements);
    gm_out.SetGlobalBuffer(gm_output + out_off, batch_elements);
    
    T weight = gm_h.GetValue(stream_idx);  // h_post[s]
    
    for (int64_t i = 0; i < tiles; ++i) {
        CopyIn(off, len);
        Compute(len, weight);  // Muls(out, in, weight)
        CopyOut(off, len);
    }
}
```

## 4. Correctness Mapping

| Paper Formula | PyTorch | CPU | NPU |
|---------------|---------|-----|-----|
| `branch_output[b, ...]` | einsum input `[B,S,D]` | `branch_output[b * E + i]` | `gm_branch + batch_idx * E` |
| `output[b×N+s, ...]` | rearrange `(b s)` | `output[(b*N+s) * E + i]` | `gm_output + (b*N+s) * E` |
| `h_post[s]` | `beta[s]` | `h_post[s]` | `gm_h.GetValue(stream_idx)` |
| `× h_post[s]` | einsum `"b...d,s->b...sd"` | `* weight` | `Muls(out, in, weight)` |

## 5. Index Calculation Proof

**Given:** B=batch, N=num_streams, S=seq_len, D=dim, E=S×D

**Paper:** `output[b×N + s, seq, d] = branch_output[b, seq, d] × h_post[s]`

**Linear index verification:**

```text
branch_output[b, seq, d]  →  b × E + seq × D + d          ✓
output[b×N + s, seq, d]   →  (b×N + s) × E + seq × D + d  ✓
```

**NPU kernel:**

```cpp
in_off  = batch_idx * batch_elements
        = b × E                       // ✓ matches branch_output[b, ...]

out_off = (batch_idx * num_streams + stream_idx) * batch_elements
        = (b × N + s) × E             // ✓ matches output[b×N+s, ...]
```

## 6. Comparison with mhc_pre

| Aspect | mhc_post | mhc_pre |
|--------|----------|--------|
| Paper Part | `H_l^{post}^T · F(...)` | `H_l^{pre} · x_l` |
| Operation | Broadcast (1 → N) | Reduce (N → 1) |
| Input | [B, S, D] | [B×N, S, D] |
| Output | [B×N, S, D] | [B, S, D] |
| Formula | `out[b×N+s] = in[b] × w[s]` | `out[b] = Σ_s in[b×N+s] × w[s]` |
| PyTorch | `"b...d, s -> b...sd"` | `"bs...d, s -> b...d"` |
| blockDim | B × N | B |

mhc_post and mhc_pre are mathematical inverses in the mHC framework.

## 7. Test Verification

```text
=== mhc_post Multi-DType Test ===
FP32: bit_exact=yes  PASS  (0 mismatch)
FP16: max_abs=1.22e-04  PASS
BF16: max_abs=9.73e-04  PASS

=== Edge Cases ===
dim=1, dim=7, dim=15      PASS  (non-aligned)
batch=1, seq=1            PASS  (boundary)
num_streams=1,2,4,8       PASS  (various N)
```

All tests confirm the implementation matches the paper formula.
