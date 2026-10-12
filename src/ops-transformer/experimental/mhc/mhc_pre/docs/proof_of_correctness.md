# mhc_pre Correctness Proof

## 1. Paper Formula

mHC Paper Equation 3:

```text
x_{l+1} = H_l^{res} · x_l + H_l^{post}^T · F(H_l^{pre} · x_l, W_l)
                                            ^^^^^^^^^^^^^^^^^
                                            mhc_pre computes this
```

**mhc_pre implements `H_l^{pre} · x_l`** — weighted sum of N streams into 1 output.

```text
output[b, seq, d] = Σ_{s=0}^{N-1} input[b × N + s, seq, d] × h_pre[s]
```

Where:

- `input`: [batch × num_streams, seq_len, dim] — N streams per batch
- `h_pre`: [num_streams] — stream weights (can be positive, negative, or non-normalized)
- `output`: [batch, seq_len, dim] — aggregated result

## 2. Implementation Scope

**This implementation follows tokenbender/mHC open-source code:**

```python
# tokenbender/mHC: hyper_connections_mhc.py
self.static_alpha = nn.Parameter(torch.zeros(num_streams))  # shape [N]
alpha = self.static_alpha.softmax(dim=-1)  # or other activation
output = einsum(streams, alpha, "b s ... d, s -> b ... d")
```

- `h_pre` is a **static weight vector** [num_streams], shared across all batches and token positions
- Weight normalization (softmax/tanh/sigmoid) is handled **upstream**, not in this operator
- For per-batch or per-token dynamic weights, the interface would need to be extended

## 3. PyTorch Reference (tokenbender/mHC)

```python
# Source: hyper_connections_mhc.py width_connection()

def width_connection(self, streams, *, alpha):
    # alpha is h_pre, shape [num_streams]
    # streams shape: [batch * num_streams, seq, dim]
    
    # Step 1: "(b s) ... d -> b s ... d"
    # Reshape: [B×N, S, D] -> [B, N, S, D]
    streams = rearrange(streams, "(b s) ... d -> b s ... d", s=num_streams)
    
    # Step 2: "b s ... d, s -> b ... d"
    # Weighted sum: [B, N, S, D] × [N] -> [B, S, D]
    # out[b,seq,d] = Σ_s streams[b,s,seq,d] × alpha[s]
    output = einsum(streams, alpha, "b s ... d, s -> b ... d")
    
    return output
```

## 4. Implementation Comparison

### CPU Reference

```cpp
void cpu_reference_fp32(
    const float* input,   // [B×N, S, D]
    const float* h_pre,   // [N]
    float* output,        // [B, S, D]
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
) {
    int64_t E = seq_len * dim;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < E; ++i) {
            float sum = 0.0f;
            for (int64_t s = 0; s < num_streams; ++s) {
                sum += input[(b * num_streams + s) * E + i] * h_pre[s];
            }
            output[b * E + i] = sum;
        }
    }
}
```

### NPU Kernel

```cpp
__aicore__ inline void ProcessOneTile(int64_t batch_idx, int64_t tile_off, int64_t len) {
    LocalTensor<T> acc = outQue.AllocTensor<T>();
    LocalTensor<T> tmp = tmpBuf.Get<T>();
    
    for (int64_t s = 0; s < num_streams; ++s) {
        int64_t in_off = (batch_idx * num_streams + s) * batch_elements + tile_off;
        gm_in.SetGlobalBuffer(gm_input + in_off, len);
        
        CopyIn(len);
        LocalTensor<T> in = inQue.DeQue<T>();
        T weight = gm_h.GetValue(s);
        
        if (s == 0) {
            Muls(acc, in, weight, len);
        } else {
            Muls(tmp, in, weight, len);
            Add(acc, acc, tmp, len);
        }
        inQue.FreeTensor(in);
    }
    outQue.EnQue(acc);
    CopyOut(tile_off, len);
}
```

## 5. Correctness Mapping

| Paper Formula | PyTorch | CPU | NPU |
|---------------|---------|-----|-----|
| `output[b, ...]` | einsum result `[B,S,D]` | `output[b * E + i]` | `gm_output + batch_idx * E` |
| `input[b×N+s, ...]` | rearrange `streams[b,s]` | `input[(b*N+s) * E + i]` | `gm_input + (b*N+s) * E` |
| `h_pre[s]` | `alpha[s]` | `h_pre[s]` | `gm_h.GetValue(s)` |
| `Σ_s ... × h_pre[s]` | einsum `"bs...d,s->b...d"` | `sum += ... * h_pre[s]` | `Muls` + `Add` loop |

## 6. Comparison with mhc_post

| Aspect | mhc_post | mhc_pre |
|--------|----------|--------|
| Paper Part | `H_l^{post}^T · F(...)` | `H_l^{pre} · x_l` |
| Operation | Broadcast (1 → N) | Reduce (N → 1) |
| Input | [B, S, D] | [B×N, S, D] |
| Output | [B×N, S, D] | [B, S, D] |
| Formula | `out[b×N+s] = in[b] × w[s]` | `out[b] = Σ_s in[b×N+s] × w[s]` |
| blockDim | B × N | B |

## 7. Test Verification

```text
=== mhc_pre Multi-DType Test ===
FP32: max_abs=8.94e-08  PASS  (accumulation rounding)
FP16: max_abs=7.00e-04  PASS
BF16: max_abs=1.95e-03  PASS

=== Edge Cases ===
dim=1, dim=7, dim=15      PASS  (non-aligned)
batch=1, seq=1            PASS  (boundary)
num_streams=1,2,4,8       PASS  (various N)
```

All tests confirm the implementation matches the paper formula.
