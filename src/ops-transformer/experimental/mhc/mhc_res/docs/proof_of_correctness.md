# mhc_res Correctness Proof

## Algorithm Definition

mhc_res implements stream mixing, corresponding to the H_res matrix operation in mHC:

```text
out[b*S + t, seq, d] = Σ_s (h_res[s, t] × x[b*S + s, seq, d])
```

In einsum notation: `"st, bsld -> btld"`

## Correspondence to Open Source Implementation

Matches [tokenbender/mHC](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections):

```python
# From hyper_connections_mhc.py
h_res = sinkhorn_log(self.H_res_logits, num_iters=self.mhc_num_iters, tau=self.mhc_tau)
residuals_out = einsum(h_res, maybe_transformed_residuals, "s t, ... s d -> ... t d")
```

## Weight Semantics

- `h_res[s, t]` = weight from source stream s to target stream t
- The weight matrix is typically made doubly stochastic via Sinkhorn normalization upstream
- Our kernel accepts pre-normalized weights (normalization is done externally)

## Precision

| Dtype | Tolerance |
|-------|----------|
| fp32 | allclose(atol=1e-6, rtol=1e-5) |
| fp16 | allclose(atol=1e-3, rtol=1e-2) |
| bf16 | allclose(atol=1e-3, rtol=4e-3) |
