# Chunked Gated Delta Rule Forward-H (chunk_gated_delta_rule_fwd_h)

## Operator pattern

Chunked state-space / delta-net recurrence that computes per-chunk states S[K,V] and the updated value v_new:

```text
v_new = u - w @ S_prev
S_next = S_prev * exp(g_last) + k.T @ (v_new * exp(g_last - g))
```

- Inputs: k [B,HK,T,K], w [B,HV,T,K], u [B,HV,T,V], g [B,HV,T]
- Outputs: states [B,HV,NT,K,V], v_new [B,HV,T,V], final_state [N,HV,K,V]
- Decay g is produced by `logsigmoid` + chunk-local cumsum, always negative.

## Proven Triton-Ascend recipe

1. **Host layout flattening**: pre-transpose k/w/u to flat `[B*HV, T, D]` on host so the kernel grid can be `(b*hv, v_tile)`.
2. **One kernel per (b,hv,V-tile)**: serial chunk loop inside the kernel, `S[K,V]` kept in fp32 accumulator registers.
3. **BLOCK_L = chunk_size**: chunk outer loop; inner positions serial within chunk.
4. **Head-group map**: when `HK != HV`, use `index_select` / modulo map to gather the correct k head for each v head.
5. **State-store semantics**: store the **post-update** state S at each chunk index to match the framework golden; v_new uses the pre-update state.
6. **Overflow-safe clamping**: clamp intermediate fp32 values (`|x| <= 1e20`) before conversion to prevent `Inf/NaN` on large dense/varlen cases.
7. **Varlen packed state layout**: varlen output state is packed as `[1, HV, total_NT, K, V]`; use the head id (not `seq*HV+head`) for state indexing.
8. **Dot precision**: use `tl.dot(..., input_precision="ieee")` for fp32 accumulator accuracy.

## Numerical correctness checklist

- [ ] Framework golden matches the operator spec: `v_new = u - w @ S` then `S = S*exp(g_last) + k.T @ (v_new*exp(g_last - g))`.
- [ ] Varlen golden uses `batch_idx=0` for data tensors and correct sequence offsets from `cu_seqlens`.
- [ ] NT (number of chunks) computed per sequence, summed across the batch for packed state layout.
- [ ] All intermediate tensors clamped to finite range before dtype conversion.
- [ ] Verify passes 8/8 before accepting speedup.

## Known pitfalls

- **State semantics bug**: storing the pre-update state instead of post-update state causes all chunks after the first to drift.
- **AICore exception on varlen**: wrong `state_bh` indexing triggers out-of-bounds GM writes.
- **Profiler collection flakiness**: medium-size head-ratio cases may fail to collect profiler data while remaining numerically correct; do not reject the kernel solely for `PROFILER_COLLECT_FAIL`.

## Typical speedup

- Ascend910B1, bf16: ~5.7x vs pure-PyTorch framework golden when target is 5.0x.
