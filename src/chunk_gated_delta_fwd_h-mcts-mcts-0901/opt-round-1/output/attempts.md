# Round 1 optimization attempts

- [已尝试] Fix state-store semantics: store POST-update state S at each chunk index.
- [已尝试] Add overflow-safe clamping (|x| <= 1e20) for fp32 intermediate values.
- [已尝试] Fix varlen state indexing to use head id for packed [1,HV,total_NT,K,V] layout.
- [已尝试] Correct framework golden for varlen mode (NT computation, batch_idx=0, finite clamp).
