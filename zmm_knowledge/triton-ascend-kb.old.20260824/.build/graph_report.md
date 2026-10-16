# OKF graph build report

params: K=8 theta_ratio=0.35 M=10 max_df=15 group_cap=12

## edges
- total: 3927
- by type: {'same_topic': 1275, 'part_of': 225, 'demonstrates': 104, 'mentions': 1831, 'explains': 73, 'exemplifies': 363, 'related': 56}
- by via: {'term': 1815, 'hier': 225, 'mention': 2109, 'tag': 457}

## term vocabulary V (|V|=313) — top by df
| term | df |
|---|---|
| matmul | 22 |
| softmax | 21 |
| fp16 | 15 |
| block | 14 |
| class | 14 |
| operator_class | 14 |
| rmsnorm | 14 |
| grid | 12 |
| rope | 12 |
| routing | 12 |
| scheduling | 12 |
| cube_vector_pipe | 11 |
| elementwise | 11 |
| fusion | 11 |
| ssm | 11 |
| vector_core | 11 |
| kv_cache | 10 |
| l0c_to_ub | 10 |
| quant | 10 |
| token | 10 |
| triton | 10 |
| ascend950dt | 9 |
| constexpr | 9 |
| flash | 9 |
| general_ascend | 9 |
| grad | 9 |
| numerical_stability | 9 |
| reduce | 9 |
| sparse_attention | 9 |
| ascend910b2 | 8 |
| gemm | 8 |
| general_perf | 8 |
| gqa | 8 |
| inference | 8 |
| reduction | 8 |
| sort | 8 |
| sparse | 8 |
| stride | 8 |
| autotune | 7 |
| fused | 7 |

## group_cap-skipped terms (group > 12, no same_topic clique): 7
block, class, fp16, matmul, operator_class, rmsnorm, softmax

## stop tags (whole-bundle / category, |stop|=115)
91093, 91095, 910b2, 910b3, add, algorithm-numerics, api, ascend, ascend910b1, ascend_extensions, attention, autotune, autotuning, backward, backward_fusion, binning, cache_protocol, causal_conv, celoss, cheatsheet, chunk_scan, code_skeleton, compilation, compute-pipeline, constexpr, conv1d, convolution, cross_entropy, cube, delta_rule, depthwise, dl, elementwise, embedding, entropy, exp, fla, flash_attention, flash_attention_causal, flash_linear_attention, format, fused_ops, fusion, gather, gather_scatter, gelu, gemm, gemm_basic, gqa, grid, hardware, hbm, index_copy, indexed_access, instances, inverse, knowledge_unit, label_smoothing, layernorm, legacy, linear_attention, logsoftmax, logsumexp, lookup, loss, mamba, matmul, max, memory-sram, meta-skills, min, moe, moe_dispatch, mqa, mrope, mte, mul, normalization, npu, numerical_stability, online_algorithm, online_softmax, operator_class, parallel_pattern, parallel_reduction, partial_rope, portability, positional_encoding, recurrent, reduction, relu, retrieval, rmsnorm, rmsnorm_fused, rope, rotary_embedding, routing, scalar, scatter, slice, softmax, solve_tril, split_k, ssm, structured_matrix, sum, tensor_create, tiling, tle, token_permutation, triangular, triton_knowledge_retrieval, ub, vector, vectorization

## isolated concept nodes (no non-part_of edge): 0
