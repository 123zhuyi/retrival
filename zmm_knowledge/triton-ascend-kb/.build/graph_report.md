# OKF graph build report

params: K=8 theta_ratio=0.35 M=10 max_df=15 group_cap=12

## edges
- total: 8115
- by type: {'same_topic': 1445, 'part_of': 475, 'mentions': 4316, 'demonstrates': 6, 'exemplifies': 1025, 'related': 644, 'explains': 204}
- by via: {'term': 2680, 'hier': 475, 'mention': 4573, 'tag': 1045}

## term vocabulary V (|V|=408) — top by df
| term | df |
|---|---|
| matmul | 55 |
| softmax | 30 |
| elementwise | 15 |
| class | 14 |
| gather_scatter | 14 |
| mask | 14 |
| operator_class | 14 |
| backward | 13 |
| loss | 13 |
| pass | 13 |
| attention-operations | 12 |
| online_softmax | 12 |
| two | 12 |
| 编译器约束/hints | 12 |
| cube_vector_pipe | 11 |
| flash_attention | 11 |
| quant | 11 |
| rmsnorm | 11 |
| routing | 11 |
| sparse | 11 |
| token | 11 |
| vector_core | 11 |
| add | 10 |
| l0c_to_ub | 10 |
| ptr | 10 |
| ascend950dt | 9 |
| autotune | 9 |
| constexpr | 9 |
| flash | 9 |
| grad | 9 |
| kv_cache | 9 |
| online | 9 |
| reduce | 9 |
| ascend910b2 | 8 |
| gdn | 8 |
| gqa | 8 |
| inference | 8 |
| precision | 8 |
| l0a_buffer | 7 |
| l0b_buffer | 7 |

## group_cap-skipped terms (group > 12, no same_topic clique): 10
backward, class, elementwise, gather_scatter, loss, mask, matmul, operator_class, pass, softmax

## stop tags (whole-bundle / category, |stop|=113)
91093, 91095, 910b2, 910b3, add, algorithm-numerics, api, ascend, ascend_extensions, attention, autotune, autotuning, backward, backward_fusion, binning, cache_protocol, causal_conv, celoss, cheatsheet, chunk_scan, code_skeleton, compilation, compute-pipeline, constexpr, conv1d, convolution, cross_entropy, cube, delta_rule, depthwise, dl, elementwise, embedding, entropy, exp, fla, flash_attention, flash_attention_causal, flash_linear_attention, format, fused_ops, fusion, gather, gather_scatter, gelu, gemm, gemm_basic, gqa, grid, hardware, hbm, index_copy, indexed_access, inverse, knowledge_unit, label_smoothing, layernorm, legacy, linear_attention, logsoftmax, logsumexp, lookup, loss, mamba, matmul, max, memory-sram, meta-skills, min, moe, moe_dispatch, mqa, mrope, mte, mul, normalization, npu, numerical_stability, online_algorithm, online_softmax, operator_class, parallel_pattern, parallel_reduction, partial_rope, portability, positional_encoding, recurrent, reduction, relu, retrieval, rmsnorm, rmsnorm_fused, rope, rotary_embedding, routing, scalar, scatter, slice, softmax, solve_tril, split_k, ssm, structured_matrix, sum, tensor_create, tiling, tle, token_permutation, triangular, triton_knowledge_retrieval, ub, vector, vectorization

## isolated concept nodes (no non-part_of edge): 0
