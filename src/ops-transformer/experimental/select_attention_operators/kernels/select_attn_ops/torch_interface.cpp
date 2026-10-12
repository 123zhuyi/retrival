/**
 * -*- coding: utf-8 -*-
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acl/acl.h"

#define DIV_ROUNDUP_MUL(x,y) ((((x)+(y)-1) / (y)) * (y))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define BYTES_PER_VAL 2
#define BYTES_PER_IDX 4
#define BYTES_ASCEND_DATA_BLOCK 32
#define DIM3 3
#define DIM2 2
#define DIM1 1
#define DIM0 0
#define DIM128 128
#define MAXMBPR 6

// Helper function to check if tensor is bfloat16
bool is_bfloat16(const at::Tensor& tensor) {
    return tensor.scalar_type() == at::kBFloat16;
}

/*****************************************************************************/
/*** Prefill metadata kernel                                               ***/
/*****************************************************************************/
void launch_quest_prefill_metadata(
    uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *k_cache,
    uint8_t *block_tables,
    uint8_t *seq_lens,
    uint8_t *metadata_block_tables,
    uint8_t *maxblocks,
    uint8_t *minblocks,
    int32_t B,
    int32_t N,
    int32_t BLOCK_SIZE,
    int32_t D,
    int32_t MKBPR,
    int32_t MMBPR
);

/**
 * This is the interface function which is invoked from the python level. 
 * It handles:
 *  1. resolving tensor shapes
 *  2. Passing the pointers of the input tensors to the kernel
 *  3. Invocation of the kernel
 *  4. Returning the pointer of the output 
 * @brief Interface the `quest prefill metadata` kernel, which initializes of
 *        quest predictor metadata after LLM prefilling of the KV cache
 *
 * @param [in] k_cache (num_kv_blocks, BLOCK_SIZE, N, D)
 * @param [in] block_tables (B, MKBPR) - for every request b: block_tables[b] is 
 *                           a list of kv block indices 
 * @param [in] seq_lens (B,) - sequence length (in token number) per request
 * @param [in] metadata_block_tables (B, MMBPR) - for every request b: block_tables[b] is 
 *                           a list of metadata block indices 
 * @param [out] maxblocks (num_meta_blocks, BLOCK_SIZE, N, D) - maxblock metadata, 
 *                        arranged in blocks of equivalent size to kv_cache blocks
 * @param [out] minblocks (num_meta_blocks, BLOCK_SIZE, N, D) - maxblock metadata, 
 *                        arranged in blocks of equivalent size to kv_cache blocks
 *
 * note: num_kv_blocks can be equal to num_kv_blocks and one may even pass maxblocks = k_cache,
 * minblocks = v_cache to reuse the same page tables of vllm
 */
void quest_prefill_metadata(at::Tensor k_cache,
                            at::Tensor block_tables,
                            at::Tensor seq_lens,
                            at::Tensor metadata_block_tables,
                            at::Tensor maxblocks,
                            at::Tensor minblocks
)
{
    // infer tensor shapes
    int32_t B = seq_lens.sizes()[DIM0]; 
    int32_t N = k_cache.sizes()[DIM2];
    int32_t BLOCK_SIZE = k_cache.sizes()[DIM1];
    int32_t D = k_cache.sizes()[DIM3];
    int32_t MKBPR = block_tables.sizes()[DIM1]; // maximm kv-blocks specifiable per request
    int32_t MMBPR = metadata_block_tables.sizes()[DIM1]; // maximm metadata-blocks specifiable per request

    // validate input shapes
    TORCH_CHECK(D == DIM128, "D must be equal to ", DIM128, " for high performance operations, got ", D);
    TORCH_CHECK(BLOCK_SIZE == DIM128, "BLOCK_SIZE must be equal to ", DIM128, " for high performance operations, got ", BLOCK_SIZE);
    TORCH_CHECK(B == block_tables.size(DIM0), "Batch size mismatch: expected ", B, " from query, got ", block_tables.size(DIM0), " from block_tables");
    TORCH_CHECK(B == metadata_block_tables.size(DIM0), "Batch size mismatch: expected ", B, " from query, got ", metadata_block_tables.size(DIM0), " from metadata_block_tables");
    TORCH_CHECK(N == maxblocks.size(DIM2), "N (num KV heads) mismatch: expected ", N, " from query, got ", maxblocks.size(DIM2), " from maxblocks");
    TORCH_CHECK(N == minblocks.size(DIM2), "N (num KV heads) mismatch: expected ", N, " from query, got ", minblocks.size(DIM2), " from minblocks");
    TORCH_CHECK(BLOCK_SIZE == maxblocks.size(DIM1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, ", got ", maxblocks.size(DIM1), " from maxblocks");
    TORCH_CHECK(BLOCK_SIZE == minblocks.size(DIM1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, ", got ", minblocks.size(DIM1), " from minblocks");
    TORCH_CHECK(D == maxblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", maxblocks.size(DIM3), " from maxblocks");
    TORCH_CHECK(D == minblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", minblocks.size(DIM3), " from minblocks");

    // allocate input tensors
    uint8_t *k_cache_ptr = reinterpret_cast<uint8_t *>(k_cache.storage().data_ptr().get());
    uint8_t *block_tables_ptr = reinterpret_cast<uint8_t *>(block_tables.storage().data_ptr().get());
    uint8_t *seq_lens_ptr = reinterpret_cast<uint8_t *>(seq_lens.storage().data_ptr().get());
    uint8_t *metadata_block_tables_ptr = reinterpret_cast<uint8_t *>(metadata_block_tables.storage().data_ptr().get());
    uint8_t *maxblocks_ptr = reinterpret_cast<uint8_t *>(maxblocks.storage().data_ptr().get());
    uint8_t *minblocks_ptr = reinterpret_cast<uint8_t *>(minblocks.storage().data_ptr().get());

    // set up aunch parameters
    uint32_t blockDims = (B * N > NUM_CORES) ? NUM_CORES : B * N;
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream();

    // launch the kernel
    launch_quest_prefill_metadata(
        blockDims, nullptr, aclStream,
        k_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        metadata_block_tables_ptr,
        maxblocks_ptr,
        minblocks_ptr,
        B,
        N,
        BLOCK_SIZE,
        D,
        MKBPR,
        MMBPR
    );
}

/*****************************************************************************/
/*** Paged version (multiple metadata blocks per request, traveresed using ***/
/*** metadata_block_tables)                                                ***/
/*****************************************************************************/
void launch_quest_block_select_paged(
    uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *query, uint8_t *maxblocks,
    uint8_t *minblocks, uint8_t *metadata_block_tables,
    uint8_t *seq_lens, uint8_t *selected_indices,
    int32_t B, int32_t N, int32_t H,
    int32_t BLOCK_SIZE, int32_t D,
    int32_t MMBPR, int32_t num_meta_blocks,
    int32_t tokens_since_metadata_update,
    int32_t k, bool use_bfloat16
);

/**
 * @brief Interface the `quest_block_select_paged` kernel which predicts the
 *        sparsity mask during decoding in the form of top-k important kv-block 
 *        indices for every KV-head in every request. The returned KV block ids 
 *        are not the indices in the KV-cache, but rather from their enumeration 
 *        from 0 to number of blocks in the sequence length being decoded.
 *
 * @param [in] query Tensor(fp16) the query vector [B,H,D]
 * @param [in] maxblocks Tensor(fp16) quest metadata with the maximum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 *                       important: zeroes must be in place of metadata of non-existing kv blocks
 * @param [in] minblocks Tensor(fp16) quest metadata with the minimum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 *                       important: zeroes must be in place of metadata of non-existing kv blocks
 * @param [in] metadata_block_tables Tensor(int32) the metadata block tables [B,MMBPR]
 * @param [in] seq_lens Tensor(int32) sequence length of each request in the batch [B]
 * @param [in] k natural number of highest indices to return for every KV head
 * 
 * @returns selected_indices - Tensor(int32) where the kernel will output the selected 
 *          indices vector. [B,N,k] 
 */
at::Tensor quest_block_select_paged(at::Tensor query,
                                    at::Tensor maxblocks,
                                    at::Tensor minblocks,
                                    at::Tensor metadata_block_tables,
                                    at::Tensor seq_lens,
                                    int k
)
{
// round the k to a number that will require a multiple of 32 bytes
    int k_round = DIV_ROUNDUP_MUL(k * BYTES_PER_IDX, BYTES_ASCEND_DATA_BLOCK) / BYTES_PER_IDX;

    // infer tensor shapes
    int32_t B = query.sizes()[DIM0]; 
    int32_t N = maxblocks.sizes()[DIM2];
    int32_t H = query.sizes()[DIM1];
    int32_t BLOCK_SIZE = maxblocks.sizes()[DIM1];
    int32_t D = query.sizes()[DIM2];
    int32_t MMBPR = metadata_block_tables.sizes()[DIM1];
    int32_t num_meta_blocks = maxblocks.sizes()[DIM0];
    
    // validate input shapes
    TORCH_CHECK(BLOCK_SIZE == DIM128, "BLOCK_SIZE must be equal to ", DIM128, " for high performance operations, got ", BLOCK_SIZE);
    TORCH_CHECK(D == DIM128, "D must be equal to ", DIM128, " for high performance operations, got ", D);
    TORCH_CHECK(H % N == 0, "H must be divisible by N (GQA/MHA requirement). H=", H, " N=", N);
    TORCH_CHECK(B == metadata_block_tables.size(DIM0), "Batch size mismatch: query batch=", B, " metadata_block_tables batch=", metadata_block_tables.size(DIM0));
    TORCH_CHECK(B == seq_lens.sizes()[DIM0], "Batch size mismatch: query batch=", B, " seq_lens batch=", seq_lens.size(DIM0));
    TORCH_CHECK(BLOCK_SIZE == minblocks.size(DIM1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, " , got ", minblocks.size(DIM1), " from minblocks");
    TORCH_CHECK(N == minblocks.size(DIM2), "N (num KV heads) mismatch: expected ", N, " from query, got ", minblocks.size(DIM2), " from minblocks");
    TORCH_CHECK(D == minblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", minblocks.size(DIM3), " from minblocks");
    TORCH_CHECK(D == maxblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", maxblocks.size(DIM3), " from maxblocks");
    TORCH_CHECK(k > 0, "k must be positive, got ", k);
    TORCH_CHECK(num_meta_blocks == minblocks.size(DIM0), "num_meta_blocks mismatch: inferred ", num_meta_blocks, " from maxblocks, got ", minblocks.size(DIM0), " from minblocks");
    TORCH_CHECK(H / N <= BLOCK_SIZE, "H/N head group size cannot exceed BLOCK_SIZE=",BLOCK_SIZE, " given H/N=", H/N);
    TORCH_CHECK(MMBPR <= MAXMBPR, "maximum metablocks per request (MMBPR) cannot exceed ", MAXMBPR, " for this kernel. Your MMBPR=", MMBPR, " as inferred from dim=1 of metadata_block_tables argument.");

    // validate data types
    bool use_bfloat16 = is_bfloat16(query);
    TORCH_CHECK(use_bfloat16 == is_bfloat16(maxblocks), "query, maxblocks, minblocks input tensors must have the same data type");
    TORCH_CHECK(use_bfloat16 == is_bfloat16(minblocks), "query, maxblocks, minblocks input tensors must have the same data type");

    // allocate output tensor
    auto output_tensor_options = at::TensorOptions(query.options()).dtype(at::kInt);
    at::Tensor selected_indices = torch::empty({B, N, k_round}, output_tensor_options); 
    uint8_t *selected_indices_ptr = reinterpret_cast<uint8_t *>(selected_indices.storage().data_ptr().get());
    
    // allocate input tensors
    uint8_t *query_ptr = reinterpret_cast<uint8_t *>(query.storage().data_ptr().get());
    uint8_t *maxblocks_ptr = reinterpret_cast<uint8_t *>(maxblocks.storage().data_ptr().get());
    uint8_t *minblocks_ptr = reinterpret_cast<uint8_t *>(minblocks.storage().data_ptr().get());
    uint8_t *metadata_block_tables_ptr = reinterpret_cast<uint8_t *>(metadata_block_tables.storage().data_ptr().get());
    uint8_t *seq_lens_ptr = reinterpret_cast<uint8_t *>(seq_lens.storage().data_ptr().get());

    // set up launch parameters
    uint32_t blockDims = (B * N > NUM_CORES) ? NUM_CORES : B * N;
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream();

    // launch the kernel
    launch_quest_block_select_paged(
        blockDims, nullptr, aclStream,
        query_ptr,  maxblocks_ptr,
        minblocks_ptr,
        metadata_block_tables_ptr,
        seq_lens_ptr, selected_indices_ptr,
        B, N, H, BLOCK_SIZE, D, MMBPR,
        num_meta_blocks,
        -1, // disable tokens_since_metadata_update feature
        k_round, use_bfloat16
    );

    if (k != k_round) {
        // Trim the tensors to the original k size before returning
        selected_indices = selected_indices.slice(/*dim=*/2, /*start=*/0, /*end=*/k);
    }

    return selected_indices;
}

/**
 * @brief Alternative interface the `quest_block_select_paged` kernel which 
 *        predicts the sparsity mask during decoding in the form of top-k 
 *        important kv-block indices for every KV-head in every request. The 
 *        returned KV block ids are not the indices in the KV-cache, but rather 
 *        from their enumeration from 0 to number of blocks in the sequence 
 *        length being decoded.
 *
 *        FEATURE 1)  PREALLOCATED OUTPUT TENSOR (selected_indices)
 *
 * @param [in] query Tensor(fp16) the query vector [B,H,D]
 * @param [in] maxblocks Tensor(fp16) quest metadata with the maximum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 *                       important: zeroes must be in place of metadata of non-existing kv blocks
 * @param [in] minblocks Tensor(fp16) quest metadata with the minimum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 *                       important: zeroes must be in place of metadata of non-existing kv blocks
 * @param [in] metadata_block_tables Tensor(int32) the metadata block tables [B,MMBPR]
 * @param [in] seq_lens Tensor(int32) sequence length of each request in the batch [B]
 * @param [in] k natural number of highest indices to return for every KV head
 * @param [out] selected_indices - Tensor(int32) where the kernel will output the selected 
 *          indices vector. [B,N,k]
 * 
 */
void quest_block_select_paged_in_out(at::Tensor query,
                              at::Tensor maxblocks,
                              at::Tensor minblocks,
                              at::Tensor metadata_block_tables,
                              at::Tensor seq_lens,
                              at::Tensor selected_indices
)
{
    // infer tensor shapes
    int32_t B = query.sizes()[DIM0]; 
    int32_t N = maxblocks.sizes()[DIM2];
    int32_t H = query.sizes()[DIM1];
    int32_t BLOCK_SIZE = maxblocks.sizes()[DIM1];
    int32_t D = query.sizes()[DIM2];
    int32_t MMBPR = metadata_block_tables.sizes()[DIM1];
    int32_t num_meta_blocks = maxblocks.sizes()[DIM0];
    int32_t k = selected_indices.sizes()[DIM2];
    int k_round = DIV_ROUNDUP_MUL(k * BYTES_PER_IDX, BYTES_ASCEND_DATA_BLOCK) / BYTES_PER_IDX;

    // validate input shapes
    TORCH_CHECK(k == k_round, "last dimenstion (", DIM2, ") of selected_indices argument must be a multiple of 8. Given:", k);
    TORCH_CHECK(D == DIM128, "D must be equal to ", DIM128," for high performance operations, got ", D);
    TORCH_CHECK(BLOCK_SIZE == DIM128, "BLOCK_SIZE must be equal to ", DIM128," for high performance operations, got ", BLOCK_SIZE);
    TORCH_CHECK(H % N == 0, "H must be divisible by N (GQA/MHA requirement). H=", H, " N=", N);
    TORCH_CHECK(B == seq_lens.sizes()[DIM0], "Batch size mismatch: query batch=", B, " seq_lens batch=", seq_lens.size(DIM0));
    TORCH_CHECK(B == metadata_block_tables.size(DIM0), "Batch size mismatch: query batch=", B, " metadata_block_tables batch=", metadata_block_tables.size(DIM0));
    TORCH_CHECK(N == minblocks.size(DIM2), "N (num KV heads) mismatch: expected ", N, " from query, got ", minblocks.size(DIM2), " from minblocks");
    TORCH_CHECK(BLOCK_SIZE == minblocks.size(DIM1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, " , got ", minblocks.size(DIM1), " from minblocks");
    TORCH_CHECK(D == maxblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", maxblocks.size(DIM3), " from maxblocks");
    TORCH_CHECK(D == minblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", minblocks.size(DIM3), " from minblocks");
    TORCH_CHECK(num_meta_blocks == minblocks.size(DIM0), "num_meta_blocks mismatch: inferred ", num_meta_blocks, " from maxblocks, got ", minblocks.size(DIM0), " from minblocks");
    TORCH_CHECK(k > 0, "k must be positive, got ", k);
    TORCH_CHECK(MMBPR <= MAXMBPR, "maximum metablocks per request (MMBPR) cannot exceed ", MAXMBPR, " for this kernel. Your MMBPR=", MMBPR, " as inferred from dim=1 of metadata_block_tables argument.");
    TORCH_CHECK(H / N <= BLOCK_SIZE, "H/N head group size cannot exceed BLOCK_SIZE=",BLOCK_SIZE, " given H/N=", H/N);
    TORCH_CHECK(B == selected_indices.size(DIM0), "selected indices 0 dim must have size: ",B, " given: ",selected_indices.size(DIM0));
    TORCH_CHECK(N == selected_indices.size(DIM1), "selected indices 1 dim must have size: ",N, " given: ",selected_indices.size(DIM1));

    // validate data types
    bool use_bfloat16 = is_bfloat16(query);
    TORCH_CHECK(use_bfloat16 == is_bfloat16(maxblocks), "query, maxblocks, minblocks input tensors must have the same data type");
    TORCH_CHECK(use_bfloat16 == is_bfloat16(minblocks), "query, maxblocks, minblocks input tensors must have the same data type");

    // allocate input tensors
    uint8_t *query_ptr = reinterpret_cast<uint8_t *>(query.storage().data_ptr().get());
    uint8_t *maxblocks_ptr = reinterpret_cast<uint8_t *>(maxblocks.storage().data_ptr().get());
    uint8_t *minblocks_ptr = reinterpret_cast<uint8_t *>(minblocks.storage().data_ptr().get());
    uint8_t *metadata_block_tables_ptr = reinterpret_cast<uint8_t *>(metadata_block_tables.storage().data_ptr().get());
    uint8_t *seq_lens_ptr = reinterpret_cast<uint8_t *>(seq_lens.storage().data_ptr().get());
    uint8_t *selected_indices_ptr = reinterpret_cast<uint8_t *>(selected_indices.storage().data_ptr().get());

    // set up launch parameters
    uint32_t blockDims = (B * N > NUM_CORES) ? NUM_CORES : B * N;
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream();

    // launch the kernel
    launch_quest_block_select_paged(blockDims, nullptr, aclStream, query_ptr, maxblocks_ptr, minblocks_ptr,
        metadata_block_tables_ptr, seq_lens_ptr, selected_indices_ptr, B, N, H, BLOCK_SIZE, D, MMBPR, num_meta_blocks,
        -1, // disable tokens_since_metadata_update feature
        k_round, use_bfloat16
    );
}

/**
 * @brief Alternative Interface the `quest_block_select_paged` kernel which 
 *        predicts the sparsity mask during decoding in the form of top-k 
 *        important kv-block indices for every KV-head in every request. The 
 *        returned KV block ids are not the indices in the KV-cache, but rather 
 *        from their enumeration from 0 to number of blocks in the sequence 
 *        length being decoded.
 *
 *        FEATURE 1) WITH PREALLOCATED OUTPUT TENSOR (selected_indices)
 *
 *        FEATURE 2) "w" 2 stands for "window" i.e. the kernel decides whether 
 *        to add local window blocks ids to the selected indices based on the 
 *        number of tokens since the last update and based on the sequence 
 *        length.
 *
 * @param [in] query Tensor(fp16) the query vector [B,H,D]
 * @param [in] maxblocks Tensor(fp16) quest metadata with the maximum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 *                       important: zeroes must be in place of metadata of non-existing kv blocks
 * @param [in] minblocks Tensor(fp16) quest metadata with the minimum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 *                       important: zeroes must be in place of metadata of non-existing kv blocks
 * @param [in] metadata_block_tables Tensor(int32) the metadata block tables [B,MMBPR]
 * @param [in] seq_lens Tensor(int32) sequence length of each request in the batch [B]
 * @param [in] tokens_since_metadata_update
 * @param [out] selected_indices - Tensor(int32) where the kernel will output the 
 *                                 selected indices vector. [B,N,k]
 * 
 */
void quest_block_select_paged_in_out_w(at::Tensor query,
                              at::Tensor maxblocks,
                              at::Tensor minblocks,
                              at::Tensor metadata_block_tables,
                              at::Tensor seq_lens,
                              int tokens_since_metadata_update,
                              at::Tensor selected_indices
)
{
    // infer tensor shapes
    int32_t N = maxblocks.sizes()[DIM2];
    int32_t B = query.sizes()[DIM0]; 
    int32_t BLOCK_SIZE = maxblocks.sizes()[DIM1];
    int32_t H = query.sizes()[DIM1];
    int32_t MMBPR = metadata_block_tables.sizes()[DIM1];
    int32_t D = query.sizes()[DIM2];
    int32_t k = selected_indices.sizes()[DIM2];
    int32_t num_meta_blocks = maxblocks.sizes()[DIM0];
    int k_round = DIV_ROUNDUP_MUL(k * BYTES_PER_IDX, BYTES_ASCEND_DATA_BLOCK) / BYTES_PER_IDX;

    // validate input shapes
    TORCH_CHECK(D == DIM128, "D must be equal to ", DIM128," for high performance operations, got ", D);
    TORCH_CHECK(k == k_round, "last dimenstion (", DIM2, ") of selected_indices argument must be a multiple of 8. Given:", k);
    TORCH_CHECK(BLOCK_SIZE == DIM128, "BLOCK_SIZE must be equal to ", DIM128," for high performance operations, got ", BLOCK_SIZE);
    TORCH_CHECK(B == seq_lens.sizes()[DIM0], "Batch size mismatch: query batch=", B, " seq_lens batch=", seq_lens.size(DIM0));
    TORCH_CHECK(H % N == 0, "H must be divisible by N (GQA/MHA requirement). H=", H, " N=", N);
    TORCH_CHECK(B == metadata_block_tables.size(DIM0), "Batch size mismatch: query batch=", B, " metadata_block_tables batch=", metadata_block_tables.size(DIM0));
    TORCH_CHECK(BLOCK_SIZE == minblocks.size(DIM1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, " , got ", minblocks.size(DIM1), " from minblocks");
    TORCH_CHECK(N == minblocks.size(DIM2), "N (num KV heads) mismatch: expected ", N, " from query, got ", minblocks.size(DIM2), " from minblocks");
    TORCH_CHECK(D == maxblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", maxblocks.size(DIM3), " from maxblocks");
    TORCH_CHECK(num_meta_blocks == minblocks.size(DIM0), "num_meta_blocks mismatch: inferred ", num_meta_blocks, " from maxblocks, got ", minblocks.size(DIM0), " from minblocks");
    TORCH_CHECK(D == minblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", minblocks.size(DIM3), " from minblocks");
    TORCH_CHECK(k > 0, "k must be positive, got ", k);
    TORCH_CHECK(MMBPR <= MAXMBPR, "maximum metablocks per request (MMBPR) cannot exceed ", MAXMBPR, " for this kernel. Your MMBPR=", MMBPR, " as inferred from dim=1 of metadata_block_tables argument.");
    TORCH_CHECK(H / N <= BLOCK_SIZE, "H/N head group size cannot exceed BLOCK_SIZE=",BLOCK_SIZE, " given H/N=", H/N);
    TORCH_CHECK(B == selected_indices.size(DIM0), "selected indices 0 dim must have size: ",B, " given: ",selected_indices.size(DIM0));
    TORCH_CHECK(N == selected_indices.size(DIM1), "selected indices 1 dim must have size: ",N, " given: ",selected_indices.size(DIM1));
    TORCH_CHECK(tokens_since_metadata_update <= BLOCK_SIZE, "tokens_since_metadata_update (given: " , tokens_since_metadata_update, ") cannot exceed BLOCK_SIZE (given:", BLOCK_SIZE, ")");
    TORCH_CHECK(tokens_since_metadata_update >= 0, "tokens_since_metadata_update (given: " , tokens_since_metadata_update, ") must be non-negative");

    // validate data types
    bool use_bfloat16 = is_bfloat16(query);
    TORCH_CHECK(use_bfloat16 == is_bfloat16(minblocks), "query, maxblocks, minblocks input tensors must have the same data type");
    TORCH_CHECK(use_bfloat16 == is_bfloat16(maxblocks), "query, maxblocks, minblocks input tensors must have the same data type");

    // allocate input tensors
    uint8_t *query_ptr = reinterpret_cast<uint8_t *>(query.storage().data_ptr().get());
    uint8_t *minblocks_ptr = reinterpret_cast<uint8_t *>(minblocks.storage().data_ptr().get());
    uint8_t *maxblocks_ptr = reinterpret_cast<uint8_t *>(maxblocks.storage().data_ptr().get());
    uint8_t *seq_lens_ptr = reinterpret_cast<uint8_t *>(seq_lens.storage().data_ptr().get());
    uint8_t *metadata_block_tables_ptr = reinterpret_cast<uint8_t *>(metadata_block_tables.storage().data_ptr().get());
    uint8_t *selected_indices_ptr = reinterpret_cast<uint8_t *>(selected_indices.storage().data_ptr().get());

    // set up launch parameters
    uint32_t blockDims = (B * N > NUM_CORES) ? NUM_CORES : B * N;
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream();

    // launch the kernel
    launch_quest_block_select_paged(
        blockDims, nullptr, aclStream,
        query_ptr, maxblocks_ptr, minblocks_ptr, metadata_block_tables_ptr, seq_lens_ptr,
        selected_indices_ptr, B, N, H, BLOCK_SIZE, D, MMBPR, num_meta_blocks, tokens_since_metadata_update,
        k_round, use_bfloat16
    );
}

/**
 * Create the binding between this CPP function and python. Expose the function towards python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("quest_prefill_metadata", &quest_prefill_metadata,
        R"DOC(
        Interface to the `quest_prefill_metadata` kernel, which initializes
        quest predictor metadata after LLM prefilling of the KV cache. Computes 
        the minimum and maximum metadata vectors for each block in the key 
        cache, enabling efficient sparse attention during the decoding phase.

        Args:
            k_cache (torch.Tensor): Key cache of shape 
                                   [num_kv_blocks, BLOCK_SIZE, N, D] (fp16)
            block_tables (torch.Tensor): KV block tables of shape [B, MKBPR] (int32)
            seq_lens (torch.Tensor): Sequence length of each request in the batch
                                   of shape [B] (int32)
            metadata_block_tables (torch.Tensor): Metadata block tables of 
                                                shape [B, MMBPR] (int32)
            maxblocks (torch.Tensor): Output tensor for maxblock metadata of shape
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16)
            minblocks (torch.Tensor): Output tensor for minblock metadata of shape
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16)

        Note:
            num_kv_blocks can be equal to num_meta_blocks, and one may even pass
            maxblocks = k_cache, minblocks = v_cache to reuse the same page tables
            of vLLM for memory efficiency.
        )DOC",
        py::arg("k_cache"),
        py::arg("block_tables"),
        py::arg("seq_lens"),
        py::arg("metadata_block_tables"),
        py::arg("maxblocks"),
        py::arg("minblocks")
    );
        
    m.def("quest_block_select_paged", &quest_block_select_paged,
        R"DOC(
        Interface to the `quest_block_select_paged` kernel which predicts the
        sparsity mask during decoding in the form of top-k important kv-block 
        indices for every KV-head in every request. The returned KV block ids 
        are not the indices in the KV-cache, but rather from their enumeration 
        from 0 to number of blocks in the sequence length being decoded.

        Args:
            query (torch.Tensor): Query vector of shape [B, H, D] (fp16 or bf16)
            maxblocks (torch.Tensor): Quest metadata with maximum vectors of 
                                    every key-cache block of shape [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                    important: zeroes must be in place of metadata of non-existing kv blocks
            minblocks (torch.Tensor): Quest metadata with minimum vectors of 
                                    every key-cache block of shape  [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                    important: zeroes must be in place of metadata of non-existing kv blocks
            metadata_block_tables (torch.Tensor): Metadata block tables of shape [B, MMBPR] (int32)
            seq_lens (torch.Tensor): Sequence length of each request in the batch
                                   of shape [B] (int32)
            k (int): Number of highest indices to return for every KV head

        Returns:
            torch.Tensor: Selected indices vector of shape [B, N, k] (int32)

        Limitations: due to kernel's internal buffer design on 910B:
            D = 128
            BLOCK_SIZE = 128
            H / N <= BLOCK_SIZE
            MMBPR < 7 (below 5 is the most stable)
        )DOC",
        py::arg("query"),
        py::arg("maxblocks"),
        py::arg("minblocks"),
        py::arg("metadata_block_tables"),
        py::arg("seq_lens"),
        py::arg("k")
    );

    m.def("quest_block_select_paged_in_out", &quest_block_select_paged_in_out,
        R"DOC(
        Alternative interface to the `quest_block_select_paged`, extended
        with the following features:

        FEATURE 1) WITH PREALLOCATED OUTPUT TENSOR (selected_indices)

        Args:
            query (torch.Tensor): Query vector of shape [B, H, D] (fp16 or bf16)
            maxblocks (torch.Tensor): Quest metadata with maximum vectors of 
                                    every key-cache block of shape 
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                    important: zeroes must be in place of metadata of non-existing kv blocks
            minblocks (torch.Tensor): Quest metadata with minimum vectors of 
                                    every key-cache block of shape 
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                    important: zeroes must be in place of metadata of non-existing kv blocks
            metadata_block_tables (torch.Tensor): Metadata block tables of 
                                                shape [B, MMBPR] (int32)
            seq_lens (torch.Tensor): Sequence length of each request in the batch
                                   of shape [B] (int32)
            selected_indices (torch.Tensor): Selected indices vector of shape [B, N, k] (int32): 
                                    Number of highest indices to return for every KV head

        Returns:
            <fills out the selected_indices tensor>
            

        Limitations: due to kernel's internal buffer design on 910B:
            D = 128
            BLOCK_SIZE = 128
            H / N <= BLOCK_SIZE
            MMBPR < 7 (below 5 is the most stable)
            k % 8 == 0
        )DOC",
        py::arg("query"),
        py::arg("maxblocks"),
        py::arg("minblocks"),
        py::arg("metadata_block_tables"),
        py::arg("seq_lens"),
        py::arg("selected_indices")
    );

    m.def("quest_block_select_paged_in_out_w", &quest_block_select_paged_in_out_w,
            R"DOC(
            Alternative interface to the `quest_block_select_paged`, extended
            with the following features:

            FEATURE 1) WITH PREALLOCATED OUTPUT TENSOR (selected_indices)
            
            FEATURE 2) "w" 2 stands for "window" i.e. the kernel decides whether to add local 
            window blocks ids to the selected indices based on the number of tokens 
            since the last update and based on th esequence length

            Args:
                query (torch.Tensor): Query vector of shape [B, H, D] (fp16 or bf16)
                maxblocks (torch.Tensor): Quest metadata with maximum vectors of 
                                        every key-cache block of shape 
                                        [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                        important: zeroes must be in place of metadata of 
                                        non-existing kv blocks
                minblocks (torch.Tensor): Quest metadata with minimum vectors of 
                                        every key-cache block of shape 
                                        [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                        important: zeroes must be in place of metadata of 
                                        non-existing kv blocks
                metadata_block_tables (torch.Tensor): Metadata block tables of 
                                                    shape [B, MMBPR] (int32)
                seq_lens (torch.Tensor): Sequence length of each request in the batch
                                    of shape [B] (int32)
                tokens_since_metadata_update (int) - number of tokens that were decoeded since the 
                                    last metadata update (note metadata update is done only on the 
                                    multiple of BLOCK_SIZE tokens whcih is lower or equal to the 
                                    sequence length at the moment of update) set to -1 to disable 
                                    selection of KV blocks for which the metadata doesn't exist.
                selected_indices (torch.Tensor): Selected indices vector of shape [B, N, k] (int32): 
                                        Number of highest indices to return for every KV head

            Returns:
                <fills out the selected_indices tensor>
                
            Limitations: 
                D = 128
                H / N <= BLOCK_SIZE
                BLOCK_SIZE = 128
                MMBPR < 7 (below 5 is the most stable)
                k % 8 == 0
            )DOC",
            py::arg("query"), py::arg("maxblocks"), py::arg("minblocks"),
            py::arg("metadata_block_tables"),
            py::arg("seq_lens"),
            py::arg("tokens_since_metadata_update"),
            py::arg("selected_indices")
        );    
}
