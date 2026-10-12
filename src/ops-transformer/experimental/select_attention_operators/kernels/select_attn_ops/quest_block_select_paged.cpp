/**
* -*- coding: utf-8 -*-
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

 
#include "kernel_operator.h"

#define BYTES_UB_BLOCK 32
#define BYTES_DATA_BLOCK 32
#define NUM_HALF_ELEMS_PER_VECTOR 128
#define NUM_FLOAT_ELEMS_PER_VECTOR 64
#define DIV_ROUNDUP(x,y) (((x)+(y)-1) / (y))
#define DIV_ROUNDUP_MUL(bytes,bytes_per_block) (DIV_ROUNDUP(bytes,bytes_per_block) * (bytes_per_block))
#define NUM_UB_BYTES(bytes) (DIV_ROUNDUP_MUL(bytes,BYTES_UB_BLOCK))
#define NUM_DATA_BLOCKS(bytes) (DIV_ROUNDUP(bytes,BYTES_DATA_BLOCK))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MINHALF -65504.0f
#define MAXHALF 65504.0f
#define MINFLOAT -3.4028235e38f
#define MAXFLOAT 3.4028235e+38f

constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_V200 = 8;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_HALF_V220 = 4;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 = 2;

/**
 * @brief bfloat16 Kernel implementation for Quest KV block selection with 
 * paged metadata in AscendC
 *
 * @param [in] query Pointer to the query vector [B,H,D]
 * @param [in] maxblocks Pointer to the quest metadata with the maximum vectors 
 *             of every K block [num_meta_blocks,BLOCK_SIZE,N,D]
 * @param [in] minblocks Pointer to the quest metadata with the minimum vectors 
 *             of every K block [num_meta_blocks,BLOCK_SIZE,N,D]
 * @param [in] metadata_block_tables Pointer to metadata block tables [B,MMBPR]
 * @param [in] seq_lens Pointer to sequence lengths [B]
 * @param [out] selected_indices Pointer to output indices vector [B,N,k] 
 * @param [in] B Batch size
 * @param [in] N Number of KV heads
 * @param [in] H Number of attention heads (query heads)
 * @param [in] BLOCK_SIZE Number of vectors in maxblock and in minblock
 * @param [in] D head dimension
 * @param [in] num_meta_blocks Total number of metadata blocks
 * @param [in] k number of top indices to return for every KV head
 * @param [in] tokens_since_metadata_update - number of most recent tokens in 
 *             the sequence, since the last update of the metadata (note the 
 *             update is assumed to be computed only on the full multiple of 
 *             BLOCK_SIZE which is below or equal to 
 *            "seq_len - tokens_since_metadata_update")
 *             set to -1 to disable the selection of sink and window kv blocks by this kernel.
 * @param [in] MMBPR Maximum number of metadata blocks per request
 */
extern "C" __global__ __aicore__ void quest_block_select_paged_bfloat16(
     GM_ADDR query, GM_ADDR maxblocks, GM_ADDR minblocks, GM_ADDR metadata_block_tables,
     GM_ADDR seq_lens, GM_ADDR selected_indices,
     int32_t B, int32_t N, int32_t H, int32_t BLOCK_SIZE, int32_t D, int32_t MMBPR,
     int32_t num_meta_blocks, int32_t tokens_since_metadata_update, int32_t k)

{
    AscendC::SetAtomicNone();
    
    // Tiling strategy - each core processes one batch*N combination
    int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t num_batch_heads = B * N;
    int32_t num_batch_heads_per_block = DIV_ROUNDUP(num_batch_heads, num_blocks);
    int32_t G = H / N; // query_heads_per_kv_head

    // Init the GM pointers
    AscendC::GlobalTensor<bfloat16_t> query_gm;
    AscendC::GlobalTensor<bfloat16_t> maxblocks_gm;
    AscendC::GlobalTensor<bfloat16_t> minblocks_gm;
    AscendC::GlobalTensor<int32_t> metadata_block_tables_gm;
    AscendC::GlobalTensor<int32_t> seq_lens_gm;
    AscendC::GlobalTensor<uint32_t> selected_indices_gm;
    
    query_gm.SetGlobalBuffer((__gm__ bfloat16_t *)query);
    maxblocks_gm.SetGlobalBuffer((__gm__ bfloat16_t *)maxblocks);
    minblocks_gm.SetGlobalBuffer((__gm__ bfloat16_t *)minblocks);
    metadata_block_tables_gm.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
    seq_lens_gm.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
    selected_indices_gm.SetGlobalBuffer((__gm__ uint32_t *)selected_indices);    

    // Allocate UB buffers - bflaot16 for incoming data only, float buffers for all the subsequent arithmetic operations
    using VecBuf_t = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    VecBuf_t input_bf16_buf;
    VecBuf_t grouped_query_float_buf, maxblock_float_buf, minblock_float_buf;
    VecBuf_t block_scores_buf, accumulated_scores_buf;
    VecBuf_t selected_indices_buf, selected_values_buf;
    VecBuf_t tmp_concat_buf, concat_buf, index_local_buf, sort_tmp_buf;

    // Calculate buffer sizes - using float for arithmetic buffers
    uint32_t INPUT_BF16_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * D * sizeof(bfloat16_t));
    uint32_t GROUPED_QUERY_FLOAT_SIZE = NUM_UB_BYTES(D * sizeof(float));
    uint32_t BLOCK_FLOAT_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * D * sizeof(float));
    uint32_t REDUCED_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * sizeof(float)); // Use float for scores
    uint32_t ACCUMULATED_SCORES_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * sizeof(float)); // Use float for scores
    uint32_t SELECTED_INDICES_BUF_SIZE = NUM_UB_BYTES(k * sizeof(uint32_t));
    uint32_t SELECTED_VALUES_BUF_SIZE = NUM_UB_BYTES(k * sizeof(float));
    uint32_t TMP_CONCAT_BUF_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 * sizeof(float));
    uint32_t CONCAT_BUF_SIZE = NUM_UB_BYTES((MMBPR * BLOCK_SIZE + MMBPR * BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220) * sizeof(float));
    uint32_t INDEX_LOCAL_BUF_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * sizeof(uint32_t));
    uint32_t SORT_TMP_BUF_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 * sizeof(float));

    // Initialize buffers = 161KB + MMBPR * 3.25KB (910B limit: 192KB)
    AscendC::TPipe pipe;
    pipe.InitBuffer(input_bf16_buf, INPUT_BF16_BUF_SIZE);               // 32KB
    pipe.InitBuffer(grouped_query_float_buf, GROUPED_QUERY_FLOAT_SIZE); // 128 * 4 = 0.5KB
    pipe.InitBuffer(maxblock_float_buf, BLOCK_FLOAT_BUF_SIZE);          // 64KB
    pipe.InitBuffer(minblock_float_buf, BLOCK_FLOAT_BUF_SIZE);          // 64KB
    pipe.InitBuffer(block_scores_buf, REDUCED_BUF_SIZE);                // 128 * 4 = 0.5KB
    pipe.InitBuffer(accumulated_scores_buf, ACCUMULATED_SCORES_SIZE);   // MMBPR * 128 * 4 = MMBPR * 0.5KB
    pipe.InitBuffer(selected_indices_buf, SELECTED_INDICES_BUF_SIZE);   // k * 4 = 32B minor
    pipe.InitBuffer(selected_values_buf, SELECTED_VALUES_BUF_SIZE);     // k * 4 = 32B minor
    pipe.InitBuffer(tmp_concat_buf, TMP_CONCAT_BUF_SIZE);               // MMBPR * 128 * 4 = MMBPR * 0.5KB
    pipe.InitBuffer(concat_buf, CONCAT_BUF_SIZE);                       // MMBPR * 128 * 3 * 4 = MMBPR * 1.5KB
    pipe.InitBuffer(index_local_buf, INDEX_LOCAL_BUF_SIZE);             // MMBPR * 128 * 4 = MMBPR * 0.5KB
    pipe.InitBuffer(sort_tmp_buf, SORT_TMP_BUF_SIZE);                   // MMBPR * 128 * 2 * 4 = MMBPR * 0.75KB

    // Get local tensors from the buffers
    AscendC::LocalTensor<bfloat16_t> input_bf16_lt = input_bf16_buf.Get<bfloat16_t>(BLOCK_SIZE * D);
    AscendC::LocalTensor<float> grouped_query_float_lt = grouped_query_float_buf.Get<float>();
    AscendC::LocalTensor<float> maxblock_float_lt = maxblock_float_buf.Get<float>();
    AscendC::LocalTensor<float> minblock_float_lt = minblock_float_buf.Get<float>();
    AscendC::LocalTensor<float> block_scores_lt = block_scores_buf.Get<float>();
    AscendC::LocalTensor<float> accumulated_scores_lt = accumulated_scores_buf.Get<float>();
    AscendC::LocalTensor<uint32_t> selected_indices_lt = selected_indices_buf.Get<uint32_t>();
    AscendC::LocalTensor<float> selected_values_lt = selected_values_buf.Get<float>();
    AscendC::LocalTensor<float> tmp_concat_lt = tmp_concat_buf.Get<float>();  
    AscendC::LocalTensor<float> concat_lt = concat_buf.Get<float>();  
    AscendC::LocalTensor<uint32_t> index_local_lt = index_local_buf.Get<uint32_t>();
    AscendC::LocalTensor<float> sort_tmp_lt = sort_tmp_buf.Get<float>();

    for (int32_t batch_head_idx = AscendC::GetBlockIdx(); batch_head_idx < num_batch_heads; batch_head_idx += num_blocks) {

        int32_t head_idx = batch_head_idx % N;  // kv-head aka kv-group index within the batch index
        int32_t query_head_start_idx = head_idx * G; // index of the first query-head of the current kv-head
        int32_t batch_idx = batch_head_idx / N; 

        // Calculate GM offsets specific to the current batch_head_idx
        int32_t query_offset = batch_idx * H * D + query_head_start_idx * D;
        int32_t output_offset = batch_head_idx * k;

        // Step 1: Load sequence length and metadata block table for this request
        int32_t seq_len = seq_lens_gm.GetValue(batch_idx);
        int32_t num_tokens_per_meta_block = BLOCK_SIZE * BLOCK_SIZE;
        int32_t num_meta_blocks_in_request = DIV_ROUNDUP(seq_len, num_tokens_per_meta_block);

        // Step 2: Reduce queries across H dimension to get grouped_query [B,N,D]
        uint16_t query_copy_block_len = NUM_DATA_BLOCKS(G * D * sizeof(bfloat16_t));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);
        AscendC::DataCopy(input_bf16_lt, query_gm[query_offset], query_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        
        // Cast bfloat16 query to float for arithmetic operations (the G float query heads will be written temporarily to maxblock_float_lt)
        AscendC::Cast<float, bfloat16_t>(maxblock_float_lt, input_bf16_lt, AscendC::RoundMode::CAST_NONE, G * D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>(); // important sync
        
        uint64_t mask = NUM_FLOAT_ELEMS_PER_VECTOR;
        uint64_t masks_per_D = D / mask;
        // Copy first query head to grouped_query_float
        AscendC::Copy(grouped_query_float_lt, maxblock_float_lt, mask, masks_per_D, { 1, 1, 8, 8 });
        AscendC::PipeBarrier<PIPE_V>();  // important sync
        
        // Add remaining query heads (if any)
        for (int32_t qhead = 1; qhead < G; qhead++) {
            AscendC::Add(grouped_query_float_lt, grouped_query_float_lt, maxblock_float_lt[qhead * D], D);
            AscendC::PipeBarrier<PIPE_V>();  // important sync
        }
        
        // Scale the grouped query
        float scale = 1.0f / static_cast<float>(G);
        AscendC::Muls(grouped_query_float_lt, grouped_query_float_lt, scale, D);

        // Step 3: Initialize accumulated scores buffer with float values
        AscendC::Duplicate(accumulated_scores_lt, (float)MINFLOAT, MMBPR * num_meta_blocks_in_request);

        // Step 4: Process each metadata block for this request-head pair
        for (int32_t meta_blk = 0; meta_blk < num_meta_blocks_in_request; meta_blk++) {
            int32_t meta_blk_id = metadata_block_tables_gm.GetValue(batch_idx * MMBPR + meta_blk);
            
            // Calculate offset for this metadata block
            int32_t meta_block_offset = /*to reach the meta_block start: */ meta_blk_id * BLOCK_SIZE * N * D 
                                        /*within the meta_block, skip the previous kv_heads: */ + head_idx * D;

            // Load maxblock and minblock for this metadata block
            AscendC::DataCopyParams gm_ub_cp;
            gm_ub_cp.blockCount = BLOCK_SIZE;
            gm_ub_cp.blockLen   = DIV_ROUNDUP(D * sizeof(bfloat16_t), BYTES_DATA_BLOCK);
            gm_ub_cp.srcStride  = DIV_ROUNDUP((N - 1) * D * sizeof(bfloat16_t), BYTES_DATA_BLOCK); // skip other heads
            gm_ub_cp.dstStride  = 0;
                        
            // Maxblock - Load, and cast to float, multiply by query
            AscendC::DataCopy(input_bf16_lt, maxblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast<float, bfloat16_t>(maxblock_float_lt, input_bf16_lt, AscendC::RoundMode::CAST_NONE, BLOCK_SIZE * D);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::BinaryRepeatParams mul_repeat_params = AscendC::BinaryRepeatParams(1, 1, 1, 8*masks_per_D, 0, 8*masks_per_D); // Adjusted for float (64 elements per vector - need to break into even+odd multiply sequences)
            AscendC::Mul(maxblock_float_lt, grouped_query_float_lt, maxblock_float_lt, NUM_FLOAT_ELEMS_PER_VECTOR, BLOCK_SIZE, mul_repeat_params); // multiply the even repeats
            AscendC::Mul(maxblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR], grouped_query_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR], maxblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR], NUM_FLOAT_ELEMS_PER_VECTOR, BLOCK_SIZE, mul_repeat_params); // multiply the odd repeats

            // Minblock - Load, and cast to float, multiply by query
            AscendC::DataCopy(input_bf16_lt, minblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::Cast<float, bfloat16_t>(minblock_float_lt, input_bf16_lt, AscendC::RoundMode::CAST_NONE, BLOCK_SIZE * D);
            AscendC::Mul(minblock_float_lt, grouped_query_float_lt, minblock_float_lt, NUM_FLOAT_ELEMS_PER_VECTOR, BLOCK_SIZE, mul_repeat_params);
            AscendC::Mul(minblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR], grouped_query_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR], minblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR], NUM_FLOAT_ELEMS_PER_VECTOR, BLOCK_SIZE, mul_repeat_params);
            
            // Find element-wise maximum
            AscendC::Max(maxblock_float_lt, maxblock_float_lt, minblock_float_lt, BLOCK_SIZE * D);
            
            // Reduce sum across D=128 dimension for each block
            // floats can be reduced using RepeatSumReduce - up to 64 floats per repeat.
            // first step: each 64 numbers are summed to 1, next 64 are sumed to 2nd, ... --> output shape: 2*BLOCK_SIZE partial sums
            // - note (i): reusing minblock as a buffer
            // - note (ii) can do no more than 255 repeats but we need 256, hence breaking into 2 calls to RepeatReduceSum (each with 128 repeats)
            AscendC::RepeatReduceSum(minblock_float_lt, maxblock_float_lt, BLOCK_SIZE, mask, 0, 1, 1, 8);
            AscendC::RepeatReduceSum(minblock_float_lt[BLOCK_SIZE], maxblock_float_lt[BLOCK_SIZE*D/masks_per_D], BLOCK_SIZE, mask, 0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();  // important synch
            // second step: every 2 numbers are reduced to 1 input: masks_per_D*BLOCK_SIZE float numbers --> output: BLOCK_SIZE float numbers
            AscendC::PairReduceSum(block_scores_lt, minblock_float_lt, masks_per_D * BLOCK_SIZE / mask, mask, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();  // important synch
            
            // Store scores in accumulated buffer with offset for this metadata block 
            uint64_t seq_len_curr_meta_blk = MIN(seq_len - (meta_blk * num_tokens_per_meta_block), BLOCK_SIZE); // (account for tail, which might not be exactly BLOCK_SIZE)
            for (int32_t sub_meta_block_id = 0; sub_meta_block_id < masks_per_D; sub_meta_block_id++) {
                int32_t block_scores_offset = sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR;
                int32_t accumulated_offset = meta_blk * BLOCK_SIZE + block_scores_offset;
                AscendC::Copy(accumulated_scores_lt[accumulated_offset], block_scores_lt[block_scores_offset], 
                              seq_len_curr_meta_blk - sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR, 1, {1,1,8,8});
                if (meta_blk < num_meta_blocks_in_request - 1){
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);     
                } 
            }
        }

        // Find top-k indices across all metadata blocks for this request-head pair
        uint32_t total_elements = num_meta_blocks_in_request * BLOCK_SIZE;
        uint32_t m_concatRepeatTimes = DIV_ROUNDUP(total_elements, 32); // For float operations
        uint32_t m_sortRepeatTimes = DIV_ROUNDUP(total_elements, 32);
        uint32_t m_extractRepeatTimes = DIV_ROUNDUP(total_elements, 32);

        // Initialize index range [0...total_elements-1]
        for (uint32_t i = 0; i < total_elements; i++) {
            index_local_lt.SetValue(i, i);
        }   

        // (add sink) Add high score to the first block, making sure that it will be selected
        if (tokens_since_metadata_update >= 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);                 
            accumulated_scores_lt.SetValue(0, MAXFLOAT);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);        
        }
        
        // Sort accumulated scores of the entire request
        AscendC::Concat(concat_lt, accumulated_scores_lt, tmp_concat_lt, m_concatRepeatTimes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort<float, true>(maxblock_float_lt, concat_lt, index_local_lt, sort_tmp_lt, m_sortRepeatTimes);  // Using float sort
       
        // Extract top-k indices - need to convert back to bfloat16 for output if needed
        AscendC::Extract(selected_values_lt, selected_indices_lt, maxblock_float_lt, m_extractRepeatTimes);

        // (local window at the end) Add check whether last index should be added
        if (tokens_since_metadata_update >= 0) {
            int32_t mru = seq_len - tokens_since_metadata_update;  // mru = sequence length of this request at the most recent metadata update
            int32_t win_size = ((mru & 0x7f) != 0) + (seq_len >> 7) - (mru >> 7); // win_size - faster computation version due to statically known fact that BLOCK_SZIE is a powers of two --> 7
            // int32_t win_size = (mru % BLOCK_SIZE != 0) + (seq_len / BLOCK_SIZE) - (mru / BLOCK_SIZE); // win_size = number of the most recent KV-blocks in the sequence, which are not yet registered by the  metadata
            for (int w=1; w<=win_size; w++){selected_indices_lt.SetValue(k - w, DIV_ROUNDUP(seq_len, BLOCK_SIZE) - w);}
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);            
        }

        // Step 6: Copy out the results
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        uint16_t indices_copy_block_len = NUM_DATA_BLOCKS(k * sizeof(int32_t));
        auto indices_copy_params = AscendC::DataCopyParams(1, indices_copy_block_len, 0, 0);
        AscendC::DataCopy(selected_indices_gm[output_offset], selected_indices_lt, indices_copy_params);
    }
}

/**
 * @brief half-precison (float16) Kernel implementation for Quest KV block 
 * selection with paged metadata in AscendC
 *
 * @param [in] query Pointer to the query vector [B,H,D]
 * @param [in] maxblocks Pointer to the quest metadata with the maximum vectors 
 *             of every K block [num_meta_blocks,BLOCK_SIZE,N,D]
 * @param [in] minblocks Pointer to the quest metadata with the minimum vectors 
 *             of every K block [num_meta_blocks,BLOCK_SIZE,N,D]
 * @param [in] metadata_block_tables Pointer to metadata block tables [B,MMBPR]
 * @param [in] seq_lens Pointer to sequence lengths [B]
 * @param [out] selected_indices Pointer to output indices vector [B,N,k] 
 * @param [in] B Batch size
 * @param [in] N Number of KV heads
 * @param [in] H Number of attention heads (query heads)
 * @param [in] BLOCK_SIZE Number of vectors in maxblock and in minblock
 * @param [in] D head dimension
 * @param [in] num_meta_blocks Total number of metadata blocks
 * @param [in] k number of top indices to return for every KV head
 * @param [in] tokens_since_metadata_update - number of most recent tokens in 
 *             the sequence, since the last update of the metadata (note the 
 *             update is assumed to be computed only on the full multiple of 
 *             BLOCK_SIZE which is below or equal to 
 *            "seq_len - tokens_since_metadata_update")
 *             set to -1 to disable the selection of sink and window kv blocks by this kernel.
 */
extern "C" __global__ __aicore__ void quest_block_select_paged_half(
    GM_ADDR query,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR metadata_block_tables,
    GM_ADDR seq_lens,
    GM_ADDR selected_indices,
    int32_t B,
    int32_t N,
    int32_t H,
    int32_t BLOCK_SIZE,
    int32_t D,
    int32_t MMBPR,
    int32_t num_meta_blocks,
    int32_t tokens_since_metadata_update,
    int32_t k)

{
    AscendC::SetAtomicNone();
    
    // Tile across AI cores - each core processes one batch*N combination
    int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t num_batch_heads = B * N;
    int32_t num_batch_heads_per_block = DIV_ROUNDUP(num_batch_heads, num_blocks);
    int32_t G = H / N; // query_heads_per_kv_head

    // Init the GM pointers
    AscendC::GlobalTensor<half> query_gm;
    AscendC::GlobalTensor<half> maxblocks_gm;
    AscendC::GlobalTensor<half> minblocks_gm;
    AscendC::GlobalTensor<int32_t> metadata_block_tables_gm;
    AscendC::GlobalTensor<int32_t> seq_lens_gm;
    AscendC::GlobalTensor<uint32_t> selected_indices_gm;
    
    query_gm.SetGlobalBuffer((__gm__ half *)query);
    maxblocks_gm.SetGlobalBuffer((__gm__ half *)maxblocks);
    minblocks_gm.SetGlobalBuffer((__gm__ half *)minblocks);
    metadata_block_tables_gm.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
    seq_lens_gm.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
    selected_indices_gm.SetGlobalBuffer((__gm__ uint32_t *)selected_indices);    

    // Allocate UB buffers
    using VecBuf_t = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    VecBuf_t query_buf, grouped_query_buf, maxblock_buf, minblock_buf;
    VecBuf_t block_scores_buf, accumulated_scores_buf;
    VecBuf_t selected_indices_buf, selected_values_buf;
    VecBuf_t tmp_concat_buf, concat_buf, index_local_buf, sort_tmp_buf;

    // Calculate buffer sizes
    uint32_t QUERY_BUF_SIZE = NUM_UB_BYTES(G * D * sizeof(half));
    uint32_t GROUPED_QUERY_SIZE = NUM_UB_BYTES(D * sizeof(half));
    uint32_t BLOCK_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * D * sizeof(half));
    uint32_t REDUCED_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * sizeof(half));
    uint32_t ACCUMULATED_SCORES_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * sizeof(half));
    uint32_t SELECTED_INDICES_BUF_SIZE = NUM_UB_BYTES(k * sizeof(uint32_t));
    uint32_t SELECTED_VALUES_BUF_SIZE = NUM_UB_BYTES(k * sizeof(half));
    uint32_t TMP_CONCAT_BUF_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_V200 * sizeof(half));
    uint32_t CONCAT_BUF_SIZE = NUM_UB_BYTES((MMBPR * BLOCK_SIZE + MMBPR * BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_V200) * sizeof(half));
    uint32_t INDEX_LOCAL_BUF_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * sizeof(uint32_t));
    uint32_t SORT_TMP_BUF_SIZE = NUM_UB_BYTES(MMBPR * BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_HALF_V220 * sizeof(half));

    // Initialize buffers
    AscendC::TPipe pipe;
    pipe.InitBuffer(query_buf, QUERY_BUF_SIZE);
    pipe.InitBuffer(grouped_query_buf, GROUPED_QUERY_SIZE);
    pipe.InitBuffer(maxblock_buf, BLOCK_BUF_SIZE);
    pipe.InitBuffer(minblock_buf, BLOCK_BUF_SIZE);
    pipe.InitBuffer(block_scores_buf, REDUCED_BUF_SIZE);
    pipe.InitBuffer(accumulated_scores_buf, ACCUMULATED_SCORES_SIZE);
    pipe.InitBuffer(selected_indices_buf, SELECTED_INDICES_BUF_SIZE);
    pipe.InitBuffer(selected_values_buf, SELECTED_VALUES_BUF_SIZE);
    pipe.InitBuffer(tmp_concat_buf, TMP_CONCAT_BUF_SIZE);  
    pipe.InitBuffer(concat_buf, CONCAT_BUF_SIZE);  
    pipe.InitBuffer(index_local_buf, INDEX_LOCAL_BUF_SIZE);
    pipe.InitBuffer(sort_tmp_buf, SORT_TMP_BUF_SIZE);

    AscendC::LocalTensor<half> query_lt = query_buf.Get<half>();
    AscendC::LocalTensor<half> grouped_query_lt = grouped_query_buf.Get<half>();
    AscendC::LocalTensor<half> maxblock_lt = maxblock_buf.Get<half>();
    AscendC::LocalTensor<half> minblock_lt = minblock_buf.Get<half>();
    AscendC::LocalTensor<half> block_scores_lt = block_scores_buf.Get<half>();
    AscendC::LocalTensor<half> accumulated_scores_lt = accumulated_scores_buf.Get<half>();
    AscendC::LocalTensor<uint32_t> selected_indices_lt = selected_indices_buf.Get<uint32_t>();
    AscendC::LocalTensor<half> selected_values_lt = selected_values_buf.Get<half>();
    AscendC::LocalTensor<half> tmp_concat_lt = tmp_concat_buf.Get<half>();  
    AscendC::LocalTensor<half> concat_lt = concat_buf.Get<half>();  
    AscendC::LocalTensor<uint32_t> index_local_lt = index_local_buf.Get<uint32_t>();
    AscendC::LocalTensor<half> sort_tmp_lt = sort_tmp_buf.Get<half>();

    for (int32_t batch_head_idx = AscendC::GetBlockIdx(); batch_head_idx < num_batch_heads; batch_head_idx += num_blocks) {

        int32_t batch_idx = batch_head_idx / N; 
        int32_t head_idx = batch_head_idx % N;  // kv-head aka kv-group index within the batch index
        int32_t query_head_start_idx = head_idx * G; // index of the first query-head of the current kv-head

        // Calculate GM offsets specific to the current batch_head_idx
        int32_t query_offset = batch_idx * H * D + query_head_start_idx * D;
        int32_t output_offset = batch_head_idx * k;

        // Load sequence length and metadata block table for this request
        int32_t seq_len = seq_lens_gm.GetValue(batch_idx);
        int32_t num_tokens_per_meta_block = BLOCK_SIZE * BLOCK_SIZE;
        int32_t num_meta_blocks_in_request = DIV_ROUNDUP(seq_len, num_tokens_per_meta_block);

        // Reduce queries across H dimension to get grouped_query [B,N,D]
        uint16_t query_copy_block_len = NUM_DATA_BLOCKS(G * D * sizeof(half));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);
        AscendC::DataCopy(query_lt, query_gm[query_offset], query_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        
        uint64_t mask = D;
        AscendC::Copy(grouped_query_lt, query_lt, mask, 1, { 1, 1, 8, 8 });
        AscendC::PipeBarrier<PIPE_V>();
        
        for (int32_t qhead = 1; qhead < G; qhead++) {
            AscendC::Add<half>(grouped_query_lt, grouped_query_lt, query_lt[qhead * D], D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        
        half scale = (half)((float)1.0 / (float)G);
        AscendC::Muls<half>(grouped_query_lt, grouped_query_lt, scale, D);

        // Initialize accumulated scores buffer
        AscendC::Duplicate(accumulated_scores_lt, (half)(MINHALF), MMBPR * num_meta_blocks_in_request);
        AscendC::PipeBarrier<PIPE_V>();

        // Process each metadata block for this request-head pair
        for (int32_t meta_blk = 0; meta_blk < num_meta_blocks_in_request; meta_blk++) {
            int32_t meta_blk_id = metadata_block_tables_gm.GetValue(batch_idx * MMBPR + meta_blk);
            
            // Calculate offset for this metadata block
            int32_t meta_block_offset = /*to reach the meta_block start: */ meta_blk_id * BLOCK_SIZE * N * D 
                                        /*within the meta_block, skip the previous kv_heads: */ + head_idx * D;

            // Load maxblock and minblock for this metadata block
            AscendC::DataCopyParams gm_ub_cp;
            gm_ub_cp.blockCount = BLOCK_SIZE;
            gm_ub_cp.blockLen   = DIV_ROUNDUP(D * sizeof(half), BYTES_DATA_BLOCK);
            gm_ub_cp.srcStride  = DIV_ROUNDUP((N - 1) * D * sizeof(half), BYTES_DATA_BLOCK); // skip other heads
            gm_ub_cp.dstStride  = 0;
                        
            AscendC::DataCopy(maxblock_lt, maxblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::DataCopy(minblock_lt, minblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            
            // Compute block scores for this metadata block
            uint8_t repeatTimes = BLOCK_SIZE;  // acceleration opportunity: possible to accelerate by setting to "seq_len_curr_meta_blk" that is defined below, this will save the unnecessary BLOCK_SIZE-seq_len_curr_meta_blk computations
            AscendC::BinaryRepeatParams mul_repeat_params = {1,1,1,8,0,8};
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Mul(maxblock_lt, grouped_query_lt, maxblock_lt, mask, repeatTimes, mul_repeat_params); // minblock_lt is not actually product query*minblock_lt 
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::Mul(minblock_lt, grouped_query_lt, minblock_lt, mask, repeatTimes, mul_repeat_params); // maxblock_lt is not actually product query*maxblock_lt
            AscendC::Max(maxblock_lt, maxblock_lt, minblock_lt, mask, repeatTimes, {1,1,1,8,8,8}); // reusing maxblock_lt to actually represent "channel_max_product_lt"
            AscendC::RepeatReduceSum<half, true>(block_scores_lt, maxblock_lt, repeatTimes, mask, 0, 1, 1, 8); 
            AscendC::PipeBarrier<PIPE_V>();  // important synch
            
            // Store scores in accumulated buffer with offset for this metadata block 
            int32_t accumulated_offset = meta_blk * BLOCK_SIZE;
            uint64_t seq_len_curr_meta_blk = MIN(seq_len - (meta_blk * num_tokens_per_meta_block), BLOCK_SIZE); // (account for tail, which might not be exactly BLOCK_SIZE)
            AscendC::Copy(accumulated_scores_lt[accumulated_offset], block_scores_lt, 
                          seq_len_curr_meta_blk, 1, {1,1,8,8});
            if (meta_blk < num_meta_blocks_in_request - 1){
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);     
            } 
        }

        // Step 5: Find top-k indices across all metadata blocks for this request-head pair
        uint32_t total_elements = num_meta_blocks_in_request * BLOCK_SIZE;
        uint32_t m_sortRepeatTimes = DIV_ROUNDUP(total_elements, 32);
        uint32_t m_concatRepeatTimes = DIV_ROUNDUP(total_elements, 32);
        uint32_t m_extractRepeatTimes = DIV_ROUNDUP(total_elements, 32);

        // Create indices range [0...total_elements-1]
        for (uint32_t i = 0; i < total_elements; i++) {index_local_lt.SetValue(i, i);}   

        // (sink) Add high score to the first block, making sure that it will be selected
        if (tokens_since_metadata_update >= 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);                 
            accumulated_scores_lt.SetValue(0, MAXHALF);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);        
        }        

        // Sort all accumulated scores
        AscendC::PipeBarrier<PIPE_V>(); // important synch because accumulated_scores_lt must be fully done
        AscendC::Concat(concat_lt, accumulated_scores_lt, tmp_concat_lt, m_concatRepeatTimes);
        AscendC::Sort<half, true>(maxblock_lt, concat_lt, index_local_lt, sort_tmp_lt, m_sortRepeatTimes);  // resuing max_block_lt buffer
        
        // Extract top-k indices
        AscendC::Extract(selected_values_lt, selected_indices_lt, maxblock_lt, m_extractRepeatTimes);

        // (local window) Add check whether last index should be added
        if (tokens_since_metadata_update >= 0) {
            int32_t mru = seq_len - tokens_since_metadata_update;  // mru = sequence length of this request at the most recent metadata update
            // int32_t win_size = (mru % BLOCK_SIZE != 0) + (seq_len / BLOCK_SIZE) - (mru / BLOCK_SIZE); // win_size = number of the most recent KV-blocks in the sequence, which are not yet registered by the  metadata
            int32_t win_size = ((mru & 0x7f) != 0) + (seq_len >> 7) - (mru >> 7); // win_size - faster computation version due to statically known fact that BLOCK_SZIE is a powers of two --> 7
            for (int w=1; w<=win_size; w++){
                selected_indices_lt.SetValue(k - w, DIV_ROUNDUP(seq_len, BLOCK_SIZE) - w);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID3);            
        }        

        // Step 6: Copy out the results
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        uint16_t indices_copy_block_len = NUM_DATA_BLOCKS(k * sizeof(int32_t));
        auto indices_copy_params = AscendC::DataCopyParams(1, indices_copy_block_len, 0, 0);
        AscendC::DataCopy(selected_indices_gm[output_offset], selected_indices_lt, indices_copy_params);
    }
}

/**
 * Kernel launch function
 */
void launch_quest_block_select_paged(
    uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *query, uint8_t *maxblocks,
    uint8_t *minblocks, uint8_t *metadata_block_tables,
    uint8_t *seq_lens, uint8_t *selected_indices,
    int32_t B, int32_t N,
    int32_t H,
    int32_t BLOCK_SIZE,
    int32_t D,
    int32_t MMBPR,
    int32_t num_meta_blocks,    
    int32_t tokens_since_metadata_update,
    int32_t k,
    bool is_bfloat16
)
{
    if (is_bfloat16) {
        quest_block_select_paged_bfloat16<<<blockDim, l2ctrl, stream>>>(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            B, N, H, BLOCK_SIZE, D, MMBPR, num_meta_blocks, tokens_since_metadata_update, k);
        } else {
        quest_block_select_paged_half<<<blockDim, l2ctrl, stream>>>(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            B, N, H, BLOCK_SIZE, D, MMBPR, num_meta_blocks, tokens_since_metadata_update, k);            
        }
}
