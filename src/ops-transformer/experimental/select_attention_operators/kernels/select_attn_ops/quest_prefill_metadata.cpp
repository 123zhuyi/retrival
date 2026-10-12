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

/*******************************************************************************
 *  quest_prefill_metadata_kernel - vector-core, 1 core = (batch, head)
 *  Loads each KV-block ONCE, keeps copy, reduces min & max logarithmically
 *******************************************************************************/
#include "kernel_operator.h"

#define DOUBLEBUFFER 2
#define SINGLEBUFFER 1

constexpr int32_t BYTES_UB_BLOCK = 32;
constexpr int32_t BYTES_DATA_BLOCK = 32;
constexpr int32_t NUM_PER_VECTOR = 128;

inline __aicore__ int32_t ceilDiv(int32_t x, int32_t d) { return (x + d - 1) / d; }
inline __aicore__ int32_t ceilDivMul(int32_t x, int32_t d) { return d * ((x + d - 1) / d); }
inline __aicore__ int32_t round32(int32_t bytes) { return (bytes + 31) & ~31; }

using namespace AscendC;

class KernelQuestMetadata {
public:
    __aicore__ inline KernelQuestMetadata() {}

    __aicore__ void Init(GM_ADDR k_cache,
                         GM_ADDR block_tables, 
                         GM_ADDR seq_lens,
                         GM_ADDR metadata_block_tables,
                         GM_ADDR maxblocks, 
                         GM_ADDR minblocks, 
                         int32_t B, int32_t N, int32_t BLOCK_SIZE, int32_t D,
                         int32_t MKBPR, int32_t MMBPR)
    {
        B_ = B; N_ = N; BLOCK_SIZE_ = BLOCK_SIZE; D_ = D;
        MKBPR_ = MKBPR;
        MMBPR_ = MMBPR;

        k_cache_gm_.SetGlobalBuffer((__gm__ half *)k_cache);
        block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)block_tables);
        seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
        metadata_block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
        maxblocks_gm_.SetGlobalBuffer((__gm__ half *)maxblocks);
        minblocks_gm_.SetGlobalBuffer((__gm__ half *)minblocks);

        int32_t tileBytes = ceilDivMul(BLOCK_SIZE * D * sizeof(half), BYTES_UB_BLOCK); // bytes for metadata 1 kv-head
        pipe_.InitBuffer(k_block_in_q_, 2, tileBytes);   // original tile of K block
        pipe_.InitBuffer(work_calc_q_,  1, tileBytes);   // working copy of K block - for reductions
        pipe_.InitBuffer(max_out_q_,    1, tileBytes);
        pipe_.InitBuffer(min_out_q_,    1, tileBytes);
    }

    __aicore__ void Process()
    {
        int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        int32_t num_batch_heads = B_ * N_;

        for (int32_t batch_head_idx = GetBlockIdx(); batch_head_idx < num_batch_heads; batch_head_idx += num_blocks) {
            int32_t r = batch_head_idx / N_;  // "request id" aka "batch index". Number in [0..B-1]
            int32_t h = batch_head_idx % N_;  // kv-head aka kv-group index within the batch index

            int32_t seq_len = seq_lens_gm_.GetValue(r);
            int32_t num_kv_blocks_in_request = ceilDiv(seq_len, BLOCK_SIZE_);
            int32_t num_meta_blocks_in_request = ceilDiv(num_kv_blocks_in_request, BLOCK_SIZE_);

            /* main loop across the BLOCK_SIZE*BLOCK_SIZE toekns to build 1 metadata block*/
            for (int32_t meta_blk = 0; meta_blk < num_meta_blocks_in_request; meta_blk++){
                /* allocate metadata output minblock and maxblock for this head */
                LocalTensor<half> max_lt = max_out_q_.AllocTensor<half>();
                LocalTensor<half> min_lt = min_out_q_.AllocTensor<half>();
                Duplicate<half>(max_lt, (half)(-65504.0f), BLOCK_SIZE_ * D_);
                Duplicate<half>(min_lt, (half)( 65504.0f), BLOCK_SIZE_ * D_);
                
                /* loop over real KV-blocks of this request that should be packed into 1 current meta_blk */
                int32_t num_kv_blocks_completed = meta_blk * BLOCK_SIZE_;
                int32_t num_kv_blocks_todo_curr_iter = min(num_kv_blocks_in_request - num_kv_blocks_completed, BLOCK_SIZE_);
                for (int32_t blk = 0; blk < num_kv_blocks_todo_curr_iter; ++blk) {
                    // tail check - set ntokens_to_reduce to the number of valid tokens in the current K block
                    int32_t ntokens_to_reduce;
                    if ((blk == num_kv_blocks_todo_curr_iter - 1) && 
                        (meta_blk == num_meta_blocks_in_request - 1)) {
                        // tail (last KV block) - do not reduce over all tokens!
                        int32_t ntokens_reduced_so_far = (meta_blk * BLOCK_SIZE_  + blk) * BLOCK_SIZE_;
                        ntokens_to_reduce = seq_len - ntokens_reduced_so_far;
                    } else {
                        ntokens_to_reduce = BLOCK_SIZE_;
                    }
                    
                    /* 1. Copy head-slice: 4-D [blocks, BLOCK_SIZE, N, D] → UB [BLOCK_SIZE, D] */
                    // MKBPR_ block slots exist for each request in block_tables 
                    int32_t kv_block_id = block_tables_gm_.GetValue(r * MKBPR_ + num_kv_blocks_completed + blk); 
                    int32_t kv_block_offset = (kv_block_id * BLOCK_SIZE_ * N_ * D_) + h * D_;                
                    LocalTensor<half> k_block_lt = k_block_in_q_.AllocTensor<half>();
                    DataCopyParams gm_ub_cp;
                    gm_ub_cp.blockCount = ntokens_to_reduce;
                    gm_ub_cp.blockLen   = ceilDiv(D_ * sizeof(half), BYTES_DATA_BLOCK);
                    gm_ub_cp.srcStride  = ceilDiv((N_ - 1) * D_ * sizeof(half), BYTES_DATA_BLOCK); // skip other heads
                    gm_ub_cp.dstStride  = 0;
                    DataCopy(k_block_lt, k_cache_gm_[kv_block_offset], gm_ub_cp);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    
                    /* 2. max-reduction (copy first so k_block_lt is preserved) */
                    uint64_t mask = D_; // D must be 128, otherwise everything breaks
                    CopyRepeatParams ub_ub_cp = { 1, 1, 8, 8 }; // contiguous --> contiguous copy
                    LocalTensor<half> work_lt = work_calc_q_.AllocTensor<half>();
                    Copy(work_lt, k_block_lt, mask, ntokens_to_reduce, ub_ub_cp);
                    ReduceTokenDim<half, true>(work_lt, ntokens_to_reduce * D_);  // true -> Max
                    // copy the first D_ numbers which are per-channel maximum acrosss BLOCK_SIZE tokens in the K-block
                    Copy(max_lt[blk * D_], work_lt, mask, 1, ub_ub_cp);  

                    /* 3. min-reduction (reuse work_lt) */
                    Copy(work_lt, k_block_lt, mask, ntokens_to_reduce, ub_ub_cp);
                    ReduceTokenDim<half, false>(work_lt, ntokens_to_reduce * D_);  // false -> Min
                    // copy the first D_ numbers which are per-channel minimum acrosss BLOCK_SIZE tokens in the K-block
                    Copy(min_lt[blk * D_], work_lt, mask, 1, ub_ub_cp);  
                    k_block_in_q_.FreeTensor(k_block_lt);
                    work_calc_q_.FreeTensor(work_lt);
                }

                // Tail filling with zero
                int32_t num_unused_metadata_rows = BLOCK_SIZE_ - num_kv_blocks_todo_curr_iter;
                if (num_unused_metadata_rows > 0) {
                    Duplicate<half>(max_lt[num_kv_blocks_todo_curr_iter * D_], (half)(0.0f), 
                                    num_unused_metadata_rows * D_);
                    Duplicate<half>(min_lt[num_kv_blocks_todo_curr_iter * D_], (half)(0.0f), 
                                    num_unused_metadata_rows * D_);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                /* 4. Copy-out with head stride */
                int32_t meta_blk_id   = metadata_block_tables_gm_.GetValue(r * MMBPR_ + meta_blk);
                int32_t meta_offset = (meta_blk_id * BLOCK_SIZE_ * N_ * D_) + h * D_;

                // struct DataCopyParams:
                // blockCount: Specifies the number of consecutive "data blocks" contained in the 
                //             command. The value range is blockCount[1, 4095].
                // blockLen:   Specifies the length of each consecutive "data block" transferred by 
                //             the instruction. The unit is data block (32 bytes). Value range: blockLen=1, 65535.
                // srcStride:  Source operand, interval between adjacent consecutive data blocks. (The interval 
                //             between the tail of the previous block and the head of the subsequent block).
                // dstStride:  Destination operand, which is the interval between adjacent consecutive data blocks. 
                //             (The interval between the tail of the previous block and the head of the suMKBPRuent block).
                DataCopyParams ub_gm_cp;
                ub_gm_cp.blockCount = BLOCK_SIZE_; 
                ub_gm_cp.blockLen   = ceilDiv(D_ * sizeof(half), BYTES_UB_BLOCK);
                ub_gm_cp.srcStride  = 0;
                ub_gm_cp.dstStride  = ceilDiv((N_ - 1) * D_ * sizeof(half), BYTES_UB_BLOCK);
                DataCopy(maxblocks_gm_[meta_offset], max_lt, ub_gm_cp);
                DataCopy(minblocks_gm_[meta_offset], min_lt, ub_gm_cp);

                max_out_q_.FreeTensor(max_lt);
                min_out_q_.FreeTensor(min_lt);
            }
        }
    }

private:
    /**
    * @brief logarithmic reduce along dim=0 of vec_lt BLOCK_SIZE dimension: 
    *   (BLOCK_SIZE, D) -> (1, D) vector 
    * @param vec_lt - local tensor to be reduced in place
    * @param initial_length - length of the vec_lt
    * @return in-place D maximum values are in the beginning of vec_lt
    */
    template <typename T, bool isMax>
    __aicore__ void ReduceTokenDim(LocalTensor<T> vec_lt, int32_t initial_length)
    {
        if (initial_length != BLOCK_SIZE_ * D_)
            AscendC::PipeBarrier<PIPE_V>();  // no need to synch otherwise for some reason

        int32_t len = initial_length;

        while (len > D_) {
            int32_t num_vec = len / D_;
            int32_t pair_vec = num_vec >> 1;
            int32_t has_tail = num_vec & 1;

            int32_t reduce_len = pair_vec * D_;

            if (reduce_len > 0) {
            if (isMax) {
                    Max(vec_lt[0], vec_lt[0], vec_lt[reduce_len], reduce_len);
            } else {       
                    Min(vec_lt[0], vec_lt[0], vec_lt[reduce_len], reduce_len);
            }
            }
            
            // If odd number of vectors, move last one forward
            if (has_tail) {
                Copy(vec_lt[reduce_len], vec_lt[(num_vec - 1) * D_], D_, 1, { 1, 1, 8, 8 });
                reduce_len += D_;
            }

            len = reduce_len;
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    TPipe pipe_;
    TQue<TPosition::VECIN,   DOUBLEBUFFER> k_block_in_q_;
    TQue<TPosition::VECCALC, SINGLEBUFFER> work_calc_q_;
    TQue<TPosition::VECOUT,  SINGLEBUFFER> max_out_q_;
    TQue<TPosition::VECOUT,  SINGLEBUFFER> min_out_q_;

    GlobalTensor<half> k_cache_gm_;
    GlobalTensor<half> maxblocks_gm_, minblocks_gm_;
    GlobalTensor<int32_t> block_tables_gm_, seq_lens_gm_, metadata_block_tables_gm_;

    int32_t B_, N_, BLOCK_SIZE_, D_, MKBPR_, MMBPR_;
};

/**
 * @brief Kernel implementation for Quest initialization of metadata after LLM prefilling of the KV cache
 * For ever request r in [0..B-1], the metadata block id is denoted meta_blk_id and is specified in the input:
 *  metadata_block_tables[r] = meta_blk_id
 * the kernel computes two metadata blocks
 *   1) maxblock - [BLOCK_SIZE, N, D] --> written into maxblocks[meta_blk_id, :, :, :]
 *   2) minblock - [BLOCK_SIZE, N, D] --> written into minblocks[meta_blk_id, :, :, :]
 * Functionality:
     assuming that request r has num_blocks_req = (seq_lens[r] / BLOCK_SIZE) blocks in its KV-cache:
     for blk,kvbid = enumerate(block_tables[r]):
       kv_block = k_cache[kvbid,:,:,:] # [BLOCK_SIZE, N, D]
       for n in range(N):
         for d in range(D)
           maxblock[blk, n, d] = max(kv_block[:,n,d], dim=0) # reduce-max token dimension BLOCK_SIZE --> 1
           minblock[blk, n, d] = min(kv_block[:,n,d], dim=0) # reduce-min token dimension BLOCK_SIZE --> 1
 *
 * @param [in] k_cache [num_blocks_total, BLOCK_SIZE, N, D] - used to hold 
                  the input context of K matrices and will be used by the 
                  kernel to write out the maxblock metadata (into block number 
                  specified by metadata_block_tables).
 * @param [in] block_tables [B, MKBPR]
 * @param [in] seq_lens [B,] number of tokens per request in batch
 * @param [in] metadata_block_tables [B, MMBPR] - metadata block number per request in batch 
 * @param [out] maxblocks - [num_metadata_blocks_total, BLOCK_SIZE, N, D] to 
                write out the minblock metadata (into block number specified by
                metadata_block_tables).
 * @param [out] minblocks - [num_metadata_blocks_total, BLOCK_SIZE, N, D] to 
                write out the minblock metadata (into block number specified by
                metadata_block_tables).
 * @param [in] B Batch size
 * @param [in] N Number of KV heads
 * @param [in] BLOCK_SIZE Number of vectors in maxblock and in minblock
 * @param [in] D head dimension
 * @param [in] MKBPR maximum number of blocks in every entry of the block_tables (num columns)
 * @param [in] MMBPR maximum number of blocks in every entry of the metadata_block_tables (num columns)
 */
extern "C" __global__ __aicore__ void
quest_prefill_metadata(GM_ADDR k_cache,
                       GM_ADDR block_tables, GM_ADDR seq_lens,
                       GM_ADDR metadata_block_tables,
                       GM_ADDR maxblocks,
                       GM_ADDR minblocks,
                       int32_t B, int32_t N,
                       int32_t BLOCK_SIZE, int32_t D, 
                       int32_t MKBPR, int32_t MMBPR)
{
    KernelQuestMetadata op;
    op.Init(k_cache, block_tables, seq_lens, metadata_block_tables,
            maxblocks, minblocks, B, N, BLOCK_SIZE, D, MKBPR, MMBPR);
    op.Process();
}

/**
 * Kernel launch function
 */
void launch_quest_prefill_metadata(
    uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *k_cache, uint8_t *block_tables,
    uint8_t *seq_lens, uint8_t *metadata_block_tables,
    uint8_t *maxblocks, uint8_t *minblocks,
    int32_t B, int32_t N, int32_t BLOCK_SIZE,
    int32_t D, int32_t MKBPR,
    int32_t MMBPR)
{
    quest_prefill_metadata<<<blockDim, l2ctrl, stream>>>(
        k_cache,
        block_tables,
        seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
        B, N, BLOCK_SIZE, D, MKBPR, MMBPR);
}
