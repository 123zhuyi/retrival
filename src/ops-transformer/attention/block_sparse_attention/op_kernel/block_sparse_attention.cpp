/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "block_sparse_attention_tilingkey.h"
#include "block_sparse_attention_interface.cpp"

extern "C" __global__ __aicore__ void block_sparse_attention(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                            __gm__ uint8_t* blockSparseMask, __gm__ uint8_t* mask, __gm__ uint8_t* blockShape,
                                                            __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKv, __gm__ uint8_t* blockTable,
                                                            __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    if (TILING_KEY_VAR >= RFA_BASE_TILING) {
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // 读取tilingKey进行kernel分发
    __gm__ BlockSparseAttentionTilingData *tilingDataPtr = 
        reinterpret_cast<__gm__ BlockSparseAttentionTilingData *>(tiling);
    uint64_t tilingKey = tilingDataPtr->tilingKey;
    
    TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING);
    TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_RFA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING);
    TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING);
    TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_RFA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING);
    TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT);
    TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_RFA_TILING_LSE_OUT);
    TILING_KEY_IS(QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT);
    TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT);
    TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_RFA_TILING_LSE_OUT);
    TILING_KEY_IS(QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT);

    #if TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        BlockSparse::BlockSparseAttentionInfer<half, float, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_RFA_TILING
        BlockSparse::BlockSparseAttentionInfer<half, half, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        BlockSparse::BlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        BlockSparse::BlockSparseAttentionInfer<half, float, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_RFA_TILING
        BlockSparse::BlockSparseAttentionInfer<half, half, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        BlockSparse::BlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT
        BlockSparse::BlockSparseAttentionInfer<half, float, Epilogue::LseMode::OUT_ONLY, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_RFA_TILING_LSE_OUT
        BlockSparse::BlockSparseAttentionInfer<half, half, Epilogue::LseMode::OUT_ONLY, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT
        BlockSparse::BlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::OUT_ONLY, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT
        BlockSparse::BlockSparseAttentionInfer<half, float, Epilogue::LseMode::OUT_ONLY, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_RFA_TILING_LSE_OUT
        BlockSparse::BlockSparseAttentionInfer<half, half, Epilogue::LseMode::OUT_ONLY, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #elif TILING_KEY_VAR == QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING_LSE_OUT
        BlockSparse::BlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::OUT_ONLY, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #else
        BlockSparse::BlockSparseAttentionInfer<half, half, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    #endif
    }
}

