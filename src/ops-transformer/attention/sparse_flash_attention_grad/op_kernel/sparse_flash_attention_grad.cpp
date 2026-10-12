/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_grad.cpp
 * \brief
 */

#if __CCE_AICORE__ == 310
#include "arch35/sparse_flash_attention_grad_template_tiling_key.h"
#include "arch35/sparse_flash_attention_grad_tiling_data_regbase.h"
#include "arch35/sparse_flash_attention_grad_entry_regbase.h"
#else
#include "arch32/sparse_flash_attention_grad_bs1_basic.h"
#include "arch32/sparse_flash_attention_grad_post.h"
#endif

#include "kernel_operator.h"

using namespace AscendC;
constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;
constexpr static uint32_t TND = 3;

#if __CCE_AICORE__ == 310

template <uint8_t inputDType, bool isTnd, uint16_t gTemplateType, uint16_t s2TemplateType, uint16_t dTemplateType, bool isRope, bool deterministic>
__global__ __aicore__ void
sparse_flash_attention_grad(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                            __gm__ uint8_t *sparse_indices, __gm__ uint8_t *d_out, __gm__ uint8_t *out, 
                            __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum, 
                            __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                            __gm__ uint8_t *query_rope, __gm__ uint8_t *key_rope,
                            __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                            __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope,
                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    REGISTER_TILING_DEFAULT(optiling::sfag::SparseFlashAttentionGradTilingDataRegbase);
    RegbaseSFAG<inputDType, isTnd, gTemplateType, s2TemplateType, dTemplateType, isRope, deterministic>(query, key, value,
                                sparse_indices, d_out, out, 
                                softmax_max, softmax_sum, 
                                actual_seq_qlen, actual_seq_kvlen,
                                query_rope, key_rope,
                                dq, dk, dv,
                                dq_rope, dk_rope,
                                workspace, tiling_data);
}


#else
#define INVOKE_SELECTED_ATTENTION_BASIC_IMPL(INPUT_TYPE, ATTEN_ENABLE, HAS_ROPE, IS_BSND)                               \
    do {                                                                                                                \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                             \
        GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionGradBasicTilingData, tiling_data_in, tiling_data);              \
        const SparseFlashAttentionGradBasicTilingData *__restrict tilingData = &tiling_data_in;                         \
        SFAG_BASIC::SelectedAttentionGradBasic<                                                                         \
            SFAG_BASIC::SFAG_TYPE<SparseFlashAttentionGradBasicTilingData, INPUT_TYPE, TND, ATTEN_ENABLE, HAS_ROPE, IS_BSND>>   \
            op;                                                                                                         \
        op.Process(query, key, value, out, d_out, softmax_max, softmax_sum, sparse_indices,                             \
                   actual_seq_qlen, actual_seq_kvlen, query_rope, key_rope,                                             \
                   dq, dk, dv, dq_rope, dk_rope, user, tilingData);                                                     \
    } while (0)


extern "C" __global__ __aicore__ void
sparse_flash_attention_grad(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                            __gm__ uint8_t *sparse_indices, __gm__ uint8_t *d_out, __gm__ uint8_t *out, 
                            __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum, 
                            __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                            __gm__ uint8_t *query_rope, __gm__ uint8_t *key_rope,
                            __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                            __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope,
                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (TILING_KEY_IS(1000)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, false, false, false);
        return;
    } else if (TILING_KEY_IS(1100)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, true, false, false);
        return;
    } else if (TILING_KEY_IS(1010)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, false, true, false);
        return;
    } else if (TILING_KEY_IS(1110)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, true, true, false);
        return;
    } else if (TILING_KEY_IS(1001)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, false, false, true);
        return;
    } else if (TILING_KEY_IS(1101)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, true, false, true);
        return;
    } else if (TILING_KEY_IS(1011)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, false, true, true);
        return;
    } else if (TILING_KEY_IS(1111)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, true, true, true);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(1000)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, false, false, false);
        return;
    } else if (TILING_KEY_IS(1100)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, true, false, false);
        return;
    } else if (TILING_KEY_IS(1010)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, false, true, false);
        return;
    } else if (TILING_KEY_IS(1110)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, true, true, false);
        return;
    } else if (TILING_KEY_IS(1001)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, false, false, true);
        return;
    } else if (TILING_KEY_IS(1101)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, true, false, true);
        return;
    } else if (TILING_KEY_IS(1011)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, false, true, true);
        return;
    } else if (TILING_KEY_IS(1111)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, true, true, true);
        return;
    }
#endif

}
#endif