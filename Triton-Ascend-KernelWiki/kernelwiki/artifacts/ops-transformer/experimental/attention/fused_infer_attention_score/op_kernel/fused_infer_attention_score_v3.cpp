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
 * \file fused_infer_attention_score_v3.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "fused_infer_attention_score_tilingkey.h"

#ifdef FIA_ENABLE_MLA
// mla模板使用私有tiling结构，框架编译时根据一组DType预编译获取keylist，根据keylist找到对应的tiling结构
// 在这组DType中，若没有mla模板的key，包含mla模板编译会报错：unknown type name 'FusedInferAttentionScoreTilingData'
#if ((ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)) || \
    ((ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16))
#ifdef NOT_DYNAMIC_COMPILE
#include "../../common/op_kernel/arch32/fia_kernel_nonquant_mla.h"
#include "../../common/op_kernel/arch32/fia_kernel_nonquant.h"
#else
#include "../common/arch32/fia_kernel_nonquant_mla.h"
#include "../common/arch32/fia_kernel_nonquant.h"
#endif
#endif
#endif // FIA_ENABLE_MLA

using namespace AscendC;

#define INVOKE_FIA_OP_GENERAL_IMPL(templateClass, CubeBlockType, VecBlockType, FdBlockType, ...)                       \
    do {                                                                                                               \
        using CubeBlockTypeT = CubeBlockType<FIAType<__VA_ARGS__>>;                                                    \
        using VecBlockTypeT = VecBlockType<FIAType<__VA_ARGS__>>;                                                      \
        using FdBlockTypeT = FdBlockType<FIAType<__VA_ARGS__>>;                                                        \
                                                                                                                       \
        templateClass<FIAType<__VA_ARGS__>, CubeBlockTypeT, VecBlockTypeT, FdBlockTypeT> op;                           \
        FIA_COPY_TILING_DATA(FusedInferAttentionScoreTilingData, tiling);                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, deqScale1, quantScale1,   \
                deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blockTable, queryPaddingSize,   \
                kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,       \
                keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale,  \
                learnableSink, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                           \
        op.Process();                                                                                                  \
    } while (0)

#define FIA_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                 \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;

extern "C" __global__ __aicore__ void fused_infer_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;

    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    // Mla NoPA bf16 kv_BNSD
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_BSH_BSND
    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);


    // Gqa NoQuant Non PA Non Perf
    TILING_KEY_IS(103000000000000000);
    TILING_KEY_IS(103000000010000001);
    TILING_KEY_IS(103000000000100000);
    TILING_KEY_IS(103000000010100001);
    TILING_KEY_IS(103000000000400000);
    TILING_KEY_IS(103000000010400001);
    TILING_KEY_IS(103000000000500000);
    TILING_KEY_IS(103000000010500001);


// Mla NoPA fp16 kv_BNSD
#if (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
// Mla NoPA fp16 kv_BSH_BSND
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, half, half, half,
                                  half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);

// Gqa NoQuant Non PA Non Perf
#elif TILING_KEY_VAR == 103000000000000000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000010000001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000000100000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               false);
#elif TILING_KEY_VAR == 103000000010100001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               false);
#elif TILING_KEY_VAR == 103000000000400000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000010400001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);
#elif TILING_KEY_VAR == 103000000000500000
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD,
                               true);
#elif TILING_KEY_VAR == 103000000010500001
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH,
                               true);


#endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16)

    // Mla NoPA bf16 kv_BNSD
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Mla NoPA bf16 kv_BSH_BSND
    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_LIST(QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    KERNEL_TASK_TYPE(C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING, KERNEL_TYPE_MIX_AIC_1_1);

    // Gqa NoQuant Non PA
    TILING_KEY_IS(103000000000022220);
    TILING_KEY_IS(103000000010022221);
    TILING_KEY_IS(103000000000122220);
    TILING_KEY_IS(103000000010122221);
    TILING_KEY_IS(103000000000422220);
    TILING_KEY_IS(103000000010422221);
    TILING_KEY_IS(103000000000522220);
    TILING_KEY_IS(103000000010522221);



// Mla NoPA bf16 kv_BNSD
#if (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
// Mla NoPA bf16 kv_BSH_BSND
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif (TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) || (TILING_KEY_VAR == C1V1_QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING) // 7buf
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuantMla, FiaBlockCubeNonQuantMla, FiaBlockVecNonQuantMla,
                               FiaBlockVecFlashDecode, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
// Gqa NoQuant Non PA
#elif TILING_KEY_VAR == 103000000000022220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010022221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000000122220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010122221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000000422220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010422221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000000522220
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false,
                               false, FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010522221
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false,
                               false, FIA_LAYOUT::BSH, true);

#endif
#endif
}
