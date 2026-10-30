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
 * \file flash_attention_score_grad_entry_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_ENTRY_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_ENTRY_REGBASE_H_

#include "common.h"

#include "flash_attention_score_grad_s1s2_bn2s2_regbase.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_regbase.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_pre_regbase.h"
#include "flash_attention_score_grad_presfmg_regbase.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_attention_score_grad_block_vec.h"
#include "flash_attention_score_grad_block_vec_quant.h"
#include "flash_attention_score_grad_block_cube.h"
#include "flash_attention_score_grad_block_cube_quant.h"
#include "flash_attention_score_grad_kernel.h"
#include "flash_attention_score_grad_kernel_deter.h"
#include "flash_attention_score_grad_kernel_quant.h"
 
 
#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(                                                                 \
    INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE,           \
    FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType)                                \
    do {                                                                                                               \
        FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<INPUT_TYPE, float, DETER_SPARSE_TYPE, IS_TND, SPLIT_AXIS, IS_TND_SWIZZLE> opPre;  \
        opPre.Init(dq, dk, dv, actual_seq_kvlen, drop_mask, user, tilingData, &pipeIn);                                \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeBase;                                                                                                \
        using CubeBlockType =                                                                                          \
            typename std::conditional<g_coreType == AscendC::AIC, FagBaseApi::FAGBlockCube<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>,               \
                                      FagBaseApi::FAGBlockCubeDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>>::type;                               \
        using VecBlockType =                                                                                           \
            typename std::conditional<g_coreType == AscendC::AIC, FagBaseApi::FAGBlockVecDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>,           \
                                      FagBaseApi::FAGBlockVec<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>>::type;                                     \
                                                                                                                       \
        typename std::conditional<(DETER_SPARSE_TYPE) == NO_DETER, FagBaseApi::FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>, FagBaseApi::FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>>::type op; \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, queryRope,     \
                keyRope, dq, dk, dv, dpse, dqRope, dkRope, user, tilingData, &pipeBase);                               \
        op.Process();                                                                                                  \
        if (ORIG_DTYPE_QUERY != DT_FLOAT) {                                                                            \
            op.SyncALLCores();                                                                                         \
            pipeBase.Destroy();                                                                                        \
            TPipe pipePost;                                                                                            \
            FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<INPUT_TYPE, float, OUTDTYPE, SPLIT_AXIS, IS_ROPE,            \
                                                          DETER_SPARSE_TYPE, IS_TND, IS_TND_SWIZZLE>                                   \
                opPost;                                                                                                \
            opPost.Init(dq, dk, dv, dqRope, dkRope, user, tilingData, &pipePost);                                      \
            opPost.Process();                                                                                          \
        } else {                                                                                                       \
            pipeBase.Destroy();                                                                                        \
        }                                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_HIFP8_REGBASE_IMPL(                                                                 \
    INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE,           \
    FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType)                                \
    do {                                                                                                               \
        FlashAttentionScoreGradPresfmgRegbase<INPUT_TYPE, CALC_TYPE, OUTDTYPE, dTemplateType, IS_D_NO_EQUAL, DETER_SPARSE_TYPE, IS_TND, SPLIT_AXIS> opPre;  \
        opPre.Init(dq, dk, dv, dy, attention_in, deqScaleDy, actual_seq_kvlen, user, tilingData, &pipeIn);                        \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeBase;                                                                                                \
        using CubeBlockType =                                                                                          \
            typename std::conditional<g_coreType == AscendC::AIC, FagBaseApi::FAGBlockCubeQuant<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>,               \
                                      FagBaseApi::FAGBlockCubeQuantDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>>::type;                               \
        using VecBlockType =                                                                                           \
            typename std::conditional<g_coreType == AscendC::AIC, FagBaseApi::FAGBlockVecQuantDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>,           \
                                      FagBaseApi::FAGBlockVecQuant<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>>::type;                                     \
                                                                                                                       \
        FagBaseApi::FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType> op; \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, dsScale, pScale, queryRope,     \
                keyRope, dq, dk, dv, dpse, dqRope, dkRope, user, tilingData, &pipeBase);                               \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                        \
        TPipe pipePost;                                                                                            \
        FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<INPUT_TYPE, float, OUTDTYPE, SPLIT_AXIS, IS_ROPE,            \
                                                        DETER_SPARSE_TYPE, IS_TND, IS_TND_SWIZZLE>                                   \
            opPost;                                                                                                \
        opPost.Init(dq, dk, dv, dqRope, dkRope, user, tilingData, &pipePost);                                      \
        opPost.Process();                                                                                          \
    } while (0)
 
#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK,    \
                                                    IS_DETER, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, SPLIT_AXIS, s1TemplateType,      \
                                                    s2TemplateType, dTemplateType, OUTDTYPE)                           \
    do {                                                                                                               \
        FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<INPUT_TYPE, float, IS_DETER, IS_TND, SPLIT_AXIS> opPre;           \
        opPre.Init(dq, dk, dv, actual_seq_kvlen, drop_mask, user, tilingData, &pipeIn);                                \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND,  \
                                                         IS_BN2_MULTIBLK, IS_DETER, 0, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, SPLIT_AXIS,       \
                                                         s1TemplateType, s2TemplateType, dTemplateType, OUTDTYPE>      \
            op;                                                                                                        \
                                                                                                                       \
        TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> dsScm;                                                            \
        TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> pScm;                                                             \
        if constexpr (IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value ||      \
                      (IS_DETER) == DETER_OLD) {                                                                       \
            pipeBase.InitBuffer(dsScm, 1,                                                                              \
                                (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE) * 2);          \
        } else {                                                                                                       \
            pipeBase.InitBuffer(dsScm, 1, (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE));    \
        }                                                                                                              \
        pipeBase.InitBuffer(pScm, 1, (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE));         \
        GlobalTscmArrayStatic tscmArray[TSCM_BUF_NUM];                                                                 \
        gTscmArray = tscmArray;                                                                                        \
        GlobalL0CArrayStatic                                                                                           \
            l0cArray[GET_L0C_BUF_NUM((uint32_t)s1TemplateType, (uint32_t)s2TemplateType, (uint32_t)dTemplateType)];    \
        gL0cArray = l0cArray;                                                                                          \
        InitTSCMBuffer<INPUT_TYPE, s1TemplateType, s2TemplateType, dTemplateType, SPLIT_AXIS,                          \
                        (IS_DETER) == DETER_OLD, IS_TND, IS_ROPE, FP8_OPEN_TSCM>(&pipeBase, gTscmArray);                              \
        InitL0CBuffer<INPUT_TYPE, s1TemplateType, s2TemplateType, dTemplateType, (IS_DETER) == DETER_OLD, IS_TND>(&pipeBase,          \
                                                                                                   gL0cArray);         \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, (TCubeTiling *)nullptr, op.mm2,                     \
                          (TCubeTiling *)nullptr, op.mm3, (TCubeTiling *)nullptr);                                     \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy,                \
                queryRope, keyRope, dq, dk, dv, dpse, dqRope, dkRope, user, tilingData, &pipeBase, dsScm, pScm);       \
        op.Process();                                                                                                  \
        if (ORIG_DTYPE_QUERY != DT_FLOAT) {                                                                            \
            op.SyncALLCores();                                                                                         \
            pipeBase.Destroy();                                                                                        \
            TPipe pipePost;                                                                                            \
            FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<INPUT_TYPE, float, OUTDTYPE, SPLIT_AXIS, IS_ROPE, IS_DETER, IS_TND> opPost;    \
            opPost.Init(dq, dk, dv, dqRope, dkRope, user, tilingData, &pipePost);                                      \
            opPost.Process();                                                                                          \
        } else {                                                                                                       \
            pipeBase.Destroy();                                                                                        \
        }                                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_FP8_OLD_REGBASE_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK,    \
                                                    IS_DETER, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, SPLIT_AXIS, s1TemplateType,      \
                                                    s2TemplateType, dTemplateType, OUTDTYPE)                           \
    do {                                                                                                               \
        FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<INPUT_TYPE, float, IS_DETER, IS_TND, SPLIT_AXIS> opPre;           \
        opPre.Init(dq, dk, dv, actual_seq_kvlen, drop_mask, user, tilingData, &pipeIn);                                \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND,  \
                                                         IS_BN2_MULTIBLK, IS_DETER, 0, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, SPLIT_AXIS,       \
                                                         s1TemplateType, s2TemplateType, dTemplateType, OUTDTYPE>      \
            op;                                                                                                        \
                                                                                                                       \
        TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> dsScm;                                                            \
        TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> pScm;                                                             \
        if constexpr (IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value ||      \
                      (IS_DETER) == DETER_OLD) {                                                                       \
            pipeBase.InitBuffer(dsScm, 1,                                                                              \
                                (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE) * 2);          \
        } else {                                                                                                       \
            pipeBase.InitBuffer(dsScm, 1, (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE));    \
        }                                                                                                              \
        pipeBase.InitBuffer(pScm, 1, (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE));         \
        GlobalTscmArrayStatic tscmArray[TSCM_BUF_NUM];                                                                 \
        gTscmArray = tscmArray;                                                                                        \
        GlobalL0CArrayStatic                                                                                           \
            l0cArray[GET_L0C_BUF_NUM((uint32_t)s1TemplateType, (uint32_t)s2TemplateType, (uint32_t)dTemplateType)];    \
        gL0cArray = l0cArray;                                                                                          \
        InitTSCMBuffer<INPUT_TYPE, s1TemplateType, s2TemplateType, dTemplateType, SPLIT_AXIS,                          \
                        (IS_DETER) == DETER_OLD, IS_TND, IS_ROPE, FP8_OPEN_TSCM>(&pipeBase, gTscmArray);                              \
        InitL0CBuffer<INPUT_TYPE, s1TemplateType, s2TemplateType, dTemplateType, (IS_DETER) == DETER_OLD, IS_TND>(&pipeBase,          \
                                                                                                   gL0cArray);         \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, (TCubeTiling *)nullptr, op.mm2,                     \
                          (TCubeTiling *)nullptr, op.mm3, (TCubeTiling *)nullptr);                                     \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy,                \
                queryRope, keyRope, dq, dk, dv, dpse, dqRope, dkRope, user, tilingData, &pipeBase, dsScm, pScm);       \
        op.Process();                                                                                                  \
        if (ORIG_DTYPE_QUERY != DT_FLOAT) {                                                                            \
            op.SyncALLCores();                                                                                         \
            pipeBase.Destroy();                                                                                        \
            TPipe pipePost;                                                                                            \
            FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<INPUT_TYPE, float, OUTDTYPE, SPLIT_AXIS, IS_ROPE, IS_DETER, IS_TND> opPost;    \
            opPost.Init(dq, dk, dv, dqRope, dkRope, user, tilingData, &pipePost);                                      \
            opPost.Process();                                                                                          \
        } else {                                                                                                       \
            pipeBase.Destroy();                                                                                        \
        }                                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_HIFP8(...)                                                         \
    if (ORIG_DTYPE_QUERY == DT_HIFLOAT8)         \
    INVOKE_FAG_GENERAL_HIFP8_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(...)                                                        \
    if (ORIG_DTYPE_QUERY == DT_FLOAT16)                                                                                \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_BF16(...)                                                        \
    if (ORIG_DTYPE_QUERY == DT_BF16)                                                                                   \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP32(...)                                                        \
    if (ORIG_DTYPE_QUERY == DT_FLOAT)                                                                                  \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_FP16(...)                                                        \
    if (ORIG_DTYPE_QUERY == DT_FLOAT16)                                                                                \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL(__VA_ARGS__)
 
#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_BF16(...)                                                        \
    if (ORIG_DTYPE_QUERY == DT_BF16)                                                                                   \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL(__VA_ARGS__)
 
#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_FP32(...)                                                        \
    if (ORIG_DTYPE_QUERY == DT_FLOAT)                                                                                  \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL(__VA_ARGS__)
 
#define INVOKE_FAG_GENERAL_S1S2_BN2S2_REGBASE_IMPL(                                                                    \
    INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE,           \
    FP8_OPEN_TSCM, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType, OUTDTYPE)                                \
    do {                                                                                                               \
        FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<INPUT_TYPE, float, DETER_SPARSE_TYPE, IS_TND, SPLIT_AXIS, IS_TND_SWIZZLE> opPre;  \
        opPre.Init(dq, dk, dv, actual_seq_kvlen, drop_mask, user, tilingData, &pipeIn);                                \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradUs1s2Bbn2s2StaticRegbase<                                                               \
            INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL,     \
            IS_ROPE, FP8_OPEN_TSCM, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>                         \
            op;                                                                                                        \
        TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> dsScm;                                                            \
        TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> pScm;                                                             \
        pipeBase.InitBuffer(dsScm, 1, (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE));        \
        pipeBase.InitBuffer(pScm, 1, (uint32_t)s1TemplateType *(uint32_t)s2TemplateType * sizeof(INPUT_TYPE));         \
        GlobalTscmArrayStatic tscmArray[TSCM_BUF_NUM];                                                                 \
        gTscmArray = tscmArray;                                                                                        \
        GlobalL0CArrayStatic                                                                                           \
            l0cArray[GET_L0C_BUF_NUM((uint32_t)s1TemplateType, (uint32_t)s2TemplateType, (uint32_t)dTemplateType)];    \
        gL0cArray = l0cArray;                                                                                          \
        InitTSCMBuffer<INPUT_TYPE, s1TemplateType, s2TemplateType, dTemplateType, SPLIT_AXIS,                          \
                       (DETER_SPARSE_TYPE) == DETER_OLD>(&pipeBase, gTscmArray);                                       \
        InitL0CBuffer<INPUT_TYPE, s1TemplateType, s2TemplateType, dTemplateType, (DETER_SPARSE_TYPE) == DETER_OLD,     \
                      IS_TND>(&pipeBase, gL0cArray);                                                                   \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, (TCubeTiling *)nullptr, op.mm2,                     \
                          (TCubeTiling *)nullptr, op.mm3, (TCubeTiling *)nullptr);                                     \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, queryRope,     \
                keyRope, dq, dk, dv, dpse, dqRope, dkRope, user, tilingData, &pipeBase, dsScm, pScm);                  \
        op.Process();                                                                                                  \
        if (ORIG_DTYPE_QUERY != DT_FLOAT) {                                                                            \
            op.SyncALLCores();                                                                                         \
            pipeBase.Destroy();                                                                                        \
            TPipe pipePost;                                                                                            \
            FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<INPUT_TYPE, float, OUTDTYPE, SPLIT_AXIS, IS_ROPE,            \
                                                          DETER_SPARSE_TYPE, IS_TND, IS_TND_SWIZZLE>                                   \
                opPost;                                                                                                \
            opPost.Init(dq, dk, dv, dqRope, dkRope, user, tilingData, &pipePost);                                      \
            opPost.Process();                                                                                          \
            pipePost.Destroy();                                                                                        \
        } else {                                                                                                       \
            pipeBase.Destroy();                                                                                        \
        }                                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2S2_REGBASE_IMPL_FP16(...)                                                           \
    if (ORIG_DTYPE_QUERY == DT_FLOAT16)                                                                                \
    INVOKE_FAG_GENERAL_S1S2_BN2S2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2S2_REGBASE_IMPL_BF16(...)                                                           \
    if (ORIG_DTYPE_QUERY == DT_BF16)                                                                                   \
    INVOKE_FAG_GENERAL_S1S2_BN2S2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2S2_REGBASE_IMPL_FP32(...)                                                           \
    if (ORIG_DTYPE_QUERY == DT_FLOAT)                                                                                  \
    INVOKE_FAG_GENERAL_S1S2_BN2S2_REGBASE_IMPL(__VA_ARGS__)


#define INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL(                                                                      \
    INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE,       \
    IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType)      \
    do {                                                                                                               \
        if constexpr (IS_BN2_MULTIBLK && IS_TND) {                                                                     \
            FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<INPUT_TYPE, float, DETER_SPARSE_TYPE, IS_TND, SPLIT_AXIS> opPre;  \
            opPre.Init(dq, dk, dv, actual_seq_kvlen, drop_mask, user, tilingData, &pipeIn);                                \
            opPre.Process();                                                                                               \
            opPre.SyncALLCores();                                                                                          \
        }                                                                                                              \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeBase;                                                                                                \
        using CubeBlockType =                                                                                          \
            typename std::conditional<g_coreType == AscendC::AIC, FagBaseApi::FAGBlockCube<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>,               \
                                      FagBaseApi::FAGBlockCubeDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>>::type;                               \
        using VecBlockType =                                                                                           \
            typename std::conditional<g_coreType == AscendC::AIC, FagBaseApi::FAGBlockVecDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>,           \
                                      FagBaseApi::FAGBlockVec<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType>>::type;                                     \
                                                                                                                       \
        FagBaseApi::FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType> op;                                     \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, queryRope,     \
                keyRope, dq, dk, dv, dpse, dqRope, dkRope, user, tilingData, &pipeBase);                               \
        op.Process();                                                                                                  \
        pipeBase.Destroy();                                                                                            \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL_FP16(...)                                                             \
    if (ORIG_DTYPE_QUERY == DT_FLOAT16)                                                                                \
    INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL_BF16(...)                                                             \
    if (ORIG_DTYPE_QUERY == DT_BF16)                                                                                   \
    INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL_FP32(...)                                                             \
    if (ORIG_DTYPE_QUERY == DT_FLOAT)                                                                                  \
    INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL(__VA_ARGS__)

// implementation of kernel function
template <uint8_t splitAxis, uint8_t inputDType, bool isTnd, bool isDrop, bool isPse, bool isAttenMask,
          uint16_t s1TemplateType, uint16_t s2TemplateType, uint16_t dTemplateType, uint8_t deterType, bool isNEqual, bool isBn2MultiBlk,
          bool isDNoEqual, bool isRope, uint8_t outDType, bool fp8OpenTscm, bool isTndSwizzle, bool isRegbase>
inline __aicore__ void
RegbaseFAG(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dy,
           __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *padding_mask,
           __gm__ uint8_t *atten_mask, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
           __gm__ uint8_t *softmax_in, __gm__ uint8_t *attention_in, __gm__ uint8_t *prefix,
           __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *deqScaleQ,
           __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy, __gm__ uint8_t *dsScale, __gm__ uint8_t *pScale, __gm__ uint8_t *queryRope,
           __gm__ uint8_t *keyRope, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
           __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipeIn;
    SetMaskNorm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    constexpr static bool needDeterPrefix = NEED_DETER_PREFIX(deterType, isTnd);
    using fagTiling = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<needDeterPrefix, isTnd, isTndSwizzle>;
    GET_TILING_DATA_WITH_STRUCT(fagTiling, tiling_data_in, tiling_data);
    const fagTiling *__restrict tilingData = &tiling_data_in;
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
        if constexpr (splitAxis == BN2GS1S2) {
            if constexpr (deterType != DETER_OLD) {
                INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(
                    half, float, half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2GS1S2,
                    S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType)); 
            } else {
                if constexpr (dTemplateType == 768){
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_FP16(
                        half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isDNoEqual, isRope, fp8OpenTscm, BN2GS1S2, S1TemplateType(s1TemplateType), 
                        S2TemplateType(s2TemplateType), DTemplateType(512), half);
                }
                else
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_FP16(
                        half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isDNoEqual, isRope, fp8OpenTscm, BN2GS1S2, S1TemplateType(s1TemplateType), 
                        S2TemplateType(s2TemplateType), DTemplateType(dTemplateType), half);
            }
            return;
        } else if constexpr (splitAxis == BN2S2) {
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(
                half, float, half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2S2,
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            return;
        } else if constexpr (splitAxis == BN2) {
            INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL_FP16(
                half, float, half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, 
                BN2, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            return;
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_BF16)
        if constexpr (splitAxis == BN2GS1S2) {
            if constexpr (deterType != DETER_OLD) {
                INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_BF16(
                    bfloat16_t, float, bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle,
                    BN2GS1S2, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            } else {
                if constexpr (dTemplateType == 768){
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_BF16(
                        bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isDNoEqual, isRope, fp8OpenTscm, BN2GS1S2, S1TemplateType(s1TemplateType), 
                        S2TemplateType(s2TemplateType), DTemplateType(512), bfloat16_t);
                }
                else
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_BF16(
                        bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isDNoEqual, isRope, fp8OpenTscm, BN2GS1S2, S1TemplateType(s1TemplateType), 
                        S2TemplateType(s2TemplateType), DTemplateType(dTemplateType), bfloat16_t);
            }
            return;
        } else if constexpr (splitAxis == BN2S2) {
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_BF16(
                bfloat16_t, float, bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2S2,
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            return;
        } else if constexpr (splitAxis == BN2) {
            INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL_BF16(bfloat16_t, float, bfloat16_t, isAttenMask, isPse, isDrop, isTnd,
                                                        isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2,
                                                        S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                                                        DTemplateType(dTemplateType));
            return;
        }
    #endif
 
    #if (ORIG_DTYPE_QUERY == DT_FLOAT)
        if constexpr (splitAxis == BN2GS1S2) {
            if constexpr (deterType != DETER_OLD) {
                INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP32(
                    float, float, float, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2GS1S2,
                    S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            } else {
                if constexpr (dTemplateType == 768){
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_FP32(
                        float, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isDNoEqual, isRope, fp8OpenTscm, BN2GS1S2, S1TemplateType(s1TemplateType), 
                        S2TemplateType(s2TemplateType), DTemplateType(512), float);
                }
                else
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_OLD_DETER_REGBASE_IMPL_FP32(
                        float, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isDNoEqual, isRope, fp8OpenTscm, BN2GS1S2, S1TemplateType(s1TemplateType), 
                        S2TemplateType(s2TemplateType), DTemplateType(dTemplateType), float);
            }
            return;
        } else if constexpr (splitAxis == BN2S2) {
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP32(
                float, float, float, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2S2,
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            return;
        } else if constexpr (splitAxis == BN2) {
            INVOKE_FAG_GENERAL_S1S2_BN2_REGBASE_IMPL_FP32(
                float, float, float, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle,
                BN2, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            return;
        }
    #endif
 

    #if (ORIG_DTYPE_QUERY == DT_HIFLOAT8)
        if constexpr (outDType == FLOAT16_PRECISION) {
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_HIFP8(
                hifloat8_t, float, half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2GS1S2,
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            return;
        } else if constexpr (outDType == BFLOAT16) {
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_HIFP8(
                hifloat8_t, float, bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk, deterType, isNEqual, isDNoEqual, isRope, fp8OpenTscm, isTndSwizzle, BN2GS1S2,
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType), DTemplateType(dTemplateType));
            return;
        }
    #endif
}
#endif // _FLASH_ATTENTION_SCORE_GRAD_ENTRY_REGBASE_H_