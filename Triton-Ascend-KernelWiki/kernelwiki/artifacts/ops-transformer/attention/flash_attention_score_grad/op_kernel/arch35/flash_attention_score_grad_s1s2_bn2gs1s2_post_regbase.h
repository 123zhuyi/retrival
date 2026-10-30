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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#define FAG_POST_CLASS_TEMPLATE                                                                                             \
    template <typename T1, typename T2, typename OUTDTYPE=T1, const uint8_t SPLIT_AXIS = 0, const bool IS_ROPE = false, const uint8_t DETER_SPARSE_TYPE = 0, const bool IS_TND = 0, const bool IS_TND_SWIZZLE = 0> 
#define FAG_POST_FUNCTION_TEMPLATE                                                                                          \
    template <typename T1, typename T2, typename OUTDTYPE, const uint8_t SPLIT_AXIS, bool IS_ROPE, const uint8_t DETER_SPARSE_TYPE, const bool IS_TND, const bool IS_TND_SWIZZLE>
#define FAG_POST_FUNCTION_PARAMS_TEMPLATE T1, T2, OUTDTYPE, SPLIT_AXIS, IS_ROPE, DETER_SPARSE_TYPE, IS_TND, IS_TND_SWIZZLE

FAG_POST_CLASS_TEMPLATE 
class FlashAttentionScoreGradS1S2BNGS1S2PostRegbase {
public:
    __aicore__ inline FlashAttentionScoreGradS1S2BNGS1S2PostRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dqRope,
                                __gm__ uint8_t *dkRope,__gm__ uint8_t *workspace,
                                FagTilingType ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBNS2Deter();

    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    uint32_t REGBASE_POST_BASE = 128 * 128;
    uint32_t ROPE_DIM = 64;
    uint32_t VALUE_DIM = 128;
    uint32_t POST_S_BASE = 96;
    TPipe *pipe;
    FagTilingType tilingData;
    TQue<QuePosition::VECIN, 1> inQueuePing;
    TQue<QuePosition::VECOUT, 1> outQueuePing;
    TQue<QuePosition::VECIN, 1> inQueuePong;
    TQue<QuePosition::VECOUT, 1> outQueuePong;
    GlobalTensor<OUTDTYPE> dqkv[3];
    GlobalTensor<OUTDTYPE> dqkRope[2];
    GlobalTensor<float> dqkvWorkspace[3];
    GlobalTensor<float> deterGm[2];
    uint32_t vBlockIdx;
    uint64_t loop;
    uint64_t inputTotalSize;
    uint64_t qPostTailNum;
};

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dqRope,
    __gm__ uint8_t *dkRope, __gm__ uint8_t *workspace,
    FagTilingType ordTilingData, TPipe *pipe_in)
{
    vBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqkv[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dqkv[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dqkv[2].SetGlobalBuffer((__gm__ OUTDTYPE *)dv);

    dqkRope[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dqRope);
    dqkRope[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dkRope);

    loop = tilingData->postTilingData.qPostBlockFactor;
    inputTotalSize = tilingData->postTilingData.qPostBlockTotal;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;

    dqkvWorkspace[0].SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    dqkvWorkspace[1].SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
    dqkvWorkspace[2].SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));

    if constexpr (IS_ROPE) {
        REGBASE_POST_BASE = POST_S_BASE * (ROPE_DIM + VALUE_DIM);
    }

    pipe->InitBuffer(inQueuePing, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePing, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));
    pipe->InitBuffer(inQueuePong, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePong, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));
}

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    if (g_coreType != AIV) {
        return;
    }
    for (int qkvIdx = 0; qkvIdx < 3; qkvIdx++) {
        if (qkvIdx == 1) {
            loop = tilingData->postTilingData.kPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.kPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.kPostTailNum;
        } else if (qkvIdx == 2) {
            loop = tilingData->postTilingData.vPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.vPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.vPostTailNum;
        }
        uint64_t blockCore = loop * REGBASE_POST_BASE;
        uint64_t begin = vBlockIdx * blockCore;
        uint64_t end = begin + blockCore;

        if (end > inputTotalSize) {
            end = inputTotalSize;
        }

        uint64_t dqPingGmOffset = vBlockIdx * loop * POST_S_BASE * VALUE_DIM;
        for (uint64_t pingIdx = begin; pingIdx < end; pingIdx = pingIdx + (REGBASE_POST_BASE << 1)) {
            LocalTensor<float> vecInPing = inQueuePing.AllocTensor<float>();
            uint64_t pongIdx = pingIdx + REGBASE_POST_BASE;
            uint64_t pingSize = pongIdx < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
            uint64_t dqPongGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE; // 96*128
            uint64_t seqPingSize = pingSize / (ROPE_DIM + VALUE_DIM);
            DataCopy(vecInPing, dqkvWorkspace[qkvIdx][pingIdx], (pingSize + 7) >> 3 << 3);
            inQueuePing.EnQue(vecInPing);
            inQueuePing.DeQue<float>();
            if (qkvIdx < 2) {
                Muls(vecInPing, vecInPing, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, pingSize);
            }
            LocalTensor<OUTDTYPE> vecOutPing = outQueuePing.AllocTensor<OUTDTYPE>();
            Cast(vecOutPing, vecInPing, RoundMode::CAST_ROUND, pingSize);
            uint64_t pongSize;
            uint64_t seqPongSize;
            LocalTensor<float> vecInPong;
            bool neeedPong = loop > 1 && pongIdx < end;
            if (neeedPong) {
                vecInPong = inQueuePong.AllocTensor<float>();
                pongSize = pongIdx + REGBASE_POST_BASE < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
                seqPongSize = pongSize / (ROPE_DIM + VALUE_DIM);
                DataCopy(vecInPong, dqkvWorkspace[qkvIdx][pongIdx], (pongSize + 7) >> 3 << 3);
            }
            outQueuePing.EnQue(vecOutPing);
            outQueuePing.DeQue<OUTDTYPE>();
            inQueuePing.FreeTensor(vecInPing);
            if constexpr (IS_ROPE) {
                if (qkvIdx == 2) {
                    DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, pingSize);
                }  else {
                    // 96 * (128 + 64)基本块
                    DataCopyParams dataCopyParams;
                    dataCopyParams.blockCount = seqPingSize;
                    // 128 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.blockLen = 4 * sizeof(OUTDTYPE);
                    // 64 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                    DataCopy(dqkv[qkvIdx][dqPingGmOffset], vecOutPing, dataCopyParams);

                    dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                    dataCopyParams.srcStride = 4 * sizeof(OUTDTYPE);
                    DataCopy(dqkRope[qkvIdx][dqPingGmOffset / 2], vecOutPing[128], dataCopyParams);
                }
                dqPingGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE;
            } else {
                DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, (pingSize + 15) >> 4 << 4);
            }
            outQueuePing.FreeTensor(vecOutPing);
            LocalTensor<OUTDTYPE> vecOutPong;
            if (neeedPong) {
                vecOutPong = outQueuePong.AllocTensor<OUTDTYPE>();
                inQueuePong.EnQue(vecInPong);
                inQueuePong.DeQue<float>();
                if (qkvIdx < 2) {
                    Muls(vecInPong, vecInPong, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, pongSize);
                }
                Cast(vecOutPong, vecInPong, RoundMode::CAST_ROUND, pongSize);
                outQueuePong.EnQue(vecOutPong);
                outQueuePong.DeQue<OUTDTYPE>();
                inQueuePong.FreeTensor(vecInPong);
                if constexpr (IS_ROPE) {
                    if (qkvIdx == 2) {
                        DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, pongSize);
                    }  else {
                        // 96 * (128 + 64)基本块
                        DataCopyParams dataCopyParams;
                        dataCopyParams.blockCount = seqPongSize;
                        // 128 * sizeof(OUTDTYPE) / 32B
                        dataCopyParams.blockLen = 4 * sizeof(OUTDTYPE);
                        // 64 * sizeof(OUTDTYPE) / 32B
                        dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                        DataCopy(dqkv[qkvIdx][dqPongGmOffset], vecOutPong, dataCopyParams);
    
                        dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                        dataCopyParams.srcStride = 4 * sizeof(OUTDTYPE);
                        DataCopy(dqkRope[qkvIdx][dqPongGmOffset / 2], vecOutPong[128], dataCopyParams);
                    }
                    dqPingGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE;
                } else {
                    DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, (pongSize + 15) >> 4 << 4);
                }
                outQueuePong.FreeTensor(vecOutPong);
            }
        }

        // bn2s2 dk dv no need to do muls and cast
        if constexpr (SPLIT_AXIS == 5) {
            break;
        }
    }
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_