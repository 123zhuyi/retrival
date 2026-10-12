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
 * \file fag_flag_data.h
 * \brief
 */

#ifndef FAG_FLAG_DATA_H
#define FAG_FLAG_DATA_H

#include "lib/matmul_intf.h"

using namespace commondef;

namespace AscendC
{
// 8字节以下不会影响通信开销, 且需要4字节对齐
struct FagTscmFlagData {
    uint64_t rightMatrixEncodingTableIdx : 5;
    uint64_t kvNeedCopy : 1;
    uint64_t leftMatrixEncodingTableIdx : 5;
    // preload params
    uint64_t nextAddr : 42;
    uint64_t offsetSign : 1;
    uint64_t nextMorN : 8;
    uint64_t copyCurrent : 1;
    uint64_t copyNext : 1;
};

// Rope专用结构体，多发一次消息
struct FagTscmRopeFlagData {
    FagTscmFlagData flagCommon{0};
    uint64_t mmIdx : 3;
    uint64_t aRopeAddr;
    uint64_t bRopeAddr;
};

// ================================= L1 =================================
static constexpr AscendC::TQueConfig TSCM_CONFIG = {.nd2nz = false,
                                                    .nz2nd = false,
                                                    .scmBlockGroup = true,
                                                    .bufferLen = 0,
                                                    .bufferNumber = 1,
                                                    .consumerSize = 0,
                                                    .consumer = {},
                                                    .enableStaticEvtId = true};
// 定义一个L1的buffer的全局变量，不同mm对象之间可以共用
struct GlobalTscmArrayStatic {
    __aicore__ inline GlobalTscmArrayStatic(){};
#ifndef __CCE_KT_TEST__
    AscendC::TSCM<AscendC::TPosition::GM, 1, &TSCM_CONFIG> localQue;
#else
    AscendC::TSCM<AscendC::TPosition::GM, 1, 0x4> localQue;
#endif
    TBuffAddr srcAddr;
};
__BLOCK_LOCAL__ __inline__ GlobalTscmArrayStatic *gTscmArray;

#define GET_Q_DX_ENCODING_TABLE_IDX(MM_IDX, PINGPONG_IDX) ((5 * MM_IDX) + (PINGPONG_IDX << 1))
#define GET_K_V_ENCODING_TABLE_IDX(MM_IDX, KV_PINGPONG_IDX) ((5 * MM_IDX) + (KV_PINGPONG_IDX << 1) + 1)
constexpr uint8_t MM1_ENCODING_TABLE[10] = {0, 1, 2, 0, 7, 3, 4, 5, 6, 8};
constexpr uint8_t MM1_NEXT_ENCODING_TABLE[9] = {2, 0, 7, 5, 0, 8, 0, 0, 3}; // preload need next tscm idx
constexpr uint8_t MM2_ENCODING_TABLE[25] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 3, 0, 5, 0, 8, 0, 0, 2, 0, 7};

template <typename T, S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          const uint8_t SPLIT_AXIS = 0, const bool IS_DETER_OLD = 0, const bool IS_TND = 0, const bool IS_ROPE = 0, const bool FP8_OPEN_TSCM = 0>
__aicore__ inline void InitTSCMBuffer(TPipe *pipeBase, GlobalTscmArrayStatic *tscmArray)
{
    if ASCEND_IS_AIC {
        if constexpr (IS_TSCM_REUSE((uint32_t)dTemplateType, T, IS_DETER_OLD, FP8_OPEN_TSCM)) {
            pipeBase->InitBuffer(tscmArray[0].localQue, 1,
                                 (uint32_t)s1TemplateType * (uint32_t)dTemplateType * sizeof(T)); // dx ping
            pipeBase->InitBuffer(tscmArray[1].localQue, 1,
                                 (uint32_t)s2TemplateType * (uint32_t)dTemplateType * sizeof(T)); // v ping
            pipeBase->InitBuffer(tscmArray[2].localQue, 1,
                                 (uint32_t)s1TemplateType * (uint32_t)dTemplateType * sizeof(T)); // dx pong
            pipeBase->InitBuffer(tscmArray[3].localQue, 1,
                                 (uint32_t)s1TemplateType * (uint32_t)dTemplateType * sizeof(T)); // q ping
            pipeBase->InitBuffer(tscmArray[4].localQue, 1,
                                 (uint32_t)s2TemplateType * (uint32_t)dTemplateType * sizeof(T)); // k ping
            pipeBase->InitBuffer(tscmArray[5].localQue, 1,
                                 (uint32_t)s1TemplateType * (uint32_t)dTemplateType * sizeof(T)); // q pong
            pipeBase->InitBuffer(tscmArray[6].localQue, 1,
                                 (uint32_t)s2TemplateType * (uint32_t)dTemplateType * sizeof(T)); // k pong
            if constexpr (IS_TSCM_PRELOAD_ROPE((uint32_t)dTemplateType, T, SPLIT_AXIS, IS_DETER_OLD, IS_TND, FP8_OPEN_TSCM, IS_ROPE)) {
                if (IS_DETER_OLD && IS_ROPE) {
                    pipeBase->InitBuffer(tscmArray[7].localQue, 1,
                            (uint32_t)s1TemplateType * ((uint32_t)(dTemplateType) / 3 << 1) * sizeof(T)); // dx next
                } else {
                    pipeBase->InitBuffer(tscmArray[7].localQue, 1,
                            (uint32_t)s1TemplateType * (uint32_t)dTemplateType * sizeof(T)); // dx next
                }
                pipeBase->InitBuffer(tscmArray[8].localQue, 1,
                                     (uint32_t)s1TemplateType * (uint32_t)dTemplateType * sizeof(T)); // q next
            }
        }
    }
}

// ================================= L0C =================================
constexpr TQueConfig CO1_CONFIG = {
    .nd2nz = false,
    .nz2nd = false,
    .scmBlockGroup = false,
    .bufferLen = 0,
    .bufferNumber = 1,
};

struct GlobalL0CArrayStatic {
    __aicore__ inline GlobalL0CArrayStatic(){};
#ifndef __CCE_KT_TEST__
    AscendC::TQue<QuePosition::CO1, 1, &CO1_CONFIG> l0cQue;
#else
    AscendC::TQue<QuePosition::CO1, 1, 0x4> l0cQue;
#endif
    TBuffAddr srcAddr;
};
__BLOCK_LOCAL__ __inline__ GlobalL0CArrayStatic *gL0cArray;

template <typename T1, S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          const bool IS_DETER_OLD = 0, const bool IS_TND = 0>
__aicore__ inline void InitL0CBuffer(AscendC::TPipe *pipeBase, GlobalL0CArrayStatic *l0cArray)
{
    if ASCEND_IS_AIC {
        if (IS_L0C_REUSE((uint32_t)s1TemplateType, (uint32_t)s2TemplateType, (uint32_t)dTemplateType, IS_DETER_OLD, T1,
                         IS_TND)) {
            constexpr uint32_t L0C_BUF_NUM =
                GET_L0C_BUF_NUM((uint32_t)s1TemplateType, (uint32_t)s2TemplateType, (uint32_t)dTemplateType);
            constexpr uint32_t MM1_MM2_MM3_MAX_L0C_SIZE = GET_MM1_MM2_MM3_MAX_L0C_SIZE(
                (uint32_t)s1TemplateType, (uint32_t)s2TemplateType, (uint32_t)dTemplateType);
            for (uint32_t i = 0; i < L0C_BUF_NUM - DK_DV_L0C_BUF_NUM; i++) {
                pipeBase->InitBuffer(l0cArray[i].l0cQue, 1, MM1_MM2_MM3_MAX_L0C_SIZE); // mm1 or mm2 or dq
            }
            pipeBase->InitBuffer(l0cArray[L0C_BUF_NUM - DK_DV_L0C_BUF_NUM].l0cQue, 1,
                                 (uint32_t)s2TemplateType * (uint32_t)dTemplateType * sizeof(float)); // dk
            pipeBase->InitBuffer(l0cArray[L0C_BUF_NUM - 1].l0cQue, 1,
                                 (uint32_t)s2TemplateType * (uint32_t)dTemplateType * sizeof(float)); // dv
        }
    }
}

} // namespace matmul

#endif