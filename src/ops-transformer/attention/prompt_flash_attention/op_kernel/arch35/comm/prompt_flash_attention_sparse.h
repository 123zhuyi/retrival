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
 * \file prompt_flash_attention_sparse.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_SPARSE_H
#define PROMPT_FLASH_ATTENTION_SPARSE_H

#include "prompt_flash_attention_comm.h"

template<typename PFAT>
__aicore__ inline void GetSparseParam(RunParam &runParam, const ConstParam &constParam)
{
    runParam.preTokensPerBatch = constParam.preTokens;
    runParam.nextTokensPerBatch = constParam.nextTokens;
    if (constParam.attentionMaskType == 3) {
        runParam.preTokensPerBatch = SPARSE_MODE_INT_MAX;
        if constexpr (!PFAT::IFA_MLA) {
            runParam.nextTokensPerBatch = runParam.actualSeqLengthKVPerBatch - runParam.actualSeqLengthPerBatch;
        } else if constexpr (PFAT::layout == PFALayout::BNSD) {
            runParam.nextTokensPerBatch = SPARSE_MODE_INT_MAX;
            runParam.nextTokensOfMlaPerBatch = runParam.actualSeqLengthKVPerBatch - runParam.actualSeqLengthOfMlaPerBatch;
        } else {
            runParam.nextTokensOfMlaPerBatch = runParam.actualSeqLengthKVPerBatch - runParam.actualSeqLengthOfMlaPerBatch;
            runParam.nextTokensPerBatch = runParam.nextTokensOfMlaPerBatch * constParam.gOfMla;
        }
    }
    if constexpr (PFAT::isBand) {
        runParam.preTokensPerBatch = constParam.preTokens - runParam.actualSeqLengthKVPerBatch +
            runParam.actualSeqLengthPerBatch;
        runParam.nextTokensPerBatch = constParam.nextTokens + runParam.actualSeqLengthKVPerBatch -
            runParam.actualSeqLengthPerBatch;
    }
}

template<typename PFAT> 
__aicore__ inline void CalAttenMasktCoreOffset(RunParam &runParam, const ConstParam &constParam,
int32_t sIdx, int64_t sOuterOffset)
{
    uint64_t attenMaskBatchOffset = 0;
    // 是否为多batch
    if (constParam.isIFA) {
        attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)constParam.attentionMaskStride * (uint64_t)constParam.maskQsSize;
        runParam.attenMaskCoreOffset = (uint64_t)runParam.kvLeftPaddingSize + attenMaskBatchOffset;
    } else {
        if (constParam.attenMaskBatch != 1) {
            attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)constParam.attentionMaskStride * (uint64_t)constParam.maskQsSize;
        }
        runParam.attenMaskCoreOffset = (uint64_t)(sOuterOffset + runParam.queryLeftPaddingSize) * \
            (uint64_t)constParam.attentionMaskStride + (uint64_t)runParam.kvLeftPaddingSize + attenMaskBatchOffset;
    }
}

template<typename PFAT>
__aicore__ inline uint64_t ComputeAttenMaskOffset(RunParam &runParam, const ConstParam &constParam,
    int64_t sInnerOffsetDataSize)
{
    uint64_t attenMaskOffset = 0;
    // 2:leftUp mode of sparseMode, 3:rightdown mode of sparseMode, 4:band mode of sparseMode
    if (constParam.attentionMaskType == 2 || constParam.attentionMaskType == 3 || constParam.attentionMaskType == 4) {
        int64_t delta = 0;
        if (constParam.attentionMaskType == 2) {
            delta = runParam.sOuterOffset - sInnerOffsetDataSize + constParam.nextTokens;
        } else if constexpr (!PFAT::IFA_MLA) {
            delta = runParam.sOuterOffset - sInnerOffsetDataSize + runParam.nextTokensPerBatch;
        } else if constexpr (PFAT::layout == PFALayout::BNSD){ // mla BNSD场景下要用到souter方向全部行的mask，souter方向不做偏移
            delta = runParam.nextTokensOfMlaPerBatch - sInnerOffsetDataSize;
        } else {
            delta = runParam.sOuterOffset / constParam.gOfMla - sInnerOffsetDataSize +
                runParam.nextTokensOfMlaPerBatch;
        }

        if (delta < 0) {
            attenMaskOffset = ((int32_t)constParam.singleProcessSOuterSizeWhole + delta) > 0 ?
                (-delta) : constParam.singleProcessSOuterSizeWhole;
        } else {
            attenMaskOffset = (((int32_t)constParam.singleProcessSInnerSize - delta) > 0 ?
                delta : constParam.singleProcessSInnerSize) * constParam.attentionMaskStride;
        }
    } else {
        attenMaskOffset = runParam.attenMaskCoreOffset + (uint64_t)sInnerOffsetDataSize;
    }
    return attenMaskOffset;
}

template<typename PFAT>
__aicore__ inline uint64_t ComputeAttenMaskOffsetPre(RunParam &runParam, const ConstParam &constParam,
    int64_t sInnerOffsetDataSize)
{                                            
    if (constParam.attentionMaskType == 0 || constParam.attentionMaskType == 1) {
        return 0;
    }
    uint64_t attenMaskOffsetPre = 0;  
    int64_t delta = runParam.sOuterOffset - sInnerOffsetDataSize - runParam.preTokensPerBatch - 1;
    if (delta < 0) {
        attenMaskOffsetPre = ((int32_t)constParam.singleProcessSOuterSizeWhole + delta) > 0 ?
            (-delta) : constParam.singleProcessSOuterSizeWhole;
    } else {
        attenMaskOffsetPre = (((int32_t)constParam.singleProcessSInnerSize - delta) > 0 ?
            delta : constParam.singleProcessSInnerSize) * constParam.attentionMaskStride;
    }
    return attenMaskOffsetPre;
}

template<typename PFAT>
__aicore__ inline void CalPseShiftCoreOffset(RunParam &runParam, const ConstParam &constParam,
int32_t sIdx, int64_t sOuterOffset)
{
    uint64_t pseShiftBatchOffset = 0;
    if (constParam.isIFA) {
        // 是否为多batch
        if (constParam.pseShiftBatch != 1) {
            pseShiftBatchOffset = (uint64_t)sIdx * (uint64_t)constParam.headNumSize * (uint64_t)constParam.gOfMla *
                (uint64_t)constParam.pseShiftS1Size * (uint64_t)constParam.pseShiftS2Size;
        }
        // 多个N
        runParam.pseShiftCoreOffset = pseShiftBatchOffset + (uint64_t)runParam.batchNOffset * (uint64_t)constParam.gOfMla *
            (uint64_t)constParam.pseShiftS1Size * (uint64_t)constParam.pseShiftS2Size +
            (uint64_t)(sOuterOffset + runParam.queryLeftPaddingSize) * (uint64_t)constParam.pseShiftS2Size +
            (uint64_t)runParam.kvLeftPaddingSize;
    } else {
        // 是否为多batch
        if (constParam.pseShiftBatch != 1) {
            pseShiftBatchOffset = (uint64_t)sIdx * (uint64_t)constParam.headNumSize *
                (uint64_t)constParam.pseShiftS1Size * (uint64_t)constParam.pseShiftS2Size;
        }
        // 多个N
        runParam.pseShiftCoreOffset = pseShiftBatchOffset + (uint64_t)runParam.batchNOffset *
            (uint64_t)constParam.pseShiftS1Size * (uint64_t)constParam.pseShiftS2Size +
            (uint64_t)(sOuterOffset + runParam.queryLeftPaddingSize) * (uint64_t)constParam.pseShiftS2Size +
            (uint64_t)runParam.kvLeftPaddingSize;
    }
}

template<typename PFAT>
__aicore__ inline uint64_t ComputePseShiftOffset(RunParam &runParam, int64_t sInnerOffsetDataSize)
{
    return (runParam.pseShiftCoreOffset + (uint64_t)sInnerOffsetDataSize);
}

#endif  // PROMPT_FLASH_ATTENTION_SPARSE_H