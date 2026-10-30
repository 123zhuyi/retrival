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
 * \file prompt_flash_attention_vector2.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_VECTOR2_H
#define PROMPT_FLASH_ATTENTION_VECTOR2_H

#include "../vf_flashupdate.h"
#include "../vf_row_invalid.h"

template <typename PFAT>
class PromptFlashAttentionNormalVector2 {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using O = typename PFAT::outputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T, PFAT::calcMode>::softmaxType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    __aicore__ inline PromptFlashAttentionNormalVector2() {};
    __aicore__ inline void Init(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse);
    __aicore__ inline void ProVector(LocalTensor<computeType>& tempBmm2Ub, LocalTensor<float>& softmaxMaxTmp,
        LocalTensor<float>& softmaxSumTmp, LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<computeType>& softmaxExpUb,
        TQue<QuePosition::VECOUT, 1>& softmaxLseQueue, const TaskParam& taskParam, const ConstParam& constParam,
        float dequantScale2);

protected:
    __aicore__ inline void Bmm1ResDoVecBmm2Compute(LocalTensor<computeType>& tempBmm2Ub, LocalTensor<float>& softmaxMaxTmp,
        LocalTensor<float>& softmaxSumTmp, LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<computeType>& softmaxExpUb,
        TQue<QuePosition::VECOUT, 1>& softmaxLseQueue, const TaskParam& taskParam, const ConstParam& constParam,
        float dequantScale2);
    __aicore__ inline void ProcessLastSouterLoopFinalRes(LocalTensor<computeType>& tempBmm2Ub,
        LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<computeType>& softmaxExpUb, LocalTensor<float>& softmaxMaxTmp,
        LocalTensor<float>& softmaxSumTmp, const TaskParam& taskParam, const ConstParam& constParam, float dequantScale2);
    __aicore__ inline void Bmm2UpdateDivNoTail(LocalTensor<computeType>& tempBmm2Ub, LocalTensor<mmOutputType>& bmm2ResUb,
        LocalTensor<float>& softmaxSumTmp, const TaskParam& taskParam, const ConstParam& constParam, float dequantScale2);
    __aicore__ inline void Bmm2FlashUpdate(LocalTensor<computeType>& tempBmm2Ub, LocalTensor<mmOutputType>& bmm2ResUb,
        LocalTensor<computeType>& softmaxExpUb, const TaskParam& taskParam, const ConstParam& constParam, float dequantScale2);
    __aicore__ inline void Bmm2FlashUpdateLast(LocalTensor<computeType>& tempBmm2Ub, LocalTensor<mmOutputType>& bmm2ResUb,
        LocalTensor<computeType>& softmaxExpUb, LocalTensor<float>& softmaxSumTmp, const TaskParam& taskParam,
        const ConstParam& constParam, float dequantScale2);
    __aicore__ inline void RowInvalid(LocalTensor<computeType>& tempBmm2Ub, LocalTensor<float>& softmaxMaxTmp,
        const TaskParam& taskParam, const ConstParam& constParam);
    __aicore__ inline void DataCopyTransposeOut(LocalTensor<computeType>& tempBmm2Ub, const TaskParam& taskParam,
        const ConstParam& constParam);
    __aicore__ inline void DataCopyTransposeOutForIFAMla(LocalTensor<computeType>& tempBmm2Ub, const TaskParam& taskParam,
        const ConstParam& constParam);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
        TQue<QuePosition::VECOUT, 1>& softmaxLseQueue, const TaskParam& taskParam, const ConstParam& constParam);
private:
    GlobalTensor<O> attentionOutGm;
    event_t attenOutCopyOut;
    GlobalTensor<float> softmaxLseGm;
    LocalTensor<float> lseUb;
    event_t tmpUbCopy;
};

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::Init(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse)
{
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
    attenOutCopyOut = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    tmpUbCopy = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::ProVector(LocalTensor<computeType>& tempBmm2Ub,
    LocalTensor<float>& softmaxMaxTmp, LocalTensor<float>& softmaxSumTmp, LocalTensor<mmOutputType>& bmm2ResUb,
    LocalTensor<computeType>& softmaxExpUb, TQue<QuePosition::VECOUT, 1>& softmaxLseQueue, const TaskParam& taskParam,
    const ConstParam& constParam, float dequantScale2)
{
    Bmm1ResDoVecBmm2Compute(tempBmm2Ub, softmaxMaxTmp, softmaxSumTmp, bmm2ResUb, softmaxExpUb, softmaxLseQueue,
        taskParam, constParam, dequantScale2);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::Bmm1ResDoVecBmm2Compute(
    LocalTensor<computeType>& tempBmm2Ub, LocalTensor<float>& softmaxMaxTmp, LocalTensor<float>& softmaxSumTmp,
    LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<computeType>& softmaxExpUb, TQue<QuePosition::VECOUT, 1>& softmaxLseQueue,
    const TaskParam& taskParam, const ConstParam& constParam, float dequantScale2)
{
    if (taskParam.isFirstInnerIter) {
        SetFlag<HardEvent::MTE3_V>(tmpUbCopy);
        WaitFlag<HardEvent::MTE3_V>(tmpUbCopy);
    }

    if (taskParam.isFirstInnerIter && !taskParam.isLastInnerIter) {
        int64_t vec2CalcSize = constParam.singleProcessSOuterSizeWhole * PFAT::vDSize;
        if constexpr (IsSameType<T, int8_t>::value) {
            AscendDequant(tempBmm2Ub, bmm2ResUb, dequantScale2, {constParam.singleProcessSOuterSizeWhole, PFAT::vDSize, PFAT::vDSize});
        } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
            Muls(tempBmm2Ub, bmm2ResUb, dequantScale2, vec2CalcSize);
        } else {
            DataCopy(tempBmm2Ub, bmm2ResUb, vec2CalcSize);
        }
    } else if (!taskParam.isFirstInnerIter && !taskParam.isLastInnerIter) { // mul + add
        Bmm2FlashUpdate(tempBmm2Ub, bmm2ResUb, softmaxExpUb, taskParam, constParam, dequantScale2);
    }

    if (taskParam.isLastInnerIter) {
        if (constParam.isSoftmaxLseEnable) {
            SoftmaxLseCopyOut(softmaxSumTmp, softmaxMaxTmp, softmaxLseQueue, taskParam, constParam);
        }
        ProcessLastSouterLoopFinalRes(tempBmm2Ub, bmm2ResUb, softmaxExpUb, softmaxMaxTmp, softmaxSumTmp, taskParam, constParam, dequantScale2);
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::SoftmaxLseCopyOut(
    LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp, TQue<QuePosition::VECOUT, 1>& softmaxLseQueue,
    const TaskParam& taskParam, const ConstParam& constParam)
{
    this->lseUb = softmaxLseQueue.template AllocTensor<float>();
    ComputeLseOutput_VF(this->lseUb, softmaxSumTmp, softmaxMaxTmp, taskParam.singleProcessSOuterSize);
    softmaxLseQueue.template EnQue(this->lseUb);
    softmaxLseQueue.DeQue<float>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = taskParam.singleProcessSOuterSize;
    intriParams1.srcStride = 0;
    if (PFAT::layout == PFALayout::TND) {
        intriParams1.dstStride = sizeof(float) * (constParam.headNumSize - 1);
    } else {
        if constexpr (PFAT::layout == PFALayout::BSH && PFAT::IFA_MLA) {
            intriParams1.dstStride = sizeof(float) * ((constParam.seqSize / constParam.gOfMla) - 1);
        } else {
            intriParams1.dstStride = 0;
        }
    }
    DataCopyPad(this->softmaxLseGm[taskParam.softmaxLseOffset], this->lseUb, intriParams1);
    softmaxLseQueue.FreeTensor(this->lseUb);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::ProcessLastSouterLoopFinalRes(
    LocalTensor<computeType>& tempBmm2Ub, LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<computeType>& softmaxExpUb,
    LocalTensor<float>& softmaxMaxTmp, LocalTensor<float>& softmaxSumTmp, const TaskParam& taskParam,
    const ConstParam& constParam, float dequantScale2)
{
    if (!taskParam.isFirstInnerIter) { // mul + add + div
        Bmm2FlashUpdateLast(tempBmm2Ub, bmm2ResUb, softmaxExpUb, softmaxSumTmp, taskParam, constParam, dequantScale2);
    } else { // div
        Bmm2UpdateDivNoTail(tempBmm2Ub, bmm2ResUb, softmaxSumTmp, taskParam, constParam, dequantScale2);
    }

    RowInvalid(tempBmm2Ub, softmaxMaxTmp, taskParam, constParam);
    if (taskParam.singleProcessSOuterSize == 0) {
        return;
    } else if constexpr (PFAT::IFA_MLA) {
        DataCopyTransposeOutForIFAMla(tempBmm2Ub, taskParam, constParam);
    } else {
        DataCopyTransposeOut(tempBmm2Ub, taskParam, constParam);
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::Bmm2UpdateDivNoTail(
    LocalTensor<computeType>& tempBmm2Ub, LocalTensor<mmOutputType>& bmm2ResUb,
    LocalTensor<float>& softmaxSumTmp, const TaskParam& taskParam,
    const ConstParam& constParam, float dequantScale2)
{
    FlashUpdateDiv<computeType, computeType, T, mmOutputType, PFAT::vDSize>(tempBmm2Ub, bmm2ResUb, softmaxSumTmp,
        taskParam.singleProcessSOuterSize, constParam.vHeadSize, dequantScale2);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::Bmm2FlashUpdate(
    LocalTensor<computeType>& tempBmm2Ub, LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<computeType>& softmaxExpUb,
    const TaskParam& taskParam, const ConstParam& constParam, float dequantScale2)
{
    FlashUpdate<computeType, computeType, T, mmOutputType, PFAT::vDSize>(tempBmm2Ub, bmm2ResUb, tempBmm2Ub, softmaxExpUb,
        taskParam.singleProcessSOuterSize, constParam.vHeadSize, dequantScale2);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::Bmm2FlashUpdateLast(
    LocalTensor<computeType>& tempBmm2Ub, LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<computeType>& softmaxExpUb,
    LocalTensor<float>& softmaxSumTmp, const TaskParam& taskParam, const ConstParam& constParam, float dequantScale2)
{
    FlashUpdateLast<computeType, computeType, T, mmOutputType, PFAT::vDSize>(tempBmm2Ub, bmm2ResUb, tempBmm2Ub, softmaxExpUb, softmaxSumTmp,
        taskParam.singleProcessSOuterSize, constParam.vHeadSize, dequantScale2);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::RowInvalid(LocalTensor<computeType>& tempBmm2Ub, 
    LocalTensor<float>& softmaxMaxTmp, const TaskParam& taskParam, const ConstParam& constParam)
{
    if constexpr (!PFAT::isHasAtten || PFAT::isBand) { // band模式没有自定义mask，无需额外处理行无效
        return;
    }
    if (!constParam.isRowInvalid || !taskParam.isLastInnerIter) {
        return;
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isRowInvalidNeedUpdate = false;
    for (uint32_t i = 0; i < taskParam.singleProcessSOuterSize; i++) {
        float maxValue = softmaxMaxTmp.GetValue(i);
        uint32_t checkValue = *(uint32_t*)&maxValue;
        if (checkValue == PFA_NEGATIVE_MIN_VAULE_FP32) {
            isRowInvalidNeedUpdate = true;
            break;
        }
    }
    if (isRowInvalidNeedUpdate) {
        RowInvalidUpdate<computeType, PFAT::vDSize>(tempBmm2Ub, softmaxMaxTmp, taskParam.singleProcessSOuterSize, constParam.vHeadSize);
    }
}

// 输出
template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::DataCopyTransposeOut(
    LocalTensor<computeType>& tempBmm2Ub, const TaskParam& taskParam, const ConstParam& constParam)
{
    // 当前默认按非量化高精度模式实现
    LocalTensor<O> finalResUb = tempBmm2Ub.template ReinterpretCast<O>();
    Cast(finalResUb, tempBmm2Ub, RoundMode::CAST_ROUND, tempBmm2Ub.GetSize());

    SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
    WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
    struct DataCopyExtParams dataCopyParams;
    dataCopyParams.srcStride = (PFAT::vDSize - constParam.vHeadSize) / constParam.outputTypeByteNum;
    dataCopyParams.blockLen = constParam.vHeadSize * sizeof(O);
    dataCopyParams.blockCount = taskParam.singleProcessSOuterSize;
    int64_t startAddr = taskParam.attentionOutOffset;
    if ((!constParam.isIFA && PFAT::layout == PFALayout::BSH) || constParam.isBSNDOut == 1 || PFAT::layout == PFALayout::TND) {
        dataCopyParams.dstStride = (constParam.headNumSize - 1) * constParam.vHeadSize * sizeof(O);
    } else {
        dataCopyParams.dstStride = 0;
    }
    DataCopyPad(attentionOutGm[startAddr], finalResUb, dataCopyParams);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector2<PFAT>::DataCopyTransposeOutForIFAMla(
    LocalTensor<computeType>& tempBmm2Ub, const TaskParam& taskParam, const ConstParam& constParam)
{
    // 当前默认按非量化高精度模式实现
    LocalTensor<O> finalResUb = tempBmm2Ub.template ReinterpretCast<O>();
    Cast(finalResUb, tempBmm2Ub, RoundMode::CAST_ROUND, tempBmm2Ub.GetSize());

    SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
    WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
    struct DataCopyParams dataCopyParams;
    dataCopyParams.srcStride = (PFAT::vDSize - constParam.vHeadSize) / constParam.outputTypeByteNum;
    dataCopyParams.blockLen = constParam.vHeadSize * sizeof(O);
    dataCopyParams.blockCount = taskParam.singleProcessSOuterSize;
    dataCopyParams.dstStride = 0;
    int64_t startAddr = taskParam.attentionOutOffset;

    if constexpr (!(PFAT::layout == PFALayout::BNSD)) {
        DataCopyPad(attentionOutGm[startAddr], finalResUb, dataCopyParams);
    } else if (taskParam.nextTokensOfMlaPerBatch >= 0 &&
        taskParam.actualSeqLengthPerBatch / constParam.gOfMla == taskParam.actualSeqLengthOfMlaPerBatch){
        DataCopyPad(attentionOutGm[startAddr], finalResUb, dataCopyParams);
    } else {
        int64_t s1OfMla = taskParam.actualSeqLengthPerBatch / constParam.gOfMla;   // q_shape里的s1
        int64_t firstS1Start = taskParam.sOuterOffset % s1OfMla;                   // 当前基本块第一块s1的起点
        int64_t firstS1Size = firstS1Start == 0 ? 0 : s1OfMla - firstS1Start;      // 当前基本块第一块s1剩余大小
        int64_t invalidRowOfS1 = s1OfMla - taskParam.actualSeqLengthOfMlaPerBatch; // actual seq导致的无效行数
        int64_t startOffset = 0;             // gm起始位置偏移
        int64_t startOffsetOfUb = 0;         // ub起始位置偏移
        int64_t nextTokenOffset = 0;         // nexttoken导致行无效在gm上的偏移
        int64_t nextTokenOffsetOfUb = 0;     // nexttoken导致行无效在ub上的偏移
        int64_t nextTokenCount = 0;          // nexttoken导致行无效的行数
        int64_t nextTokenCountOfFirstS1 = 0; // 在第一块s1上nexttoken导致行无效的行数
        if (taskParam.nextTokensOfMlaPerBatch < 0) {
            nextTokenCount = -taskParam.nextTokensOfMlaPerBatch;
            nextTokenOffset = nextTokenCount * constParam.vHeadSize;
            nextTokenOffsetOfUb = nextTokenCount * PFAT::vDSize;
            if (nextTokenCount > firstS1Start) {
                nextTokenCountOfFirstS1 = nextTokenCount - firstS1Start;
                startOffset = nextTokenCountOfFirstS1 * constParam.vHeadSize;
                startOffsetOfUb = nextTokenCountOfFirstS1 * PFAT::vDSize;
            }
        }
        int64_t tmpBlockCount = firstS1Size < taskParam.singleProcessSOuterSize ?
            firstS1Size : taskParam.singleProcessSOuterSize;
        // 下面DataCopyPad需要注意blockCount不能为0
        if (tmpBlockCount > invalidRowOfS1) {
            dataCopyParams.blockCount = static_cast<uint16_t>(tmpBlockCount - invalidRowOfS1 - nextTokenCountOfFirstS1);
            DataCopyPad(attentionOutGm[startAddr + startOffset], finalResUb[startOffsetOfUb], dataCopyParams);
        }
        startOffset = firstS1Size * constParam.vHeadSize;
        startOffsetOfUb = firstS1Size * PFAT::vDSize;
        dataCopyParams.blockCount = static_cast<uint16_t>(taskParam.actualSeqLengthOfMlaPerBatch - nextTokenCount);
        for (int64_t i = 1; i * s1OfMla + firstS1Size <= taskParam.singleProcessSOuterSize; i++) {
            DataCopyPad(attentionOutGm[startAddr + startOffset + nextTokenOffset],
                finalResUb[startOffsetOfUb + nextTokenOffsetOfUb], dataCopyParams);
            startOffset += s1OfMla * constParam.vHeadSize;
            startOffsetOfUb += s1OfMla * PFAT::vDSize;
        }
        if (taskParam.singleProcessSOuterSize - firstS1Size > 0) {
            tmpBlockCount = (taskParam.singleProcessSOuterSize - firstS1Size) % s1OfMla;
            tmpBlockCount = tmpBlockCount < taskParam.actualSeqLengthOfMlaPerBatch ?
                tmpBlockCount : taskParam.actualSeqLengthOfMlaPerBatch;
            tmpBlockCount -= nextTokenCount;
            if (tmpBlockCount > 0) {
                dataCopyParams.blockCount = static_cast<uint16_t>(tmpBlockCount);
                DataCopyPad(attentionOutGm[startAddr + startOffset + nextTokenOffset],
                    finalResUb[startOffsetOfUb + nextTokenOffsetOfUb], dataCopyParams);
            }
        }
    }
}
#endif  // PROMPT_FLASH_ATTENTION_VECTOR2_H