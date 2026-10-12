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
 * \file prompt_flash_attention_vector1.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_VECTOR1_H
#define PROMPT_FLASH_ATTENTION_VECTOR1_H

#include "../vf_softmaxflashv3.h"
#include "../vf_softmaxflashv3_dn.h"
#include "../vf_row_invalid.h"
#include "adv_api/activation/softmax.h"

template <typename PFAT>
class PromptFlashAttentionNormalVector1 {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using U = typename PFAT::maskType;
    using computeType = typename PromptFlashAttentionTypeTraits<T, PFAT::calcMode>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    __aicore__ inline PromptFlashAttentionNormalVector1() {};
    __aicore__ inline void Init(__gm__ uint8_t* attenMask, __gm__ uint8_t* pseShift);
    __aicore__ inline void ProVector(LocalTensor<T>& tmpSoftmaxResUb, TQue<QuePosition::VECIN, 1>& maskQueue,
        TQue<QuePosition::VECIN, 1>& pseQueue, LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
        LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb, LocalTensor<uint8_t>& sfmaxTmpUb,
        const TaskParam& taskParam, const ConstParam& constParam, float dequantScale1, float quantScale1);

protected:
    __aicore__ inline void Bmm1VecInputCopyIn(TQue<QuePosition::VECIN, 1>& maskQueue,
        TQue<QuePosition::VECIN, 1>& pseQueue, const TaskParam& taskParam, const ConstParam& constParam);
    __aicore__ inline void MaskCopyIn(TQue<QuePosition::VECIN, 1>& maskQueue, uint32_t souterSize,
        const TaskParam& taskParam, const ConstParam& constParam);
    __aicore__ inline void PseCopyIn(TQue<QuePosition::VECIN, 1>& pseQueue, uint32_t souterSize,
        const TaskParam& taskParam, const ConstParam& constParam);
    __aicore__ inline void Res1VecCompute(LocalTensor<T>& tmpSoftmaxResUb, TQue<QuePosition::VECIN, 1>& maskQueue,
        TQue<QuePosition::VECIN, 1>& pseQueue, LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
        LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb, LocalTensor<uint8_t>& sfmaxTmpUb,
        const TaskParam& taskParam, const ConstParam& constParam, float dequantScale1, float quantScale1);
    __aicore__ inline void SoftmaxBasicFirstCompute(LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<mmOutputType>& mmResUb,
        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, LocalTensor<uint8_t>& sfmaxTmpUb,
        const LocalTensor<U>& mask, const LocalTensor<pseShiftType>& pse, uint32_t souterSize, SoftMaxShapeInfo& softmaxShapeInfo, const ConstParam& constParam, 
        float dequantScale1, float quantScale1);
    __aicore__ inline void SoftmaxBasicUpdateCompute(LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<mmOutputType>& mmResUb,
        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb,
        LocalTensor<uint8_t>& sfmaxTmpUb, const LocalTensor<U>& mask, const LocalTensor<pseShiftType>& pse, uint32_t souterSize,
        SoftMaxShapeInfo& softmaxShapeInfo, const ConstParam& constParam, float dequantScale1, float quantScale1);
    __aicore__ inline void SoftmaxFirstComputeDN(LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<computeType>& mmResUb,
        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
        SoftMaxShapeInfo& softmaxShapeInfo, const TaskParam& taskParam, const ConstParam& constParam);
    __aicore__ inline void SoftmaxUpdateComputeDN(LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<computeType>& mmResUb,
        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb,
        SoftMaxShapeInfo& softmaxShapeInfo, const TaskParam& taskParam, const ConstParam& constParam);
private:
    GlobalTensor<U> attenMaskGm;
    GlobalTensor<pseShiftType> pseShiftGm;
    LocalTensor<U> attenMaskUb;
    LocalTensor<pseShiftType> pseShiftUb;
    uint32_t negativeScalar = PFA_NEGATIVE_MIN_VAULE_FP32;
};

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::Init(__gm__ uint8_t* attenMask,
    __gm__ uint8_t* pseShift)
{
    attenMaskGm.SetGlobalBuffer((__gm__ U*)attenMask);
    pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType*)pseShift);
    if constexpr ((PFAT::calcMode != RunMode::HighPrecision) &&
        (IsSameType<T, half>::value || IsSameType<T, int8_t>::value)) {
        this->negativeScalar = PFA_NEGATIVE_MIN_VAULE_FP16;
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::ProVector(LocalTensor<T>& tmpSoftmaxResUb,
    TQue<QuePosition::VECIN, 1>& maskQueue, TQue<QuePosition::VECIN, 1>& pseQueue, LocalTensor<mmOutputType>& mmResUb,
    LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb,
    LocalTensor<uint8_t>& sfmaxTmpUb, const TaskParam& taskParam, const ConstParam& constParam, float dequantScale1, float quantScale1)
{
    if constexpr ((PFAT::isHasAtten) || (PFAT::isHasPse)) {
        Bmm1VecInputCopyIn(maskQueue, pseQueue, taskParam, constParam);
    }
    Res1VecCompute(tmpSoftmaxResUb, maskQueue, pseQueue, mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, sfmaxTmpUb,
        taskParam, constParam, dequantScale1, quantScale1);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::Bmm1VecInputCopyIn(TQue<QuePosition::VECIN, 1>& maskQueue,
    TQue<QuePosition::VECIN, 1>& pseQueue, const TaskParam& taskParam, const ConstParam& constParam)
{
    uint32_t souterSize = constParam.softmaxFlashTilingDataSrcM;
    // 优化尾块，softmax循环次数
    if (constParam.softmaxFlashTilingDataSrcK != taskParam.singleProcessSInnerSizeNow) {
        souterSize = constParam.softmaxFlashTilingDataSrcSize / taskParam.singleProcessSInnerSizeNow / 8 * 8; // 8对齐
        souterSize = ((souterSize > taskParam.singleProcessSOuterSize) || (souterSize == 0)) ? \
        taskParam.singleProcessSOuterSize : souterSize;
    }

    if constexpr (PFAT::isHasPse) {
        PseCopyIn(pseQueue, souterSize, taskParam, constParam);
    }
    if constexpr (PFAT::isHasAtten) {
        MaskCopyIn(maskQueue, souterSize, taskParam, constParam);
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::MaskCopyIn(TQue<QuePosition::VECIN, 1>& maskQueue,
    uint32_t souterSize, const TaskParam& taskParam, const ConstParam& constParam)
{
    uint64_t attenMaskOffsetNext = taskParam.attenMaskOffset;
    uint64_t attenMaskOffsetPre = taskParam.attenMaskOffsetPre;
    int64_t alignSInner = static_cast<int64_t>(taskParam.maskCopyInCol);
    int64_t unalignSInner = static_cast<int64_t>(taskParam.singleProcessSInnerBmmTail);
    uint32_t maskPadSize = taskParam.padSize;
    uint32_t neededSouterSize = souterSize; // 使用真实需要的Q_S，针对小Q_S优化
    if (constParam.seqSize < neededSouterSize) {
        neededSouterSize = constParam.seqSize;
    }

    size_t lenOfType = sizeof(U);;  // 每个数据的长度
    this->attenMaskUb = maskQueue.template AllocTensor<U>();

    DataCopyExtParams intriParams;
    intriParams.blockCount = neededSouterSize;  // 此处应该是非对齐
    intriParams.blockLen = alignSInner * lenOfType;
    intriParams.srcStride = (static_cast<int64_t>(constParam.attentionMaskStride) - alignSInner) * lenOfType;
    if (taskParam.isInnerTail) {
        intriParams.blockLen = unalignSInner * lenOfType;
        intriParams.srcStride = (static_cast<int64_t>(constParam.attentionMaskStride) - unalignSInner) * lenOfType;
    }
    intriParams.dstStride = 0;

    DataCopyPadExtParams<U> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 1;
    if (taskParam.isInnerTail) {
        padParams.rightPadding = maskPadSize;
    } else {
        padParams.rightPadding = 0;
    }
    if (constParam.isIFA) {
        intriParams.blockCount = 1;
        DataCopyPad(this->attenMaskUb, attenMaskGm[attenMaskOffsetNext], intriParams, padParams);
    } else if constexpr (!PFAT::IFA_MLA) {
        DataCopyPad(this->attenMaskUb, attenMaskGm[attenMaskOffsetNext], intriParams, padParams);
    } else if constexpr (!(PFAT::layout == PFALayout::BNSD)) {
        intriParams.blockCount = 1;
        DataCopyPad(this->attenMaskUb, attenMaskGm[attenMaskOffsetNext], intriParams, padParams);
    } else {
        int64_t s1OfMla = taskParam.actualSeqLengthPerBatch / constParam.gOfMla; // q_shape里的s1
        int64_t firstS1Start = taskParam.sOuterOffset % s1OfMla; // 当前基本块第一块s1的起点
        intriParams.blockCount = (s1OfMla - firstS1Start) < neededSouterSize ? 
            (s1OfMla - firstS1Start) : neededSouterSize;
        // 下面DataCopyPad需要注意blockLen不能为0
        DataCopyPad(this->attenMaskUb, attenMaskGm[attenMaskOffsetNext + firstS1Start *
            static_cast<int64_t>(constParam.attentionMaskStride)], intriParams, padParams);
        if (firstS1Start != 0 && (s1OfMla - firstS1Start) < neededSouterSize) {
            intriParams.blockCount = firstS1Start < (neededSouterSize - s1OfMla + firstS1Start) ? 
                firstS1Start : (neededSouterSize - s1OfMla + firstS1Start);
            DataCopyPad(this->attenMaskUb[(s1OfMla - firstS1Start) * alignSInner],
                attenMaskGm[attenMaskOffsetNext], intriParams, padParams);
        }
        for (int64_t i = 1; (i + 1) * s1OfMla <= neededSouterSize; i++) { // 复制前s1行
            DataCopy(this->attenMaskUb[s1OfMla * i * alignSInner], this->attenMaskUb, s1OfMla * alignSInner);
        }
        if (neededSouterSize > s1OfMla && neededSouterSize % s1OfMla != 0) {
            DataCopy(this->attenMaskUb[s1OfMla * (neededSouterSize / s1OfMla) * alignSInner], this->attenMaskUb,
                (neededSouterSize % s1OfMla) * alignSInner);
        }
    }
    if constexpr (PFAT::isBand) {
        DataCopyPad(this->attenMaskUb[PFAT::vsOuter * PFAT::sInner],
                    attenMaskGm[attenMaskOffsetPre], intriParams, padParams);
    }
    maskQueue.template EnQue<U>(this->attenMaskUb);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::PseCopyIn(TQue<QuePosition::VECIN, 1>& pseQueue,
    uint32_t souterSize, const TaskParam& taskParam, const ConstParam& constParam)
{
    uint64_t pseShiftOffset = taskParam.pseShiftOffset;
    int64_t alignSInner = static_cast<int64_t>(taskParam.pseShiftCopyInCol);
    int64_t unalignSInner = static_cast<int64_t>(taskParam.singleProcessSInnerBmmTail);
    uint32_t pseShiftPadSize = taskParam.pseShiftPadSize;
    // 使用真实需要的Q_S，针对小Q_S优化
    uint32_t neededSouterSize = souterSize;
    if (constParam.seqSize < neededSouterSize) {
        neededSouterSize = constParam.seqSize;
    }

    size_t lenOfType = sizeof(pseShiftType);  // 每个数据的长度
    this->pseShiftUb = pseQueue.template AllocTensor<pseShiftType>();
    this->pseShiftUb.SetSize(souterSize * alignSInner);

    DataCopyExtParams intriParams;
    intriParams.blockCount = neededSouterSize;  // 此处应该是非对齐
    intriParams.blockLen = alignSInner * lenOfType;
    intriParams.srcStride = (static_cast<int64_t>(constParam.pseShiftStride) - alignSInner) * lenOfType;
    if (taskParam.isInnerTail) {
        intriParams.blockLen = unalignSInner * lenOfType;
        intriParams.srcStride = (static_cast<int64_t>(constParam.pseShiftStride) - unalignSInner) * lenOfType;
    }
    intriParams.dstStride = 0;

    DataCopyPadExtParams<pseShiftType> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 1;
    if (taskParam.isInnerTail) {
        padParams.rightPadding = pseShiftPadSize;
    } else {
        padParams.rightPadding = 0;
    }
    DataCopyPad(this->pseShiftUb, pseShiftGm[pseShiftOffset], intriParams, padParams);
    pseQueue.template EnQue<pseShiftType>(this->pseShiftUb);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::Res1VecCompute(LocalTensor<T>& tmpSoftmaxResUb,
    TQue<QuePosition::VECIN, 1>& maskQueue, TQue<QuePosition::VECIN, 1>& pseQueue, LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb, LocalTensor<uint8_t>& sfmaxTmpUb,
    const TaskParam& taskParam, const ConstParam& constParam, float dequantScale1, float quantScale1)
{
    uint32_t mm1ResGmOffset = 0;
    uint32_t nextMm1ResGmOffset = 0;
    uint64_t attenMaskOffset = taskParam.attenMaskOffset;
    uint64_t attenMaskOffsetPre = taskParam.attenMaskOffsetPre;
    uint64_t pseShiftOffset = taskParam.pseShiftOffset;
    LocalTensor<float> softmaxMaxUbSub = softmaxMaxUb;
    LocalTensor<float> softmaxSumUbSub = softmaxSumUb;
    LocalTensor<computeType> softmaxExpUbSub = softmaxExpUb;

    uint32_t souterSize = taskParam.singleProcessSOuterSize;
    uint32_t computeSize = souterSize * taskParam.singleProcessSInnerSizeNow;

    mmResUb.SetSize(computeSize);

    // softmaxflash
    const uint32_t basicSoftmaxSinner = 64; // 64是softmax sinner基本块大小
    const uint32_t basicSoftmaxSouter = 8; // 8是softmax souter基本块大小
    const uint32_t basicSoftmaxK = 128; // 128是softmax k轴基本块大小

    SoftMaxShapeInfo softmaxShapeInfo;
    if (taskParam.isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(taskParam.singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(taskParam.singleProcessSInnerBmmTail)
        };
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(taskParam.singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(taskParam.singleProcessSInnerSizeNow)
        };
    }

    if constexpr (PFAT::isHasPse) {
        this->pseShiftUb = pseQueue.template DeQue<pseShiftType>();
    }
    if constexpr (PFAT::isHasAtten) {
        this->attenMaskUb = maskQueue.template DeQue<U>();
    }
    if constexpr (PFAT::useDN) {
        if (taskParam.isFirstInnerIter) {
            this->SoftmaxFirstComputeDN(tmpSoftmaxResUb, mmResUb, softmaxMaxUbSub, softmaxSumUbSub,
                softmaxShapeInfo, taskParam, constParam);
        } else {
            this->SoftmaxUpdateComputeDN(tmpSoftmaxResUb, mmResUb, softmaxMaxUbSub, softmaxSumUbSub, softmaxExpUbSub,
                softmaxShapeInfo, taskParam, constParam);
        }
    } else {
        if (taskParam.isFirstInnerIter) {
            this->SoftmaxBasicFirstCompute(tmpSoftmaxResUb, mmResUb, softmaxMaxUbSub, softmaxSumUbSub, sfmaxTmpUb,
                this->attenMaskUb, this->pseShiftUb, souterSize, softmaxShapeInfo, constParam, dequantScale1, quantScale1);
        } else {
            this->SoftmaxBasicUpdateCompute(tmpSoftmaxResUb, mmResUb, softmaxMaxUbSub, softmaxSumUbSub, softmaxExpUbSub,
                sfmaxTmpUb, this->attenMaskUb, this->pseShiftUb, souterSize, softmaxShapeInfo, constParam, dequantScale1, quantScale1);
        }
        if constexpr (PFAT::isHasPse) {
            pseQueue.FreeTensor(this->pseShiftUb);
        }
        if constexpr (PFAT::isHasAtten) {
            maskQueue.FreeTensor(this->attenMaskUb);
        }
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::SoftmaxFirstComputeDN(
    LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, SoftMaxShapeInfo& softmaxShapeInfo,
    const TaskParam& taskParam, const ConstParam& constParam)
{
    LocalTensor<computeType> null;
    auto scale = static_cast<computeType>(constParam.scaleValue);
    computeType minScalar;
    if constexpr (PFAT::calcMode == RunMode::HighPrecision || IsSameType<T, bfloat16_t>::value) {
        uint32_t tmp = 0xFF7FFFFF; // fp32最小值
        minScalar = *((float*)&tmp);
    } else {
        uint32_t tmp = 0xFBFF; // fp16最小值
        minScalar = *((half*)&tmp);
    }
    uint32_t srcMAlign = ((taskParam.cubeSOuterSize + 31) >> 5 << 5) / 2;

    ProcessVec1VfDn<computeType, T, false, PFAT::sInner>(tmpSoftmaxResUb, softmaxSumUb,
        softmaxMaxUb, mmResUb, null, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::SoftmaxUpdateComputeDN(
    LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb, SoftMaxShapeInfo& softmaxShapeInfo,
    const TaskParam& taskParam, const ConstParam& constParam)
{
    auto scale = static_cast<computeType>(constParam.scaleValue);
    computeType minScalar;
    if constexpr (PFAT::calcMode == RunMode::HighPrecision || IsSameType<T, bfloat16_t>::value) {
        uint32_t tmp = 0xFF7FFFFF; // fp32最小值
        minScalar = *((float*)&tmp);
    } else {
        uint32_t tmp = 0xFBFF; // fp16最小值
        minScalar = *((half*)&tmp);
    }
    uint32_t srcMAlign = ((taskParam.cubeSOuterSize + 31) >> 5 << 5) / 2;

    ProcessVec1VfDn<computeType, T, true, PFAT::sInner>(tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb,
        mmResUb, softmaxExpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::SoftmaxBasicFirstCompute(
    LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, LocalTensor<uint8_t>& sfmaxTmpUb, const LocalTensor<U>& mask, const LocalTensor<pseShiftType>& pse, uint32_t souterSize,
    SoftMaxShapeInfo& softmaxShapeInfo, const ConstParam& constParam, float dequantScale1, float quantScale1)
{
    LocalTensor<computeType> null;
    auto scale = static_cast<computeType>(constParam.scaleValue);
    computeType minScalar;
    if constexpr (PFAT::calcMode == RunMode::HighPrecision || IsSameType<T, bfloat16_t>::value) {
        uint32_t tmp = 0xFF7FFFFF; // fp32最小值
        minScalar = *((float*)&tmp);
    } else {
        uint32_t tmp = 0xFBFF; // fp16最小值
        minScalar = *((half*)&tmp);
    }
    uint32_t srcMAlign = (softmaxShapeInfo.srcM + constParam.typeByteNum - 1) / constParam.typeByteNum * constParam.typeByteNum;
    if (constParam.isIFA) {
        if (softmaxShapeInfo.oriSrcK <= 64) {
            // mode 2: 0 < oriSrcK <= 64
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 2, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK == 128) {
            // mode 1: oriSrcK = 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 1, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK > 64 && softmaxShapeInfo.oriSrcK < 128) {
            // mode 0: 64 < oriSrcK < 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 0, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else {
            // mode 3: 128 < oriSrcK <= 256
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 3, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        }
    } else {
        if (softmaxShapeInfo.oriSrcK <= 64) {
            // mode 2: 0 < oriSrcK <= 64
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 2, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK == 128) {
            // mode 1: oriSrcK = 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 1, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK > 64 && softmaxShapeInfo.oriSrcK < 128) {
            // mode 0: 64 < oriSrcK < 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 0, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else {
            // mode 3: 128 < oriSrcK <= 256
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, false, 3, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, null, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        }
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalVector1<PFAT>::SoftmaxBasicUpdateCompute(
    LocalTensor<T>& tmpSoftmaxResUb, LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb, LocalTensor<uint8_t>& sfmaxTmpUb,
    const LocalTensor<U>& mask, const LocalTensor<pseShiftType>& pse, uint32_t souterSize, SoftMaxShapeInfo& softmaxShapeInfo, const ConstParam& constParam, 
    float dequantScale1, float quantScale1)
{
    LocalTensor<computeType> null;
    auto scale = static_cast<computeType>(constParam.scaleValue);
    computeType minScalar;
    if constexpr (PFAT::calcMode == RunMode::HighPrecision || IsSameType<T, bfloat16_t>::value) {
        uint32_t tmp = 0xFF7FFFFF; // fp32最小值
        minScalar = *((float*)&tmp);
    } else {
        uint32_t tmp = 0xFBFF; // fp16最小值
        minScalar = *((half*)&tmp);
    }
    uint32_t srcMAlign = (softmaxShapeInfo.srcM + constParam.typeByteNum - 1) / constParam.typeByteNum * constParam.typeByteNum;
    if (constParam.isIFA) { 
        if (softmaxShapeInfo.oriSrcK <= 64) {
            // mode 2: 0 < oriSrcK <= 64
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 2, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK == 128) {
            // mode 1: oriSrcK = 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 1, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask,
                pse, sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK > 64 && softmaxShapeInfo.oriSrcK < 128) {
            // mode 0: 64 < oriSrcK < 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 0, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask,
                pse, sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else {
            // mode 3: 128 < oriSrcK <= 256
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 3, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                true, PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask,
                pse, sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        }
    } else {
        if (softmaxShapeInfo.oriSrcK <= 64) {
            // mode 2: 0 < oriSrcK <= 64
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 2, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask, pse,
                sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK == 128) {
            // mode 1: oriSrcK = 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 1, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask,
                pse, sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else if (softmaxShapeInfo.oriSrcK > 64 && softmaxShapeInfo.oriSrcK < 128) {
            // mode 0: 64 < oriSrcK < 128
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 0, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask,
                pse, sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        } else {
            // mode 3: 128 < oriSrcK <= 256
            SoftmaxFlashV3_VF<computeType, T, pseShiftType, true, 3, PFAT::isHasAtten, PFAT::isHasPse, PFAT::isBand,
                PFAT::IFA_MLA && !(PFAT::layout == PFALayout::BNSD), PFAT::isBmm2Concat, PFAT::vsOuter, PFAT::sInner, mmOutputType>(
                tmpSoftmaxResUb, softmaxSumUb, softmaxMaxUb, softmaxExpUb, mmResUb, softmaxSumUb, softmaxMaxUb, mask,
                pse, sfmaxTmpUb, srcMAlign, softmaxShapeInfo.oriSrcK, scale, minScalar, dequantScale1, quantScale1);
        }
    }
}

#endif  // PROMPT_FLASH_ATTENTION_VECTOR1_H