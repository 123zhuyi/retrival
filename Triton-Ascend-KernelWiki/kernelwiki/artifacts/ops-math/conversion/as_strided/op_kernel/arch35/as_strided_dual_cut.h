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
 * \file as_strided_dual_cut.h
 * \brief as_strided_dual_cut
 */

#ifndef ASCENDC_AS_STRIDED_DUAL_CUT_H_
#define ASCENDC_AS_STRIDED_DUAL_CUT_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include "as_strided.h"
#include "kernel_operator.h"

namespace AsStrided {
using namespace AscendC;

#define IS_CASE_NORMAL ((TILING_KEY == 200 || TILING_KEY == 201))
#define BLOCK_ELEMENT_NUM ((32 / sizeof(T)))

template <typename T, int TILING_KEY>
class KernelAsStridedDualCut {
public:
    __aicore__ inline KernelAsStridedDualCut()
    {}

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR outShape, GM_ADDR outStride, GM_ADDR output, const AsStridedTilingData* tiling)
    {
        storageOffset_ = 0;

        if (IS_CASE_NORMAL) {
            storageOffset_ = tiling->storageOffset;
            cutAxisNum_ = tiling->axisOutTotalFactor;
            outerAxisNum_ = tiling->outerAxisNum;
            innerAxisNum_ = tiling->innerAxisNum;
            ubSize_ = tiling->ubSize;

            cutAxisIdx01_ = tiling->nddmaTailLoop[0];
            cutAxisIdx02_ = tiling->nddmaTailLoop[1];
            cutAxisTail01_ = tiling->nddmaTailLoop[NDDMA_INDEX2];
            cutAxisTail02_ = tiling->nddmaTailLoop[NDDMA_INDEX3];

            blockNum_ = tiling->blockNum;
            loopsPerCore_ = tiling->loopsPerCore;
            loopsTailCore_ = tiling->innerAxisFactorTail;

            CopyArray(tiling->nddmaLoop, ubShape_, TILING_NDDMA_LEN);
            CopyArray(tiling->nddmaSrcStride, ubInStride_, TILING_NDDMA_LEN);
            CopyArray(tiling->nddmaDstStride, ubOutStride_, TILING_NDDMA_LEN);

            CopyArray(tiling->gmShape, gmShape_, TILING_ARRAY_LEN);
            CopyArray(tiling->gmInStride, gmInStride_, TILING_ARRAY_LEN);
            CopyArray(tiling->gmOutStride, gmOutStride_, TILING_ARRAY_LEN);

            finalDimAligned_ = (ubShape_[NDDMA_INDEX4] + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;

            ubDstStride_[0] = ubShape_[1] * ubShape_[NDDMA_INDEX2] * ubShape_[NDDMA_INDEX3] * finalDimAligned_;
            ubDstStride_[1] = ubShape_[NDDMA_INDEX2] * ubShape_[NDDMA_INDEX3] * finalDimAligned_;
            ubDstStride_[NDDMA_INDEX2] = ubShape_[NDDMA_INDEX3] * finalDimAligned_;
            ubDstStride_[NDDMA_INDEX3] = finalDimAligned_;
            ubDstStride_[NDDMA_INDEX4] = 1;
            for (int i = 0; i < TILING_NDDMA_LEN - innerAxisNum_; i++) {
                ubDstStride_[i] = 0;
            }
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)input + storageOffset_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
        pipe_.InitBuffer(inOutQueue, BUFFER_NUM, ubSize_ * sizeof(T));
    }

    __aicore__ inline void InitCopyParams()
    {
        dmaParam_ = {
            {
                // src stride
                {static_cast<uint64_t>(ubInStride_[NDDMA_INDEX4]), static_cast<uint64_t>(ubInStride_[NDDMA_INDEX3]),
                 static_cast<uint64_t>(ubInStride_[NDDMA_INDEX2]), static_cast<uint64_t>(ubInStride_[NDDMA_INDEX1]),
                 static_cast<uint64_t>(ubInStride_[NDDMA_INDEX0])},
                //  dst stride
                {ubDstStride_[NDDMA_INDEX4], ubDstStride_[NDDMA_INDEX3], ubDstStride_[NDDMA_INDEX2],
                 ubDstStride_[NDDMA_INDEX1], ubDstStride_[NDDMA_INDEX0]},
                //  loop size
                {ubShape_[NDDMA_INDEX4], ubShape_[NDDMA_INDEX3], ubShape_[NDDMA_INDEX2], ubShape_[NDDMA_INDEX1],
                 ubShape_[NDDMA_INDEX0]},
                {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}, // left pad
                {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}  //  right pad
            },
            0};
    }

    __aicore__ inline void Process()
    {
        InitCopyParams();
        if (this->cutAxisNum_ == 1) {
            SubProcessCutOne();
        } else {
            SubProcessCutTwo();
        }
    }

    __aicore__ inline void MoveOutWithLoop(
        LocalTensor<T>& outLocal, int64_t outAddr, LoopModeParams& loopMode, DataCopyExtParams& copyOutParamV2)
    {
        loopMode.loop2Size = dmaParam_.loopInfo.loopSize[NDDMA_INDEX3];
        loopMode.loop2SrcStride = ubDstStride_[1] * sizeof(T);
        loopMode.loop2DstStride = ubOutStride_[1] * sizeof(T);

        loopMode.loop1Size = dmaParam_.loopInfo.loopSize[NDDMA_INDEX2];
        loopMode.loop1SrcStride = ubDstStride_[NDDMA_INDEX2] * sizeof(T);
        loopMode.loop1DstStride = ubOutStride_[NDDMA_INDEX2] * sizeof(T);

        copyOutParamV2.blockCount = dmaParam_.loopInfo.loopSize[1];
        copyOutParamV2.blockLen = dmaParam_.loopInfo.loopSize[0] * sizeof(T);
        copyOutParamV2.srcStride = ubDstStride_[NDDMA_INDEX3] * sizeof(T);
        copyOutParamV2.dstStride = ubOutStride_[NDDMA_INDEX3] * sizeof(T);
        // add changes
        SetLoopModePara(loopMode, DataCopyMVType::UB_TO_OUT);
        for (int aIdx = 0; aIdx < dmaParam_.loopInfo.loopSize[NDDMA_INDEX4]; ++aIdx) {
            DataCopyExtParams copyParams{
                copyOutParamV2.blockCount, copyOutParamV2.blockLen,
                (copyOutParamV2.srcStride - copyOutParamV2.blockLen) / 32,
                copyOutParamV2.dstStride - copyOutParamV2.blockLen, 0};
            DataCopyPad(outputGm_[outAddr + aIdx * ubOutStride_[0]], outLocal[aIdx * ubDstStride_[0]], copyParams);
        }
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }

    __aicore__ inline void SubProcessCutTwo()
    {
        // Block_dim = 64, final block deal with 5
        int64_t coreLoops = (GetBlockIdx() == GetBlockNum() - 1) ? loopsTailCore_ : loopsPerCore_;
        int64_t loopId = GetBlockIdx() * loopsPerCore_;

        int64_t cutAxisLoopIdx01 = TILING_NDDMA_LEN - 1 - this->cutAxisIdx01_;
        int64_t cutAxisLoopIdx02 = TILING_NDDMA_LEN - 1 - this->cutAxisIdx02_;
        int64_t cutAxisMain01 = this->dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01]; // 1,
        int64_t cutAxisMain02 = this->dmaParam_.loopInfo.loopSize[cutAxisLoopIdx02]; // 2,

        // tmp
        int64_t tmpId = 0;
        int64_t cutAxisOuter01 = this->gmShape_[outerAxisNum_ - cutAxisNum_];     // ao
        int64_t cutAxisOuter02 = this->gmShape_[outerAxisNum_ - cutAxisNum_ + 1]; // bo
        int64_t currentCutAxisOuter01, currentCutAxisOuter02;

        int64_t axisId = -1;
        int64_t inputAddr = 0;
        int64_t outAddr = 0;

        LoopModeParams loopMode_;
        DataCopyExtParams copyOutParamV2_;

        for (int64_t loop = 0; loop < coreLoops; loop++) {
            tmpId = loopId + loop;
            inputAddr = 0;
            outAddr = 0;
            int64_t baseIdx = 0;
            for (; baseIdx < outerAxisNum_ - cutAxisNum_; baseIdx++) {
                axisId = tmpId % gmShape_[baseIdx];
                tmpId /= gmShape_[baseIdx];
                inputAddr += axisId * gmInStride_[baseIdx];
                outAddr += axisId * gmOutStride_[baseIdx];
            }

            currentCutAxisOuter01 = tmpId % gmShape_[baseIdx];
            tmpId /= gmShape_[baseIdx];
            inputAddr += currentCutAxisOuter01 * gmInStride_[baseIdx];
            outAddr += currentCutAxisOuter01 * gmOutStride_[baseIdx];
            baseIdx++;
            currentCutAxisOuter02 = tmpId % gmShape_[baseIdx];
            tmpId /= gmShape_[baseIdx];
            inputAddr += currentCutAxisOuter02 * gmInStride_[baseIdx];
            outAddr += currentCutAxisOuter02 * gmOutStride_[baseIdx];

            if (unlikely(currentCutAxisOuter01 == cutAxisOuter01 - 1 && currentCutAxisOuter02 == cutAxisOuter02 - 1)) {
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01] = cutAxisTail01_;
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx02] = cutAxisTail02_;
            } else if (currentCutAxisOuter01 == cutAxisOuter01 - 1) {
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01] = cutAxisTail01_;
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx02] = cutAxisMain02;
            } else if (currentCutAxisOuter02 == cutAxisOuter02 - 1) {
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01] = cutAxisMain01;
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx02] = cutAxisTail02_;
            } else {
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01] = cutAxisMain01;
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx02] = cutAxisMain02;
            }

            LocalTensor<T> inLocal = inOutQueue.template AllocTensor<T>();
            DataCopy(inLocal, inputGm_[inputAddr], dmaParam_);
            inOutQueue.EnQue(inLocal);
            LocalTensor<T> outLocal = inOutQueue.template DeQue<T>();

            MoveOutWithLoop(outLocal, outAddr, loopMode_, copyOutParamV2_);
            inOutQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void SubProcessCutOne()
    {
        int64_t coreLoops = (GetBlockIdx() == GetBlockNum() - 1) ? loopsTailCore_ : loopsPerCore_;
        int64_t loopId = GetBlockIdx() * loopsPerCore_;

        int64_t cutAxisLoopIdx01 = TILING_NDDMA_LEN - 1 - this->cutAxisIdx01_;
        int64_t cutAxisMain01 = this->dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01]; // 1,

        // tmp
        int64_t tmpId = 0;
        int64_t cutAxisOuter01 = this->gmShape_[outerAxisNum_ - cutAxisNum_]; // ao
        int64_t currentCutAxisOuter01;

        int64_t axisId = -1;
        int64_t inputAddr = 0;
        int64_t outAddr = 0;

        LoopModeParams loopMode_;
        DataCopyExtParams copyOutParamV2_;

        for (int64_t loop = 0; loop < coreLoops; loop++) {
            tmpId = loopId + loop;
            inputAddr = 0;
            outAddr = 0;
            int64_t baseIdx = 0;
            for (; baseIdx < outerAxisNum_ - cutAxisNum_; baseIdx++) {
                axisId = tmpId % gmShape_[baseIdx];
                tmpId /= gmShape_[baseIdx];
                inputAddr += axisId * gmInStride_[baseIdx];
                outAddr += axisId * gmOutStride_[baseIdx];
            }

            currentCutAxisOuter01 = tmpId % gmShape_[outerAxisNum_ - 1];
            tmpId /= gmShape_[outerAxisNum_ - 1];
            inputAddr += currentCutAxisOuter01 * gmInStride_[outerAxisNum_ - 1];
            outAddr += currentCutAxisOuter01 * gmOutStride_[outerAxisNum_ - 1];

            if (currentCutAxisOuter01 == cutAxisOuter01 - 1) {
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01] = cutAxisTail01_;
            } else {
                dmaParam_.loopInfo.loopSize[cutAxisLoopIdx01] = cutAxisMain01;
            }

            LocalTensor<T> inLocal = inOutQueue.template AllocTensor<T>();
            DataCopy(inLocal, inputGm_[inputAddr], dmaParam_);
            inOutQueue.EnQue(inLocal);
            LocalTensor<T> outLocal = inOutQueue.template DeQue<T>();

            MoveOutWithLoop(outLocal, outAddr, loopMode_, copyOutParamV2_);

            inOutQueue.FreeTensor(outLocal);
        }
    }

    template <typename U>
    __aicore__ inline void CopyArray(const U* src, U* dst, int64_t size)
    {
        for (int64_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }

private:
    TPipe pipe_;

    GlobalTensor<T> inputGm_, outputGm_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> inOutQueue;

    MultiCopyParams<T, NDDMA_DIM> dmaParam_;

    uint32_t ubShape_[TILING_NDDMA_LEN] = {0};
    uint64_t ubInStride_[TILING_NDDMA_LEN] = {0};
    uint32_t ubOutStride_[TILING_NDDMA_LEN] = {0};
    uint32_t ubDstStride_[TILING_NDDMA_LEN] = {0};

    int32_t gmShape_[TILING_ARRAY_LEN] = {0};
    int32_t gmInStride_[TILING_ARRAY_LEN] = {0};
    int32_t gmOutStride_[TILING_ARRAY_LEN] = {0};

    uint32_t cutAxisNum_ = 0;
    uint32_t outerAxisNum_ = 0;
    uint32_t innerAxisNum_ = 0;
    uint32_t cutAxisIdx01_ = 0;
    uint32_t cutAxisIdx02_ = 0;
    uint32_t cutAxisTail01_ = 0;
    uint32_t cutAxisTail02_ = 0;

    uint32_t finalDimAligned_ = 0;

    uint32_t blockNum_ = 0;
    uint32_t loopsPerCore_ = 0;
    uint32_t loopsTailCore_ = 0;

    int64_t storageOffset_ = 0;
    uint32_t ubSize_ = 0;
};

} // namespace AsStrided

#endif