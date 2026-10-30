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
 * \file as_strided_move_align.h
 * \brief as_strided_move_align
 */

#ifndef ASCENDC_AS_STRIDED_MOVEALIGN_H_
#define ASCENDC_AS_STRIDED_MOVEALIGN_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include "as_strided.h"
#include "as_strided_zero_stride.h"
#include "kernel_operator.h"

namespace AsStrided {
using namespace AscendC;

constexpr size_t MOVEALIGN_DIM2 = 2;
constexpr size_t INNER_AXIS_DIM3 = 3;
constexpr int64_t ALIGN_BYTE = 32;

template <typename T>
class KernelAsStridedMoveAlign {
public:

    __aicore__ inline KernelAsStridedMoveAlign()  {}

    template<typename U>
    __aicore__ inline void CopyArray(const U *src, U *dst, int64_t size)
    {
        for (int64_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR outShape, GM_ADDR outStride, GM_ADDR output,
        AsStridedTilingData tilingData)
    {
        blockNum_ = tilingData.blockNum;
        loopsTailCore_ = tilingData.loopsTailCore;
        tilingAxisIdx_ = tilingData.tilingAxisIdx;
        outerAxisFactor_ = tilingData.outerAxisFactor;
        innerAxisFactor_ = tilingData.innerAxisFactor;
        axisOutTotalFactor_ = tilingData.axisOutTotalFactor;
        outerAxisNum_ = tilingData.outerAxisNum;
        innerAxisNum_ = tilingData.innerAxisNum;
        loopsPerCore_ = tilingData.loopsPerCore;
        ubFactor_ = tilingData.ubFactor;
        ubFactorTail_ = tilingData.ubFactorTail;
        innerAxisFactorTail_ = tilingData.innerAxisFactorTail;
        ubSize_ = tilingData.ubSize;
        storageOffset_ = tilingData.storageOffset;
        en32BAligned_ = tilingData.en32BAligned;
        
        CopyArray(tilingData.outStrideArr, outStrideArr_, TILING_ARRAY_LEN);
        CopyArray(tilingData.outLoopArr, outLoopArr_, TILING_ARRAY_LEN);
        CopyArray(tilingData.nddmaLoop, nddmaLoop_, TILING_NDDMA_LEN);
        CopyArray(tilingData.nddmaTailLoop, nddmaTailLoop_, TILING_NDDMA_LEN);
        CopyArray(tilingData.nddmaSrcStride, nddmaSrcStride_, TILING_NDDMA_LEN);
        CopyArray(tilingData.nddmaDstStride, nddmaDstStride_, TILING_NDDMA_LEN);

        tileOffset_ = innerAxisFactorTail_ == 0 ? ubFactor_ * outerAxisFactor_ : ubFactor_ * (outerAxisFactor_ - 1) +
                      ubFactorTail_;

        inputGm_.SetGlobalBuffer((__gm__ T*)input + storageOffset_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, ubSize_ * sizeof(T));
    }

    __aicore__ inline int64_t Product(int32_t* outLoopArr, size_t size)
    {
        int64_t result = 1;
        for (size_t i = size; i < TILING_ARRAY_LEN; i++) {
            result *= static_cast<int64_t>(outLoopArr[i]);
        }
        return result;
    }

    __aicore__ inline void SetCopyOutAlignParams()
    {
        copyParams_.blockCount = loopMode_.loop1Size * loopMode_.loop2Size *
                                copyInParam_.blockCount;
        copyParams_.blockLen =  copyInParam_.blockLen;
        copyParams_.dstStride = 0;
        copyParams_.srcStride = 0;

        copyParamsTail_.blockCount = loopModeTail_.loop1Size * loopModeTail_.loop2Size *
                                copyInParamTail_.blockCount;
        copyParamsTail_.blockLen = copyInParamTail_.blockLen;
        copyParamsTail_.dstStride = 0;
        copyParamsTail_.srcStride = 0;
    }

    __aicore__ inline void SetCopyInAlignParam()
    {
        loopMode_.loop1Size = static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX2]);
        loopMode_.loop2Size = static_cast<uint32_t>(nddmaLoop_[1]);
        loopMode_.loop1SrcStride = static_cast<uint32_t>(nddmaSrcStride_[NDDMA_INDEX2]) * sizeof(T);
        loopMode_.loop2SrcStride = static_cast<uint32_t>(nddmaSrcStride_[1]) * sizeof(T);

        loopModeTail_.loop1Size = static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX2]);
        loopModeTail_.loop2Size = static_cast<uint32_t>(nddmaTailLoop_[1]);
        loopModeTail_.loop1SrcStride = static_cast<uint32_t>(nddmaSrcStride_[NDDMA_INDEX2]) * sizeof(T);
        loopModeTail_.loop2SrcStride = static_cast<uint32_t>(nddmaSrcStride_[1]) * sizeof(T);

        copyInParam_.blockCount = static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX3]);
        copyInParam_.blockLen = static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX4]) * sizeof(T);
        copyInParam_.srcStride = static_cast<uint32_t>(nddmaSrcStride_[NDDMA_INDEX3]) * sizeof(T);
        copyInParam_.dstStride = static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX4]) * sizeof(T);

        copyInParamTail_.blockCount = static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX3]);
        copyInParamTail_.blockLen = static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX4]) * sizeof(T);
        copyInParamTail_.srcStride = static_cast<uint32_t>(nddmaSrcStride_[NDDMA_INDEX3]) * sizeof(T);
        copyInParamTail_.dstStride = static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX4]) * sizeof(T);

        // 计算loop模式的dstStride
        // copyInParam_
        int64_t wAlignedBytes = CeilDiv(static_cast<int64_t>(copyInParam_.blockLen), ALIGN_BYTE) * ALIGN_BYTE;
        loopMode_.loop1DstStride = wAlignedBytes * copyInParam_.blockCount;
        loopMode_.loop2DstStride = loopMode_.loop1DstStride * loopMode_.loop1Size;
        
        //copyInParamTail_
        int64_t wAlignedTailBytes = CeilDiv(static_cast<int64_t>(copyInParamTail_.blockLen), ALIGN_BYTE) * ALIGN_BYTE;
        loopModeTail_.loop1DstStride = wAlignedTailBytes * copyInParam_.blockCount;
        loopModeTail_.loop2DstStride = loopModeTail_.loop1DstStride * loopModeTail_.loop1Size;
    }

    __aicore__ inline void Process()
    {
        SetCopyInAlignParam();
        SetCopyOutAlignParams();
        SubProcess();
    }

    __aicore__ inline void CopyIn(int64_t srcOffset, uint32_t totalIdx) 
    {
        if (innerAxisFactorTail_ != 0 && (totalIdx + 1) % outerAxisFactor_ == 0) {
            LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
            SetLoopModePara(loopModeTail_, DataCopyMVType::OUT_TO_UB);
            DataCopyExtParams copyParams{
                copyInParamTail_.blockCount, 
                copyInParamTail_.blockLen,
                copyInParamTail_.srcStride - copyInParamTail_.blockLen, 
                (copyInParam_.dstStride - copyInParam_.blockLen-padParams_.leftPadding - padParams_.rightPadding) / 32,
                0
            };
            DataCopyPadExtParams<T> padParames{
                false, padParams_.leftPadding, padParams_.rightPadding,0
            };
            DataCopyPad(srcLocal,inputGm_[srcOffset],copyParams,padParames);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            inQueue_.EnQue(srcLocal);
        } else {
            LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
            SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
            DataCopyExtParams copyParams{
                copyInParam_.blockCount, 
                copyInParam_.blockLen,
                copyInParam_.srcStride - copyInParam_.blockLen, 
                (copyInParam_.dstStride - copyInParam_.blockLen - padParams_.leftPadding-padParams_.rightPadding) / 32,
                0
            };
            DataCopyPadExtParams<T> padParames{
                    false, padParams_.leftPadding, padParams_.rightPadding, 0
                };
            DataCopyPad(srcLocal, inputGm_[srcOffset], copyParams, padParames);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            inQueue_.EnQue(srcLocal);
        } 
    }

    __aicore__ inline void CopyOut(int64_t dstOffset, uint32_t totalIdx)
    {   
        if (innerAxisFactorTail_ != 0 && (totalIdx + 1) % outerAxisFactor_ == 0) {
            LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
            DataCopyPad(outputGm_[dstOffset], dstLocal, copyParamsTail_);
            inQueue_.FreeTensor(dstLocal);
        } else {
            LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
            DataCopyPad(outputGm_[dstOffset], dstLocal, copyParams_);
            inQueue_.FreeTensor(dstLocal);
        } 
    }

    __aicore__ inline void SubProcess()
    {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        int64_t useIdxLoop = 0;

        for (int64_t loop = 0; loop < loopsPerCore_; loop++) {
            int64_t currentIdx = loop * GetBlockNum() + GetBlockIdx();
            if (currentIdx >= axisOutTotalFactor_) {
                break;
            }
            int64_t totalIdx = static_cast<uint32_t>(currentIdx);
            srcOffset = 0;
            useIdxLoop = TILING_ARRAY_LEN - 1 - outerAxisNum_;
            for (int64_t useIdx = TILING_ARRAY_LEN - 1; useIdx > useIdxLoop; useIdx--) {
                srcOffset += (
                        (static_cast<int64_t>(totalIdx) / Product(outLoopArr_, useIdx + 1)) 
                        % static_cast<int64_t>(outLoopArr_[useIdx])
                    ) * static_cast<int64_t>(outStrideArr_[useIdx]);
            }
            dstOffset = 
                (static_cast<int64_t>(totalIdx) / static_cast<int64_t>(outerAxisFactor_)) * static_cast<int64_t>(tileOffset_) + 
                (static_cast<int64_t>(totalIdx) % static_cast<int64_t>(outerAxisFactor_)) * static_cast<int64_t>(ubFactor_);
            
            CopyIn(srcOffset, totalIdx);
            CopyOut(dstOffset, totalIdx);

            int32_t eventIdMTE32S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIdMTE32S);
            WaitFlag<HardEvent::MTE3_S>(eventIdMTE32S);
        }
    }

private:

    TPipe pipe_;

    GlobalTensor<T> inputGm_, outputGm_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> inQueue_;

    DataCopyExtParams copyParams_;
    DataCopyExtParams copyParamsTail_;

    uint32_t loopsTailCore_ = 0;
    uint32_t tilingAxisIdx_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t outerAxisFactor_ = 0;
    uint32_t innerAxisFactor_ = 0;
    uint32_t outerAxisNum_ = 0;
    uint32_t loopsPerCore_ = 0;
    uint32_t ubFactor_ = 0;
    uint32_t ubFactorTail_ = 0;
    uint32_t ubSize_ = 0;
    uint32_t innerAxisNum_ = 0;
    int64_t storageOffset_ = 0;
    uint32_t innerAxisFactorTail_ = 0;
    uint32_t tileOffset_ = 0;
    uint32_t axisOutTotalFactor_ = 0;
    uint32_t en32BAligned_ = 0;
    int32_t outLoopArr_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int32_t outStrideArr_[TILING_ARRAY_LEN] = {0};
    uint32_t nddmaLoop_[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint32_t nddmaTailLoop_[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint32_t nddmaDstStride_[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint64_t nddmaSrcStride_[TILING_NDDMA_LEN] = {0};

    LoopModeParams loopMode_;
    LoopModeParams loopModeTail_;
    DataCopyExtParams copyInParam_;
    DataCopyExtParams copyInParamTail_;
    DataCopyPadExtParams<T> padParams_ {false, 0, 0, 0};
};

} // namespace AsStrided

#endif
