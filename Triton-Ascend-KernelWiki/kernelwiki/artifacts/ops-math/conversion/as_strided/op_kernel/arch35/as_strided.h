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
 * \file as_strided.h
 * \brief as_strided
 */

#ifndef ASCENDC_AS_STRIDED_H_
#define ASCENDC_AS_STRIDED_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include "kernel_operator.h"
#include "as_strided_struct.h"

namespace AsStrided {
using namespace AscendC;

constexpr uint8_t ZERO_U8 = 0;
constexpr int64_t TILING_ARRAY_LEN = 10;
constexpr int64_t TILING_NDDMA_LEN = 5;
constexpr uint8_t NDDMA_DIM = 5;
constexpr int16_t BUFFER_NUM = 2;
constexpr size_t NDDMA_INDEX0 = 0;
constexpr size_t NDDMA_INDEX1 = 1;
constexpr size_t NDDMA_INDEX2 = 2;
constexpr size_t NDDMA_INDEX3 = 3;
constexpr size_t NDDMA_INDEX4 = 4;

template <typename T>
class KernelAsStrided {
public:
    __aicore__ inline KernelAsStrided()
    {}

    template <typename U>
    __aicore__ inline void CopyArray(const U* src, U* dst, int64_t size)
    {
        for (int64_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR outShape, GM_ADDR outStride, GM_ADDR output, AsStridedTilingData tilingData)
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
        CopyArray(tilingData.outStrideArr, outStrideArr_, TILING_ARRAY_LEN);
        CopyArray(tilingData.outLoopArr, outLoopArr_, TILING_ARRAY_LEN);
        CopyArray(tilingData.nddmaLoop, nddmaLoop_, TILING_NDDMA_LEN);
        CopyArray(tilingData.nddmaTailLoop, nddmaTailLoop_, TILING_NDDMA_LEN);
        CopyArray(tilingData.nddmaSrcStride, nddmaSrcStride_, TILING_NDDMA_LEN);
        CopyArray(tilingData.nddmaDstStride, nddmaDstStride_, TILING_NDDMA_LEN);

        blockOffset_ = loopsPerCore_ * ubFactor_;
        tileOffset_ =
            ubFactorTail_ == 0 ? ubFactor_ * outerAxisFactor_ : ubFactor_ * (outerAxisFactor_ - 1) + ubFactorTail_;

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

    __aicore__ inline void InitCopyParams()
    {
        dmaParam_ = {
            {
                {// src stride
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX4]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX3]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX2]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX1]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX0])},
                {//  dst stride
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX4]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX3]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX2]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX1]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX0])},
                {//  loop size
                 static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX4]), static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX3]),
                 static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX2]), static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX1]),
                 static_cast<uint32_t>(nddmaLoop_[NDDMA_INDEX0])},
                {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}, // left pad
                {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}  //  right pad
            },
            0};

        dmaTailParam_ = {
            {
                {// src stride
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX4]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX3]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX2]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX1]),
                 static_cast<uint64_t>(nddmaSrcStride_[NDDMA_INDEX0])},
                {//  dst stride
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX4]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX3]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX2]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX1]),
                 static_cast<uint32_t>(nddmaDstStride_[NDDMA_INDEX0])},
                {//  loop size
                 static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX4]),
                 static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX3]),
                 static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX2]),
                 static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX1]),
                 static_cast<uint32_t>(nddmaTailLoop_[NDDMA_INDEX0])},
                {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}, // left pad
                {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}  //  right pad
            },
            0};

        copyParams_ = {
            static_cast<uint16_t>(1), static_cast<uint32_t>(ubFactor_ * sizeof(T)), static_cast<uint32_t>(0), 0, 0};

        copyParamsTail_ = {
            static_cast<uint16_t>(1), static_cast<uint32_t>(ubFactorTail_ * sizeof(T)), static_cast<uint32_t>(0), 0, 0};
    }

    __aicore__ inline void Process()
    {
        InitCopyParams();
        SubProcess();
    }

    __aicore__ inline void CopyProcess(int64_t srcOffset, int64_t dstOffset, uint32_t totalIdx)
    {
        static constexpr MultiCopyConfig Config = {false, 0, 0, false};
        if (innerAxisFactorTail_ == 0) {
            LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
            DataCopy<T, NDDMA_DIM, Config>(srcLocal, inputGm_[srcOffset], dmaParam_);
            inQueue_.EnQue(srcLocal);
            LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
            DataCopyPad(outputGm_[dstOffset], dstLocal, copyParams_);
            inQueue_.FreeTensor(dstLocal);
        } else {
            if ((totalIdx + 1) % outerAxisFactor_ != 0) {
                LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
                DataCopy<T, NDDMA_DIM, Config>(srcLocal, inputGm_[srcOffset], dmaParam_);
                inQueue_.EnQue(srcLocal);
                LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
                DataCopyPad(outputGm_[dstOffset], dstLocal, copyParams_);
                inQueue_.FreeTensor(dstLocal);
            } else {
                LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
                DataCopy<T, NDDMA_DIM, Config>(srcLocal, inputGm_[srcOffset], dmaTailParam_);
                inQueue_.EnQue(srcLocal);
                LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
                DataCopyPad(outputGm_[dstOffset], dstLocal, copyParamsTail_);
                inQueue_.FreeTensor(dstLocal);
            }
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
            int64_t totalIdx = currentIdx;
            srcOffset = 0;
            useIdxLoop = TILING_ARRAY_LEN - 1 - outerAxisNum_;
            for (int64_t useIdx = TILING_ARRAY_LEN - 1; useIdx > useIdxLoop; useIdx--) {
                srcOffset += ((static_cast<int64_t>(totalIdx) / Product(outLoopArr_, useIdx + 1)) %
                              static_cast<int64_t>(outLoopArr_[useIdx])) *
                             static_cast<int64_t>(outStrideArr_[useIdx]);
            }
            dstOffset = (static_cast<int64_t>(totalIdx) / static_cast<int64_t>(outerAxisFactor_)) *
                            static_cast<int64_t>(tileOffset_) +
                        (static_cast<int64_t>(totalIdx) % static_cast<int64_t>(outerAxisFactor_)) *
                            static_cast<int64_t>(ubFactor_);
            CopyProcess(srcOffset, dstOffset, totalIdx);
        }
    }

private:
    TPipe pipe_;

    GlobalTensor<T> inputGm_, outputGm_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;

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
    uint32_t blockOffset_ = 0;
    uint32_t innerAxisNum_ = 0;
    int64_t storageOffset_ = 0;
    uint32_t innerAxisFactorTail_ = 0;
    uint32_t axisOutTotalFactor_ = 0;
    uint32_t tileOffset_ = 0;
    int32_t outLoopArr_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int32_t outStrideArr_[TILING_ARRAY_LEN] = {0};
    uint32_t nddmaLoop_[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint32_t nddmaTailLoop_[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint32_t nddmaDstStride_[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint64_t nddmaSrcStride_[TILING_NDDMA_LEN] = {0};
    MultiCopyParams<T, NDDMA_DIM> dmaParam_;
    MultiCopyParams<T, NDDMA_DIM> dmaTailParam_;
};

} // namespace AsStrided

#endif