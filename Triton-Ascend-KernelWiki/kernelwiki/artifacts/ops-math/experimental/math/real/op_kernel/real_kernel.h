/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file real_kernel.h
 * \brief real kernel
 */

#ifndef REAL_KERNEL_H
#define REAL_KERNEL_H

#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "real_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t COMPLEX_COEFFICIENT = 2;
constexpr int32_t DEPTH_NUM = 1;

namespace RealNs {

template <typename S, typename T>
class RealKernel {
public:
    __aicore__ inline RealKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RealTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ExtractRealPart(LocalTensor<T>& src, uint32_t count);
    __aicore__ inline void ExtractRealPartNonInplace(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count);
    __aicore__ inline void ProcessRealIdentity();
    __aicore__ inline void ProcessComplexTiling();
    __aicore__ inline void ProcessComplexNonInplace();

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DEPTH_NUM> inQueue;
    TQue<QuePosition::VECOUT, DEPTH_NUM> outQueue;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DEPTH_NUM> ioQueue; // For pure copy
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t blockIdx_ = 0;
    uint64_t blockLength = 0; // uint64_t to prevent overflow for large tensors
    uint64_t blockOffset = 0; // uint64_t to prevent overflow for large tensors
    uint32_t perOfCore = 0;   // Bounded by UB size, uint32_t is sufficient
    uint32_t loopOfCore = 0;  // Number of loops, uint32_t is sufficient
    uint32_t tailOfCore = 0;  // Bounded by UB size, uint32_t is sufficient
    uint32_t useNonInplace_ = 0;
};

template <typename S, typename T>
__aicore__ inline void RealKernel<S, T>::ExtractRealPart(LocalTensor<T>& src, uint32_t count)
{
    // GatherMask inplace: count * COMPLEX_COEFFICIENT * sizeof(T) must be divisible by 256
    constexpr uint8_t GATHER_MASK_MODE_ONE = 1;
    GatherMaskParams params = {1, 4, 8, 0};
    params.repeatTimes = count * COMPLEX_COEFFICIENT * sizeof(T) / 256;
    uint64_t rsvdCnt = 0;
    uint32_t mask = 0;
    GatherMask(src, src, GATHER_MASK_MODE_ONE, false, mask, params, rsvdCnt);
    PipeBarrier<PIPE_V>();
}

template <typename S, typename T>
__aicore__ inline void RealKernel<S, T>::ExtractRealPartNonInplace(
    LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
{
    constexpr uint8_t GATHER_MASK_MODE_ONE = 1;
    GatherMaskParams params;
    params.repeatTimes = 1;
    uint64_t rsvdCnt = 0;
    uint32_t mask = count * COMPLEX_COEFFICIENT;
    GatherMask(dst, src, GATHER_MASK_MODE_ONE, true, mask, params, rsvdCnt);
    PipeBarrier<PIPE_V>();
}

template <typename S, typename T>
__aicore__ inline void RealKernel<S, T>::Init(
    GM_ADDR x, GM_ADDR y, const RealTilingData* __restrict tilingData, TPipe* pipeIn)
{
    pipe_ = pipeIn;
    blockIdx_ = GetBlockIdx();

    // Initial offset: assume all cores are big cores
    uint64_t globalOffset = static_cast<uint64_t>(tilingData->bigCoreDataNum) * blockIdx_;

    if (static_cast<uint64_t>(tilingData->tailBlockNum) > 0 &&
        blockIdx_ < static_cast<uint64_t>(tilingData->tailBlockNum)) {
        // Big core: first tailBlockNum cores
        blockLength = static_cast<uint64_t>(tilingData->bigCoreDataNum);
        loopOfCore = static_cast<uint32_t>(tilingData->bigCoreLoopNum);
        tailOfCore = static_cast<uint32_t>(tilingData->bigCoreTailDataNum);
    } else {
        // Small core: rest of cores
        blockLength = static_cast<uint64_t>(tilingData->smallCoreDataNum);
        loopOfCore = static_cast<uint32_t>(tilingData->smallCoreLoopNum);
        tailOfCore = static_cast<uint32_t>(tilingData->smallCoreTailDataNum);
        // Adjust offset: subtract over-counted data from big cores
        if (tilingData->bigCoreDataNum > 0) {
            globalOffset -= (static_cast<uint64_t>(tilingData->bigCoreDataNum) -
                             static_cast<uint64_t>(tilingData->smallCoreDataNum)) *
                            (blockIdx_ - static_cast<uint64_t>(tilingData->tailBlockNum));
        }
    }

    perOfCore = static_cast<uint32_t>(tilingData->ubPartDataNum);
    useNonInplace_ = static_cast<uint32_t>(tilingData->useNonInplace);
    blockOffset = globalOffset;

    if constexpr (IsSameType<S, T>::value) {
        xGm.SetGlobalBuffer((__gm__ S*)x + blockOffset, blockLength);
    } else {
        // Cast to uint64_t to prevent overflow when blockOffset > 2^31
        uint64_t complexOffset = blockOffset * COMPLEX_COEFFICIENT;
        uint64_t complexLength = static_cast<uint64_t>(blockLength) * COMPLEX_COEFFICIENT;
        xGm.SetGlobalBuffer((__gm__ T*)x + complexOffset, complexLength);
    }
    yGm.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength);
}

template <typename S, typename T>
__aicore__ inline void RealKernel<S, T>::ProcessComplexTiling()
{
    pipe_->InitBuffer(inQueue, BUFFER_NUM, perOfCore * sizeof(T) * COMPLEX_COEFFICIENT);

    uint64_t currentOffset = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // Prologue: launch DMA in for first tile
    uint32_t firstTileLen = (loopOfCore > 1) ? perOfCore : tailOfCore;
    LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
    DataCopyExtParams copyFirstIn{
        static_cast<uint16_t>(1), static_cast<uint32_t>(firstTileLen * sizeof(T) * COMPLEX_COEFFICIENT),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(xLocal, xGm[0], copyFirstIn, padParams);
    inQueue.EnQue(xLocal);

    for (uint32_t loopIdx = 0; loopIdx < loopOfCore; loopIdx++) {
        uint32_t curLen = (loopIdx < loopOfCore - 1) ? perOfCore : tailOfCore;

        // Wait for DMA in completion
        LocalTensor<T> input = inQueue.DeQue<T>();

        // Prefetch: launch DMA in for next tile (overlaps with GatherMask below)
        if (loopIdx + 1 < loopOfCore) {
            uint32_t nextLen = (loopIdx + 1 < loopOfCore - 1) ? perOfCore : tailOfCore;
            LocalTensor<T> nextBuf = inQueue.AllocTensor<T>();
            DataCopyExtParams copyNextIn{
                static_cast<uint16_t>(1), static_cast<uint32_t>(nextLen * sizeof(T) * COMPLEX_COEFFICIENT),
                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
            DataCopyPad(nextBuf, xGm[(currentOffset + curLen) * COMPLEX_COEFFICIENT], copyNextIn, padParams);
            inQueue.EnQue(nextBuf);
        }

        ExtractRealPart(input, curLen);
        DataCopyExtParams copyOut{
            static_cast<uint16_t>(1), static_cast<uint32_t>(curLen * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(yGm[currentOffset], input, copyOut);
        inQueue.FreeTensor(input);

        currentOffset += curLen;
    }
}

template <typename S, typename T>
__aicore__ inline void RealKernel<S, T>::ProcessComplexNonInplace()
{
    pipe_->InitBuffer(inQueue, BUFFER_NUM, perOfCore * sizeof(T) * COMPLEX_COEFFICIENT);
    pipe_->InitBuffer(outQueue, BUFFER_NUM, perOfCore * sizeof(T));

    uint64_t currentOffset = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (uint32_t loopIdx = 0; loopIdx < loopOfCore; loopIdx++) {
        uint32_t curLen = (loopIdx < loopOfCore - 1) ? perOfCore : tailOfCore;
        uint32_t complexTileLen = curLen * COMPLEX_COEFFICIENT;

        // DMA in: read complex data
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        DataCopyExtParams copyIn{
            static_cast<uint16_t>(1), static_cast<uint32_t>(complexTileLen * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(xLocal, xGm[currentOffset * COMPLEX_COEFFICIENT], copyIn, padParams);
        inQueue.EnQue(xLocal);

        // GatherMask non-inplace: extract real part to outQueue
        LocalTensor<T> input = inQueue.DeQue<T>();
        LocalTensor<T> result = outQueue.AllocTensor<T>();
        ExtractRealPartNonInplace(result, input, curLen);
        outQueue.EnQue(result);
        inQueue.FreeTensor(input);

        // DMA out: write real data
        LocalTensor<T> output = outQueue.DeQue<T>();
        DataCopyExtParams copyOut{
            static_cast<uint16_t>(1), static_cast<uint32_t>(curLen * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(yGm[currentOffset], output, copyOut);
        outQueue.FreeTensor(output);

        currentOffset += curLen;
    }
}

template <typename S, typename T>
__aicore__ inline void RealKernel<S, T>::ProcessRealIdentity()
{
    pipe_->InitBuffer(ioQueue, BUFFER_NUM, perOfCore * sizeof(T));

    uint64_t currentOffset = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // Prologue: launch DMA in for first tile
    uint32_t firstTileLen = (loopOfCore > 1) ? perOfCore : tailOfCore;
    LocalTensor<T> firstBuf = ioQueue.AllocTensor<T>();
    DataCopyExtParams copyFirstIn{
        static_cast<uint16_t>(1), static_cast<uint32_t>(firstTileLen * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(firstBuf, xGm[0], copyFirstIn, padParams);
    ioQueue.EnQue(firstBuf);

    for (uint32_t loopIdx = 0; loopIdx < loopOfCore; loopIdx++) {
        uint32_t curLen = (loopIdx < loopOfCore - 1) ? perOfCore : tailOfCore;

        // Wait for DMA in completion
        LocalTensor<T> input = ioQueue.DeQue<T>();

        // Prefetch: launch DMA in for next tile (overlaps with DMA out below)
        if (loopIdx + 1 < loopOfCore) {
            uint32_t nextLen = (loopIdx + 1 < loopOfCore - 1) ? perOfCore : tailOfCore;
            LocalTensor<T> nextBuf = ioQueue.AllocTensor<T>();
            DataCopyExtParams copyNextIn{
                static_cast<uint16_t>(1), static_cast<uint32_t>(nextLen * sizeof(T)), static_cast<uint32_t>(0),
                static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
            DataCopyPad(nextBuf, xGm[currentOffset + curLen], copyNextIn, padParams);
            ioQueue.EnQue(nextBuf);
        }

        // DMA out: copy to yGm
        DataCopyExtParams copyOut{
            static_cast<uint16_t>(1), static_cast<uint32_t>(curLen * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(yGm[currentOffset], input, copyOut);
        ioQueue.FreeTensor(input);

        currentOffset += curLen;
    }
}

template <typename S, typename T>
__aicore__ inline void RealKernel<S, T>::Process()
{
    if constexpr (IsSameType<S, T>::value) {
        ProcessRealIdentity();
    } else {
        if (useNonInplace_) {
            ProcessComplexNonInplace();
        } else {
            ProcessComplexTiling();
        }
    }
}

} // namespace RealNs

#endif // REAL_KERNEL_H
