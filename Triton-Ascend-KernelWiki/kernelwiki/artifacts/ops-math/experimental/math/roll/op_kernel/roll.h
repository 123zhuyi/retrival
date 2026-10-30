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
 * \file roll.h
 * \brief Roll Ascend C kernel.
 */

#ifndef ROLL_H_
#define ROLL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "roll_tiling_data.h"

namespace RollKernel {
using namespace AscendC;

constexpr int32_t ROLL_BUFFER_NUM = 1;
constexpr int64_t ROLL_GM_BLOCK_BYTES = 32;
constexpr int64_t ROLL_STRIDED_SEGMENT_MIN_BYTES = 12;
constexpr int64_t ROLL_MAX_DATACOPY_BLOCK_COUNT = 4095;
constexpr int64_t ROLL_FLAT_PATCH_MAX_BLOCK_BYTES = 768;
constexpr int64_t ROLL_FLAT_PATCH_MIN_TOTAL_BYTES = 4096;

template <typename T>
class Roll {
public:
    __aicore__ inline Roll() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RollTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t ComputeSourceRowIndex(int64_t outputRowIndex) const;
    __aicore__ inline int64_t ComputeContiguousSourceRowRun(int64_t outputRowIndex, int64_t maxRows) const;
    __aicore__ inline int64_t ComputeSourceBlockIndex(int64_t outputBlockIndex, int64_t lastActiveDim) const;
    __aicore__ inline int64_t ComputeContiguousSourceBlockRun(
        int64_t outputBlockIndex, int64_t maxBlocks, int64_t lastActiveDim) const;
    __aicore__ inline int64_t ComputeInputIndex(int64_t outputIndex) const;
    __aicore__ inline void CopySegmentByScalar(int64_t dstIndex, int64_t srcIndex, int64_t elementCount);
    __aicore__ inline void CopySegmentBySourceAligned(int64_t dstIndex, int64_t srcIndex, int64_t elementCount);
    __aicore__ inline void CopyStridedSourceSegments(
        int64_t dstIndex, int64_t srcIndex, int64_t segmentElements, int64_t strideElements, int64_t segmentCount);
    __aicore__ inline bool CopyStridedSingleElementByRowGather(int64_t dstIndex,
                                                               int64_t srcBlockBase,
                                                               int64_t sourceOffset,
                                                               int64_t blockSize,
                                                               int64_t blockCount,
                                                               bool preferRowGatherPatch);
    __aicore__ inline bool CopyBlockRollByFlatPatch(int64_t dstIndex,
                                                    int64_t srcIndex,
                                                    int64_t blockSize,
                                                    int64_t firstElements,
                                                    int64_t secondElements,
                                                    int64_t blockCount);
    __aicore__ inline void CopySegment(int64_t dstIndex, int64_t srcIndex, int64_t elementCount);
    __aicore__ inline void CopyIdentity();
    __aicore__ inline void CopyFlattenRoll();
    __aicore__ inline void CopyFlattenRollBySource();
    __aicore__ inline void CopyLeadingDimRollBySource();
    __aicore__ inline void CopyLastDimRoll();
    __aicore__ inline void CopyLastDimRollByRows();
    __aicore__ inline void CopyLastDimFullRows(int64_t dstIndex, int64_t rowCount);
    __aicore__ inline void CopyRowsRollInUb(
        LocalTensor<T>& outLocal, LocalTensor<T>& inLocal, int64_t rowCount, int64_t dimSize,
        int64_t alignedRowElements, int64_t shift);
    __aicore__ inline void CopyLastDimFullRowsBySegments(int64_t dstIndex, int64_t rowCount);
    __aicore__ inline void CopyMultiDimLastDimFullRowsFromSource(int64_t dstIndex, int64_t srcRowIndex, int64_t rowCount);
    __aicore__ inline void CopyMultiDimLastDimFullRowsBySegments(int64_t dstIndex, int64_t srcRowIndex, int64_t rowCount);
    __aicore__ inline void CopyMultiDimLastDimFullRows(int64_t dstIndex, int64_t rowCount);
    __aicore__ inline void CopyMultiDimLastDimRollByRows();
    __aicore__ inline void CopyMultiDimNonLastBlockPartial(int64_t lastActiveDim, int64_t& dstIndex, int64_t& remain);
    __aicore__ inline void CopyMultiDimNonLastFullBlocks(
        int64_t lastActiveDim, int64_t dstIndex, int64_t srcBlockIndex, int64_t blockCount);
    __aicore__ inline void CopyMultiDimNonLastRollByBlocks(int64_t lastActiveDim);
    __aicore__ inline void CopyLastDimPartial(int64_t& dstIndex, int64_t& remain);
    __aicore__ inline void CopySingleDimRoll();
    __aicore__ inline void CopySingleDimRollByBlocks();
    __aicore__ inline void CopySingleDimFullBlocks(int64_t dstIndex, int64_t blockCount);
    __aicore__ inline void CopySingleDimPartial(int64_t& dstIndex, int64_t& remain);
    __aicore__ inline void CopySegmentedRoll();
    __aicore__ inline void ProcessScalar();

private:
    TPipe* pipe_ = nullptr;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, ROLL_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, ROLL_BUFFER_NUM> outQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    const RollTilingData* tilingData_ = nullptr;
    int64_t startIndex_ = 0;
    int64_t elementCount_ = 0;
    int64_t ubElements_ = 1;
};

template <typename T>
__aicore__ inline void Roll<T>::Init(GM_ADDR x, GM_ADDR y, const RollTilingData* tilingData, TPipe* pipe)
{
    tilingData_ = tilingData;
    pipe_ = pipe;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t perCoreElements =
        tilingData_->perCoreElements > 0 ? tilingData_->perCoreElements : tilingData_->blockFactor;
    const int64_t lastCoreElements =
        tilingData_->lastCoreElements > 0 ? tilingData_->lastCoreElements : perCoreElements;
    startIndex_ = blockIdx * perCoreElements;
    if (blockIdx >= tilingData_->usedCoreNum) {
        elementCount_ = 0;
    } else {
        elementCount_ = (blockIdx == tilingData_->usedCoreNum - 1) ? lastCoreElements : perCoreElements;
    }
    if (startIndex_ >= tilingData_->totalNum) {
        elementCount_ = 0;
    } else if (startIndex_ + elementCount_ > tilingData_->totalNum) {
        elementCount_ = tilingData_->totalNum - startIndex_;
    }
    ubElements_ = tilingData_->ubElements > 0 ? tilingData_->ubElements : tilingData_->ubFactor;
    if (ubElements_ <= 0) {
        ubElements_ = 1;
    }
    pipe_->InitBuffer(inQueue_, ROLL_BUFFER_NUM, ubElements_ * sizeof(T));
    pipe_->InitBuffer(outQueue_, ROLL_BUFFER_NUM, ubElements_ * sizeof(T));
}

template <typename T>
__aicore__ inline int64_t Roll<T>::ComputeSourceRowIndex(int64_t outputRowIndex) const
{
    int64_t remain = outputRowIndex;
    int64_t inputRow = 0;
    for (int64_t dim = 0; dim < tilingData_->dimNum - 1; ++dim) {
        const int64_t rowStride = tilingData_->strides[dim] / tilingData_->shapes[tilingData_->dimNum - 1];
        const int64_t coord = remain / rowStride;
        remain %= rowStride;
        const int64_t shape = tilingData_->shapes[dim];
        const int64_t sourceCoord = (coord - tilingData_->shifts[dim] + shape) % shape;
        inputRow += sourceCoord * rowStride;
    }
    return inputRow;
}

template <typename T>
__aicore__ inline int64_t Roll<T>::ComputeContiguousSourceRowRun(int64_t outputRowIndex, int64_t maxRows) const
{
    if (maxRows <= 1) {
        return 1;
    }
    const int64_t lastDimSize = tilingData_->shapes[tilingData_->dimNum - 1];
    int64_t runRows = maxRows;
    for (int64_t dim = 0; dim < tilingData_->dimNum - 1; ++dim) {
        const int64_t shift = tilingData_->shifts[dim];
        if (shift == 0) {
            continue;
        }
        const int64_t rowStride = tilingData_->strides[dim] / lastDimSize;
        if (rowStride <= 0) {
            continue;
        }
        const int64_t shape = tilingData_->shapes[dim];
        const int64_t coord = (outputRowIndex / rowStride) % shape;
        const int64_t lowerOffset = outputRowIndex % rowStride;
        int64_t contiguous = 0;
        if (coord < shift) {
            contiguous = (shift - coord) * rowStride - lowerOffset;
        } else {
            contiguous = (shape - coord) * rowStride - lowerOffset;
        }
        if (contiguous <= 0) {
            contiguous = 1;
        }
        if (runRows > contiguous) {
            runRows = contiguous;
        }
    }
    return runRows > 0 ? runRows : 1;
}

template <typename T>
__aicore__ inline int64_t Roll<T>::ComputeSourceBlockIndex(int64_t outputBlockIndex, int64_t lastActiveDim) const
{
    const int64_t blockSize = tilingData_->shapes[lastActiveDim] * tilingData_->strides[lastActiveDim];
    int64_t remain = outputBlockIndex;
    int64_t inputBlock = 0;
    for (int64_t dim = 0; dim < lastActiveDim; ++dim) {
        const int64_t blockStride = tilingData_->strides[dim] / blockSize;
        const int64_t coord = remain / blockStride;
        remain %= blockStride;
        const int64_t shape = tilingData_->shapes[dim];
        const int64_t sourceCoord = (coord - tilingData_->shifts[dim] + shape) % shape;
        inputBlock += sourceCoord * blockStride;
    }
    return inputBlock;
}

template <typename T>
__aicore__ inline int64_t Roll<T>::ComputeContiguousSourceBlockRun(
    int64_t outputBlockIndex, int64_t maxBlocks, int64_t lastActiveDim) const
{
    if (maxBlocks <= 1) {
        return 1;
    }
    const int64_t blockSize = tilingData_->shapes[lastActiveDim] * tilingData_->strides[lastActiveDim];
    int64_t runBlocks = maxBlocks;
    for (int64_t dim = 0; dim < lastActiveDim; ++dim) {
        const int64_t shift = tilingData_->shifts[dim];
        if (shift == 0) {
            continue;
        }
        const int64_t blockStride = tilingData_->strides[dim] / blockSize;
        if (blockStride <= 0) {
            continue;
        }
        const int64_t shape = tilingData_->shapes[dim];
        const int64_t coord = (outputBlockIndex / blockStride) % shape;
        const int64_t lowerOffset = outputBlockIndex % blockStride;
        int64_t contiguous = 0;
        if (coord < shift) {
            contiguous = (shift - coord) * blockStride - lowerOffset;
        } else {
            contiguous = (shape - coord) * blockStride - lowerOffset;
        }
        if (contiguous <= 0) {
            contiguous = 1;
        }
        if (runBlocks > contiguous) {
            runBlocks = contiguous;
        }
    }
    return runBlocks > 0 ? runBlocks : 1;
}

template <typename T>
__aicore__ inline int64_t Roll<T>::ComputeInputIndex(int64_t outputIndex) const
{
    if (tilingData_->dimNum <= 0 || tilingData_->dimNum > static_cast<int64_t>(ROLL_MAX_DIM_NUM)) {
        return outputIndex;
    }

    int64_t remain = outputIndex;
    int64_t inputIndex = 0;
    for (int64_t dim = 0; dim < tilingData_->dimNum; ++dim) {
        const int64_t stride = tilingData_->strides[dim];
        const int64_t coord = remain / stride;
        remain %= stride;
        const int64_t shape = tilingData_->shapes[dim];
        const int64_t sourceCoord = (coord - tilingData_->shifts[dim] + shape) % shape;
        inputIndex += sourceCoord * stride;
    }
    return inputIndex;
}

template <typename T>
__aicore__ inline void Roll<T>::CopySegmentByScalar(int64_t dstIndex, int64_t srcIndex, int64_t elementCount)
{
    int64_t copied = 0;
    while (copied < elementCount) {
        int64_t current = elementCount - copied;
        if (current > ubElements_) {
            current = ubElements_;
        }

        LocalTensor<T> local = inQueue_.AllocTensor<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = static_cast<uint32_t>(current * sizeof(T));
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, xGm_[srcIndex + copied], copyInParams, padParams);
        inQueue_.EnQue(local);

        LocalTensor<T> result = inQueue_.DeQue<T>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = static_cast<uint32_t>(current * sizeof(T));
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm_[dstIndex + copied], result, copyOutParams);
        inQueue_.FreeTensor(result);
        copied += current;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopySegmentBySourceAligned(int64_t dstIndex, int64_t srcIndex, int64_t elementCount)
{
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    const int64_t elementsPerBlock = ROLL_GM_BLOCK_BYTES / typeBytes;
    int64_t copied = 0;
    while (copied < elementCount) {
        const int64_t src = srcIndex + copied;
        const int64_t srcResidue = src % elementsPerBlock;
        if (srcResidue != 0) {
            int64_t current = elementsPerBlock - srcResidue;
            if (current > elementCount - copied) {
                current = elementCount - copied;
            }
            CopySegmentByScalar(dstIndex + copied, src, current);
            copied += current;
            continue;
        }

        int64_t current = elementCount - copied;
        if (current > ubElements_) {
            current = ubElements_;
        }
        LocalTensor<T> local = inQueue_.AllocTensor<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = static_cast<uint32_t>(current * typeBytes);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, xGm_[src], copyInParams, padParams);
        inQueue_.EnQue(local);
        LocalTensor<T> inLocal = inQueue_.DeQue<T>();

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = static_cast<uint32_t>(current * typeBytes);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm_[dstIndex + copied], inLocal, copyOutParams);
        inQueue_.FreeTensor(inLocal);
        copied += current;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyStridedSourceSegments(
    int64_t dstIndex, int64_t srcIndex, int64_t segmentElements, int64_t strideElements, int64_t segmentCount)
{
    if (segmentElements <= 0 || segmentCount <= 0) {
        return;
    }
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    int64_t copiedSegments = 0;
    while (copiedSegments < segmentCount) {
        int64_t currentSegments = segmentCount - copiedSegments;
        const int64_t maxSegments = ubElements_ / segmentElements;
        if (currentSegments > maxSegments) {
            currentSegments = maxSegments;
        }
        if (currentSegments > ROLL_MAX_DATACOPY_BLOCK_COUNT) {
            currentSegments = ROLL_MAX_DATACOPY_BLOCK_COUNT;
        }
        if (currentSegments <= 0) {
            currentSegments = 1;
        }

        const int64_t curSrc = srcIndex + copiedSegments * strideElements;
        const int64_t curDst = dstIndex + copiedSegments * strideElements;
        LocalTensor<T> local = inQueue_.AllocTensor<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = static_cast<uint16_t>(currentSegments);
        copyInParams.blockLen = static_cast<uint32_t>(segmentElements * typeBytes);
        copyInParams.srcStride = static_cast<uint32_t>((strideElements - segmentElements) * typeBytes);
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, xGm_[curSrc], copyInParams, padParams);
        inQueue_.EnQue(local);
        LocalTensor<T> inLocal = inQueue_.DeQue<T>();

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = static_cast<uint16_t>(currentSegments);
        copyOutParams.blockLen = static_cast<uint32_t>(segmentElements * typeBytes);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = static_cast<uint32_t>((strideElements - segmentElements) * typeBytes);
        DataCopyPad(yGm_[curDst], inLocal, copyOutParams);
        inQueue_.FreeTensor(inLocal);
        copiedSegments += currentSegments;
    }
}

template <typename T>
__aicore__ inline bool Roll<T>::CopyStridedSingleElementByRowGather(
    int64_t dstIndex, int64_t srcBlockBase, int64_t sourceOffset, int64_t blockSize, int64_t blockCount,
    bool preferRowGatherPatch)
{
    if (!preferRowGatherPatch || sizeof(T) != 1 || blockCount <= 1 || blockSize <= 0 || blockSize > 64 ||
        sourceOffset < 0 || sourceOffset >= blockSize) {
        return false;
    }
    if (tilingData_->dimNum == 2 && blockSize == 31) {
        return false;
    }
    const bool useRankGt2NarrowLayout =
        tilingData_->dimNum > 2 && (blockSize == 7 || (blockSize >= 15 && blockSize <= 31));
    const int64_t alignedBlockElements =
        ((blockSize + ROLL_GM_BLOCK_BYTES - 1) / ROLL_GM_BLOCK_BYTES) * ROLL_GM_BLOCK_BYTES;
    if (alignedBlockElements <= 0 || alignedBlockElements > ubElements_) {
        return false;
    }
    int64_t maxRows = ubElements_ / alignedBlockElements;
    if (maxRows > ubElements_) {
        maxRows = ubElements_;
    }
    if (maxRows > ROLL_MAX_DATACOPY_BLOCK_COUNT) {
        maxRows = ROLL_MAX_DATACOPY_BLOCK_COUNT;
    }
    if (blockSize >= 15 && blockSize <= 31) {
        const int64_t rowGatherCap = tilingData_->dimNum == 2 && blockSize == 31
                                         ? 256
                                         : (blockSize == 31 ? 384 : (tilingData_->dimNum == 2 ? 512 : 1024));
        if (maxRows > rowGatherCap) {
            maxRows = rowGatherCap;
        }
    }
    if (maxRows <= 0) {
        return false;
    }

    int64_t copiedRows = 0;
    while (copiedRows < blockCount) {
        int64_t currentRows = blockCount - copiedRows;
        if (currentRows > maxRows) {
            currentRows = maxRows;
        }
        LocalTensor<T> local = inQueue_.AllocTensor<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = static_cast<uint16_t>(currentRows);
        copyInParams.blockLen = static_cast<uint32_t>(blockSize);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, xGm_[srcBlockBase + copiedRows * blockSize], copyInParams, padParams);
        inQueue_.EnQue(local);
        LocalTensor<T> inLocal = inQueue_.DeQue<T>();

        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
        auto inPtr = (__ubuf__ T*)inLocal.GetPhyAddr();
        auto outPtr = (__ubuf__ T*)outLocal.GetPhyAddr();
        for (int64_t row = 0; row < currentRows; ++row) {
            outPtr[row] = inPtr[row * alignedBlockElements + sourceOffset];
        }
        inQueue_.FreeTensor(inLocal);
        outQueue_.EnQue(outLocal);

        LocalTensor<T> result = outQueue_.DeQue<T>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = static_cast<uint32_t>(currentRows * sizeof(T));
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm_[dstIndex + copiedRows], result, copyOutParams);
        outQueue_.FreeTensor(result);
        copiedRows += currentRows;
    }
    return true;
}

template <typename T>
__aicore__ inline bool Roll<T>::CopyBlockRollByFlatPatch(
    int64_t dstIndex, int64_t srcIndex, int64_t blockSize, int64_t firstElements, int64_t secondElements,
    int64_t blockCount)
{
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    if (blockCount <= 0 || blockSize <= 0 || blockSize > ubElements_ || blockSize * typeBytes > ROLL_FLAT_PATCH_MAX_BLOCK_BYTES ||
        blockCount * blockSize * typeBytes < ROLL_FLAT_PATCH_MIN_TOTAL_BYTES) {
        return false;
    }

    const int64_t totalElements = blockCount * blockSize;
    LocalTensor<T> local = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = static_cast<uint32_t>(totalElements * typeBytes);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(local, xGm_[srcIndex], copyInParams, padParams);
    inQueue_.EnQue(local);
    LocalTensor<T> inLocal = inQueue_.DeQue<T>();

    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    auto inPtr = (__ubuf__ T*)inLocal.GetPhyAddr();
    auto outPtr = (__ubuf__ T*)outLocal.GetPhyAddr();
    for (int64_t block = 0; block < blockCount; ++block) {
        int64_t base = block * blockSize;
        for (int64_t i = 0; i < firstElements; ++i) {
            outPtr[base + i] = inPtr[base + secondElements + i];
        }
        for (int64_t i = 0; i < secondElements; ++i) {
            outPtr[base + firstElements + i] = inPtr[base + i];
        }
    }
    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);

    LocalTensor<T> result = outQueue_.DeQue<T>();
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = static_cast<uint32_t>(totalElements * typeBytes);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyPad(yGm_[dstIndex], result, copyOutParams);
    outQueue_.FreeTensor(result);
    return true;
}

template <typename T>
__aicore__ inline void Roll<T>::CopySegment(int64_t dstIndex, int64_t srcIndex, int64_t elementCount)
{
    if (elementCount <= 0) {
        return;
    }
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    const int64_t minStridedBytes = ROLL_STRIDED_SEGMENT_MIN_BYTES > typeBytes ? ROLL_STRIDED_SEGMENT_MIN_BYTES : typeBytes;
    if (elementCount * typeBytes >= minStridedBytes) {
        CopySegmentBySourceAligned(dstIndex, srcIndex, elementCount);
    } else {
        CopySegmentByScalar(dstIndex, srcIndex, elementCount);
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyIdentity()
{
    CopySegment(startIndex_, startIndex_, elementCount_);
}

template <typename T>
__aicore__ inline void Roll<T>::CopyFlattenRoll()
{
    CopySegmentedRoll();
}

template <typename T>
__aicore__ inline void Roll<T>::CopyFlattenRollBySource()
{
    const int64_t shift = tilingData_->shifts[0];
    const int64_t split = tilingData_->totalNum - shift;
    int64_t remain = elementCount_;
    int64_t dst = startIndex_;
    while (remain > 0) {
        int64_t src = 0;
        int64_t current = remain;
        if (dst < shift) {
            src = split + dst;
            int64_t boundary = shift - dst;
            if (current > boundary) {
                current = boundary;
            }
        } else {
            src = dst - shift;
            int64_t boundary = tilingData_->totalNum - dst;
            if (current > boundary) {
                current = boundary;
            }
        }
        CopySegment(dst, src, current);
        dst += current;
        remain -= current;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyLeadingDimRollBySource()
{
    const int64_t blockSize = tilingData_->dimSize * tilingData_->innerSize;
    const int64_t shift = tilingData_->activeShift;
    const int64_t split = (tilingData_->dimSize - shift) * tilingData_->innerSize;
    int64_t remain = elementCount_;
    int64_t dst = startIndex_;
    while (remain > 0) {
        const int64_t blockOffset = dst % blockSize;
        int64_t src = 0;
        int64_t current = remain;
        if (blockOffset < shift * tilingData_->innerSize) {
            src = dst - blockOffset + split + blockOffset;
            int64_t boundary = shift * tilingData_->innerSize - blockOffset;
            if (current > boundary) {
                current = boundary;
            }
        } else {
            src = dst - blockOffset + (blockOffset - shift * tilingData_->innerSize);
            int64_t boundary = blockSize - blockOffset;
            if (current > boundary) {
                current = boundary;
            }
        }
        CopySegment(dst, src, current);
        dst += current;
        remain -= current;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyLastDimRoll()
{
    CopySegmentedRoll();
}

template <typename T>
__aicore__ inline void Roll<T>::CopyLastDimFullRows(int64_t dstIndex, int64_t rowCount)
{
    const int64_t dimSize = tilingData_->dimSize;
    const int64_t shift = tilingData_->activeShift;
    const int64_t firstElements = shift;
    const int64_t secondElements = dimSize - shift;
    const int64_t srcBase = (dstIndex / dimSize) * dimSize;
    if (CopyBlockRollByFlatPatch(dstIndex, srcBase, dimSize, firstElements, secondElements, rowCount)) {
        return;
    }
    CopyLastDimFullRowsBySegments(dstIndex, rowCount);
}

template <typename T>
__aicore__ inline void Roll<T>::CopyRowsRollInUb(
    LocalTensor<T>& outLocal, LocalTensor<T>& inLocal, int64_t rowCount, int64_t dimSize,
    int64_t alignedRowElements, int64_t shift)
{
    auto inPtr = (__ubuf__ T*)inLocal.GetPhyAddr();
    auto outPtr = (__ubuf__ T*)outLocal.GetPhyAddr();
    for (int64_t row = 0; row < rowCount; ++row) {
        auto inRow = inPtr + row * alignedRowElements;
        auto outRow = outPtr + row * dimSize;
        for (int64_t i = 0; i < shift; ++i) {
            outRow[i] = inRow[dimSize - shift + i];
        }
        for (int64_t i = 0; i < dimSize - shift; ++i) {
            outRow[shift + i] = inRow[i];
        }
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyLastDimFullRowsBySegments(int64_t dstIndex, int64_t rowCount)
{
    const int64_t dimSize = tilingData_->dimSize;
    const int64_t shift = tilingData_->activeShift;
    const int64_t firstElements = shift;
    const int64_t secondElements = dimSize - shift;
    const int64_t srcBase = (dstIndex / dimSize) * dimSize;
    CopyStridedSourceSegments(dstIndex, srcBase + secondElements, firstElements, dimSize, rowCount);
    CopyStridedSourceSegments(dstIndex + firstElements, srcBase, secondElements, dimSize, rowCount);
}

template <typename T>
__aicore__ inline void Roll<T>::CopyMultiDimLastDimFullRowsFromSource(
    int64_t dstIndex, int64_t srcRowIndex, int64_t rowCount)
{
    const int64_t dimSize = tilingData_->shapes[tilingData_->dimNum - 1];
    const int64_t shift = tilingData_->shifts[tilingData_->dimNum - 1];
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    const int64_t rowBytes = dimSize * typeBytes;
    const int64_t alignedRowBytes = ((rowBytes + ROLL_GM_BLOCK_BYTES - 1) / ROLL_GM_BLOCK_BYTES) * ROLL_GM_BLOCK_BYTES;
    const int64_t alignedRowElements = alignedRowBytes / typeBytes;

    LocalTensor<T> local = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = static_cast<uint16_t>(rowCount);
    copyInParams.blockLen = static_cast<uint32_t>(rowBytes);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(local, xGm_[srcRowIndex * dimSize], copyInParams, padParams);
    inQueue_.EnQue(local);
    LocalTensor<T> inLocal = inQueue_.DeQue<T>();

    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    CopyRowsRollInUb(outLocal, inLocal, rowCount, dimSize, alignedRowElements, shift);
    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);

    LocalTensor<T> result = outQueue_.DeQue<T>();
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = static_cast<uint32_t>(rowCount * rowBytes);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyPad(yGm_[dstIndex], result, copyOutParams);
    outQueue_.FreeTensor(result);
}

template <typename T>
__aicore__ inline void Roll<T>::CopyMultiDimLastDimFullRowsBySegments(
    int64_t dstIndex, int64_t srcRowIndex, int64_t rowCount)
{
    const int64_t dimSize = tilingData_->shapes[tilingData_->dimNum - 1];
    const int64_t shift = tilingData_->shifts[tilingData_->dimNum - 1];
    const int64_t firstElements = shift;
    const int64_t secondElements = dimSize - shift;
    const int64_t srcBase = srcRowIndex * dimSize;
    if (CopyBlockRollByFlatPatch(dstIndex, srcBase, dimSize, firstElements, secondElements, rowCount)) {
        return;
    }
    CopyStridedSourceSegments(dstIndex, srcBase + secondElements, firstElements, dimSize, rowCount);
    CopyStridedSourceSegments(dstIndex + firstElements, srcBase, secondElements, dimSize, rowCount);
}

template <typename T>
__aicore__ inline void Roll<T>::CopyMultiDimLastDimFullRows(int64_t dstIndex, int64_t rowCount)
{
    const int64_t dimSize = tilingData_->shapes[tilingData_->dimNum - 1];
    const int64_t shift = tilingData_->shifts[tilingData_->dimNum - 1];
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    const int64_t rowBytes = dimSize * typeBytes;
    const int64_t alignedRowBytes = ((rowBytes + ROLL_GM_BLOCK_BYTES - 1) / ROLL_GM_BLOCK_BYTES) * ROLL_GM_BLOCK_BYTES;
    const int64_t alignedRowElements = alignedRowBytes / typeBytes;

    LocalTensor<T> local = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = static_cast<uint32_t>(rowBytes);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    int64_t copiedRows = 0;
    while (copiedRows < rowCount) {
        const int64_t dstRow = dstIndex / dimSize + copiedRows;
        const int64_t srcRow = ComputeInputIndex(dstRow * dimSize) / dimSize;
        int64_t runRows = 1;
        while (copiedRows + runRows < rowCount) {
            const int64_t nextDstRow = dstRow + runRows;
            const int64_t nextSrcRow = ComputeInputIndex(nextDstRow * dimSize) / dimSize;
            if (nextSrcRow != srcRow + runRows) {
                break;
            }
            ++runRows;
        }
        copyInParams.blockCount = static_cast<uint16_t>(runRows);
        DataCopyPad(local[copiedRows * alignedRowElements], xGm_[srcRow * dimSize], copyInParams, padParams);
        copiedRows += runRows;
    }
    inQueue_.EnQue(local);
    LocalTensor<T> inLocal = inQueue_.DeQue<T>();

    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    CopyRowsRollInUb(outLocal, inLocal, rowCount, dimSize, alignedRowElements, shift);
    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);

    LocalTensor<T> result = outQueue_.DeQue<T>();
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = static_cast<uint32_t>(rowCount * rowBytes);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyPad(yGm_[dstIndex], result, copyOutParams);
    outQueue_.FreeTensor(result);
}

template <typename T>
__aicore__ inline void Roll<T>::CopyMultiDimLastDimRollByRows()
{
    const int64_t dimSize = tilingData_->shapes[tilingData_->dimNum - 1];
    const int64_t shift = tilingData_->shifts[tilingData_->dimNum - 1];
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    const int64_t rowBytes = dimSize * typeBytes;
    const int64_t alignedRowBytes = ((rowBytes + ROLL_GM_BLOCK_BYTES - 1) / ROLL_GM_BLOCK_BYTES) * ROLL_GM_BLOCK_BYTES;
    const int64_t alignedRowElements = alignedRowBytes / typeBytes;
    if (dimSize <= 1 || shift <= 0 || alignedRowElements <= 0 || alignedRowElements > ubElements_) {
        CopySegmentedRoll();
        return;
    }
    const int64_t maxInputRows = ubElements_ / alignedRowElements;
    const int64_t maxOutputRows = ubElements_ / dimSize;
    int64_t maxRows = maxInputRows < maxOutputRows ? maxInputRows : maxOutputRows;
    if (maxRows <= 0) {
        CopySegmentedRoll();
        return;
    }
    if (maxRows > 4095) {
        maxRows = 4095;
    }

    int64_t remain = elementCount_;
    int64_t dst = startIndex_;
    while (remain > 0 && (dst % dimSize) != 0) {
        const int64_t src = ComputeInputIndex(dst);
        int64_t current = remain;
        const int64_t rowOffset = dst % dimSize;
        const int64_t srcOffset = src % dimSize;
        const int64_t dstContiguous = dimSize - rowOffset;
        const int64_t srcContiguous = dimSize - srcOffset;
        if (current > dstContiguous) {
            current = dstContiguous;
        }
        if (current > srcContiguous) {
            current = srcContiguous;
        }
        CopySegment(dst, src, current);
        dst += current;
        remain -= current;
    }

    int64_t fullRows = remain / dimSize;
    while (fullRows > 0) {
        int64_t currentRows = fullRows > maxRows ? maxRows : fullRows;
        if (tilingData_->dimNum == 2) {
            const int64_t outerSize = tilingData_->shapes[0];
            const int64_t rowShift = tilingData_->shifts[0];
            const int64_t dstRow = dst / dimSize;
            int64_t srcRow = dstRow - rowShift;
            if (srcRow < 0) {
                srcRow += outerSize;
            }
            const int64_t srcContiguousRows = outerSize - srcRow;
            if (currentRows > srcContiguousRows) {
                currentRows = srcContiguousRows;
            }
            if (IsSameType<T, bfloat16_t>::value && tilingData_->activeDimCount > 1 && dimSize == 3 && rowShift != 0 &&
                shift != 0 && tilingData_->totalNum * static_cast<int64_t>(sizeof(T)) >= 512 &&
                tilingData_->totalNum * static_cast<int64_t>(sizeof(T)) <= 4096) {
                CopyMultiDimLastDimFullRowsFromSource(dst, srcRow, currentRows);
            } else {
                CopyMultiDimLastDimFullRowsBySegments(dst, srcRow, currentRows);
            }
        } else {
            const int64_t dstRow = dst / dimSize;
            int64_t srcRow = ComputeSourceRowIndex(dstRow);
            currentRows = ComputeContiguousSourceRowRun(dstRow, currentRows);
            CopyMultiDimLastDimFullRowsBySegments(dst, srcRow, currentRows);
        }
        const int64_t copied = currentRows * dimSize;
        dst += copied;
        remain -= copied;
        fullRows -= currentRows;
    }

    while (remain > 0) {
        const int64_t src = ComputeInputIndex(dst);
        int64_t current = remain;
        const int64_t rowOffset = dst % dimSize;
        const int64_t srcOffset = src % dimSize;
        const int64_t dstContiguous = dimSize - rowOffset;
        const int64_t srcContiguous = dimSize - srcOffset;
        if (current > dstContiguous) {
            current = dstContiguous;
        }
        if (current > srcContiguous) {
            current = srcContiguous;
        }
        CopySegment(dst, src, current);
        dst += current;
        remain -= current;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyMultiDimNonLastBlockPartial(
    int64_t lastActiveDim, int64_t& dstIndex, int64_t& remain)
{
    const int64_t inner = tilingData_->strides[lastActiveDim];
    const int64_t dimSize = tilingData_->shapes[lastActiveDim];
    const int64_t shift = tilingData_->shifts[lastActiveDim];
    const int64_t blockSize = dimSize * inner;
    const int64_t firstElements = shift * inner;
    const int64_t secondElements = (dimSize - shift) * inner;

    const int64_t dstBlock = dstIndex / blockSize;
    const int64_t srcBlock = ComputeSourceBlockIndex(dstBlock, lastActiveDim);
    const int64_t inBlockOffset = dstIndex % blockSize;
    int64_t src = 0;
    int64_t contiguous = 0;
    if (inBlockOffset < firstElements) {
        src = srcBlock * blockSize + secondElements + inBlockOffset;
        contiguous = firstElements - inBlockOffset;
    } else {
        src = srcBlock * blockSize + inBlockOffset - firstElements;
        contiguous = blockSize - inBlockOffset;
    }
    int64_t current = remain;
    if (current > contiguous) {
        current = contiguous;
    }
    CopySegment(dstIndex, src, current);
    dstIndex += current;
    remain -= current;
}

template <typename T>
__aicore__ inline void Roll<T>::CopyMultiDimNonLastFullBlocks(
    int64_t lastActiveDim, int64_t dstIndex, int64_t srcBlockIndex, int64_t blockCount)
{
    const int64_t inner = tilingData_->strides[lastActiveDim];
    const int64_t dimSize = tilingData_->shapes[lastActiveDim];
    const int64_t shift = tilingData_->shifts[lastActiveDim];
    const int64_t blockSize = dimSize * inner;
    const int64_t firstElements = shift * inner;
    const int64_t secondElements = (dimSize - shift) * inner;
    const int64_t srcIndex = srcBlockIndex * blockSize;
    if (CopyBlockRollByFlatPatch(dstIndex, srcIndex, blockSize, firstElements, secondElements, blockCount)) {
        return;
    }
    if (firstElements <= ubElements_ && secondElements <= ubElements_) {
        CopyStridedSourceSegments(dstIndex, srcIndex + secondElements, firstElements, blockSize, blockCount);
        CopyStridedSourceSegments(dstIndex + firstElements, srcIndex, secondElements, blockSize, blockCount);
        return;
    }
    int64_t copiedBlocks = 0;
    while (copiedBlocks < blockCount) {
        const int64_t dst = dstIndex + copiedBlocks * blockSize;
        const int64_t src = srcIndex + copiedBlocks * blockSize;
        CopySegment(dst, src + secondElements, firstElements);
        CopySegment(dst + firstElements, src, secondElements);
        ++copiedBlocks;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyMultiDimNonLastRollByBlocks(int64_t lastActiveDim)
{
    const int64_t inner = tilingData_->strides[lastActiveDim];
    const int64_t dimSize = tilingData_->shapes[lastActiveDim];
    const int64_t shift = tilingData_->shifts[lastActiveDim];
    const int64_t blockSize = dimSize * inner;
    if (inner <= 0 || dimSize <= 1 || shift <= 0 || blockSize <= 0) {
        CopySegmentedRoll();
        return;
    }

    int64_t remain = elementCount_;
    int64_t dst = startIndex_;
    while (remain > 0 && (dst % blockSize) != 0) {
        CopyMultiDimNonLastBlockPartial(lastActiveDim, dst, remain);
    }

    int64_t fullBlocks = remain / blockSize;
    while (fullBlocks > 0) {
        const int64_t dstBlock = dst / blockSize;
        int64_t srcBlock = ComputeSourceBlockIndex(dstBlock, lastActiveDim);
        int64_t runBlocks = ComputeContiguousSourceBlockRun(dstBlock, fullBlocks, lastActiveDim);
        if (runBlocks > 4095) {
            runBlocks = 4095;
        }
        CopyMultiDimNonLastFullBlocks(lastActiveDim, dst, srcBlock, runBlocks);
        const int64_t copied = runBlocks * blockSize;
        dst += copied;
        remain -= copied;
        fullBlocks -= runBlocks;
    }

    while (remain > 0) {
        CopyMultiDimNonLastBlockPartial(lastActiveDim, dst, remain);
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyLastDimPartial(int64_t& dstIndex, int64_t& remain)
{
    const int64_t dimSize = tilingData_->dimSize;
    const int64_t shift = tilingData_->activeShift;
    const int64_t secondElements = dimSize - shift;
    const int64_t rowBase = (dstIndex / dimSize) * dimSize;
    const int64_t rowOffset = dstIndex % dimSize;
    int64_t src = 0;
    int64_t contiguous = 0;
    if (rowOffset < shift) {
        src = rowBase + secondElements + rowOffset;
        contiguous = shift - rowOffset;
    } else {
        src = rowBase + rowOffset - shift;
        contiguous = dimSize - rowOffset;
    }
    int64_t current = remain;
    if (current > contiguous) {
        current = contiguous;
    }
    CopySegment(dstIndex, src, current);
    dstIndex += current;
    remain -= current;
}

template <typename T>
__aicore__ inline void Roll<T>::CopySingleDimRoll()
{
    CopySegmentedRoll();
}

template <typename T>
__aicore__ inline void Roll<T>::CopySingleDimFullBlocks(int64_t dstIndex, int64_t blockCount)
{
    const int64_t blockSize = tilingData_->dimSize * tilingData_->innerSize;
    const int64_t firstElements = tilingData_->activeShift * tilingData_->innerSize;
    const int64_t secondElements = blockSize - firstElements;
    const int64_t srcIndex = ComputeInputIndex(dstIndex);
    if (CopyBlockRollByFlatPatch(dstIndex, srcIndex, blockSize, firstElements, secondElements, blockCount)) {
        return;
    }
    CopyStridedSourceSegments(dstIndex, srcIndex + secondElements, firstElements, blockSize, blockCount);
    CopyStridedSourceSegments(dstIndex + firstElements, srcIndex, secondElements, blockSize, blockCount);
}

template <typename T>
__aicore__ inline void Roll<T>::CopySingleDimPartial(int64_t& dstIndex, int64_t& remain)
{
    const int64_t blockSize = tilingData_->dimSize * tilingData_->innerSize;
    const int64_t firstElements = tilingData_->activeShift * tilingData_->innerSize;
    const int64_t secondElements = blockSize - firstElements;
    const int64_t blockOffset = dstIndex % blockSize;
    int64_t src = 0;
    int64_t contiguous = 0;
    if (blockOffset < firstElements) {
        src = dstIndex - blockOffset + secondElements + blockOffset;
        contiguous = firstElements - blockOffset;
    } else {
        src = dstIndex - blockOffset + blockOffset - firstElements;
        contiguous = blockSize - blockOffset;
    }
    int64_t current = remain;
    if (current > contiguous) {
        current = contiguous;
    }
    CopySegment(dstIndex, src, current);
    dstIndex += current;
    remain -= current;
}

template <typename T>
__aicore__ inline void Roll<T>::CopySingleDimRollByBlocks()
{
    const int64_t blockSize = tilingData_->dimSize * tilingData_->innerSize;
    if (blockSize <= 0 || tilingData_->activeShift <= 0 || tilingData_->dimSize <= 1) {
        CopySingleDimRoll();
        return;
    }

    int64_t remain = elementCount_;
    int64_t dst = startIndex_;
    while (remain > 0 && (dst % blockSize) != 0) {
        CopySingleDimPartial(dst, remain);
    }

    int64_t fullBlocks = remain / blockSize;
    while (fullBlocks > 0) {
        int64_t currentBlocks = fullBlocks > 4095 ? 4095 : fullBlocks;
        CopySingleDimFullBlocks(dst, currentBlocks);
        const int64_t copied = currentBlocks * blockSize;
        dst += copied;
        remain -= copied;
        fullBlocks -= currentBlocks;
    }

    while (remain > 0) {
        CopySingleDimPartial(dst, remain);
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyLastDimRollByRows()
{
    const int64_t dimSize = tilingData_->dimSize;
    const int64_t shift = tilingData_->activeShift;
    const int64_t typeBytes = static_cast<int64_t>(sizeof(T));
    const int64_t rowBytes = dimSize * typeBytes;
    const int64_t alignedRowBytes = ((rowBytes + ROLL_GM_BLOCK_BYTES - 1) / ROLL_GM_BLOCK_BYTES) * ROLL_GM_BLOCK_BYTES;
    const int64_t alignedRowElements = alignedRowBytes / typeBytes;
    if (dimSize <= 1 || shift <= 0 || alignedRowElements <= 0 || alignedRowElements > ubElements_) {
        CopyLastDimRoll();
        return;
    }

    const int64_t maxInputRows = ubElements_ / alignedRowElements;
    const int64_t maxOutputRows = ubElements_ / dimSize;
    int64_t maxRows = maxInputRows < maxOutputRows ? maxInputRows : maxOutputRows;
    if (maxRows <= 0) {
        CopyLastDimRoll();
        return;
    }
    if (maxRows > 4095) {
        maxRows = 4095;
    }

    int64_t remain = elementCount_;
    int64_t dst = startIndex_;
    while (remain > 0 && (dst % dimSize) != 0) {
        CopyLastDimPartial(dst, remain);
    }

    int64_t fullRows = remain / dimSize;
    while (fullRows > 0) {
        int64_t currentRows = fullRows > maxRows ? maxRows : fullRows;
        const int64_t tailElements = dimSize - shift;
        const int64_t largeSegmentBytes = (shift > tailElements ? shift : tailElements) * typeBytes;
        const bool preferSegmentPath = (largeSegmentBytes >= ROLL_STRIDED_SEGMENT_MIN_BYTES) ||
                                       (sizeof(T) == 1 && dimSize >= 7 && dimSize <= 31);
        if (preferSegmentPath) {
            CopyLastDimFullRowsBySegments(dst, currentRows);
        } else {
            CopyLastDimFullRows(dst, currentRows);
        }
        const int64_t copied = currentRows * dimSize;
        dst += copied;
        remain -= copied;
        fullRows -= currentRows;
    }

    while (remain > 0) {
        CopyLastDimPartial(dst, remain);
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopySegmentedRoll()
{
    int64_t lastActiveDim = -1;
    for (int64_t dim = 0; dim < tilingData_->dimNum; ++dim) {
        if (tilingData_->shifts[dim] != 0) {
            lastActiveDim = dim;
        }
    }
    if (lastActiveDim < 0) {
        CopyIdentity();
        return;
    }

    int64_t remain = elementCount_;
    int64_t dst = startIndex_;
    while (remain > 0) {
        const int64_t src = ComputeInputIndex(dst);
        int64_t current = remain;
        if (lastActiveDim == tilingData_->dimNum - 1) {
            const int64_t dimSize = tilingData_->shapes[lastActiveDim];
            const int64_t shift = tilingData_->shifts[lastActiveDim];
            const int64_t dstOffset = dst % dimSize;
            const int64_t srcOffset = (dstOffset - shift + dimSize) % dimSize;
            const int64_t dstContiguous = dimSize - dstOffset;
            const int64_t srcContiguous = dimSize - srcOffset;
            if (current > dstContiguous) {
                current = dstContiguous;
            }
            if (current > srcContiguous) {
                current = srcContiguous;
            }
        } else {
            const int64_t contiguous = tilingData_->strides[lastActiveDim];
            const int64_t offset = dst % contiguous;
            const int64_t dstContiguous = contiguous - offset;
            if (current > dstContiguous) {
                current = dstContiguous;
            }
        }
        CopySegment(dst, src, current);
        dst += current;
        remain -= current;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessScalar()
{
    CopySegmentedRoll();
}

template <typename T>
__aicore__ inline void Roll<T>::Process()
{
    if (elementCount_ <= 0 || tilingData_->totalNum <= 0) {
        return;
    }
    if (tilingData_->activeDimCount == 0) {
        CopyIdentity();
    } else if (tilingData_->dimNum == 1 && tilingData_->strides[0] == 1) {
        CopyFlattenRollBySource();
    } else if (tilingData_->activeDimCount == 1 && tilingData_->activeDim == 0 && tilingData_->innerSize > 0 &&
               (tilingData_->dimNum == 2 || tilingData_->dimSize <= 4 ||
                (IsSameType<T, uint8_t>::value && tilingData_->dimNum > 2 &&
                 tilingData_->innerSize % ROLL_GM_BLOCK_BYTES == 0 &&
                 tilingData_->totalNum >= 32 * 1024 * 1024)) &&
               !(sizeof(T) == 1 && tilingData_->dimNum == 2 && tilingData_->innerSize == 127)) {
        CopyLeadingDimRollBySource();
    } else if (tilingData_->activeDimCount == 1 && tilingData_->innerSize == 1 &&
               tilingData_->activeDim == tilingData_->dimNum - 1) {
        CopyLastDimRollByRows();
    } else if (tilingData_->activeDimCount == 1 && tilingData_->innerSize > 0) {
        CopySingleDimRollByBlocks();
    } else {
        int64_t lastActiveDim = -1;
        for (int64_t dim = 0; dim < tilingData_->dimNum; ++dim) {
            if (tilingData_->shifts[dim] != 0) {
                lastActiveDim = dim;
            }
        }
        if (lastActiveDim == tilingData_->dimNum - 1) {
            CopyMultiDimLastDimRollByRows();
            return;
        } else if (lastActiveDim >= 0) {
            CopyMultiDimNonLastRollByBlocks(lastActiveDim);
            return;
        }
        CopySegmentedRoll();
    }
}

} // namespace RollKernel

#endif // ROLL_H_
