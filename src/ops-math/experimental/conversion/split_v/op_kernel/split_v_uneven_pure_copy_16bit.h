/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_UNEVEN_PURE_COPY_16BIT_H_
#define OP_KERNEL_SPLIT_V_UNEVEN_PURE_COPY_16BIT_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVUnevenPureCopy16Bit : private SplitVPureCopyState<T, 1> {
public:
    __aicore__ inline SplitVUnevenPureCopy16Bit() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR yList, const SplitVTilingDataUnevenPureCopy16Bit* tilingData,
                                TPipe* pipe)
    {
        blockIdx = GetBlockIdx();
        totalLength = tilingData->totalLength;
        outerLength = tilingData->outerLength;
        rowLength = tilingData->rowLength;
        splitNum = tilingData->splitNum;
        maxSplitSize = tilingData->maxSplitSize;
        for (uint32_t i = 0; i < splitNum; ++i) {
            sizeSplits[i] = tilingData->sizeSplits[i];
            splitStarts[i] = tilingData->splitStarts[i];
        }
        LoadPureCopySchedule(tilingData);

        const uint32_t bufferElems = mode == MODE_SPLIT_MAJOR ? outerTile * splitPitch : colTilePitch;
        InitPureCopyBuffer(x, yList, bufferElems, pipe);
    }

    __aicore__ inline void Process()
    {
        if (!PreparePureCopyCore(IsValidTiling())) {
            return;
        }
        if (mode == MODE_SPLIT_MAJOR) {
            ProcessSplitMajor();
        } else {
            ProcessRowLengthChunk();
        }
    }

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t TQUE_BIND_DEPTH = 1;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t MODE_SPLIT_MAJOR = 0;
    static constexpr uint32_t MODE_ROW_LENGTH_CHUNK = 1;
    static constexpr uint32_t B16_ELEMS_PER_BLOCK = 16;

    using Base = SplitVPureCopyState<T, TQUE_BIND_DEPTH>;
    SPLIT_V_USE_PURE_COPY_STATE(Base);

    __aicore__ inline bool IsAligned32Bytes(uint32_t bytes) const { return (bytes & (BLOCK_SIZE - 1U)) == 0; }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const { return SplitVAlignUp(value, align); }

    __aicore__ inline bool IsValidTiling() const
    {
        if (splitNum == 0) {
            return false;
        }
        return IsPureCopyScheduleValid(maxSplitSize,
                                       splitPitch >= maxSplitSize && IsAligned32Bytes(splitPitch * sizeof(T)),
                                       IsAligned32Bytes(colTilePitch * sizeof(T)));
    }

    __aicore__ inline uint32_t GetSplitPitch(uint32_t splitLen) const { return AlignUp(splitLen, B16_ELEMS_PER_BLOCK); }

    __aicore__ inline LocalTensor<T> CopyInSplitMajor(uint64_t outerBase, uint32_t outerReal, uint32_t splitIndex,
                                                      uint32_t splitLen, uint32_t curPitch)
    {
        return CopyInPureSplitMajor(outerBase, outerReal, splitStarts[splitIndex], splitLen, curPitch);
    }

    __aicore__ inline void CopyOutSplitMajor(const LocalTensor<T>& local, uint64_t outerBase, uint32_t outerReal,
                                             uint32_t splitIndex, uint32_t splitLen)
    {
        CopyOutPureSplitMajor(local, outerBase, outerReal, splitIndex, splitLen);
    }

    __aicore__ inline void ProcessSplitMajor()
    {
        for (uint32_t outerOff = 0; outerOff < coreRows; outerOff += outerTile) {
            const uint32_t remainRows = coreRows - outerOff;
            const uint32_t outerReal = remainRows > outerTile ? outerTile : remainRows;
            const uint64_t outerBase = coreOuterBase + outerOff;
            for (uint32_t splitIndex = 0; splitIndex < splitNum; ++splitIndex) {
                const uint32_t splitLen = sizeSplits[splitIndex];
                const uint32_t curPitch = GetSplitPitch(splitLen);
                LocalTensor<T> local = CopyInSplitMajor(outerBase, outerReal, splitIndex, splitLen, curPitch);
                CopyOutSplitMajor(local, outerBase, outerReal, splitIndex, splitLen);
                queue.FreeTensor(local);
            }
        }
    }

    __aicore__ inline LocalTensor<T> CopyInRowChunk(uint64_t srcOffset, uint32_t copyElems)
    {
        return CopyInPureRowChunk(srcOffset, copyElems, colTilePitch);
    }

    __aicore__ inline void CopyOutRowChunk(const LocalTensor<T>& local, uint64_t outerIndex, uint32_t splitIndex,
                                           uint32_t splitOffset, uint32_t copyElems)
    {
        const uint32_t splitLen = sizeSplits[splitIndex];
        outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIndex),
                                 outerLength * static_cast<uint64_t>(splitLen));
        const uint64_t dstOffset = outerIndex * static_cast<uint64_t>(splitLen) + splitOffset;
        CopyOutPureRowChunk(local, outputGm, dstOffset, copyElems);
    }

    __aicore__ inline void ProcessRowLengthChunk()
    {
        for (uint32_t rowOff = 0; rowOff < coreRows; ++rowOff) {
            const uint64_t outerIndex = coreOuterBase + rowOff;
            const uint64_t rowBase = outerIndex * static_cast<uint64_t>(rowLength);
            for (uint32_t splitIndex = 0; splitIndex < splitNum; ++splitIndex) {
                const uint32_t splitLen = sizeSplits[splitIndex];
                const uint32_t splitBase = splitStarts[splitIndex];
                for (uint32_t splitOff = 0; splitOff < splitLen; splitOff += colTileLength) {
                    const uint32_t remain = splitLen - splitOff;
                    const uint32_t copyElems = remain > colTileLength ? colTileLength : remain;
                    LocalTensor<T> local = CopyInRowChunk(rowBase + splitBase + splitOff, copyElems);
                    CopyOutRowChunk(local, outerIndex, splitIndex, splitOff, copyElems);
                    queue.FreeTensor(local);
                }
            }
        }
    }

    uint32_t splitNum = 0;
    uint32_t sizeSplits[maxSplitNum];
    uint32_t splitStarts[maxSplitNum];
    uint32_t maxSplitSize = 0;
};

#endif // OP_KERNEL_SPLIT_V_UNEVEN_PURE_COPY_16BIT_H_
