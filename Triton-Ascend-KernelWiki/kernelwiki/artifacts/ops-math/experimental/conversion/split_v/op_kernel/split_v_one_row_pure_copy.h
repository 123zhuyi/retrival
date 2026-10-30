/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_ONE_ROW_PURE_COPY_H_
#define OP_KERNEL_SPLIT_V_ONE_ROW_PURE_COPY_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVOneRowPureCopy {
public:
    __aicore__ inline SplitVOneRowPureCopy() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const SplitVTilingDataOneRowPureCopy* tiling,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const;
    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) const;
    __aicore__ inline void LocateTask(uint64_t taskIdx, uint32_t& splitIdx, uint32_t& chunkIdx,
                                      uint32_t& splitStart) const;
    __aicore__ inline void CopyOne(uint32_t splitIdx, uint32_t splitStart, uint32_t chunkOffset, uint32_t copyLength);

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;

    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queue;

    uint64_t totalLength = 0;
    uint32_t splitNum = 0;
    uint32_t sizeSplits[maxSplitNum];

    uint64_t loopOffset = 0;
    uint64_t loopNum = 0;
    uint32_t chunkLength = 0;
    uint32_t chunkPitch = 0;
    uint32_t alignedNum = 1;
    uint32_t blockIdx = 0;
    uint32_t blockNum = 0;
};

template <typename T>
__aicore__ inline uint32_t SplitVOneRowPureCopy<T>::AlignUp(uint32_t value, uint32_t align) const
{
    return (value + align - 1U) & ~(align - 1U);
}

template <typename T>
__aicore__ inline uint32_t SplitVOneRowPureCopy<T>::MinU32(uint32_t a, uint32_t b) const
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline void SplitVOneRowPureCopy<T>::Init(GM_ADDR input, GM_ADDR output,
                                                     const SplitVTilingDataOneRowPureCopy* tiling, TPipe* pipe)
{
    blockIdx = GetBlockIdx();
    blockNum = GetBlockNum();
    totalLength = tiling->totalLength;
    splitNum = tiling->splitNum;
    chunkLength = tiling->chunkLength;
    alignedNum = BLOCK_SIZE / sizeof(T);
    if (alignedNum == 0) {
        alignedNum = 1;
    }
    chunkPitch = AlignUp(chunkLength, alignedNum);

    for (uint32_t i = 0; i < splitNum; ++i) {
        sizeSplits[i] = tiling->sizeSplits[i];
    }

    if (blockIdx < tiling->formerNum) {
        loopNum = tiling->formerTaskNum;
        loopOffset = static_cast<uint64_t>(blockIdx) * tiling->formerTaskNum;
    } else {
        loopNum = tiling->tailTaskNum;
        loopOffset = tiling->formerNum * tiling->formerTaskNum +
                     static_cast<uint64_t>(blockIdx - tiling->formerNum) * tiling->tailTaskNum;
    }

    inputGm.SetGlobalBuffer((__gm__ T*)input, totalLength);
    outList.Init((__gm__ void*)output);
    pipe->InitBuffer(queue, BUFFER_NUM, chunkPitch * sizeof(T));
}

template <typename T>
__aicore__ inline void SplitVOneRowPureCopy<T>::LocateTask(uint64_t taskIdx, uint32_t& splitIdx, uint32_t& chunkIdx,
                                                           uint32_t& splitStart) const
{
    uint64_t taskBase = 0;
    splitStart = 0;
    for (splitIdx = 0; splitIdx < splitNum; ++splitIdx) {
        const uint32_t splitLen = sizeSplits[splitIdx];
        const uint64_t splitTaskNum = (static_cast<uint64_t>(splitLen) + chunkLength - 1U) / chunkLength;
        if (taskIdx < taskBase + splitTaskNum) {
            chunkIdx = static_cast<uint32_t>(taskIdx - taskBase);
            return;
        }
        taskBase += splitTaskNum;
        splitStart += splitLen;
    }
    chunkIdx = 0;
}

template <typename T>
__aicore__ inline void SplitVOneRowPureCopy<T>::Process()
{
    if (blockIdx >= blockNum || splitNum == 0 || chunkLength == 0 || loopNum == 0) {
        return;
    }

    uint32_t splitIdx = 0;
    uint32_t chunkIdx = 0;
    uint32_t splitStart = 0;
    LocateTask(loopOffset, splitIdx, chunkIdx, splitStart);
    for (uint64_t i = 0; i < loopNum && splitIdx < splitNum; ++i) {
        const uint32_t splitLen = sizeSplits[splitIdx];
        const uint32_t chunkOffset = static_cast<uint32_t>(static_cast<uint64_t>(chunkIdx) * chunkLength);
        const uint32_t remain = splitLen - chunkOffset;
        const uint32_t copyLength = MinU32(remain, chunkLength);
        CopyOne(splitIdx, splitStart, chunkOffset, copyLength);
        ++chunkIdx;
        if (static_cast<uint64_t>(chunkIdx) * chunkLength >= splitLen) {
            splitStart += splitLen;
            ++splitIdx;
            chunkIdx = 0;
        }
    }
}

template <typename T>
__aicore__ inline void SplitVOneRowPureCopy<T>::CopyOne(uint32_t splitIdx, uint32_t splitStart, uint32_t chunkOffset,
                                                        uint32_t copyLength)
{
    LocalTensor<T> local = queue.AllocTensor<T>();
    const uint32_t alignedLength = AlignUp(copyLength, alignedNum);
    const uint64_t srcOffset = static_cast<uint64_t>(splitStart) + chunkOffset;
    const uint64_t srcByteOffset = srcOffset * sizeof(T);
    const uint64_t dstByteOffset = static_cast<uint64_t>(chunkOffset) * sizeof(T);
    const bool canCopyInDirect = copyLength == alignedLength && (srcByteOffset & (BLOCK_SIZE - 1U)) == 0;
    const bool canCopyOutDirect = copyLength == alignedLength && (dstByteOffset & (BLOCK_SIZE - 1U)) == 0;
    if (canCopyInDirect) {
        DataCopy(local, inputGm[srcOffset], copyLength);
    } else {
        DataCopyParams copyParams = {1, 0, 0, 0};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        copyParams.blockLen = copyLength * sizeof(T);
        padParams.rightPadding = alignedLength - copyLength;
        DataCopyPad(local, inputGm[srcOffset], copyParams, padParams);
    }
    queue.EnQue<T>(local);

    LocalTensor<T> outLocal = queue.DeQue<T>();
    outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(splitIdx), sizeSplits[splitIdx]);
    if (canCopyOutDirect) {
        DataCopy(outputGm[chunkOffset], outLocal, copyLength);
    } else {
        DataCopyParams copyParams = {1, 0, 0, 0};
        copyParams.blockLen = copyLength * sizeof(T);
        DataCopyPad(outputGm[chunkOffset], outLocal, copyParams);
    }
    queue.FreeTensor(outLocal);
}

#endif // OP_KERNEL_SPLIT_V_ONE_ROW_PURE_COPY_H_
