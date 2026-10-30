/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_UNEVEN_COMPACT_H_
#define OP_KERNEL_SPLIT_V_UNEVEN_COMPACT_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVUnevenCompact : private SplitVUnevenCompactTilingState {
public:
    __aicore__ inline SplitVUnevenCompact() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataUnevenCompact* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t TRANS_BLOCK = 16;
    static constexpr uint32_t MODE_FULL_ROW = 0;
    static constexpr uint32_t MODE_ROW_CHUNK = 1;
    static constexpr uint32_t MODE_PURE_COPY = 2;

    using Base = SplitVUnevenCompactTilingState;
    SPLIT_V_USE_UNEVEN_COMPACT_TILING_STATE(Base);

    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queue;
    TBuf<TPosition::VECCALC> transBuf;
    TBuf<TPosition::VECCALC> segBuf;
    LocalTensor<T> transTensor;
    LocalTensor<T> segTensor;

    uint32_t rowPitch = 0;

    DataCopyExtParams copyInParam{0, 0, 0, 0, 0};
    DataCopyExtParams copyOutParam{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> copyInPadParam{false, 0, 0, 0};
    DataCopyParams segCopyParam{0, 0, 0, 0};

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const;
    __aicore__ inline uint32_t Max(uint32_t a, uint32_t b) const;
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const;
    template <typename VT>
    __aicore__ inline void ProcessFullRowTyped(uint64_t outerBase, uint32_t outerReal);
    template <typename VT>
    __aicore__ inline void ProcessChunkTaskTyped(uint64_t taskIdx);
    __aicore__ inline void ProcessPureTask(uint64_t tileIdx);
    __aicore__ inline void CopyOutCompact(const LocalTensor<T>& local, uint64_t outerBase, uint32_t outerReal,
                                          uint32_t splitIdx, uint32_t splitLen);
    __aicore__ inline void WaitMte2ToV();
    __aicore__ inline void WaitVToMte2();
    __aicore__ inline void WaitVToMte3();
    __aicore__ inline void WaitMte3ToV();
};

template <typename T>
__aicore__ inline uint32_t SplitVUnevenCompact<T>::Min(uint32_t a, uint32_t b) const
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline uint32_t SplitVUnevenCompact<T>::Max(uint32_t a, uint32_t b) const
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline uint32_t SplitVUnevenCompact<T>::AlignUp(uint32_t value, uint32_t align) const
{
    return SplitVAlignUp(value, align);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::Init(GM_ADDR input, GM_ADDR output,
                                                    SplitVTilingDataUnevenCompact* tiling, TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= GetBlockNum()) {
        return;
    }

    outList.Init((__gm__ void*)output);
    LoadUnevenCompactTiling(tiling, blockIdx);
    rowPitch = tiling->rowTransLen == 0 ? rowLength : tiling->rowTransLen / TRANS_BLOCK;
    colChunkNum = colChunkNum == 0 ? 1 : colChunkNum;

    inputGm.SetGlobalBuffer((__gm__ T*)input, totalLength);

    const uint32_t stagePitch = rowPitch == 0 ? rowLength : rowPitch;
    const uint32_t inputElems = outerTile * stagePitch;
    const uint32_t transElems = mode == MODE_PURE_COPY ? TRANS_BLOCK : outerTile * stagePitch;
    const uint32_t segElems = mode == MODE_PURE_COPY ? TRANS_BLOCK : outerTile * stagePitch;
    pipe->InitBuffer(queue, BUFFER_NUM, inputElems * sizeof(T));
    pipe->InitBuffer(transBuf, transElems * sizeof(T));
    pipe->InitBuffer(segBuf, segElems * sizeof(T));
    transTensor = transBuf.Get<T>();
    segTensor = segBuf.Get<T>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::WaitMte2ToV()
{
    SplitVSync<HardEvent::MTE2_V>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::WaitVToMte2()
{
    SplitVSync<HardEvent::V_MTE2>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::WaitVToMte3()
{
    SplitVSync<HardEvent::V_MTE3>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::WaitMte3ToV()
{
    SplitVSync<HardEvent::MTE3_V>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::CopyOutCompact(const LocalTensor<T>& local, uint64_t outerBase,
                                                              uint32_t outerReal, uint32_t splitIdx, uint32_t splitLen)
{
    outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx),
                             outerLength * static_cast<uint64_t>(splitLen));
    const uint32_t outputElems = outerReal * splitLen;
    if (outerReal == outerTile) {
        DataCopy(outputGm[outerBase * static_cast<uint64_t>(splitLen)], local, outputElems);
    } else {
        copyOutParam.blockCount = 1;
        copyOutParam.blockLen = outputElems * sizeof(T);
        copyOutParam.srcStride = 0;
        copyOutParam.dstStride = 0;
        DataCopyPad(outputGm[outerBase * static_cast<uint64_t>(splitLen)], local, copyOutParam);
    }
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVUnevenCompact<T>::ProcessFullRowTyped(uint64_t outerBase, uint32_t outerReal)
{
    LocalTensor<T> inLocalRaw = queue.AllocTensor<T>();
    const uint32_t inputElems = outerReal * rowLength;
    if (outerReal == outerTile) {
        DataCopy(inLocalRaw, inputGm[outerBase * static_cast<uint64_t>(rowLength)], inputElems);
    } else {
        copyInParam.blockCount = 1;
        copyInParam.blockLen = inputElems * sizeof(T);
        copyInParam.srcStride = 0;
        copyInParam.dstStride = 0;
        DataCopyPad(inLocalRaw, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyInParam, copyInPadParam);
    }
    queue.EnQue<T>(inLocalRaw);

    LocalTensor<T> inReadyRaw = queue.DeQue<T>();
    WaitMte2ToV();
    LocalTensor<VT> inBits = inReadyRaw.template ReinterpretCast<VT>();
    LocalTensor<VT> transBits = transTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> segBits = segTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> outBits = inReadyRaw.template ReinterpretCast<VT>();

    SplitVFirstVnchw<VT, TRANS_BLOCK>(inBits, transBits, rowLength);

    for (uint32_t g = 0; g < splitNum; ++g) {
        const uint32_t splitLen = sizeSplits[g];
        const uint32_t srcOffset = splitStarts[g] * TRANS_BLOCK;

        segCopyParam.blockCount = TRANS_BLOCK;
        segCopyParam.blockLen = splitLen;
        segCopyParam.srcStride = rowLength - splitLen;
        segCopyParam.dstStride = 0;

        WaitVToMte2();
        DataCopy(segBits, transBits[srcOffset], segCopyParam);
        WaitMte2ToV();

        SplitVSecondVnchw<VT, TRANS_BLOCK>(segBits, outBits, splitLen);
        WaitVToMte3();

        CopyOutCompact(inReadyRaw, outerBase, outerReal, g, splitLen);
        WaitMte3ToV();
    }
    queue.FreeTensor(inReadyRaw);
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVUnevenCompact<T>::ProcessChunkTaskTyped(uint64_t taskIdx)
{
    const uint64_t outerTileIdx = taskIdx / colChunkNum;
    const uint32_t splitIdx = static_cast<uint32_t>(taskIdx - outerTileIdx * colChunkNum);
    if (splitIdx >= splitNum) {
        return;
    }

    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint32_t splitLen = sizeSplits[splitIdx];
    const uint32_t splitStart = splitStarts[splitIdx];

    LocalTensor<T> inLocalRaw = queue.AllocTensor<T>();
    copyInParam.blockCount = static_cast<uint16_t>(outerReal);
    copyInParam.blockLen = splitLen * sizeof(T);
    copyInParam.srcStride = (rowLength - splitLen) * sizeof(T);
    copyInParam.dstStride = 0;
    copyInPadParam.isPad = rowPitch != splitLen;
    copyInPadParam.leftPadding = 0;
    copyInPadParam.rightPadding = static_cast<uint8_t>(rowPitch - splitLen);
    copyInPadParam.paddingValue = 0;
    const uint64_t srcOffset = outerBase * static_cast<uint64_t>(rowLength) + splitStart;
    DataCopyPad(inLocalRaw, inputGm[srcOffset], copyInParam, copyInPadParam);
    queue.EnQue<T>(inLocalRaw);

    LocalTensor<T> inReadyRaw = queue.DeQue<T>();
    WaitMte2ToV();
    LocalTensor<VT> inBits = inReadyRaw.template ReinterpretCast<VT>();
    LocalTensor<VT> transBits = transTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> segBits = segTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> outBits = inReadyRaw.template ReinterpretCast<VT>();

    SplitVFirstVnchw<VT, TRANS_BLOCK>(inBits, transBits, rowPitch);

    segCopyParam.blockCount = TRANS_BLOCK;
    segCopyParam.blockLen = splitLen;
    segCopyParam.srcStride = rowPitch - splitLen;
    segCopyParam.dstStride = 0;

    WaitVToMte2();
    DataCopy(segBits, transBits, segCopyParam);
    WaitMte2ToV();

    SplitVSecondVnchw<VT, TRANS_BLOCK>(segBits, outBits, splitLen);
    WaitVToMte3();

    CopyOutCompact(inReadyRaw, outerBase, outerReal, splitIdx, splitLen);
    WaitMte3ToV();
    queue.FreeTensor(inReadyRaw);
}
template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::ProcessPureTask(uint64_t tileIdx)
{
    const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
    const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

    for (uint32_t g = 0; g < splitNum; ++g) {
        const uint32_t splitLen = sizeSplits[g];
        const uint32_t splitStart = splitStarts[g];
        LocalTensor<T> local = queue.AllocTensor<T>();

        copyInParam.blockCount = static_cast<uint16_t>(outerReal);
        copyInParam.blockLen = splitLen * sizeof(T);
        copyInParam.srcStride = (rowLength - splitLen) * sizeof(T);
        copyInParam.dstStride = 0;
        copyInPadParam.isPad = rowPitch != splitLen;
        copyInPadParam.leftPadding = 0;
        copyInPadParam.rightPadding = static_cast<uint8_t>(rowPitch - splitLen);
        copyInPadParam.paddingValue = 0;
        const uint64_t srcOffset = outerBase * static_cast<uint64_t>(rowLength) + splitStart;
        DataCopyPad(local, inputGm[srcOffset], copyInParam, copyInPadParam);
        queue.EnQue<T>(local);

        LocalTensor<T> ready = queue.DeQue<T>();
        outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(g), outerLength * static_cast<uint64_t>(splitLen));
        copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
        copyOutParam.blockLen = splitLen * sizeof(T);
        copyOutParam.srcStride = 0;
        copyOutParam.dstStride = 0;
        DataCopyPad(outputGm[outerBase * static_cast<uint64_t>(splitLen)], ready, copyOutParam);
        queue.FreeTensor(ready);
    }
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact<T>::Process()
{
    if (outerTile == 0 || rowLength == 0 || maxSplitSize == 0 || splitNum == 0 || rowPitch == 0) {
        return;
    }

    for (uint64_t i = 0; i < tileNum; ++i) {
        const uint64_t tileIdx = loopOff + i;
        if (mode == MODE_PURE_COPY) {
            ProcessPureTask(tileIdx);
            continue;
        }
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            if (mode == MODE_ROW_CHUNK) {
                ProcessChunkTaskTyped<half>(tileIdx);
            } else {
                const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
                const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
                ProcessFullRowTyped<half>(outerBase, outerReal);
            }
        } else if constexpr (AscendC::Std::is_same<T, uint16_t>::value) {
            if (mode == MODE_ROW_CHUNK) {
                ProcessChunkTaskTyped<int16_t>(tileIdx);
            } else {
                const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
                const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
                ProcessFullRowTyped<int16_t>(outerBase, outerReal);
            }
        } else {
            if (mode == MODE_ROW_CHUNK) {
                ProcessChunkTaskTyped<T>(tileIdx);
            } else {
                const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
                const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
                ProcessFullRowTyped<T>(outerBase, outerReal);
            }
        }
    }
}

#endif // OP_KERNEL_SPLIT_V_UNEVEN_COMPACT_H_
