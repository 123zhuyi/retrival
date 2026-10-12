/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please
 * refer to the License for details. You may not use this file except in compliance with the License. THIS SOFTWARE IS
 * PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 * repository for the full text of the License.
 */

/*!
 * \file split_v_same_len.h
 * \brief Same-length split for innerLength == 1.
 *
 * The previous version copied a large [outerTile, rowLength] block into UB, but
 * wrote each [splitSize] row to GM one by one. For small splitSize this creates
 * too many tiny MTE3 transactions.
 *
 * This version processes one output group at a time:
 *   GM [outerTile, rowLength] --2D strided DataCopyPad--> UB [outerTile, splitSize]
 *   UB [outerTile, splitSize] --2D DataCopyPad----------> GM output
 *
 * For b16 types, splitTileLength == 16 enables a TransDataTo5HD path that maps
 * to the TBE scatter_vnchwconv_b16 primitive:
 *   [16, rowLength] -> [rowLength, 16] -> [16, splitSize]
 * and then writes each output by one 2D DataCopyPad.
 */

#ifndef OP_KERNEL_SPLIT_V_SAME_LEN_H_
#define OP_KERNEL_SPLIT_V_SAME_LEN_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVSameLen {
public:
    __aicore__ inline SplitVSameLen() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataSameLen* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t VNCHW_SIDE = 16;

    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queue;
    TBuf<TPosition::VECCALC> calcBuf;

    uint64_t outerLength;
    uint32_t rowLength;
    uint32_t splitSize;
    uint32_t tailSplitSize;
    uint32_t maxSplitSize;
    uint32_t splitNum;

    uint32_t outerTile;
    uint32_t outerTail;
    uint32_t outerTileNum;

    uint64_t tileNum;
    uint64_t loopOff;
    uint32_t alignedNum;
    uint32_t splitAligned;
    uint32_t rowAligned;
    bool useVnchw;
    bool useFullRowDma;

    DataCopyExtParams copyInParam{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    DataCopyExtParams copyOutParam{0, 0, 0, 0, 0};

    __aicore__ inline void ProcessDma2d();
    __aicore__ inline void ProcessFullRowDma();
    __aicore__ inline void ProcessVnchwB16();
    __aicore__ inline void WaitMte2ToV();
    __aicore__ inline void WaitVToMte3();
    __aicore__ inline void WaitMte3ToV();
    __aicore__ inline void WaitMte2ToMte3();
    __aicore__ inline void WaitMte3ToMte2();
    __aicore__ inline uint32_t GetSplitSize(uint32_t splitIdx) const;
};

template <typename T>
__aicore__ inline void SplitVSameLen<T>::Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataSameLen* tiling,
                                              TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= GetBlockNum()) {
        return;
    }

    outList.Init((__gm__ void*)output);

    outerLength = tiling->outerLength;
    rowLength = static_cast<uint32_t>(tiling->midLength);
    splitSize = tiling->splitSize;
    tailSplitSize = tiling->tailSplitSize == 0 ? splitSize : tiling->tailSplitSize;
    maxSplitSize = tailSplitSize > splitSize ? tailSplitSize : splitSize;
    splitNum = tiling->splitNum;
    outerTile = tiling->outerTile;
    outerTail = tiling->outerTail;
    outerTileNum = tiling->outerTileNum;

    if (blockIdx < tiling->formerNum) {
        tileNum = tiling->formerOuterTileNum;
        loopOff = blockIdx * tiling->formerOuterTileNum;
    } else {
        tileNum = tiling->tailOuterTileNum;
        loopOff = tiling->formerNum * tiling->formerOuterTileNum +
                  (blockIdx - tiling->formerNum) * tiling->tailOuterTileNum;
    }

    inputGm.SetGlobalBuffer((__gm__ T*)input, outerLength * rowLength);

    alignedNum = BLOCK_SIZE / sizeof(T);
    if (alignedNum == 0) {
        alignedNum = 1;
    }
    splitAligned = ((maxSplitSize + alignedNum - 1) / alignedNum) * alignedNum;
    rowAligned = ((rowLength + alignedNum - 1) / alignedNum) * alignedNum;
    useVnchw = (tiling->splitTileLength == VNCHW_SIDE) && (sizeof(T) == sizeof(uint16_t));
    useFullRowDma = tiling->splitTileLength == splitVSameLenFullRowDma;

    if (useFullRowDma) {
        pipe->InitBuffer(calcBuf, outerTile * rowAligned * sizeof(T));
    } else if (useVnchw) {
        const uint32_t inputElems = VNCHW_SIDE * rowAligned;
        const uint32_t transElems = rowAligned * VNCHW_SIDE;
        const uint32_t outputElems = VNCHW_SIDE * splitAligned;
        pipe->InitBuffer(calcBuf, (inputElems + transElems + outputElems) * sizeof(T));
    } else {
        // DataCopyPad stores each unaligned row in UB with 32B alignment.
        pipe->InitBuffer(queue, BUFFER_NUM, outerTile * splitAligned * sizeof(T));
    }
}

template <typename T>
__aicore__ inline uint32_t SplitVSameLen<T>::GetSplitSize(uint32_t splitIdx) const
{
    return (splitIdx + 1 == splitNum) ? tailSplitSize : splitSize;
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::Process()
{
    if (useFullRowDma) {
        ProcessFullRowDma();
    } else if (useVnchw) {
        ProcessVnchwB16();
    } else {
        ProcessDma2d();
    }
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::ProcessDma2d()
{
    for (uint64_t i = 0; i < tileNum; ++i) {
        const uint64_t tileIdx = loopOff + i;
        const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
        uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

        for (uint32_t g = 0; g < splitNum; ++g) {
            const uint32_t curSplitSize = GetSplitSize(g);
            // Compact this split group from strided input rows into UB.
            const uint64_t srcOff = outerBase * rowLength + static_cast<uint64_t>(g) * splitSize;
            LocalTensor<T> inLocal = queue.AllocTensor<T>();

            copyInParam.blockCount = static_cast<uint16_t>(outerReal);
            copyInParam.blockLen = static_cast<uint32_t>(curSplitSize * sizeof(T));
            copyInParam.srcStride = static_cast<uint32_t>((rowLength - curSplitSize) * sizeof(T));
            copyInParam.dstStride = 0;
            DataCopyPad(inLocal, inputGm[srcOff], copyInParam, padParam);
            queue.EnQue<T>(inLocal);

            // Write compact [outerReal, splitSize] to the selected output in one
            // 2D MTE3 transaction. Source rows in UB are 32B-aligned by CopyIn.
            LocalTensor<T> data = queue.DeQue<T>();
            outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(g), outerLength * curSplitSize);
            const uint64_t dstBase = outerBase * curSplitSize;

            copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
            copyOutParam.blockLen = static_cast<uint32_t>(curSplitSize * sizeof(T));
            copyOutParam.srcStride = 0;
            copyOutParam.dstStride = 0;
            DataCopyPad(outputGm[dstBase], data, copyOutParam);

            queue.FreeTensor(data);
        }
    }
}

#endif // OP_KERNEL_SPLIT_V_SAME_LEN_H_

template <typename T>
__aicore__ inline void SplitVSameLen<T>::WaitMte2ToV()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::WaitVToMte3()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::WaitMte3ToV()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::WaitMte2ToMte3()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::WaitMte3ToMte2()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::ProcessFullRowDma()
{
    LocalTensor<T> inputLocal = calcBuf.Get<T>();
    GlobalTensor<T> outputGm0;
    GlobalTensor<T> outputGm1;
    GlobalTensor<T> outputGm2;
    GlobalTensor<T> outputGm3;
    const bool splitNum4 = splitNum == 4;
    if (splitNum4) {
        outputGm0.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(0), outerLength * splitSize);
        outputGm1.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(1), outerLength * splitSize);
        outputGm2.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(2), outerLength * splitSize);
        outputGm3.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(3), outerLength * splitSize);
    }

    copyInParam.blockLen = static_cast<uint32_t>(rowLength * sizeof(T));
    copyInParam.srcStride = 0;
    copyInParam.dstStride = static_cast<uint32_t>((rowAligned - rowLength) * sizeof(T));
    DataCopyPadExtParams<T> fullRowPadParam{true, 0, static_cast<uint8_t>(rowAligned - rowLength), 0};

    copyOutParam.blockLen = static_cast<uint32_t>(splitSize * sizeof(T));
    copyOutParam.srcStride = static_cast<uint32_t>((rowAligned - splitSize) * sizeof(T));
    copyOutParam.dstStride = 0;

    DataCopyParams copyOutAlignedParam{0, 0, 0, 0};
    const bool useAlignedOutputCopy = rowAligned == rowLength && (splitSize * sizeof(T)) % BLOCK_SIZE == 0 &&
                                      ((rowLength - splitSize) * sizeof(T)) % BLOCK_SIZE == 0;
    if (useAlignedOutputCopy) {
        copyOutAlignedParam.blockLen = static_cast<uint16_t>(splitSize * sizeof(T));
        copyOutAlignedParam.srcStride = static_cast<uint16_t>(((rowLength - splitSize) * sizeof(T)) / BLOCK_SIZE);
        copyOutAlignedParam.dstStride = 0;
    }

    for (uint64_t i = 0; i < tileNum; ++i) {
        const uint64_t tileIdx = loopOff + i;
        const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
        const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

        copyInParam.blockCount = static_cast<uint16_t>(outerReal);
        const uint64_t srcBase = outerBase * static_cast<uint64_t>(rowLength);
        if (rowAligned == rowLength) {
            const uint32_t copyElems = static_cast<uint32_t>(static_cast<uint64_t>(outerReal) * rowLength);
            DataCopy(inputLocal, inputGm[srcBase], copyElems);
        } else {
            DataCopyPad(inputLocal, inputGm[srcBase], copyInParam, fullRowPadParam);
        }
        WaitMte2ToMte3();

        copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
        copyOutAlignedParam.blockCount = static_cast<uint16_t>(outerReal);
        const uint64_t dstBase = outerBase * static_cast<uint64_t>(splitSize);
        if (splitNum4) {
            if (useAlignedOutputCopy) {
                DataCopyPad(outputGm0[dstBase], inputLocal, copyOutAlignedParam);
                DataCopyPad(outputGm1[dstBase], inputLocal[splitSize], copyOutAlignedParam);
                DataCopyPad(outputGm2[dstBase], inputLocal[splitSize * 2], copyOutAlignedParam);
                DataCopyPad(outputGm3[dstBase], inputLocal[splitSize * 3], copyOutAlignedParam);
            } else {
                DataCopyPad(outputGm0[dstBase], inputLocal, copyOutParam);
                DataCopyPad(outputGm1[dstBase], inputLocal[splitSize], copyOutParam);
                DataCopyPad(outputGm2[dstBase], inputLocal[splitSize * 2], copyOutParam);
                DataCopyPad(outputGm3[dstBase], inputLocal[splitSize * 3], copyOutParam);
            }
        } else {
            for (uint32_t g = 0; g < splitNum; ++g) {
                outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(g), outerLength * splitSize);
                const uint32_t localOffset = static_cast<uint32_t>(static_cast<uint64_t>(g) * splitSize);
                if (useAlignedOutputCopy) {
                    DataCopyPad(outputGm[dstBase], inputLocal[localOffset], copyOutAlignedParam);
                } else {
                    DataCopyPad(outputGm[dstBase], inputLocal[localOffset], copyOutParam);
                }
            }
        }
        if (i + 1 < tileNum) {
            WaitMte3ToMte2();
        }
    }
}

template <typename T>
__aicore__ inline void SplitVSameLen<T>::ProcessVnchwB16()
{
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        uint64_t ubOffset = 0;
        LocalTensor<T> inputLocalT = calcBuf.GetWithOffset<T>(VNCHW_SIDE * rowAligned, ubOffset);
        ubOffset += static_cast<uint64_t>(VNCHW_SIDE) * rowAligned * sizeof(T);
        LocalTensor<T> transLocalT = calcBuf.GetWithOffset<T>(rowAligned * VNCHW_SIDE, ubOffset);
        ubOffset += static_cast<uint64_t>(rowAligned) * VNCHW_SIDE * sizeof(T);
        LocalTensor<T> outLocalT = calcBuf.GetWithOffset<T>(VNCHW_SIDE * splitAligned, ubOffset);

        LocalTensor<uint16_t> inputLocal = inputLocalT.template ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> transLocal = transLocalT.template ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> outLocal = outLocalT.template ReinterpretCast<uint16_t>();

        uint64_t firstSrcList[VNCHW_SIDE];
        uint64_t firstDstList[VNCHW_SIDE];
        uint64_t secondSrcList[VNCHW_SIDE];
        uint64_t secondDstList[VNCHW_SIDE];

        TransDataTo5HDParams firstParams;
        firstParams.dstHighHalf = false;
        firstParams.srcHighHalf = false;
        firstParams.repeatTimes = static_cast<uint8_t>(rowAligned / VNCHW_SIDE);
        firstParams.srcRepStride = 1;
        firstParams.dstRepStride = VNCHW_SIDE;
        if (firstParams.repeatTimes == 1) {
            firstParams.srcRepStride = 0;
            firstParams.dstRepStride = 0;
        }

        TransDataTo5HDParams secondParams;
        secondParams.dstHighHalf = false;
        secondParams.srcHighHalf = false;
        secondParams.repeatTimes = 1;
        secondParams.srcRepStride = 0;
        secondParams.dstRepStride = 0;

        copyInParam.blockLen = static_cast<uint32_t>(rowLength * sizeof(T));
        copyInParam.srcStride = 0;
        copyInParam.dstStride = 0;

        copyOutParam.blockLen = static_cast<uint32_t>(splitSize * sizeof(T));
        copyOutParam.srcStride = 0;
        copyOutParam.dstStride = 0;

        for (uint64_t i = 0; i < tileNum; ++i) {
            const uint64_t tileIdx = loopOff + i;
            const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
            const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

            copyInParam.blockCount = static_cast<uint16_t>(outerReal);
            const uint64_t srcOff = outerBase * rowLength;
            DataCopyPad(inputLocalT, inputGm[srcOff], copyInParam, padParam);
            WaitMte2ToV();

            // First scatter_vnchwconv_b16 equivalent:
            // source rows -> 16-row VNCHW blocks, grouped by 16 columns.
            for (uint32_t r = 0; r < VNCHW_SIDE; ++r) {
                firstSrcList[r] = reinterpret_cast<uint64_t>(inputLocal[r * rowAligned].GetPhyAddr());
                firstDstList[r] = reinterpret_cast<uint64_t>(transLocal[r * VNCHW_SIDE].GetPhyAddr());
            }
            TransDataTo5HD<uint16_t>(firstDstList, firstSrcList, firstParams);
            PipeBarrier<PIPE_V>();

            for (uint32_t g = 0; g < splitNum; ++g) {
                for (uint32_t splitBase = 0; splitBase < splitSize; splitBase += VNCHW_SIDE) {
                    const uint32_t realCols = (splitBase + VNCHW_SIDE <= splitSize) ? VNCHW_SIDE :
                                                                                      (splitSize - splitBase);

                    // Second scatter_vnchwconv_b16 equivalent:
                    // selected split columns -> row-major [16, splitSize] UB staging.
                    for (uint32_t c = 0; c < VNCHW_SIDE; ++c) {
                        if (c < realCols) {
                            const uint32_t col = static_cast<uint32_t>(g) * splitSize + splitBase + c;
                            const uint32_t transOff = (col / VNCHW_SIDE) * VNCHW_SIDE * VNCHW_SIDE +
                                                      (col % VNCHW_SIDE) * VNCHW_SIDE;
                            secondSrcList[c] = reinterpret_cast<uint64_t>(transLocal[transOff].GetPhyAddr());
                        } else {
                            // Padding columns are not copied to GM. Point them to a valid 32B block.
                            secondSrcList[c] = reinterpret_cast<uint64_t>(transLocal[0].GetPhyAddr());
                        }
                        secondDstList[c] = reinterpret_cast<uint64_t>(
                            outLocal[c * splitAligned + splitBase].GetPhyAddr());
                    }
                    TransDataTo5HD<uint16_t>(secondDstList, secondSrcList, secondParams);
                }
                WaitVToMte3();

                outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(g), outerLength * splitSize);
                const uint64_t dstBase = outerBase * splitSize;
                copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
                DataCopyPad(outputGm[dstBase], outLocalT, copyOutParam);
                WaitMte3ToV();
            }
        }
    }
}
