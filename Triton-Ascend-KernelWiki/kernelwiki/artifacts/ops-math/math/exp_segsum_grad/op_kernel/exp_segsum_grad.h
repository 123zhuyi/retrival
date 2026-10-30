/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file exp_segsum_grad.h
 * \brief
 */

#ifndef EXP_SEGSUM_GRAD_H
#define EXP_SEGSUM_GRAD_H

#include <type_traits>
#include "kernel_operator.h"

namespace ExpSegsumGrad {
using namespace AscendC;
constexpr float ZERO_FLOAT = 0;
constexpr int32_t NO_BUFFER_NUM = 1;
constexpr int32_t BLOCK_LEN = 8;

template <typename T, int32_t MODE>
class ExpSegsumGradND {
public:
    TPipe pipe;

    __aicore__ inline ExpSegsumGradND(){};
    __aicore__ inline void Init(GM_ADDR gradOut, GM_ADDR output, GM_ADDR gradIn,
                                GM_ADDR workspace, ExpSegsumGradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b) {
        return a < b ? a : b;
    };

    __aicore__ inline void ParseTilingData(ExpSegsumGradTilingData *tilingData);
    __aicore__ inline void ClearGM();
    __aicore__ inline void ComputeBase();
    __aicore__ inline void ComputeBatches();
    __aicore__ inline void ComputeMul(int64_t num, LocalTensor<float> currentRow, LocalTensor<float> castRow);
    __aicore__ inline void ComputeSum(
        int64_t colLoopTimes, LocalTensor<float> currentRow, LocalTensor<float> castRow, LocalTensor<T> yTensor);
    __aicore__ inline void ComputeMulBatches(
        int64_t row, int64_t calNumAlign, int64_t rowIdx, LocalTensor<float> currentRow, LocalTensor<float> castRow);
    __aicore__ inline void CopyInBatch(int64_t offset, int64_t batchesEachCopy, uint32_t calSize);
    __aicore__ inline void CopyOutBatch(int64_t offset, int64_t batchesEachCopy, int32_t calNum, uint32_t calSize);
    __aicore__ inline void getLastRow(int64_t offset, uint32_t calSize, LocalTensor<float> lastRow);
    __aicore__ inline void setLastRow(int64_t offset, uint32_t calSize, LocalTensor<float> lastRow);

private:
    TBuf<QuePosition::VECCALC> lastQueue;
    TBuf<QuePosition::VECCALC> currentQueue;
    TBuf<QuePosition::VECCALC> castQueue;
    TQue<QuePosition::VECIN, NO_BUFFER_NUM> gradOutQueue;
    TQue<QuePosition::VECIN, NO_BUFFER_NUM> outputQueue;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> gradInQueue;

    GlobalTensor<T> gradOutTensorsGM;
    GlobalTensor<T> outputTensorsGM;
    GlobalTensor<T> gradInTensorsGM;
    GlobalTensor<float> lastTensorsGM;

    int64_t blockIdx = 0;
    int64_t batchStart = 0;
    int64_t batchEnd = 0;
    int64_t needCoreNum = 0;
    int64_t slideSize = 0;
    
    int64_t tailDimLength = 0;
    int64_t calCount = 0;
    uint64_t blockSize = 8;
};

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::Init(GM_ADDR gradOut, GM_ADDR output, GM_ADDR gradIn,
                                                   GM_ADDR workspace, ExpSegsumGradTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    ParseTilingData(tilingData);
    pipe.InitBuffer(gradOutQueue, NO_BUFFER_NUM, slideSize * sizeof(T));
    pipe.InitBuffer(outputQueue, NO_BUFFER_NUM, slideSize * sizeof(T));
    pipe.InitBuffer(gradInQueue, NO_BUFFER_NUM, slideSize * sizeof(T));
    pipe.InitBuffer(lastQueue, slideSize * sizeof(float));
    pipe.InitBuffer(currentQueue, slideSize * sizeof(float));
    pipe.InitBuffer(castQueue, slideSize * sizeof(float));

    gradOutTensorsGM.SetGlobalBuffer((__gm__ T *)gradOut);
    outputTensorsGM.SetGlobalBuffer((__gm__ T *)output);
    gradInTensorsGM.SetGlobalBuffer((__gm__ T *)gradIn);
    lastTensorsGM.SetGlobalBuffer((__gm__ float *)workspace, tailDimLength * needCoreNum * sizeof(float));
};

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::Process()
{
    if (blockIdx >= needCoreNum) {
        return;
    }
    if (MODE == 0) {
        ClearGM();
        SyncAll();
        ComputeBase();
    } else {
        ComputeBatches();
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::ClearGM()
{
    if (blockIdx >= needCoreNum) {
        return;
    }
    LocalTensor<float> clearRow =lastQueue.Get<float>();
    Duplicate(clearRow, ZERO_FLOAT, slideSize);
    int64_t clearTimes = CeilA2B(tailDimLength, slideSize);
    for (int i = 0; i < clearTimes; i++) {
        int64_t offset = blockIdx * tailDimLength + slideSize * i;
        int64_t length = Min(tailDimLength - i * slideSize, slideSize);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0};
        DataCopyPad(lastTensorsGM[offset], clearRow, copyParams);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::ComputeBase() 
{
    LocalTensor<float> lastRow = lastQueue.Get<float>();
    LocalTensor<float> currentRow = currentQueue.Get<float>();
    LocalTensor<float> castRow = castQueue.Get<float>();
    for (int64_t batchIdx = batchStart; batchIdx < batchEnd; batchIdx++) {
        // 从最后一行开始，逐行向上累加
        for (int64_t rowIdx = tailDimLength - 1; rowIdx >= 0; rowIdx --) {
            int64_t colLoopTimes = CeilA2B(tailDimLength, slideSize);
            LocalTensor<T> yTensor = gradInQueue.AllocTensor<T>();
            if constexpr (std::is_same<T, bfloat16_t>::value){
                Duplicate(yTensor, ToBfloat16(0), BLOCK_LEN * colLoopTimes);
            } else {
                Duplicate(yTensor, static_cast<T>(ZERO_FLOAT), BLOCK_LEN * colLoopTimes);
            }
            PipeBarrier<PIPE_V>();
            // 一次无法处理完整的一行，分多次处理
            for (int64_t colIdx = 0; colIdx < tailDimLength; colIdx += slideSize) {
                // 位于对角线右上方的元素（不含对角线），会进行两次置0操作
                // 不参与最终结果的计算，不处理这些元素
                if (rowIdx < colIdx) {
                    break;
                }

                int64_t num1 = Min(rowIdx - colIdx + 1, slideSize); //第一次masked_fill
                int64_t inOffset = batchIdx * tailDimLength * tailDimLength + rowIdx * tailDimLength + colIdx;
                CopyInBatch(inOffset, 1, num1 * sizeof(T));
                ComputeMul(num1, currentRow, castRow);

                int64_t lastOffset = blockIdx * tailDimLength + colIdx;
                getLastRow(lastOffset, num1 * sizeof(float), lastRow);

                Add(currentRow, currentRow, lastRow, num1);
                PipeBarrier<PIPE_V>();
                // 更新上一行的累加结果
                DataCopy(lastRow, currentRow, slideSize);
                PipeBarrier<PIPE_V>();
                setLastRow(lastOffset, num1 * sizeof(float), lastRow);

                int64_t num2 = Min(rowIdx - colIdx, slideSize); //第二次masked_fill
                if (num2 == 0) {
                    break;
                }
                uint32_t shape[] = { 1, static_cast<uint32_t>(num2)};
                int64_t yOffset= colIdx / slideSize * BLOCK_LEN;
                if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value){
                    ReduceSum<float, Pattern::Reduce::AR, false>(castRow[yOffset], currentRow, shape, true);
                } else if constexpr (std::is_same<T, float>::value) {
                    ReduceSum<float, Pattern::Reduce::AR, false>(yTensor[yOffset], currentRow, shape, true);
                } 
                PipeBarrier<PIPE_V>();
            }
            ComputeSum(colLoopTimes, currentRow, castRow, yTensor);
            gradInQueue.EnQue(yTensor);
            int64_t outOffset = batchIdx * tailDimLength + rowIdx;
            CopyOutBatch(outOffset, 1, 1, sizeof(T));
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::ComputeMul(
    int64_t num, LocalTensor<float> currentRow, LocalTensor<float> castRow)
{
    LocalTensor<T> gradOutTensor = gradOutQueue.DeQue<T>();
    LocalTensor<T> outputTensor = outputQueue.DeQue<T>();
    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value){
        Cast(currentRow, gradOutTensor, RoundMode::CAST_NONE, num);
        Cast(castRow, outputTensor, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        Mul(currentRow, currentRow, castRow, num);
    } else if constexpr (std::is_same<T, float>::value) {
        Mul(currentRow, gradOutTensor, outputTensor, num);
    }
    PipeBarrier<PIPE_V>();
    gradOutQueue.FreeTensor(gradOutTensor);
    outputQueue.FreeTensor(outputTensor);
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::ComputeSum(
    int64_t colLoopTimes, LocalTensor<float> currentRow, LocalTensor<float> castRow, LocalTensor<T> yTensor)
{
    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value){
        DataCopy(currentRow, castRow, BLOCK_LEN * colLoopTimes);
        PipeBarrier<PIPE_V>();
        uint32_t shape[] = { 1, static_cast<uint32_t>(BLOCK_LEN * colLoopTimes)};
        ReduceSum<float, Pattern::Reduce::AR, false>(castRow, currentRow, shape, true);
        PipeBarrier<PIPE_V>();
        Cast(yTensor, castRow, RoundMode::CAST_ROUND, 1);
    } else if constexpr (std::is_same<T, float>::value) {
        DataCopy(currentRow, yTensor, BLOCK_LEN * colLoopTimes);
        PipeBarrier<PIPE_V>();
        uint32_t shape[] = { 1, static_cast<uint32_t>(BLOCK_LEN * colLoopTimes)};
        ReduceSum<float, Pattern::Reduce::AR, false>(yTensor, currentRow, shape, true);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::getLastRow(
    int64_t offset, uint32_t calSize, LocalTensor<float> lastRow)
{
    event_t eventId1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId1);
    WaitFlag<HardEvent::V_MTE2>(eventId1);
    DataCopyExtParams copyParams{1, calSize, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(lastRow, lastTensorsGM[offset], copyParams, padParams);
    event_t eventId2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId2);
    WaitFlag<HardEvent::MTE2_V>(eventId2);
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::setLastRow(
    int64_t offset, uint32_t calSize, LocalTensor<float> lastRow)
{
    event_t eventId1 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId1);
    WaitFlag<HardEvent::V_MTE3>(eventId1);
    DataCopyExtParams copyParams{1, calSize, 0, 0, 0};
    DataCopyPad(lastTensorsGM[offset], lastRow, copyParams);
    event_t eventId2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId2);
    WaitFlag<HardEvent::MTE3_V>(eventId2);
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::ComputeMulBatches(
    int64_t row, int64_t calNumAlign, int64_t rowIdx, LocalTensor<float> currentRow, LocalTensor<float> castRow)
{
    LocalTensor<T> gradOutTensor = gradOutQueue.DeQue<T>();
    LocalTensor<T> outputTensor = outputQueue.DeQue<T>();
    for (int64_t i = 0; i < row; i++) {
        int64_t offset = i * calNumAlign;
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value){
            Cast(currentRow[offset], gradOutTensor[offset], RoundMode::CAST_NONE, rowIdx + 2 + i - row);
            Cast(castRow[offset], outputTensor[offset], RoundMode::CAST_NONE, rowIdx + 2 + i - row);
            PipeBarrier<PIPE_V>();
            Mul(currentRow[offset], currentRow[offset], castRow[offset], rowIdx + 2 + i - row);
        } else if constexpr (std::is_same<T, float>::value) {
            Mul(currentRow[offset], gradOutTensor[offset], outputTensor[offset], rowIdx + 2 + i - row);
        }
    }
    PipeBarrier<PIPE_V>();
    gradOutQueue.FreeTensor(gradOutTensor);
    outputQueue.FreeTensor(outputTensor);
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::ComputeBatches()
{
    LocalTensor<float> lastRow = lastQueue.Get<float>();
    LocalTensor<float> currentRow = currentQueue.Get<float>();
    LocalTensor<float> castRow = castQueue.Get<float>();
    uint32_t calSize = calCount * sizeof(T);
    int64_t calNumAlign = CeilA2B(calCount, blockSize) * blockSize;
    int64_t rowNum = slideSize / calNumAlign;
    for (int64_t batchIdx = batchStart; batchIdx < batchEnd; batchIdx++) { 
        Duplicate(lastRow, ZERO_FLOAT, slideSize);
        for (int64_t rowIdx = tailDimLength - 1; rowIdx >= 0; rowIdx -= rowNum) {
            int64_t row = Min(rowIdx + 1, rowNum);
            int64_t inOffset = batchIdx * tailDimLength * tailDimLength + (rowIdx - row + 1) * tailDimLength;
            CopyInBatch(inOffset, row, tailDimLength * sizeof(T));

            Duplicate(currentRow, ZERO_FLOAT, slideSize);
            PipeBarrier<PIPE_V>();
            ComputeMulBatches(row, calNumAlign, rowIdx, currentRow, castRow);
            for (int64_t i = 0; i < row; i++) {
                int64_t offset = (row - i - 1) * calNumAlign;
                Add(currentRow[offset], currentRow[offset], lastRow, calNumAlign);
                PipeBarrier<PIPE_V>();
                DataCopy(lastRow, currentRow[offset], calNumAlign);
                PipeBarrier<PIPE_V>();
            }
            Duplicate(castRow, ZERO_FLOAT, slideSize);
            PipeBarrier<PIPE_V>();
            
            for (int64_t i = 0; i < row; i++) {
                Adds(castRow[i * calNumAlign], currentRow[i * calNumAlign], ZERO_FLOAT, rowIdx + 1 + i - row);
            }
            PipeBarrier<PIPE_V>();

            LocalTensor<T> yTensor = gradInQueue.AllocTensor<T>();
            uint32_t shape[] = { static_cast<uint32_t>(row), static_cast<uint32_t>(calNumAlign)};
            if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value){
                ReduceSum<float, Pattern::Reduce::AR, false>(currentRow, castRow, shape, true);
                PipeBarrier<PIPE_V>();
                Cast(yTensor, currentRow, RoundMode::CAST_RINT, row);
            } else if constexpr (std::is_same<T, float>::value) {
                ReduceSum<float, Pattern::Reduce::AR, false>(yTensor, castRow, shape, true);
            }
            int64_t outOffset = batchIdx * tailDimLength + rowIdx - row + 1;

            gradInQueue.EnQue(yTensor);
            CopyOutBatch(outOffset, 1, row, row * sizeof(T));
        }
    }
}
template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::CopyInBatch(int64_t offset, int64_t batchesEachCopy, uint32_t calSize)
{
    LocalTensor<T> gradOutTensor = gradOutQueue.AllocTensor<T>();
    LocalTensor<T> outputTensor = outputQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(batchesEachCopy), calSize, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(gradOutTensor, gradOutTensorsGM[offset], copyParams, padParams);
    DataCopyPad(outputTensor, outputTensorsGM[offset], copyParams, padParams);
    gradOutQueue.EnQue(gradOutTensor);
    outputQueue.EnQue(outputTensor);
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::CopyOutBatch(
    int64_t offset, int64_t batchesEachCopy, int32_t calNum, uint32_t calSize)
{
    LocalTensor<T> yTensor = gradInQueue.DeQue<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(batchesEachCopy), calSize, 0, 0, 0};
    DataCopyPad(gradInTensorsGM[offset], yTensor, copyParams);
    gradInQueue.FreeTensor(yTensor);
}

template <typename T, int32_t MODE>
__aicore__ inline void ExpSegsumGradND<T, MODE>::ParseTilingData(ExpSegsumGradTilingData *tilingData)
{
    slideSize = tilingData->slideSize;
    tailDimLength = tilingData->tailDimLength;
    calCount = tailDimLength;
    needCoreNum = tilingData->needCoreNum;
    batchStart = tilingData->batchStart[blockIdx];
    batchEnd = tilingData->batchEnd[blockIdx];

    blockSize = 32 / sizeof(T);
}
}  // namespace ExpSegsumGrad
#endif