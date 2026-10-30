/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file split.h
 * \brief
 */
#ifndef __SPLIT_H__
#define __SPLIT_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_tiling_data.h"
#include "split_tiling_key.h"
#include "split_utils.h"

namespace NsSplit {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t INDICES_LIMIT = 10;
constexpr uint32_t DIM_LIMIT = 8;

template <typename T>
class Split {
public:
    __aicore__ inline Split(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, SplitTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyWorkIn(int32_t progress);
    __aicore__ inline void CopyOutIsEven(int32_t progress);
    __aicore__ inline void CopyOutNotEven(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe* pipe;
    TQueBind<TPosition::VECOUT, TPosition::GM, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> WorkBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> workGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t blockSize;

    uint32_t globalBufferIndex;

    uint32_t indices_or_sections[INDICES_LIMIT];
    uint32_t splitLen[INDICES_LIMIT + 1];
    uint32_t shape[DIM_LIMIT];
    uint32_t srcdim;
    uint32_t totalNums;
    uint32_t unit;
    uint32_t indices_len;
    int64_t axis;
    bool isEven;
    GM_ADDR ybase;
};

template <typename T>
__aicore__ inline void Split<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, SplitTilingData* tilingData, TPipe* pipe_)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    pipe = pipe_;
    uint32_t coreIdx = GetBlockIdx();
    this->globalBufferIndex = tilingData->bigCoreDataNum * coreIdx;
    this->tileDataNum = tilingData->tileDataNum;
    this->blockSize = tilingData->blockSize;
    this->srcdim = tilingData->srcdim;
    this->totalNums = tilingData->totalNums;
    this->unit = tilingData->unit;
    this->indices_len = tilingData->indices_len;
    this->ybase = y; // 标记tensorlist起始地址
    this->axis = tilingData->axis;
    this->isEven = tilingData->isEven;
    for (int i = 0; i < INDICES_LIMIT; ++i) {
        this->indices_or_sections[i] = tilingData->indices_or_sections[i];
        this->splitLen[i] = tilingData->splitLen[i];
    }
    this->splitLen[INDICES_LIMIT] = tilingData->splitLen[INDICES_LIMIT];
    for (int i = 0; i < DIM_LIMIT; ++i) {
        this->shape[i] = tilingData->shape[i];
    }
    if (coreIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum; // （单位：元素数）
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (coreIdx - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ T*)x);
    workGm.SetGlobalBuffer((__gm__ T*)workspace);
    // 分配本地缓冲区（双缓冲）
    pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe->InitBuffer(WorkBuf, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void Split<T>::CopyWorkIn(int32_t progress)
{
    LocalTensor<T> yLocal = outQueue.AllocTensor<T>();

    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = processDataNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(yLocal, workGm[progress * tileDataNum], copyParams, {false, 0, 0, 0});

    outQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Split<T>::CopyOutIsEven(int32_t progress)
{
    LocalTensor<T> yLocal = outQueue.DeQue<T>();
    uint32_t tileStart = globalBufferIndex + static_cast<uint32_t>(progress) * tileDataNum;
    uint32_t remaining = processDataNum;

    uint32_t outIdx = 0;
    uint32_t acc = 0;
    uint32_t perLen = splitLen[0];
    uint32_t outCount = indices_or_sections[0];

    while (outIdx < outCount && tileStart >= acc + perLen) {
        acc += perLen;
        ++outIdx;
    }

    uint32_t srcOff = 0;
    uint32_t dstOff = (tileStart >= acc) ? (tileStart - acc) : 0;

    while (remaining > 0 && outIdx < outCount) {
        uint32_t space = (perLen > dstOff) ? (perLen - dstOff) : 0;
        uint32_t toCopy = (remaining < space) ? remaining : space;

        yGm.SetGlobalBuffer(GetTensorAddr<T>(outIdx, ybase), perLen);
        CopyOutRange(yLocal, srcOff, yGm, dstOff, toCopy, blockSize);

        remaining -= toCopy;
        srcOff += toCopy;
        ++outIdx;
        dstOff = 0;
    }
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Split<T>::CopyOutNotEven(int32_t progress)
{
    LocalTensor<T> yLocal = outQueue.DeQue<T>();
    uint32_t tileStart = globalBufferIndex + static_cast<uint32_t>(progress) * tileDataNum;
    uint32_t remaining = processDataNum;

    uint32_t outIdx = 0;
    uint32_t acc = 0;
    while (outIdx <= indices_len) {
        uint32_t len = splitLen[outIdx];
        if (tileStart < acc + len)
            break;
        acc += len;
        ++outIdx;
    }

    uint32_t srcOff = 0;
    uint32_t dstOff = (tileStart >= acc) ? (tileStart - acc) : 0;

    while (remaining > 0 && outIdx <= indices_len) {
        uint32_t len = splitLen[outIdx];
        if (len == 0) {
            ++outIdx;
            continue;
        }
        uint32_t space = (len > dstOff) ? (len - dstOff) : 0;
        uint32_t toCopy = (remaining < space) ? remaining : space;

        yGm.SetGlobalBuffer(GetTensorAddr<T>(outIdx, ybase), len);
        CopyOutRange(yLocal, srcOff, yGm, dstOff, toCopy, blockSize);

        remaining -= toCopy;
        srcOff += toCopy;
        ++outIdx;
        dstOff = 0;
    }
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Split<T>::Compute(int32_t progress)
{
    LocalTensor<T> workLocal = WorkBuf.Get<T>();
    uint32_t linearBase = globalBufferIndex + progress * tileDataNum;
    for (uint32_t i = 0; i < processDataNum; ++i) {
        uint32_t globalIdx = linearBase + i;
        if (globalIdx >= totalNums) {
            break; // 防越界
        }
        uint32_t relIdx = globalIdx;
        uint32_t InOffset = 0;

        uint32_t sliceIdx = 0;
        uint32_t localIndex = 0;
        if (isEven) {
            uint32_t section = indices_or_sections[0];
            uint32_t sliceLen = (shape[axis] / section) * unit;
            sliceIdx = relIdx / sliceLen;
            localIndex = relIdx % sliceLen;
            InOffset = CalIndexEven(shape, srcdim, axis, section, localIndex, sliceIdx, unit);
        } else {
            uint32_t acc = 0;
            bool found = false;
            for (uint32_t s = 0; s <= indices_len; ++s) {
                uint32_t len = splitLen[s];
                if (len == 0) {
                    continue; // 跳过长度为0的slice
                }
                if (relIdx < acc + len) {
                    sliceIdx = s;
                    localIndex = relIdx - acc;
                    found = true;
                    break;
                }
                acc += len;
            }
            if (!found) {
                sliceIdx = indices_len; // 最后一个 slice
                localIndex = 0;
            }
            InOffset =
                CalIndexByIndices(shape, srcdim, axis, indices_or_sections, indices_len, localIndex, sliceIdx, unit);
        }
        T origalValue = xGm.GetValue(InOffset);
        workLocal.SetValue(i, origalValue);
    }

    int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventIDSToMTE3);

    AscendC::DataCopy(workGm[linearBase], workLocal, processDataNum);
}

template <typename T>
__aicore__ inline void Split<T>::Process()
{
    uint32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        Compute(i);
    }
    this->processDataNum = this->tailDataNum;
    Compute(loopCount - 1);

    SyncAll();

    this->processDataNum = this->tileDataNum;
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        CopyWorkIn(i);
        if (this->isEven) {
            CopyOutIsEven(i);
        } else {
            CopyOutNotEven(i);
        }
    }
    this->processDataNum = this->tailDataNum;
    CopyWorkIn(loopCount - 1);
    if (this->isEven) {
        CopyOutIsEven(loopCount - 1);
    } else {
        CopyOutNotEven(loopCount - 1);
    }
}
} // namespace NsSplit
#endif // SPLIT_H