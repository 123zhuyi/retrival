/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
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
 * \file sort_v2.h
 * \brief
 */
#ifndef SORT_H
#define SORT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sort_v2_tiling_data.h"
#include "sort_v2_tiling_key.h"
#include <cfloat>

namespace NsSortV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ELEMENT_16 = 16U;
constexpr uint32_t ELEMENT_32 = 32U;
constexpr uint32_t BUFFER_SIZE = 9U;

template <typename T>
class SortV2 {
public:
    __aicore__ inline SortV2(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR dstIndex, const SortV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t tileIndex);
    __aicore__ inline void CopyOut(int32_t tileIndex);
    __aicore__ inline void Compute(int32_t tileIndex);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> dstIndexQ;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> calcQ;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> tmpQ;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> concatQ;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<uint32_t> indexGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<uint32_t> dstIndexGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t processDataNum = 0;
    uint32_t axis = 0;
    bool descending = false;
    uint32_t dimH = 1;
    uint32_t dimW = 1;
    uint32_t sliceLen = 0;
    uint32_t realSortLen = 0;
    uint32_t concatRepeat = 0;
    uint32_t sortRepeat = 0;
    uint32_t extractRepeat = 0;
    uint32_t inBufferSize = 0;
    uint32_t outBufferSize = 0;
    uint32_t calcBufferSize = 0;
    uint32_t tmpBufferSize = 0;
    uint32_t concatTmpBytes = 0;
    uint32_t align8 = 0;
    uint32_t padLen = 0;
    uint32_t dupCount = 0;
};

template <typename T>
__aicore__ inline void SortV2<T>::Init(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR dstIndex, const SortV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    const uint32_t coreIdx = AscendC::GetBlockIdx();
    const uint32_t validCoreNum = tilingData->startBlockIdx;
    if (coreIdx >= validCoreNum) {
        coreDataNum = 0;
        return;
    }
    const uint32_t smallCoreNum = tilingData->smallCoreNum;
    const uint32_t bigCoreNum = tilingData->bigCoreNum;
    const uint32_t bigCoreDataNum = tilingData->bigCoreDataNum;
    const uint32_t smallCoreDataNum = tilingData->smallCoreDataNum;

    axis = tilingData->axis;
    descending = tilingData->descending;
    dimH = tilingData->dimH;
    dimW = tilingData->dimW;
    sliceLen = tilingData->sliceLen;
    realSortLen = tilingData->realSortLen;
    align8 = tilingData->align8;
    dupCount = tilingData->dupCount;
    padLen = tilingData->padLen;

    // 计算当前核心的起始地址偏移
    uint32_t bigBefore = coreIdx < bigCoreNum ? coreIdx : bigCoreNum;
    uint32_t smallBefore = coreIdx > bigCoreNum ? (coreIdx - bigCoreNum) : 0;
    uint32_t startSlice = bigBefore * bigCoreDataNum + smallBefore * smallCoreDataNum;
    uint32_t coreDataStart = (axis == 0) ? startSlice // 列排序时列号就是元素偏移
                                           :
                                           startSlice * dimW; // 行排序时整行连续
    // 计算当前核心的数据量
    if (coreIdx < bigCoreNum) {
        coreDataNum = bigCoreDataNum;
    } else {
        coreDataNum = smallCoreDataNum;
    }
    if (coreDataNum == 0) {
        return;
    }

    // 计算临时缓存
    concatRepeat = realSortLen / ELEMENT_16;
    sortRepeat = realSortLen / ELEMENT_32;
    extractRepeat = realSortLen / ELEMENT_32;
    uint32_t maxTypeSize = sizeof(T) > sizeof(uint32_t) ? sizeof(T) : sizeof(uint32_t);
    inBufferSize = realSortLen * maxTypeSize;
    outBufferSize = realSortLen * sizeof(T);
    calcBufferSize = realSortLen * BUFFER_SIZE;
    tmpBufferSize = realSortLen * BUFFER_SIZE;
    // 用作搬运地址
    this->tileDataNum = (axis == 0) ? 1 : dimW;
    const uint32_t totalElem = dimH * dimW;
    const uint32_t remainElem = totalElem > coreDataStart ? totalElem - coreDataStart : 0;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + coreDataStart, remainElem);
    indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(index) + coreDataStart, remainElem);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + coreDataStart, remainElem);
    dstIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(dstIndex) + coreDataStart, remainElem);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, inBufferSize);
    pipe.InitBuffer(outQueueY, 1, outBufferSize);
    pipe.InitBuffer(dstIndexQ, 1, realSortLen * sizeof(uint32_t));
    pipe.InitBuffer(calcQ, 1, calcBufferSize * sizeof(T));
    pipe.InitBuffer(tmpQ, 1, tmpBufferSize * sizeof(T));
    pipe.InitBuffer(concatQ, 1, tmpBufferSize * sizeof(T));
}

template <typename T>
__aicore__ inline void SortV2<T>::CopyIn(int32_t tileIndex)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<uint32_t> indexLocal = inQueueX.AllocTensor<uint32_t>();
    if (axis == 0) {
        T padVal = descending ? static_cast<T>(-FLT_MAX) : static_cast<T>(FLT_MAX);
        AscendC::Duplicate(xLocal, padVal, realSortLen);
        uint32_t idxPadVal = descending ? static_cast<uint32_t>(-FLT_MAX) : static_cast<uint32_t>(FLT_MAX);
        AscendC::Duplicate(indexLocal, idxPadVal, realSortLen);
        AscendC::DataCopyExtParams copyParams1{
            static_cast<uint16_t>(dimH), static_cast<uint32_t>(sizeof(T)),
            static_cast<uint32_t>((dimW - 1) * sizeof(T)), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams1{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, xGm[tileIndex], copyParams1, padParams1);
        AscendC::DataCopyExtParams copyParams2{
            static_cast<uint16_t>(dimH), static_cast<uint32_t>(sizeof(uint32_t)),
            static_cast<uint32_t>((dimW - 1) * sizeof(uint32_t)), 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> padParams2{false, 0, 0, 0};
        AscendC::DataCopyPad(indexLocal, indexGm[tileIndex], copyParams2, padParams2);
    } else {
        AscendC::DataCopyExtParams copyParams1{
            static_cast<uint16_t>(1), static_cast<uint32_t>(this->dimW * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams1{
            true, 0, static_cast<uint8_t>(padLen), descending ? static_cast<T>(-FLT_MAX) : static_cast<T>(FLT_MAX)};
        AscendC::DataCopyPad(xLocal, xGm[tileIndex * tileDataNum], copyParams1, padParams1);
        if (dupCount > 0) {
            AscendC::Duplicate(
                xLocal[align8], descending ? static_cast<T>(-FLT_MAX) : static_cast<T>(FLT_MAX), dupCount);
        }
        AscendC::DataCopyExtParams copyParams2{
            static_cast<uint16_t>(1), static_cast<uint32_t>(this->dimW * sizeof(uint32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> padParams2{
            true, 0, static_cast<uint8_t>(padLen),
            descending ? static_cast<uint32_t>(-FLT_MAX) : static_cast<uint32_t>(FLT_MAX)};
        AscendC::DataCopyPad(indexLocal, indexGm[tileIndex * tileDataNum], copyParams2, padParams2);
        if (dupCount > 0) {
            AscendC::Duplicate(
                indexLocal[align8], descending ? static_cast<uint32_t>(-FLT_MAX) : static_cast<uint32_t>(FLT_MAX),
                dupCount);
        }
    }
    inQueueX.EnQue(xLocal);
    inQueueX.EnQue(indexLocal);
}

template <typename T>
__aicore__ inline void SortV2<T>::CopyOut(int32_t tileIndex)
{
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::LocalTensor<uint32_t> dstIndexLocal = dstIndexQ.DeQue<uint32_t>();
    if (axis == 0) {
        AscendC::DataCopyExtParams copyParams1{
            static_cast<uint16_t>(dimH), static_cast<uint32_t>(sizeof(T)), 0,
            static_cast<uint32_t>((dimW - 1) * sizeof(T)), 0};
        AscendC::DataCopyPad(yGm[tileIndex], yLocal, copyParams1);
        AscendC::DataCopyExtParams copyParams2{
            static_cast<uint16_t>(dimH), static_cast<uint32_t>(sizeof(uint32_t)), 0,
            static_cast<uint32_t>((dimW - 1) * sizeof(uint32_t)), 0};
        AscendC::DataCopyPad(dstIndexGm[tileIndex], dstIndexLocal, copyParams2);
    } else if (axis == 1) {
        AscendC::DataCopyExtParams copyParams1{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sliceLen * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[tileIndex * this->tileDataNum], yLocal, copyParams1);
        AscendC::DataCopyExtParams copyParams2{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sliceLen * sizeof(uint32_t)), 0, 0, 0};
        AscendC::DataCopyPad(dstIndexGm[tileIndex * this->tileDataNum], dstIndexLocal, copyParams2);
    }
    outQueueY.FreeTensor(yLocal);
    dstIndexQ.FreeTensor(dstIndexLocal);
}

template <typename T>
__aicore__ inline void SortV2<T>::Compute(int32_t tileIndex)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<uint32_t> indexLocal = inQueueX.DeQue<uint32_t>();
    AscendC::LocalTensor<T> sortedLocal = calcQ.AllocTensor<T>();
    AscendC::LocalTensor<T> concatTmpLocal = concatQ.AllocTensor<T>();
    AscendC::LocalTensor<T> sortTmpLocal = tmpQ.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint32_t> dstIndexLocal = dstIndexQ.AllocTensor<uint32_t>();
    AscendC::LocalTensor<T> concatLocal;
    AscendC::Concat(concatLocal, xLocal, concatTmpLocal, concatRepeat);
    xLocal.SetSize(realSortLen);
    sortedLocal.SetSize(realSortLen);
    sortTmpLocal.SetSize(realSortLen);
    if (!descending) {
        AscendC::Muls(concatLocal, concatLocal, static_cast<T>(-1), realSortLen);
    }
    AscendC::Sort<T, true>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, sortRepeat);
    AscendC::Extract(yLocal, dstIndexLocal, sortedLocal, extractRepeat);
    if (!descending) {
        AscendC::Muls(yLocal, yLocal, static_cast<T>(-1), realSortLen);
    }
    outQueueY.EnQue(yLocal);
    dstIndexQ.EnQue(dstIndexLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueX.FreeTensor(indexLocal);
    calcQ.FreeTensor(sortedLocal);
    concatQ.FreeTensor(concatTmpLocal);
    tmpQ.FreeTensor(sortTmpLocal);
}

template <typename T>
__aicore__ inline void SortV2<T>::Process()
{
    if (coreDataNum == 0) {
        return;
    }
    for (uint32_t i = 0; i < coreDataNum; ++i) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsSortV2
#endif // SortV2_H