/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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
 * \file masked_fill.h
 * \brief
 */
#ifndef MASKED_FILL_H
#define MASKED_FILL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "masked_fill_tiling_data.h"
#include "masked_fill_tiling_key.h"

namespace NsMaskedFill {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;

template <typename T>
class MaskedFill {
public:
    __aicore__ inline MaskedFill() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR mask, GM_ADDR valueT, GM_ADDR z, const MaskedFillTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MaskedFillTilingData* tilingData);
    __aicore__ inline uint32_t MapXGlobalIdxToMaskGlobalIdx(uint32_t xGlobalIdx);
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align);
    __aicore__ inline uint32_t GetCopyCount(uint32_t tileIdx);
    __aicore__ inline void CopyIn(uint32_t tileIdx);
    __aicore__ inline void Compute(uint32_t tileIdx);
    __aicore__ inline void CopyOut(uint32_t tileIdx);

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueM;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<T> xGm;
    GlobalTensor<bool> maskGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<T> zGm;

    // -------------------------- 广播变量 ------------------------------
    uint32_t size;                   // x总元素数
    uint32_t shape[128];             // x原始shape
    uint32_t numshapes;              // x维度数
    uint32_t osize;                  // mask总元素数
    uint32_t oshape[128];            // mask原始shape
    uint32_t onumshapes;             // mask维度数
    uint32_t dimOffsetXList[128];    // x的维度偏移表
    uint32_t dimOffsetMaskList[128]; // mask的维度偏移表
    int32_t alignStart;              // 广播对齐起始维度

    // -------------------------- 分片变量 ------------------------------
    uint32_t tileNum;          // 当前Core的Tile总数
    uint32_t alignNum_x;       // x的对齐元素数
    uint32_t tileLength_x;     // x的常规Tile长度（对齐后）
    uint32_t lastTileLength_x; // x的尾Tile长度（对齐后）
    uint32_t gmStartIdx;       // 当前Core的GM起始索引
    uint32_t localValidElem;   // 当前Core的有效元素数

    // -------------------------- 缓存变量 ------------------------------
    uint32_t xCopyCountCache[BUFFER_NUM] = {0U}; // x拷贝数缓存
    T value;                                     // 填充
};

template <typename T>
__aicore__ inline void MaskedFill<T>::Init(
    GM_ADDR x, GM_ADDR mask, GM_ADDR valueT, GM_ADDR z, const MaskedFillTilingData* tilingData)
{
    valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(valueT), 1);
    this->value = valueGm.GetValue(0);
    ParseTilingData(tilingData);

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + this->gmStartIdx, this->localValidElem);
    maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(mask), this->osize);
    zGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(z) + this->gmStartIdx, this->localValidElem);

    uint32_t xBufSize = this->tileLength_x * sizeof(T);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, xBufSize);
    pipe.InitBuffer(inQueueM, BUFFER_NUM, xBufSize); // mask缓冲区大小与x一致
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, xBufSize);
}

template <typename T>
__aicore__ inline void MaskedFill<T>::Process()
{
    if (this->localValidElem == 0U || this->tileNum == 0U)
        return;
    for (uint32_t i = 0U; i < this->tileNum; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

template <typename T>
__aicore__ inline void MaskedFill<T>::ParseTilingData(const MaskedFillTilingData* tilingData)
{
    this->alignNum_x = ALIGN_BYTES / sizeof(T);
    this->size = tilingData->xSize;
    this->osize = tilingData->maskSize;
    this->numshapes = tilingData->xNumShapes;
    this->onumshapes = tilingData->maskNumShapes;
    for (uint32_t i = 0; i < this->numshapes; i++) {
        this->shape[i] = tilingData->sShape[i];
    }
    for (uint32_t i = 0; i < this->onumshapes; i++) {
        this->oshape[i] = tilingData->maskShape[i];
    }

    uint32_t dimOffsetX = 1;
    dimOffsetXList[this->numshapes - 1] = 1;
    for (int32_t i = this->numshapes - 2; i >= 0; i--) {
        dimOffsetX *= this->shape[i + 1];
        dimOffsetXList[i] = dimOffsetX;
    }
    uint32_t dimOffsetMask = 1;
    dimOffsetMaskList[this->onumshapes - 1] = 1;
    for (int32_t i = this->onumshapes - 2; i >= 0; i--) {
        dimOffsetMask *= this->oshape[i + 1];
        dimOffsetMaskList[i] = dimOffsetMask;
    }
    this->alignStart = this->numshapes - this->onumshapes;

    uint32_t blockIdx = GetBlockIdx();
    uint32_t gmStartIdx = 0U;
    uint32_t blockTotalElemAligned = 0U;

    if (blockIdx < tilingData->formerNum) {
        blockTotalElemAligned = tilingData->formerLength;
        gmStartIdx = blockTotalElemAligned * blockIdx;
        this->tileNum = tilingData->formerTileNum;
        this->tileLength_x = tilingData->formerTileLength;
        this->lastTileLength_x = tilingData->formerLastTileLength;
    } else {
        blockTotalElemAligned = tilingData->tailLength;
        gmStartIdx = tilingData->formerLength * tilingData->formerNum +
                     blockTotalElemAligned * (blockIdx - tilingData->formerNum);
        this->tileNum = tilingData->tailTileNum;
        this->tileLength_x = tilingData->tailTileLength;
        this->lastTileLength_x = tilingData->tailLastTileLength;
    }

    this->gmStartIdx = gmStartIdx;
    if (this->gmStartIdx >= tilingData->xSize) {
        this->localValidElem = 0U;
        return;
    }
    this->localValidElem = (this->gmStartIdx + blockTotalElemAligned > tilingData->xSize) ?
                               (tilingData->xSize - this->gmStartIdx) :
                               blockTotalElemAligned;
}

template <typename T>
__aicore__ inline uint32_t MaskedFill<T>::MapXGlobalIdxToMaskGlobalIdx(uint32_t xGlobalIdx)
{
    uint32_t xIndice[128] = {0};
    uint32_t temp = xGlobalIdx;
    for (int32_t i = this->numshapes - 1; i >= 0; i--) {
        xIndice[i] = temp % this->shape[i];
        temp /= this->shape[i];
    }

    uint32_t maskIndice[128] = {0};
    for (int32_t i = 0; i < this->onumshapes; i++) {
        int32_t xDimIdx = this->alignStart + i;
        uint32_t maskDimSize = this->oshape[i];

        maskIndice[i] = (maskDimSize == 1) ? 0 : xIndice[xDimIdx];
    }

    uint32_t maskGlobalIdx = 0;
    for (int32_t i = 0; i < this->onumshapes; i++) {
        maskGlobalIdx += maskIndice[i] * dimOffsetMaskList[i];
    }
    return maskGlobalIdx;
}

template <typename T>
__aicore__ inline uint32_t MaskedFill<T>::AlignUp(uint32_t value, uint32_t align)
{
    return (value == 0U) ? 0U : ((value + align - 1U) & ~(align - 1U));
}

template <typename T>
__aicore__ inline uint32_t MaskedFill<T>::GetCopyCount(uint32_t tileIdx)
{
    uint32_t tileLen = (tileIdx == this->tileNum - 1U) ? this->lastTileLength_x : this->tileLength_x;
    uint32_t offsetInCore = tileIdx * this->tileLength_x;
    uint32_t maxValid = this->localValidElem - offsetInCore;

    return (maxValid <= 0U) ? 0U : ((tileLen <= maxValid) ? tileLen : AlignUp(maxValid, this->alignNum_x));
}

template <typename T>
__aicore__ inline void MaskedFill<T>::CopyIn(uint32_t tileIdx)
{
    uint32_t offsetInCore = tileIdx * this->tileLength_x;
    if (offsetInCore >= this->localValidElem) {
        this->xCopyCountCache[tileIdx % BUFFER_NUM] = 0U;
        return;
    }

    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    LocalTensor<bool> maskLocal = inQueueM.AllocTensor<bool>();

    uint32_t xCopyCount = GetCopyCount(tileIdx);
    this->xCopyCountCache[tileIdx % BUFFER_NUM] = xCopyCount;

    if (xCopyCount > 0U) {
        DataCopy(xLocal, xGm[offsetInCore], xCopyCount);
    }
    if (xCopyCount > 0U) {
        for (int32_t i = xCopyCount - 1; i >= 0; i--) {
            uint32_t xGlobalIdx = this->gmStartIdx + offsetInCore + i;
            uint32_t maskGlobalIdx = MapXGlobalIdxToMaskGlobalIdx(xGlobalIdx);
            maskLocal.SetValue(i, maskGm.GetValue(maskGlobalIdx));
        }
    }
    inQueueX.EnQue(xLocal);
    inQueueM.EnQue(maskLocal);
}

template <typename T>
__aicore__ inline void MaskedFill<T>::Compute(uint32_t tileIdx)
{
    uint32_t offsetInCore = tileIdx * this->tileLength_x;
    uint32_t bufIdx = tileIdx % BUFFER_NUM;
    if (offsetInCore >= this->localValidElem)
        return;

    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<bool> maskLocal = inQueueM.DeQue<bool>();
    LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

    uint32_t xCopyCount = this->xCopyCountCache[bufIdx];
    uint32_t maxValid = this->localValidElem - offsetInCore;
    uint32_t computeCount = Std::min(xCopyCount, maxValid);

    // 只支持half/float
    // AscendC::Select(zLocal, maskLocal, xLocal, static_cast<float>(this->value),
    // AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);

    for (int32_t i = computeCount - 1; i >= 0; i--) {
        // int16_t/uint16_t/half/int32_t/uint32_t/float/bfloat16_t 适用Duplicate接口
        if constexpr (
            Std::is_same<T, bfloat16_t>::value || Std::is_same<T, float>::value || Std::is_same<T, uint32_t>::value ||
            Std::is_same<T, int32_t>::value || Std::is_same<T, half>::value || Std::is_same<T, uint16_t>::value ||
            Std::is_same<T, int16_t>::value) {
            Duplicate<T>(zLocal, maskLocal.GetValue(i) ? this->value : xLocal.GetValue(i), i + 1);
        } else {
            zLocal.SetValue(i, maskLocal.GetValue(i) ? this->value : xLocal.GetValue(i));
        }
    }

    outQueueZ.EnQue(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueM.FreeTensor(maskLocal);
}

template <typename T>
__aicore__ inline void MaskedFill<T>::CopyOut(uint32_t tileIdx)
{
    uint32_t offsetInCore = tileIdx * this->tileLength_x;
    uint32_t bufIdx = tileIdx % BUFFER_NUM;
    if (offsetInCore >= this->localValidElem)
        return;

    LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    uint32_t copyCount = GetCopyCount(tileIdx);
    if (copyCount > 0U) {
        DataCopy(zGm[offsetInCore], zLocal, copyCount);
    }

    outQueueZ.FreeTensor(zLocal);
}

} // namespace NsMaskedFill
#endif // MASKED_FILL_H