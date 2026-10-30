/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
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
 * \file tril.h
 * \brief
 */
#ifndef __TRIL_H__
#define __TRIL_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tril_tiling_data.h"
#include "tril_tiling_key.h"

namespace NsTril {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t minNum = 1;
constexpr int keyOne = 1;
constexpr int keyTwo = 2;
constexpr int keyThree = 3;
constexpr int keyFour = 4;
constexpr int computeBatchSize = 256;
template <typename T>
class Tril {
public:
    __aicore__ inline Tril(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TrilTilingData* tilingData, uint32_t key);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SheerDup();
    __aicore__ inline void SheerZero();
    __aicore__ inline void NaivePath();
    __aicore__ inline void FastPath();
    __aicore__ inline void AllZero(uint32_t tileLength);
    __aicore__ inline void CopyIn(uint32_t GmOffset, uint32_t tileLength);
    __aicore__ inline void CopyOut(uint32_t GmOffset, uint32_t tileLength);
    __aicore__ inline void Compute(int32_t cnt, uint32_t initLength, int32_t adjust);

private:
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM>
        queBind; // Use TQueBind to replace QueI，QueO
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_X> yGm;
    uint32_t totalLengthAligned;
    int32_t matrixNum;
    int32_t matrixSize;
    int32_t rowLength;
    int32_t columnLength;
    int32_t diagVal;
    int32_t loopCnt;
    uint32_t fullTileLength;
    uint32_t lastTileLength;
    int32_t fullCnt;
    int32_t lastCnt;
    int32_t alignNum;
    uint32_t typeSize;
    uint32_t fullRowInc;
    uint32_t initLength;
    int32_t repeatTimes;
    uint32_t key;
};
template <typename T>
__aicore__ inline void Tril<T>::Init(GM_ADDR x, GM_ADDR y, const TrilTilingData* tilingData, uint32_t key)
{
    this->matrixNum = tilingData->matrixNum;
    this->matrixSize = tilingData->matrixSize;
    this->rowLength = tilingData->rowLength;
    this->columnLength = tilingData->columnLength;
    this->diagVal = tilingData->diagVal;
    this->fullCnt = tilingData->fullCnt;
    this->lastCnt = tilingData->lastCnt;
    if (tilingData->columnLength == 0) {
        this->columnLength = minNum;
    }
    this->fullRowInc = tilingData->fullTileLength / tilingData->columnLength;
    this->initLength = 1;
    // The result would not be the expected
    this->typeSize = tilingData->typeSize;
    if (this->typeSize == 0) {
        this->typeSize = sizeof(float);
    }
    this->repeatTimes = columnLength / (computeBatchSize / this->typeSize);
    this->key = key;
    uint64_t gmBuffer = tilingData->totalLengthAligned;
    xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, gmBuffer);
    yGm.SetGlobalBuffer((__gm__ DTYPE_X*)y, gmBuffer);
    this->loopCnt = tilingData->loopCnt;
    this->fullTileLength = tilingData->fullTileLength;
    this->lastTileLength = tilingData->lastTileLength;
    uint32_t singleBuffer = tilingData->fullTileLength;
    if (singleBuffer < tilingData->lastTileLength) {
        singleBuffer = tilingData->lastTileLength;
    }
    if (key == keyThree || key == keyFour) {
        pipe.InitBuffer(inQueueX, BUFFER_NUM, singleBuffer * this->typeSize);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, singleBuffer * this->typeSize);
    } else {
        pipe.InitBuffer(queBind, BUFFER_NUM, singleBuffer * this->typeSize);
    }
}
template <typename T>
__aicore__ inline void Tril<T>::SheerDup()
{
    uint32_t GmOffset = 0;
    for (int i = 0; i < this->loopCnt - 1; i++, GmOffset += this->fullTileLength) {
        auto bindLocal = queBind.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(bindLocal, xGm[GmOffset], this->fullTileLength);
        queBind.EnQue(bindLocal);
        bindLocal = queBind.DeQue<DTYPE_X>();
        AscendC::DataCopy(yGm[GmOffset], bindLocal, this->fullTileLength);
        queBind.FreeTensor(bindLocal);
    }
    auto bindLocal = queBind.AllocTensor<DTYPE_X>();
    AscendC::DataCopy(bindLocal, xGm[GmOffset], this->lastTileLength);
    queBind.EnQue(bindLocal);
    bindLocal = queBind.DeQue<DTYPE_X>();
    AscendC::DataCopy(yGm[GmOffset], bindLocal, this->lastTileLength);
    queBind.FreeTensor(bindLocal);
}
template <typename T>
__aicore__ inline void Tril<T>::SheerZero()
{
    uint32_t GmOffset = 0;
    for (int i = 0; i < this->loopCnt - 1; i++, GmOffset += this->fullTileLength) {
        CopyIn(GmOffset, this->fullTileLength);
        AllZero(this->fullTileLength);
        CopyOut(GmOffset, this->fullTileLength);
    }
    CopyIn(GmOffset, this->lastTileLength);
    AllZero(this->lastTileLength);
    CopyOut(GmOffset, this->lastTileLength);
}
template <typename T>
__aicore__ inline void Tril<T>::NaivePath()
{
    int32_t cnt = 0;
    for (int32_t i = 0; i < this->matrixNum; i++) {
        for (int32_t j = 0; j < this->rowLength; j++) {
            int32_t k = 0;
            while (k < this->columnLength && k - j <= this->diagVal) {
                DTYPE_X curr = xGm.GetValue(cnt);
                yGm.SetValue(cnt, curr);
                k++;
                cnt++;
            }
            while (k < this->columnLength) {
                yGm.SetValue(cnt, (DTYPE_X)0);
                k++;
                cnt++;
            }
        }
    }
}
template <typename T>
__aicore__ inline void Tril<T>::FastPath()
{
    uint32_t GmOffset = 0;
    int32_t init_row = 0;
    for (int num = 0; num < this->matrixNum; num++) {
        uint32_t calLength = this->initLength;
        if (this->diagVal <= 0) {
            init_row = 1 - diagVal;
        }
        for (int32_t i = 0; i < this->loopCnt - 1; i++) {
            CopyIn(GmOffset, this->fullTileLength);
            Compute(this->fullCnt, calLength, init_row);
            CopyOut(GmOffset, this->fullTileLength);
            if (init_row > 0) {
                init_row -= this->fullRowInc;
                if (init_row < 0) {
                    calLength -= init_row;
                    init_row = 0;
                }
            } else {
                calLength += this->fullRowInc;
            }
            GmOffset += this->fullTileLength;
        }
        CopyIn(GmOffset, this->lastTileLength);
        Compute(this->lastCnt, calLength, init_row);
        CopyOut(GmOffset, this->lastTileLength);
        GmOffset += this->lastTileLength;
    }
}
template <typename T>
__aicore__ inline void Tril<T>::AllZero(uint32_t tileLength)
{
    auto xLocal = inQueueX.DeQue<DTYPE_X>();
    auto yLocal = outQueueY.AllocTensor<DTYPE_X>();
    AscendC::Sub(yLocal, xLocal, xLocal, tileLength);
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
template <typename T>
__aicore__ inline void Tril<T>::CopyIn(uint32_t GmOffset, uint32_t tileLength)
{
    auto xLocal = inQueueX.AllocTensor<DTYPE_X>();
    AscendC::DataCopy(xLocal, xGm[GmOffset], tileLength);
    inQueueX.EnQue(xLocal);
}
template <typename T>
__aicore__ inline void Tril<T>::CopyOut(uint32_t GmOffset, uint32_t tileLength)
{
    auto yLocal = outQueueY.DeQue<DTYPE_X>();
    AscendC::DataCopy(yGm[GmOffset], yLocal, tileLength);
    outQueueY.FreeTensor(yLocal);
}
template <typename T>
__aicore__ inline void Tril<T>::Compute(int32_t cnt, uint32_t initLength, int32_t adjust)
{
    auto xLocal = inQueueX.DeQue<DTYPE_X>();
    auto yLocal = outQueueY.AllocTensor<DTYPE_X>();
    uint32_t localOffset = 0;
    uint32_t currLength = initLength;
    DTYPE_X scalarZero = 0;
    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};
    AscendC::Adds(yLocal, xLocal, scalarZero, mask, this->repeatTimes * cnt, {1, 1, 8, 8});
    for (int32_t i = 0; i < adjust; i++) {
        AscendC::Sub(yLocal[localOffset], xLocal[localOffset], xLocal[localOffset], currLength);
        currLength--;
        localOffset += this->columnLength;
    }
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
template <typename T>
__aicore__ inline void Tril<T>::Process()
{
    if (this->key == keyOne) {
        NaivePath();
    } else if (this->key == keyTwo) {
        SheerDup();
    } else if (this->key == keyThree) {
        SheerZero();
    } else if (key == keyFour) {
        FastPath();
    }
}
} // namespace NsTril
#endif // TRIL_H