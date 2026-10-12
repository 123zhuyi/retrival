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
 * \file lin_space_d.h
 * \brief
 */
#ifndef LIN_SPACE_D_H
#define LIN_SPACE_D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lin_space_d_tiling_data.h"
#include "lin_space_d_tiling_key.h"

namespace NsLinSpaceD {
using namespace AscendC;

template <typename TStart, typename TEnd>
class LinSpaceD {
public:
    __aicore__ inline LinSpaceD(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const LinSpaceDTilingData* tilingData);
    __aicore__ inline void Process();

    constexpr static uint8_t FLOAT_DATA_TYPE_SIZE = 4;
    constexpr static uint8_t BLOCK_SIZE = 32;
    constexpr static uint8_t TILE_ELEM_NUM = BLOCK_SIZE / FLOAT_DATA_TYPE_SIZE; // float的Tile元素数（8）
    constexpr static uint8_t BUFFER_NUM = 2;                                    // 双缓冲

private:
    __aicore__ inline void ParseTilingData(const LinSpaceDTilingData* tilingData);
    __aicore__ inline void Compute(int32_t tileIdx);
    __aicore__ inline void CopyOut(int32_t tileIdx);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpHalf, tmpFloat;
    TBuf<QuePosition::VECCALC> tmpStart, tmpEnd;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> zGm;
    GlobalTensor<TStart> startGm;
    GlobalTensor<TEnd> endGm;
    uint32_t currentBlockLength;
    uint32_t blockLength;    // 当前核心处理的float总元素数
    uint32_t tileNum;        // 当前核心的Tile总数
    uint32_t lastTileLength; // 最后一个Tile的float元素数
    uint32_t blockOffset;    // 当前核心的全局float元素偏移
    uint32_t origSize;       // 原始有效元素数
    float step, startF, endF;
    TStart start;
    TEnd end;
};

template <typename TStart, typename TEnd>
__aicore__ inline void LinSpaceD<TStart, TEnd>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, const LinSpaceDTilingData* tilingData)
{
    startGm.SetGlobalBuffer((__gm__ TStart*)x, 1);
    endGm.SetGlobalBuffer((__gm__ TEnd*)y, 1);
    this->start = startGm.GetValue(0);
    this->end = endGm.GetValue(0);

    ParseTilingData(tilingData);

    zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + this->blockOffset, this->currentBlockLength);
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_ELEM_NUM * sizeof(float));
    pipe.InitBuffer(tmpHalf, sizeof(half));
    pipe.InitBuffer(tmpFloat, sizeof(float));
    pipe.InitBuffer(tmpStart, sizeof(TStart));
    pipe.InitBuffer(tmpEnd, sizeof(TEnd));
}

template <typename TStart, typename TEnd>
__aicore__ inline void LinSpaceD<TStart, TEnd>::Process()
{
    if constexpr (Std::is_same<TStart, bfloat16_t>::value) {
        this->startF = ToFloat(this->start);
    } else if constexpr (Std::is_same<TStart, uint8_t>::value) {
        LocalTensor<TStart> uint8Local = tmpStart.Get<TStart>();
        LocalTensor<half> halfLocal = tmpHalf.Get<half>();
        LocalTensor<float> floatLocal = tmpFloat.Get<float>();
        uint8Local.SetValue(0, this->start);
        Cast(halfLocal, uint8Local, RoundMode::CAST_NONE, 1);
        Cast(floatLocal, halfLocal, RoundMode::CAST_NONE, 1);
        this->startF = floatLocal.GetValue(0);
    } else if constexpr (Std::is_same<TStart, float>::value) {
        this->startF = this->start;
    } else {
        this->startF = static_cast<float>(this->start);
    }
    if constexpr (Std::is_same<TEnd, bfloat16_t>::value) {
        this->endF = ToFloat(this->end);
    } else if constexpr (Std::is_same<TEnd, uint8_t>::value) {
        LocalTensor<half> halfLocal = tmpHalf.Get<half>();
        LocalTensor<float> floatLocal = tmpFloat.Get<float>();
        LocalTensor<TEnd> uint8Local = tmpEnd.Get<TEnd>();
        uint8Local.SetValue(0, this->end);
        Cast(halfLocal, uint8Local, RoundMode::CAST_NONE, 1);
        Cast(floatLocal, halfLocal, RoundMode::CAST_NONE, 1);
        this->endF = floatLocal.GetValue(0);
    } else if constexpr (Std::is_same<TEnd, float>::value) {
        this->endF = this->end;
    } else {
        this->endF = static_cast<float>(this->end);
    }
    if (this->origSize == 1) {
        uint32_t coreId = GetBlockIdx();
        if (coreId >= 1)
            return;
        LocalTensor<float> tileOutput = outQueueZ.AllocTensor<float>();
        Duplicate<float>(tileOutput, startF, TILE_ELEM_NUM);
        outQueueZ.EnQue(tileOutput);
        CopyOut(0);
    } else {
        this->step = (endF - startF) / (origSize - 1);
        for (int32_t tileIdx = 0; tileIdx < tileNum; tileIdx++) {
            Compute(tileIdx);
            CopyOut(tileIdx);
        }
    }
}

template <typename TStart, typename TEnd>
__aicore__ inline void LinSpaceD<TStart, TEnd>::ParseTilingData(const LinSpaceDTilingData* tilingData)
{
    uint32_t coreId = GetBlockIdx();
    this->currentBlockLength = 0;

    if (coreId < tilingData->formerNum) {
        this->blockLength = tilingData->formerLength;
        this->tileNum = tilingData->formerTileNum;
        this->lastTileLength = tilingData->formerLastTileLength;
        this->blockOffset = tilingData->formerLength * coreId;
        this->currentBlockLength = tilingData->formerLength;
    } else {
        this->blockLength = tilingData->tailLength;
        this->tileNum = tilingData->tailTileNum;
        this->lastTileLength = tilingData->tailLastTileLength;
        this->blockOffset = tilingData->formerLength * tilingData->formerNum +
                            tilingData->tailLength * (coreId - tilingData->formerNum);
        this->currentBlockLength = tilingData->tailLength;
    }
    this->origSize = tilingData->totalLength;
}

template <typename TStart, typename TEnd>
__aicore__ inline void LinSpaceD<TStart, TEnd>::Compute(int32_t tileIdx)
{
    LocalTensor<float> tileOutput = outQueueZ.AllocTensor<float>();
    int32_t tileElemNum = TILE_ELEM_NUM; // 8
    int32_t tileGlobalStart = blockOffset + tileIdx * tileElemNum;
    float firstValue = startF + step * static_cast<float>(tileGlobalStart);
    AscendC::ArithProgression<float>(tileOutput, firstValue, step, tileElemNum);
    outQueueZ.EnQue(tileOutput);
}

template <typename TStart, typename TEnd>
__aicore__ inline void LinSpaceD<TStart, TEnd>::CopyOut(int32_t tileIdx)
{
    LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    int32_t copyElemNum;
    if (tileIdx == tileNum - 1) {
        copyElemNum = lastTileLength;
    } else {
        copyElemNum = TILE_ELEM_NUM;
    }

    int32_t copyOffset = tileIdx * TILE_ELEM_NUM;
    if (copyOffset + copyElemNum > blockLength) {
        copyElemNum = blockLength - copyOffset;
    }
    DataCopy(zGm[copyOffset], zLocal, copyElemNum);
    outQueueZ.FreeTensor(zLocal);
}

} // namespace NsLinSpaceD
#endif // LIN_SPACE_D_H