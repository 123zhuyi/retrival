/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill_diagonal_v2_dense.h
 * \brief
 */
#ifndef FILL_DIAGONAL_DENSE_V2_H
#define FILL_DIAGONAL_DENSE_V2_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace FillDiagonalV2Dense {
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t UB_BLOCK_SIZE = 32;

template<typename T, typename MTE_T>
class Kernel {
public:
    __aicore__ inline Kernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR fill_value, const FillDiagonalV2TilingData* tilingData)
    {
        uint64_t typeMult = sizeof(T) / sizeof(MTE_T);
        const uint64_t maxTileLength = tilingData->ubSize / BUFFER_NUM / sizeof(T);
        if (AscendC::GetBlockIdx() < AscendC::GetBlockNum() - 1) {
            blockLength = tilingData->blockLength;
        } else {
            blockLength = tilingData->lastBlockLength;
        }
        blockStart = tilingData->blockLength * AscendC::GetBlockIdx();
        tileNum = blockLength / maxTileLength;
        tileLength = tileNum ? maxTileLength : blockLength;
        miniTileLength = blockLength - tileNum * tileLength;
        xGm.SetGlobalBuffer((__gm__ MTE_T *)x + blockStart * typeMult, blockLength * typeMult);
        this->valGm.SetGlobalBuffer((__gm__ T *)fill_value, 1);
        pipe.InitBuffer(que, BUFFER_NUM, tileLength * sizeof(T));
        step = tilingData->step;
        end = tilingData->end;
        this->val = this->valGm.GetValue(0);
    }

    __aicore__ inline void Process()
    {
        uint64_t loopCount = tileNum;
        for (uint64_t progress = 0; progress < loopCount; ++progress) {
            CopyIn(progress, tileLength);
            Compute(progress, tileLength);
            CopyOut(progress, tileLength);
        }
        if (miniTileLength) {
            CopyIn(tileNum, miniTileLength);
            Compute(tileNum, miniTileLength);
            CopyOut(tileNum, miniTileLength);
        }
    }
private:
    __aicore__ inline void CopyIn(uint64_t progress, uint64_t length)
    {
        uint64_t typeMult = sizeof(T) / sizeof(MTE_T);
        AscendC::LocalTensor<MTE_T> xLocal = this->que.template AllocTensor<MTE_T>();
        const uint64_t ubBlockLength = UB_BLOCK_SIZE / sizeof(MTE_T);
        uint64_t alignedLength = length * typeMult / ubBlockLength * ubBlockLength;
        if (alignedLength != 0) {
            AscendC::DataCopy(xLocal, xGm[progress * this->tileLength * typeMult], alignedLength);
        }
        if (length * typeMult > alignedLength) {
            #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
            AscendC::DataCopy(xLocal[alignedLength], xGm[progress * this->tileLength * typeMult + alignedLength], ubBlockLength);
            #else
            AscendC::DataCopyParams dataCopyParams{1, static_cast<uint16_t>((length * typeMult - alignedLength) * sizeof(MTE_T)), 0, 0};
            uint8_t rightPad = static_cast<uint8_t>(ubBlockLength + alignedLength - length * typeMult);
            AscendC::DataCopyPadParams padParams{true, 0, rightPad, 0};
            AscendC::DataCopyPad(xLocal[alignedLength], xGm[progress * this->tileLength * typeMult + alignedLength], dataCopyParams, padParams);
            #endif
        }
        this->que.template EnQue<AscendC::QuePosition::GM, AscendC::QuePosition::VECIN, MTE_T>(xLocal);
    }

    __aicore__ inline void Compute(uint64_t progress, uint64_t length)
    {
        #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
        const uint64_t ubBlockLength = UB_BLOCK_SIZE / sizeof(T);
        if (length % ubBlockLength != 0) {
            length += ubBlockLength - (length % ubBlockLength);
        }
        #endif
        AscendC::LocalTensor<MTE_T> xLocal = this->que.template DeQue<AscendC::QuePosition::GM, AscendC::QuePosition::VECIN, MTE_T>();
        AscendC::LocalTensor<T> dataLocal = xLocal.template ReinterpretCast<T>();
        uint64_t tileStart = blockStart + progress * this->tileLength;
        uint64_t tileEnd = tileStart + length;
        uint64_t diagIndex;
        if (tileStart % this->step == 0) {
            diagIndex = tileStart;
        } else {
            diagIndex = (tileStart / this->step + 1) * this->step;
        }
        while (diagIndex < this->end && diagIndex < tileEnd) {
            dataLocal(diagIndex - tileStart) = this->val;
            diagIndex += this->step;
        }
        this->que.template EnQue<AscendC::QuePosition::VECOUT, AscendC::QuePosition::GM, MTE_T>(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t progress, uint64_t length)
    {
        uint64_t typeMult = sizeof(T) / sizeof(MTE_T);
        AscendC::LocalTensor<MTE_T> xLocal = this->que.template DeQue<AscendC::QuePosition::VECOUT, AscendC::QuePosition::GM, MTE_T>();
        const uint64_t ubBlockLength = UB_BLOCK_SIZE / sizeof(MTE_T);
        uint64_t alignedLength = length * typeMult / ubBlockLength * ubBlockLength;
        if (alignedLength != 0) {
            AscendC::DataCopy(xGm[progress * this->tileLength * typeMult], xLocal, alignedLength);
        }
        if (length * typeMult > alignedLength) {
            #if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
            AscendC::DataCopy(xGm[progress * this->tileLength * typeMult + alignedLength], xLocal[alignedLength], ubBlockLength);
            #else
            AscendC::DataCopyParams dataCopyParams{1, static_cast<uint16_t>((length * typeMult - alignedLength) * sizeof(MTE_T)), 0, 0};
            AscendC::DataCopyPad(xGm[progress * this->tileLength * typeMult + alignedLength], xLocal[alignedLength], dataCopyParams);
            #endif
        }
        this->que.template FreeTensor(xLocal);
    }

    T val;
    uint64_t step;
    uint64_t end;
    uint64_t blockLength;
    uint64_t blockStart;
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t miniTileLength;
    AscendC::GlobalTensor<MTE_T> xGm;
    AscendC::GlobalTensor<T> valGm;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> que;
    AscendC::TPipe pipe;
};
} // namespace FillDiagoanlV2Dense
#endif