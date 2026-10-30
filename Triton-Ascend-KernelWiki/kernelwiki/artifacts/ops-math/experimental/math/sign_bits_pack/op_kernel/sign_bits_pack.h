/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sign_bits_pack.h
 * \brief
 */
#ifndef SIGN_BITS_PACK_H_
#define SIGN_BITS_PACK_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sign_bits_pack_tiling_data.h"
#include "sign_bits_pack_tiling_key.h"

namespace NsSignBitsPack {

using namespace AscendC;

template <typename T>
class KernelSignBitsPack {
public:
    __aicore__ inline KernelSignBitsPack(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SignBitsPackTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyInLast(int32_t progress);
    __aicore__ inline void CopyOutLast(int32_t progress);

private:
    TQue<QuePosition::VECIN, 1> inQueueIN;
    TQue<QuePosition::VECOUT, 1> outQueueOUT;
    GlobalTensor<T> xGm;
    GlobalTensor<uint8_t> yGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    int32_t bufferNum = 2;
    uint32_t rightPaddingElemNums = 0;
    uint32_t lastCopyLength = 0;
    uint32_t lastCalcLength = 0;
};

template <typename T>
__aicore__ inline void KernelSignBitsPack<T>::Init(
    GM_ADDR x, GM_ADDR y, const SignBitsPackTilingData* tilingData, TPipe* pipeIn)
{
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }

    this->rightPaddingElemNums = tilingData->rightPaddingElemNums;
    this->lastCopyLength = tilingData->lastCopyLength;
    this->lastCalcLength = tilingData->lastCalcLength;
    this->bufferNum = 1;
    if (static_cast<int32_t>(tilingData->usedDb) == 1) {
        this->bufferNum = 2;
    }
    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ uint8_t*)y + globalBufferIndex / 8, this->coreDataNum / 8);
    pipeIn->InitBuffer(inQueueIN, this->bufferNum, this->tileDataNum * sizeof(T));
    pipeIn->InitBuffer(outQueueOUT, this->bufferNum, this->tileDataNum / 4 * sizeof(uint8_t));
}

template <typename T>
__aicore__ inline void KernelSignBitsPack<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inQueueIN.AllocTensor<T>();

    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);

    inQueueIN.EnQue(xLocal);
}

template <typename T> // DatacopyPad
__aicore__ inline void KernelSignBitsPack<T>::CopyInLast(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inQueueIN.AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->lastCopyLength * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(this->rightPaddingElemNums), (T)(-1)};
    AscendC::DataCopyPad(xLocal, xGm[progress * this->tileDataNum], copyParams, padParams);

    inQueueIN.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void KernelSignBitsPack<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<uint8_t> yLocal = outQueueOUT.DeQue<uint8_t>();

    AscendC::DataCopyExtParams copyParams{
        1, static_cast<uint32_t>((this->processDataNum / 8) * sizeof(uint8_t)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[progress * this->tileDataNum / 8], yLocal, copyParams);

    outQueueOUT.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void KernelSignBitsPack<T>::CopyOutLast(int32_t progress)
{
    AscendC::LocalTensor<uint8_t> yLocal = outQueueOUT.DeQue<uint8_t>();

    AscendC::DataCopyExtParams copyParams{
        1, static_cast<uint32_t>(this->lastCopyLength / 8 * sizeof(uint8_t)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[progress * this->tileDataNum / 8], yLocal, copyParams);

    outQueueOUT.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void KernelSignBitsPack<T>::Compute(int32_t progress)
{
    LocalTensor<T> xLocal = inQueueIN.DeQue<T>();
    LocalTensor<uint8_t> yLocal = outQueueOUT.AllocTensor<uint8_t>();

    CompareScalar(yLocal, xLocal, (T)(0), CMPMODE::GE, this->processDataNum);

    outQueueOUT.EnQue(yLocal);
    inQueueIN.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void KernelSignBitsPack<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    if (AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) // 最后一个核
    {
        this->processDataNum = this->lastCalcLength;
        CopyInLast(loopCount - 1);
    } else {
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount - 1);
    }

    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsSignBitsPack
#endif // SIGN_BITS_PACK_H