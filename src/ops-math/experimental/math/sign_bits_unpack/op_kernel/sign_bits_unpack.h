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
 * \file sign_bits_unpack.h
 * \brief
 */
#ifndef SIGN_BITS_UNPACK_H_
#define SIGN_BITS_UNPACK_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sign_bits_unpack_tiling_data.h"
#include "sign_bits_unpack_tiling_key.h"

#include "kernel_operator.h"

constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t SINGLE_BUFFER = 1;

namespace NsSignBitsUnpack {

using namespace AscendC;

template <typename T>
class KernelSignBitsUnpack {
public:
    __aicore__ inline KernelSignBitsUnpack(){};

    __aicore__ inline void Init(
        GM_ADDR self, GM_ADDR out, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
        uint64_t tailBlockNum, uint64_t bufferOpen);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void CopyOut(int64_t progress);
    __aicore__ inline void Compute(int64_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> inQueueSelf;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> outQueueOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0;

    AscendC::GlobalTensor<uint8_t> selfGm;
    AscendC::GlobalTensor<T> outGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t tileDataNumOut = 0;
    uint64_t processDataNumIn = 0;
    uint64_t processDataNumOut = 0;
};

template <typename T>
__aicore__ inline void KernelSignBitsUnpack<T>::Init(
    GM_ADDR self, GM_ADDR out, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
    uint64_t tailBlockNum, uint64_t bufferOpen)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    uint64_t outCoreIndx = bigCoreDataNum * coreId * 8;
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
        outCoreIndx -= 8 * (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }
    uint64_t BUFFER_NUM = DOUBLE_BUFFER;
    if (bufferOpen == 0) {
        BUFFER_NUM = SINGLE_BUFFER;
    }
    uint64_t outCoreNum = this->coreDataNum * 8;
    this->tileDataNumOut = this->tileDataNum * 8;
    selfGm.SetGlobalBuffer((__gm__ uint8_t*)self + globalBufferIndex, this->coreDataNum);
    outGm.SetGlobalBuffer((__gm__ T*)out + outCoreIndx, outCoreNum);
    pipe.InitBuffer(inQueueSelf, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
    pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileDataNumOut * sizeof(T));
    if constexpr (std::is_same_v<T, float>) {
        pipe.InitBuffer(tmpQueue0, this->tileDataNumOut * sizeof(half));
    }
}

template <typename T>
__aicore__ inline void KernelSignBitsUnpack<T>::CopyIn(int64_t progress)
{
    AscendC::LocalTensor<uint8_t> selfLocal = inQueueSelf.template AllocTensor<uint8_t>();
    AscendC::DataCopy(selfLocal, selfGm[progress * this->tileDataNum], this->processDataNumIn);
    inQueueSelf.EnQue(selfLocal);
}

template <typename T>
__aicore__ inline void KernelSignBitsUnpack<T>::CopyOut(int64_t progress)
{
    AscendC::LocalTensor<T> outLocal = outQueueOut.template DeQue<T>();
    AscendC::DataCopy(outGm[progress * this->tileDataNumOut], outLocal, this->processDataNumOut);
    outQueueOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void KernelSignBitsUnpack<T>::Compute(int64_t progress)
{
    if constexpr (std::is_same_v<T, float>) {
        AscendC::LocalTensor<uint8_t> selfLocal = inQueueSelf.template DeQue<uint8_t>();
        AscendC::LocalTensor<float> outLocal = outQueueOut.template AllocTensor<float>();
        AscendC::LocalTensor<half> tmp0Local = tmpQueue0.AllocTensor<half>();
        AscendC::Duplicate(outLocal, static_cast<float>(1.0), this->processDataNumOut);
        AscendC::Select(
            outLocal, selfLocal, outLocal, static_cast<float>(-1.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            this->processDataNumOut);
        outQueueOut.template EnQue<float>(outLocal);
        inQueueSelf.FreeTensor(selfLocal);
    } else {
        AscendC::LocalTensor<uint8_t> selfLocal = inQueueSelf.template DeQue<uint8_t>();
        AscendC::LocalTensor<half> outLocal = outQueueOut.template AllocTensor<half>();
        AscendC::Duplicate(outLocal, static_cast<half>(1.0), this->processDataNumOut);
        // 数据对齐
        AscendC::Select(
            outLocal, selfLocal, outLocal, static_cast<half>(1.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            this->processDataNumOut);
        AscendC::Select(
            outLocal, selfLocal, outLocal, static_cast<half>(-1.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            this->processDataNumOut);
        outQueueOut.template EnQue<half>(outLocal);
        inQueueSelf.FreeTensor(selfLocal);
    }
}

template <typename T>
__aicore__ inline void KernelSignBitsUnpack<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNumIn = this->tileDataNum;
    this->processDataNumOut = this->processDataNumIn * 8;
    for (int64_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNumIn = this->tailDataNum;
    this->processDataNumOut = this->processDataNumIn * 8;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsSignBitsUnpack
#endif // SIGN_BITS_UNPACK_H