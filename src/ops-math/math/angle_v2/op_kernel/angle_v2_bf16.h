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
 * \file angle_v2_bf16.h
 * \brief
 */
#ifndef _ANGLE_V2_BF16_H_
#define _ANGLE_V2_BF16_H_

#include "angle_v2_base.h"
#include "op_kernel/platform_util.h"

namespace AngleV2N {
using namespace AscendC;

template <typename xType, typename yType>
class AngleV2Bf16 : public AngleV2Base<yType> {
public:
    __aicore__ inline AngleV2Bf16() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData, TPipe* inputPipe)
    {
        pipe = inputPipe;
        this->BaseMemberDataInit(tilingData);
        this->ubBlockSize = Ops::Base::GetUbBlockSize();
        this->repeatTimes = (this->tileLength + this->mask - 1) / this->mask;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x) + this->offset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);

        // pipe alloc memory to queue, the unit is Bytes
        pipe->InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(xType));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));

        pipe->InitBuffer(maskBuf1, this->tileLength * sizeof(uint8_t));
        pipe->InitBuffer(zeroBuf, this->tileLength * sizeof(float));
        pipe->InitBuffer(piBuf, this->tileLength * sizeof(float));
        pipe->InitBuffer(nanBuf, this->tileLength * sizeof(float));
        pipe->InitBuffer(castBuf, this->tileLength * sizeof(float) * COEFFICENT);
    }

    __aicore__ inline void Process()
    {
        BufferGet();
        int64_t dataPerBlockIn = ubBlockSize / sizeof(xType);
        int64_t dataPerBlockOut = ubBlockSize / sizeof(yType);

        blockLenIn = this->tileLength / dataPerBlockIn;
        blockLenOut = this->tileLength / dataPerBlockOut;
        // loop count need to be doubled, due to double buffer
        for (int64_t i = 0; i < this->tileNum; i++) {
            int64_t coreOffset = i * this->tileLength;
            CopyIn(coreOffset);
            Compute(this->tileLength);
            CopyOut(coreOffset);
        }

        if (this->lastTileLength > 0) {
            int64_t coreOffset = this->blockLength - this->lastTileLength;
            repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;

            blockLenIn = this->lastTileLength / dataPerBlockIn;
            blockLenOut = this->lastTileLength / dataPerBlockOut;
            CopyIn(coreOffset);
            Compute(this->lastTileLength);
            CopyOut(coreOffset);
        }
    }

private:
    __aicore__ inline void BufferGet()
    {
        zeroTensor = zeroBuf.Get<float>();
        piTensor = piBuf.Get<float>();
        nanTensor = nanBuf.Get<float>();
        mask1 = maskBuf1.Get<uint8_t>();
        inputCast = castBuf.Get<float>();
        resCast = inputCast[this->tileLength];

        Duplicate(zeroTensor, static_cast<float>(0.0), this->mask, this->repeatTimes, this->dupDstBlockStride,
                  this->dupDstRepeatStride);
        Duplicate(piTensor, static_cast<float>(constData.const_pi), this->mask, this->repeatTimes,
                  this->dupDstBlockStride, this->dupDstRepeatStride);
        Duplicate(nanTensor, static_cast<float>(NAN), this->mask, this->repeatTimes, this->dupDstBlockStride,
                  this->dupDstRepeatStride);
    }

    __aicore__ inline void CopyIn(int64_t coreOffset)
    {
        // alloc tensor from queue memory
        LocalTensor<xType> xLocal = inQueue.AllocTensor<xType>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[coreOffset], {1, blockLenIn, 0, 0});
        // enque input tensors to VECIN queue
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t calCount)
    {
        // deque input tensors from VECIN queue
        LocalTensor<xType> input = inQueue.DeQue<xType>();
        LocalTensor<yType> result = outQueue.AllocTensor<yType>();

        Cast(inputCast, input, RoundMode::CAST_NONE, this->mask, this->repeatTimes, this->CastHighParams);
        // result = if input >= 0 then 0 else pi
        Compare(mask1, inputCast, zeroTensor, CMPMODE::GE, this->mask, this->repeatTimes, this->repeatParams);
        this->DoSelect(resCast, mask1, zeroTensor, piTensor, this->mask, this->repeatTimes);

        // select nan
        Compare(mask1, inputCast, inputCast, CMPMODE::EQ, this->mask, this->repeatTimes, this->repeatParams);
        this->DoSelect(resCast, mask1, resCast, nanTensor, this->mask, this->repeatTimes);
        Cast(result, resCast, RoundMode::CAST_RINT, this->mask, this->repeatTimes, this->CastDownParams);

        // enque the output tensor to VECOUT queue
        outQueue.EnQue<yType>(result);
        // free input tensors for reuse
        inQueue.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(int64_t coreOffset)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<yType> result = outQueue.DeQue<yType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(yGm[coreOffset], result, {1, blockLenOut, 0, 0});
        // free output tensor for reuse
        outQueue.FreeTensor(result);
    }

private:
    TPipe* pipe;
    ConstData constData;
    uint8_t repeatTimes;
    GlobalTensor<xType> xGm;
    GlobalTensor<yType> yGm;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> maskBuf1;
    TBuf<TPosition::VECCALC> piBuf;
    TBuf<TPosition::VECCALC> nanBuf;
    TBuf<TPosition::VECCALC> zeroBuf;
    TBuf<TPosition::VECCALC> castBuf;

    LocalTensor<float> zeroTensor;
    LocalTensor<float> piTensor;
    LocalTensor<float> nanTensor;
    LocalTensor<float> inputCast;
    LocalTensor<float> resCast;
    LocalTensor<uint8_t> mask1;
    uint16_t blockLenIn = 1;
    uint16_t blockLenOut = 1;
    uint32_t ubBlockSize;
};
} // namespace AngleV2N
#endif // _ANGLE_V2_BF16_H_
