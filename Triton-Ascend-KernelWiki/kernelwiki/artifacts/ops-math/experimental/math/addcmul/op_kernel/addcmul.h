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
 * \file addcmul.h
 * \brief
 */
#ifndef __ADDCMUL_H__
#define __ADDCMUL_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "addcmul_tiling_data.h"
#include "addcmul_tiling_key.h"

namespace NsAddcmul {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(
        GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
        uint32_t bigCoreDataNum, uint32_t finalBigTileNum, uint32_t finalSmallTileNum, uint32_t tileDataNum,
        uint32_t smallTailDataNum, uint32_t bigTailDataNum, uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t blockIdx = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }

        input_dataGm.SetGlobalBuffer((__gm__ TYPE_X*)input_data + globalBufferIndex, this->coreDataNum);
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X*)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X*)x2 + globalBufferIndex, this->coreDataNum);
        valueGm.SetGlobalBuffer((__gm__ TYPE_X*)value);
        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueINPUT_DATA, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
            this->f_value = ToFloat(valueGm.GetValue(0));
        } else {
            this->m_value = valueGm.GetValue(0);
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> input_dataLocal = inQueueINPUT_DATA.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_X> x1Local = inQueueX1.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_X> x2Local = inQueueX2.AllocTensor<TYPE_X>();
        DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(input_dataLocal, input_dataGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
        inQueueINPUT_DATA.EnQue(input_dataLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> x1Local = inQueueX1.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> x2Local = inQueueX2.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> input_dataLocal = inQueueINPUT_DATA.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            LocalTensor<float> p1 = tmp1.Get<float>();
            LocalTensor<float> p2 = tmp2.Get<float>();
            Cast(p1, x1Local, RoundMode::CAST_NONE, this->processDataNum);
            Cast(p2, x2Local, RoundMode::CAST_NONE, this->processDataNum);
            Mul(p1, p1, p2, this->processDataNum);
            Muls(p1, p1, this->f_value, this->processDataNum);
            Cast(p2, input_dataLocal, RoundMode::CAST_NONE, this->processDataNum);
            Add(p2, p1, p2, this->processDataNum);

            Cast(yLocal, p2, RoundMode::CAST_RINT, this->processDataNum);
        } else {
            Mul(x1Local, x1Local, x2Local, this->processDataNum);
            Muls(x1Local, x1Local, this->m_value, this->processDataNum);
            Add(yLocal, x1Local, input_dataLocal, this->processDataNum);
        }
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        inQueueINPUT_DATA.FreeTensor(input_dataLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2, inQueueINPUT_DATA;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<TYPE_X> input_dataGm;
    GlobalTensor<TYPE_X> x1Gm;
    GlobalTensor<TYPE_X> x2Gm;
    GlobalTensor<TYPE_X> valueGm;
    GlobalTensor<TYPE_X> yGm;
    TYPE_X m_value;
    float f_value = 0;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};

template <typename TYPE_X>
class KernelAddcmulTensor {
public:
    __aicore__ inline KernelAddcmulTensor() {}
    __aicore__ inline void Init(
        GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
        uint32_t bigCoreDataNum, uint32_t finalBigTileNum, uint32_t finalSmallTileNum, uint32_t tileDataNum,
        uint32_t smallTailDataNum, uint32_t bigTailDataNum, uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t blockIdx = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        input_dataGm.SetGlobalBuffer((__gm__ TYPE_X*)input_data + globalBufferIndex, this->coreDataNum);
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X*)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X*)x2 + globalBufferIndex, this->coreDataNum);
        valueGm.SetGlobalBuffer((__gm__ TYPE_X*)value + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueVALUE, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueINPUT_DATA, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> input_dataLocal = inQueueINPUT_DATA.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_X> x1Local = inQueueX1.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_X> x2Local = inQueueX2.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_X> valueLocal = inQueueVALUE.AllocTensor<TYPE_X>();
        DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(input_dataLocal, input_dataGm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(valueLocal, valueGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
        inQueueINPUT_DATA.EnQue(input_dataLocal);
        inQueueVALUE.EnQue(valueLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> x1Local = inQueueX1.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> x2Local = inQueueX2.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> valueLocal = inQueueVALUE.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> input_dataLocal = inQueueINPUT_DATA.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            LocalTensor<float> p1 = tmp1.Get<float>();
            LocalTensor<float> p2 = tmp2.Get<float>();
            Cast(p1, x1Local, RoundMode::CAST_NONE, this->processDataNum);
            Cast(p2, x2Local, RoundMode::CAST_NONE, this->processDataNum);
            Mul(p1, p1, p2, this->processDataNum);
            Cast(p2, valueLocal, RoundMode::CAST_NONE, this->processDataNum);
            Mul(p1, p1, p2, this->processDataNum);
            Cast(p2, input_dataLocal, RoundMode::CAST_NONE, this->processDataNum);
            Add(p2, p1, p2, this->processDataNum);
            Cast(yLocal, p2, RoundMode::CAST_RINT, this->processDataNum);
        } else {
            Mul(x1Local, x1Local, x2Local, this->processDataNum);
            Mul(x1Local, x1Local, valueLocal, this->processDataNum);
            Add(yLocal, x1Local, input_dataLocal, this->processDataNum);
        }
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        inQueueVALUE.FreeTensor(valueLocal);
        inQueueINPUT_DATA.FreeTensor(input_dataLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2, inQueueVALUE, inQueueINPUT_DATA;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<TYPE_X> input_dataGm;
    GlobalTensor<TYPE_X> x1Gm;
    GlobalTensor<TYPE_X> x2Gm;
    GlobalTensor<TYPE_X> valueGm;
    GlobalTensor<TYPE_X> yGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};

} // namespace NsAddcmul
#endif // ADDCMUL_H
