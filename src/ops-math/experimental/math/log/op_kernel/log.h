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
 * \file log.h
 * \brief
 */
#ifndef __LOG_H__
#define __LOG_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log_tiling_data.h"
#include "log_tiling_key.h"

namespace NsLog {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
class LogKernel {
public:
    __aicore__ inline LogKernel(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum,
        uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum,
        uint64_t tailBlockNum, float base, float scale, float shift);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf1;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t ubPartDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    float base = 0.0;
    float shift = 0.0;
    float scale = 0.0;
};

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void LogKernel<TYPE_X, TYPE_Y, IsExistBigCore>::Init(
    GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum,
    uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum,
    uint64_t tailBlockNum, float base, float scale, float shift)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t blockIdx = GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
    this->ubPartDataNum = ubPartDataNum;

    if constexpr (IsExistBigCore) {
        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = bigCoreLoopNum;
            this->tailDataNum = bigCoreTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = smallCoreLoopNum;
        this->tailDataNum = smallCoreTailDataNum;
        globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
    }

    // 设置log算子特有属性
    this->base = base;
    this->shift = shift;
    this->scale = scale;

    // 初始化全局内存
    xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + globalBufferIndex, this->coreDataNum);

    // 初始化管道和缓冲区
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(DTYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(DTYPE_Y));

    // 为浮点转换准备缓冲区
    if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
        pipe.InitBuffer(calcBuf1, this->ubPartDataNum * sizeof(float));
    }
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void LogKernel<TYPE_X, TYPE_Y, IsExistBigCore>::CopyIn(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void LogKernel<TYPE_X, TYPE_Y, IsExistBigCore>::CopyOut(int32_t progress)
{
    LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void LogKernel<TYPE_X, TYPE_Y, IsExistBigCore>::Compute(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

    if constexpr (std::is_same_v<TYPE_X, float> || std::is_same_v<TYPE_X, float16_t>) {
        // 浮点类型直接计算
        Muls(yLocal, xLocal, (TYPE_X)this->scale, this->processDataNum);
        Adds(yLocal, yLocal, (TYPE_X)this->shift, this->processDataNum);
        Log(yLocal, yLocal, this->processDataNum);
        Muls(yLocal, yLocal, (TYPE_X)this->base, this->processDataNum);
    } else {
        // 非浮点类型先转换为float计算
        LocalTensor<float> xLocalFp32 = calcBuf1.Get<float>();

        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Muls(xLocalFp32, xLocalFp32, this->scale, this->processDataNum);
        Adds(xLocalFp32, xLocalFp32, this->shift, this->processDataNum);
        Log(xLocalFp32, xLocalFp32, this->processDataNum);
        Muls(xLocalFp32, xLocalFp32, this->base, this->processDataNum);
        Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, this->processDataNum);
    }

    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void LogKernel<TYPE_X, TYPE_Y, IsExistBigCore>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->ubPartDataNum;
    // 处理完整数据块
    for (int32_t i = 0; i < loopCount - 1; i++) {
        // 流水线处理
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsLog
#endif // LOG_H