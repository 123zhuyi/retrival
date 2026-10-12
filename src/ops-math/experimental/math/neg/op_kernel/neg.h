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
 * \file neg.h
 * \brief
 */
#ifndef __NEG_H__
#define __NEG_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "neg_tiling_data.h"
#include "neg_tiling_key.h"

namespace NsNeg {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t INT64_COMPARE_ALIGN_NUM = 32;
constexpr uint32_t INT64_WORD_NUM = 2;
constexpr uint32_t INT64_HALF_WORD_NUM = 4;

template <typename T, bool IsExistBigCore>
class KernelNeg {
public:
    __aicore__ inline KernelNeg() {}
    __aicore__ inline void Init(
        GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum,
        uint32_t smallCoreLoopNum, uint32_t ubPartDataNum, uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum,
        uint32_t tailBlockNum, TPipe* pipeIn)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (IsExistBigCore) {
            if (coreNum < tailBlockNum) {
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
        src_global.SetGlobalBuffer((__gm__ T*)src_gm + globalBufferIndex, this->coreDataNum);
        dst_global.SetGlobalBuffer((__gm__ T*)dst_gm + globalBufferIndex, this->coreDataNum);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        if constexpr (std::is_same_v<T, int8_t>) {
            pipe->InitBuffer(QueueTmp, this->ubPartDataNum * sizeof(half));
            pipe->InitBuffer(QueueTmp2, this->ubPartDataNum * sizeof(int16_t));
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            pipe->InitBuffer(QueueTmp, this->ubPartDataNum * sizeof(half));
            pipe->InitBuffer(QueueTmp2, this->ubPartDataNum * sizeof(int16_t));
            pipe->InitBuffer(QueueTmp1, this->ubPartDataNum * sizeof(int16_t));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            pipe->InitBuffer(QueueTmp, this->ubPartDataNum * INT64_WORD_NUM * sizeof(float));
            pipe->InitBuffer(QueueTmp1, this->ubPartDataNum * INT64_WORD_NUM * sizeof(uint32_t));
            pipe->InitBuffer(QueueTmp2, this->ubPartDataNum * INT64_WORD_NUM * sizeof(uint8_t));
        } else if constexpr (!(std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int64_t> ||
                               std::is_same_v<T, float> || std::is_same_v<T, half>)) {
            pipe->InitBuffer(QueueTmp1, this->ubPartDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->tileNum == 0) {
            return;
        }
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
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
    __aicore__ inline void CopyIn(uint32_t process)
    {
        LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        if constexpr (std::is_same_v<T, int64_t>) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint16_t>(this->processDataNum * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(srcLocal, src_global[process * this->ubPartDataNum], copyParams, {false, 0, 0, 0});
        } else {
            DataCopy(srcLocal, src_global[process * this->ubPartDataNum], this->processDataNum);
        }
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(uint32_t process)
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            Duplicate(dstLocal, T(-1), this->processDataNum);
            Mul(dstLocal, srcLocal, dstLocal, this->processDataNum);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            LocalTensor<half> tmp = QueueTmp.Get<half>();
            Cast(tmp, srcLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(tmp, tmp, half(-1), this->processDataNum);
            // 移位操作实现溢出处理
            LocalTensor<int16_t> tmp2 = QueueTmp2.Get<int16_t>();
            Cast(tmp2, tmp, RoundMode::CAST_RINT, this->processDataNum); // float16 -> int16
            // 处理溢出 (模拟 int8 计算的行为)
            ShiftLeft(tmp2, tmp2, int16_t(8), this->processDataNum);
            ShiftRight(tmp2, tmp2, int16_t(8), this->processDataNum);
            // 转回 half
            Cast(tmp, tmp2, RoundMode::CAST_NONE, this->processDataNum);
            // 转回int8
            Cast(dstLocal, tmp, RoundMode::CAST_NONE, this->processDataNum);
            QueueTmp2.FreeTensor(tmp2);
            QueueTmp.FreeTensor(tmp);
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            LocalTensor<half> tmp = QueueTmp.Get<half>();
            Cast(tmp, srcLocal, RoundMode::CAST_NONE, this->processDataNum);
            // Match torch.neg(uint8): keep uint8 dtype and wrap modulo 256.
            Muls(tmp, tmp, half(-1), this->processDataNum);
            Adds(tmp, tmp, half(256), this->processDataNum);
            LocalTensor<int16_t> tmp2 = QueueTmp2.Get<int16_t>();
            Cast(tmp2, tmp, RoundMode::CAST_RINT, this->processDataNum);
            LocalTensor<int16_t> mask = QueueTmp1.Get<int16_t>();
            Duplicate(mask, int16_t(255), this->processDataNum);
            And(tmp2, tmp2, mask, this->processDataNum);
            Cast(tmp, tmp2, RoundMode::CAST_NONE, this->processDataNum);
            Cast(dstLocal, tmp, RoundMode::CAST_NONE, this->processDataNum);
            QueueTmp1.FreeTensor(mask);
            QueueTmp2.FreeTensor(tmp2);
            QueueTmp.FreeTensor(tmp);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            uint32_t calcDataNum = (this->processDataNum + INT64_COMPARE_ALIGN_NUM - 1) / INT64_COMPARE_ALIGN_NUM *
                                   INT64_COMPARE_ALIGN_NUM;
            uint32_t lane32Count = calcDataNum * INT64_WORD_NUM;
            uint32_t repeatTimes = calcDataNum / INT64_COMPARE_ALIGN_NUM;

            auto dst16 = dstLocal.template ReinterpretCast<uint16_t>();
            auto src16 = srcLocal.template ReinterpretCast<uint16_t>();
            auto dst32 = dstLocal.template ReinterpretCast<int32_t>();
            Not(dst16, src16, static_cast<int32_t>(calcDataNum * INT64_HALF_WORD_NUM));

            uint64_t lowWordMask[1] = {0x5555555555555555ULL};
            Adds<int32_t>(
                dst32, dst32, static_cast<int32_t>(1), lowWordMask, static_cast<uint8_t>(repeatTimes), {1, 1, 8, 8});

            LocalTensor<float> tmpFloat = QueueTmp.Get<float>();
            LocalTensor<uint32_t> carry32 = QueueTmp1.Get<uint32_t>();
            LocalTensor<uint8_t> cmpMask = QueueTmp2.Get<uint8_t>();

            Cast<float, int32_t>(tmpFloat, dst32, RoundMode::CAST_NONE, lane32Count);

            CompareScalar<float, uint8_t>(cmpMask, tmpFloat, 0.0f, CMPMODE::EQ, lane32Count);

            Duplicate<float>(tmpFloat, 1.0f, lane32Count);
            Select<float, uint8_t>(tmpFloat, cmpMask, tmpFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, lane32Count);
            Cast<int32_t, float>(
                carry32.template ReinterpretCast<int32_t>(), tmpFloat, RoundMode::CAST_RINT, lane32Count);

            uint64_t highWordMask[1] = {0xAAAAAAAAAAAAAAAAULL};
            Duplicate(carry32, static_cast<uint32_t>(0), highWordMask, static_cast<uint8_t>(repeatTimes), 1, 8);

            Cast<float, int32_t>(
                tmpFloat, carry32.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, lane32Count);
            CompareScalar<float, uint8_t>(cmpMask, tmpFloat, 0.0f, CMPMODE::NE, lane32Count);
            ShiftLeft<uint32_t>(
                cmpMask.template ReinterpretCast<uint32_t>(), cmpMask.template ReinterpretCast<uint32_t>(),
                static_cast<uint32_t>(1), static_cast<int32_t>(lane32Count / 32));
            Duplicate<float>(tmpFloat, 1.0f, lane32Count);
            Select<float, uint8_t>(tmpFloat, cmpMask, tmpFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, lane32Count);
            Cast<int32_t, float>(
                carry32.template ReinterpretCast<int32_t>(), tmpFloat, RoundMode::CAST_RINT, lane32Count);
            Add<int32_t>(dst32, dst32, carry32.template ReinterpretCast<int32_t>(), lane32Count);

            QueueTmp2.FreeTensor(cmpMask);
            QueueTmp1.FreeTensor(carry32);
            QueueTmp.FreeTensor(tmpFloat);
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            Muls(dstLocal, srcLocal, T(-1), this->processDataNum);
        }
        // Muls不支持bfloat16类型
        else {
            LocalTensor<float> tmp1 = QueueTmp1.Get<float>();
            Cast(tmp1, srcLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(tmp1, tmp1, float(-1), this->processDataNum);
            Cast(dstLocal, tmp1, RoundMode::CAST_RINT, this->processDataNum);
            QueueTmp1.FreeTensor(tmp1);
        }
        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(uint32_t process)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        if constexpr (std::is_same_v<T, int64_t>) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint16_t>(this->processDataNum * sizeof(T));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(dst_global[process * this->ubPartDataNum], dstLocal, copyParams);
        } else {
            DataCopy(dst_global[process * this->ubPartDataNum], dstLocal, this->processDataNum);
        }
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src_global;
    GlobalTensor<T> dst_global;
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> QueueTmp, QueueTmp2, QueueTmp1;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

} // namespace NsNeg
#endif // NEG_H
