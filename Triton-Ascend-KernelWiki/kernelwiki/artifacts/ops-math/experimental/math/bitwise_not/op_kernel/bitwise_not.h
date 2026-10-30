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
 * \file bitwise_not.h
 * \brief
 */
#ifndef __BITWISE_NOT_H__
#define __BITWISE_NOT_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bitwise_not_tiling_data.h"
#include "bitwise_not_tiling_key.h"

namespace NsBitwiseNot {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// 用 const（非 constexpr）：CPU 编译环境（tikicpulib）下 half(const float&) 构造非 constexpr，
// constexpr half 会阻塞 CPU 侧编译；const half 运行期初始化，数值/语义不变，设备构建同样有效。
const half BN_ONE = 1.0f;
const half BN_NEGATIVE_ONE = -1.0f;
constexpr uint32_t ALIGN_NUM_I16 = 16; // 16 个 int16 = 32B

template <typename TYPE_X, bool IsExistBigCore>
class BitwiseNot {
public:
    __aicore__ inline BitwiseNot() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t bigCoreLoopNum, uint32_t smallCoreLoopNum, uint32_t ubPartDataNum,
                                uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum, uint32_t tailBlockNum,
                                uint32_t isBool, uint32_t lastCoreId, uint32_t lastCoreTailDataNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        this->isBool = isBool;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;
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
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
            }
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * blockIdx;
        }

        // GM 序最后一个核（= lastCoreId，必为最后一个 small core）的最后一个 tile 用真实剩余元素数
        // lastCoreTailDataNum（= smallCoreTailDataNum - pad，可非 32B 对齐）写回 GM，
        // 不再写出末尾 pad 个对齐填充元素，避免末核越界写；其余核尾 tile 仍用对齐的 small/bigCoreTailDataNum。
        this->isLastCore = (blockIdx == lastCoreId);
        this->lastTailDataNum = this->isLastCore ? lastCoreTailDataNum : this->tailDataNum;

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        // half 临时 buffer：仅 BOOL 逻辑非分支使用；整型分支不读写（预算已在 host tiling 计入）。
        pipe.InitBuffer(tmpBuffer, this->ubPartDataNum * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        if (loopCount <= 0) {
            return; // 空 Tensor 安全空跑（host tiling 对 0 元素已置 loop=0）
        }
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i, this->ubPartDataNum);
            Compute(this->ubPartDataNum);
            CopyOut(i, this->ubPartDataNum);
        }
        // 末核最后一个 tile 用真实剩余元素数 lastTailDataNum（= smallCoreTailDataNum - pad，可非 32B 对齐），
        // 其余核用对齐的 tailDataNum；DataCopyPad 按 dataNum*sizeof(TYPE_X)（Byte）安全处理非对齐尾块。
        CopyIn(loopCount - 1, this->lastTailDataNum);
        Compute(this->lastTailDataNum);
        CopyOut(loopCount - 1, this->lastTailDataNum);
    }

private:
    __aicore__ inline uint32_t AlignUpI16(uint32_t n)
    {
        return (n + ALIGN_NUM_I16 - 1) / ALIGN_NUM_I16 * ALIGN_NUM_I16;
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        // DataCopyPad 安全处理对齐 / 非对齐尾块：blockLen 单位 Byte = dataNum * sizeof(TYPE_X)。
        AscendC::DataCopyExtParams copyParams{1, dataNum * (uint32_t)sizeof(TYPE_X), 0, 0, 0};
        AscendC::DataCopyPadExtParams<TYPE_X> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->ubPartDataNum], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t dataNum)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();

        if constexpr (std::is_same_v<TYPE_X, int16_t> || std::is_same_v<TYPE_X, uint16_t>) {
            // INT16 / UINT16：A2 原生支持，直接 Not
            AscendC::Not(yLocal, xLocal, (int32_t)AlignUpI16(dataNum));
        } else if constexpr (std::is_same_v<TYPE_X, int32_t>) {
            // INT32：每元素 4 字节 = 2 个 int16
            auto xI16 = xLocal.template ReinterpretCast<int16_t>();
            auto yI16 = yLocal.template ReinterpretCast<int16_t>();
            AscendC::Not(yI16, xI16, (int32_t)AlignUpI16(dataNum * 2));
        } else if constexpr (std::is_same_v<TYPE_X, int64_t>) {
            // INT64：每元素 8 字节 = 4 个 int16（无需标量兜底）
            auto xI16 = xLocal.template ReinterpretCast<int16_t>();
            auto yI16 = yLocal.template ReinterpretCast<int16_t>();
            AscendC::Not(yI16, xI16, (int32_t)AlignUpI16(dataNum * 4));
        } else if constexpr (std::is_same_v<TYPE_X, uint8_t> || std::is_same_v<TYPE_X, int8_t>) {
            // INT8 / UINT8 按位取反：每元素 1 字节，按 (n+1)/2 个 int16 视图翻转；
            // 奇数尾多翻字节落 UB 内安全，CopyOut 按原 dtype 元素数写回不越界。
            // BOOL 与 INT8 是不同的 DTYPE_X（bool vs int8_t），编译期 if constexpr 已区分语义，
            // 此分支只处理纯位运算的 INT8/UINT8（isBool 恒为 0，仅作可读性保留）。
            auto xI16 = xLocal.template ReinterpretCast<int16_t>();
            auto yI16 = yLocal.template ReinterpretCast<int16_t>();
            AscendC::Not(yI16, xI16, (int32_t)AlignUpI16((dataNum + 1) / 2));
        } else { // bool：逻辑非（0<->1），非裸位翻转
            // Cast 无 bool 直变体，bool 与 int8 位等价，先重解释为 int8 再走 logical_not 实测链路。
            auto xI8 = xLocal.template ReinterpretCast<int8_t>();
            auto yI8 = yLocal.template ReinterpretCast<int8_t>();
            AscendC::LocalTensor<half> tmpLocal = tmpBuffer.Get<half>();
            AscendC::Cast(tmpLocal, xI8, AscendC::RoundMode::CAST_NONE, dataNum);
            AscendC::Abs(tmpLocal, tmpLocal, dataNum);
            AscendC::Mins(tmpLocal, tmpLocal, BN_ONE, dataNum);
            AscendC::Adds(tmpLocal, tmpLocal, BN_NEGATIVE_ONE, dataNum);
            AscendC::Abs(tmpLocal, tmpLocal, dataNum);
            AscendC::Cast(yI8, tmpLocal, AscendC::RoundMode::CAST_NONE, dataNum);
        }

        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t dataNum)
    {
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
        // CopyOut 严格按原 dtype 元素数（dataNum）写回 GM，避免 int16 视图多翻字节越界。
        AscendC::DataCopyExtParams copyParams{1, dataNum * (uint32_t)sizeof(TYPE_X), 0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->ubPartDataNum], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> yGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t ubPartDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t isBool = 0;
    uint32_t lastTailDataNum = 0; // 末核最后一个 tile 的真实剩余元素数
    bool isLastCore = false;      // 当前核是否为 GM 序最后一个核
};
} // namespace NsBitwiseNot
#endif
