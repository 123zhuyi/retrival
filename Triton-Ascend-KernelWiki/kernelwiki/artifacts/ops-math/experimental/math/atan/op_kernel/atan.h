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
 * \file atan.h
 * \brief
 */
#ifndef ATAN_H
#define ATAN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atan_tiling_data.h"
#include "atan_tiling_key.h"

namespace NsAtan {

using namespace AscendC;

#define TAN_PI_BY_EIGHT 0.4142135623730950f
#define NEG_TAN_PI_BY_EIGHT -0.4142135623730950f
#define CONST_PI_BY_FOUR 0.78539816339744830961566084581988f
#define CONST_PI_BY_EIGHT 0.39269908169872415480783042290994f

constexpr int32_t BUFFER_NUM = 2;
constexpr float CONST_POS_ONE = 1.0f;
constexpr float TAYLOR[] = {1.0f, -1.0f / 3, 1.0f / 5, -1.0f / 7, 1.0f / 9, -1.0f / 11, 1.0f / 13};

template <typename T>
class Atan {
public:
    __aicore__ inline Atan(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AtanTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void DoTaylor(
        LocalTensor<float>& p1, LocalTensor<float>& p2, LocalTensor<float>& p3, LocalTensor<float>& p4);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    TBuf<QuePosition::VECCALC> compare_bits;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4, tmp_xLocal, tmp_yLocal, tmp_sign;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int32_t coreDataNum = 0;
    int32_t tileNum = 0;
    int32_t tileDataNum = 0;
    int32_t tailDataNum = 0;
    int32_t processDataNum = 0;
};

template <typename T>
__aicore__ inline void Atan<T>::Init(GM_ADDR x, GM_ADDR y, const AtanTilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;

    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (GetBlockIdx() - tilingData->tailBlockNum);
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(compare_bits, this->tileDataNum * sizeof(uint8_t));
    pipe.InitBuffer(tmp_sign, this->tileDataNum * sizeof(float));
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        pipe.InitBuffer(tmp_xLocal, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp_yLocal, this->tileDataNum * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void Atan<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Atan<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->tileDataNum], yLocal, this->processDataNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Atan<T>::DoTaylor(
    LocalTensor<float>& p1, LocalTensor<float>& p2, LocalTensor<float>& p3, LocalTensor<float>& p4)
{
    Muls(p3, p1, TAN_PI_BY_EIGHT, this->processDataNum);
    Adds(p3, p3, CONST_POS_ONE, this->processDataNum);
    Adds(p2, p1, NEG_TAN_PI_BY_EIGHT, this->processDataNum);
    Div(p3, p2, p3, this->processDataNum);
    Abs(p4, p3, this->processDataNum);

    Mul(p2, p4, p4, this->processDataNum);
    Duplicate(p3, static_cast<float>(TAYLOR[6]), this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[5], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[4], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[3], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[2], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[1], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[0], this->processDataNum);
    Mul(p3, p3, p4, this->processDataNum);
    Adds(p4, p3, CONST_PI_BY_EIGHT, this->processDataNum);

    Mul(p2, p1, p1, this->processDataNum);
    Duplicate(p3, static_cast<float>(TAYLOR[4]), this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[3], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[2], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[1], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[0], this->processDataNum);
    Mul(p3, p3, p1, this->processDataNum);

    Min(p1, p4, p3, this->processDataNum);
}

template <typename T>
__aicore__ inline void Atan<T>::Compute(int32_t progress)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    auto p1 = tmp1.Get<float>();
    auto p2 = tmp2.Get<float>();
    auto p3 = tmp3.Get<float>();
    auto p4 = tmp4.Get<float>();
    auto bits = compare_bits.Get<uint8_t>();
    auto sign = tmp_sign.Get<float>();

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        auto xLocal_f = tmp_xLocal.Get<float>();
        auto yLocal_f = tmp_yLocal.Get<float>();
        Cast(xLocal_f, xLocal, RoundMode::CAST_NONE, this->processDataNum);

        // when x's value is too large the first caculator of _do_taylor will be overflow. when epsilon is 0.0001,
        // the approximate value of `tan(pi/2 - 0.0001)` is 10000
        float max_input_value = 10000.0f;
        float min_input_value = -max_input_value;
        Mins(p1, xLocal_f, max_input_value, this->processDataNum);
        Maxs(xLocal_f, p1, min_input_value, this->processDataNum);

        Abs(p1, xLocal_f, this->processDataNum);       // p1 = abs_data
        Div(sign, xLocal_f, p1, this->processDataNum); // sign = xLocal_f / abs_data

        Duplicate(p2, CONST_POS_ONE, this->processDataNum);

        Sub(p3, p1, p2, this->processDataNum);
        Add(p4, p1, p2, this->processDataNum);
        Div(xLocal_f, p3, p4, this->processDataNum);
        Abs(xLocal_f, xLocal_f, this->processDataNum); // xLocal_f = abs_data2

        DoTaylor(p1, p2, p3, p4);
        DoTaylor(xLocal_f, p2, p3, p4);

        Adds(xLocal_f, xLocal_f, CONST_PI_BY_FOUR, this->processDataNum);
        Min(p1, p1, xLocal_f, this->processDataNum);
        Mul(yLocal_f, p1, sign, this->processDataNum);
        Cast(yLocal, yLocal_f, RoundMode::CAST_NONE, this->processDataNum);
    } else if constexpr (std::is_same_v<T, float>) {
        // when x's value is too large the first caculator of _do_taylor will be overflow. when epsilon is 0.0001,
        // the approximate value of `tan(pi/2 - 0.0001)` is 10000
        float max_input_value = 10000.0f;
        float min_input_value = -max_input_value;
        Mins(p1, xLocal, max_input_value, this->processDataNum);
        Maxs(xLocal, p1, min_input_value, this->processDataNum);

        Abs(p1, xLocal, this->processDataNum);       // p1 = abs_data
        Div(sign, xLocal, p1, this->processDataNum); // sign = xLocal / abs_data

        Duplicate(p2, CONST_POS_ONE, this->processDataNum);

        Sub(p3, p1, p2, this->processDataNum);
        Add(p4, p1, p2, this->processDataNum);
        Div(xLocal, p3, p4, this->processDataNum);
        Abs(xLocal, xLocal, this->processDataNum); // xLocal = abs_data2

        DoTaylor(p1, p2, p3, p4);
        DoTaylor(xLocal, p2, p3, p4);

        Adds(xLocal, xLocal, CONST_PI_BY_FOUR, this->processDataNum);
        Min(p1, p1, xLocal, this->processDataNum);
        Mul(yLocal, p1, sign, this->processDataNum);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Atan<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == loopCount - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsAtan
#endif // ATAN_H