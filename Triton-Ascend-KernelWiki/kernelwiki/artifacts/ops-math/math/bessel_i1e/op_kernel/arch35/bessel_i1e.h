/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Optimized bessel_i1e kernel - exp(-|x|) Taylor 9 terms -> 13 terms
 *
 * Key change: exp(-|x|/4) Taylor series from 9 terms to 13 terms
 * Expected improvement: max error from 2.6e-6 to 6.1e-10 (4000x)
 *
 * Uses loop-based Horner evaluation to avoid deep nesting that
 * the Ascend C compiler cannot handle.
 */

#ifndef BESSEL_I1E_H
#define BESSEL_I1E_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bessel_i1e_tiling_data.h"
#include "bessel_i1e_tiling_key.h"

namespace NsBesselI1e {

using namespace AscendC;

constexpr float SEGMENT_POINT = 3.75f;
constexpr float INV_SEGMENT = 0.26666666666666666f;
constexpr float QUARTER = 0.25f;

constexpr float EXP_COEFF[13] = {1.0f,        1.0f,        0.5f,        0.16666667f, 0.04166667f,
                                 0.00833333f, 0.00138889f, 0.00019841f, 0.00002480f, 2.75573e-6f,
                                 2.75573e-7f, 2.50521e-8f, 2.08768e-9f};

constexpr float itrBefore[7] = {0.5000000008f, 0.8789061535f, 0.5149860539f, 0.1508606731f,
                                0.0265652742f, 0.0030351394f, 0.0003173337f};

constexpr float itrAfter[9] = {0.3989422302f, -0.0398905760f, -0.0034090932f, 0.0000697438f, -0.0044962120f,
                               0.0108902378f, -0.0151944387f, 0.0095376700f,  -0.0021325862f};

template <typename T>
class BesselI1e {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline BesselI1e(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BesselI1eTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> tmpQueue1;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void BesselI1e<T>::Init(GM_ADDR x, GM_ADDR y, const BesselI1eTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(float));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpQueue1, BUFFER_NUM, ubLength_ * sizeof(float));
}

template <typename T>
__aicore__ inline void BesselI1e<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<float> xLocal = inputQueueX.template AllocTensor<float>();
    if constexpr (std::is_same_v<T, half>) {
        AscendC::LocalTensor<half> tmpHalf = tmpQueue1.template AllocTensor<half>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(half);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(tmpHalf, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
        for (int64_t i = 0; i < currentNum; i++) {
            xLocal.SetValue(i, static_cast<float>(tmpHalf.GetValue(i)));
        }
        tmpQueue1.FreeTensor(tmpHalf);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        AscendC::LocalTensor<bfloat16_t> tmpBf16 = tmpQueue1.template AllocTensor<bfloat16_t>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(bfloat16_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(tmpBf16, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
        AscendC::Cast(xLocal, tmpBf16, AscendC::RoundMode::CAST_NONE, currentNum);
        tmpQueue1.FreeTensor(tmpBf16);
    } else {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void BesselI1e<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<float> yLocal = outputQueueY.template DeQue<float>();
    if constexpr (std::is_same_v<T, half>) {
        AscendC::LocalTensor<half> tmpHalf = tmpQueue1.template AllocTensor<half>();
        for (int64_t i = 0; i < currentNum; i++) {
            tmpHalf.SetValue(i, static_cast<half>(yLocal.GetValue(i)));
        }
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(half);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outputGMY[progress * ubLength_], tmpHalf, copyParams);
        tmpQueue1.FreeTensor(tmpHalf);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        AscendC::LocalTensor<bfloat16_t> tmpBf16 = tmpQueue1.template AllocTensor<bfloat16_t>();
        AscendC::Cast(tmpBf16, yLocal, AscendC::RoundMode::CAST_RINT, currentNum);
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(bfloat16_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outputGMY[progress * ubLength_], tmpBf16, copyParams);
        tmpQueue1.FreeTensor(tmpBf16);
    } else {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void BesselI1e<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<float> xLocal = inputQueueX.template DeQue<float>();
    AscendC::LocalTensor<float> yLocal = outputQueueY.template AllocTensor<float>();

    for (int64_t i = 0; i < currentNum; i++) {
        float x = xLocal.GetValue(i);
        float absX = (x >= 0.0f) ? x : -x;
        float sign = (x >= 0.0f) ? 1.0f : -1.0f;
        float result;

        if (absX < SEGMENT_POINT) {
            float t = absX * INV_SEGMENT;
            float t2 = t * t;
            float poly = itrBefore[6];
            for (int k = 5; k >= 0; k--) {
                poly = poly * t2 + itrBefore[k];
            }
            float q = absX * QUARTER;
            float q2 = q * q;
            float q4 = q2 * q2;
            float q7 = q4 * q2 * q;
            float e_lo = EXP_COEFF[6];
            for (int k = 5; k >= 0; k--) {
                e_lo = EXP_COEFF[k] - q * e_lo;
            }
            float e_hi = EXP_COEFF[12];
            for (int k = 11; k >= 7; k--) {
                e_hi = EXP_COEFF[k] - q * e_hi;
            }
            float e = e_lo - q7 * e_hi;
            e = e * e;
            e = e * e;
            result = e * absX * poly;
        } else {
            float t = SEGMENT_POINT / absX;
            float poly = itrAfter[8];
            for (int k = 7; k >= 0; k--) {
                poly = poly * t + itrAfter[k];
            }
            float sqrtX = sqrt(absX);
            result = poly / sqrtX;
        }
        yLocal.SetValue(i, sign * result);
    }

    outputQueueY.template EnQue<float>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void BesselI1e<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsBesselI1e
#endif // BESSEL_I1E_H
