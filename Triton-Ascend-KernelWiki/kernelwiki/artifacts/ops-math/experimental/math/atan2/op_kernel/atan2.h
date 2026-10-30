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
 * \file atan2.h
 * \brief
 */
#ifndef ATAN2_H
#define ATAN2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atan2_tiling_data.h"
#include "atan2_tiling_key.h"

namespace NsAtan2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr float CONST_PI = 3.14159265358979323846f;
constexpr float CONST_PI_BY_TWO = 1.57079632679489661923f;
constexpr float EPSILON = 1e-37f;

template <typename T>
class Atan2 {
public:
    __aicore__ inline Atan2(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Atan2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX1; // y arg of atan2(y,x)
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX2; // x arg
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // float32 scratch buffers
    TBuf<QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2, tmpBuf3, tmpBuf4;
    // uint8_t temp space for AscendC::Atan internal use
    TBuf<QuePosition::VECCALC> atanTmpBuf;

    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
    uint32_t atanTmpSize = 0; // byte size of atanTmpBuf per tile, from tiling
};

// ── Init ───────────────────────────────────────────────────────────────────

template <typename T>
__aicore__ inline void Atan2<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Atan2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * coreIdx;

    this->tileDataNum = tilingData->tileDataNum;
    this->atanTmpSize = tilingData->atanTmpSize;

    if (coreIdx < (uint32_t)tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (coreIdx - (uint32_t)tilingData->tailBlockNum);
    }

    inputGMX1.SetGlobalBuffer((__gm__ T*)x1 + globalBufferIndex, this->coreDataNum);
    inputGMX2.SetGlobalBuffer((__gm__ T*)x2 + globalBufferIndex, this->coreDataNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf3, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf4, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(atanTmpBuf, this->atanTmpSize);
}

// ── CopyIn / CopyOut ───────────────────────────────────────────────────────

template <typename T>
__aicore__ inline void Atan2<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> x1Local = inputQueueX1.AllocTensor<T>();
    LocalTensor<T> x2Local = inputQueueX2.AllocTensor<T>();
    AscendC::DataCopy(x1Local, inputGMX1[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(x2Local, inputGMX2[progress * this->tileDataNum], this->processDataNum);
    inputQueueX1.EnQue(x1Local);
    inputQueueX2.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void Atan2<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress * this->tileDataNum], yLocal, this->processDataNum);
    outputQueueY.FreeTensor(yLocal);
}

// ── Compute ────────────────────────────────────────────────────────────────
//
// Scratch assignment during Compute (all float32 except atanTmp):
//   tmp0: step1→yF, step6→sign_x_soft→unsigned_out
//   tmp1: step1→xF, step5→blend_xne0
//   tmp2: step2→ind_xlt0, step6→blend*(base_atan−π/2)
//   tmp3: step2→scratch for ind denominator, step3→sign_y
//   tmp4: step4→base_atan = Atan(|yF/xF|)  [dst≠src per Atan constraint]
//   atanTmpBuf: uint8_t scratch for Atan

template <typename T>
__aicore__ inline void Atan2<T>::Compute(int32_t progress)
{
    LocalTensor<T> yLocal = inputQueueX1.DeQue<T>();
    LocalTensor<T> xLocal = inputQueueX2.DeQue<T>();
    LocalTensor<T> outLocal = outputQueueY.AllocTensor<T>();

    LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
    LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2.Get<float>();
    LocalTensor<float> tmp3 = tmpBuf3.Get<float>();
    LocalTensor<float> tmp4 = tmpBuf4.Get<float>();
    LocalTensor<uint8_t> atanTmp = atanTmpBuf.Get<uint8_t>();

    uint32_t n = this->processDataNum;

    // ── Step 1: cast inputs to float32 → tmp0(yF), tmp1(xF) ──────────────
    if constexpr (AscendC::Std::is_same<T, half>::value || AscendC::Std::is_same<T, bfloat16_t>::value) {
        AscendC::Cast(tmp0, yLocal, RoundMode::CAST_NONE, n);
        AscendC::Cast(tmp1, xLocal, RoundMode::CAST_NONE, n);
    } else {
        // T == float: copy into float scratch
        AscendC::Adds(tmp0, yLocal.template ReinterpretCast<float>(), 0.0f, n);
        AscendC::Adds(tmp1, xLocal.template ReinterpretCast<float>(), 0.0f, n);
    }
    // tmp0 = yF,  tmp1 = xF

    // ── Step 2: ind_xlt0 = (x<0)?1:0 → tmp2 ─────────────────────────────
    // neg_x = min(xF, 0)  → 0 when x≥0, negative when x<0
    // tmp2 = |neg_x| / (|neg_x| + ε)  → 1 when x<0, 0 when x≥0
    AscendC::Mins(tmp2, tmp1, 0.0f, n);    // tmp2 = min(xF, 0)
    AscendC::Abs(tmp2, tmp2, n);           // tmp2 = |min(xF,0)|
    AscendC::Adds(tmp3, tmp2, EPSILON, n); // tmp3 = |min(xF,0)| + ε (borrow tmp3 momentarily)
    AscendC::Div(tmp2, tmp2, tmp3, n);     // tmp2 = ind_xlt0 ∈ [0,1]

    // ── Step 3: sign_y = yF/(|yF|+ε) → tmp3 ─────────────────────────────
    AscendC::Abs(tmp3, tmp0, n);           // tmp3 = |yF|
    AscendC::Adds(tmp3, tmp3, EPSILON, n); // tmp3 = |yF| + ε
    AscendC::Div(tmp3, tmp0, tmp3, n);     // tmp3 = sign_y ∈ (-1,+1]

    // ── Step 4: base_atan = Atan(|yF/xF|) → tmp4 ─────────────────────────
    // Use tmp0 as scratch for |yF/xF| (yF is no longer needed after step 3)
    AscendC::Div(tmp0, tmp0, tmp1, n); // tmp0 = yF/xF  (±Inf when xF=0; handled by blend)
    AscendC::Abs(tmp0, tmp0, n);       // tmp0 = |yF/xF| ≥ 0
    // Atan: dst(tmp4) ≠ src(tmp0), sharedTmpBuffer = atanTmp
    AscendC::Atan(tmp4, tmp0, atanTmp, n); // tmp4 = Atan(|yF/xF|) ∈ [0, π/2]

    // ── Step 5: blend_xne0 = |xF|/(|xF|+ε) → tmp1 ───────────────────────
    // xF is in tmp1; compute |xF| in-place
    AscendC::Abs(tmp1, tmp1, n);           // tmp1 = |xF|
    AscendC::Adds(tmp0, tmp1, EPSILON, n); // tmp0 = |xF| + ε
    AscendC::Div(tmp1, tmp1, tmp0, n);     // tmp1 = blend_xne0

    // ── Step 6: unsigned_out accumulation → tmp0 ──────────────────────────
    // Correct formula: unsigned_out = sign_x_soft * blend*(base_atan − π/2) + π/2
    //   where sign_x_soft = 1 − 2*ind_xlt0  (+1 when x≥0, −1 when x<0)
    //
    // Derivation check:
    //   x>0 (ind=0): blend*(base−π/2)+π/2   → base when blend=1  ∈[0,π/2] ✓
    //   x<0 (ind=1): −blend*(base−π/2)+π/2  → π−base when blend=1         ✓
    //   x=0 (blend=0): π/2                                                  ✓
    //
    // tmp1=blend_xne0, tmp2=ind_xlt0, tmp4=base_atan
    AscendC::Muls(tmp0, tmp2, -2.0f, n);            // tmp0 = −2*ind_xlt0
    AscendC::Adds(tmp0, tmp0, 1.0f, n);             // tmp0 = sign_x_soft = 1−2*ind_xlt0
    AscendC::Adds(tmp2, tmp4, -CONST_PI_BY_TWO, n); // tmp2 = base_atan − π/2
    AscendC::Mul(tmp2, tmp1, tmp2, n);              // tmp2 = blend*(base_atan−π/2)
    AscendC::Mul(tmp0, tmp0, tmp2, n);              // tmp0 = sign_x_soft*blend*(base_atan−π/2)
    AscendC::Adds(tmp0, tmp0, CONST_PI_BY_TWO, n);  // tmp0 = unsigned_out

    // ── Step 7: out = sign_y * unsigned_out → tmp0 ────────────────────────
    AscendC::Mul(tmp0, tmp3, tmp0, n);

    // ── Step 8: cast float32 result back to T → outLocal ─────────────────
    if constexpr (AscendC::Std::is_same<T, half>::value || AscendC::Std::is_same<T, bfloat16_t>::value) {
        AscendC::Cast(outLocal, tmp0, RoundMode::CAST_ROUND, n);
    } else {
        AscendC::Adds(outLocal.template ReinterpretCast<float>(), tmp0, 0.0f, n);
    }

    outputQueueY.EnQue<T>(outLocal);
    inputQueueX1.FreeTensor(yLocal);
    inputQueueX2.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Atan2<T>::Process()
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

} // namespace NsAtan2
#endif // ATAN2_H
