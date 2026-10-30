/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file population_count.h
 * \brief PopulationCount SWAR kernel implementation (arch35)
 *
 * Template parameters:
 *   - D_T_X:       input dtype (int16_t / uint16_t)
 *   - BUFFER_MODE: 0 = single buffer, 1 = double buffer
 *
 * Algorithm: 16-bit SWAR popcount (4 steps + high-byte mask):
 *   x = x - ((x >> 1) & 0x5555);
 *   x = (x & 0x3333) + ((x >> 2) & 0x3333);
 *   x = (x + (x >> 4)) & 0x0F0F;
 *   y = x + (x >> 8);            // low byte holds popcount [0..16]
 *   y = y & 0x00FF;              // CRITICAL: clear high byte before Cast
 *   out_u8 = Cast<uint8_t>(y, CAST_NONE);
 *
 * For int16 input: ReinterpretCast to uint16 first so all right-shifts are logical.
 *
 * IMPORTANT (Cast saturation, verified by probe_uint16_small / probe_int16_dbuf /
 *            probe_unaligned_tail on NPU 2026-04-20):
 *   arch35 `Cast<uint8_t, uint16_t, CAST_NONE>` performs **numeric saturation**
 *   (>255 -> 255), NOT low-byte truncation. After SWAR step 4, `u`'s high byte
 *   still carries residual data (the "high-byte popcount" that was never
 *   cleared), so values > 255 are common. Without masking with 0x00FF, every
 *   non-zero popcount element ends up as 0xFF. Hence we MUST insert
 *   `Ands(u, u, 0x00FF)` between step 4 and Cast.
 */

#ifndef POPULATION_COUNT_H
#define POPULATION_COUNT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "population_count_tiling_data.h"
#include "population_count_tiling_key.h"

namespace NsPopulationCount {

using namespace AscendC;

template <typename D_T_X, uint32_t BUFFER_MODE>
class PopulationCount {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline PopulationCount() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PopulationCountTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf;

    GlobalTensor<D_T_X>   inputGMX;
    GlobalTensor<uint8_t> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_    = 0;
};

template <typename D_T_X, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<D_T_X, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const PopulationCountTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ D_T_X*)x + tilingData->blockFactor * blockIdx, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ uint8_t*)y + tilingData->blockFactor * blockIdx, blockLength_);

    pipe.InitBuffer(inputQueueX,  BUFFER_NUM, ubLength_ * sizeof(D_T_X));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(uint8_t));
    // tmpBuf holds intermediate SWAR results as uint16
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(uint16_t));
}

template <typename D_T_X, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<D_T_X, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<D_T_X> xLocal = inputQueueX.template AllocTensor<D_T_X>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(D_T_X));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename D_T_X, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<D_T_X, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<uint8_t> yLocal = outputQueueY.template DeQue<uint8_t>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(uint8_t));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename D_T_X, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<D_T_X, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<D_T_X>   xLocal = inputQueueX.template DeQue<D_T_X>();
    AscendC::LocalTensor<uint8_t> yLocal = outputQueueY.template AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint16_t> tmp   = tmpBuf.Get<uint16_t>();

    // Reinterpret int16 -> uint16 to force logical right shift
    AscendC::LocalTensor<uint16_t> u;
    if constexpr (std::is_same_v<D_T_X, int16_t>) {
        u = xLocal.template ReinterpretCast<uint16_t>();
    } else {
        u = xLocal;
    }

    const int32_t count = static_cast<int32_t>(currentNum);

    // SWAR step 1: u = u - ((u >> 1) & 0x5555)
    AscendC::ShiftRight(tmp, u, static_cast<uint16_t>(1), count);
    AscendC::Ands(tmp, tmp, static_cast<uint16_t>(0x5555), count);
    AscendC::Sub(u, u, tmp, count);

    // SWAR step 2: u = (u & 0x3333) + ((u >> 2) & 0x3333)
    AscendC::ShiftRight(tmp, u, static_cast<uint16_t>(2), count);
    AscendC::Ands(tmp, tmp, static_cast<uint16_t>(0x3333), count);
    AscendC::Ands(u,   u,   static_cast<uint16_t>(0x3333), count);
    AscendC::Add(u, u, tmp, count);

    // SWAR step 3: u = (u + (u >> 4)) & 0x0F0F
    AscendC::ShiftRight(tmp, u, static_cast<uint16_t>(4), count);
    AscendC::Add(u, u, tmp, count);
    AscendC::Ands(u, u, static_cast<uint16_t>(0x0F0F), count);

    // SWAR step 4: u = u + (u >> 8)  (low byte holds popcount [0..16])
    AscendC::ShiftRight(tmp, u, static_cast<uint16_t>(8), count);
    AscendC::Add(u, u, tmp, count);

    // CRITICAL: mask high byte BEFORE Cast.
    //   arch35 `Cast<uint8_t, uint16_t, CAST_NONE>` is numeric saturation, not
    //   low-byte truncation. After step 4, the high byte of `u` still holds
    //   residual data from SWAR, making `u` > 255 for many inputs. Without
    //   this mask, every non-zero popcount element saturates to 0xFF.
    //   Verified on NPU by probe_uint16_small / probe_int16_dbuf /
    //   probe_unaligned_tail (2026-04-20).
    AscendC::Ands(u, u, static_cast<uint16_t>(0x00FF), count);

    // Cast uint16 -> uint8 (CAST_NONE). After the 0x00FF mask above, u in [0,16]
    // so the Cast is exact and saturation-safe.
    AscendC::Cast(yLocal, u, AscendC::RoundMode::CAST_NONE, count);

    outputQueueY.template EnQue<uint8_t>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename D_T_X, uint32_t BUFFER_MODE>
__aicore__ inline void PopulationCount<D_T_X, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
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

} // namespace NsPopulationCount
#endif // POPULATION_COUNT_H
