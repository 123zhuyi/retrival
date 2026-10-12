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
 * \file inv_grad.h
 * \brief InvGrad kernel class definition (arch35)
 *
 * Formula: dx = -dy * y * y
 *
 * Precision strategy:
 *   - float32: Mul(yy, y, y) -> Mul(dx, dy, yy) -> Muls(dx, dx, -1.0f)
 *   - float16/bfloat16: Cast(y, dy -> fp32) -> Mul -> Mul -> Muls(-1)
 *                        -> Cast(fp32 -> T)
 *
 * Template parameter T (mapped by TilingKey):
 *   - float:      TilingKey 0
 *   - half:       TilingKey 1
 *   - bfloat16_t: TilingKey 2
 *
 * Buffer layout (single buffer, depth=1):
 *   yQueue:   ubFactor * sizeof(T)     (input y)
 *   dyQueue:  ubFactor * sizeof(T)     (input dy)
 *   outQueue: ubFactor * sizeof(T)     (output dx)
 *   tmpBuf1:  ubFactor * sizeof(float) (fp32 intermediate yy / yFloat)
 *   tmpBuf2:  ubFactor * sizeof(float) (fp32 intermediate dyFloat)
 */

#ifndef EXPERIMENTAL_MATH_INV_GRAD_OP_KERNEL_INV_GRAD_H
#define EXPERIMENTAL_MATH_INV_GRAD_OP_KERNEL_INV_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inv_grad_tiling_data.h"
#include "inv_grad_tiling_key.h"

namespace NsInvGrad {

using AscendC::Cast;
using AscendC::DataCopyPad;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename T>
class InvGrad {
public:
    __aicore__ inline InvGrad() {}

    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, const InvGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);

    // float32 direct path: Mul(yy, y, y) -> Mul(dx, dy, yy) -> Muls(-1)
    __aicore__ inline void ComputeFloat32(
        LocalTensor<float>& yLocal, LocalTensor<float>& dyLocal, LocalTensor<float>& dxLocal, int64_t alignedNum);
    // Non-float32 path: Cast(T->f32) -> compute -> Cast(f32->T)
    template <typename SrcT>
    __aicore__ inline void ComputeWithCast(
        LocalTensor<SrcT>& yLocal, LocalTensor<SrcT>& dyLocal, LocalTensor<SrcT>& dxLocal, int64_t alignedNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> yQueue_;    // input y
    TQue<QuePosition::VECIN, 1> dyQueue_;   // input dy
    TQue<QuePosition::VECOUT, 1> outQueue_; // output dx
    TBuf<QuePosition::VECCALC> tmpBuf1_;    // fp32 scratch 1 (yy / yFloat)
    TBuf<QuePosition::VECCALC> tmpBuf2_;    // fp32 scratch 2 (dyFloat)

    GlobalTensor<T> yGM_;
    GlobalTensor<T> dyGM_;
    GlobalTensor<T> dxGM_;

    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubFactor_ = 0;
};

// =============================================================================
// Init
// =============================================================================
template <typename T>
__aicore__ inline void InvGrad<T>::Init(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, const InvGradTilingData* tilingData)
{
    ubFactor_ = tilingData->ubFactor;

    if (tilingData->totalElements == 0 || tilingData->blockFactor == 0) {
        blockOffset_ = 0;
        blockLen_ = 0;
        return;
    }

    blockOffset_ = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    int64_t remaining = tilingData->totalElements - blockOffset_;
    if (remaining <= 0) {
        blockLen_ = 0;
        return;
    }
    blockLen_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;

    yGM_.SetGlobalBuffer((__gm__ T*)y + blockOffset_, blockLen_);
    dyGM_.SetGlobalBuffer((__gm__ T*)dy + blockOffset_, blockLen_);
    dxGM_.SetGlobalBuffer((__gm__ T*)dx + blockOffset_, blockLen_);

    pipe.InitBuffer(yQueue_, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(dyQueue_, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outQueue_, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1_, ubFactor_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2_, ubFactor_ * sizeof(float));
}

// =============================================================================
// CopyIn: GM -> UB for both y and dy
// =============================================================================
template <typename T>
__aicore__ inline void InvGrad<T>::CopyIn(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = yQueue_.template AllocTensor<T>();
    LocalTensor<T> dyLocal = dyQueue_.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(yLocal, yGM_[gmOffset], copyParams, {false, 0, 0, 0});
    DataCopyPad(dyLocal, dyGM_[gmOffset], copyParams, {false, 0, 0, 0});
    yQueue_.EnQue(yLocal);
    dyQueue_.EnQue(dyLocal);
}

// =============================================================================
// CopyOut: UB -> GM for dx
// =============================================================================
template <typename T>
__aicore__ inline void InvGrad<T>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> dxLocal = outQueue_.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(dxGM_[gmOffset], dxLocal, copyParams);
    outQueue_.FreeTensor(dxLocal);
}

// =============================================================================
// ComputeFloat32: Mul(yy, y, y) -> Mul(dx, dy, yy) -> Muls(dx, dx, -1)
// =============================================================================
template <typename T>
__aicore__ inline void InvGrad<T>::ComputeFloat32(
    LocalTensor<float>& yLocal, LocalTensor<float>& dyLocal, LocalTensor<float>& dxLocal, int64_t alignedNum)
{
    LocalTensor<float> yy = tmpBuf1_.template Get<float>();
    Mul(yy, yLocal, yLocal, static_cast<int32_t>(alignedNum));   // yy = y * y
    Mul(dxLocal, dyLocal, yy, static_cast<int32_t>(alignedNum)); // dx = dy * yy
    Muls(dxLocal, dxLocal, static_cast<float>(-1.0f),
         static_cast<int32_t>(alignedNum)); // dx = -dx
}

// =============================================================================
// ComputeWithCast: Cast(T -> fp32) -> compute in fp32 -> Cast(fp32 -> T)
// =============================================================================
template <typename T>
template <typename SrcT>
__aicore__ inline void InvGrad<T>::ComputeWithCast(
    LocalTensor<SrcT>& yLocal, LocalTensor<SrcT>& dyLocal, LocalTensor<SrcT>& dxLocal, int64_t alignedNum)
{
    LocalTensor<float> yFloat = tmpBuf1_.template Get<float>();
    LocalTensor<float> dyFloat = tmpBuf2_.template Get<float>();

    // Up-cast: half/bf16 -> float32
    Cast(yFloat, yLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedNum));
    Cast(dyFloat, dyLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedNum));

    // fp32 compute:
    //   yFloat = y * y
    //   yFloat = dy * yFloat
    //   yFloat = -yFloat
    Mul(yFloat, yFloat, yFloat, static_cast<int32_t>(alignedNum));
    Mul(yFloat, dyFloat, yFloat, static_cast<int32_t>(alignedNum));
    Muls(yFloat, yFloat, static_cast<float>(-1.0f), static_cast<int32_t>(alignedNum));

    // Down-cast: float32 -> half/bf16
    Cast(dxLocal, yFloat, RoundMode::CAST_ROUND, static_cast<uint32_t>(alignedNum));
}

// =============================================================================
// Compute: dispatch to float32 direct path or cast path via if constexpr
// =============================================================================
template <typename T>
__aicore__ inline void InvGrad<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> yLocal = yQueue_.template DeQue<T>();
    LocalTensor<T> dyLocal = dyQueue_.template DeQue<T>();
    LocalTensor<T> dxLocal = outQueue_.template AllocTensor<T>();

    // Align count to the max of float-block and T-block
    constexpr int64_t floatBlock = 32 / sizeof(float); // 8
    constexpr int64_t typeBlock = 32 / sizeof(T);
    constexpr int64_t alignBlock = (floatBlock > typeBlock) ? floatBlock : typeBlock;
    int64_t alignedNum = ((currentNum + alignBlock - 1) / alignBlock) * alignBlock;

    if constexpr (std::is_same_v<T, float>) {
        ComputeFloat32(yLocal, dyLocal, dxLocal, alignedNum);
    } else {
        ComputeWithCast(yLocal, dyLocal, dxLocal, alignedNum);
    }

    outQueue_.template EnQue<T>(dxLocal);
    yQueue_.FreeTensor(yLocal);
    dyQueue_.FreeTensor(dyLocal);
}

// =============================================================================
// Process: main loop over UB chunks
// =============================================================================
template <typename T>
__aicore__ inline void InvGrad<T>::Process()
{
    if (blockLen_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLen_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t gmOffset = i * ubFactor_;
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLen_ - gmOffset) : ubFactor_;
        CopyIn(gmOffset, currentNum);
        Compute(currentNum);
        CopyOut(gmOffset, currentNum);
    }
}

} // namespace NsInvGrad
#endif // EXPERIMENTAL_MATH_INV_GRAD_OP_KERNEL_INV_GRAD_H
