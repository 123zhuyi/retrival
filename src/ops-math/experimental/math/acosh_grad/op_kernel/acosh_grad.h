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
 * \file acosh_grad.h
 * \brief AcoshGrad AscendC kernel — double-buffer TPipe implementation
 *
 * 算法: z = dy / sinh(y)
 * sinh(y) = repeat(repeat(repeat(taylor_sinh(y/8))))
 * repeat(v) = 2 * v * sqrt(v^2 + 1)
 *
 * CopyIn/CopyOut 使用 DataCopyPad 处理非 32B 对齐尾块。
 * Compute 内部全 fp32 计算，fp16/bf16 通过 Cast 转换。
 */

#ifndef ACOSHGRAD_H
#define ACOSHGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "acosh_grad_tiling_data.h"
#include "acosh_grad_tiling_key.h"

namespace NsAcoshGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t UB_BLOCK_BYTES = 32U;

constexpr float TAYLOR_C3 = 0.16666666666666666f;
constexpr float TAYLOR_C5 = 0.00833333333333333f;
constexpr float TAYLOR_C7 = 0.00019841269841270f;
constexpr float SCALE_1_8 = 0.125f;

template <typename T>
class AcoshGrad {
public:
    __aicore__ inline AcoshGrad() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, const AcoshGradTilingData* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t tileIdx, int64_t count);
    __aicore__ inline void Compute(int64_t count);
    __aicore__ inline void CopyOut(int64_t tileIdx, int64_t count);

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDy_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    TBuf<QuePosition::VECCALC> sinhBufPool_;
    TBuf<QuePosition::VECCALC> dyBufPool_;
    TBuf<QuePosition::VECCALC> tmpBufPool_;

    GlobalTensor<T> yGm_;
    GlobalTensor<T> dyGm_;
    GlobalTensor<T> zGm_;

    int64_t startOffset_ = 0;
    int64_t blockCount_ = 0;
    int64_t tileLength_ = 0;
    int64_t tileNum_ = 0;
    int64_t lastTileLen_ = 0;
};

template <typename T>
__aicore__ inline void AcoshGrad<T>::Init(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, const AcoshGradTilingData* td)
{
    uint32_t blockIdx = GetBlockIdx();
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    dyGm_.SetGlobalBuffer((__gm__ T*)dy);
    zGm_.SetGlobalBuffer((__gm__ T*)dx);

    bool isTailCore = (blockIdx >= td->formerCoreNum);
    if (!isTailCore) {
        startOffset_ = static_cast<int64_t>(blockIdx) * static_cast<int64_t>(td->blockLength);
        blockCount_ = static_cast<int64_t>(td->blockLength);
        tileNum_ = static_cast<int64_t>(td->tileNum);
        lastTileLen_ = static_cast<int64_t>(td->lastTileLength);
    } else {
        uint32_t tailIdx = blockIdx - td->formerCoreNum;
        startOffset_ = static_cast<int64_t>(td->formerCoreNum) * static_cast<int64_t>(td->blockLength) +
                       static_cast<int64_t>(tailIdx) * static_cast<int64_t>(td->tailBlockLength);
        blockCount_ = static_cast<int64_t>(td->tailBlockLength);
        tileNum_ = static_cast<int64_t>(td->tailTileNum);
        lastTileLen_ = static_cast<int64_t>(td->tailLastTileLength);
    }
    tileLength_ = static_cast<int64_t>(td->tileLength);

    if (blockCount_ == 0) {
        return;
    }

    // UB sizes rounded up to platform block boundaries.
    constexpr uint32_t IOBLK = UB_BLOCK_BYTES / static_cast<uint32_t>(sizeof(T));
    constexpr uint32_t F32BLK = UB_BLOCK_BYTES / static_cast<uint32_t>(sizeof(float));
    uint32_t ioAligned = ((static_cast<uint32_t>(tileLength_) + IOBLK - 1U) / IOBLK) * IOBLK *
                         static_cast<uint32_t>(sizeof(T));
    uint32_t f32Aligned = ((static_cast<uint32_t>(tileLength_) + F32BLK - 1U) / F32BLK) * F32BLK *
                          static_cast<uint32_t>(sizeof(float));

    pipe_.InitBuffer(inQueueY_, BUFFER_NUM, static_cast<int64_t>(ioAligned));
    pipe_.InitBuffer(inQueueDy_, BUFFER_NUM, static_cast<int64_t>(ioAligned));
    pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, static_cast<int64_t>(ioAligned));
    pipe_.InitBuffer(sinhBufPool_, static_cast<int64_t>(f32Aligned));
    pipe_.InitBuffer(dyBufPool_, static_cast<int64_t>(f32Aligned));
    pipe_.InitBuffer(tmpBufPool_, static_cast<int64_t>(f32Aligned));
}

template <typename T>
__aicore__ inline void AcoshGrad<T>::CopyIn(int64_t tileIdx, int64_t count)
{
    LocalTensor<T> yLocal = inQueueY_.AllocTensor<T>();
    LocalTensor<T> dyLocal = inQueueDy_.AllocTensor<T>();
    int64_t offset = startOffset_ + tileIdx * tileLength_;

    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(T));
    cp.srcStride = 0;
    cp.dstStride = 0;
    cp.rsv = 0;

    DataCopyPadExtParams<T> pad;
    pad.isPad = false;
    pad.leftPadding = 0;
    pad.rightPadding = 0;

    DataCopyPad(yLocal, yGm_[offset], cp, pad);
    DataCopyPad(dyLocal, dyGm_[offset], cp, pad);

    inQueueY_.EnQue(yLocal);
    inQueueDy_.EnQue(dyLocal);
}

template <typename T>
__aicore__ inline void AcoshGrad<T>::Compute(int64_t count)
{
    LocalTensor<T> yLocal = inQueueY_.DeQue<T>();
    LocalTensor<T> dyLocal = inQueueDy_.DeQue<T>();
    LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
    LocalTensor<float> sinhBuf = sinhBufPool_.Get<float>();
    LocalTensor<float> dyBuf = dyBufPool_.Get<float>();
    LocalTensor<float> tmpBuf = tmpBufPool_.Get<float>();
    uint32_t cnt = static_cast<uint32_t>(count);

    Cast(sinhBuf, yLocal, RoundMode::CAST_NONE, cnt);
    Cast(dyBuf, dyLocal, RoundMode::CAST_NONE, cnt);

    Muls(sinhBuf, sinhBuf, SCALE_1_8, cnt);
    Mul(tmpBuf, sinhBuf, sinhBuf, cnt);

    Muls(dyBuf, tmpBuf, TAYLOR_C7, cnt);
    Adds(dyBuf, dyBuf, TAYLOR_C5, cnt);
    Mul(dyBuf, dyBuf, tmpBuf, cnt);
    Adds(dyBuf, dyBuf, TAYLOR_C3, cnt);
    Mul(dyBuf, dyBuf, tmpBuf, cnt);
    Adds(dyBuf, dyBuf, 1.0f, cnt);
    Mul(sinhBuf, dyBuf, sinhBuf, cnt);

    Cast(dyBuf, dyLocal, RoundMode::CAST_NONE, cnt);

    for (int32_t iter = 0; iter < 3; iter++) {
        Mul(tmpBuf, sinhBuf, sinhBuf, cnt);
        Adds(tmpBuf, tmpBuf, 1.0f, cnt);
        Sqrt(tmpBuf, tmpBuf, cnt);
        Mul(sinhBuf, sinhBuf, tmpBuf, cnt);
        Muls(sinhBuf, sinhBuf, 2.0f, cnt);
    }

    Div(dyBuf, dyBuf, sinhBuf, cnt);
    Cast(zLocal, dyBuf, RoundMode::CAST_ROUND, cnt);

    inQueueY_.FreeTensor(yLocal);
    inQueueDy_.FreeTensor(dyLocal);
    outQueueZ_.EnQue(zLocal);
}

// bfloat16 specialisation: Cast(bf16→fp32) uses CAST_NONE, Cast(fp32→bf16) uses CAST_ROUND
template <>
__aicore__ inline void AcoshGrad<bfloat16_t>::Compute(int64_t count)
{
    LocalTensor<bfloat16_t> yLocal = inQueueY_.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> dyLocal = inQueueDy_.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> zLocal = outQueueZ_.AllocTensor<bfloat16_t>();
    LocalTensor<float> sinhBuf = sinhBufPool_.Get<float>();
    LocalTensor<float> dyBuf = dyBufPool_.Get<float>();
    LocalTensor<float> tmpBuf = tmpBufPool_.Get<float>();
    uint32_t cnt = static_cast<uint32_t>(count);

    // bf16 → float: CAST_NONE is correct
    Cast(sinhBuf, yLocal, RoundMode::CAST_NONE, cnt);
    Cast(dyBuf, dyLocal, RoundMode::CAST_NONE, cnt);

    Muls(sinhBuf, sinhBuf, SCALE_1_8, cnt);
    Mul(tmpBuf, sinhBuf, sinhBuf, cnt);

    Muls(dyBuf, tmpBuf, TAYLOR_C7, cnt);
    Adds(dyBuf, dyBuf, TAYLOR_C5, cnt);
    Mul(dyBuf, dyBuf, tmpBuf, cnt);
    Adds(dyBuf, dyBuf, TAYLOR_C3, cnt);
    Mul(dyBuf, dyBuf, tmpBuf, cnt);
    Adds(dyBuf, dyBuf, 1.0f, cnt);
    Mul(sinhBuf, dyBuf, sinhBuf, cnt);

    // Restore dy
    Cast(dyBuf, dyLocal, RoundMode::CAST_NONE, cnt);

    for (int32_t iter = 0; iter < 3; iter++) {
        Mul(tmpBuf, sinhBuf, sinhBuf, cnt);
        Adds(tmpBuf, tmpBuf, 1.0f, cnt);
        Sqrt(tmpBuf, tmpBuf, cnt);
        Mul(sinhBuf, sinhBuf, tmpBuf, cnt);
        Muls(sinhBuf, sinhBuf, 2.0f, cnt);
    }

    Div(dyBuf, dyBuf, sinhBuf, cnt);

    // float → bf16: must use CAST_ROUND (CAST_NONE not supported on 910B)
    Cast(zLocal, dyBuf, RoundMode::CAST_ROUND, cnt);

    inQueueY_.FreeTensor(yLocal);
    inQueueDy_.FreeTensor(dyLocal);
    outQueueZ_.EnQue(zLocal);
}

template <>
__aicore__ inline void AcoshGrad<float>::Compute(int64_t count)
{
    LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
    LocalTensor<float> dyLocal = inQueueDy_.DeQue<float>();
    LocalTensor<float> zLocal = outQueueZ_.AllocTensor<float>();
    LocalTensor<float> sinhBuf = sinhBufPool_.Get<float>();
    LocalTensor<float> dyBuf = dyBufPool_.Get<float>();
    LocalTensor<float> tmpBuf = tmpBufPool_.Get<float>();
    uint32_t cnt = static_cast<uint32_t>(count);

    Muls(sinhBuf, yLocal, SCALE_1_8, cnt);
    Mul(tmpBuf, sinhBuf, sinhBuf, cnt);

    Muls(dyBuf, tmpBuf, TAYLOR_C7, cnt);
    Adds(dyBuf, dyBuf, TAYLOR_C5, cnt);
    Mul(dyBuf, dyBuf, tmpBuf, cnt);
    Adds(dyBuf, dyBuf, TAYLOR_C3, cnt);
    Mul(dyBuf, dyBuf, tmpBuf, cnt);
    Adds(dyBuf, dyBuf, 1.0f, cnt);
    Mul(sinhBuf, dyBuf, sinhBuf, cnt);

    // 尽早把 dy 拷入 VECCALC，随后不再读输入队列张量 dyLocal，
    // 让 VECIN double-buffer 尽快释放给下一 tile 的 MTE2 预取，
    // 高延迟的 Div 只在 VECCALC 上做，不背队列同步 flag。
    Muls(dyBuf, dyLocal, 1.0f, cnt);

    for (int32_t iter = 0; iter < 3; iter++) {
        Mul(tmpBuf, sinhBuf, sinhBuf, cnt);
        Adds(tmpBuf, tmpBuf, 1.0f, cnt);
        Sqrt(tmpBuf, tmpBuf, cnt);
        Mul(sinhBuf, sinhBuf, tmpBuf, cnt);
        Muls(sinhBuf, sinhBuf, 2.0f, cnt);
    }

    Div(dyBuf, dyBuf, sinhBuf, cnt);
    Muls(zLocal, dyBuf, 1.0f, cnt);

    inQueueY_.FreeTensor(yLocal);
    inQueueDy_.FreeTensor(dyLocal);
    outQueueZ_.EnQue(zLocal);
}

template <typename T>
__aicore__ inline void AcoshGrad<T>::CopyOut(int64_t tileIdx, int64_t count)
{
    LocalTensor<T> zLocal = outQueueZ_.DeQue<T>();
    int64_t offset = startOffset_ + tileIdx * tileLength_;

    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(T));
    cp.srcStride = 0;
    cp.dstStride = 0;
    cp.rsv = 0;

    DataCopyPad(zGm_[offset], zLocal, cp);
    outQueueZ_.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void AcoshGrad<T>::Process()
{
    if (blockCount_ == 0) {
        return;
    }
    for (int64_t i = 0; i < tileNum_; i++) {
        int64_t count = (i < tileNum_ - 1) ? tileLength_ : lastTileLen_;
        CopyIn(i, count);
        Compute(count);
        CopyOut(i, count);
    }
}

} // namespace NsAcoshGrad
#endif // ACOSHGRAD_H
