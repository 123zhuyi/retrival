/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_HALF_H_
#define DETAIL_GCD_KERNEL_HALF_H_

template <>
class GcdKernel<half> {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 1) / 2));
        x1HalfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1));
        x2HalfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2));
        yHalfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
        vectorEnabled_ = IsContiguousElementwise(tiling_) && tiling_.totalNum >= GCD_FP16_VECTOR_MIN_ELEMENTS;
        if (vectorEnabled_) {
            InitGcdVectorWorkBuffers(buffers_, GCD_FP16_VECTOR_TILE, sizeof(half), sizeof(half));
        }
    }

    __aicore__ inline void Process()
    {
        ProcessPackedKernel<GcdKernel<half>, 2, 16>(*this, tiling_, IsContiguousElementwise(tiling_), vectorEnabled_,
                                                    GCD_FP16_VECTOR_ALIGN_ELEMENTS, GCD_FP16_VECTOR_TILE, yGm_);
    }

    __aicore__ inline uint16_t ComputeContiguousLaneBits(int64_t word, int64_t lane)
    {
        return ComputeFp16Bits(word * 2 + lane, word * 2 + lane);
    }

    __aicore__ inline uint16_t ComputeStridedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        return ComputeFp16Bits(x1Offset, x2Offset);
    }

    __aicore__ inline bool ComputePackedVectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        ComputeFp16VectorTile(linear, count, syncBeforeScalarTail);
        return true;
    }

private:
    __aicore__ inline uint16_t ComputeFp16Bits(int64_t x1Offset, int64_t x2Offset)
    {
        uint64_t lhs = GcdOp::AbsFp16BitsToU64(x1Gm_.GetValue(static_cast<uint64_t>(x1Offset)));
        uint64_t rhs = GcdOp::AbsFp16BitsToU64(x2Gm_.GetValue(static_cast<uint64_t>(x2Offset)));
        return GcdOp::U64ToFp16Bits(GcdOp::GcdUnsigned64(lhs, rhs));
    }

    __aicore__ inline void ComputeFp16VectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        LocalTensor<half> x1Local = buffers_.x1Buf.Get<half>();
        LocalTensor<half> x2Local = buffers_.x2Buf.Get<half>();
        LocalTensor<half> yLocal = buffers_.yBuf.Get<half>();
        LocalTensor<float> aLocal = buffers_.aBuf.Get<float>();
        LocalTensor<float> bLocal = buffers_.bBuf.Get<float>();
        LocalTensor<float> t1 = buffers_.t1Buf.Get<float>();
        LocalTensor<float> t2 = buffers_.t2Buf.Get<float>();
        LocalTensor<float> t3 = buffers_.t3Buf.Get<float>();

        DataCopy(x1Local, x1HalfGm_[static_cast<uint64_t>(linear)], count);
        DataCopy(x2Local, x2HalfGm_[static_cast<uint64_t>(linear)], count);
        SyncPipe<HardEvent::MTE2_V>();
        Cast(aLocal, x1Local, RoundMode::CAST_NONE, count);
        Cast(bLocal, x2Local, RoundMode::CAST_NONE, count);
        Abs(aLocal, aLocal, count);
        Abs(bLocal, bLocal, count);
        FloorFloatVectorInputs(aLocal, bLocal, t1, count);
        RunFloatEuclid(aLocal, bLocal, t1, t2, t3, count, GCD_FP16_VECTOR_MAX_ITER);

        Cast(yLocal, aLocal, RoundMode::CAST_RINT, count);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(yHalfGm_[static_cast<uint64_t>(linear)], yLocal, count);
        SyncMte3ToScalarIf(syncBeforeScalarTail);
    }

    GcdTilingData tiling_;
    bool vectorEnabled_ = false;
    GcdVectorWorkBuffers buffers_;
    GlobalTensor<uint16_t> x1Gm_;
    GlobalTensor<uint16_t> x2Gm_;
    GlobalTensor<uint32_t> yGm_;
    GlobalTensor<half> x1HalfGm_;
    GlobalTensor<half> x2HalfGm_;
    GlobalTensor<half> yHalfGm_;
};

#endif // DETAIL_GCD_KERNEL_HALF_H_
