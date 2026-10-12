/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_INT16_H_
#define DETAIL_GCD_KERNEL_INT16_H_

template <>
class GcdKernel<int16_t> {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x2));
        x1WordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x1));
        x2WordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 1) / 2));
        x1IntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(x1));
        x2IntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(x2));
        yIntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(y));
        vectorEnabled_ = IsContiguousElementwise(tiling_) && tiling_.totalNum >= GCD_INT16_VECTOR_MIN_ELEMENTS;
        if (vectorEnabled_) {
            InitGcdVectorWorkBuffers(buffers_, GCD_INT16_VECTOR_TILE, sizeof(int16_t), sizeof(int16_t));
        }
    }

    __aicore__ inline void Process()
    {
        ProcessPackedKernel<GcdKernel<int16_t>, 2, 16>(*this, tiling_, IsContiguousElementwise(tiling_), vectorEnabled_,
                                                       GCD_INT16_VECTOR_ALIGN_ELEMENTS, GCD_INT16_VECTOR_TILE, yGm_);
    }

    __aicore__ inline uint16_t ComputeContiguousLaneBits(int64_t word, int64_t lane)
    {
        uint32_t lhsWord = x1WordGm_.GetValue(static_cast<uint64_t>(word));
        uint32_t rhsWord = x2WordGm_.GetValue(static_cast<uint64_t>(word));
        uint16_t lhsBits = static_cast<uint16_t>((lhsWord >> (lane * 16)) & 0xffffU);
        uint16_t rhsBits = static_cast<uint16_t>((rhsWord >> (lane * 16)) & 0xffffU);
        return GcdOp::GcdInt16RawBits(lhsBits, rhsBits);
    }

    __aicore__ inline uint16_t ComputeStridedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        return GcdOp::GcdInt16RawBits(x1Gm_.GetValue(static_cast<uint64_t>(x1Offset)),
                                      x2Gm_.GetValue(static_cast<uint64_t>(x2Offset)));
    }

    __aicore__ inline bool ComputePackedVectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        ComputeInt16VectorTile(linear, count, syncBeforeScalarTail);
        return true;
    }

private:
    __aicore__ inline void ComputeInt16VectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        LocalTensor<int16_t> x1Local = buffers_.x1Buf.Get<int16_t>();
        LocalTensor<int16_t> x2Local = buffers_.x2Buf.Get<int16_t>();
        LocalTensor<int16_t> yLocal = buffers_.yBuf.Get<int16_t>();
        LocalTensor<float> aLocal = buffers_.aBuf.Get<float>();
        LocalTensor<float> bLocal = buffers_.bBuf.Get<float>();
        LocalTensor<float> t1 = buffers_.t1Buf.Get<float>();
        LocalTensor<float> t2 = buffers_.t2Buf.Get<float>();
        LocalTensor<float> t3 = buffers_.t3Buf.Get<float>();

        DataCopy(x1Local, x1IntGm_[static_cast<uint64_t>(linear)], count);
        DataCopy(x2Local, x2IntGm_[static_cast<uint64_t>(linear)], count);
        SyncPipe<HardEvent::MTE2_V>();
        Cast(aLocal, x1Local, RoundMode::CAST_NONE, count);
        Cast(bLocal, x2Local, RoundMode::CAST_NONE, count);
        Abs(aLocal, aLocal, count);
        Abs(bLocal, bLocal, count);

        RunFloatEuclid(aLocal, bLocal, t1, t2, t3, count, GCD_INT16_VECTOR_MAX_ITER);

        Muls(t1, aLocal, 1.0f / 32768.0f, count);
        Cast(t2.ReinterpretCast<int32_t>(), t1, RoundMode::CAST_FLOOR, count);
        Cast(t1, t2.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
        Muls(t1, t1, 65536.0f, count);
        Sub(aLocal, aLocal, t1, count);
        Cast(yLocal, aLocal, RoundMode::CAST_RINT, count);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(yIntGm_[static_cast<uint64_t>(linear)], yLocal, count);
        SyncMte3ToScalarIf(syncBeforeScalarTail);
    }

    GcdTilingData tiling_;
    bool vectorEnabled_ = false;
    GcdVectorWorkBuffers buffers_;
    GlobalTensor<uint16_t> x1Gm_;
    GlobalTensor<uint16_t> x2Gm_;
    GlobalTensor<uint32_t> x1WordGm_;
    GlobalTensor<uint32_t> x2WordGm_;
    GlobalTensor<int16_t> x1IntGm_;
    GlobalTensor<int16_t> x2IntGm_;
    GlobalTensor<int16_t> yIntGm_;
    GlobalTensor<uint32_t> yGm_;
};

#endif // DETAIL_GCD_KERNEL_INT16_H_
