/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_FLOAT_H_
#define DETAIL_GCD_KERNEL_FLOAT_H_

template <>
class GcdKernel<float> {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint64_t>(tiling_.totalNum));
        vectorEnabled_ = IsContiguousElementwise(tiling_) && tiling_.totalNum >= GCD_FLOAT_VECTOR_MIN_ELEMENTS;
        if (vectorEnabled_) {
            InitGcdVectorWorkBuffers(buffers_, GCD_FLOAT_VECTOR_TILE, sizeof(float), sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        bool contiguousElementwise = IsContiguousElementwise(tiling_);
        ProcessLinearKernel(*this, tiling_, contiguousElementwise, vectorEnabled_, GCD_FLOAT_VECTOR_ALIGN_ELEMENTS,
                            GCD_FLOAT_VECTOR_TILE, x1Gm_, x2Gm_, yGm_);
    }

    __aicore__ inline bool ComputeVectorTile(int64_t yOffset, int64_t x1Offset, int64_t x2Offset, int32_t count,
                                             bool syncBeforeScalarTail)
    {
        LocalTensor<float> x1Local = buffers_.x1Buf.Get<float>();
        LocalTensor<float> x2Local = buffers_.x2Buf.Get<float>();
        LocalTensor<float> yLocal = buffers_.yBuf.Get<float>();
        LocalTensor<float> aLocal = buffers_.aBuf.Get<float>();
        LocalTensor<float> bLocal = buffers_.bBuf.Get<float>();
        LocalTensor<float> t1 = buffers_.t1Buf.Get<float>();
        LocalTensor<float> t2 = buffers_.t2Buf.Get<float>();
        LocalTensor<float> t3 = buffers_.t3Buf.Get<float>();

        DataCopy(x1Local, x1Gm_[static_cast<uint64_t>(x1Offset)], count);
        DataCopy(x2Local, x2Gm_[static_cast<uint64_t>(x2Offset)], count);
        SyncPipe<HardEvent::MTE2_V>();
        Abs(aLocal, x1Local, count);
        Abs(bLocal, x2Local, count);
        ReduceMax<float>(t1, aLocal, t3, count, false);
        ReduceMax<float>(t2, bLocal, t3, count, false);
        SyncPipe<HardEvent::V_S>();
        float lhsMax = t1.GetValue(0);
        float rhsMax = t2.GetValue(0);
        if (lhsMax > GCD_FLOAT_VECTOR_SAFE_ABS_F || rhsMax > GCD_FLOAT_VECTOR_SAFE_ABS_F) {
            return false;
        }

        FloorFloatVectorInputs(aLocal, bLocal, t1, count);
        RunFloatEuclid(aLocal, bLocal, t1, t2, t3, count, GCD_FLOAT_VECTOR_MAX_ITER);

        Cast(yLocal, aLocal, RoundMode::CAST_RINT, count);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(yGm_[static_cast<uint64_t>(yOffset)], yLocal, count);
        SyncMte3ToScalarIf(syncBeforeScalarTail);
        return true;
    }

private:
    GcdTilingData tiling_;
    bool vectorEnabled_ = false;
    GcdVectorWorkBuffers buffers_;
    GlobalTensor<float> x1Gm_;
    GlobalTensor<float> x2Gm_;
    GlobalTensor<float> yGm_;
};

#endif // DETAIL_GCD_KERNEL_FLOAT_H_
