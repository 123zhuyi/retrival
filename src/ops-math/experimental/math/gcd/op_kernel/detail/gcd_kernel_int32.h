/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_INT32_H_
#define DETAIL_GCD_KERNEL_INT32_H_

template <>
class GcdKernel<int32_t> {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y), static_cast<uint64_t>(tiling_.totalNum));
        contiguousElementwise_ = IsContiguousElementwise(tiling_);
        vectorEnabled_ = tiling_.totalNum >= GCD_INT32_VECTOR_MIN_ELEMENTS && contiguousElementwise_;
        if (vectorEnabled_) {
            InitGcdVectorWorkBuffers(buffers_, GCD_INT32_VECTOR_TILE, sizeof(int32_t), sizeof(int32_t));
            buffers_.pipe.InitBuffer(cmpBuf_, GCD_INT32_VECTOR_TILE * sizeof(uint8_t));
        }
    }

    __aicore__ inline void Process()
    {
        ProcessLinearKernel(*this, tiling_, contiguousElementwise_, vectorEnabled_, GCD_INT32_VECTOR_ALIGN_ELEMENTS,
                            GCD_INT32_VECTOR_TILE, x1Gm_, x2Gm_, yGm_);
    }

    __aicore__ inline bool ComputeVectorTile(int64_t yOffset, int64_t x1Offset, int64_t x2Offset, int32_t count,
                                             bool syncBeforeScalarTail)
    {
        LocalTensor<int32_t> x1Local = buffers_.x1Buf.Get<int32_t>();
        LocalTensor<int32_t> x2Local = buffers_.x2Buf.Get<int32_t>();
        LocalTensor<int32_t> yLocal = buffers_.yBuf.Get<int32_t>();
        LocalTensor<float> aLocal = buffers_.aBuf.Get<float>();
        LocalTensor<float> bLocal = buffers_.bBuf.Get<float>();
        LocalTensor<float> t1 = buffers_.t1Buf.Get<float>();
        LocalTensor<float> t2 = buffers_.t2Buf.Get<float>();
        LocalTensor<float> t3 = buffers_.t3Buf.Get<float>();
        LocalTensor<uint8_t> cmpMask = cmpBuf_.Get<uint8_t>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(int32_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(x1Local, x1Gm_[static_cast<uint64_t>(x1Offset)], copyParams, padParams);
        DataCopyPad(x2Local, x2Gm_[static_cast<uint64_t>(x2Offset)], copyParams, padParams);
        SyncPipe<HardEvent::MTE2_V>();
        Cast(aLocal, x1Local, RoundMode::CAST_NONE, count);
        Cast(bLocal, x2Local, RoundMode::CAST_NONE, count);
        Abs(aLocal, aLocal, count);
        Abs(bLocal, bLocal, count);
        ReduceMax<float>(t1, aLocal, t3, count, false);
        ReduceMax<float>(t2, bLocal, t3, count, false);
        SyncPipe<HardEvent::V_S>();
        float lhsMax = t1.GetValue(0);
        float rhsMax = t2.GetValue(0);
        if (lhsMax > GCD_INT32_VECTOR_SAFE_ABS_F || rhsMax > GCD_INT32_VECTOR_SAFE_ABS_F) {
            return false;
        }

        RunInt32Euclid(aLocal, bLocal, t1, t2, t3, cmpMask, count);

        Cast(yLocal, aLocal, RoundMode::CAST_RINT, count);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopyPad(yGm_[static_cast<uint64_t>(yOffset)], yLocal, copyParams);
        SyncMte3ToScalarIf(syncBeforeScalarTail);
        return true;
    }

private:
    GcdTilingData tiling_;
    bool contiguousElementwise_ = false;
    bool vectorEnabled_ = false;
    GcdVectorWorkBuffers buffers_;
    TBuf<TPosition::VECCALC> cmpBuf_;
    GlobalTensor<int32_t> x1Gm_;
    GlobalTensor<int32_t> x2Gm_;
    GlobalTensor<int32_t> yGm_;
};

#endif // DETAIL_GCD_KERNEL_INT32_H_
