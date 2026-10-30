/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file eye_simd.h
 * \brief eye_simd
 */

#ifndef ASCENDC_EYE_SIMD_H_
#define ASCENDC_EYE_SIMD_H_

#include "kernel_operator.h"

namespace Eye {
using namespace AscendC;

constexpr uint32_t VECTOR_LENGTH = 256U;

template <typename T, typename U>
class EyeKernelSimd {
public:
    __aicore__ inline EyeKernelSimd(const EyeForAscendCTilingData& tilingData, TPipe& pipe)
        : td_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void EyeCompute();
    template <typename V>
    __aicore__ inline void GenScatterIndex();
    __aicore__ inline void GenEyeBuf();

private:
    GlobalTensor<T> y_;
    TQue<QuePosition::VECOUT, 1> zeroBuf_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TPipe& pipe_;

    int64_t blockIdx_{0};
    int64_t loopNum_{0};
    int64_t ubProNum_{0};
    int64_t tailLoopLength_{0};
    int64_t curCoreData_{0};
    const EyeForAscendCTilingData& td_;
};

template <typename T, typename U>
__aicore__ inline void EyeKernelSimd<T, U>::Init(GM_ADDR y)
{
    ubProNum_ = td_.loopLength * td_.numRows * td_.numColumns;

    y_.SetGlobalBuffer((__gm__ T*)(y));
    pipe_.InitBuffer(zeroBuf_, 1, ubProNum_ * sizeof(T));
    pipe_.InitBuffer(indexBuf_, VECTOR_LENGTH);

    blockIdx_ = GetBlockIdx();
    curCoreData_ = blockIdx_ != (td_.usedCoreNum - 1) ? td_.normBlockData : td_.tailBlockData;
    loopNum_ = curCoreData_ / td_.loopLength;
    tailLoopLength_ = curCoreData_ - loopNum_ * td_.loopLength;
}

template <typename T, typename U>
template <typename V>
__aicore__ inline void EyeKernelSimd<T, U>::GenScatterIndex()
{
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    uint32_t vfLen = VECTOR_LENGTH / sizeof(U);
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    uint32_t rows = td_.numRows;
    uint32_t cols = td_.numColumns;
#if defined(__CCE_KT_TEST__)
    uint32_t oneInBatchNum = std::min(td_.numRows, td_.numColumns);
#else
    uint32_t oneInBatchNum = min(td_.numRows, td_.numColumns);
#endif
    __VEC_SCOPE__
    {
        Reg::RegTensor<U> index;
        Reg::RegTensor<V> tmp;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        uint32_t num = vfLen;
        Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
        Reg::Arange(tmp, 0);
        index = (Reg::RegTensor<U>&)tmp;
        Reg::Duplicate(v0, (U)(cols + 1), p0);
        Reg::Duplicate(v1, (U)(oneInBatchNum), p0);
        Reg::Duplicate(v2, (U)(rows * cols), p0);
        Reg::Div(vd2, index, v1, p0);
        Reg::Mul(vd3, vd2, v1, p0);
        Reg::Sub(vd4, index, vd3, p0);

        Reg::Mul(vd5, vd2, v2, p0);
        Reg::Mul(vd1, vd4, v0, p0);
        Reg::Add(vd6, vd5, vd1, p0);
        Reg::DataCopy(dstAddr, vd6, p0);
    }
}

template <typename T, typename U>
__aicore__ inline void EyeKernelSimd<T, U>::GenEyeBuf()
{
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<T> tmpLocal = zeroBuf_.AllocTensor<T>();
    Duplicate(tmpLocal, static_cast<T>(0), ubProNum_);
    uint32_t vfLen = VECTOR_LENGTH / sizeof(U);
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto zeroAddr = (__ubuf__ T*)tmpLocal.GetPhyAddr();

    uint32_t rows = td_.numRows;
    uint32_t cols = td_.numColumns;
#if defined(__CCE_KT_TEST__)
    uint32_t oneInBatchNum = std::min(td_.numRows, td_.numColumns);
#else
    uint32_t oneInBatchNum = min(td_.numRows, td_.numColumns);
#endif
    if (oneInBatchNum < vfLen) {
        uint16_t regFactor0 = vfLen / oneInBatchNum;
        uint16_t size0 = td_.loopLength / regFactor0;
        uint16_t tailRegFactor0 = td_.loopLength - size0 * regFactor0;
        uint32_t stride = regFactor0 * rows * cols;
        __VEC_SCOPE__
        {
            uint32_t main = regFactor0 * oneInBatchNum;
            uint32_t tail = tailRegFactor0 * oneInBatchNum;
            Reg::MaskReg p0 = Reg::UpdateMask<U>(main);
            Reg::MaskReg p1 = Reg::UpdateMask<U>(tail);
            Reg::RegTensor<U> vd0;
            Reg::RegTensor<U> vd1;
            Reg::RegTensor<T> one;
            Reg::Duplicate(one, (T)(1), p0);
            Reg::DataCopy(vd0, indexAddr);
            for (uint16_t i = 0; i < size0; i++) {
                Reg::Adds(vd1, vd0, (U)(i * stride), p0);
                Reg::DataCopyScatter(zeroAddr, one, vd1, p0);
            }
            Reg::Adds(vd1, vd0, (U)(size0 * stride), p1);
            Reg::DataCopyScatter(zeroAddr, one, vd1, p1);
        }
    } else {
        uint16_t regFactor1 = vfLen;
        uint16_t size1 = oneInBatchNum / regFactor1;
        uint16_t tailRegFactor1 = oneInBatchNum - size1 * regFactor1;
        uint16_t size0 = td_.loopLength;
        uint32_t stride = regFactor1 * (cols + 1);
        uint32_t batchEleNum = rows * cols;
        __VEC_SCOPE__
        {
            uint32_t main = regFactor1;
            uint32_t tail = tailRegFactor1;
            Reg::MaskReg p0 = Reg::UpdateMask<U>(main);
            Reg::MaskReg p1 = Reg::UpdateMask<U>(tail);
            Reg::RegTensor<U> vd0;
            Reg::RegTensor<U> vd1;
            Reg::RegTensor<U> vd2;
            Reg::RegTensor<U> index;
            Reg::RegTensor<T> one;
            Reg::Duplicate(one, (T)(1), p0);
            Reg::DataCopy(index, indexAddr);
            for (uint16_t i = 0; i < size0; i++) {
                Reg::Adds(vd0, index, (U)(i * batchEleNum), p0);
                for (uint16_t j = 0; j < size1; j++) {
                    Reg::Adds(vd1, vd0, (U)(j * stride), p0);
                    Reg::DataCopyScatter(zeroAddr, one, vd1, p0);
                }
                Reg::Adds(vd2, vd0, (U)(size1 * stride), p1);
                Reg::DataCopyScatter(zeroAddr, one, vd2, p1);
            }
        }
    }
    zeroBuf_.EnQue<T>(tmpLocal);
}

template <typename T, typename U>
__aicore__ inline void EyeKernelSimd<T, U>::EyeCompute()
{
    if constexpr (IsSameType<U, uint32_t>::value) {
        GenScatterIndex<int32_t>();
    } else if constexpr (IsSameType<U, uint16_t>::value) {
        GenScatterIndex<int16_t>();
    } else if constexpr (IsSameType<U, uint64_t>::value) {
        GenScatterIndex<int64_t>();
    }
    GenEyeBuf();
    LocalTensor<T> srcLocal = zeroBuf_.DeQue<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(ubProNum_ * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    int64_t offset = blockIdx_ * td_.normBlockData * td_.numRows * td_.numColumns;
    for (int64_t i = 0; i < loopNum_; i++) {
        DataCopyPad(y_[offset + i * ubProNum_], srcLocal, copyParams);
    }
    if (tailLoopLength_ > 0) {
        copyParams.blockLen = tailLoopLength_ * td_.numRows * td_.numColumns * sizeof(T);
        DataCopyPad(y_[offset + loopNum_ * ubProNum_], srcLocal, copyParams);
    }
    zeroBuf_.FreeTensor(srcLocal);
}

template <typename T, typename U>
__aicore__ inline void EyeKernelSimd<T, U>::Process()
{
    if (blockIdx_ < td_.usedCoreNum) {
        EyeCompute();
    }
}
} // namespace Eye

#endif // ASCENDC_EYE_SIMD_H_
