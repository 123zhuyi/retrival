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
 * \file pad_grad_constant_simt.h
 * \brief pad_grad_constant_simt
 */

#ifndef PAD_V3_GRAD_CONSTANT_SIMT_H
#define PAD_V3_GRAD_CONSTANT_SIMT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "pad_v3_grad_struct.h"

#ifdef __DAV_FPGA__
constexpr int32_t CONSTANT_GRAD_THREAD_DIM = 512;
constexpr int32_t CONSTANT_GRAD_HALF_THREAD_DIM = 512;
constexpr int32_t CONSTANT_GRAD_EIGHTH_THREAD_DIM = 256;
#else
constexpr int32_t CONSTANT_GRAD_THREAD_DIM = 2048;
constexpr int32_t CONSTANT_GRAD_HALF_THREAD_DIM = 1024;
constexpr int32_t CONSTANT_GRAD_EIGHTH_THREAD_DIM = 256;
#endif

namespace PadV3Grad {
using namespace AscendC;

template <typename T>
class PadV3GradConstantSimt {
public:
    __aicore__ inline PadV3GradConstantSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData);
    template <typename U, typename S>
    __aicore__ inline void Process();

private:
    GlobalTensor<T> mInputGM_;
    GlobalTensor<T> mOutputGM_;

    uint32_t mBlockIdx_;
    const PadV3GradACTilingData* mTD_;
};

template <typename T>
__aicore__ inline void PadV3GradConstantSimt<T>::Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T, typename U, typename S>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_THREAD_DIM) __aicore__
    void SimtComputeDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                           uint32_t blockNum, U inShape0, S left0)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U yIdx = idx;

        S inIndex0 = yIdx + left0;
        if (inIndex0 < 0 || inIndex0 >= inShape0) {
            outputGM[idx] = 0;
        } else {
            outputGM[idx] = inputGM[inIndex0];
        }
    }
}

template <typename T, typename U, typename S, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_THREAD_DIM) __aicore__
    void SimtComputeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                           uint32_t blockNum, __ubuf__ U* outStrides, __ubuf__ U* inStrides, __ubuf__ U* inShapes, U m0,
                           U s0, __ubuf__ S* lefts)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        S inIndex[DIM]{0};
        U yIdx = idx;

        inIndex[0] = Simt::UintDiv(yIdx, m0, s0);
        yIdx -= inIndex[0] * outStrides[0];
        inIndex[1] = yIdx;

        inIndex[0] = inIndex[0] + lefts[0];
        inIndex[1] = inIndex[1] + lefts[1];

        if (inIndex[0] < 0 || inIndex[0] >= inShapes[0] || inIndex[1] < 0 || inIndex[1] >= inShapes[1]) {
            outputGM[idx] = 0;
        } else {
            U inputOffset = U(inIndex[0]) * inStrides[0] + U(inIndex[1]);
            outputGM[idx] = inputGM[inputOffset];
        }
    }
}

template <typename T, typename U, typename S, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_THREAD_DIM) __aicore__
    void SimtComputeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                             uint32_t blockNum, __ubuf__ U* outStrides, __ubuf__ U* inStrides, __ubuf__ U* inShapes,
                             U m0, U m1, U s0, U s1, __ubuf__ S* lefts)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        S inIndex[DIM]{0};

        U yIdx = idx;

        inIndex[0] = Simt::UintDiv(yIdx, m0, s0);
        yIdx -= inIndex[0] * outStrides[0];
        inIndex[1] = Simt::UintDiv(yIdx, m1, s1);
        yIdx -= inIndex[1] * outStrides[1];
        inIndex[2] = yIdx;

        inIndex[0] = inIndex[0] + lefts[0];
        inIndex[1] = inIndex[1] + lefts[1];
        inIndex[2] = inIndex[2] + lefts[2];

        if (inIndex[0] < 0 || inIndex[0] >= inShapes[0] || inIndex[1] < 0 || inIndex[1] >= inShapes[1] ||
            inIndex[2] < 0 || inIndex[2] >= inShapes[2]) {
            outputGM[idx] = 0;
        } else {
            U inputOffset = U(inIndex[0]) * inStrides[0] + U(inIndex[1]) * inStrides[1] + U(inIndex[2]);
            outputGM[idx] = inputGM[inputOffset];
        }
    }
}

template <typename T, typename U, typename S, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_THREAD_DIM) __aicore__
    void SimtComputeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                            uint32_t blockNum, __ubuf__ U* outStrides, __ubuf__ U* inStrides, __ubuf__ U* inShapes,
                            U m0, U m1, U m2, U s0, U s1, U s2, __ubuf__ S* lefts)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        S inIndex[DIM]{0};

        U yIdx = idx;

        inIndex[0] = Simt::UintDiv(yIdx, m0, s0);
        yIdx -= inIndex[0] * outStrides[0];
        inIndex[1] = Simt::UintDiv(yIdx, m1, s1);
        yIdx -= inIndex[1] * outStrides[1];
        inIndex[2] = Simt::UintDiv(yIdx, m2, s2);
        yIdx -= inIndex[2] * outStrides[2];
        inIndex[3] = yIdx;

        inIndex[0] = inIndex[0] + lefts[0];
        inIndex[1] = inIndex[1] + lefts[1];
        inIndex[2] = inIndex[2] + lefts[2];
        inIndex[3] = inIndex[3] + lefts[3];

        if (inIndex[0] < 0 || inIndex[0] >= inShapes[0] || inIndex[1] < 0 || inIndex[1] >= inShapes[1] ||
            inIndex[2] < 0 || inIndex[2] >= inShapes[2] || inIndex[3] < 0 || inIndex[3] >= inShapes[3]) {
            outputGM[idx] = 0;
        } else {
            U inputOffset = U(inIndex[0]) * inStrides[0] + U(inIndex[1]) * inStrides[1] + U(inIndex[2]) * inStrides[2] +
                            U(inIndex[3]);
            outputGM[idx] = inputGM[inputOffset];
        }
    }
}

template <typename T, typename U, typename S, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_THREAD_DIM) __aicore__
    void SimtComputeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                            uint32_t blockNum, __ubuf__ U* outStrides, __ubuf__ U* inStrides, __ubuf__ U* inShapes,
                            U m0, U m1, U m2, U m3, U s0, U s1, U s2, U s3, __ubuf__ S* lefts)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        S inIndex[DIM]{0};

        U yIdx = idx;

        inIndex[0] = Simt::UintDiv(yIdx, m0, s0);
        yIdx -= inIndex[0] * static_cast<S>(outStrides[0]);
        inIndex[1] = Simt::UintDiv(yIdx, m1, s1);
        yIdx -= inIndex[1] * static_cast<S>(outStrides[1]);
        inIndex[2] = Simt::UintDiv(yIdx, m2, s2);
        yIdx -= inIndex[2] * static_cast<S>(outStrides[2]);
        inIndex[3] = Simt::UintDiv(yIdx, m3, s3);
        yIdx -= inIndex[3] * static_cast<S>(outStrides[3]);
        inIndex[4] = yIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] = inIndex[i] + static_cast<S>(lefts[i]);
        }

        if (inIndex[0] < 0 || inIndex[0] >= inShapes[0] || inIndex[1] < 0 || inIndex[1] >= inShapes[1] ||
            inIndex[2] < 0 || inIndex[2] >= inShapes[2] || inIndex[3] < 0 || inIndex[3] >= inShapes[3] ||
            inIndex[4] < 0 || inIndex[4] >= inShapes[4]) {
            outputGM[idx] = 0;
        } else {
            U inputOffset = static_cast<U>(inIndex[0]) * static_cast<U>(inStrides[0]) +
                            static_cast<U>(inIndex[1]) * static_cast<U>(inStrides[1]) +
                            static_cast<U>(inIndex[2]) * static_cast<U>(inStrides[2]) +
                            static_cast<U>(inIndex[3]) * static_cast<U>(inStrides[3]) + static_cast<U>(inIndex[4]);
            outputGM[idx] = inputGM[inputOffset];
        }
    }
}

template <typename T, typename U, typename S, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_HALF_THREAD_DIM) __aicore__
    void SimtComputeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                           uint32_t blockNum, __ubuf__ U* outStrides, __ubuf__ U* inStrides, __ubuf__ U* inShapes, U m0,
                           U m1, U m2, U m3, U m4, U s0, U s1, U s2, U s3, U s4, __ubuf__ S* lefts)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        S inIndex[DIM]{0};

        U yIdx = idx;

        inIndex[0] = Simt::UintDiv(yIdx, m0, s0);
        yIdx -= inIndex[0] * static_cast<S>(outStrides[0]);
        inIndex[1] = Simt::UintDiv(yIdx, m1, s1);
        yIdx -= inIndex[1] * static_cast<S>(outStrides[1]);
        inIndex[2] = Simt::UintDiv(yIdx, m2, s2);
        yIdx -= inIndex[2] * static_cast<S>(outStrides[2]);
        inIndex[3] = Simt::UintDiv(yIdx, m3, s3);
        yIdx -= inIndex[3] * static_cast<S>(outStrides[3]);
        inIndex[4] = Simt::UintDiv(yIdx, m4, s4);
        yIdx -= inIndex[4] * static_cast<S>(outStrides[4]);
        inIndex[5] = yIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] = inIndex[i] + static_cast<S>(lefts[i]);
        }

        if (inIndex[0] < 0 || inIndex[0] >= inShapes[0] || inIndex[1] < 0 || inIndex[1] >= inShapes[1] ||
            inIndex[2] < 0 || inIndex[2] >= inShapes[2] || inIndex[3] < 0 || inIndex[3] >= inShapes[3] ||
            inIndex[4] < 0 || inIndex[4] >= inShapes[4] || inIndex[5] < 0 || inIndex[5] >= inShapes[5]) {
            outputGM[idx] = 0;
        } else {
            U inputOffset = static_cast<U>(inIndex[0]) * static_cast<U>(inStrides[0]) +
                            static_cast<U>(inIndex[1]) * static_cast<U>(inStrides[1]) +
                            static_cast<U>(inIndex[2]) * static_cast<U>(inStrides[2]) +
                            static_cast<U>(inIndex[3]) * static_cast<U>(inStrides[3]) +
                            static_cast<U>(inIndex[4]) * static_cast<U>(inStrides[4]) + static_cast<U>(inIndex[5]);
            outputGM[idx] = inputGM[inputOffset];
        }
    }
}

template <typename T, typename U, typename S, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_HALF_THREAD_DIM) __aicore__
    void SimtComputeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                             uint32_t blockNum, __ubuf__ U* outStrides, __ubuf__ U* inStrides, __ubuf__ U* inShapes,
                             U m0, U m1, U m2, U m3, U m4, U m5, U s0, U s1, U s2, U s3, U s4, U s5, __ubuf__ S* lefts)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        S inIndex[DIM]{0};

        U yIdx = idx;

        inIndex[0] = Simt::UintDiv(yIdx, m0, s0);
        yIdx -= inIndex[0] * static_cast<S>(outStrides[0]);
        inIndex[1] = Simt::UintDiv(yIdx, m1, s1);
        yIdx -= inIndex[1] * static_cast<S>(outStrides[1]);
        inIndex[2] = Simt::UintDiv(yIdx, m2, s2);
        yIdx -= inIndex[2] * static_cast<S>(outStrides[2]);
        inIndex[3] = Simt::UintDiv(yIdx, m3, s3);
        yIdx -= inIndex[3] * static_cast<S>(outStrides[3]);
        inIndex[4] = Simt::UintDiv(yIdx, m4, s4);
        yIdx -= inIndex[4] * static_cast<S>(outStrides[4]);
        inIndex[5] = Simt::UintDiv(yIdx, m5, s5);
        yIdx -= inIndex[5] * static_cast<S>(outStrides[5]);
        inIndex[6] = yIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] = inIndex[i] + static_cast<S>(lefts[i]);
        }

        if (inIndex[0] < 0 || inIndex[0] >= inShapes[0] || inIndex[1] < 0 || inIndex[1] >= inShapes[1] ||
            inIndex[2] < 0 || inIndex[2] >= inShapes[2] || inIndex[3] < 0 || inIndex[3] >= inShapes[3] ||
            inIndex[4] < 0 || inIndex[4] >= inShapes[4] || inIndex[5] < 0 || inIndex[5] >= inShapes[5] ||
            inIndex[6] < 0 || inIndex[6] >= inShapes[6]) {
            outputGM[idx] = 0;
        } else {
            U inputOffset = static_cast<U>(inIndex[0]) * static_cast<U>(inStrides[0]) +
                            static_cast<U>(inIndex[1]) * static_cast<U>(inStrides[1]) +
                            static_cast<U>(inIndex[2]) * static_cast<U>(inStrides[2]) +
                            static_cast<U>(inIndex[3]) * static_cast<U>(inStrides[3]) +
                            static_cast<U>(inIndex[4]) * static_cast<U>(inStrides[4]) +
                            static_cast<U>(inIndex[5]) * static_cast<U>(inStrides[5]) + static_cast<U>(inIndex[6]);
            outputGM[idx] = inputGM[inputOffset];
        }
    }
}

template <typename T, typename U, typename S, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CONSTANT_GRAD_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, U outputSize, uint32_t blockIdx,
                             uint32_t blockNum, __ubuf__ U* outStrides, __ubuf__ U* inStrides, __ubuf__ U* inShapes,
                             U m0, U m1, U m2, U m3, U m4, U m5, U m6, U s0, U s1, U s2, U s3, U s4, U s5, U s6,
                             __ubuf__ S* lefts)
{
    for (U idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        S inIndex[DIM]{0};

        U yIdx = idx;

        inIndex[0] = Simt::UintDiv(yIdx, m0, s0);
        yIdx -= inIndex[0] * static_cast<S>(outStrides[0]);
        inIndex[1] = Simt::UintDiv(yIdx, m1, s1);
        yIdx -= inIndex[1] * static_cast<S>(outStrides[1]);
        inIndex[2] = Simt::UintDiv(yIdx, m2, s2);
        yIdx -= inIndex[2] * static_cast<S>(outStrides[2]);
        inIndex[3] = Simt::UintDiv(yIdx, m3, s3);
        yIdx -= inIndex[3] * static_cast<S>(outStrides[3]);
        inIndex[4] = Simt::UintDiv(yIdx, m4, s4);
        yIdx -= inIndex[4] * static_cast<S>(outStrides[4]);
        inIndex[5] = Simt::UintDiv(yIdx, m5, s5);
        yIdx -= inIndex[5] * static_cast<S>(outStrides[5]);
        inIndex[6] = Simt::UintDiv(yIdx, m6, s6);
        yIdx -= inIndex[6] * static_cast<S>(outStrides[6]);
        inIndex[7] = yIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] = inIndex[i] + static_cast<S>(lefts[i]);
        }

        if (inIndex[0] < 0 || inIndex[0] >= inShapes[0] || inIndex[1] < 0 || inIndex[1] >= inShapes[1] ||
            inIndex[2] < 0 || inIndex[2] >= inShapes[2] || inIndex[3] < 0 || inIndex[3] >= inShapes[3] ||
            inIndex[4] < 0 || inIndex[4] >= inShapes[4] || inIndex[5] < 0 || inIndex[5] >= inShapes[5] ||
            inIndex[6] < 0 || inIndex[6] >= inShapes[6] || inIndex[7] < 0 || inIndex[7] >= inShapes[7]) {
            outputGM[idx] = 0;
        } else {
            U inputOffset = static_cast<U>(inIndex[0]) * static_cast<U>(inStrides[0]) +
                            static_cast<U>(inIndex[1]) * static_cast<U>(inStrides[1]) +
                            static_cast<U>(inIndex[2]) * static_cast<U>(inStrides[2]) +
                            static_cast<U>(inIndex[3]) * static_cast<U>(inStrides[3]) +
                            static_cast<U>(inIndex[4]) * static_cast<U>(inStrides[4]) +
                            static_cast<U>(inIndex[5]) * static_cast<U>(inStrides[5]) +
                            static_cast<U>(inIndex[6]) * static_cast<U>(inStrides[6]) + static_cast<U>(inIndex[7]);
            outputGM[idx] = inputGM[inputOffset];
        }
    }
}

template <typename T>
template <typename U, typename S>
__aicore__ inline void PadV3GradConstantSimt<T>::Process()
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD_->dimNum; // 维度数

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeDimOne<T, U, S>>(dim3(CONSTANT_GRAD_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), mTD_->outShape[0],
                                                mBlockIdx_, blockNum, mTD_->inShape[0], mTD_->leftPad[0]);
        return;
    }

    U outputSize = mTD_->outShape[0] * mTD_->outStride[0];
    if (outputSize == 0) {
        return;
    }

    __ubuf__ U inShapes[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U inStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U outStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ S leftPads[PAD_GRAD_MAX_DIMS_NUM];
    for (uint32_t i = 0; i < mDimNum; ++i) {
        inShapes[i] = static_cast<U>(mTD_->inShape[i]);
        inStrides[i] = static_cast<U>(mTD_->inStride[i]);
        outStrides[i] = static_cast<U>(mTD_->outStride[i]);
        leftPads[i] = static_cast<S>(mTD_->leftPad[i]);
    }

    U s[8];
    U m[8];

    for (uint32_t i = 0; i < mDimNum; ++i) {
        GetUintDivMagicAndShift(m[i], s[i], static_cast<U>(mTD_->outStride[i]));
    }

    if (mDimNum == 2) {
        asc_vf_call<SimtComputeDimTwo<T, U, S, 2>>(dim3(CONSTANT_GRAD_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                   (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize,
                                                   mBlockIdx_, blockNum, outStrides, inStrides, inShapes, m[0], s[0],
                                                   leftPads);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeDimThree<T, U, S, 3>>(
            dim3(CONSTANT_GRAD_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, outStrides, inStrides,
            inShapes, m[0], m[1], s[0], s[1], leftPads);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeDimFour<T, U, S, 4>>(dim3(CONSTANT_GRAD_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                    (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize,
                                                    mBlockIdx_, blockNum, outStrides, inStrides, inShapes, m[0], m[1],
                                                    m[2], s[0], s[1], s[2], leftPads);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeDimFive<T, U, S, 5>>(dim3(CONSTANT_GRAD_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                    (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize,
                                                    mBlockIdx_, blockNum, outStrides, inStrides, inShapes, m[0], m[1],
                                                    m[2], m[3], s[0], s[1], s[2], s[3], leftPads);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeDimSix<T, U, S, 6>>(
            dim3(CONSTANT_GRAD_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, outStrides, inStrides,
            inShapes, m[0], m[1], m[2], m[3], m[4], s[0], s[1], s[2], s[3], s[4], leftPads);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeDimSeven<T, U, S, 7>>(
            dim3(CONSTANT_GRAD_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, outStrides, inStrides,
            inShapes, m[0], m[1], m[2], m[3], m[4], m[5], s[0], s[1], s[2], s[3], s[4], s[5], leftPads);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeDimEight<T, U, S, 8>>(
            dim3(CONSTANT_GRAD_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, outStrides, inStrides,
            inShapes, m[0], m[1], m[2], m[3], m[4], m[5], m[6], s[0], s[1], s[2], s[3], s[4], s[5], s[6], leftPads);
    }
}
} // namespace PadV3Grad

#endif
