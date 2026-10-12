/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pad_circular_simt_huge.h
 * \brief pad_circular_simt_huge
 */

#ifndef PAD_CIRCULAR_SIMT_HUGE_H
#define PAD_CIRCULAR_SIMT_HUGE_H

#include "pad_common.h"
#include "pad_v3_struct.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t CIRCULAR_HUGE_THREAD_DIM = 512;
#else
constexpr int32_t CIRCULAR_HUGE_THREAD_DIM = 2048;
constexpr int32_t CIRCULAR_HUGE_HALF_THREAD_DIM = 1024;
constexpr int32_t CIRCULAR_HUGE_QUATER_THREAD_DIM = 512;
constexpr int32_t CIRCULAR_HUGE_EIGHTH_THREAD_DIM = 256;
#endif

namespace PadV3 {
using namespace AscendC;

template <typename T>
class PadCircularSimtHuge {
public:
    __aicore__ inline PadCircularSimtHuge(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    GlobalTensor<T> mInputGM_;  // GM x
    GlobalTensor<T> mOutputGM_; // GM y

    uint32_t mBlockIdx_;         // 核号
    const PadACTilingData* mTD_; // tilingData
};

template <typename T>
__aicore__ inline void PadCircularSimtHuge<T>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,
                                                    const PadACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                       uint32_t blockIdx, uint32_t blockNum, uint64_t inShape0, int64_t left0)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        int64_t inIndexCirHuge1 = idx - left0;

        if (inIndexCirHuge1 < 0) {
            inIndexCirHuge1 += inShape0;
        } else if (inIndexCirHuge1 >= inShape0) {
            inIndexCirHuge1 -= inShape0;
        }
        outputGM[idx] = inputGM[inIndexCirHuge1];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                       uint32_t blockIdx, uint32_t blockNum, uint64_t outStride0, uint64_t inShape0,
                                       uint64_t inShape1, uint64_t m0, uint64_t s0, int64_t left0, int64_t left1)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexCirHuge2[DIM] = {0};

        inIndexCirHuge2[0] = Simt::UintDiv(dstIdx, m0, s0);
        inIndexCirHuge2[1] = dstIdx - inIndexCirHuge2[0] * outStride0;

        inIndexCirHuge2[0] -= left0;
        inIndexCirHuge2[1] -= left1;

        if (inIndexCirHuge2[0] < 0) {
            inIndexCirHuge2[0] += inShape0;
        } else if (inIndexCirHuge2[0] >= inShape0) {
            inIndexCirHuge2[0] -= inShape0;
        }

        if (inIndexCirHuge2[1] < 0) {
            inIndexCirHuge2[1] += inShape1;
        } else if (inIndexCirHuge2[1] >= inShape1) {
            inIndexCirHuge2[1] -= inShape1;
        }

        uint64_t inputOffset = uint64_t(inIndexCirHuge2[0]) * inShape1 + uint64_t(inIndexCirHuge2[1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                         uint32_t blockIdx, uint32_t blockNum, uint64_t outStride0, uint64_t outStride1,
                                         uint64_t inShape0, uint64_t inShape1, uint64_t inShape2, uint64_t m0,
                                         uint64_t m1, uint64_t s0, uint64_t s1, int64_t left0, int64_t left1,
                                         int64_t left2)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexCirHuge3[DIM] = {0};

        inIndexCirHuge3[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexCirHuge3[0] * outStride0;
        inIndexCirHuge3[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexCirHuge3[1] * outStride1;
        inIndexCirHuge3[DIM - 1] = dstIdx;

        inIndexCirHuge3[0] -= left0;
        inIndexCirHuge3[1] -= left1;
        inIndexCirHuge3[DIM - 1] -= left2;

        if (inIndexCirHuge3[0] < 0) {
            inIndexCirHuge3[0] += inShape0;
        } else if (inIndexCirHuge3[0] >= inShape0) {
            inIndexCirHuge3[0] -= inShape0;
        }

        if (inIndexCirHuge3[1] < 0) {
            inIndexCirHuge3[1] += inShape1;
        } else if (inIndexCirHuge3[1] >= inShape1) {
            inIndexCirHuge3[1] -= inShape1;
        }

        if (inIndexCirHuge3[2] < 0) {
            inIndexCirHuge3[2] += inShape2;
        } else if (inIndexCirHuge3[2] >= inShape2) {
            inIndexCirHuge3[2] -= inShape2;
        }

        uint64_t inputOffset = uint64_t(inIndexCirHuge3[0]) * inShape1 * inShape2 +
                               uint64_t(inIndexCirHuge3[1]) * inShape2 + uint64_t(inIndexCirHuge3[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                        uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                        uint64_t m1, uint64_t m2, uint64_t s0, uint64_t s1, uint64_t s2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexCirHuge4[DIM] = {0};

        inIndexCirHuge4[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexCirHuge4[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexCirHuge4[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexCirHuge4[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexCirHuge4[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexCirHuge4[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexCirHuge4[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexCirHuge4[i] -= static_cast<int64_t>(tD->leftPad[i]);
            if (inIndexCirHuge4[i] < 0) {
                inIndexCirHuge4[i] += static_cast<int64_t>(tD->inShape[i]);
            } else if (inIndexCirHuge4[i] >= static_cast<int64_t>(tD->inShape[i])) {
                inIndexCirHuge4[i] -= static_cast<int64_t>(tD->inShape[i]);
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexCirHuge4[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndexCirHuge4[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndexCirHuge4[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndexCirHuge4[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                        uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                        uint64_t m1, uint64_t m2, uint64_t m3, uint64_t s0, uint64_t s1, uint64_t s2,
                                        uint64_t s3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexCirHuge5[DIM] = {0};

        inIndexCirHuge5[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexCirHuge5[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexCirHuge5[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexCirHuge5[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexCirHuge5[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexCirHuge5[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexCirHuge5[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexCirHuge5[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexCirHuge5[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexCirHuge5[i] -= static_cast<int64_t>(tD->leftPad[i]);
            if (inIndexCirHuge5[i] < 0) {
                inIndexCirHuge5[i] += static_cast<int64_t>(tD->inShape[i]);
            } else if (inIndexCirHuge5[i] >= static_cast<int64_t>(tD->inShape[i])) {
                inIndexCirHuge5[i] -= static_cast<int64_t>(tD->inShape[i]);
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexCirHuge5[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndexCirHuge5[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndexCirHuge5[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndexCirHuge5[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndexCirHuge5[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                       uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                       uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t s0, uint64_t s1,
                                       uint64_t s2, uint64_t s3, uint64_t s4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexCirHuge6[DIM] = {0};

        inIndexCirHuge6[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexCirHuge6[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexCirHuge6[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexCirHuge6[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexCirHuge6[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexCirHuge6[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexCirHuge6[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexCirHuge6[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexCirHuge6[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexCirHuge6[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndexCirHuge6[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexCirHuge6[i] -= static_cast<int64_t>(tD->leftPad[i]);
            if (inIndexCirHuge6[i] < 0) {
                inIndexCirHuge6[i] += static_cast<int64_t>(tD->inShape[i]);
            } else if (inIndexCirHuge6[i] >= static_cast<int64_t>(tD->inShape[i])) {
                inIndexCirHuge6[i] -= static_cast<int64_t>(tD->inShape[i]);
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexCirHuge6[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndexCirHuge6[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndexCirHuge6[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndexCirHuge6[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndexCirHuge6[4]) * static_cast<uint64_t>(tD->inStride[4]) +
                               static_cast<uint64_t>(inIndexCirHuge6[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                         uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                         uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t m5, uint64_t s0,
                                         uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexCirHuge7[DIM] = {0};

        inIndexCirHuge7[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexCirHuge7[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexCirHuge7[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexCirHuge7[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexCirHuge7[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexCirHuge7[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexCirHuge7[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexCirHuge7[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexCirHuge7[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexCirHuge7[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndexCirHuge7[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndexCirHuge7[5] * static_cast<int64_t>(tD->outStride[5]);
        inIndexCirHuge7[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexCirHuge7[i] -= static_cast<int64_t>(tD->leftPad[i]);
            if (inIndexCirHuge7[i] < 0) {
                inIndexCirHuge7[i] += static_cast<int64_t>(tD->inShape[i]);
            } else if (inIndexCirHuge7[i] >= static_cast<int64_t>(tD->inShape[i])) {
                inIndexCirHuge7[i] -= static_cast<int64_t>(tD->inShape[i]);
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexCirHuge7[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndexCirHuge7[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndexCirHuge7[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndexCirHuge7[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndexCirHuge7[4]) * static_cast<uint64_t>(tD->inStride[4]) +
                               static_cast<uint64_t>(inIndexCirHuge7[5]) * static_cast<uint64_t>(tD->inStride[5]) +
                               static_cast<uint64_t>(inIndexCirHuge7[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_HUGE_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeCircularHugeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                         uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                         uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t m5, uint64_t m6,
                                         uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5,
                                         uint64_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexCirHuge8[DIM] = {0};

        inIndexCirHuge8[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexCirHuge8[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexCirHuge8[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexCirHuge8[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexCirHuge8[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexCirHuge8[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexCirHuge8[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexCirHuge8[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexCirHuge8[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexCirHuge8[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndexCirHuge8[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndexCirHuge8[5] * static_cast<int64_t>(tD->outStride[5]);
        inIndexCirHuge8[6] = Simt::UintDiv(dstIdx, m6, s6);
        dstIdx -= inIndexCirHuge8[6] * static_cast<int64_t>(tD->outStride[6]);
        inIndexCirHuge8[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexCirHuge8[i] -= static_cast<int64_t>(tD->leftPad[i]);
            if (inIndexCirHuge8[i] < 0) {
                inIndexCirHuge8[i] += static_cast<int64_t>(tD->inShape[i]);
            } else if (inIndexCirHuge8[i] >= static_cast<int64_t>(tD->inShape[i])) {
                inIndexCirHuge8[i] -= static_cast<int64_t>(tD->inShape[i]);
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexCirHuge8[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndexCirHuge8[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndexCirHuge8[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndexCirHuge8[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndexCirHuge8[4]) * static_cast<uint64_t>(tD->inStride[4]) +
                               static_cast<uint64_t>(inIndexCirHuge8[5]) * static_cast<uint64_t>(tD->inStride[5]) +
                               static_cast<uint64_t>(inIndexCirHuge8[6]) * static_cast<uint64_t>(tD->inStride[6]) +
                               static_cast<uint64_t>(inIndexCirHuge8[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T>
__aicore__ inline void PadCircularSimtHuge<T>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD_->dimNum;

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeCircularHugeDimOne<T>>(dim3(CIRCULAR_HUGE_HALF_THREAD_DIM),
                                                      (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                      (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), mTD_->outShape[0],
                                                      mBlockIdx_, blockNum, mTD_->inShape[0], mTD_->leftPad[0]);
        return;
    }

    uint64_t outputSize = mTD_->outShape[0] * mTD_->outStride[0];

    uint64_t s[8];
    uint64_t m[8];

    for (uint32_t i = 0; i < mDimNum - 1; ++i) {
        GetUintDivMagicAndShift(m[i], s[i], static_cast<uint64_t>(mTD_->outStride[i]));
    }

    if (mDimNum == 2) {
        asc_vf_call<SimtComputeCircularHugeDimTwo<T, 2>>(
            dim3(CIRCULAR_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->inShape[0], mTD_->inShape[1], m[0], s[0], mTD_->leftPad[0], mTD_->leftPad[1]);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeCircularHugeDimThree<T, 3>>(
            dim3(CIRCULAR_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->outStride[1], mTD_->inShape[0], mTD_->inShape[1], mTD_->inShape[2], m[0], m[1], s[0], s[1],
            mTD_->leftPad[0], mTD_->leftPad[1], mTD_->leftPad[2]);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeCircularHugeDimFour<T, 4>>(
            dim3(CIRCULAR_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            s[0], s[1], s[2]);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeCircularHugeDimFive<T, 5>>(
            dim3(CIRCULAR_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], s[0], s[1], s[2], s[3]);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeCircularHugeDimSix<T, 6>>(
            dim3(CIRCULAR_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], s[0], s[1], s[2], s[3], s[4]);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeCircularHugeDimSeven<T, 7>>(
            dim3(CIRCULAR_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], s[0], s[1], s[2], s[3], s[4], s[5]);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeCircularHugeDimEight<T, 8>>(
            dim3(CIRCULAR_HUGE_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], m[6], s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
    }
}
} // namespace PadV3

#endif
