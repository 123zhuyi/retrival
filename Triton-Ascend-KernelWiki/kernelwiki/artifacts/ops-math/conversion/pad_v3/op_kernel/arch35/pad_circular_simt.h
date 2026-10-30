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
 * \file pad_circular_simt.h
 * \brief pad_circular_simt
 */

#ifndef PAD_CIRCULAR_SIMT_H
#define PAD_CIRCULAR_SIMT_H

#include "pad_common.h"
#include "pad_v3_struct.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t CIRCULAR_THREAD_DIM = 512;
#else
constexpr int32_t CIRCULAR_THREAD_DIM = 2048;
#endif

namespace PadV3 {
using namespace AscendC;

template <typename T>
class PadCircularSimt {
public:
    __aicore__ inline PadCircularSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    GlobalTensor<T> mInputGM_;  // GM x
    GlobalTensor<T> mOutputGM_; // GM y

    uint32_t mBlockIdx_;         // 核号
    const PadACTilingData* mTD_; // tilingData
};

template <typename T>
__aicore__ inline void PadCircularSimt<T>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,
                                                const PadACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint32_t inShape0, int32_t left0)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        int32_t inIndex0 = idx - left0; // 可能为负数或超出边界

        if (inIndex0 < 0) {
            inIndex0 += inShape0;
        } else if (inIndex0 >= inShape0) {
            inIndex0 -= inShape0;
        }
        outputGM[idx] = inputGM[inIndex0];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint32_t outStride0, uint32_t inShape0,
                                   uint32_t inShape1, uint32_t m0, uint32_t s0, int32_t left0, int32_t left1)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0); // 对dstIdx的每个元素做除以outStride的运算
        inIndex[1] = dstIdx - inIndex[0] * outStride0;

        inIndex[0] -= left0;
        inIndex[1] -= left1;

        if (inIndex[0] < 0) {
            inIndex[0] += inShape0;
        } else if (inIndex[0] >= inShape0) {
            inIndex[0] -= inShape0;
        }

        if (inIndex[1] < 0) {
            inIndex[1] += inShape1;
        } else if (inIndex[1] >= inShape1) {
            inIndex[1] -= inShape1;
        }

        uint32_t inputOffset = uint32_t(inIndex[0]) * inShape1 + uint32_t(inIndex[1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t outputSize,
                                     uint32_t blockIdx, uint32_t blockNum, uint32_t outStride0, uint32_t outStride1,
                                     uint32_t inShape0, uint32_t inShape1, uint32_t inShape2, uint32_t m0, uint32_t m1,
                                     uint32_t s0, uint32_t s1, int32_t left0, int32_t left1, int32_t left2)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * outStride0;
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * outStride1;
        inIndex[DIM - 1] = dstIdx;

        inIndex[0] -= left0;
        inIndex[1] -= left1;
        inIndex[2] -= left2;

        if (inIndex[0] < 0) {
            inIndex[0] += inShape0;
        } else if (inIndex[0] >= inShape0) {
            inIndex[0] -= inShape0;
        }

        if (inIndex[1] < 0) {
            inIndex[1] += inShape1;
        } else if (inIndex[1] >= inShape1) {
            inIndex[1] -= inShape1;
        }

        if (inIndex[2] < 0) {
            inIndex[2] += inShape2;
        } else if (inIndex[2] >= inShape2) {
            inIndex[2] -= inShape2;
        }

        uint32_t inputOffset = uint32_t(inIndex[0]) * inShape1 * inShape2 + uint32_t(inIndex[1]) * inShape2 +
                               uint32_t(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                    uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                    uint32_t s0, uint32_t s1, uint32_t s2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndex[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] -= static_cast<int32_t>(tD->leftPad[i]);
            if (inIndex[i] < 0) {
                inIndex[i] += static_cast<int32_t>(tD->inShape[i]);
            } else if (inIndex[i] >= static_cast<int32_t>(tD->inShape[i])) {
                inIndex[i] -= static_cast<int32_t>(tD->inShape[i]);
            }
        }
        uint32_t inputOffset = static_cast<uint32_t>(inIndex[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndex[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndex[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                    uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                    uint32_t m3, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndex[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] -= static_cast<int32_t>(tD->leftPad[i]);
            if (inIndex[i] < 0) {
                inIndex[i] += static_cast<int32_t>(tD->inShape[i]);
            } else if (inIndex[i] >= static_cast<int32_t>(tD->inShape[i])) {
                inIndex[i] -= static_cast<int32_t>(tD->inShape[i]);
            }
        }

        uint32_t inputOffset = static_cast<uint32_t>(inIndex[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndex[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndex[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndex[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                   uint32_t m3, uint32_t m4, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
                                   uint32_t s4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndex[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndex[4] * static_cast<int32_t>(tD->outStride[4]);
        inIndex[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] -= static_cast<int32_t>(tD->leftPad[i]);
            if (inIndex[i] < 0) {
                inIndex[i] += static_cast<int32_t>(tD->inShape[i]);
            } else if (inIndex[i] >= static_cast<int32_t>(tD->inShape[i])) {
                inIndex[i] -= static_cast<int32_t>(tD->inShape[i]);
            }
        }
        uint32_t inputOffset = static_cast<uint32_t>(inIndex[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndex[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndex[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndex[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndex[4]) * static_cast<uint32_t>(tD->inStride[4]) +
                               static_cast<uint32_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                     uint32_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint32_t m0,
                                     uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t s0,
                                     uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndex[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndex[4] * static_cast<int32_t>(tD->outStride[4]);
        inIndex[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndex[5] * static_cast<int32_t>(tD->outStride[5]);
        inIndex[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] -= static_cast<int32_t>(tD->leftPad[i]);
            if (inIndex[i] < 0) {
                inIndex[i] += static_cast<int32_t>(tD->inShape[i]);
            } else if (inIndex[i] >= static_cast<int32_t>(tD->inShape[i])) {
                inIndex[i] -= static_cast<int32_t>(tD->inShape[i]);
            }
        }
        uint32_t inputOffset = static_cast<uint32_t>(inIndex[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndex[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndex[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndex[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndex[4]) * static_cast<uint32_t>(tD->inStride[4]) +
                               static_cast<uint32_t>(inIndex[5]) * static_cast<uint32_t>(tD->inStride[5]) +
                               static_cast<uint32_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(CIRCULAR_THREAD_DIM) __aicore__
    void SimtComputeCircularDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                     uint32_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint32_t m0,
                                     uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6,
                                     uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                                     uint32_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndex[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndex[4] * static_cast<int32_t>(tD->outStride[4]);
        inIndex[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndex[5] * static_cast<int32_t>(tD->outStride[5]);
        inIndex[6] = Simt::UintDiv(dstIdx, m6, s6);
        dstIdx -= inIndex[6] * static_cast<int32_t>(tD->outStride[6]);
        inIndex[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndex[i] -= static_cast<int32_t>(tD->leftPad[i]);
            if (inIndex[i] < 0) {
                inIndex[i] += static_cast<int32_t>(tD->inShape[i]);
            } else if (inIndex[i] >= static_cast<int32_t>(tD->inShape[i])) {
                inIndex[i] -= static_cast<int32_t>(tD->inShape[i]);
            }
        }
        uint32_t inputOffset = static_cast<uint32_t>(inIndex[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndex[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndex[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndex[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndex[4]) * static_cast<uint32_t>(tD->inStride[4]) +
                               static_cast<uint32_t>(inIndex[5]) * static_cast<uint32_t>(tD->inStride[5]) +
                               static_cast<uint32_t>(inIndex[6]) * static_cast<uint32_t>(tD->inStride[6]) +
                               static_cast<uint32_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T>
__aicore__ inline void PadCircularSimt<T>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD_->dimNum;

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeCircularDimOne<T>>(dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                  (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), mTD_->outShape[0],
                                                  mBlockIdx_, blockNum, mTD_->inShape[0], mTD_->leftPad[0]);
        return;
    }

    uint32_t outputSize = mTD_->outShape[0] * mTD_->outStride[0];
    if (outputSize == 0) {
        return;
    }

    uint32_t s[8];
    uint32_t m[8];

    for (uint32_t i = 0; i < mDimNum - 1; ++i) {
        GetUintDivMagicAndShift(m[i], s[i], static_cast<uint32_t>(mTD_->outStride[i]));
    }

    if (mDimNum == 2) {
        asc_vf_call<SimtComputeCircularDimTwo<T, 2>>(dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                     (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize,
                                                     mBlockIdx_, blockNum, mTD_->outStride[0], mTD_->inShape[0],
                                                     mTD_->inShape[1], m[0], s[0], mTD_->leftPad[0], mTD_->leftPad[1]);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeCircularDimThree<T, 3>>(
            dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->outStride[1], mTD_->inShape[0], mTD_->inShape[1], mTD_->inShape[2], m[0], m[1], s[0], s[1],
            mTD_->leftPad[0], mTD_->leftPad[1], mTD_->leftPad[2]);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeCircularDimFour<T, 4>>(dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                      (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize,
                                                      mBlockIdx_, blockNum, m[0], m[1], m[2], s[0], s[1], s[2]);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeCircularDimFive<T, 5>>(dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                      (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize,
                                                      mBlockIdx_, blockNum, m[0], m[1], m[2], m[3], s[0], s[1], s[2],
                                                      s[3]);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeCircularDimSix<T, 6>>(dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                     (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize,
                                                     mBlockIdx_, blockNum, m[0], m[1], m[2], m[3], m[4], s[0], s[1],
                                                     s[2], s[3], s[4]);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeCircularDimSeven<T, 7>>(dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                       (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling,
                                                       outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2], m[3], m[4],
                                                       m[5], s[0], s[1], s[2], s[3], s[4], s[5]);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeCircularDimEight<T, 8>>(dim3(CIRCULAR_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                       (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling,
                                                       outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2], m[3], m[4],
                                                       m[5], m[6], s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
    }
}
} // namespace PadV3

#endif
