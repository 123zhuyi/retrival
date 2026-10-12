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
 * \file pad_edge_simt_huge.h
 * \brief pad_edge_simt_huge
 */

#ifndef PAD_EDGE_SIMT_HUGE_H
#define PAD_EDGE_SIMT_HUGE_H

#include "pad_common.h"
#include "pad_v3_struct.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t EDGE_HUGE_THREAD_DIM = 512;
#else
constexpr int32_t EDGE_HUGE_THREAD_DIM = 2048;
constexpr int32_t EDGE_HUGE_HALF_THREAD_DIM = 1024;
constexpr int32_t EDGE_HUGE_QUATER_THREAD_DIM = 512;
#endif

namespace PadV3 {
using namespace AscendC;

template <typename T>
class PadEdgeSimtHuge {
public:
    __aicore__ inline PadEdgeSimtHuge(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    GlobalTensor<T> mInputGM_;  // GM x
    GlobalTensor<T> mOutputGM_; // GM y

    uint32_t mBlockIdx_;         // 核号
    const PadACTilingData* mTD_; // tilingData
};

template <typename T>
__aicore__ inline void PadEdgeSimtHuge<T>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,
                                                const PadACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint64_t inShape0, int64_t left0)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        int64_t inIndexEdgeHuge1 = static_cast<int64_t>(idx);

        inIndexEdgeHuge1 = min(max(inIndexEdgeHuge1, left0), int64_t(inShape0 + left0 - 1)) - left0;
        outputGM[idx] = inputGM[inIndexEdgeHuge1];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint64_t outStride0, uint64_t inShape0,
                                   uint64_t inShape1, uint64_t m0, uint64_t s0, int64_t left0, int64_t left1)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexEdgeHuge2[DIM] = {0};

        inIndexEdgeHuge2[0] = Simt::UintDiv(dstIdx, m0, s0);
        inIndexEdgeHuge2[1] = dstIdx - inIndexEdgeHuge2[0] * static_cast<int64_t>(outStride0);

        inIndexEdgeHuge2[0] = min(max(inIndexEdgeHuge2[0], left0), int64_t(inShape0 + left0 - 1)) - left0;
        inIndexEdgeHuge2[1] = min(max(inIndexEdgeHuge2[1], left1), int64_t(inShape1 + left1 - 1)) - left1;

        uint64_t inputOffset = uint64_t(inIndexEdgeHuge2[0]) * inShape1 + uint64_t(inIndexEdgeHuge2[1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                     uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                     uint64_t m1, uint64_t s0, uint64_t s1, int64_t left0, int64_t left1, int64_t left2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexEdgeHuge3[DIM] = {0};

        inIndexEdgeHuge3[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexEdgeHuge3[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexEdgeHuge3[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexEdgeHuge3[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexEdgeHuge3[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexEdgeHuge3[i] = min(max(inIndexEdgeHuge3[i], static_cast<int64_t>(tD->leftPad[i])),
                                      static_cast<int64_t>(tD->inShape[i] + static_cast<int64_t>(tD->leftPad[i]) - 1)) -
                                  static_cast<int64_t>(tD->leftPad[i]);
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexEdgeHuge3[0]) * tD->inStride[0] +
                               static_cast<uint64_t>(inIndexEdgeHuge3[1]) * tD->inStride[1] +
                               static_cast<uint64_t>(inIndexEdgeHuge3[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint64_t outputSize,
                                    uint32_t blockIdx, uint32_t blockNum, uint64_t m0, uint64_t m1, uint64_t m2,
                                    uint64_t s0, uint64_t s1, uint64_t s2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexEdgeHuge4[DIM] = {0};

        inIndexEdgeHuge4[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexEdgeHuge4[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexEdgeHuge4[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexEdgeHuge4[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexEdgeHuge4[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexEdgeHuge4[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexEdgeHuge4[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexEdgeHuge4[i] = min(max(inIndexEdgeHuge4[i], static_cast<int64_t>(tD->leftPad[i])),
                                      static_cast<int64_t>(tD->inShape[i] + static_cast<int64_t>(tD->leftPad[i]) - 1)) -
                                  static_cast<int64_t>(tD->leftPad[i]);
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexEdgeHuge4[0]) * tD->inStride[0] +
                               static_cast<uint64_t>(inIndexEdgeHuge4[1]) * tD->inStride[1] +
                               static_cast<uint64_t>(inIndexEdgeHuge4[2]) * tD->inStride[2] +
                               static_cast<uint64_t>(inIndexEdgeHuge4[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint64_t outputSize,
                                    uint32_t blockIdx, uint32_t blockNum, uint64_t m0, uint64_t m1, uint64_t m2,
                                    uint64_t m3, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexEdgeHuge5[DIM] = {0};

        inIndexEdgeHuge5[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexEdgeHuge5[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexEdgeHuge5[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexEdgeHuge5[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexEdgeHuge5[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexEdgeHuge5[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexEdgeHuge5[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexEdgeHuge5[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexEdgeHuge5[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexEdgeHuge5[i] = min(max(inIndexEdgeHuge5[i], static_cast<int64_t>(tD->leftPad[i])),
                                      static_cast<int64_t>(tD->inShape[i] + static_cast<int64_t>(tD->leftPad[i]) - 1)) -
                                  static_cast<int64_t>(tD->leftPad[i]);
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexEdgeHuge5[0]) * tD->inStride[0] +
                               static_cast<uint64_t>(inIndexEdgeHuge5[1]) * tD->inStride[1] +
                               static_cast<uint64_t>(inIndexEdgeHuge5[2]) * tD->inStride[2] +
                               static_cast<uint64_t>(inIndexEdgeHuge5[3]) * tD->inStride[3] +
                               static_cast<uint64_t>(inIndexEdgeHuge5[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint64_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint64_t m0, uint64_t m1, uint64_t m2,
                                   uint64_t m3, uint64_t m4, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3,
                                   uint64_t s4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexEdgeHuge6[DIM] = {0};

        inIndexEdgeHuge6[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexEdgeHuge6[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexEdgeHuge6[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexEdgeHuge6[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexEdgeHuge6[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexEdgeHuge6[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexEdgeHuge6[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexEdgeHuge6[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexEdgeHuge6[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexEdgeHuge6[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndexEdgeHuge6[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexEdgeHuge6[i] = min(max(inIndexEdgeHuge6[i], static_cast<int64_t>(tD->leftPad[i])),
                                      static_cast<int64_t>(tD->inShape[i] + static_cast<int64_t>(tD->leftPad[i]) - 1)) -
                                  static_cast<int64_t>(tD->leftPad[i]);
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexEdgeHuge6[0]) * tD->inStride[0] +
                               static_cast<uint64_t>(inIndexEdgeHuge6[1]) * tD->inStride[1] +
                               static_cast<uint64_t>(inIndexEdgeHuge6[2]) * tD->inStride[2] +
                               static_cast<uint64_t>(inIndexEdgeHuge6[3]) * tD->inStride[3] +
                               static_cast<uint64_t>(inIndexEdgeHuge6[4]) * tD->inStride[4] +
                               static_cast<uint64_t>(inIndexEdgeHuge6[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                     uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                     uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t m5, uint64_t s0,
                                     uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexEdgeHuge7[DIM] = {0};

        inIndexEdgeHuge7[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexEdgeHuge7[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexEdgeHuge7[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexEdgeHuge7[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexEdgeHuge7[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexEdgeHuge7[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexEdgeHuge7[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexEdgeHuge7[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexEdgeHuge7[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexEdgeHuge7[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndexEdgeHuge7[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndexEdgeHuge7[5] * static_cast<int64_t>(tD->outStride[5]);
        inIndexEdgeHuge7[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexEdgeHuge7[i] = min(max(inIndexEdgeHuge7[i], static_cast<int64_t>(tD->leftPad[i])),
                                      static_cast<int64_t>(tD->inShape[i] + static_cast<int64_t>(tD->leftPad[i]) - 1)) -
                                  static_cast<int64_t>(tD->leftPad[i]);
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexEdgeHuge7[0]) * tD->inStride[0] +
                               static_cast<uint64_t>(inIndexEdgeHuge7[1]) * tD->inStride[1] +
                               static_cast<uint64_t>(inIndexEdgeHuge7[2]) * tD->inStride[2] +
                               static_cast<uint64_t>(inIndexEdgeHuge7[3]) * tD->inStride[3] +
                               static_cast<uint64_t>(inIndexEdgeHuge7[4]) * tD->inStride[4] +
                               static_cast<uint64_t>(inIndexEdgeHuge7[5]) * tD->inStride[5] +
                               static_cast<uint64_t>(inIndexEdgeHuge7[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM>
__simt_vf__ LAUNCH_BOUND(EDGE_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeEdgeHugeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                     uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                     uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t m5, uint64_t m6,
                                     uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5,
                                     uint64_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndexEdgeHuge8[DIM] = {0};

        inIndexEdgeHuge8[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexEdgeHuge8[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndexEdgeHuge8[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexEdgeHuge8[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndexEdgeHuge8[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexEdgeHuge8[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndexEdgeHuge8[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexEdgeHuge8[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndexEdgeHuge8[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexEdgeHuge8[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndexEdgeHuge8[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndexEdgeHuge8[5] * static_cast<int64_t>(tD->outStride[5]);
        inIndexEdgeHuge8[6] = Simt::UintDiv(dstIdx, m6, s6);
        dstIdx -= inIndexEdgeHuge8[6] * static_cast<int64_t>(tD->outStride[6]);
        inIndexEdgeHuge8[DIM - 1] = dstIdx;

        for (int32_t i = 0; i < DIM; i++) {
            inIndexEdgeHuge8[i] = min(max(inIndexEdgeHuge8[i], static_cast<int64_t>(tD->leftPad[i])),
                                      static_cast<int64_t>(tD->inShape[i] + static_cast<int64_t>(tD->leftPad[i]) - 1)) -
                                  static_cast<int64_t>(tD->leftPad[i]);
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndexEdgeHuge8[0]) * tD->inStride[0] +
                               static_cast<uint64_t>(inIndexEdgeHuge8[1]) * tD->inStride[1] +
                               static_cast<uint64_t>(inIndexEdgeHuge8[2]) * tD->inStride[2] +
                               static_cast<uint64_t>(inIndexEdgeHuge8[3]) * tD->inStride[3] +
                               static_cast<uint64_t>(inIndexEdgeHuge8[4]) * tD->inStride[4] +
                               static_cast<uint64_t>(inIndexEdgeHuge8[5]) * tD->inStride[5] +
                               static_cast<uint64_t>(inIndexEdgeHuge8[6]) * tD->inStride[6] +
                               static_cast<uint64_t>(inIndexEdgeHuge8[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T>
__aicore__ inline void PadEdgeSimtHuge<T>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD_->dimNum;

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeEdgeHugeDimOne<T>>(dim3(EDGE_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                  (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), mTD_->outShape[0],
                                                  mBlockIdx_, blockNum, mTD_->inShape[0], mTD_->leftPad[0]);
        return;
    }

    uint64_t outputSize = mTD_->outShape[0] * mTD_->outStride[0];

    uint64_t s[8];
    uint64_t m[8];

    for (uint32_t i = 0; i < mDimNum - 1; ++i) {
        GetUintDivMagicAndShift(m[i], s[i], mTD_->outStride[i]);
    }

    if (mDimNum == 2) {
        asc_vf_call<SimtComputeEdgeHugeDimTwo<T, 2>>(
            dim3(EDGE_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->inShape[0], mTD_->inShape[1], m[0], s[0], mTD_->leftPad[0], mTD_->leftPad[1]);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeEdgeHugeDimThree<T, 3>>(
            dim3(EDGE_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], s[0],
            s[1], mTD_->leftPad[0], mTD_->leftPad[1], mTD_->leftPad[2]);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeEdgeHugeDimFour<T, 4>>(dim3(EDGE_HUGE_QUATER_THREAD_DIM),
                                                      (__gm__ T*)(mInputGM_.GetPhyAddr()),
                                                      (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize,
                                                      mBlockIdx_, blockNum, m[0], m[1], m[2], s[0], s[1], s[2]);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeEdgeHugeDimFive<T, 5>>(
            dim3(EDGE_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], s[0], s[1], s[2], s[3]);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeEdgeHugeDimSix<T, 6>>(
            dim3(EDGE_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], s[0], s[1], s[2], s[3], s[4]);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeEdgeHugeDimSeven<T, 7>>(
            dim3(EDGE_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], s[0], s[1], s[2], s[3], s[4], s[5]);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeEdgeHugeDimEight<T, 8>>(
            dim3(EDGE_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], m[6], s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
    }
}
} // namespace PadV3

#endif //  PAD_EDGE_SIMT_HUGE_H
