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
 * \file pad_reflect_simt_huge.h
 * \brief pad_reflect_simt_huge
 */

#ifndef PAD_REFLECT_SIMT_HUGE_H
#define PAD_REFLECT_SIMT_HUGE_H

#include "pad_common.h"
#include "pad_v3_struct.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t REFLECT_HUGE_THREAD_DIM = 512;
#else
constexpr int32_t REFLECT_HUGE_THREAD_DIM = 2048;
constexpr int32_t REFLECT_HUGE_HALF_THREAD_DIM = 1024;
constexpr int32_t REFLECT_HUGE_QUATER_THREAD_DIM = 512;
constexpr int32_t REFLECT_HUGE_AN_EIGHTH_THREAD_DIM = 256;
#endif

namespace PadV3 {
using namespace AscendC;

template <typename T, int32_t KEY>
class PadReflectSimtHuge {
public:
    __aicore__ inline PadReflectSimtHuge(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    GlobalTensor<T> mInputGM_;  // GM x
    GlobalTensor<T> mOutputGM_; // GM y

    uint32_t mBlockIdx_;                                       // 核号
    const PadACTilingData* mTD_;                               // tilingData
    constexpr static bool IS_REFLECT = (KEY / 1000) % 10 == 1; // TilingKey倒数第四维标识reflect还是symmetric
};

template <typename T, int32_t KEY>
__aicore__ inline void PadReflectSimtHuge<T, KEY>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,
                                                        const PadACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                      uint32_t blockIdx, uint32_t blockNum, uint64_t inShape0, int64_t left0)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        int64_t inIndex0 = idx;

        if constexpr (IS_REFLECT) {
            inIndex0 = abs(inIndex0 - left0) - abs(inIndex0 - (int64_t(inShape0) + left0 - 1)) - inIndex0 + left0 +
                       int64_t(inShape0) - 1;
        } else {
            inIndex0 = (abs((inIndex0 - left0) * 2 + 1) - abs((inIndex0 - (int64_t(inShape0) + left0 - 1)) * 2 - 1)) /
                           2 -
                       inIndex0 + left0 + int64_t(inShape0) - 1;
        }
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                      uint32_t blockIdx, uint32_t blockNum, uint64_t outStride0, uint64_t inShape0,
                                      uint64_t inShape1, uint64_t m0, uint64_t s0, int64_t left0, int64_t left1)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        inIndex[1] = dstIdx - inIndex[0] * outStride0;

        inIndex[0] -= left0;
        inIndex[1] -= left1;

        if constexpr (IS_REFLECT) {
            inIndex[0] = abs(inIndex[0]) - abs(inIndex[0] - (int64_t(inShape0) - 1)) - inIndex[0] + int64_t(inShape0) -
                         1;
            inIndex[1] = abs(inIndex[1]) - abs(inIndex[1] - (int64_t(inShape1) - 1)) - inIndex[1] + int64_t(inShape1) -
                         1;
        } else {
            inIndex[0] = (abs(inIndex[0] * 2 + 1) - abs((inIndex[0] - (int64_t(inShape0) - 1)) * 2 - 1)) / 2 -
                         inIndex[0] + int64_t(inShape0) - 1;
            inIndex[1] = (abs(inIndex[1] * 2 + 1) - abs((inIndex[1] - (int64_t(inShape1) - 1)) * 2 - 1)) / 2 -
                         inIndex[1] + int64_t(inShape1) - 1;
        }

        uint64_t inputOffset = uint64_t(inIndex[0]) * inShape1 + uint64_t(inIndex[1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint64_t outputSize,
                                        uint32_t blockIdx, uint32_t blockNum, uint64_t outStride0, uint64_t outStride1,
                                        uint64_t inShape0, uint64_t inShape1, uint64_t inShape2, uint64_t m0,
                                        uint64_t m1, uint64_t s0, uint64_t s1, int64_t left0, int64_t left1,
                                        int64_t left2)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * outStride0;
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * outStride1;
        inIndex[DIM - 1] = dstIdx;

        inIndex[0] -= left0;
        inIndex[1] -= left1;
        inIndex[DIM - 1] -= left2;

        if constexpr (IS_REFLECT) {
            inIndex[0] = abs(inIndex[0]) - abs(inIndex[0] - (int64_t(inShape0) - 1)) - inIndex[0] + int64_t(inShape0) -
                         1;
            inIndex[1] = abs(inIndex[1]) - abs(inIndex[1] - (int64_t(inShape1) - 1)) - inIndex[1] + int64_t(inShape1) -
                         1;
            inIndex[DIM - 1] = abs(inIndex[DIM - 1]) - abs(inIndex[DIM - 1] - (int64_t(inShape2) - 1)) -
                               inIndex[DIM - 1] + int64_t(inShape2) - 1;
        } else {
            inIndex[0] = (abs(inIndex[0] * 2 + 1) - abs((inIndex[0] - (int64_t(inShape0) - 1)) * 2 - 1)) / 2 -
                         inIndex[0] + int64_t(inShape0) - 1;
            inIndex[1] = (abs(inIndex[1] * 2 + 1) - abs((inIndex[1] - (int64_t(inShape1) - 1)) * 2 - 1)) / 2 -
                         inIndex[1] + int64_t(inShape1) - 1;
            inIndex[DIM - 1] = (abs(inIndex[DIM - 1] * 2 + 1) -
                                abs((inIndex[DIM - 1] - (int64_t(inShape2) - 1)) * 2 - 1)) /
                                   2 -
                               inIndex[DIM - 1] + int64_t(inShape2) - 1;
        }

        uint64_t inputOffset = uint64_t(inIndex[0]) * inShape1 * inShape2 + uint64_t(inIndex[1]) * inShape2 +
                               uint64_t(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                       uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                       uint64_t m1, uint64_t m2, uint64_t s0, uint64_t s1, uint64_t s2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndex[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = abs(inIndex[i]) - abs(inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = (abs(inIndex[i] * 2 + 1) -
                              abs((inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                 2 -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndex[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndex[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndex[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                       uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                       uint64_t m1, uint64_t m2, uint64_t m3, uint64_t s0, uint64_t s1, uint64_t s2,
                                       uint64_t s3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndex[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = abs(inIndex[i]) - abs(inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = (abs(inIndex[i] * 2 + 1) -
                              abs((inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                 2 -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndex[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndex[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndex[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndex[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                      uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                      uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t s0, uint64_t s1,
                                      uint64_t s2, uint64_t s3, uint64_t s4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndex[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndex[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndex[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = abs(inIndex[i]) - abs(inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = (abs(inIndex[i] * 2 + 1) -
                              abs((inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                 2 -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndex[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndex[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndex[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndex[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndex[4]) * static_cast<uint64_t>(tD->inStride[4]) +
                               static_cast<uint64_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_QUATER_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                        uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                        uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t m5, uint64_t s0,
                                        uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndex[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndex[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndex[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndex[5] * static_cast<int64_t>(tD->outStride[5]);
        inIndex[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = abs(inIndex[i]) - abs(inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = (abs(inIndex[i] * 2 + 1) -
                              abs((inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                 2 -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndex[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndex[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndex[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndex[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndex[4]) * static_cast<uint64_t>(tD->inStride[4]) +
                               static_cast<uint64_t>(inIndex[5]) * static_cast<uint64_t>(tD->inStride[5]) +
                               static_cast<uint64_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_HUGE_AN_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeReflectHugeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling,
                                        uint64_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint64_t m0,
                                        uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4, uint64_t m5, uint64_t m6,
                                        uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5,
                                        uint64_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t dstIdx = idx;
        int64_t inIndex[DIM] = {0};

        inIndex[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndex[0] * static_cast<int64_t>(tD->outStride[0]);
        inIndex[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndex[1] * static_cast<int64_t>(tD->outStride[1]);
        inIndex[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndex[2] * static_cast<int64_t>(tD->outStride[2]);
        inIndex[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndex[3] * static_cast<int64_t>(tD->outStride[3]);
        inIndex[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndex[4] * static_cast<int64_t>(tD->outStride[4]);
        inIndex[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndex[5] * static_cast<int64_t>(tD->outStride[5]);
        inIndex[6] = Simt::UintDiv(dstIdx, m6, s6);
        dstIdx -= inIndex[6] * static_cast<int64_t>(tD->outStride[6]);
        inIndex[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = abs(inIndex[i]) - abs(inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndex[i] -= static_cast<int64_t>(tD->leftPad[i]);
                inIndex[i] = (abs(inIndex[i] * 2 + 1) -
                              abs((inIndex[i] - (static_cast<int64_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                 2 -
                             inIndex[i] + static_cast<int64_t>(tD->inShape[i]) - 1;
            }
        }

        uint64_t inputOffset = static_cast<uint64_t>(inIndex[0]) * static_cast<uint64_t>(tD->inStride[0]) +
                               static_cast<uint64_t>(inIndex[1]) * static_cast<uint64_t>(tD->inStride[1]) +
                               static_cast<uint64_t>(inIndex[2]) * static_cast<uint64_t>(tD->inStride[2]) +
                               static_cast<uint64_t>(inIndex[3]) * static_cast<uint64_t>(tD->inStride[3]) +
                               static_cast<uint64_t>(inIndex[4]) * static_cast<uint64_t>(tD->inStride[4]) +
                               static_cast<uint64_t>(inIndex[5]) * static_cast<uint64_t>(tD->inStride[5]) +
                               static_cast<uint64_t>(inIndex[6]) * static_cast<uint64_t>(tD->inStride[6]) +
                               static_cast<uint64_t>(inIndex[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void PadReflectSimtHuge<T, KEY>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD_->dimNum;

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeReflectHugeDimOne<T, IS_REFLECT>>(
            dim3(REFLECT_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), mTD_->outShape[0], mBlockIdx_, blockNum, mTD_->inShape[0],
            mTD_->leftPad[0]);
        return;
    }

    uint64_t outputSize = mTD_->outShape[0] * mTD_->outStride[0];

    uint64_t s[8];
    uint64_t m[8];

    for (uint32_t i = 0; i < mDimNum - 1; ++i) {
        GetUintDivMagicAndShift(m[i], s[i], static_cast<uint64_t>(mTD_->outStride[i]));
    }

    if (mDimNum == 2) {
        asc_vf_call<SimtComputeReflectHugeDimTwo<T, 2, IS_REFLECT>>(
            dim3(REFLECT_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->inShape[0], mTD_->inShape[1], m[0], s[0], mTD_->leftPad[0], mTD_->leftPad[1]);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeReflectHugeDimThree<T, 3, IS_REFLECT>>(
            dim3(REFLECT_HUGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->outStride[1], mTD_->inShape[0], mTD_->inShape[1], mTD_->inShape[2], m[0], m[1], s[0], s[1],
            mTD_->leftPad[0], mTD_->leftPad[1], mTD_->leftPad[2]);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeReflectHugeDimFour<T, 4, IS_REFLECT>>(
            dim3(REFLECT_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            s[0], s[1], s[2]);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeReflectHugeDimFive<T, 5, IS_REFLECT>>(
            dim3(REFLECT_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], s[0], s[1], s[2], s[3]);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeReflectHugeDimSix<T, 6, IS_REFLECT>>(
            dim3(REFLECT_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], s[0], s[1], s[2], s[3], s[4]);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeReflectHugeDimSeven<T, 7, IS_REFLECT>>(
            dim3(REFLECT_HUGE_QUATER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], s[0], s[1], s[2], s[3], s[4], s[5]);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeReflectHugeDimEight<T, 8, IS_REFLECT>>(
            dim3(REFLECT_HUGE_AN_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], m[6], s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
    }
}
} // namespace PadV3

#endif //  PAD_REFLECT_SIMT_HUGE_H
