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
 * \file pad_reflect_simt.h
 * \brief pad_reflect_simt
 */

#ifndef PAD_REFLECT_SIMT_H
#define PAD_REFLECT_SIMT_H

#include "pad_common.h"
#include "pad_v3_struct.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t REFLECT_THREAD_DIM = 512;
#else
constexpr int32_t REFLECT_THREAD_DIM = 2048;
#endif

namespace PadV3 {
using namespace AscendC;

template <typename T, int32_t KEY>
class PadReflectSimt {
public:
    __aicore__ inline PadReflectSimt(){};
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
__aicore__ inline void PadReflectSimt<T, KEY>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,
                                                    const PadACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t outputSize,
                                  uint32_t blockIdx, uint32_t blockNum, uint32_t inShape0, int32_t left0)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        int32_t inIndexMir1 = idx - left0;
        if constexpr (IS_REFLECT) {
            inIndexMir1 = abs(inIndexMir1) - abs(inIndexMir1 - (int32_t(inShape0) - 1)) - inIndexMir1 +
                          int32_t(inShape0) - 1;
        } else {
            inIndexMir1 = (abs(inIndexMir1 * 2 + 1) - abs((inIndexMir1 - (int32_t(inShape0) - 1)) * 2 - 1)) / 2 -
                          inIndexMir1 + int32_t(inShape0) - 1;
        }
        outputGM[idx] = inputGM[inIndexMir1];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t outputSize,
                                  uint32_t blockIdx, uint32_t blockNum, uint32_t outStride0, uint32_t inShape0,
                                  uint32_t inShape1, uint32_t m0, uint32_t s0, int32_t left0, int32_t left1)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndexMir2[DIM] = {0};

        inIndexMir2[0] = Simt::UintDiv(dstIdx, m0, s0);
        inIndexMir2[1] = dstIdx - inIndexMir2[0] * outStride0;

        inIndexMir2[0] -= left0;
        inIndexMir2[1] -= left1;

        if constexpr (IS_REFLECT) {
            inIndexMir2[0] = abs(inIndexMir2[0]) - abs(inIndexMir2[0] - (int32_t(inShape0) - 1)) - inIndexMir2[0] +
                             int32_t(inShape0) - 1;
            inIndexMir2[1] = abs(inIndexMir2[1]) - abs(inIndexMir2[1] - (int32_t(inShape1) - 1)) - inIndexMir2[1] +
                             int32_t(inShape1) - 1;
        } else {
            inIndexMir2[0] = (abs(inIndexMir2[0] * 2 + 1) - abs((inIndexMir2[0] - (int32_t(inShape0) - 1)) * 2 - 1)) /
                                 2 -
                             inIndexMir2[0] + int32_t(inShape0) - 1;
            inIndexMir2[1] = (abs(inIndexMir2[1] * 2 + 1) - abs((inIndexMir2[1] - (int32_t(inShape1) - 1)) * 2 - 1)) /
                                 2 -
                             inIndexMir2[1] + int32_t(inShape1) - 1;
        }

        uint32_t inputOffset = uint32_t(inIndexMir2[0]) * inShape1 + uint32_t(inIndexMir2[1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t outputSize,
                                    uint32_t blockIdx, uint32_t blockNum, uint32_t outStride0, uint32_t outStride1,
                                    uint32_t inShape0, uint32_t inShape1, uint32_t inShape2, uint32_t m0, uint32_t m1,
                                    uint32_t s0, uint32_t s1, int32_t left0, int32_t left1, int32_t left2)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndexMir3[DIM] = {0};

        inIndexMir3[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexMir3[0] * outStride0;
        inIndexMir3[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexMir3[1] * outStride1;
        inIndexMir3[DIM - 1] = dstIdx;

        inIndexMir3[0] -= left0;
        inIndexMir3[1] -= left1;
        inIndexMir3[2] -= left2;

        if constexpr (IS_REFLECT) {
            inIndexMir3[0] = abs(inIndexMir3[0]) - abs(inIndexMir3[0] - (int32_t(inShape0) - 1)) - inIndexMir3[0] +
                             int32_t(inShape0) - 1;
            inIndexMir3[1] = abs(inIndexMir3[1]) - abs(inIndexMir3[1] - (int32_t(inShape1) - 1)) - inIndexMir3[1] +
                             int32_t(inShape1) - 1;
            inIndexMir3[DIM - 1] = abs(inIndexMir3[DIM - 1]) - abs(inIndexMir3[DIM - 1] - (int32_t(inShape2) - 1)) -
                                   inIndexMir3[DIM - 1] + int32_t(inShape2) - 1;
        } else {
            inIndexMir3[0] = (abs(inIndexMir3[0] * 2 + 1) - abs((inIndexMir3[0] - (int32_t(inShape0) - 1)) * 2 - 1)) /
                                 2 -
                             inIndexMir3[0] + int32_t(inShape0) - 1;
            inIndexMir3[1] = (abs(inIndexMir3[1] * 2 + 1) - abs((inIndexMir3[1] - (int32_t(inShape1) - 1)) * 2 - 1)) /
                                 2 -
                             inIndexMir3[1] + int32_t(inShape1) - 1;
            inIndexMir3[DIM - 1] = (abs(inIndexMir3[DIM - 1] * 2 + 1) -
                                    abs((inIndexMir3[DIM - 1] - (int32_t(inShape2) - 1)) * 2 - 1)) /
                                       2 -
                                   inIndexMir3[DIM - 1] + int32_t(inShape2) - 1;
        }

        uint32_t inputOffset = uint32_t(inIndexMir3[0]) * inShape1 * inShape2 + uint32_t(inIndexMir3[1]) * inShape2 +
                               uint32_t(inIndexMir3[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                   uint32_t s0, uint32_t s1, uint32_t s2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndexMir4[DIM] = {0};

        inIndexMir4[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexMir4[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndexMir4[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexMir4[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndexMir4[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexMir4[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndexMir4[DIM - 1] = dstIdx;
        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir4[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir4[i] = abs(inIndexMir4[i]) -
                                 abs(inIndexMir4[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) - inIndexMir4[i] +
                                 static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir4[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir4[i] = (abs(inIndexMir4[i] * 2 + 1) -
                                  abs((inIndexMir4[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                     2 -
                                 inIndexMir4[i] + static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        }
        uint32_t inputOffset = static_cast<uint32_t>(inIndexMir4[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndexMir4[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndexMir4[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndexMir4[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                   uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                   uint32_t m3, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndexMir5[DIM] = {0};

        inIndexMir5[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexMir5[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndexMir5[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexMir5[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndexMir5[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexMir5[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndexMir5[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexMir5[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndexMir5[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir5[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir5[i] = abs(inIndexMir5[i]) -
                                 abs(inIndexMir5[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) - inIndexMir5[i] +
                                 static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir5[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir5[i] = (abs(inIndexMir5[i] * 2 + 1) -
                                  abs((inIndexMir5[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                     2 -
                                 inIndexMir5[i] + static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        }

        uint32_t inputOffset = static_cast<uint32_t>(inIndexMir5[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndexMir5[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndexMir5[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndexMir5[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndexMir5[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                  uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                  uint32_t m3, uint32_t m4, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
                                  uint32_t s4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndexMir6[DIM] = {0};

        inIndexMir6[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexMir6[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndexMir6[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexMir6[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndexMir6[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexMir6[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndexMir6[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexMir6[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndexMir6[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexMir6[4] * static_cast<int32_t>(tD->outStride[4]);
        inIndexMir6[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir6[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir6[i] = abs(inIndexMir6[i]) -
                                 abs(inIndexMir6[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) - inIndexMir6[i] +
                                 static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir6[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir6[i] = (abs(inIndexMir6[i] * 2 + 1) -
                                  abs((inIndexMir6[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                     2 -
                                 inIndexMir6[i] + static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        }

        uint32_t inputOffset = static_cast<uint32_t>(inIndexMir6[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndexMir6[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndexMir6[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndexMir6[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndexMir6[4]) * static_cast<uint32_t>(tD->inStride[4]) +
                               static_cast<uint32_t>(inIndexMir6[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                    uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                    uint32_t m3, uint32_t m4, uint32_t m5, uint32_t s0, uint32_t s1, uint32_t s2,
                                    uint32_t s3, uint32_t s4, uint32_t s5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndexMir7[DIM] = {0};

        inIndexMir7[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexMir7[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndexMir7[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexMir7[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndexMir7[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexMir7[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndexMir7[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexMir7[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndexMir7[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexMir7[4] * static_cast<int32_t>(tD->outStride[4]);
        inIndexMir7[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndexMir7[5] * static_cast<int32_t>(tD->outStride[5]);
        inIndexMir7[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir7[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir7[i] = abs(inIndexMir7[i]) -
                                 abs(inIndexMir7[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) - inIndexMir7[i] +
                                 static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir7[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir7[i] = (abs(inIndexMir7[i] * 2 + 1) -
                                  abs((inIndexMir7[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                     2 -
                                 inIndexMir7[i] + static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        }

        uint32_t inputOffset = static_cast<uint32_t>(inIndexMir7[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndexMir7[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndexMir7[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndexMir7[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndexMir7[4]) * static_cast<uint32_t>(tD->inStride[4]) +
                               static_cast<uint32_t>(inIndexMir7[5]) * static_cast<uint32_t>(tD->inStride[5]) +
                               static_cast<uint32_t>(inIndexMir7[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t DIM, bool IS_REFLECT>
__simt_vf__ LAUNCH_BOUND(REFLECT_THREAD_DIM) __aicore__
    void SimtComputeReflectDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, uint32_t outputSize,
                                    uint32_t blockIdx, uint32_t blockNum, uint32_t m0, uint32_t m1, uint32_t m2,
                                    uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6, uint32_t s0, uint32_t s1,
                                    uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5, uint32_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t dstIdx = idx;
        int32_t inIndexMir8[DIM] = {0};

        inIndexMir8[0] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= inIndexMir8[0] * static_cast<int32_t>(tD->outStride[0]);
        inIndexMir8[1] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= inIndexMir8[1] * static_cast<int32_t>(tD->outStride[1]);
        inIndexMir8[2] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= inIndexMir8[2] * static_cast<int32_t>(tD->outStride[2]);
        inIndexMir8[3] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= inIndexMir8[3] * static_cast<int32_t>(tD->outStride[3]);
        inIndexMir8[4] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= inIndexMir8[4] * static_cast<int32_t>(tD->outStride[4]);
        inIndexMir8[5] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= inIndexMir8[5] * static_cast<int32_t>(tD->outStride[5]);
        inIndexMir8[6] = Simt::UintDiv(dstIdx, m6, s6);
        dstIdx -= inIndexMir8[6] * static_cast<int32_t>(tD->outStride[6]);
        inIndexMir8[DIM - 1] = dstIdx;

        if constexpr (IS_REFLECT) {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir8[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir8[i] = abs(inIndexMir8[i]) -
                                 abs(inIndexMir8[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) - inIndexMir8[i] +
                                 static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        } else {
            for (int32_t i = 0; i < DIM; i++) {
                inIndexMir8[i] -= static_cast<int32_t>(tD->leftPad[i]);
                inIndexMir8[i] = (abs(inIndexMir8[i] * 2 + 1) -
                                  abs((inIndexMir8[i] - (static_cast<int32_t>(tD->inShape[i]) - 1)) * 2 - 1)) /
                                     2 -
                                 inIndexMir8[i] + static_cast<int32_t>(tD->inShape[i]) - 1;
            }
        }

        uint32_t inputOffset = static_cast<uint32_t>(inIndexMir8[0]) * static_cast<uint32_t>(tD->inStride[0]) +
                               static_cast<uint32_t>(inIndexMir8[1]) * static_cast<uint32_t>(tD->inStride[1]) +
                               static_cast<uint32_t>(inIndexMir8[2]) * static_cast<uint32_t>(tD->inStride[2]) +
                               static_cast<uint32_t>(inIndexMir8[3]) * static_cast<uint32_t>(tD->inStride[3]) +
                               static_cast<uint32_t>(inIndexMir8[4]) * static_cast<uint32_t>(tD->inStride[4]) +
                               static_cast<uint32_t>(inIndexMir8[5]) * static_cast<uint32_t>(tD->inStride[5]) +
                               static_cast<uint32_t>(inIndexMir8[6]) * static_cast<uint32_t>(tD->inStride[6]) +
                               static_cast<uint32_t>(inIndexMir8[DIM - 1]);
        outputGM[idx] = inputGM[inputOffset];
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void PadReflectSimt<T, KEY>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD_->dimNum;

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeReflectDimOne<T, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), mTD_->outShape[0], mBlockIdx_, blockNum, mTD_->inShape[0],
            mTD_->leftPad[0]);
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
        asc_vf_call<SimtComputeReflectDimTwo<T, 2, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->inShape[0], mTD_->inShape[1], m[0], s[0], mTD_->leftPad[0], mTD_->leftPad[1]);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeReflectDimThree<T, 3, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mTD_->outStride[0],
            mTD_->outStride[1], mTD_->inShape[0], mTD_->inShape[1], mTD_->inShape[2], m[0], m[1], s[0], s[1],
            mTD_->leftPad[0], mTD_->leftPad[1], mTD_->leftPad[2]);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeReflectDimFour<T, 4, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            s[0], s[1], s[2]);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeReflectDimFive<T, 5, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], s[0], s[1], s[2], s[3]);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeReflectDimSix<T, 6, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], s[0], s[1], s[2], s[3], s[4]);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeReflectDimSeven<T, 7, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], s[0], s[1], s[2], s[3], s[4], s[5]);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeReflectDimEight<T, 8, IS_REFLECT>>(
            dim3(REFLECT_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), tiling, outputSize, mBlockIdx_, blockNum, m[0], m[1], m[2],
            m[3], m[4], m[5], m[6], s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
    }
}
} // namespace PadV3

#endif //  PAD_EDGE_SIMT_H
