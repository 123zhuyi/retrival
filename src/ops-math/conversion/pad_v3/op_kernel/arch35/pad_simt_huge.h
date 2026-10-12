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
 * \file pad_simt.h
 * \brief pad_simt
 */

#ifndef PAD_SIMT_HUGE
#define PAD_SIMT_HUGE

#include "pad_common.h"
#include "pad_simt.h"
#include "pad_v3_struct.h"
#include "simt_api/asc_simt.h"

namespace PadV3 {
using namespace AscendC;

template <typename T, typename U = int8_t>
class PadSimtHuge {
public:
    __aicore__ inline PadSimtHuge(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData,
                                GM_ADDR constValue = nullptr);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    GlobalTensor<T> mInputGMHuge;  // GM x huge
    GlobalTensor<T> mOutputGMHuge; // GM y huge
    GlobalTensor<T> constValueGM_Huge;
    T constValue_{0}; // huge 默认值

    uint32_t mBlockIdx;         // huge 核号
    const PadACTilingData* mTD; // huge tilingData
};

template <typename T, typename U>
__aicore__ inline void PadSimtHuge<T, U>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y,
                                               const PadACTilingData* tilingData, GM_ADDR constValue)
{
    mBlockIdx = GetBlockIdx();
    mTD = tilingData;
    mInputGMHuge.SetGlobalBuffer((__gm__ T*)x);
    mOutputGMHuge.SetGlobalBuffer((__gm__ T*)y);

    if (constValue != nullptr) {
        constValueGM_Huge.SetGlobalBuffer((__gm__ T*)constValue);
        constValue_ = constValueGM_Huge(0);
        if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
            T padValue = constValueGM_Huge(0);
            T low4bit = padValue & 0x0F;
            constValue_ = (low4bit << 4) | low4bit;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeHugeDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, T fillValue, uint64_t outputSize,
                               uint32_t blockIdx, uint32_t blockNum, uint64_t inShape0, int64_t left0)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        int64_t dimOneInIndex0 = idx - left0;

        if (dimOneInIndex0 >= 0 && dimOneInIndex0 < inShape0) {
            outputGM[idx] = inputGM[dimOneInIndex0];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeHugeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, T fillValue, uint64_t outputSize,
                               uint32_t blockIdx, uint32_t blockNum, uint64_t outShape1, uint64_t inShape0,
                               uint64_t inShape1, uint64_t m1, uint64_t s1, int64_t left0, int64_t left1)
{
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        // uint64_t yIdx = idx;     // idx 并未修改

        uint64_t dimTwoIndex1 = idx - Simt::UintDiv(idx, m1, s1) * outShape1;
        uint64_t dimTwoIndex0 = Simt::UintDiv(idx, m1, s1);

        int64_t dimTwoInIndex1 = dimTwoIndex1 - left1;
        int64_t dimTwoInIndex0 = dimTwoIndex0 - left0;

        if (dimTwoInIndex1 >= 0 && dimTwoInIndex1 < inShape1 && dimTwoInIndex0 >= 0 && dimTwoInIndex0 < inShape0) {
            outputGM[idx] = inputGM[dimTwoInIndex0 * inShape1 + dimTwoInIndex1];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeHugeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                                 uint32_t blockIdx, uint32_t blockNum, uint64_t inShape0, uint64_t inShape1,
                                 uint64_t inShape2, uint64_t m1, uint64_t m2, uint64_t s1, uint64_t s2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint64_t outputSize = tD->outShape[0] * tD->outStride[0];
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t yIdx = idx; // 采用 idx 会被修改

        uint64_t dimThreeIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * tD->outShape[2];
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint64_t dimThreeIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * tD->outShape[1];
        uint64_t dimThreeIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int64_t dimThreeInIndex2 = dimThreeIndex2 - tD->leftPad[2];
        int64_t dimThreeInIndex1 = dimThreeIndex1 - tD->leftPad[1];
        int64_t dimThreeInIndex0 = dimThreeIndex0 - tD->leftPad[0];

        if (dimThreeInIndex2 >= 0 && dimThreeInIndex2 < inShape2 && dimThreeInIndex1 >= 0 &&
            dimThreeInIndex1 < inShape1 && dimThreeInIndex0 >= 0 && dimThreeInIndex0 < inShape0) {
            outputGM[idx] = inputGM[dimThreeInIndex0 * inShape1 * inShape2 + dimThreeInIndex1 * inShape2 +
                                    dimThreeInIndex2];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeHugeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                                uint32_t blockIdx, uint32_t blockNum, uint64_t m1, uint64_t m2, uint64_t m3,
                                uint64_t s1, uint64_t s2, uint64_t s3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint64_t outputSize = tD->outShape[0] * tD->outStride[0];
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t yIdx = idx; // 保存原始输出索引

        uint64_t dimFourIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * tD->outShape[3];
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint64_t dimFourIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * tD->outShape[2];
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint64_t dimFourIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * tD->outShape[1];
        uint64_t dimFourIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int64_t dimFourInIndex3 = dimFourIndex3 - tD->leftPad[3];
        int64_t dimFourInIndex2 = dimFourIndex2 - tD->leftPad[2];
        int64_t dimFourInIndex1 = dimFourIndex1 - tD->leftPad[1];
        int64_t dimFourInIndex0 = dimFourIndex0 - tD->leftPad[0];

        if (dimFourInIndex3 >= 0 && dimFourInIndex3 < tD->inShape[3] && dimFourInIndex2 >= 0 &&
            dimFourInIndex2 < tD->inShape[2] && dimFourInIndex1 >= 0 && dimFourInIndex1 < tD->inShape[1] &&
            dimFourInIndex0 >= 0 && dimFourInIndex0 < tD->inShape[0]) {
            outputGM[idx] = inputGM[dimFourInIndex0 * tD->inShape[1] * tD->inShape[2] * tD->inShape[3] +
                                    dimFourInIndex1 * tD->inShape[2] * tD->inShape[3] +
                                    dimFourInIndex2 * tD->inShape[3] + dimFourInIndex3];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeHugeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                                uint32_t blockIdx, uint32_t blockNum, uint64_t m1, uint64_t m2, uint64_t m3,
                                uint64_t m4, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint64_t outputSize = tD->outShape[0] * tD->outStride[0];
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t yIdx = idx; // 保存原始输出索引

        uint64_t dimFiveIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * tD->outShape[4];
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint64_t dimFiveIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * tD->outShape[3];
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint64_t dimFiveIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * tD->outShape[2];
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint64_t dimFiveIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * tD->outShape[1];
        uint64_t dimFiveIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int64_t dimFiveInIndex4 = dimFiveIndex4 - tD->leftPad[4];
        int64_t dimFiveInIndex3 = dimFiveIndex3 - tD->leftPad[3];
        int64_t dimFiveInIndex2 = dimFiveIndex2 - tD->leftPad[2];
        int64_t dimFiveInIndex1 = dimFiveIndex1 - tD->leftPad[1];
        int64_t dimFiveInIndex0 = dimFiveIndex0 - tD->leftPad[0];

        if (dimFiveInIndex4 >= 0 && dimFiveInIndex4 < tD->inShape[4] && dimFiveInIndex3 >= 0 &&
            dimFiveInIndex3 < tD->inShape[3] && dimFiveInIndex2 >= 0 && dimFiveInIndex2 < tD->inShape[2] &&
            dimFiveInIndex1 >= 0 && dimFiveInIndex1 < tD->inShape[1] && dimFiveInIndex0 >= 0 &&
            dimFiveInIndex0 < tD->inShape[0]) {
            outputGM[idx] = inputGM[dimFiveInIndex0 * tD->inShape[1] * tD->inShape[2] * tD->inShape[3] *
                                        tD->inShape[4] +
                                    dimFiveInIndex1 * tD->inShape[2] * tD->inShape[3] * tD->inShape[4] +
                                    dimFiveInIndex2 * tD->inShape[3] * tD->inShape[4] +
                                    dimFiveInIndex3 * tD->inShape[4] + dimFiveInIndex4];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(HALF_THREAD_DIM) __aicore__
    void SimtComputeHugeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                               uint32_t blockIdx, uint32_t blockNum, uint64_t m1, uint64_t m2, uint64_t m3, uint64_t m4,
                               uint64_t m5, uint64_t s1, uint64_t s2, uint64_t s3, uint64_t s4, uint64_t s5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint64_t outputSize = tD->outShape[0] * tD->outStride[0];
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t yIdx = idx; // 保存原始输出索引

        uint64_t dimSixIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * tD->outShape[5];
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint64_t dimSixIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * tD->outShape[4];
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint64_t dimSixIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * tD->outShape[3];
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint64_t dimSixIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * tD->outShape[2];
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint64_t dimSixIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * tD->outShape[1];
        uint64_t dimSixIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int64_t dimSixInIndex5 = dimSixIndex5 - tD->leftPad[5];
        int64_t dimSixInIndex4 = dimSixIndex4 - tD->leftPad[4];
        int64_t dimSixInIndex3 = dimSixIndex3 - tD->leftPad[3];
        int64_t dimSixInIndex2 = dimSixIndex2 - tD->leftPad[2];
        int64_t dimSixInIndex1 = dimSixIndex1 - tD->leftPad[1];
        int64_t dimSixInIndex0 = dimSixIndex0 - tD->leftPad[0];

        if (dimSixInIndex5 >= 0 && dimSixInIndex5 < tD->inShape[5] && dimSixInIndex4 >= 0 &&
            dimSixInIndex4 < tD->inShape[4] && dimSixInIndex3 >= 0 && dimSixInIndex3 < tD->inShape[3] &&
            dimSixInIndex2 >= 0 && dimSixInIndex2 < tD->inShape[2] && dimSixInIndex1 >= 0 &&
            dimSixInIndex1 < tD->inShape[1] && dimSixInIndex0 >= 0 && dimSixInIndex0 < tD->inShape[0]) {
            outputGM[idx] = inputGM[dimSixInIndex0 * tD->inShape[1] * tD->inShape[2] * tD->inShape[3] * tD->inShape[4] *
                                        tD->inShape[5] +
                                    dimSixInIndex1 * tD->inShape[2] * tD->inShape[3] * tD->inShape[4] * tD->inShape[5] +
                                    dimSixInIndex2 * tD->inShape[3] * tD->inShape[4] * tD->inShape[5] +
                                    dimSixInIndex3 * tD->inShape[4] * tD->inShape[5] + dimSixInIndex4 * tD->inShape[5] +
                                    dimSixInIndex5];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(HALF_THREAD_DIM) __aicore__
    void SimtComputeHugeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                                 uint32_t blockIdx, uint32_t blockNum, uint64_t m1, uint64_t m2, uint64_t m3,
                                 uint64_t m4, uint64_t m5, uint64_t m6, uint64_t s1, uint64_t s2, uint64_t s3,
                                 uint64_t s4, uint64_t s5, uint64_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint64_t outputSize = tD->outShape[0] * tD->outStride[0];
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t yIdx = idx; // 保存原始输出索引

        uint64_t dimSevenIndex6 = yIdx - Simt::UintDiv(yIdx, m6, s6) * tD->outShape[6];
        yIdx = Simt::UintDiv(yIdx, m6, s6);
        uint64_t dimSevenIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * tD->outShape[5];
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint64_t dimSevenIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * tD->outShape[4];
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint64_t dimSevenIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * tD->outShape[3];
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint64_t dimSevenIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * tD->outShape[2];
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint64_t dimSevenIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * tD->outShape[1];
        uint64_t dimSevenIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int64_t dimSevenInIndex6 = dimSevenIndex6 - tD->leftPad[6];
        int64_t dimSevenInIndex5 = dimSevenIndex5 - tD->leftPad[5];
        int64_t dimSevenInIndex4 = dimSevenIndex4 - tD->leftPad[4];
        int64_t dimSevenInIndex3 = dimSevenIndex3 - tD->leftPad[3];
        int64_t dimSevenInIndex2 = dimSevenIndex2 - tD->leftPad[2];
        int64_t dimSevenInIndex1 = dimSevenIndex1 - tD->leftPad[1];
        int64_t dimSevenInIndex0 = dimSevenIndex0 - tD->leftPad[0];

        if (dimSevenInIndex6 >= 0 && dimSevenInIndex6 < tD->inShape[6] && dimSevenInIndex5 >= 0 &&
            dimSevenInIndex5 < tD->inShape[6] && dimSevenInIndex4 >= 0 && dimSevenInIndex4 < tD->inShape[4] &&
            dimSevenInIndex3 >= 0 && dimSevenInIndex3 < tD->inShape[3] && dimSevenInIndex2 >= 0 &&
            dimSevenInIndex2 < tD->inShape[2] && dimSevenInIndex1 >= 0 && dimSevenInIndex1 < tD->inShape[1] &&
            dimSevenInIndex0 >= 0 && dimSevenInIndex0 < tD->inShape[0]) {
            outputGM[idx] = inputGM[dimSevenInIndex0 * tD->inShape[1] * tD->inShape[2] * tD->inShape[3] *
                                        tD->inShape[4] * tD->inShape[5] * tD->inShape[6] +
                                    dimSevenInIndex1 * tD->inShape[2] * tD->inShape[3] * tD->inShape[4] *
                                        tD->inShape[5] * tD->inShape[6] +
                                    dimSevenInIndex2 * tD->inShape[3] * tD->inShape[4] * tD->inShape[5] *
                                        tD->inShape[6] +
                                    dimSevenInIndex3 * tD->inShape[4] * tD->inShape[5] * tD->inShape[6] +
                                    dimSevenInIndex4 * tD->inShape[5] * tD->inShape[6] +
                                    dimSevenInIndex5 * tD->inShape[6] + dimSevenInIndex6];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(AN_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeHugeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                                 uint32_t blockIdx, uint32_t blockNum, uint64_t m1, uint64_t m2, uint64_t m3,
                                 uint64_t m4, uint64_t m5, uint64_t m6, uint64_t m7, uint64_t s1, uint64_t s2,
                                 uint64_t s3, uint64_t s4, uint64_t s5, uint64_t s6, uint64_t s7)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint64_t outputSize = tD->outShape[0] * tD->outStride[0];

    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint64_t yIdx = idx; // 保存原始输出索引

        uint64_t dimEightIndex7 = yIdx - Simt::UintDiv(yIdx, m7, s7) * tD->outShape[7];
        yIdx = Simt::UintDiv(yIdx, m7, s7);
        uint64_t dimEightIndex6 = yIdx - Simt::UintDiv(yIdx, m6, s6) * tD->outShape[6];
        yIdx = Simt::UintDiv(yIdx, m6, s6);
        uint64_t dimEightIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * tD->outShape[5];
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint64_t dimEightIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * tD->outShape[4];
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint64_t dimEightIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * tD->outShape[3];
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint64_t dimEightIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * tD->outShape[2];
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint64_t dimEightIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * tD->outShape[1];
        uint64_t dimEightIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int64_t dimEightInIndex7 = dimEightIndex7 - tD->leftPad[7];
        int64_t dimEightInIndex6 = dimEightIndex6 - tD->leftPad[6];
        int64_t dimEightInIndex5 = dimEightIndex5 - tD->leftPad[5];
        int64_t dimEightInIndex4 = dimEightIndex4 - tD->leftPad[4];
        int64_t dimEightInIndex3 = dimEightIndex3 - tD->leftPad[3];
        int64_t dimEightInIndex2 = dimEightIndex2 - tD->leftPad[2];
        int64_t dimEightInIndex1 = dimEightIndex1 - tD->leftPad[1];
        int64_t dimEightInIndex0 = dimEightIndex0 - tD->leftPad[0];

        if (dimEightInIndex7 >= 0 && dimEightInIndex7 < tD->inShape[7] && dimEightInIndex6 >= 0 &&
            dimEightInIndex6 < tD->inShape[6] && dimEightInIndex5 >= 0 && dimEightInIndex5 < tD->inShape[5] &&
            dimEightInIndex4 >= 0 && dimEightInIndex4 < tD->inShape[4] && dimEightInIndex3 >= 0 &&
            dimEightInIndex3 < tD->inShape[3] && dimEightInIndex2 >= 0 && dimEightInIndex2 < tD->inShape[2] &&
            dimEightInIndex1 >= 0 && dimEightInIndex1 < tD->inShape[1] && dimEightInIndex0 >= 0 &&
            dimEightInIndex0 < tD->inShape[0]) {
            outputGM[idx] = inputGM[dimEightInIndex0 * tD->inShape[1] * tD->inShape[2] * tD->inShape[3] *
                                        tD->inShape[4] * tD->inShape[5] * tD->inShape[6] * tD->inShape[7] +
                                    dimEightInIndex1 * tD->inShape[2] * tD->inShape[3] * tD->inShape[4] *
                                        tD->inShape[5] * tD->inShape[6] * tD->inShape[7] +
                                    dimEightInIndex2 * tD->inShape[3] * tD->inShape[4] * tD->inShape[5] *
                                        tD->inShape[6] * tD->inShape[7] +
                                    dimEightInIndex3 * tD->inShape[4] * tD->inShape[5] * tD->inShape[6] *
                                        tD->inShape[7] +
                                    dimEightInIndex4 * tD->inShape[5] * tD->inShape[6] * tD->inShape[7] +
                                    dimEightInIndex5 * tD->inShape[6] * tD->inShape[7] +
                                    dimEightInIndex6 * tD->inShape[7] + dimEightInIndex7];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void PadSimtHuge<T, U>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数

    if (mBlockIdx >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD->dimNum; // 维度数
    uint64_t outputSize = mTD->outShape[0] * mTD->outStride[0];

    uint64_t outShape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t inShape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t leftPad[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (uint32_t i = 0; i < mDimNum; ++i) {
        outShape[i] = mTD->outShape[i];
        inShape[i] = mTD->inShape[i];
        leftPad[i] = mTD->leftPad[i];
    }

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeHugeDimOne<T>>(dim3(THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()),
                                              (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()), constValue_, outputSize,
                                              mBlockIdx, blockNum, inShape[0], leftPad[0]);
        return;
    }

    uint64_t s[8];
    uint64_t m[8];

    for (uint32_t i = 1; i < mDimNum; ++i) {
        GetUintDivMagicAndShift(m[i], s[i], outShape[i]);
    }

    if (mDimNum == 2) {
        asc_vf_call<SimtComputeHugeDimTwo<T>>(dim3(THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()),
                                              (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()), constValue_, outputSize,
                                              mBlockIdx, blockNum, outShape[1], inShape[0], inShape[1], m[1], s[1],
                                              leftPad[0], leftPad[1]);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeHugeDimThree<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()), (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()),
            tiling, constValue_, mBlockIdx, blockNum, inShape[0], inShape[1], inShape[2], m[1], m[2], s[1], s[2]);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeHugeDimFour<T>>(dim3(THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()),
                                               (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()), tiling, constValue_,
                                               mBlockIdx, blockNum, m[1], m[2], m[3], s[1], s[2], s[3]);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeHugeDimFive<T>>(dim3(THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()),
                                               (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()), tiling, constValue_,
                                               mBlockIdx, blockNum, m[1], m[2], m[3], m[4], s[1], s[2], s[3], s[4]);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeHugeDimSix<T>>(dim3(HALF_THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()),
                                              (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()), tiling, constValue_,
                                              mBlockIdx, blockNum, m[1], m[2], m[3], m[4], m[5], s[1], s[2], s[3], s[4],
                                              s[5]);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeHugeDimSeven<T>>(dim3(HALF_THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()),
                                                (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()), tiling, constValue_,
                                                mBlockIdx, blockNum, m[1], m[2], m[3], m[4], m[5], m[6], s[1], s[2],
                                                s[3], s[4], s[5], s[6]);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeHugeDimEight<T>>(dim3(AN_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGMHuge.GetPhyAddr()),
                                                (__gm__ volatile T*)(mOutputGMHuge.GetPhyAddr()), tiling, constValue_,
                                                mBlockIdx, blockNum, m[1], m[2], m[3], m[4], m[5], m[6], m[7], s[1],
                                                s[2], s[3], s[4], s[5], s[6], s[7]);
    }
}
} // namespace PadV3

#endif //  PAD_SIMT_HUGE
