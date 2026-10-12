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

#ifndef PAD_SIMT_H
#define PAD_SIMT_H

#include "pad_common.h"
#include "pad_v3_struct.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t THREAD_DIM = 512;
constexpr int32_t HALF_THREAD_DIM = 512;
constexpr int32_t QUARTER_THREAD_DIM = 512;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256;
#else
constexpr int32_t THREAD_DIM = 2048;
constexpr int32_t HALF_THREAD_DIM = 1024;
constexpr int32_t QUARTER_THREAD_DIM = 512;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256;
#endif

namespace PadV3 {
using namespace AscendC;

template <typename T, typename U = int8_t>
class PadSimt {
public:
    __aicore__ inline PadSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData,
                                GM_ADDR constValue = nullptr);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    GlobalTensor<T> mInputGM;  // GM x
    GlobalTensor<T> mOutputGM; // GM y
    GlobalTensor<T> constValueGM_;
    T constValue_{0}; // 默认值

    uint32_t mBlockIdx;         // 核号
    const PadACTilingData* mTD; // tilingData
};

template <typename T, typename U>
__aicore__ inline void PadSimt<T, U>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData,
                                           GM_ADDR constValue)
{
    mBlockIdx = GetBlockIdx();
    mTD = tilingData;
    mInputGM.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM.SetGlobalBuffer((__gm__ T*)y);

    if (constValue != nullptr) {
        constValueGM_.SetGlobalBuffer((__gm__ T*)constValue);
        constValue_ = constValueGM_(0);
        if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
            T padValue = constValueGM_(0);
            T low4bit = padValue & 0x0F;
            constValue_ = (low4bit << 4) | low4bit;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, T fillValue, uint32_t outputSize,
                           uint32_t blockIdx, uint32_t blockNum, uint32_t inShape0, int32_t left0)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        int32_t inIndex0 = idx - left0;

        if (inIndex0 >= 0 && inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, T fillValue, uint32_t outputSize,
                           uint32_t blockIdx, uint32_t blockNum, uint32_t outShape1, uint32_t inShape0,
                           uint32_t inShape1, uint32_t m1, uint32_t s1, int32_t left0, int32_t left1)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t Index1 = idx - Simt::UintDiv(idx, m1, s1) * outShape1;
        uint32_t Index0 = Simt::UintDiv(idx, m1, s1);

        int32_t inIndex1 = Index1 - left1;
        int32_t inIndex0 = Index0 - left0;

        if (inIndex1 >= 0 && inIndex1 < inShape1 && inIndex0 >= 0 && inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0 * inShape1 + inIndex1];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, T fillValue, uint32_t outputSize,
                             uint32_t blockIdx, uint32_t blockNum, uint32_t outShape1, uint32_t outShape2,
                             uint32_t inShape0, uint32_t inShape1, uint32_t inShape2, uint32_t m1, uint32_t m2,
                             uint32_t s1, uint32_t s2, int32_t left0, int32_t left1, int32_t left2)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t yIdx = idx; // 采用 idx 会被修改

        uint32_t Index2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outShape2; // a % b = a - (a / b) * b
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t Index1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outShape1;
        uint32_t Index0 = Simt::UintDiv(yIdx, m1, s1);

        int32_t inIndex2 = Index2 - left2;
        int32_t inIndex1 = Index1 - left1;
        int32_t inIndex0 = Index0 - left0;

        if (inIndex2 >= 0 && inIndex2 < inShape2 && inIndex1 >= 0 && inIndex1 < inShape1 && inIndex0 >= 0 &&
            inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0 * inShape1 * inShape2 + inIndex1 * inShape2 + inIndex2];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, T fillValue, uint32_t outputSize,
                            uint32_t blockIdx, uint32_t blockNum, uint32_t outShape1, uint32_t outShape2,
                            uint32_t outShape3, uint32_t inShape0, uint32_t inShape1, uint32_t inShape2,
                            uint32_t inShape3, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t s1, uint32_t s2,
                            uint32_t s3, int32_t left0, int32_t left1, int32_t left2, int32_t left3)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t yIdx = idx; // 记录原来值

        uint32_t Index3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * outShape3;
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t Index2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outShape2;
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t Index1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outShape1;
        uint32_t Index0 = Simt::UintDiv(yIdx, m1, s1);

        int32_t inIndex3 = Index3 - left3;
        int32_t inIndex2 = Index2 - left2;
        int32_t inIndex1 = Index1 - left1;
        int32_t inIndex0 = Index0 - left0;

        if (inIndex3 >= 0 && inIndex3 < inShape3 && inIndex2 >= 0 && inIndex2 < inShape2 && inIndex1 >= 0 &&
            inIndex1 < inShape1 && inIndex0 >= 0 && inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0 * inShape1 * inShape2 * inShape3 + inIndex1 * inShape2 * inShape3 +
                                    inIndex2 * inShape3 + inIndex3];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, T fillValue, uint32_t outputSize,
                            uint32_t blockIdx, uint32_t blockNum, uint32_t outShape1, uint32_t outShape2,
                            uint32_t outShape3, uint32_t outShape4, uint32_t inShape0, uint32_t inShape1,
                            uint32_t inShape2, uint32_t inShape3, uint32_t inShape4, uint32_t m1, uint32_t m2,
                            uint32_t m3, uint32_t m4, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, int32_t left0,
                            int32_t left1, int32_t left2, int32_t left3, int32_t left4)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t yIdx = idx;

        uint32_t Index4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * outShape4;
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint32_t Index3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * outShape3;
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t Index2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * outShape2;
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t Index1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * outShape1;
        uint32_t Index0 = Simt::UintDiv(yIdx, m1, s1);

        int32_t inIndex4 = Index4 - left4;
        int32_t inIndex3 = Index3 - left3;
        int32_t inIndex2 = Index2 - left2;
        int32_t inIndex1 = Index1 - left1;
        int32_t inIndex0 = Index0 - left0;

        if (inIndex4 >= 0 && inIndex4 < inShape4 && inIndex3 >= 0 && inIndex3 < inShape3 && inIndex2 >= 0 &&
            inIndex2 < inShape2 && inIndex1 >= 0 && inIndex1 < inShape1 && inIndex0 >= 0 && inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0 * inShape1 * inShape2 * inShape3 * inShape4 +
                                    inIndex1 * inShape2 * inShape3 * inShape4 + inIndex2 * inShape3 * inShape4 +
                                    inIndex3 * inShape4 + inIndex4];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                           uint32_t blockIdx, uint32_t blockNum, uint32_t inShape0, uint32_t inShape1,
                           uint32_t inShape2, uint32_t inShape3, uint32_t inShape4, uint32_t inShape5, uint32_t m1,
                           uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t s1, uint32_t s2, uint32_t s3,
                           uint32_t s4, uint32_t s5, int32_t left0, int32_t left1, int32_t left2, int32_t left3,
                           int32_t left4, int32_t left5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint32_t outputSize = tD->outShape[0] * tD->outStride[0];
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t yIdx = idx; // 保存原始输出索引

        uint32_t simtDimSixIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * static_cast<uint32_t>(tD->outShape[5]);
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint32_t simtDimSixIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * static_cast<uint32_t>(tD->outShape[4]);
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint32_t simtDimSixIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * static_cast<uint32_t>(tD->outShape[3]);
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t simtDimSixIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * static_cast<uint32_t>(tD->outShape[2]);
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t simtDimSixIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * static_cast<uint32_t>(tD->outShape[1]);
        uint32_t simtDimSixIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int32_t inIndex5 = simtDimSixIndex5 - left5;
        int32_t inIndex4 = simtDimSixIndex4 - left4;
        int32_t inIndex3 = simtDimSixIndex3 - left3;
        int32_t inIndex2 = simtDimSixIndex2 - left2;
        int32_t inIndex1 = simtDimSixIndex1 - left1;
        int32_t inIndex0 = simtDimSixIndex0 - left0;

        if (inIndex5 >= 0 && inIndex5 < inShape5 && inIndex4 >= 0 && inIndex4 < inShape4 && inIndex3 >= 0 &&
            inIndex3 < inShape3 && inIndex2 >= 0 && inIndex2 < inShape2 && inIndex1 >= 0 && inIndex1 < inShape1 &&
            inIndex0 >= 0 && inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0 * inShape1 * inShape2 * inShape3 * inShape4 * inShape5 +
                                    inIndex1 * inShape2 * inShape3 * inShape4 * inShape5 +
                                    inIndex2 * inShape3 * inShape4 * inShape5 + inIndex3 * inShape4 * inShape5 +
                                    inIndex4 * inShape5 + inIndex5];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                             uint32_t outputSize, uint32_t blockIdx, uint32_t blockNum, uint32_t inShape0,
                             uint32_t inShape1, uint32_t inShape2, uint32_t inShape3, uint32_t inShape4,
                             uint32_t inShape5, uint32_t inShape6, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4,
                             uint32_t m5, uint32_t m6, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                             uint32_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t yIdx = idx; // 保存原始输出索引

        uint32_t simtDimSevenIndex6 = yIdx - Simt::UintDiv(yIdx, m6, s6) * static_cast<uint32_t>(tD->outShape[6]);
        yIdx = Simt::UintDiv(yIdx, m6, s6);
        uint32_t simtDimSevenIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * static_cast<uint32_t>(tD->outShape[5]);
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint32_t simtDimSevenIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * static_cast<uint32_t>(tD->outShape[4]);
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint32_t simtDimSevenIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * static_cast<uint32_t>(tD->outShape[3]);
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t simtDimSevenIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * static_cast<uint32_t>(tD->outShape[2]);
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t simtDimSevenIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * static_cast<uint32_t>(tD->outShape[1]);
        uint32_t simtDimSevenIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int32_t inIndex6 = simtDimSevenIndex6 - static_cast<uint32_t>(tD->leftPad[6]);
        int32_t inIndex5 = simtDimSevenIndex5 - static_cast<uint32_t>(tD->leftPad[5]);
        int32_t inIndex4 = simtDimSevenIndex4 - static_cast<uint32_t>(tD->leftPad[4]);
        int32_t inIndex3 = simtDimSevenIndex3 - static_cast<uint32_t>(tD->leftPad[3]);
        int32_t inIndex2 = simtDimSevenIndex2 - static_cast<uint32_t>(tD->leftPad[2]);
        int32_t inIndex1 = simtDimSevenIndex1 - static_cast<uint32_t>(tD->leftPad[1]);
        int32_t inIndex0 = simtDimSevenIndex0 - static_cast<uint32_t>(tD->leftPad[0]);

        if (inIndex6 >= 0 && inIndex6 < inShape6 && inIndex5 >= 0 && inIndex5 < inShape5 && inIndex4 >= 0 &&
            inIndex4 < inShape4 && inIndex3 >= 0 && inIndex3 < inShape3 && inIndex2 >= 0 && inIndex2 < inShape2 &&
            inIndex1 >= 0 && inIndex1 < inShape1 && inIndex0 >= 0 && inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0 * inShape1 * inShape2 * inShape3 * inShape4 * inShape5 * inShape6 +
                                    inIndex1 * inShape2 * inShape3 * inShape4 * inShape5 * inShape6 +
                                    inIndex2 * inShape3 * inShape4 * inShape5 * inShape6 +
                                    inIndex3 * inShape4 * inShape5 * inShape6 + inIndex4 * inShape5 * inShape6 +
                                    inIndex5 * inShape6 + inIndex6];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, T fillValue,
                             uint32_t blockIdx, uint32_t blockNum, uint32_t inShape0, uint32_t inShape1,
                             uint32_t inShape2, uint32_t inShape3, uint32_t inShape4, uint32_t inShape5,
                             uint32_t inShape6, uint32_t inShape7, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4,
                             uint32_t m5, uint32_t m6, uint32_t m7, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4,
                             uint32_t s5, uint32_t s6, uint32_t s7)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(PadACTilingData, tD, tiling);
    uint32_t outputSize = tD->outShape[0] * tD->outStride[0];
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        uint32_t yIdx = idx; // 保存原始输出索引

        uint32_t simtDimEightIndex7 = yIdx - Simt::UintDiv(yIdx, m7, s7) * static_cast<uint32_t>(tD->outShape[7]);
        yIdx = Simt::UintDiv(yIdx, m7, s7);
        uint32_t simtDimEightIndex6 = yIdx - Simt::UintDiv(yIdx, m6, s6) * static_cast<uint32_t>(tD->outShape[6]);
        yIdx = Simt::UintDiv(yIdx, m6, s6);
        uint32_t simtDimEightIndex5 = yIdx - Simt::UintDiv(yIdx, m5, s5) * static_cast<uint32_t>(tD->outShape[5]);
        yIdx = Simt::UintDiv(yIdx, m5, s5);
        uint32_t simtDimEightIndex4 = yIdx - Simt::UintDiv(yIdx, m4, s4) * static_cast<uint32_t>(tD->outShape[4]);
        yIdx = Simt::UintDiv(yIdx, m4, s4);
        uint32_t simtDimEightIndex3 = yIdx - Simt::UintDiv(yIdx, m3, s3) * static_cast<uint32_t>(tD->outShape[3]);
        yIdx = Simt::UintDiv(yIdx, m3, s3);
        uint32_t simtDimEightIndex2 = yIdx - Simt::UintDiv(yIdx, m2, s2) * static_cast<uint32_t>(tD->outShape[2]);
        yIdx = Simt::UintDiv(yIdx, m2, s2);
        uint32_t simtDimEightIndex1 = yIdx - Simt::UintDiv(yIdx, m1, s1) * static_cast<uint32_t>(tD->outShape[1]);
        uint32_t simtDimEightIndex0 = Simt::UintDiv(yIdx, m1, s1);

        int32_t inIndex7 = simtDimEightIndex7 - static_cast<uint32_t>(tD->leftPad[7]);
        int32_t inIndex6 = simtDimEightIndex6 - static_cast<uint32_t>(tD->leftPad[6]);
        int32_t inIndex5 = simtDimEightIndex5 - static_cast<uint32_t>(tD->leftPad[5]);
        int32_t inIndex4 = simtDimEightIndex4 - static_cast<uint32_t>(tD->leftPad[4]);
        int32_t inIndex3 = simtDimEightIndex3 - static_cast<uint32_t>(tD->leftPad[3]);
        int32_t inIndex2 = simtDimEightIndex2 - static_cast<uint32_t>(tD->leftPad[2]);
        int32_t inIndex1 = simtDimEightIndex1 - static_cast<uint32_t>(tD->leftPad[1]);
        int32_t inIndex0 = simtDimEightIndex0 - static_cast<uint32_t>(tD->leftPad[0]);

        if (inIndex7 >= 0 && inIndex7 < inShape7 && inIndex6 >= 0 && inIndex6 < inShape6 && inIndex5 >= 0 &&
            inIndex5 < inShape5 && inIndex4 >= 0 && inIndex4 < inShape4 && inIndex3 >= 0 && inIndex3 < inShape3 &&
            inIndex2 >= 0 && inIndex2 < inShape2 && inIndex1 >= 0 && inIndex1 < inShape1 && inIndex0 >= 0 &&
            inIndex0 < inShape0) {
            outputGM[idx] = inputGM[inIndex0 * inShape1 * inShape2 * inShape3 * inShape4 * inShape5 * inShape6 *
                                        inShape7 +
                                    inIndex1 * inShape2 * inShape3 * inShape4 * inShape5 * inShape6 * inShape7 +
                                    inIndex2 * inShape3 * inShape4 * inShape5 * inShape6 * inShape7 +
                                    inIndex3 * inShape4 * inShape5 * inShape6 * inShape7 +
                                    inIndex4 * inShape5 * inShape6 * inShape7 + inIndex5 * inShape6 * inShape7 +
                                    inIndex6 * inShape7 + inIndex7];
        } else {
            outputGM[idx] = fillValue;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void PadSimt<T, U>::Process(GM_ADDR tiling)
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD->dimNum; // 维度数

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeDimOne<T>>(dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()),
                                          (__gm__ volatile T*)(mOutputGM.GetPhyAddr()), constValue_, mTD->outShape[0],
                                          mBlockIdx, blockNum, mTD->inShape[0], mTD->leftPad[0]);
        return;
    }

    uint32_t outputSize = mTD->outShape[0] * mTD->outStride[0];
    if (outputSize == 0) {
        return;
    }

    uint32_t s[8];
    uint32_t m[8];

    for (uint32_t i = 1; i < mDimNum; ++i) {
        GetUintDivMagicAndShift(m[i], s[i], static_cast<uint32_t>(mTD->outShape[i]));
    }

    if (mDimNum == 2) {
        asc_vf_call<SimtComputeDimTwo<T>>(dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()),
                                          (__gm__ volatile T*)(mOutputGM.GetPhyAddr()), constValue_, outputSize,
                                          mBlockIdx, blockNum, mTD->outShape[1], mTD->inShape[0], mTD->inShape[1], m[1],
                                          s[1], mTD->leftPad[0], mTD->leftPad[1]);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeDimThree<T>>(dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()),
                                            (__gm__ volatile T*)(mOutputGM.GetPhyAddr()), constValue_, outputSize,
                                            mBlockIdx, blockNum, mTD->outShape[1], mTD->outShape[2], mTD->inShape[0],
                                            mTD->inShape[1], mTD->inShape[2], m[1], m[2], s[1], s[2], mTD->leftPad[0],
                                            mTD->leftPad[1], mTD->leftPad[2]);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeDimFour<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()), (__gm__ volatile T*)(mOutputGM.GetPhyAddr()),
            constValue_, outputSize, mBlockIdx, blockNum, mTD->outShape[1], mTD->outShape[2], mTD->outShape[3],
            mTD->inShape[0], mTD->inShape[1], mTD->inShape[2], mTD->inShape[3], m[1], m[2], m[3], s[1], s[2], s[3],
            mTD->leftPad[0], mTD->leftPad[1], mTD->leftPad[2], mTD->leftPad[3]);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeDimFive<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()), (__gm__ volatile T*)(mOutputGM.GetPhyAddr()),
            constValue_, outputSize, mBlockIdx, blockNum, mTD->outShape[1], mTD->outShape[2], mTD->outShape[3],
            mTD->outShape[4], mTD->inShape[0], mTD->inShape[1], mTD->inShape[2], mTD->inShape[3], mTD->inShape[4], m[1],
            m[2], m[3], m[4], s[1], s[2], s[3], s[4], mTD->leftPad[0], mTD->leftPad[1], mTD->leftPad[2],
            mTD->leftPad[3], mTD->leftPad[4]);
    } else if (mDimNum == 6) {
        asc_vf_call<SimtComputeDimSix<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()), (__gm__ volatile T*)(mOutputGM.GetPhyAddr()), tiling,
            constValue_, mBlockIdx, blockNum, mTD->inShape[0], mTD->inShape[1], mTD->inShape[2], mTD->inShape[3],
            mTD->inShape[4], mTD->inShape[5], m[1], m[2], m[3], m[4], m[5], s[1], s[2], s[3], s[4], s[5],
            mTD->leftPad[0], mTD->leftPad[1], mTD->leftPad[2], mTD->leftPad[3], mTD->leftPad[4], mTD->leftPad[5]);
    } else if (mDimNum == 7) {
        asc_vf_call<SimtComputeDimSeven<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()), (__gm__ volatile T*)(mOutputGM.GetPhyAddr()), tiling,
            constValue_, outputSize, mBlockIdx, blockNum, mTD->inShape[0], mTD->inShape[1], mTD->inShape[2],
            mTD->inShape[3], mTD->inShape[4], mTD->inShape[5], mTD->inShape[6], m[1], m[2], m[3], m[4], m[5], m[6],
            s[1], s[2], s[3], s[4], s[5], s[6]);
    } else if (mDimNum == 8) {
        asc_vf_call<SimtComputeDimEight<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(mInputGM.GetPhyAddr()), (__gm__ volatile T*)(mOutputGM.GetPhyAddr()), tiling,
            constValue_, mBlockIdx, blockNum, mTD->inShape[0], mTD->inShape[1], mTD->inShape[2], mTD->inShape[3],
            mTD->inShape[4], mTD->inShape[5], mTD->inShape[6], mTD->inShape[7], m[1], m[2], m[3], m[4], m[5], m[6],
            m[7], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
    }
}
} // namespace PadV3

#endif //  PAD_SIMT_H
