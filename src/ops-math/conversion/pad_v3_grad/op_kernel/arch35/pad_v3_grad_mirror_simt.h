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
 * \file
 * \brief
 */

#ifndef PAD_V3_GRAD_MIRROR_SIMT_H
#define PAD_V3_GRAD_MIRROR_SIMT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "pad_v3_grad_struct.h"

constexpr int32_t MIRROR_THREAD_DIM = 2048;
constexpr int32_t MIRROR_HALF_THREAD_DIM = 1024;
constexpr int32_t MIRROR_QUARTER_THREAD_DIM = 512;
constexpr int32_t MIRROR_EIGHTH_THREAD_DIM = 256;
constexpr int32_t MIRROR_SIXTEENTH_THREAD_DIM = 128;

namespace PadV3Grad {
using namespace AscendC;

template <typename T, uint8_t KEY>
class PadV3GradMirrorSimt {
public:
    __aicore__ inline PadV3GradMirrorSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData);
    template <typename U>
    __aicore__ inline void Process();

private:
    GlobalTensor<T> mInputGM_;
    GlobalTensor<T> mOutputGM_;
    uint32_t mBlockIdx_;               // 核号
    const PadV3GradACTilingData* mTD_; // tilingData
};

template <typename T, uint8_t KEY>
__aicore__ inline void PadV3GradMirrorSimt<T, KEY>::Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;

    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <uint8_t DIM_NUM, typename U>
__simt_callee__ __aicore__ void ReflectDimOffset(IdxAndTimes<U>* inIdxCnt, U* inIndex, U* outIndex,
                                                 __ubuf__ U* inStrides, __ubuf__ U* outShapes, __ubuf__ U* leftPads,
                                                 __ubuf__ U* rightPads)
{
    for (uint8_t i = 0; i < DIM_NUM; i++) {
        inIdxCnt[i].inGmIdx[0] = inIndex[i] * inStrides[i];
        if (outIndex[i] - 1 < leftPads[i] && outIndex[i] > 0) // left
        {
            // 计算该点该维度左pad在输入GM上的偏移
            inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt] = (leftPads[i] - outIndex[i]) * inStrides[i];
            inIdxCnt[i].cnt++;
        }
        if (outShapes[i] - outIndex[i] - 1 <= rightPads[i] && outShapes[i] - outIndex[i] - 1 > 0) // right
        {
            // 计算该点该维度右pad在输入GM上的偏移
            inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt] = (2 * outShapes[i] - outIndex[i] + leftPads[i] - 2) * inStrides[i];
            inIdxCnt[i].cnt++;
        }
    }
}

template <uint8_t DIM_NUM, typename U>
__simt_callee__ __aicore__ void SymmetricDimOffset(IdxAndTimes<U>* inIdxCnt, U* inIndex, U* outIndex,
                                                   __ubuf__ U* inStrides, __ubuf__ U* outShapes, __ubuf__ U* leftPads,
                                                   __ubuf__ U* rightPads)
{
    for (uint8_t i = 0; i < DIM_NUM; i++) {
        inIdxCnt[i].inGmIdx[0] = inIndex[i] * inStrides[i];
        if (outIndex[i] < leftPads[i]) {
            inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt] = (leftPads[i] - outIndex[i] - 1) * inStrides[i];
            inIdxCnt[i].cnt++;
        }
        if (outShapes[i] - outIndex[i] <= rightPads[i]) {
            inIdxCnt[i].inGmIdx[inIdxCnt[i].cnt] = (2 * outShapes[i] - outIndex[i] + leftPads[i] - 1) * inStrides[i];
            inIdxCnt[i].cnt++;
        }
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                              uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                              __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                              __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                              __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> inIdxCnt[DIM_NUM];
        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < inIdxCnt[0].cnt; a0++) {
            if (inIdxCnt[0].inGmIdx[a0] < 0 || inIdxCnt[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = inIdxCnt[0].inGmIdx[a0];
            CastType tmpVal;
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                tmpVal = __bfloat162float(inputGM[a0Offset]);
            } else if constexpr (std::is_same_v<T, float16_t>) {
                tmpVal = __half2float(inputGM[a0Offset]);
            } else {
                tmpVal = inputGM[a0Offset];
            }
            total += tmpVal;
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                              uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                              __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                              __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                              __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        uint64_t yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> inIdxCnt[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < inIdxCnt[0].cnt; a0++) {
            if (inIdxCnt[0].inGmIdx[a0] < 0 || inIdxCnt[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = static_cast<uint64_t>(inIdxCnt[0].inGmIdx[a0]);
            for (uint8_t a1 = 0; a1 < inIdxCnt[1].cnt; a1++) {
                if (inIdxCnt[1].inGmIdx[a1] < 0 || inIdxCnt[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + static_cast<uint64_t>(inIdxCnt[1].inGmIdx[a1]);
                CastType tmpVal;
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    tmpVal = __bfloat162float(inputGM[a1Offset]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    tmpVal = __half2float(inputGM[a1Offset]);
                } else {
                    tmpVal = inputGM[a1Offset];
                }
                total += tmpVal;
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                                uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                                __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                                __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                                __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> inIdxCnt[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < inIdxCnt[0].cnt; a0++) {
            if (inIdxCnt[0].inGmIdx[a0] < 0 || inIdxCnt[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = inIdxCnt[0].inGmIdx[a0];
            for (uint8_t a1 = 0; a1 < inIdxCnt[1].cnt; a1++) {
                if (inIdxCnt[1].inGmIdx[a1] < 0 || inIdxCnt[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + inIdxCnt[1].inGmIdx[a1];
                for (uint8_t a2 = 0; a2 < inIdxCnt[2].cnt; a2++) {
                    if (inIdxCnt[2].inGmIdx[a2] < 0 || inIdxCnt[2].inGmIdx[a2] >= cutBounds[2]) {
                        continue;
                    }
                    GmOffsetType a2Offset = a1Offset + inIdxCnt[2].inGmIdx[a2];
                    CastType tmpVal;
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        tmpVal = __bfloat162float(inputGM[a2Offset]);
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        tmpVal = __half2float(inputGM[a2Offset]);
                    } else {
                        tmpVal = inputGM[a2Offset];
                    }
                    total += tmpVal;
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                               __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                               __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> inIdxCnt[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < inIdxCnt[0].cnt; a0++) {
            if (inIdxCnt[0].inGmIdx[a0] < 0 || inIdxCnt[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = inIdxCnt[0].inGmIdx[a0];
            for (uint8_t a1 = 0; a1 < inIdxCnt[1].cnt; a1++) {
                if (inIdxCnt[1].inGmIdx[a1] < 0 || inIdxCnt[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + inIdxCnt[1].inGmIdx[a1];
                for (uint8_t a2 = 0; a2 < inIdxCnt[2].cnt; a2++) {
                    if (inIdxCnt[2].inGmIdx[a2] < 0 || inIdxCnt[2].inGmIdx[a2] >= cutBounds[2]) {
                        continue;
                    }
                    GmOffsetType a2Offset = a1Offset + inIdxCnt[2].inGmIdx[a2];
                    for (uint8_t a3 = 0; a3 < inIdxCnt[3].cnt; a3++) {
                        if (inIdxCnt[3].inGmIdx[a3] < 0 || inIdxCnt[3].inGmIdx[a3] >= cutBounds[3]) {
                            continue;
                        }
                        GmOffsetType a3Offset = a2Offset + inIdxCnt[3].inGmIdx[a3];
                        CastType tmpVal;
                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                            tmpVal = __bfloat162float(inputGM[a3Offset]);
                        } else if constexpr (std::is_same_v<T, float16_t>) {
                            tmpVal = __half2float(inputGM[a3Offset]);
                        } else {
                            tmpVal = inputGM[a3Offset];
                        }
                        total += tmpVal;
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t DIM_NUM, typename U, typename GmOffsetType, typename CastType, uint8_t KEY>
__simt_vf__ LAUNCH_BOUND(MIRROR_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeMirrorFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                               __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts,
                               __ubuf__ U* cutBounds)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U outIndex[DIM_NUM]{0};
        U inIndex[DIM_NUM]{0};
        GmOffsetType yIdx = idx;

        // 计算输出索引
        CalPos<DIM_NUM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        // 在每一维上填充的个数（包括自身）及其偏移
        IdxAndTimes<U> inIdxCnt[DIM_NUM];

        if constexpr (KEY == 2) {
            ReflectDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        } else {
            SymmetricDimOffset<DIM_NUM, U>(inIdxCnt, inIndex, outIndex, inStrides, outShapes, leftPads, rightPads);
        }

        CastType total = 0;
        for (uint8_t a0 = 0; a0 < inIdxCnt[0].cnt; a0++) {
            if (inIdxCnt[0].inGmIdx[a0] < 0 || inIdxCnt[0].inGmIdx[a0] >= cutBounds[0]) {
                continue;
            }
            GmOffsetType a0Offset = inIdxCnt[0].inGmIdx[a0];
            for (uint8_t a1 = 0; a1 < inIdxCnt[1].cnt; a1++) {
                if (inIdxCnt[1].inGmIdx[a1] < 0 || inIdxCnt[1].inGmIdx[a1] >= cutBounds[1]) {
                    continue;
                }
                GmOffsetType a1Offset = a0Offset + inIdxCnt[1].inGmIdx[a1];
                for (uint8_t a2 = 0; a2 < inIdxCnt[2].cnt; a2++) {
                    if (inIdxCnt[2].inGmIdx[a2] < 0 || inIdxCnt[2].inGmIdx[a2] >= cutBounds[2]) {
                        continue;
                    }
                    GmOffsetType a2Offset = a1Offset + inIdxCnt[2].inGmIdx[a2];
                    for (uint8_t a3 = 0; a3 < inIdxCnt[3].cnt; a3++) {
                        if (inIdxCnt[3].inGmIdx[a3] < 0 || inIdxCnt[3].inGmIdx[a3] >= cutBounds[3]) {
                            continue;
                        }
                        GmOffsetType a3Offset = a2Offset + inIdxCnt[3].inGmIdx[a3];
                        for (uint8_t a4 = 0; a4 < inIdxCnt[4].cnt; a4++) {
                            if (inIdxCnt[4].inGmIdx[a4] < 0 || inIdxCnt[4].inGmIdx[a4] >= cutBounds[4]) {
                                continue;
                            }
                            GmOffsetType a4Offset = a3Offset + inIdxCnt[4].inGmIdx[a4];
                            CastType tmpVal;
                            if constexpr (std::is_same_v<T, bfloat16_t>) {
                                tmpVal = __bfloat162float(inputGM[a4Offset]);
                            } else if constexpr (std::is_same_v<T, float16_t>) {
                                tmpVal = __half2float(inputGM[a4Offset]);
                            } else {
                                tmpVal = inputGM[a4Offset];
                            }
                            total += tmpVal;
                        }
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, uint8_t KEY>
template <typename U>
__aicore__ inline void PadV3GradMirrorSimt<T, KEY>::Process()
{
    using CastType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float32_t,
                                        std::conditional_t<std::is_same_v<T, float16_t>, float32_t, T>>;
    using GmOffsetType = std::conditional_t<std::is_same_v<U, int64_t>, uint64_t, uint32_t>;

    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    GmOffsetType outputSize = 1;
    for (uint8_t i = 0; i < mTD_->dimNum; i++) {
        outputSize *= mTD_->outShape[i];
    }

    if (outputSize == 0) {
        return;
    }
    // 快速除参数
    __ubuf__ GmOffsetType magics[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ GmOffsetType shifts[PAD_GRAD_MAX_DIMS_NUM];
    // tiling data
    __ubuf__ U inShapes[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U outShapes[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U inStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U outStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U leftPads[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U rightPads[PAD_GRAD_MAX_DIMS_NUM];
    // 裁剪边界
    __ubuf__ U cutBounds[PAD_GRAD_MAX_DIMS_NUM];

    GmOffsetType m = 0, s = 0;
    for (int i = 0; i < mTD_->dimNum; i++) {
        inShapes[i] = static_cast<U>(mTD_->inShape[i]);
        outShapes[i] = static_cast<U>(mTD_->outShape[i]);
        inStrides[i] = static_cast<U>(mTD_->inStride[i]);
        outStrides[i] = static_cast<U>(mTD_->outStride[i]);
        leftPads[i] = mTD_->leftPad[i];
        rightPads[i] = mTD_->rightPad[i];

        GetUintDivMagicAndShift(m, s, static_cast<GmOffsetType>(mTD_->outStride[i]));
        magics[i] = m;
        shifts[i] = s;

        cutBounds[i] = static_cast<U>(mTD_->inShape[i]) * static_cast<U>(mTD_->inStride[i]);
    }

    if (mTD_->dimNum == 1) {
        asc_vf_call<SimtComputeMirrorOne<T, 1, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mTD_->dimNum == 2) {
        asc_vf_call<SimtComputeMirrorTwo<T, 2, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mTD_->dimNum == 3) {
        asc_vf_call<SimtComputeMirrorThree<T, 3, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mTD_->dimNum == 4) {
        asc_vf_call<SimtComputeMirrorFour<T, 4, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    } else if (mTD_->dimNum == 5) {
        asc_vf_call<SimtComputeMirrorFive<T, 5, U, GmOffsetType, CastType, KEY>>(
            dim3(MIRROR_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts, cutBounds);
    }
}

} // namespace PadV3Grad
#endif
