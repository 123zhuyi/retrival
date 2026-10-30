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
 * \file pad_v3_grad_edge_simt.h
 * \brief pad_v3_grad_edge_simt
 */
#ifndef PAD_V3_GRAD_EDGE_SIMT_H
#define PAD_V3_GRAD_EDGE_SIMT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "pad_v3_grad_struct.h"

constexpr int32_t EDGE_THREAD_DIM = 2048;
constexpr int32_t EDGE_HALF_THREAD_DIM = 1024;
constexpr int32_t EDGE_QUARTER_THREAD_DIM = 512;
constexpr int32_t EDGE_EIGHTH_THREAD_DIM = 256;
constexpr int32_t EDGE_SIXTEENTH_THREAD_DIM = 128;
namespace PadV3Grad {
using namespace AscendC;

template <typename T>
class PadV3GradEdgeSimt {
public:
    __aicore__ inline PadV3GradEdgeSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData);
    template <typename U>
    __aicore__ inline void Process();

private:
    GlobalTensor<T> mInputGM_;         // GM x
    GlobalTensor<T> mOutputGM_;        // GM y
    uint32_t mBlockIdx_;               // 核号
    const PadV3GradACTilingData* mTD_; // tilingData
};

template <typename T>
__aicore__ inline void PadV3GradEdgeSimt<T>::Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <uint8_t DIM, typename U>
__simt_callee__ __aicore__ void CalScope(U (*scopeIndex)[2], U* inIndex, U* outIndex, __ubuf__ U* rightPads,
                                         __ubuf__ U* inShapes, __ubuf__ U* outShapes)
{
    int8_t flag = 0;
    for (uint8_t i = 0; i < DIM; i++) {
        // 判断是否被裁剪
        flag = 0;
        if (inIndex[i] < 0 || inIndex[i] >= inShapes[i]) {
            flag = 1;
        }
        // 判断起始范围
        if (outIndex[i] == 0) {
            scopeIndex[i][0] = 0;
        } else {
            scopeIndex[i][0] = inIndex[i] + flag;
        }
        // 判断结束范围
        if (outIndex[i] == outShapes[i] - 1) {
            scopeIndex[i][1] = inIndex[i] + rightPads[i];
        } else {
            scopeIndex[i][1] = inIndex[i] - flag;
        }
    }
}

template <typename T, int32_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(EDGE_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimOne(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                               __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U inIndex[DIM]{0};
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;

        // 计算输入输出索引
        CalPos<DIM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        U scopeIndex[DIM][2]; // 每个维度的padding范围
        CalScope<DIM, U>(scopeIndex, inIndex, outIndex, rightPads, inShapes, outShapes);

        CastType total = 0;
        for (U a0 = scopeIndex[0][0]; a0 <= scopeIndex[0][1]; ++a0) {
            GmOffsetType inputOffset = static_cast<GmOffsetType>(a0);
            CastType tmpVal;
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                tmpVal = __bfloat162float(inputGM[inputOffset]);
            } else if constexpr (std::is_same_v<T, float16_t>) {
                tmpVal = __half2float(inputGM[inputOffset]);
            } else {
                tmpVal = inputGM[inputOffset];
            }
            total += tmpVal;
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, int32_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(EDGE_HALF_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                               uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                               __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                               __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U inIndex[DIM]{0};
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;

        // 计算输入输出索引
        CalPos<DIM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        U scopeIndex[DIM][2]; // 每个维度的padding范围

        CalScope<DIM, U>(scopeIndex, inIndex, outIndex, rightPads, inShapes, outShapes);

        CastType total = 0;
        for (U a0 = scopeIndex[0][0]; a0 <= scopeIndex[0][1]; ++a0) {
            for (U a1 = scopeIndex[1][0]; a1 <= scopeIndex[1][1]; ++a1) {
                GmOffsetType inputOffset = static_cast<GmOffsetType>(a0 * inStrides[0] + a1);
                CastType tmpVal;
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    tmpVal = __bfloat162float(inputGM[inputOffset]);
                } else if constexpr (std::is_same_v<T, float16_t>) {
                    tmpVal = __half2float(inputGM[inputOffset]);
                } else {
                    tmpVal = inputGM[inputOffset];
                }
                total += tmpVal;
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, int32_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(EDGE_QUARTER_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                                 uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                                 __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                                 __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U inIndex[DIM]{0};
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;

        // 计算输入输出索引
        CalPos<DIM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        U scopeIndex[DIM][2]; // 每个维度的padding范围

        CalScope<DIM, U>(scopeIndex, inIndex, outIndex, rightPads, inShapes, outShapes);

        CastType total = 0;
        for (U a0 = scopeIndex[0][0]; a0 <= scopeIndex[0][1]; ++a0) {
            for (U a1 = scopeIndex[1][0]; a1 <= scopeIndex[1][1]; ++a1) {
                for (U a2 = scopeIndex[2][0]; a2 <= scopeIndex[2][1]; ++a2) {
                    GmOffsetType inputOffset = static_cast<GmOffsetType>(a0 * inStrides[0] + a1 * inStrides[1] + a2);
                    CastType tmpVal;
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        tmpVal = __bfloat162float(inputGM[inputOffset]);
                    } else if constexpr (std::is_same_v<T, float16_t>) {
                        tmpVal = __half2float(inputGM[inputOffset]);
                    } else {
                        tmpVal = inputGM[inputOffset];
                    }
                    total += tmpVal;
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, int32_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(EDGE_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                                uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                                __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                                __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U inIndex[DIM]{0};
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;

        // 计算输入输出索引
        CalPos<DIM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        U scopeIndex[DIM][2]; // 每个维度的padding范围

        CalScope<DIM, U>(scopeIndex, inIndex, outIndex, rightPads, inShapes, outShapes);

        CastType total = 0;
        for (U a0 = scopeIndex[0][0]; a0 <= scopeIndex[0][1]; ++a0) {
            for (U a1 = scopeIndex[1][0]; a1 <= scopeIndex[1][1]; ++a1) {
                for (U a2 = scopeIndex[2][0]; a2 <= scopeIndex[2][1]; ++a2) {
                    for (U a3 = scopeIndex[3][0]; a3 <= scopeIndex[3][1]; ++a3) {
                        GmOffsetType inputOffset = static_cast<GmOffsetType>(a0 * inStrides[0] + a1 * inStrides[1] +
                                                                             a2 * inStrides[2] + a3);
                        CastType tmpVal;
                        if constexpr (std::is_same_v<T, bfloat16_t>) {
                            tmpVal = __bfloat162float(inputGM[inputOffset]);
                        } else if constexpr (std::is_same_v<T, float16_t>) {
                            tmpVal = __half2float(inputGM[inputOffset]);
                        } else {
                            tmpVal = inputGM[inputOffset];
                        }
                        total += tmpVal;
                    }
                }
            }
        }
        CopyOut<T, CastType, GmOffsetType>(idx, outputGM, total);
    }
}

template <typename T, int32_t DIM, typename U, typename GmOffsetType, typename CastType>
__simt_vf__ LAUNCH_BOUND(EDGE_EIGHTH_THREAD_DIM) __aicore__
    void SimtComputeEdgeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, GmOffsetType outputSize,
                                uint32_t blockIdx, uint32_t blockNum, __ubuf__ U* inShapes, __ubuf__ U* outShapes,
                                __ubuf__ U* inStrides, __ubuf__ U* outStrides, __ubuf__ U* leftPads,
                                __ubuf__ U* rightPads, __ubuf__ GmOffsetType* magics, __ubuf__ GmOffsetType* shifts)
{
    for (GmOffsetType idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize; idx += blockNum * blockDim.x) {
        U inIndex[DIM]{0};
        U outIndex[DIM]{0};
        GmOffsetType yIdx = idx;

        // 计算输入输出索引
        CalPos<DIM, U, GmOffsetType>(yIdx, inIndex, outIndex, outStrides, leftPads, magics, shifts);

        U scopeIndex[DIM][2]; // 每个维度的padding范围

        CalScope<DIM, U>(scopeIndex, inIndex, outIndex, rightPads, inShapes, outShapes);

        CastType total = 0;
        for (U a0 = scopeIndex[0][0]; a0 <= scopeIndex[0][1]; ++a0) {
            for (U a1 = scopeIndex[1][0]; a1 <= scopeIndex[1][1]; ++a1) {
                for (U a2 = scopeIndex[2][0]; a2 <= scopeIndex[2][1]; ++a2) {
                    for (U a3 = scopeIndex[3][0]; a3 <= scopeIndex[3][1]; ++a3) {
                        for (U a4 = scopeIndex[4][0]; a4 <= scopeIndex[4][1]; ++a4) {
                            GmOffsetType inputOffset = static_cast<GmOffsetType>(
                                a0 * inStrides[0] + a1 * inStrides[1] + a2 * inStrides[2] + a3 * inStrides[3] + a4);
                            CastType tmpVal;
                            if constexpr (std::is_same_v<T, bfloat16_t>) {
                                tmpVal = __bfloat162float(inputGM[inputOffset]);
                            } else if constexpr (std::is_same_v<T, float16_t>) {
                                tmpVal = __half2float(inputGM[inputOffset]);
                            } else {
                                tmpVal = inputGM[inputOffset];
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

template <typename T>
template <typename U>
__aicore__ inline void PadV3GradEdgeSimt<T>::Process()
{
    using CastType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float32_t,
                                        std::conditional_t<std::is_same_v<T, float16_t>, float32_t, T>>;
    using GmOffsetType = std::conditional_t<std::is_same_v<U, int64_t>, uint64_t, uint32_t>;

    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t mDimNum = mTD_->dimNum;
    GmOffsetType outputSize = mTD_->outShape[0] * mTD_->outStride[0];
    if (outputSize == 0) {
        return;
    }

    // 快速除参数
    __ubuf__ GmOffsetType shifts[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ GmOffsetType magics[PAD_GRAD_MAX_DIMS_NUM];
    // tiling data
    __ubuf__ U inShapes[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U outShapes[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U inStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U outStrides[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U leftPads[PAD_GRAD_MAX_DIMS_NUM];
    __ubuf__ U rightPads[PAD_GRAD_MAX_DIMS_NUM];

    GmOffsetType m = 0, s = 0;
    for (uint32_t i = 0; i < mDimNum; ++i) {
        inShapes[i] = static_cast<U>(mTD_->inShape[i]);
        outShapes[i] = static_cast<U>(mTD_->outShape[i]);
        inStrides[i] = static_cast<U>(mTD_->inStride[i]);
        outStrides[i] = static_cast<U>(mTD_->outStride[i]);
        leftPads[i] = mTD_->leftPad[i];
        rightPads[i] = mTD_->rightPad[i];
        GetUintDivMagicAndShift(m, s, static_cast<GmOffsetType>(mTD_->outStride[i]));
        magics[i] = m;
        shifts[i] = s;
    }
    DataSyncBarrier<MemDsbT::UB>();

    if (mDimNum == 1) {
        asc_vf_call<SimtComputeEdgeDimOne<T, 1, U, GmOffsetType, CastType>>(
            dim3(EDGE_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()), (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()),
            outputSize, mBlockIdx_, blockNum, inShapes, outShapes, inStrides, outStrides, leftPads, rightPads, magics,
            shifts);
    } else if (mDimNum == 2) {
        asc_vf_call<SimtComputeEdgeDimTwo<T, 2, U, GmOffsetType, CastType>>(
            dim3(EDGE_HALF_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts);
    } else if (mDimNum == 3) {
        asc_vf_call<SimtComputeEdgeDimThree<T, 3, U, GmOffsetType, CastType>>(
            dim3(EDGE_QUARTER_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts);
    } else if (mDimNum == 4) {
        asc_vf_call<SimtComputeEdgeDimFour<T, 4, U, GmOffsetType, CastType>>(
            dim3(EDGE_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts);
    } else if (mDimNum == 5) {
        asc_vf_call<SimtComputeEdgeDimFive<T, 5, U, GmOffsetType, CastType>>(
            dim3(EDGE_EIGHTH_THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, inShapes, outShapes,
            inStrides, outStrides, leftPads, rightPads, magics, shifts);
    }
}

} // namespace PadV3Grad

#endif // PAD_V3_GRAD_EDGE_SIMT_H
