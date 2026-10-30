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
 * \file im2col_simt_NHWC.h
 * \brief
 */
#ifndef IM2COL_SIMT_NHWC_H
#define IM2COL_SIMT_NHWC_H

#include <type_traits>
#include "kernel_operator.h"
#include "im2col_tilingdata.h"
#include "simt_api/asc_simt.h"

namespace Im2ColAsc {
using namespace AscendC;

template <typename T, typename U>
class Im2ColSIMT_NHWC {
public:
    __aicore__ inline Im2ColSIMT_NHWC(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Im2ColSIMTTilingData* tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    uint32_t blockIdx_ = 0;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint32_t threadNum_;

    // 分核信息
    uint32_t realCoreNum_;
    uint32_t mainCoreNum_;
    uint64_t perCoreElement_;
    uint64_t curCoreElement_;

    // shape 信息
    uint32_t batchSize_; // N
    U channel_;   // C

    // kernel 信息
    U hKSize_;
    U wKSize_;

    // 卷积输出shape
    U convKernelNumInHeight_;
    U convKernelNumInWidth_;

    uint64_t n_;

    __aicore__ inline void ParseSIMTTilingData(const Im2ColSIMTTilingData* tilingData);
};

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void Im2ColSIMTNHWCCompute(
    U channel, U hKSize, U wKSize, U convKernelNumInHeight, U convKernelNumInWidth, U yBaseIdx,
    uint64_t curCoreElement, U magic0, U shift0, U magic1, U shift1, U magic2,
    U shift2, U magic3, U shift3, U magic4, U shift4, GM_ADDR tiling, __gm__ T* x,
    __gm__ volatile T* y)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(Im2ColSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U index = U(yBaseIdx + idx);

        U indexDivC = Simt::UintDiv(index, magic0, shift0);
        // co=channel * hKSize * wKSize
        U indexDivCKernelH = Simt::UintDiv(indexDivC, magic1, shift1);
        U indexDivCo = Simt::UintDiv(indexDivCKernelH, magic2, shift2);
        U indexDivCoConvW = Simt::UintDiv(indexDivCo, magic3, shift3);
        // 计算 index 在卷积输出shape的位置
        U bOut = Simt::UintDiv(indexDivCoConvW, magic4, shift4);
        U cOut = index - indexDivC * channel;
        U wOut = indexDivCo - indexDivCoConvW * convKernelNumInWidth;
        U hOut = indexDivCoConvW - bOut * convKernelNumInHeight;
        // 计算 index在卷积核中的相对位置
        U kIdx = index - indexDivCo * channel * hKSize * wKSize;
        U kIdxDivC = Simt::UintDiv(kIdx, magic0, shift0);
        U kIdxDivCKernelW = Simt::UintDiv(kIdxDivC, magic2, shift2);
        U kIdxDivCKernelWH = Simt::UintDiv(kIdxDivCKernelW, magic1, shift1);
        U kW = kIdxDivC - kIdxDivCKernelW * wKSize;
        U kH = kIdxDivCKernelW - kIdxDivCKernelWH * hKSize;
        // 把输出坐标映射回输入坐标
        U imRow = hOut * tdGmPtr->input.hStride - tdGmPtr->input.hPaddingBefore + kH * tdGmPtr->input.hDilation;
        U imCol = wOut * tdGmPtr->input.wStride - tdGmPtr->input.wPaddingBefore + kW * tdGmPtr->input.wDilation;
        if (imRow >= 0 && imRow < tdGmPtr->input.H && imCol >= 0 && imCol < tdGmPtr->input.W) {
            U xIdx = bOut * tdGmPtr->input.H * tdGmPtr->input.W * channel + imRow * tdGmPtr->input.W * channel + imCol * channel + cOut;
            y[index] = x[xIdx];
        } else {
            y[index] = 0;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void Im2ColSIMT_NHWC<T, U>::Init(
    GM_ADDR x, GM_ADDR y, const Im2ColSIMTTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    this->ParseSIMTTilingData(tilingData);
    threadNum_ = HALF_THREAD_NUM_LAUNCH_BOUND;
}

template <typename T, typename U>
__aicore__ inline void Im2ColSIMT_NHWC<T, U>::ParseSIMTTilingData(const Im2ColSIMTTilingData* tilingData)
{
    // shape 信息
    batchSize_ = tilingData->input.N;
    channel_ = U(tilingData->input.C);

    // kernel 信息
    hKSize_ = U(tilingData->input.hKernelSize);
    wKSize_ = U(tilingData->input.wKernelSize);

    // 卷积输出shape
    convKernelNumInHeight_ = U(tilingData->convKernelNumInHeight);
    convKernelNumInWidth_ = U(tilingData->convKernelNumInWidth);

    // 分核信息
    realCoreNum_ = tilingData->realCoreNum;
    mainCoreNum_ = tilingData->mainCoreNum;

    perCoreElement_ = tilingData->blockFactor;
    if (blockIdx_ >= mainCoreNum_) {
        curCoreElement_ = tilingData->blockTailFactor;
    } else {
        curCoreElement_ = tilingData->blockFactor;
    }
    n_ = U(batchSize_) * channel_ * hKSize_ * wKSize_ * convKernelNumInHeight_ * convKernelNumInWidth_;
}

template <typename T, typename U>
__aicore__ inline void Im2ColSIMT_NHWC<T, U>::Process(GM_ADDR tiling)
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    U yBlockOffset = 0;
    if (blockIdx_ >= mainCoreNum_) {
        yBlockOffset = U(mainCoreNum_ * perCoreElement_ + (blockIdx_ - mainCoreNum_) * curCoreElement_);
    } else {
        yBlockOffset = U(blockIdx_ * perCoreElement_);
    }
    if (blockIdx_ == realCoreNum_ - 1) {
        curCoreElement_ = n_ >= yBlockOffset ? n_ - yBlockOffset : 0;
    }
    // 快速除
    U magic0 = 0;
    U shift0 = 0;
    U magic1 = 0;
    U shift1 = 0;
    U magic2 = 0;
    U shift2 = 0;
    U magic3 = 0;
    U shift3 = 0;
    U magic4 = 0;
    U shift4 = 0;
    GetUintDivMagicAndShift(magic0, shift0, channel_);
    GetUintDivMagicAndShift(magic1, shift1, hKSize_);
    GetUintDivMagicAndShift(magic2, shift2, wKSize_);
    GetUintDivMagicAndShift(magic3, shift3, convKernelNumInWidth_);
    GetUintDivMagicAndShift(magic4, shift4, convKernelNumInHeight_);
    // SMIT搬移
    asc_vf_call<Im2ColSIMTNHWCCompute<T, U>>(
        dim3(threadNum_), channel_, hKSize_, wKSize_, convKernelNumInHeight_, convKernelNumInWidth_, yBlockOffset, curCoreElement_,
        magic0, shift0, magic1, shift1, magic2, shift2, magic3, shift3, magic4, shift4, tiling,
        (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ volatile T*)(yGm_.GetPhyAddr()));
}
} // namespace Im2ColAsc
#endif // IM2COL_SIMT_NHWC_H