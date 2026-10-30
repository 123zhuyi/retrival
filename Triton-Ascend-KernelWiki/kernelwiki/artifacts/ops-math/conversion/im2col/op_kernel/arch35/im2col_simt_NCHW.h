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
 * \file im2col_simt_NCHW.h
 * \brief
 */
#ifndef IM2COL_SIMT_NCHW_H
#define IM2COL_SIMT_NCHW_H

#include <type_traits>
#include "kernel_operator.h"
#include "im2col_tilingdata.h"
#include "simt_api/asc_simt.h"
namespace Im2ColAsc {
using namespace AscendC;
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
constexpr uint32_t HALF_THREAD_NUM_LAUNCH_BOUND = 256;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;
constexpr uint32_t HALF_THREAD_NUM_LAUNCH_BOUND = 1024;
#endif
template <typename T, typename U>
class Im2ColSIMT_NCHW {
    public:
        __aicore__ inline Im2ColSIMT_NCHW(){};
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Im2ColSIMTTilingData* tilingData);
        __aicore__ inline void Process(GM_ADDR tiling);
    private:
        uint32_t blockIdx_ = 0;
        GlobalTensor<T> xGm_;
        GlobalTensor<T> yGm_;
        uint32_t realCoreNum_;
        uint32_t mainCoreNum_;
        uint64_t perCoreElement_;
        uint64_t curCoreElement_;

        uint32_t batchSize_;//N
        uint32_t channel_;//C

        U hKSize_;//卷积核高
        U wKSize_;//卷积核宽
        U convKernelNumInHeight_;//卷积输出shape的H大小
        U convKernelNumInWidth_;//卷积输出shape的w大小

        uint64_t n_;//总数
        uint32_t threadNum_;
        __aicore__ inline void ParseSIMTTilingData(const Im2ColSIMTTilingData* tilingData);
};

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void Im2ColSIMTCompute(
    U yBaseIdx,  uint64_t curCoreElement, U convKernelNumInHeight, U convKernelNumInWidth,
    U wKSize, U hKSize,
    U magic0, U shift0,U magic1, U shift1, U magic2, U shift2, U magic3, U shift3,
    GM_ADDR tiling, __gm__ T* x, __gm__ volatile T* y)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(Im2ColSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = yBaseIdx + U(idx);
        U yIdxH = Simt::UintDiv(yIdx, magic1, shift1);
        U kIdx = Simt::UintDiv(yIdxH, magic0, shift0);
        U wOut = yIdx - yIdxH * convKernelNumInWidth;
        U hOut = yIdxH - kIdx * convKernelNumInHeight;

        //该索引在当前卷积核中的哪个位置  b * c * kernelsize *kernelsize
        U kIdxH = Simt::UintDiv(kIdx, magic3, shift3);
        U batchAndChannel = Simt::UintDiv(kIdxH, magic2, shift2);
        U kW = kIdx - kIdxH * wKSize;
        U kH = kIdxH - batchAndChannel * hKSize;

        //该索引在输入shape上的具体位置
        U imRow = hOut * tdGmPtr->input.hStride - tdGmPtr->input.hPaddingBefore + kH * tdGmPtr->input.hDilation;
        U imCol = wOut * tdGmPtr->input.wStride - tdGmPtr->input.wPaddingBefore + kW * tdGmPtr->input.wDilation;

        if (imRow >= 0 && imRow < tdGmPtr->input.H && imCol >= 0 && imCol < tdGmPtr->input.W) {
            U xIdx = (batchAndChannel * tdGmPtr->input.H + imRow) * tdGmPtr->input.W + imCol;
            y[yIdx] = x[xIdx];
        }
        else {
            y[yIdx] = 0;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void Im2ColSIMT_NCHW<T, U>::Init(GM_ADDR x, GM_ADDR y, const Im2ColSIMTTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    this->ParseSIMTTilingData(tilingData);
}

template <typename T, typename U>
__aicore__ inline void Im2ColSIMT_NCHW<T, U>::ParseSIMTTilingData(const Im2ColSIMTTilingData* tilingData)
{
    realCoreNum_ = tilingData->realCoreNum;
    mainCoreNum_ = tilingData->mainCoreNum;
    perCoreElement_ = tilingData->blockFactor;
    batchSize_ = tilingData->input.N;
    channel_ = tilingData->input.C;
    hKSize_ = U(tilingData->input.hKernelSize);
    wKSize_ = U(tilingData->input.wKernelSize);
    convKernelNumInHeight_ = U(tilingData->convKernelNumInHeight);
    convKernelNumInWidth_ = U(tilingData->convKernelNumInWidth);
    threadNum_ = tilingData->threadNum;
    if (blockIdx_ >= mainCoreNum_) {
        curCoreElement_ = tilingData->blockTailFactor;
    } else {
        curCoreElement_ = tilingData->blockFactor;
    }
    n_ = U(batchSize_) * channel_ * hKSize_ * wKSize_ * convKernelNumInHeight_ * convKernelNumInWidth_;
}

template <typename T, typename U>
__aicore__ inline void Im2ColSIMT_NCHW<T, U>::Process(GM_ADDR tiling)
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
    if (blockIdx_ == realCoreNum_ - 1){
        curCoreElement_ = n_ >= yBlockOffset ? n_ - yBlockOffset : 0;
    }
    //快速除
    U magic0 = 0;
    U shift0 = 0;
    U magic1 = 0;
    U shift1 = 0;
    U magic2 = 0;
    U shift2 = 0;
    U magic3 = 0;
    U shift3 = 0;
    GetUintDivMagicAndShift(magic0, shift0, convKernelNumInHeight_);
    GetUintDivMagicAndShift(magic1, shift1, convKernelNumInWidth_);
    GetUintDivMagicAndShift(magic2, shift2, hKSize_);
    GetUintDivMagicAndShift(magic3, shift3, wKSize_);
    //SMIT搬移
    asc_vf_call<Im2ColSIMTCompute<T, U>>(
        dim3(threadNum_), yBlockOffset, curCoreElement_, convKernelNumInHeight_, convKernelNumInWidth_,
    wKSize_, hKSize_,
    magic0, shift0, magic1, shift1, magic2, shift2, magic3, shift3,
    tiling, (__gm__ T*) (xGm_.GetPhyAddr()), (__gm__ volatile T*) (yGm_.GetPhyAddr()));
}
}// namespace Im2ColAsc
#endif // IM2COL_SIMT_NCHW_H