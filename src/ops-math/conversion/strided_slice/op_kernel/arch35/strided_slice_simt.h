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
 * \file strided_slice_simt.h
 * \brief
 */
#ifndef STRIDED_SLICE_SIMT_H
#define STRIDED_SLICE_SIMT_H

#include "strided_slice_base.h"

namespace StridedSlice {
using namespace AscendC;

template <typename T, typename U>
class StridedSliceSIMT : public StridedSliceBase<T, U>
{
public:
    __aicore__ inline StridedSliceSIMT(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y, const StridedSliceSIMTTilingData* tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    __aicore__ inline void ProcessNormalShape(GM_ADDR tiling, uint32_t inputDims);

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int32_t blockIdx_ = 0;
    int32_t blockNums_ = 1;
    int64_t begin_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};
    const StridedSliceSIMTTilingData* tilingData_;
};

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim1Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, const uint32_t outputSize, const int32_t begin0,
    const int32_t strides0, int32_t blockIdx, int32_t blockNums)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;
        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = 0;
        inputIndex += (begin0 + strides0 * outputIndex[DIM0_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim2Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, const uint32_t outputSize, const uint32_t outputShapeProd1,
    const uint32_t inputShapeProd1, const int32_t begin0, const int32_t begin1, const int32_t strides0,
    const int32_t strides1, uint32_t m0, uint32_t s0, int32_t blockIdx, int32_t blockNums)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;
        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= outputIndex[DIM0_INDEX] * outputShapeProd1;

        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = 0;
        inputIndex += (begin0 + strides0 * outputIndex[DIM0_INDEX]) * inputShapeProd1;
        inputIndex += (begin1 + strides1 * outputIndex[DIM1_INDEX]);

        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim3Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, const uint32_t outputSize, const uint32_t outputShapeProd1,
    const uint32_t outputShapeProd2, const uint32_t inputShapeProd1, const uint32_t inputShapeProd2,
    const int32_t begin0, const int32_t begin1, const int32_t begin2, const int32_t strides0, const int32_t strides1,
    const int32_t strides2, uint32_t m0, uint32_t m1, uint32_t s0, uint32_t s1, int32_t blockIdx, int32_t blockNums)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= outputIndex[DIM0_INDEX] * outputShapeProd1;
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= outputIndex[DIM1_INDEX] * outputShapeProd2;

        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = 0;
        inputIndex += (begin0 + strides0 * outputIndex[DIM0_INDEX]) * inputShapeProd1;
        inputIndex += (begin1 + strides1 * outputIndex[DIM1_INDEX]) * inputShapeProd2;
        inputIndex += (begin2 + strides2 * outputIndex[DIM2_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim4Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, const uint32_t outputSize, const uint32_t outputShapeProd1,
    const uint32_t outputShapeProd2, const uint32_t outputShapeProd3, const uint32_t inputShapeProd1,
    const uint32_t inputShapeProd2, const uint32_t inputShapeProd3, const int32_t begin0, const int32_t begin1,
    const int32_t begin2, const int32_t begin3, const int32_t strides0, const int32_t strides1, const int32_t strides2,
    const int32_t strides3, uint32_t m0, uint32_t m1, uint32_t m2, uint32_t s0, uint32_t s1, uint32_t s2,
    int32_t blockIdx, int32_t blockNums)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= outputIndex[DIM0_INDEX] * outputShapeProd1;
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= outputIndex[DIM1_INDEX] * outputShapeProd2;
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= outputIndex[DIM2_INDEX] * outputShapeProd3;

        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = 0;
        inputIndex += (begin0 + strides0 * outputIndex[DIM0_INDEX]) * inputShapeProd1;
        inputIndex += (begin1 + strides1 * outputIndex[DIM1_INDEX]) * inputShapeProd2;
        inputIndex += (begin2 + strides2 * outputIndex[DIM2_INDEX]) * inputShapeProd3;
        inputIndex += (begin3 + strides3 * outputIndex[DIM3_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim5Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint32_t outputSize, int32_t blockIdx,
    int32_t blockNums, const uint32_t beginSum, const int32_t strides0, 
    const int32_t strides1, const int32_t strides2, const int32_t strides3, const int32_t strides4, uint32_t m0, 
    uint32_t m1, uint32_t m2, uint32_t m3, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= outputIndex[DIM0_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM1_INDEX]);
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= outputIndex[DIM1_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM2_INDEX]);
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= outputIndex[DIM2_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM3_INDEX]);
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= outputIndex[DIM3_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM4_INDEX]);

        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = beginSum;
        inputIndex += strides0 * outputIndex[DIM0_INDEX];
        inputIndex += strides1 * outputIndex[DIM1_INDEX];
        inputIndex += strides2 * outputIndex[DIM2_INDEX];
        inputIndex += strides3 * outputIndex[DIM3_INDEX];
        inputIndex += strides4 * outputIndex[DIM4_INDEX];
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim6Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint32_t outputSize, int32_t blockIdx,
    int32_t blockNums, const uint32_t beginSum, const int32_t strides0, 
    const int32_t strides1, const int32_t strides2, const int32_t strides3, const int32_t strides4, const int32_t strides5,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t s0, uint32_t s1,
    uint32_t s2, uint32_t s3, uint32_t s4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= outputIndex[DIM0_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM1_INDEX]);
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= outputIndex[DIM1_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM2_INDEX]);
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= outputIndex[DIM2_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM3_INDEX]);
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= outputIndex[DIM3_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM4_INDEX]);
        outputIndex[DIM4_INDEX] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= outputIndex[DIM4_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM5_INDEX]);

        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = beginSum;
        inputIndex += strides0 * outputIndex[DIM0_INDEX];
        inputIndex += strides1 * outputIndex[DIM1_INDEX];
        inputIndex += strides2 * outputIndex[DIM2_INDEX];
        inputIndex += strides3 * outputIndex[DIM3_INDEX];
        inputIndex += strides4 * outputIndex[DIM4_INDEX];
        inputIndex += strides5 * outputIndex[DIM5_INDEX];
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim7Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint32_t outputSize, int32_t blockIdx,
    int32_t blockNums, const uint32_t beginSum, const int32_t strides0, 
    const int32_t strides1, const int32_t strides2, const int32_t strides3, const int32_t strides4, const int32_t strides5, 
    const int32_t strides6, uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t s0,
    uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= outputIndex[DIM0_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM1_INDEX]);
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= outputIndex[DIM1_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM2_INDEX]);
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= outputIndex[DIM2_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM3_INDEX]);
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= outputIndex[DIM3_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM4_INDEX]);
        outputIndex[DIM4_INDEX] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= outputIndex[DIM4_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM5_INDEX]);
        outputIndex[DIM5_INDEX] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= outputIndex[DIM5_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM6_INDEX]);

        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = beginSum;
        inputIndex += strides0 * outputIndex[DIM0_INDEX];
        inputIndex += strides1 * outputIndex[DIM1_INDEX];
        inputIndex += strides2 * outputIndex[DIM2_INDEX];
        inputIndex += strides3 * outputIndex[DIM3_INDEX];
        inputIndex += strides4 * outputIndex[DIM4_INDEX];
        inputIndex += strides5 * outputIndex[DIM5_INDEX];
        inputIndex += strides6 * outputIndex[DIM6_INDEX];
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(HALF_THREAD_DIM) __aicore__ void StridedSliceDim8Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint32_t outputSize, int32_t blockIdx,
    int32_t blockNums, const uint32_t beginSum, const int32_t strides0, 
    const int32_t strides1, const int32_t strides2, const int32_t strides3, const int32_t strides4, const int32_t strides5, 
    const int32_t strides6, const int32_t strides7, uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5, uint32_t s6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int32_t outputIndex[Dim] = {0};
        uint32_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, m0, s0);
        dstIdx -= outputIndex[DIM0_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM1_INDEX]);
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, m1, s1);
        dstIdx -= outputIndex[DIM1_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM2_INDEX]);
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, m2, s2);
        dstIdx -= outputIndex[DIM2_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM3_INDEX]);
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, m3, s3);
        dstIdx -= outputIndex[DIM3_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM4_INDEX]);
        outputIndex[DIM4_INDEX] = Simt::UintDiv(dstIdx, m4, s4);
        dstIdx -= outputIndex[DIM4_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM5_INDEX]);
        outputIndex[DIM5_INDEX] = Simt::UintDiv(dstIdx, m5, s5);
        dstIdx -= outputIndex[DIM5_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM6_INDEX]);
        outputIndex[DIM6_INDEX] = Simt::UintDiv(dstIdx, m6, s6);
        dstIdx -= outputIndex[DIM6_INDEX] * int32_t(tdGmPtr->outputShapeProd[DIM7_INDEX]);

        outputIndex[Dim - 1] = dstIdx;

        uint32_t inputIndex = beginSum;
        inputIndex += strides0 * outputIndex[DIM0_INDEX];
        inputIndex += strides1 * outputIndex[DIM1_INDEX];
        inputIndex += strides2 * outputIndex[DIM2_INDEX];
        inputIndex += strides3 * outputIndex[DIM3_INDEX];
        inputIndex += strides4 * outputIndex[DIM4_INDEX];
        inputIndex += strides5 * outputIndex[DIM5_INDEX];
        inputIndex += strides6 * outputIndex[DIM6_INDEX];
        inputIndex += strides7 * outputIndex[DIM7_INDEX];
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceSIMT<T, U>::Init(
    GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y, const StridedSliceSIMTTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockNums_ = GetBlockNum();

    for (int32_t i = 0; i < tilingData_->inputDims; i++) {
        if (tilingData_->begin[i] >= 0) {
            begin_[i] = tilingData_->begin[i];
        } else {
            int64_t realIdx = UNCONST_BEGIN_VALUE - tilingData_->begin[i];
            begin_[i] = static_cast<int64_t>(((__gm__ U*)begin)[realIdx]);
            int64_t curInShape = (i == tilingData_->inputDims - 1) ? tilingData_->inputShapeProd[i] :
                (tilingData_->inputShapeProd[i] / tilingData_->inputShapeProd[i + 1]);
            begin_[i] = begin_[i] < 0 ? (begin_[i] + curInShape): begin_[i];
            if (begin_[i] < 0) {
                begin_[i] = 0;
            } else if (begin_[i] >= curInShape) {
                begin_[i] = curInShape - 1;
            }
        }
    }

    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceSIMT<T, U>::ProcessNormalShape(GM_ADDR tiling, uint32_t inputDims)
{
    uint32_t outputSize = tilingData_->outputShapeProd[DIM0_INDEX];
    uint32_t shift[DIMS_8] = {0};
    uint32_t m[DIMS_8] = {0};
    for (uint32_t i = 0; i < inputDims - 1; i++) {
        GetUintDivMagicAndShift(m[i], shift[i], uint32_t(tilingData_->outputShapeProd[i + 1]));
    }

    uint32_t strideTmp[DIMS_8] = {0};
    uint32_t beginSum = 0;
    for (uint32_t i = 0; i < inputDims - 1; i++) {
        beginSum += begin_[i] * tilingData_->inputShapeProd[i + 1];
        strideTmp[i] = tilingData_->strides[i] * tilingData_->inputShapeProd[i + 1];
    }
    strideTmp[inputDims - 1] = tilingData_->strides[inputDims - 1];
    beginSum += begin_[inputDims - 1];


    if (inputDims == DIMS_1) {
        asc_vf_call<StridedSliceDim1Compute<T, DIMS_1>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            outputSize, int32_t(begin_[DIM0_INDEX]), int32_t(tilingData_->strides[DIM0_INDEX]), blockIdx_,
            blockNums_);
    } else if (inputDims == DIMS_2) {
        asc_vf_call<StridedSliceDim2Compute<T, DIMS_2>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            outputSize, tilingData_->outputShapeProd[DIM1_INDEX], tilingData_->inputShapeProd[DIM1_INDEX],
            int32_t(begin_[DIM0_INDEX]), int32_t(begin_[DIM1_INDEX]),
            int32_t(tilingData_->strides[DIM0_INDEX]), int32_t(tilingData_->strides[DIM1_INDEX]), m[DIM0_INDEX],
            shift[DIM0_INDEX], blockIdx_, blockNums_);
    } else if (inputDims == DIMS_3) {
        asc_vf_call<StridedSliceDim3Compute<T, DIMS_3>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            outputSize, tilingData_->outputShapeProd[DIM1_INDEX], tilingData_->outputShapeProd[DIM2_INDEX],
            tilingData_->inputShapeProd[DIM1_INDEX], tilingData_->inputShapeProd[DIM2_INDEX],
            int32_t(begin_[DIM0_INDEX]), int32_t(begin_[DIM1_INDEX]),
            int32_t(begin_[DIM2_INDEX]), int32_t(tilingData_->strides[DIM0_INDEX]),
            int32_t(tilingData_->strides[DIM1_INDEX]), int32_t(tilingData_->strides[DIM2_INDEX]), m[DIM0_INDEX],
            m[DIM1_INDEX], shift[DIM0_INDEX], shift[DIM1_INDEX], blockIdx_, blockNums_);
    } else if (inputDims == DIMS_4) {
        asc_vf_call<StridedSliceDim4Compute<T, DIMS_4>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            outputSize, tilingData_->outputShapeProd[DIM1_INDEX], tilingData_->outputShapeProd[DIM2_INDEX],
            tilingData_->outputShapeProd[DIM3_INDEX], tilingData_->inputShapeProd[DIM1_INDEX],
            tilingData_->inputShapeProd[DIM2_INDEX], tilingData_->inputShapeProd[DIM3_INDEX],
            int32_t(begin_[DIM0_INDEX]), int32_t(begin_[DIM1_INDEX]),
            int32_t(begin_[DIM2_INDEX]), int32_t(begin_[DIM3_INDEX]),
            int32_t(tilingData_->strides[DIM0_INDEX]), int32_t(tilingData_->strides[DIM1_INDEX]),
            int32_t(tilingData_->strides[DIM2_INDEX]), int32_t(tilingData_->strides[DIM3_INDEX]), m[DIM0_INDEX],
            m[DIM1_INDEX], m[DIM2_INDEX], shift[DIM0_INDEX], shift[DIM1_INDEX], shift[DIM2_INDEX], blockIdx_,
            blockNums_);
    } else if (inputDims == DIMS_5) {
        asc_vf_call<StridedSliceDim5Compute<T, DIMS_5>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_, beginSum,
            int32_t(strideTmp[DIM0_INDEX]), int32_t(strideTmp[DIM1_INDEX]),
            int32_t(strideTmp[DIM2_INDEX]), int32_t(strideTmp[DIM3_INDEX]), int32_t(strideTmp[DIM4_INDEX]), m[DIM0_INDEX],
            m[DIM1_INDEX], m[DIM2_INDEX], m[DIM3_INDEX], shift[DIM0_INDEX], shift[DIM1_INDEX], shift[DIM2_INDEX],
            shift[DIM3_INDEX]);
    } else if (inputDims == DIMS_6) {
        asc_vf_call<StridedSliceDim6Compute<T, DIMS_6>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_, beginSum,
            int32_t(strideTmp[DIM0_INDEX]), int32_t(strideTmp[DIM1_INDEX]),
            int32_t(strideTmp[DIM2_INDEX]), int32_t(strideTmp[DIM3_INDEX]), int32_t(strideTmp[DIM4_INDEX]), int32_t(strideTmp[DIM5_INDEX]), m[DIM0_INDEX],
            m[DIM1_INDEX], m[DIM2_INDEX], m[DIM3_INDEX], m[DIM4_INDEX], shift[DIM0_INDEX], shift[DIM1_INDEX],
            shift[DIM2_INDEX], shift[DIM3_INDEX], shift[DIM4_INDEX]);
    } else if (inputDims == DIMS_7) {
        asc_vf_call<StridedSliceDim7Compute<T, DIMS_7>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_, beginSum,
            int32_t(strideTmp[DIM0_INDEX]), int32_t(strideTmp[DIM1_INDEX]),
            int32_t(strideTmp[DIM2_INDEX]), int32_t(strideTmp[DIM3_INDEX]), int32_t(strideTmp[DIM4_INDEX]), int32_t(strideTmp[DIM5_INDEX]), 
            int32_t(strideTmp[DIM6_INDEX]), m[DIM0_INDEX],
            m[DIM1_INDEX], m[DIM2_INDEX], m[DIM3_INDEX], m[DIM4_INDEX], m[DIM5_INDEX], shift[DIM0_INDEX],
            shift[DIM1_INDEX], shift[DIM2_INDEX], shift[DIM3_INDEX], shift[DIM4_INDEX], shift[DIM5_INDEX]);
    } else if (inputDims == DIMS_8) {
        asc_vf_call<StridedSliceDim8Compute<T, DIMS_8>>(
            dim3(HALF_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_, beginSum,
            int32_t(strideTmp[DIM0_INDEX]), int32_t(strideTmp[DIM1_INDEX]),
            int32_t(strideTmp[DIM2_INDEX]), int32_t(strideTmp[DIM3_INDEX]), int32_t(strideTmp[DIM4_INDEX]), int32_t(strideTmp[DIM5_INDEX]), 
            int32_t(strideTmp[DIM6_INDEX]), int32_t(strideTmp[DIM7_INDEX]), m[DIM0_INDEX],
            m[DIM1_INDEX], m[DIM2_INDEX], m[DIM3_INDEX], m[DIM4_INDEX], m[DIM5_INDEX], m[DIM6_INDEX], shift[DIM0_INDEX],
            shift[DIM1_INDEX], shift[DIM2_INDEX], shift[DIM3_INDEX], shift[DIM4_INDEX], shift[DIM5_INDEX],
            shift[DIM6_INDEX]);
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceSIMT<T, U>::Process(GM_ADDR tiling)
{
    if (blockIdx_ >= blockNums_ || tilingData_->isEmptyTensor) {
        return;
    }

    ProcessNormalShape(tiling, tilingData_->inputDims);
}

} // namespace StridedSlice

#endif // STRIDED_SLICE_SIMT_H