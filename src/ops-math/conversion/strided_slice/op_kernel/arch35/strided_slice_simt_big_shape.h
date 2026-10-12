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
 * \file strided_slice_simt_big_shape.h
 * \brief
 */
#ifndef STRIDED_SLICE_SIMT_BIG_SHAPE_H
#define STRIDED_SLICE_SIMT_BIG_SHAPE_H

#include "strided_slice_base.h"

namespace StridedSlice {
using namespace AscendC;

constexpr int64_t SIMT_PARAM_LEN = 2 * STRIDED_SLICE_MAX_AXIS_NUM * sizeof(int64_t);

template <typename T>
struct FastDivParam {
    T magic0 = 0;
    T magic1 = 0;
    T magic2 = 0;
    T magic3 = 0;
    T magic4 = 0;
    T magic5 = 0;
    T magic6 = 0;
    T shift0 = 0;
    T shift1 = 0;
    T shift2 = 0;
    T shift3 = 0;
    T shift4 = 0;
    T shift5 = 0;
    T shift6 = 0;
};

template <typename T, typename U>
class StridedSliceSIMTBigShape : public StridedSliceBase<T, U>
{
public:
    __aicore__ inline StridedSliceSIMTBigShape(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y, const StridedSliceSIMTTilingData* tilingData,
        TPipe* pipeIn);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    __aicore__ inline void ProcessBigShape(GM_ADDR tiling, uint32_t inputDims);
    __aicore__ inline void SetSimtDivMagic(LocalTensor<uint64_t>& simtParamLocal, uint32_t inputDims);

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    TBuf<TPosition::VECCALC> simtParamBuf_;

    int32_t blockIdx_ = 0;
    int32_t blockNums_ = 1;
    int64_t begin_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};
    TPipe* pipe_;
    const StridedSliceSIMTTilingData* tilingData_;
};

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void StridedSliceDim1B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, const uint64_t outputSize, const int32_t begin0,
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
__simt_vf__ LAUNCH_BOUND(HALF_THREAD_DIM) __aicore__ void StridedSliceDim2B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint64_t outputSize, int32_t blockIdx,
    int32_t blockNums, __ubuf__ uint64_t* fastDivBuf, int64_t begin0, int64_t begin1)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int64_t outputIndex[Dim] = {0};
        uint64_t dstIdx = idx;
        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM0_INDEX], fastDivBuf[DIM0_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM0_INDEX] * tdGmPtr->outputShapeProd[DIM1_INDEX];

        outputIndex[Dim - 1] = dstIdx;

        uint64_t inputIndex = 0;
        inputIndex += (begin0 + tdGmPtr->strides[DIM0_INDEX] * outputIndex[DIM0_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM1_INDEX];
        inputIndex += (begin1 + tdGmPtr->strides[DIM1_INDEX] * outputIndex[DIM1_INDEX]);

        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(QUARTER_THREAD_DIM) __aicore__ void StridedSliceDim3B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint64_t outputSize, int32_t blockIdx,
    int32_t blockNums, __ubuf__ uint64_t* fastDivBuf, int64_t begin0, int64_t begin1, int64_t begin2)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int64_t outputIndex[Dim] = {0};
        uint64_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM0_INDEX], fastDivBuf[DIM0_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM0_INDEX] * tdGmPtr->outputShapeProd[DIM1_INDEX];
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM1_INDEX], fastDivBuf[DIM1_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM1_INDEX] * tdGmPtr->outputShapeProd[DIM2_INDEX];

        outputIndex[Dim - 1] = dstIdx;

        uint64_t inputIndex = 0;
        inputIndex += (begin0 + tdGmPtr->strides[DIM0_INDEX] * outputIndex[DIM0_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM1_INDEX];
        inputIndex += (begin1 + tdGmPtr->strides[DIM1_INDEX] * outputIndex[DIM1_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM2_INDEX];
        inputIndex += (begin2 + tdGmPtr->strides[DIM2_INDEX] * outputIndex[DIM2_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(QUARTER_THREAD_DIM) __aicore__ void StridedSliceDim4B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint64_t outputSize, int32_t blockIdx,
    int32_t blockNums, __ubuf__ uint64_t* fastDivBuf, int64_t begin0, int64_t begin1, int64_t begin2, int64_t begin3)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int64_t outputIndex[Dim] = {0};
        uint64_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM0_INDEX], fastDivBuf[DIM0_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM0_INDEX] * tdGmPtr->outputShapeProd[DIM1_INDEX];
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM1_INDEX], fastDivBuf[DIM1_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM1_INDEX] * tdGmPtr->outputShapeProd[DIM2_INDEX];
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM2_INDEX], fastDivBuf[DIM2_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM2_INDEX] * tdGmPtr->outputShapeProd[DIM3_INDEX];

        outputIndex[Dim - 1] = dstIdx;

        uint64_t inputIndex = 0;
        inputIndex += (begin0 + tdGmPtr->strides[DIM0_INDEX] * outputIndex[DIM0_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM1_INDEX];
        inputIndex += (begin1 + tdGmPtr->strides[DIM1_INDEX] * outputIndex[DIM1_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM2_INDEX];
        inputIndex += (begin2 + tdGmPtr->strides[DIM2_INDEX] * outputIndex[DIM2_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM3_INDEX];
        inputIndex += (begin3 + tdGmPtr->strides[DIM3_INDEX] * outputIndex[DIM3_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(AN_EIGHTH_THREAD_DIM) __aicore__ void StridedSliceDim5B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint64_t outputSize, int32_t blockIdx,
    int32_t blockNums, __ubuf__ uint64_t* fastDivBuf, int64_t begin0, int64_t begin1, int64_t begin2,
    int64_t begin3, int64_t begin4)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int64_t outputIndex[Dim] = {0};
        uint64_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM0_INDEX], fastDivBuf[DIM0_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM0_INDEX] * tdGmPtr->outputShapeProd[DIM1_INDEX];
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM1_INDEX], fastDivBuf[DIM1_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM1_INDEX] * tdGmPtr->outputShapeProd[DIM2_INDEX];
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM2_INDEX], fastDivBuf[DIM2_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM2_INDEX] * tdGmPtr->outputShapeProd[DIM3_INDEX];
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM3_INDEX], fastDivBuf[DIM3_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM3_INDEX] * tdGmPtr->outputShapeProd[DIM4_INDEX];

        outputIndex[Dim - 1] = dstIdx;

        uint64_t inputIndex = 0;
        inputIndex += (begin0 + tdGmPtr->strides[DIM0_INDEX] * outputIndex[DIM0_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM1_INDEX];
        inputIndex += (begin1 + tdGmPtr->strides[DIM1_INDEX] * outputIndex[DIM1_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM2_INDEX];
        inputIndex += (begin2 + tdGmPtr->strides[DIM2_INDEX] * outputIndex[DIM2_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM3_INDEX];
        inputIndex += (begin3 + tdGmPtr->strides[DIM3_INDEX] * outputIndex[DIM3_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM4_INDEX];
        inputIndex += (begin4 + tdGmPtr->strides[DIM4_INDEX] * outputIndex[DIM4_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(AN_EIGHTH_THREAD_DIM) __aicore__ void StridedSliceDim6B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint64_t outputSize, int32_t blockIdx,
    int32_t blockNums, __ubuf__ uint64_t* fastDivBuf, int64_t begin0, int64_t begin1, int64_t begin2,
    int64_t begin3, int64_t begin4, int64_t begin5)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int64_t outputIndex[Dim] = {0};
        uint64_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM0_INDEX], fastDivBuf[DIM0_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM0_INDEX] * tdGmPtr->outputShapeProd[DIM1_INDEX];
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM1_INDEX], fastDivBuf[DIM1_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM1_INDEX] * tdGmPtr->outputShapeProd[DIM2_INDEX];
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM2_INDEX], fastDivBuf[DIM2_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM2_INDEX] * tdGmPtr->outputShapeProd[DIM3_INDEX];
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM3_INDEX], fastDivBuf[DIM3_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM3_INDEX] * tdGmPtr->outputShapeProd[DIM4_INDEX];
        outputIndex[DIM4_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM4_INDEX], fastDivBuf[DIM4_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM4_INDEX] * tdGmPtr->outputShapeProd[DIM5_INDEX];

        outputIndex[Dim - 1] = dstIdx;

        uint64_t inputIndex = 0;
        inputIndex += (begin0 + tdGmPtr->strides[DIM0_INDEX] * outputIndex[DIM0_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM1_INDEX];
        inputIndex += (begin1 + tdGmPtr->strides[DIM1_INDEX] * outputIndex[DIM1_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM2_INDEX];
        inputIndex += (begin2 + tdGmPtr->strides[DIM2_INDEX] * outputIndex[DIM2_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM3_INDEX];
        inputIndex += (begin3 + tdGmPtr->strides[DIM3_INDEX] * outputIndex[DIM3_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM4_INDEX];
        inputIndex += (begin4 + tdGmPtr->strides[DIM4_INDEX] * outputIndex[DIM4_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM5_INDEX];
        inputIndex += (begin5 + tdGmPtr->strides[DIM5_INDEX] * outputIndex[DIM5_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(AN_EIGHTH_THREAD_DIM) __aicore__ void StridedSliceDim7B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint64_t outputSize, int32_t blockIdx,
    int32_t blockNums, __ubuf__ uint64_t* fastDivBuf, int64_t begin0, int64_t begin1, int64_t begin2,
    int64_t begin3, int64_t begin4, int64_t begin5, int64_t begin6)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int64_t outputIndex[Dim] = {0};
        uint64_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM0_INDEX], fastDivBuf[DIM0_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM0_INDEX] * tdGmPtr->outputShapeProd[DIM1_INDEX];
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM1_INDEX], fastDivBuf[DIM1_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM1_INDEX] * tdGmPtr->outputShapeProd[DIM2_INDEX];
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM2_INDEX], fastDivBuf[DIM2_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM2_INDEX] * tdGmPtr->outputShapeProd[DIM3_INDEX];
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM3_INDEX], fastDivBuf[DIM3_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM3_INDEX] * tdGmPtr->outputShapeProd[DIM4_INDEX];
        outputIndex[DIM4_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM4_INDEX], fastDivBuf[DIM4_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM4_INDEX] * tdGmPtr->outputShapeProd[DIM5_INDEX];
        outputIndex[DIM5_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM5_INDEX], fastDivBuf[DIM5_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM5_INDEX] * tdGmPtr->outputShapeProd[DIM6_INDEX];

        outputIndex[Dim - 1] = dstIdx;

        uint64_t inputIndex = 0;
        inputIndex += (begin0 + tdGmPtr->strides[DIM0_INDEX] * outputIndex[DIM0_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM1_INDEX];
        inputIndex += (begin1 + tdGmPtr->strides[DIM1_INDEX] * outputIndex[DIM1_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM2_INDEX];
        inputIndex += (begin2 + tdGmPtr->strides[DIM2_INDEX] * outputIndex[DIM2_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM3_INDEX];
        inputIndex += (begin3 + tdGmPtr->strides[DIM3_INDEX] * outputIndex[DIM3_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM4_INDEX];
        inputIndex += (begin4 + tdGmPtr->strides[DIM4_INDEX] * outputIndex[DIM4_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM5_INDEX];
        inputIndex += (begin5 + tdGmPtr->strides[DIM5_INDEX] * outputIndex[DIM5_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM6_INDEX];
        inputIndex += (begin6 + tdGmPtr->strides[DIM6_INDEX] * outputIndex[DIM6_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, int Dim>
__simt_vf__ LAUNCH_BOUND(AN_EIGHTH_THREAD_DIM) __aicore__ void StridedSliceDim8B64Compute(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, GM_ADDR tiling, const uint64_t outputSize, int32_t blockIdx,
    int32_t blockNums, __ubuf__ uint64_t* fastDivBuf, int64_t begin0, int64_t begin1, int64_t begin2,
    int64_t begin3, int64_t begin4, int64_t begin5, int64_t begin6, int64_t begin7)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(StridedSliceSIMTTilingData, tdGmPtr, tiling);
    for (uint64_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNums * blockDim.x) {
        int64_t outputIndex[Dim] = {0};
        uint64_t dstIdx = idx;

        outputIndex[DIM0_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM0_INDEX], fastDivBuf[DIM0_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM0_INDEX] * tdGmPtr->outputShapeProd[DIM1_INDEX];
        outputIndex[DIM1_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM1_INDEX], fastDivBuf[DIM1_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM1_INDEX] * tdGmPtr->outputShapeProd[DIM2_INDEX];
        outputIndex[DIM2_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM2_INDEX], fastDivBuf[DIM2_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM2_INDEX] * tdGmPtr->outputShapeProd[DIM3_INDEX];
        outputIndex[DIM3_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM3_INDEX], fastDivBuf[DIM3_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM3_INDEX] * tdGmPtr->outputShapeProd[DIM4_INDEX];
        outputIndex[DIM4_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM4_INDEX], fastDivBuf[DIM4_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM4_INDEX] * tdGmPtr->outputShapeProd[DIM5_INDEX];
        outputIndex[DIM5_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM5_INDEX], fastDivBuf[DIM5_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM5_INDEX] * tdGmPtr->outputShapeProd[DIM6_INDEX];
        outputIndex[DIM6_INDEX] = Simt::UintDiv(dstIdx, fastDivBuf[DIM6_INDEX], fastDivBuf[DIM6_INDEX + DIM7_INDEX]);
        dstIdx -= outputIndex[DIM6_INDEX] * tdGmPtr->outputShapeProd[DIM7_INDEX];

        outputIndex[Dim - 1] = dstIdx;

        uint64_t inputIndex = 0;
        inputIndex += (begin0 + tdGmPtr->strides[DIM0_INDEX] * outputIndex[DIM0_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM1_INDEX];
        inputIndex += (begin1 + tdGmPtr->strides[DIM1_INDEX] * outputIndex[DIM1_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM2_INDEX];
        inputIndex += (begin2 + tdGmPtr->strides[DIM2_INDEX] * outputIndex[DIM2_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM3_INDEX];
        inputIndex += (begin3 + tdGmPtr->strides[DIM3_INDEX] * outputIndex[DIM3_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM4_INDEX];
        inputIndex += (begin4 + tdGmPtr->strides[DIM4_INDEX] * outputIndex[DIM4_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM5_INDEX];
        inputIndex += (begin5 + tdGmPtr->strides[DIM5_INDEX] * outputIndex[DIM5_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM6_INDEX];
        inputIndex += (begin6 + tdGmPtr->strides[DIM6_INDEX] * outputIndex[DIM6_INDEX]) *
                      tdGmPtr->inputShapeProd[DIM7_INDEX];
        inputIndex += (begin7 + tdGmPtr->strides[DIM7_INDEX] * outputIndex[DIM7_INDEX]);
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceSIMTBigShape<T, U>::Init(
    GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y, const StridedSliceSIMTTilingData* tilingData,
    TPipe* pipeIn)
{
    pipe_ = pipeIn;
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
__aicore__ inline void StridedSliceSIMTBigShape<T, U>::SetSimtDivMagic(
    LocalTensor<uint64_t>& simtParamLocal, uint32_t inputDims)
{
    FastDivParam<uint64_t> fastDivParam;
    if (inputDims == DIMS_8) {
        GetUintDivMagicAndShift(
            fastDivParam.magic0, fastDivParam.shift0, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM1_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic1, fastDivParam.shift1, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM2_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic2, fastDivParam.shift2, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM3_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic3, fastDivParam.shift3, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM4_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic4, fastDivParam.shift4, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM5_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic5, fastDivParam.shift5, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM6_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic6, fastDivParam.shift6, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM7_INDEX]));
        simtParamLocal.SetValue(DIM0_INDEX, fastDivParam.magic0);
        simtParamLocal.SetValue(DIM1_INDEX, fastDivParam.magic1);
        simtParamLocal.SetValue(DIM2_INDEX, fastDivParam.magic2);
        simtParamLocal.SetValue(DIM3_INDEX, fastDivParam.magic3);
        simtParamLocal.SetValue(DIM4_INDEX, fastDivParam.magic4);
        simtParamLocal.SetValue(DIM5_INDEX, fastDivParam.magic5);
        simtParamLocal.SetValue(DIM6_INDEX, fastDivParam.magic6);
        simtParamLocal.SetValue(DIM0_INDEX + DIM7_INDEX, fastDivParam.shift0);
        simtParamLocal.SetValue(DIM1_INDEX + DIM7_INDEX, fastDivParam.shift1);
        simtParamLocal.SetValue(DIM2_INDEX + DIM7_INDEX, fastDivParam.shift2);
        simtParamLocal.SetValue(DIM3_INDEX + DIM7_INDEX, fastDivParam.shift3);
        simtParamLocal.SetValue(DIM4_INDEX + DIM7_INDEX, fastDivParam.shift4);
        simtParamLocal.SetValue(DIM5_INDEX + DIM7_INDEX, fastDivParam.shift5);
        simtParamLocal.SetValue(DIM6_INDEX + DIM7_INDEX, fastDivParam.shift6);
    } else if (inputDims == DIMS_7) {
        GetUintDivMagicAndShift(
            fastDivParam.magic0, fastDivParam.shift0, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM1_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic1, fastDivParam.shift1, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM2_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic2, fastDivParam.shift2, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM3_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic3, fastDivParam.shift3, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM4_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic4, fastDivParam.shift4, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM5_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic5, fastDivParam.shift5, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM6_INDEX]));
        simtParamLocal.SetValue(DIM0_INDEX, fastDivParam.magic0);
        simtParamLocal.SetValue(DIM1_INDEX, fastDivParam.magic1);
        simtParamLocal.SetValue(DIM2_INDEX, fastDivParam.magic2);
        simtParamLocal.SetValue(DIM3_INDEX, fastDivParam.magic3);
        simtParamLocal.SetValue(DIM4_INDEX, fastDivParam.magic4);
        simtParamLocal.SetValue(DIM5_INDEX, fastDivParam.magic5);
        simtParamLocal.SetValue(DIM0_INDEX + DIM7_INDEX, fastDivParam.shift0);
        simtParamLocal.SetValue(DIM1_INDEX + DIM7_INDEX, fastDivParam.shift1);
        simtParamLocal.SetValue(DIM2_INDEX + DIM7_INDEX, fastDivParam.shift2);
        simtParamLocal.SetValue(DIM3_INDEX + DIM7_INDEX, fastDivParam.shift3);
        simtParamLocal.SetValue(DIM4_INDEX + DIM7_INDEX, fastDivParam.shift4);
        simtParamLocal.SetValue(DIM5_INDEX + DIM7_INDEX, fastDivParam.shift5);
    } else if (inputDims == DIMS_6) {
        GetUintDivMagicAndShift(
            fastDivParam.magic0, fastDivParam.shift0, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM1_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic1, fastDivParam.shift1, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM2_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic2, fastDivParam.shift2, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM3_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic3, fastDivParam.shift3, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM4_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic4, fastDivParam.shift4, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM5_INDEX]));
        simtParamLocal.SetValue(DIM0_INDEX, fastDivParam.magic0);
        simtParamLocal.SetValue(DIM1_INDEX, fastDivParam.magic1);
        simtParamLocal.SetValue(DIM2_INDEX, fastDivParam.magic2);
        simtParamLocal.SetValue(DIM3_INDEX, fastDivParam.magic3);
        simtParamLocal.SetValue(DIM4_INDEX, fastDivParam.magic4);
        simtParamLocal.SetValue(DIM0_INDEX + DIM7_INDEX, fastDivParam.shift0);
        simtParamLocal.SetValue(DIM1_INDEX + DIM7_INDEX, fastDivParam.shift1);
        simtParamLocal.SetValue(DIM2_INDEX + DIM7_INDEX, fastDivParam.shift2);
        simtParamLocal.SetValue(DIM3_INDEX + DIM7_INDEX, fastDivParam.shift3);
        simtParamLocal.SetValue(DIM4_INDEX + DIM7_INDEX, fastDivParam.shift4);
    } else if (inputDims == DIMS_5) {
        GetUintDivMagicAndShift(
            fastDivParam.magic0, fastDivParam.shift0, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM1_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic1, fastDivParam.shift1, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM2_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic2, fastDivParam.shift2, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM3_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic3, fastDivParam.shift3, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM4_INDEX]));
        simtParamLocal.SetValue(DIM0_INDEX, fastDivParam.magic0);
        simtParamLocal.SetValue(DIM1_INDEX, fastDivParam.magic1);
        simtParamLocal.SetValue(DIM2_INDEX, fastDivParam.magic2);
        simtParamLocal.SetValue(DIM3_INDEX, fastDivParam.magic3);
        simtParamLocal.SetValue(DIM0_INDEX + DIM7_INDEX, fastDivParam.shift0);
        simtParamLocal.SetValue(DIM1_INDEX + DIM7_INDEX, fastDivParam.shift1);
        simtParamLocal.SetValue(DIM2_INDEX + DIM7_INDEX, fastDivParam.shift2);
        simtParamLocal.SetValue(DIM3_INDEX + DIM7_INDEX, fastDivParam.shift3);
    } else if (inputDims == DIMS_4) {
        GetUintDivMagicAndShift(
            fastDivParam.magic0, fastDivParam.shift0, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM1_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic1, fastDivParam.shift1, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM2_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic2, fastDivParam.shift2, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM3_INDEX]));
        simtParamLocal.SetValue(DIM0_INDEX, fastDivParam.magic0);
        simtParamLocal.SetValue(DIM1_INDEX, fastDivParam.magic1);
        simtParamLocal.SetValue(DIM2_INDEX, fastDivParam.magic2);
        simtParamLocal.SetValue(DIM0_INDEX + DIM7_INDEX, fastDivParam.shift0);
        simtParamLocal.SetValue(DIM1_INDEX + DIM7_INDEX, fastDivParam.shift1);
        simtParamLocal.SetValue(DIM2_INDEX + DIM7_INDEX, fastDivParam.shift2);
    } else if (inputDims == DIMS_3) {
        GetUintDivMagicAndShift(
            fastDivParam.magic0, fastDivParam.shift0, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM1_INDEX]));
        GetUintDivMagicAndShift(
            fastDivParam.magic1, fastDivParam.shift1, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM2_INDEX]));
        simtParamLocal.SetValue(DIM0_INDEX, fastDivParam.magic0);
        simtParamLocal.SetValue(DIM1_INDEX, fastDivParam.magic1);
        simtParamLocal.SetValue(DIM0_INDEX + DIM7_INDEX, fastDivParam.shift0);
        simtParamLocal.SetValue(DIM1_INDEX + DIM7_INDEX, fastDivParam.shift1);
    } else if (inputDims == DIMS_2) {
        GetUintDivMagicAndShift(
            fastDivParam.magic0, fastDivParam.shift0, static_cast<uint64_t>(tilingData_->outputShapeProd[DIM1_INDEX]));
        simtParamLocal.SetValue(DIM0_INDEX, fastDivParam.magic0);
        simtParamLocal.SetValue(DIM0_INDEX + DIM7_INDEX, fastDivParam.shift0);
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceSIMTBigShape<T, U>::ProcessBigShape(GM_ADDR tiling, uint32_t inputDims)
{
    pipe_->InitBuffer(simtParamBuf_, SIMT_PARAM_LEN);
    LocalTensor<uint64_t> simtParamLocal = simtParamBuf_.Get<uint64_t>();
    uint64_t outputSize = tilingData_->outputShapeProd[0];

    SetSimtDivMagic(simtParamLocal, inputDims);    
    DataSyncBarrier<MemDsbT::UB>();

    if (inputDims == DIMS_1) {
        asc_vf_call<StridedSliceDim1B64Compute<T, DIMS_1>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            outputSize, begin_[0], tilingData_->strides[0], blockIdx_, blockNums_);
    } else if (inputDims == DIMS_2) {
        asc_vf_call<StridedSliceDim2B64Compute<T, DIMS_2>>(
            dim3(HALF_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_,
            (__ubuf__ uint64_t*)(simtParamLocal.GetPhyAddr()), begin_[0], begin_[1]);
    } else if (inputDims == DIMS_3) {
        asc_vf_call<StridedSliceDim3B64Compute<T, DIMS_3>>(
            dim3(QUARTER_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_,
            (__ubuf__ uint64_t*)(simtParamLocal.GetPhyAddr()), begin_[0], begin_[1], begin_[DIM2_INDEX]);
    } else if (inputDims == DIMS_4) {
        asc_vf_call<StridedSliceDim4B64Compute<T, DIMS_4>>(
            dim3(QUARTER_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_,
            (__ubuf__ uint64_t*)(simtParamLocal.GetPhyAddr()), begin_[0], begin_[1], begin_[DIM2_INDEX],
            begin_[DIM3_INDEX]);
    } else if (inputDims == DIMS_5) {
        asc_vf_call<StridedSliceDim5B64Compute<T, DIMS_5>>(
            dim3(AN_EIGHTH_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_,
            (__ubuf__ uint64_t*)(simtParamLocal.GetPhyAddr()), begin_[0], begin_[1], begin_[DIM2_INDEX],
            begin_[DIM3_INDEX], begin_[DIM4_INDEX]);
    } else if (inputDims == DIMS_6) {
        asc_vf_call<StridedSliceDim6B64Compute<T, DIMS_6>>(
            dim3(AN_EIGHTH_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_,
            (__ubuf__ uint64_t*)(simtParamLocal.GetPhyAddr()), begin_[0], begin_[1], begin_[DIM2_INDEX],
            begin_[DIM3_INDEX], begin_[DIM4_INDEX], begin_[DIM5_INDEX]);
    } else if (inputDims == DIMS_7) {
        asc_vf_call<StridedSliceDim7B64Compute<T, DIMS_7>>(
            dim3(AN_EIGHTH_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_,
            (__ubuf__ uint64_t*)(simtParamLocal.GetPhyAddr()), begin_[0], begin_[1], begin_[DIM2_INDEX],
            begin_[DIM3_INDEX], begin_[DIM4_INDEX], begin_[DIM5_INDEX], begin_[DIM6_INDEX]);
    } else if (inputDims == DIMS_8) {
        asc_vf_call<StridedSliceDim8B64Compute<T, DIMS_8>>(
            dim3(AN_EIGHTH_THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), tiling, outputSize, blockIdx_, blockNums_,
            (__ubuf__ uint64_t*)(simtParamLocal.GetPhyAddr()), begin_[0], begin_[1], begin_[DIM2_INDEX],
            begin_[DIM3_INDEX], begin_[DIM4_INDEX], begin_[DIM5_INDEX], begin_[DIM6_INDEX], begin_[DIM7_INDEX]);
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceSIMTBigShape<T, U>::Process(GM_ADDR tiling)
{
    if (blockIdx_ >= blockNums_ || tilingData_->isEmptyTensor) {
        return;
    }

    ProcessBigShape(tiling, tilingData_->inputDims);
}

} // namespace StridedSlice

#endif // STRIDED_SLICE_SIMT_BIG_SHAPE_H