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
* \file as_strided_simt.h
* \brief as_strided_simt
*/

#ifndef AS_STRIDED_SIMT_H_
#define AS_STRIDED_SIMT_H_

#include "as_strided.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace AsStrided {
using namespace AscendC;

constexpr int64_t THREAD_NUM = 512;
constexpr uint16_t DIMS_1 = 1;
constexpr uint16_t DIMS_2 = 2;
constexpr uint16_t DIMS_3 = 3;
constexpr uint16_t DIMS_4 = 4;
constexpr uint16_t DIMS_5 = 5;
constexpr uint16_t DIMS_6 = 6;
constexpr uint16_t DIMS_7 = 7;
constexpr uint16_t DIMS_8 = 8;

template <typename T>
class AsStridedSimt
{
public:
    __aicore__ inline AsStridedSimt(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const AsStridedSimtTilingData* tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    const AsStridedSimtTilingData* tilingData_;
    TPipe pipe_;

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    uint32_t dimNum_ = 0;
    uint32_t blockIdx_ = 0;

    int64_t storageOffset_ = 0;
    int64_t curCoreBaseIndex_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t perCoreElements_ = 0;
};

template <typename T>
__aicore__ inline void AsStridedSimt<T>::Init(GM_ADDR input, GM_ADDR output, const AsStridedSimtTilingData* tilingData)
{
    tilingData_ = tilingData;
    storageOffset_ = tilingData->storageOffset;
    dimNum_ = tilingData_->outDimNum;
    inputGm_.SetGlobalBuffer((__gm__ T*)(input) + storageOffset_);
    outputGm_.SetGlobalBuffer((__gm__ T*)output);

    blockIdx_ = GetBlockIdx();
    perCoreElements_ = tilingData_->mainBlockFactor;
    if(blockIdx_ == GetBlockNum() - 1) {
        curCoreElements_ = tilingData_->tailBlockFactor;
    } else {
        curCoreElements_ = perCoreElements_;
    }
    curCoreBaseIndex_ = perCoreElements_ * blockIdx_;
}
template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim1(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0)
{  
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputDimIdx = outputIdx;
        uint32_t inputIdx = strideArr0 * inputDimIdx;

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}

template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim2(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0,
    uint32_t strideArr1, uint32_t sizeStride0, uint32_t m0, uint32_t s0)
{  
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputCurIdx = outputIdx;
        uint32_t inputDimIdx[Dim] = {0};
        inputDimIdx[0] = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= inputDimIdx[0] * sizeStride0;
        inputDimIdx[1] = inputCurIdx;

        uint32_t inputIdx = 0;
        inputIdx += strideArr0 * inputDimIdx[0];
        inputIdx += strideArr1 * inputDimIdx[1];

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}

template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim3(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0,
    uint32_t strideArr1, uint32_t strideArr2, uint32_t sizeStride0, uint32_t sizeStride1, uint32_t m0, uint32_t s0, uint32_t m1, uint32_t s1)
{  
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputCurIdx = outputIdx;
        uint32_t inputDimIdx[Dim] = {0};
        inputDimIdx[0] = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= inputDimIdx[0] * sizeStride0;
        inputDimIdx[1] = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= inputDimIdx[1] * sizeStride1;
        inputDimIdx[2] = inputCurIdx;

        uint32_t inputIdx = 0;
        inputIdx += strideArr0 * inputDimIdx[0];
        inputIdx += strideArr1 * inputDimIdx[1];
        inputIdx += strideArr2 * inputDimIdx[2];

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}

template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim4(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0, 
    uint32_t strideArr1, uint32_t strideArr2, uint32_t strideArr3, uint32_t sizeStride0, uint32_t sizeStride1, uint32_t sizeStride2,
    uint32_t m0, uint32_t s0, uint32_t m1, uint32_t s1, uint32_t m2, uint32_t s2)
{  
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputCurIdx = outputIdx;
        uint32_t inputDimIdx[Dim] = {0};
        inputDimIdx[0] = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= inputDimIdx[0] * sizeStride0;
        inputDimIdx[1] = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= inputDimIdx[1] * sizeStride1;
        inputDimIdx[2] = Simt::UintDiv(inputCurIdx, m2, s2);
        inputCurIdx -= inputDimIdx[2] * sizeStride2;
        inputDimIdx[3] = inputCurIdx;

        uint32_t inputIdx = 0;
        inputIdx += strideArr0 * inputDimIdx[0];
        inputIdx += strideArr1 * inputDimIdx[1];
        inputIdx += strideArr2 * inputDimIdx[2];
        inputIdx += strideArr3 * inputDimIdx[3];

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}

template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim5(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, GM_ADDR tiling, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0,
    uint32_t strideArr1, uint32_t strideArr2, uint32_t strideArr3, uint32_t strideArr4, uint32_t m0, uint32_t s0, uint32_t m1, uint32_t s1,
    uint32_t m2, uint32_t s2, uint32_t m3, uint32_t s3)
{  
    GET_TILING_DATA_PTR_WITH_STRUCT(AsStridedSimtTilingData, tdGmPtr, tiling);
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputCurIdx = outputIdx;
        uint32_t inputDimIdx[Dim] = {0};
        inputDimIdx[0] = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= inputDimIdx[0] * tdGmPtr->outSizeStride[0];
        inputDimIdx[1] = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= inputDimIdx[1] * tdGmPtr->outSizeStride[1];
        inputDimIdx[2] = Simt::UintDiv(inputCurIdx, m2, s2);
        inputCurIdx -= inputDimIdx[2] * tdGmPtr->outSizeStride[2];
        inputDimIdx[3] = Simt::UintDiv(inputCurIdx, m3, s3);
        inputCurIdx -= inputDimIdx[3] * tdGmPtr->outSizeStride[3];
        inputDimIdx[4] = inputCurIdx;

        uint32_t inputIdx = 0;
        inputIdx += strideArr0 * inputDimIdx[0];
        inputIdx += strideArr1 * inputDimIdx[1];
        inputIdx += strideArr2 * inputDimIdx[2];
        inputIdx += strideArr3 * inputDimIdx[3];
        inputIdx += strideArr4 * inputDimIdx[4];

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}

template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim6(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, GM_ADDR tiling, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0,
    uint32_t strideArr1, uint32_t strideArr2, uint32_t strideArr3, uint32_t strideArr4, uint32_t strideArr5, uint32_t m0, uint32_t s0,
    uint32_t m1, uint32_t s1, uint32_t m2, uint32_t s2, uint32_t m3, uint32_t s3, uint32_t m4, uint32_t s4)
{  
    GET_TILING_DATA_PTR_WITH_STRUCT(AsStridedSimtTilingData, tdGmPtr, tiling);
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputCurIdx = outputIdx;
        uint32_t inputDimIdx[Dim] = {0};
        inputDimIdx[0] = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= inputDimIdx[0] * tdGmPtr->outSizeStride[0];
        inputDimIdx[1] = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= inputDimIdx[1] * tdGmPtr->outSizeStride[1];
        inputDimIdx[2] = Simt::UintDiv(inputCurIdx, m2, s2);
        inputCurIdx -= inputDimIdx[2] * tdGmPtr->outSizeStride[2];
        inputDimIdx[3] = Simt::UintDiv(inputCurIdx, m3, s3);
        inputCurIdx -= inputDimIdx[3] * tdGmPtr->outSizeStride[3];
        inputDimIdx[4] = Simt::UintDiv(inputCurIdx, m4, s4);
        inputCurIdx -= inputDimIdx[4] * tdGmPtr->outSizeStride[4];
        inputDimIdx[5] = inputCurIdx;

        uint32_t inputIdx = 0;
        inputIdx += strideArr0 * inputDimIdx[0];
        inputIdx += strideArr1 * inputDimIdx[1];
        inputIdx += strideArr2 * inputDimIdx[2];
        inputIdx += strideArr3 * inputDimIdx[3];
        inputIdx += strideArr4 * inputDimIdx[4];
        inputIdx += strideArr5 * inputDimIdx[5];

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}

template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim7(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, GM_ADDR tiling, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0,
    uint32_t strideArr1, uint32_t strideArr2, uint32_t strideArr3, uint32_t strideArr4, uint32_t strideArr5, uint32_t strideArr6,uint32_t m0, uint32_t s0,
    uint32_t m1, uint32_t s1, uint32_t m2, uint32_t s2, uint32_t m3, uint32_t s3, uint32_t m4, uint32_t s4, uint32_t m5, uint32_t s5)
{  
    GET_TILING_DATA_PTR_WITH_STRUCT(AsStridedSimtTilingData, tdGmPtr, tiling);
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputCurIdx = outputIdx;
        uint32_t inputDimIdx[Dim] = {0};
        inputDimIdx[0] = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= inputDimIdx[0] * tdGmPtr->outSizeStride[0];
        inputDimIdx[1] = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= inputDimIdx[1] * tdGmPtr->outSizeStride[1];
        inputDimIdx[2] = Simt::UintDiv(inputCurIdx, m2, s2);
        inputCurIdx -= inputDimIdx[2] * tdGmPtr->outSizeStride[2];
        inputDimIdx[3] = Simt::UintDiv(inputCurIdx, m3, s3);
        inputCurIdx -= inputDimIdx[3] * tdGmPtr->outSizeStride[3];
        inputDimIdx[4] = Simt::UintDiv(inputCurIdx, m4, s4);
        inputCurIdx -= inputDimIdx[4] * tdGmPtr->outSizeStride[4];
        inputDimIdx[5] = Simt::UintDiv(inputCurIdx, m5, s5);
        inputCurIdx -= inputDimIdx[5] * tdGmPtr->outSizeStride[5];
        inputDimIdx[6] = inputCurIdx;

        uint32_t inputIdx = 0;
        inputIdx += strideArr0 * inputDimIdx[0];
        inputIdx += strideArr1 * inputDimIdx[1];
        inputIdx += strideArr2 * inputDimIdx[2];
        inputIdx += strideArr3 * inputDimIdx[3];
        inputIdx += strideArr4 * inputDimIdx[4];
        inputIdx += strideArr5 * inputDimIdx[5];
        inputIdx += strideArr6 * inputDimIdx[6];

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}


template <typename T, uint16_t Dim>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtStridedDim8(
    __gm__ T* inputGmAddr,  __gm__ T* outputGmAddr, GM_ADDR tiling, uint32_t outputBasicIndex, uint32_t count, uint32_t strideArr0,
    uint32_t strideArr1, uint32_t strideArr2, uint32_t strideArr3, uint32_t strideArr4, uint32_t strideArr5, uint32_t strideArr6,
    uint32_t strideArr7, uint32_t m0, uint32_t s0, uint32_t m1, uint32_t s1, uint32_t m2, uint32_t s2, uint32_t m3, uint32_t s3,
    uint32_t m4, uint32_t s4, uint32_t m5, uint32_t s5, uint32_t m6, uint32_t s6)
{  
    GET_TILING_DATA_PTR_WITH_STRUCT(AsStridedSimtTilingData, tdGmPtr, tiling);
    for(uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t outputIdx = outputBasicIndex + index;
        uint32_t inputCurIdx = outputIdx;
        uint32_t inputDimIdx[Dim] = {0};
        inputDimIdx[0] = Simt::UintDiv(inputCurIdx, m0, s0);
        inputCurIdx -= inputDimIdx[0] * tdGmPtr->outSizeStride[0];
        inputDimIdx[1] = Simt::UintDiv(inputCurIdx, m1, s1);
        inputCurIdx -= inputDimIdx[1] * tdGmPtr->outSizeStride[1];
        inputDimIdx[2] = Simt::UintDiv(inputCurIdx, m2, s2);
        inputCurIdx -= inputDimIdx[2] * tdGmPtr->outSizeStride[2];
        inputDimIdx[3] = Simt::UintDiv(inputCurIdx, m3, s3);
        inputCurIdx -= inputDimIdx[3] * tdGmPtr->outSizeStride[3];
        inputDimIdx[4] = Simt::UintDiv(inputCurIdx, m4, s4);
        inputCurIdx -= inputDimIdx[4] * tdGmPtr->outSizeStride[4];
        inputDimIdx[5] = Simt::UintDiv(inputCurIdx, m5, s5);
        inputCurIdx -= inputDimIdx[5] * tdGmPtr->outSizeStride[5];
        inputDimIdx[6] = Simt::UintDiv(inputCurIdx, m6, s6);
        inputCurIdx -= inputDimIdx[6] * tdGmPtr->outSizeStride[6];
        inputDimIdx[7] = inputCurIdx;

        uint32_t inputIdx = 0;
        inputIdx += strideArr0 * inputDimIdx[0];
        inputIdx += strideArr1 * inputDimIdx[1];
        inputIdx += strideArr2 * inputDimIdx[2];
        inputIdx += strideArr3 * inputDimIdx[3];
        inputIdx += strideArr4 * inputDimIdx[4];
        inputIdx += strideArr5 * inputDimIdx[5];
        inputIdx += strideArr6 * inputDimIdx[6];
        inputIdx += strideArr7 * inputDimIdx[7];

        outputGmAddr[outputIdx] = inputGmAddr[inputIdx];
    }
}


template <typename T>
__aicore__ inline void AsStridedSimt<T>::Process(GM_ADDR tiling)
{
    uint32_t magic[DIMS_8] = {0};
    uint32_t shift[DIMS_8] = {0};
    for(int32_t dim = 0; dim < dimNum_ - 1; dim++) {
        GetUintDivMagicAndShift(magic[dim], shift[dim], tilingData_->outSizeStride[dim]);
    }

    __gm__ T* inputGmAddr = (__gm__ T*)inputGm_.GetPhyAddr();
    __gm__ T* outputGmAddr = (__gm__ T*)outputGm_.GetPhyAddr();
    
    if (dimNum_ == DIMS_1) {
        asc_vf_call<SimtStridedDim1<T, DIMS_1>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0]); 
    } else if (dimNum_ == DIMS_2) {
        asc_vf_call<SimtStridedDim2<T, DIMS_2>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0],
            tilingData_->strideArr[1], tilingData_->outSizeStride[0], magic[0], shift[0]); 
    } else if (dimNum_ == DIMS_3) {
        asc_vf_call<SimtStridedDim3<T, DIMS_3>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0],
            tilingData_->strideArr[1], tilingData_->strideArr[2], tilingData_->outSizeStride[0], tilingData_->outSizeStride[1],
            magic[0], shift[0], magic[1], shift[1]); 
    } else if (dimNum_ == DIMS_4) {
        asc_vf_call<SimtStridedDim4<T, DIMS_4>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0],
            tilingData_->strideArr[1], tilingData_->strideArr[2], tilingData_->strideArr[3], tilingData_->outSizeStride[0],
            tilingData_->outSizeStride[1], tilingData_->outSizeStride[2], magic[0], shift[0], magic[1], shift[1], magic[2], shift[2]); 
    } else if (dimNum_ == DIMS_5) {
        asc_vf_call<SimtStridedDim5<T, DIMS_5>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, tiling, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0],
            tilingData_->strideArr[1], tilingData_->strideArr[2], tilingData_->strideArr[3], tilingData_->strideArr[4],
            magic[0], shift[0], magic[1], shift[1], magic[2], shift[2], magic[3], shift[3]); 
    } else if (dimNum_ == DIMS_6) {
        asc_vf_call<SimtStridedDim6<T, DIMS_6>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, tiling, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0],
            tilingData_->strideArr[1], tilingData_->strideArr[2], tilingData_->strideArr[3], tilingData_->strideArr[4],
            tilingData_->strideArr[5], magic[0], shift[0], magic[1], shift[1], magic[2], shift[2], magic[3], shift[3], magic[4], shift[4]); 
    } else if (dimNum_ == DIMS_7) {
        asc_vf_call<SimtStridedDim7<T, DIMS_7>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, tiling, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0],
            tilingData_->strideArr[1], tilingData_->strideArr[2], tilingData_->strideArr[3], tilingData_->strideArr[4],
            tilingData_->strideArr[5], tilingData_->strideArr[6], magic[0], shift[0], magic[1], shift[1], magic[2], shift[2],
            magic[3], shift[3], magic[4], shift[4], magic[5], shift[5]); 
    } else if (dimNum_ == DIMS_8) {
        asc_vf_call<SimtStridedDim8<T, DIMS_8>>(
            dim3{THREAD_NUM, 1, 1}, inputGmAddr, outputGmAddr, tiling, curCoreBaseIndex_, curCoreElements_, tilingData_->strideArr[0],
            tilingData_->strideArr[1], tilingData_->strideArr[2], tilingData_->strideArr[3], tilingData_->strideArr[4], 
            tilingData_->strideArr[5], tilingData_->strideArr[6], tilingData_->strideArr[7], magic[0], shift[0], magic[1], shift[1],
            magic[2], shift[2], magic[3], shift[3], magic[4], shift[4], magic[5], shift[5], magic[6], shift[6]);   
    }
}

}

#endif
