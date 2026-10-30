/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _BATCH_TO_SPACE_N_D_SIMT_H_
#define _BATCH_TO_SPACE_N_D_SIMT_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "batch_to_space_nd_tiling_data.h"

#include "simt_api/asc_simt.h"
namespace B2SND {
using namespace AscendC;
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
constexpr uint32_t HALF_THREAD_NUM_LAUNCH_BOUND = 256;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;
constexpr uint32_t HALF_THREAD_NUM_LAUNCH_BOUND = 1024;
#endif
template <typename T, typename U>
class BatchToSpaceNDSIMT {
public:
    __aicore__ inline BatchToSpaceNDSIMT(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR block_shape, GM_ADDR crops, GM_ADDR y,
                                const B2SNDSimtTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int32_t SIMT_BUFFER_SIZE = 32 * 1024;
    const B2SNDSimtTilingData* mTD;
    TQue<QuePosition::VECOUT, 1> que_;
    LocalTensor<T> inTensorY_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint32_t blockIdx_ = 0;
    uint32_t threadNum_;
    // 分核信息
    uint32_t realCoreNum_;
    uint32_t mainCoreNum_;
    uint32_t tailBlockSize_;
    uint32_t blockSize_;
    uint64_t mainCoreBlock_;
    int64_t curBlockLoopNumStart_ = 0;
    U batchSize_; // N
    U channel_;   // C
    U blockDim_;

    __aicore__ inline void ParseSIMTTilingData(const B2SNDSimtTilingData* tilingData);
};

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void SimtComputeDimOne(
    U batchSize, U channel, uint64_t curCoreElement, U startIdx, U yShape1, U xShape1, U bShape0, U crops0, U m0, U s0,
    U m1, U s1, U m2, U s2, __gm__ T* x, __ubuf__ T* y)
{
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = U(startIdx + idx);
        U xIdx = 0;
        U xStride = 1;

        U tmp = Simt::UintDiv(yIdx, m0, s0); //  yIdx / channel_;
        U cOut = yIdx - tmp * channel;
        yIdx = tmp;
        xIdx += cOut;
        xStride *= channel;
        U bIdx = 0;
        U bStride = batchSize;

        tmp = Simt::UintDiv(yIdx, m1, s1);
        U indexDivH = yIdx - tmp * yShape1;
        indexDivH += crops0;

        U xCol = Simt::UintDiv(indexDivH, m2, s2); // indexDivH / bShape0;
        U indexDivB = indexDivH - xCol * bShape0;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape1;
        bIdx += indexDivB * bStride;

        xIdx += yIdx * xStride + bIdx * xStride;
        y[idx] = x[xIdx];
    }
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void SimtComputeDimTwo(
    U batchSize, U channel, uint64_t curCoreElement, U startIdx, U yShape1, U xShape1, U bShape0, U crops0, U yShape2,
    U xShape2, U bShape1, U crops1, U m0, U s0, U m1, U s1, U m2, U s2, U m3, U s3, U m4, U s4, __gm__ T* x,
    __ubuf__ T* y)
{
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = U(startIdx + idx);
        U xIdx = 0;
        U xStride = 1;
        U tmp = Simt::UintDiv(yIdx, m0, s0);
        U cOut = yIdx - tmp * channel;
        yIdx = tmp;
        xIdx += cOut;
        xStride *= channel;
        U bIdx = 0;
        U bStride = batchSize;

        tmp = Simt::UintDiv(yIdx, m3, s3);
        U indexDivH = yIdx - tmp * yShape2;
        indexDivH += crops1;
        U xCol = Simt::UintDiv(indexDivH, m4, s4);
        U indexDivB = indexDivH - xCol * bShape1;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape2;
        bIdx += indexDivB * bStride;
        bStride *= bShape1;

        tmp = Simt::UintDiv(yIdx, m1, s1);
        indexDivH = yIdx - tmp * yShape1;
        indexDivH += crops0;
        xCol = Simt::UintDiv(indexDivH, m2, s2);
        indexDivB = indexDivH - xCol * bShape0;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape1;
        bIdx += indexDivB * bStride;

        xIdx += yIdx * xStride + bIdx * xStride;
        y[idx] = x[xIdx];
    }
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void SimtComputeDimThree(
    U batchSize, U channel, uint64_t curCoreElement, U startIdx, U yShape1, U xShape1, U bShape0, U crops0, U yShape2,
    U xShape2, U bShape1, U crops1, U yShape3, U xShape3, U bShape2, U crops2, U m0, U s0, U m1, U s1, U m2, U s2, U m3,
    U s3, U m4, U s4, U m5, U s5, U m6, U s6, __gm__ T* x, __ubuf__ T* y)
{
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = U(startIdx + idx);
        U xIdx = 0;
        U xStride = 1;

        U tmp = Simt::UintDiv(yIdx, m0, s0);
        U cOut = yIdx - tmp * channel;
        yIdx = tmp;
        xIdx += cOut;
        xStride *= channel;
        U bIdx = 0;
        U bStride = batchSize;

        tmp = Simt::UintDiv(yIdx, m5, s5);
        U indexDivH = yIdx - tmp * yShape3;
        indexDivH += crops2;
        U xCol = Simt::UintDiv(indexDivH, m6, s6);
        U indexDivB = indexDivH - xCol * bShape2;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape3;
        bIdx += indexDivB * bStride;
        bStride *= bShape2;

        tmp = Simt::UintDiv(yIdx, m3, s3);
        indexDivH = yIdx - tmp * yShape2;
        indexDivH += crops1;
        xCol = Simt::UintDiv(indexDivH, m4, s4);
        indexDivB = indexDivH - xCol * bShape1;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape2;
        bIdx += indexDivB * bStride;
        bStride *= bShape1;

        tmp = Simt::UintDiv(yIdx, m1, s1);
        indexDivH = yIdx - tmp * yShape1;
        indexDivH += crops0;
        xCol = Simt::UintDiv(indexDivH, m2, s2);
        indexDivB = indexDivH - xCol * bShape0;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape1;
        bIdx += indexDivB * bStride;

        xIdx += yIdx * xStride + bIdx * xStride;
        y[idx] = x[xIdx];
    }
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void SimtComputeDimFour(
    U batchSize, U channel, uint64_t curCoreElement, U startIdx, U yShape1, U xShape1, U bShape0, U crops0, U yShape2,
    U xShape2, U bShape1, U crops1, U yShape3, U xShape3, U bShape2, U crops2, U yShape4, U xShape4, U bShape3,
    U crops3, U m0, U s0, U m1, U s1, U m2, U s2, U m3, U s3, U m4, U s4, U m5, U s5, U m6, U s6, U m7, U s7, U m8,
    U s8, __gm__ T* x, __ubuf__ T* y)
{
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = U(startIdx + idx);
        U xIdx = 0;
        U xStride = 1;

        U tmp = Simt::UintDiv(yIdx, m0, s0);
        U cOut = yIdx - tmp * channel;
        yIdx = tmp;
        xIdx += cOut;
        xStride *= channel;
        U bIdx = 0;
        U bStride = batchSize;

        tmp = Simt::UintDiv(yIdx, m7, s7);
        U indexDivH = yIdx - tmp * yShape4;
        indexDivH += crops3;
        U xCol = Simt::UintDiv(indexDivH, m8, s8);
        U indexDivB = indexDivH - xCol * bShape3;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape4;
        bIdx += indexDivB * bStride;
        bStride *= bShape3;

        tmp = Simt::UintDiv(yIdx, m5, s5);
        indexDivH = yIdx - tmp * yShape3;
        indexDivH += crops2;
        xCol = Simt::UintDiv(indexDivH, m6, s6);
        indexDivB = indexDivH - xCol * bShape2;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape3;
        bIdx += indexDivB * bStride;
        bStride *= bShape2;

        tmp = Simt::UintDiv(yIdx, m3, s3);
        indexDivH = yIdx - tmp * yShape2;
        indexDivH += crops1;
        xCol = Simt::UintDiv(indexDivH, m4, s4);
        indexDivB = indexDivH - xCol * bShape1;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape2;
        bIdx += indexDivB * bStride;
        bStride *= bShape1;

        tmp = Simt::UintDiv(yIdx, m1, s1);
        indexDivH = yIdx - tmp * yShape1;
        indexDivH += crops0;
        xCol = Simt::UintDiv(indexDivH, m2, s2);
        indexDivB = indexDivH - xCol * bShape0;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape1;
        bIdx += indexDivB * bStride;

        xIdx += yIdx * xStride + bIdx * xStride;
        y[idx] = x[xIdx];
    }
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void SimtComputeDimFive(
    U batchSize, U channel, uint64_t curCoreElement, U startIdx, U yShape1, U xShape1, U bShape0, U crops0, U yShape2,
    U xShape2, U bShape1, U crops1, U yShape3, U xShape3, U bShape2, U crops2, U yShape4, U xShape4, U bShape3,
    U crops3, U yShape5, U xShape5, U bShape4, U crops4, U m0, U s0, U m1, U s1, U m2, U s2, U m3, U s3, U m4, U s4,
    U m5, U s5, U m6, U s6, U m7, U s7, U m8, U s8, U m9, U s9, U m10, U s10, __gm__ T* x, __ubuf__ T* y)
{
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = U(startIdx + idx);
        U xIdx = 0;
        U xStride = 1;

        U tmp = Simt::UintDiv(yIdx, m0, s0);
        U cOut = yIdx - tmp * channel;
        yIdx = tmp;
        xIdx += cOut;
        xStride *= channel;
        U bIdx = 0;
        U bStride = batchSize;

        tmp = Simt::UintDiv(yIdx, m9, s9);
        U indexDivH = yIdx - tmp * yShape5;
        indexDivH += crops4;
        U xCol = Simt::UintDiv(indexDivH, m10, s10);
        U indexDivB = indexDivH - xCol * bShape4;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape5;
        bIdx += indexDivB * bStride;
        bStride *= bShape4;

        tmp = Simt::UintDiv(yIdx, m7, s7);
        indexDivH = yIdx - tmp * yShape4;
        indexDivH += crops3;
        xCol = Simt::UintDiv(indexDivH, m8, s8);
        indexDivB = indexDivH - xCol * bShape3;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape4;
        bIdx += indexDivB * bStride;
        bStride *= bShape3;

        tmp = Simt::UintDiv(yIdx, m5, s5);
        indexDivH = yIdx - tmp * yShape3;
        indexDivH += crops2;
        xCol = Simt::UintDiv(indexDivH, m6, s6);
        indexDivB = indexDivH - xCol * bShape2;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape3;
        bIdx += indexDivB * bStride;
        bStride *= bShape2;

        tmp = Simt::UintDiv(yIdx, m3, s3);
        indexDivH = yIdx - tmp * yShape2;
        indexDivH += crops1;
        xCol = Simt::UintDiv(indexDivH, m4, s4);
        indexDivB = indexDivH - xCol * bShape1;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape2;
        bIdx += indexDivB * bStride;
        bStride *= bShape1;

        tmp = Simt::UintDiv(yIdx, m1, s1);
        indexDivH = yIdx - tmp * yShape1;
        indexDivH += crops0;
        xCol = Simt::UintDiv(indexDivH, m2, s2);
        indexDivB = indexDivH - xCol * bShape0;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape1;
        bIdx += indexDivB * bStride;

        xIdx += yIdx * xStride + bIdx * xStride;
        y[idx] = x[xIdx];
    }
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void SimtComputeDimSix(
    U batchSize, U channel, uint64_t curCoreElement, U startIdx, U yShape1, U xShape1, U bShape0, U crops0, U yShape2,
    U xShape2, U bShape1, U crops1, U yShape3, U xShape3, U bShape2, U crops2, U yShape4, U xShape4, U bShape3,
    U crops3, U yShape5, U xShape5, U bShape4, U crops4, U yShape6, U xShape6, U bShape5, U crops5, U m0, U s0, U m1,
    U s1, U m2, U s2, U m3, U s3, U m4, U s4, U m5, U s5, U m6, U s6, U m7, U s7, U m8, U s8, U m9, U s9, U m10, U s10,
    U m11, U s11, U m12, U s12, __gm__ T* x, __ubuf__ T* y)
{
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = U(startIdx + idx);
        U xIdx = 0;
        U xStride = 1;

        U tmp = Simt::UintDiv(yIdx, m0, s0);
        U cOut = yIdx - tmp * channel;
        yIdx = tmp;
        xIdx += cOut;
        xStride *= channel;
        U bIdx = 0;
        U bStride = batchSize;

        tmp = Simt::UintDiv(yIdx, m11, s11);
        U indexDivH = yIdx - tmp * yShape6;
        indexDivH += crops5;
        U xCol = Simt::UintDiv(indexDivH, m12, s12);
        U indexDivB = indexDivH - xCol * bShape5;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape6;
        bIdx += indexDivB * bStride;
        bStride *= bShape5;

        tmp = Simt::UintDiv(yIdx, m9, s9);
        indexDivH = yIdx - tmp * yShape5;
        indexDivH += crops4;
        xCol = Simt::UintDiv(indexDivH, m10, s10);
        indexDivB = indexDivH - xCol * bShape4;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape5;
        bIdx += indexDivB * bStride;
        bStride *= bShape4;

        tmp = Simt::UintDiv(yIdx, m7, s7);
        indexDivH = yIdx - tmp * yShape4;
        indexDivH += crops3;
        xCol = Simt::UintDiv(indexDivH, m8, s8);
        indexDivB = indexDivH - xCol * bShape3;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape4;
        bIdx += indexDivB * bStride;
        bStride *= bShape3;

        tmp = Simt::UintDiv(yIdx, m5, s5);
        indexDivH = yIdx - tmp * yShape3;
        indexDivH += crops2;
        xCol = Simt::UintDiv(indexDivH, m6, s6);
        indexDivB = indexDivH - xCol * bShape2;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape3;
        bIdx += indexDivB * bStride;
        bStride *= bShape2;

        tmp = Simt::UintDiv(yIdx, m3, s3);
        indexDivH = yIdx - tmp * yShape2;
        indexDivH += crops1;
        xCol = Simt::UintDiv(indexDivH, m4, s4);
        indexDivB = indexDivH - xCol * bShape1;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape2;
        bIdx += indexDivB * bStride;
        bStride *= bShape1;

        tmp = Simt::UintDiv(yIdx, m1, s1);
        indexDivH = yIdx - tmp * yShape1;
        indexDivH += crops0;
        xCol = Simt::UintDiv(indexDivH, m2, s2);
        indexDivB = indexDivH - xCol * bShape0;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape1;
        bIdx += indexDivB * bStride;

        xIdx += yIdx * xStride + bIdx * xStride;
        y[idx] = x[xIdx];
    }
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void SimtComputeDimSeven(
    U batchSize, U channel, uint64_t curCoreElement, U startIdx, U yShape1, U xShape1, U bShape0, U crops0, U yShape2,
    U xShape2, U bShape1, U crops1, U yShape3, U xShape3, U bShape2, U crops2, U yShape4, U xShape4, U bShape3,
    U crops3, U yShape5, U xShape5, U bShape4, U crops4, U yShape6, U xShape6, U bShape5, U crops5, U yShape7,
    U xShape7, U bShape6, U crops6, U m0, U s0, U m1, U s1, U m2, U s2, U m3, U s3, U m4, U s4, U m5, U s5, U m6, U s6,
    U m7, U s7, U m8, U s8, U m9, U s9, U m10, U s10, U m11, U s11, U m12, U s12, U m13, U s13, U m14, U s14,
    __gm__ T* x, __ubuf__ T* y)
{
    for (uint64_t idx = threadIdx.x; idx < curCoreElement; idx += blockDim.x) {
        U yIdx = U(startIdx + idx);
        U xIdx = 0;
        U xStride = 1;

        U tmp = Simt::UintDiv(yIdx, m0, s0);
        U cOut = yIdx - tmp * channel;
        yIdx = tmp;
        xIdx += cOut;
        xStride *= channel;
        U bIdx = 0;
        U bStride = batchSize;

        tmp = Simt::UintDiv(yIdx, m13, s13);
        U indexDivH = yIdx - tmp * yShape7;
        indexDivH += crops6;
        U xCol = Simt::UintDiv(indexDivH, m14, s14);
        U indexDivB = indexDivH - xCol * bShape6;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape7;
        bIdx += indexDivB * bStride;
        bStride *= bShape6;

        tmp = Simt::UintDiv(yIdx, m11, s11);
        indexDivH = yIdx - tmp * yShape6;
        indexDivH += crops5;
        xCol = Simt::UintDiv(indexDivH, m12, s12);
        indexDivB = indexDivH - xCol * bShape5;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape6;
        bIdx += indexDivB * bStride;
        bStride *= bShape5;

        tmp = Simt::UintDiv(yIdx, m9, s9);
        indexDivH = yIdx - tmp * yShape5;
        indexDivH += crops4;
        xCol = Simt::UintDiv(indexDivH, m10, s10);
        indexDivB = indexDivH - xCol * bShape4;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape5;
        bIdx += indexDivB * bStride;
        bStride *= bShape4;

        tmp = Simt::UintDiv(yIdx, m7, s7);
        indexDivH = yIdx - tmp * yShape4;
        indexDivH += crops3;
        xCol = Simt::UintDiv(indexDivH, m8, s8);
        indexDivB = indexDivH - xCol * bShape3;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape4;
        bIdx += indexDivB * bStride;
        bStride *= bShape3;

        tmp = Simt::UintDiv(yIdx, m5, s5);
        indexDivH = yIdx - tmp * yShape3;
        indexDivH += crops2;
        xCol = Simt::UintDiv(indexDivH, m6, s6);
        indexDivB = indexDivH - xCol * bShape2;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape3;
        bIdx += indexDivB * bStride;
        bStride *= bShape2;

        tmp = Simt::UintDiv(yIdx, m3, s3);
        indexDivH = yIdx - tmp * yShape2;
        indexDivH += crops1;
        xCol = Simt::UintDiv(indexDivH, m4, s4);
        indexDivB = indexDivH - xCol * bShape1;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape2;
        bIdx += indexDivB * bStride;
        bStride *= bShape1;

        tmp = Simt::UintDiv(yIdx, m1, s1);
        indexDivH = yIdx - tmp * yShape1;
        indexDivH += crops0;
        xCol = Simt::UintDiv(indexDivH, m2, s2);
        indexDivB = indexDivH - xCol * bShape0;
        yIdx = tmp;
        xIdx += xCol * xStride;
        xStride *= xShape1;
        bIdx += indexDivB * bStride;

        xIdx += yIdx * xStride + bIdx * xStride;
        y[idx] = x[xIdx];
    }
}

template <typename T, typename U>
__aicore__ inline void BatchToSpaceNDSIMT<T, U>::Init(GM_ADDR x, GM_ADDR block_shape, GM_ADDR crops, GM_ADDR y,
                                                      const B2SNDSimtTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    mTD = tilingData;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    this->ParseSIMTTilingData(tilingData);
    threadNum_ = HALF_THREAD_NUM_LAUNCH_BOUND;
    pipe->InitBuffer(que_, BUFFER_NUM, tilingData->blockSize * sizeof(T));
}

template <typename T, typename U>
__aicore__ inline void BatchToSpaceNDSIMT<T, U>::ParseSIMTTilingData(const B2SNDSimtTilingData* tilingData)
{
    // shape 信息
    batchSize_ = U(tilingData->input.outShape[0]);
    channel_ = U(tilingData->input.inShape[tilingData->input.rank - 1]);
    blockDim_ = U(tilingData->input.rank - 2);

    // 分核信息
    realCoreNum_ = tilingData->needCoreNum;     // 开核数量
    mainCoreNum_ = tilingData->mainCoreNum;     // 主核数量
    mainCoreBlock_ = tilingData->mainCoreBlock; // 主核块数
    tailBlockSize_ = tilingData->tailBlockSize; // 尾块元素数
    blockSize_ = tilingData->blockSize;         // 每块元素数
}

template <typename T, typename U>
__aicore__ inline void BatchToSpaceNDSIMT<T, U>::Process(GM_ADDR tiling)
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    int64_t ubLoopNum = 0;
    U startIdx = 0;
    U ubProcessNum = 0;
    DataCopyExtParams copyOutParams{1, 0, 0, 0, 0};
    // 所有核的起始地址
    curBlockLoopNumStart_ = blockIdx_ < mainCoreNum_ ?
                                blockIdx_ * mainCoreBlock_ :
                                mainCoreNum_ * mainCoreBlock_ + (blockIdx_ - mainCoreNum_) * (mainCoreBlock_ - 1);
    // 尾核
    if (blockIdx_ >= mainCoreNum_) {
        ubLoopNum = mainCoreBlock_ - 1;
    } else {
        ubLoopNum = mainCoreBlock_;
    }

    // 快速除
    U s[15] = {0};
    U m[15] = {0};

    GetUintDivMagicAndShift(m[0], s[0], channel_);
    for (int64_t i = 1; i <= blockDim_; ++i) {
        GetUintDivMagicAndShift(m[2 * i - 1], s[2 * i - 1], U(mTD->input.outShape[i]));
        GetUintDivMagicAndShift(m[2 * i], s[2 * i], U(mTD->input.blockShape[i - 1]));
    }

    for (int64_t i = 0; i < ubLoopNum; ++i) {
        inTensorY_ = que_.AllocTensor<T>();
        ubProcessNum = blockSize_;
        if (blockIdx_ == realCoreNum_ - 1 && i == ubLoopNum - 1) {
            ubProcessNum = tailBlockSize_;
        }
        if (ubProcessNum == 0) {
            return;
        }
        startIdx = (curBlockLoopNumStart_ + i) * blockSize_;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = ubProcessNum * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        if (blockDim_ == 1) {
            asc_vf_call<SimtComputeDimOne<T, U>>(
                dim3(threadNum_), batchSize_, channel_, ubProcessNum, startIdx, U(mTD->input.outShape[1]),
                U(mTD->input.inShape[1]), U(mTD->input.blockShape[0]), U(mTD->input.crops[0][0]), m[0], s[0], m[1],
                s[1], m[2], s[2], (__gm__ T*)(xGm_.GetPhyAddr()), (__ubuf__ T*)inTensorY_.GetPhyAddr());
        } else if (blockDim_ == 2) {
            asc_vf_call<SimtComputeDimTwo<T, U>>(
                dim3(threadNum_), batchSize_, channel_, ubProcessNum, startIdx, U(mTD->input.outShape[1]),
                U(mTD->input.inShape[1]), U(mTD->input.blockShape[0]), U(mTD->input.crops[0][0]),
                U(mTD->input.outShape[2]), U(mTD->input.inShape[2]), U(mTD->input.blockShape[1]),
                U(mTD->input.crops[1][0]), m[0], s[0], m[1], s[1], m[2], s[2], m[3], s[3], m[4], s[4],
                (__gm__ T*)(xGm_.GetPhyAddr()), (__ubuf__ T*)inTensorY_.GetPhyAddr());
        } else if (blockDim_ == 3) {
            asc_vf_call<SimtComputeDimThree<T, U>>(
                dim3(threadNum_), batchSize_, channel_, ubProcessNum, startIdx, U(mTD->input.outShape[1]),
                U(mTD->input.inShape[1]), U(mTD->input.blockShape[0]), U(mTD->input.crops[0][0]),
                U(mTD->input.outShape[2]), U(mTD->input.inShape[2]), U(mTD->input.blockShape[1]),
                U(mTD->input.crops[1][0]), U(mTD->input.outShape[3]), U(mTD->input.inShape[3]),
                U(mTD->input.blockShape[2]), U(mTD->input.crops[2][0]), m[0], s[0], m[1], s[1], m[2], s[2], m[3], s[3],
                m[4], s[4], m[5], s[5], m[6], s[6], (__gm__ T*)(xGm_.GetPhyAddr()),
                (__ubuf__ T*)inTensorY_.GetPhyAddr());
        } else if (blockDim_ == 4) {
            asc_vf_call<SimtComputeDimFour<T, U>>(
                dim3(threadNum_), batchSize_, channel_, ubProcessNum, startIdx, U(mTD->input.outShape[1]),
                U(mTD->input.inShape[1]), U(mTD->input.blockShape[0]), U(mTD->input.crops[0][0]),
                U(mTD->input.outShape[2]), U(mTD->input.inShape[2]), U(mTD->input.blockShape[1]),
                U(mTD->input.crops[1][0]), U(mTD->input.outShape[3]), U(mTD->input.inShape[3]),
                U(mTD->input.blockShape[2]), U(mTD->input.crops[2][0]), U(mTD->input.outShape[4]),
                U(mTD->input.inShape[4]), U(mTD->input.blockShape[3]), U(mTD->input.crops[3][0]), m[0], s[0], m[1],
                s[1], m[2], s[2], m[3], s[3], m[4], s[4], m[5], s[5], m[6], s[6], m[7], s[7], m[8], s[8],
                (__gm__ T*)(xGm_.GetPhyAddr()), (__ubuf__ T*)inTensorY_.GetPhyAddr());
        } else if (blockDim_ == 5) {
            asc_vf_call<SimtComputeDimFive<T, U>>(
                dim3(threadNum_), batchSize_, channel_, ubProcessNum, startIdx, U(mTD->input.outShape[1]),
                U(mTD->input.inShape[1]), U(mTD->input.blockShape[0]), U(mTD->input.crops[0][0]),
                U(mTD->input.outShape[2]), U(mTD->input.inShape[2]), U(mTD->input.blockShape[1]),
                U(mTD->input.crops[1][0]), U(mTD->input.outShape[3]), U(mTD->input.inShape[3]),
                U(mTD->input.blockShape[2]), U(mTD->input.crops[2][0]), U(mTD->input.outShape[4]),
                U(mTD->input.inShape[4]), U(mTD->input.blockShape[3]), U(mTD->input.crops[3][0]),
                U(mTD->input.outShape[5]), U(mTD->input.inShape[5]), U(mTD->input.blockShape[4]),
                U(mTD->input.crops[4][0]), m[0], s[0], m[1], s[1], m[2], s[2], m[3], s[3], m[4], s[4], m[5], s[5], m[6],
                s[6], m[7], s[7], m[8], s[8], m[9], s[9], m[10], s[10], (__gm__ T*)(xGm_.GetPhyAddr()),
                (__ubuf__ T*)inTensorY_.GetPhyAddr());
        } else if (blockDim_ == 6) {
            asc_vf_call<SimtComputeDimSix<T, U>>(
                dim3(threadNum_), batchSize_, channel_, ubProcessNum, startIdx, U(mTD->input.outShape[1]),
                U(mTD->input.inShape[1]), U(mTD->input.blockShape[0]), U(mTD->input.crops[0][0]),
                U(mTD->input.outShape[2]), U(mTD->input.inShape[2]), U(mTD->input.blockShape[1]),
                U(mTD->input.crops[1][0]), U(mTD->input.outShape[3]), U(mTD->input.inShape[3]),
                U(mTD->input.blockShape[2]), U(mTD->input.crops[2][0]), U(mTD->input.outShape[4]),
                U(mTD->input.inShape[4]), U(mTD->input.blockShape[3]), U(mTD->input.crops[3][0]),
                U(mTD->input.outShape[5]), U(mTD->input.inShape[5]), U(mTD->input.blockShape[4]),
                U(mTD->input.crops[4][0]), U(mTD->input.outShape[6]), U(mTD->input.inShape[6]),
                U(mTD->input.blockShape[5]), U(mTD->input.crops[5][0]), m[0], s[0], m[1], s[1], m[2], s[2], m[3], s[3],
                m[4], s[4], m[5], s[5], m[6], s[6], m[7], s[7], m[8], s[8], m[9], s[9], m[10], s[10], m[11], s[11],
                m[12], s[12], (__gm__ T*)(xGm_.GetPhyAddr()), (__ubuf__ T*)inTensorY_.GetPhyAddr());
        } else if (blockDim_ == 7) {
            asc_vf_call<SimtComputeDimSeven<T, U>>(
                dim3(threadNum_), batchSize_, channel_, ubProcessNum, startIdx, U(mTD->input.outShape[1]),
                U(mTD->input.inShape[1]), U(mTD->input.blockShape[0]), U(mTD->input.crops[0][0]),
                U(mTD->input.outShape[2]), U(mTD->input.inShape[2]), U(mTD->input.blockShape[1]),
                U(mTD->input.crops[1][0]), U(mTD->input.outShape[3]), U(mTD->input.inShape[3]),
                U(mTD->input.blockShape[2]), U(mTD->input.crops[2][0]), U(mTD->input.outShape[4]),
                U(mTD->input.inShape[4]), U(mTD->input.blockShape[3]), U(mTD->input.crops[3][0]),
                U(mTD->input.outShape[5]), U(mTD->input.inShape[5]), U(mTD->input.blockShape[4]),
                U(mTD->input.crops[4][0]), U(mTD->input.outShape[6]), U(mTD->input.inShape[6]),
                U(mTD->input.blockShape[5]), U(mTD->input.crops[5][0]), U(mTD->input.outShape[7]),
                U(mTD->input.inShape[7]), U(mTD->input.blockShape[6]), U(mTD->input.crops[6][0]), m[0], s[0], m[1],
                s[1], m[2], s[2], m[3], s[3], m[4], s[4], m[5], s[5], m[6], s[6], m[7], s[7], m[8], s[8], m[9], s[9],
                m[10], s[10], m[11], s[11], m[12], s[12], m[13], s[13], m[14], s[14], (__gm__ T*)(xGm_.GetPhyAddr()),
                (__ubuf__ T*)inTensorY_.GetPhyAddr());
        }

        que_.EnQue(inTensorY_);
        LocalTensor<T> outTensor = que_.DeQue<T>();
        DataCopyPad(yGm_[startIdx], outTensor, copyOutParams);
        que_.FreeTensor(inTensorY_);
    }
}

} // namespace B2SND

#endif // _BATCH_TO_SPACE_N_D_SIMT_H_
