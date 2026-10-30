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
 * \file roll_simt.h
 * \brief
 */

#ifndef __ROLL_SIMT_H__
#define __ROLL_SIMT_H__

#include "kernel_operator.h"
#include "roll_struct.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"

namespace Roll {
using namespace AscendC;

constexpr int64_t THREAD_NUM = 2048;

template <typename T>
class RollSimt {
public:
    __aicore__ inline RollSimt(TPipe* pipe, const RollTilingData* tiling) : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    const RollTilingData* tilingData_;
    TPipe* pipe_;

    TBuf<TPosition::VECCALC> shapeBuffer_;
    TBuf<TPosition::VECCALC> stridesBuffer_;
    TBuf<TPosition::VECCALC> shiftsBuffer_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t curCoreBaseIndex_ = 0;

    int64_t perCoreEmelents_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void RollSimt<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    perCoreEmelents_ = tilingData_->perCoreElements;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreElements_ = tilingData_->lastCoreElements;
    } else {
        curCoreElements_ = tilingData_->perCoreElements;
    }
    curCoreBaseIndex_ = perCoreEmelents_ * blockIdx_;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(shapeBuffer_, MAX_DIM_NUM * sizeof(int64_t));
    pipe_->InitBuffer(stridesBuffer_, MAX_DIM_NUM * sizeof(int64_t));
    pipe_->InitBuffer(shiftsBuffer_, MAX_DIM_NUM * sizeof(int64_t));
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtDataShift(
    __gm__ T* xGmAddr, __gm__ T* yGmAddr, int64_t inputBasicIndex, int64_t count, int64_t dimNum,
    __ubuf__ int64_t* shapes, __ubuf__ int64_t* strides, __ubuf__ int64_t* shifts)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < count;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t inputIndex = inputBasicIndex + index;
        int64_t inputDimIndex = inputIndex;
        int64_t outputIndex = 0;
        for (int64_t dim = 0; dim < dimNum; dim++) {
            int64_t curDimInputIndex = inputDimIndex / strides[dim];
            inputDimIndex = inputDimIndex % strides[dim];
            outputIndex += (curDimInputIndex + shifts[dim]) % shapes[dim] * strides[dim];
        }
        yGmAddr[outputIndex] = xGmAddr[inputIndex];
    }
}

template <typename T>
__aicore__ inline void RollSimt<T>::Process()
{
    LocalTensor<int64_t> shapes = shapeBuffer_.Get<int64_t>();
    LocalTensor<int64_t> strides = stridesBuffer_.Get<int64_t>();
    LocalTensor<int64_t> shifts = shiftsBuffer_.Get<int64_t>();
    for (int64_t dim = 0; dim < tilingData_->dimNum; dim++) {
        shapes.SetValue(dim, tilingData_->shapes[dim]);
        strides.SetValue(dim, tilingData_->strides[dim]);
        shifts.SetValue(dim, tilingData_->shifts[dim]);
    }

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __gm__ T* xGmAddr = (__gm__ T*)xGm_.GetPhyAddr();
    __gm__ T* yGmAddr = (__gm__ T*)yGm_.GetPhyAddr();
    __ubuf__ int64_t* shapesAddr = (__ubuf__ int64_t*)shapes.GetPhyAddr();
    __ubuf__ int64_t* stridesAddr = (__ubuf__ int64_t*)strides.GetPhyAddr();
    __ubuf__ int64_t* shiftsAddr = (__ubuf__ int64_t*)shifts.GetPhyAddr();
    asc_vf_call<SimtDataShift<T>>(
        dim3{THREAD_NUM, 1, 1}, xGmAddr, yGmAddr, curCoreBaseIndex_, curCoreElements_, tilingData_->dimNum,
        shapesAddr, stridesAddr, shiftsAddr);
}

} // namespace Roll
#endif // __ROLL_SIMT_H__