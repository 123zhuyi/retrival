/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file dynamic_stitch_simt.h
 * \brief
 */
#ifndef DYNAMIC_STITCH_SIMT_H
#define DYNAMIC_STITCH_SIMT_H

#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"

namespace DynamicStitch {

constexpr uint32_t FIRST_DIM_THREAD_NUM = 2;

template <typename T>
class DynamicStitchScatterSimt
{
public:
    __aicore__ inline DynamicStitchScatterSimt(TPipe* pipe, const DynamicStitchTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    
    TPipe* pipe_;
    const DynamicStitchTilingData* tilingData_;
    TBuf<TPosition::VECCALC> assistBuffer_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> workspaceGm_;

    GM_ADDR xGmAddr_ = nullptr;

    int32_t blockIdx_ = 0;
    int64_t sliceSize_ = 0;
    uint16_t startTensorIndex_ = 0;
    uint16_t endTensorIndex_ = 0;
    int64_t startOffset_ = 0;
    int64_t endOffset_ = 0;
    int64_t curTensorNum_ = 0;
};

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtDataCopy(
    GM_ADDR inTensorsPtr, int64_t startTensorIndex, int64_t endTensorIndex, int64_t startOffset, int64_t endOffset,
    __ubuf__ int64_t* tensorCumsum, __gm__ int32_t* workspaceGmAddr, __gm__ T* yGmAddr, const int64_t sliceSize)
{
    for (int32_t tensorIndex = startTensorIndex + static_cast<int32_t>(threadIdx.x);
         tensorIndex <= endTensorIndex; tensorIndex += static_cast<int32_t>(blockDim.x)) {
        __gm__ T* inputTensor = GetTensorSimtAddr<T>(tensorIndex, inTensorsPtr);
        int64_t curElemCount = tensorCumsum[tensorIndex + 1] - tensorCumsum[tensorIndex]; // 获取当前tensor的个数
        int64_t curStartIndex = 0;
        int64_t curEndIndex = curElemCount - 1;
        if (tensorIndex == startTensorIndex) {
            curStartIndex = startOffset;
        }

        if (tensorIndex == endTensorIndex) {
            curEndIndex = endOffset;
        }

        for (int32_t index = curStartIndex + static_cast<int32_t>(threadIdx.y); index <= curEndIndex;
             index += static_cast<int32_t>(blockDim.y)) {
            int64_t workspaceIndex = tensorCumsum[tensorIndex] + index;
            int32_t dstIndex = workspaceGmAddr[workspaceIndex];
            if (dstIndex < 0) {
                continue;
            }

            for (int32_t sliceIndex = static_cast<int32_t>(threadIdx.z); sliceIndex < sliceSize;
                 sliceIndex += static_cast<int32_t>(blockDim.z)) {
                int64_t yGmBaseIndex = dstIndex * sliceSize;
                int64_t xGmBaseIndex = index * sliceSize;
                yGmAddr[yGmBaseIndex + sliceIndex] = inputTensor[xGmBaseIndex + sliceIndex];
            }
        }
    }
}

template <typename T>
__aicore__ inline void DynamicStitchScatterSimt<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    ParseTilingData();
    curTensorNum_ = endTensorIndex_ - startTensorIndex_ + 1;
    xGmAddr_ = x;
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace);
    pipe_->InitBuffer(
        assistBuffer_, AlignUp((MAX_LIST_TENSOR_CNT + 1) * sizeof(int64_t), Ops::Base::GetUbBlockSize()));
}

template <typename T>
__aicore__ inline void DynamicStitchScatterSimt<T>::ParseTilingData()
{
    sliceSize_ = tilingData_->sliceSize;
    startTensorIndex_ = tilingData_->tensorStartList[blockIdx_];
    endTensorIndex_ = tilingData_->tensorEndList[blockIdx_];
    startOffset_ = tilingData_->tensorStartOffsetList[blockIdx_];
    endOffset_ = tilingData_->tensorEndOffsetList[blockIdx_];
}

template <typename T>
__aicore__ inline void DynamicStitchScatterSimt<T>::Process()
{
    if (blockIdx_ < tilingData_->usedCoreNum) {
        LocalTensor<int64_t> tensorCumsumListLocalTensor = assistBuffer_.Get<int64_t>();
        for (int i = startTensorIndex_; i <= endTensorIndex_ + 1; i++) {
            tensorCumsumListLocalTensor.SetValue(i, tilingData_->tensorCumsumList[i]);
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        __gm__ T* yGmAddr = (__gm__ T*)yGm_.GetPhyAddr();
        __gm__ int32_t* workspaceGmAddr = (__gm__ int32_t*)workspaceGm_.GetPhyAddr();
        __ubuf__ int64_t* tensorCumsumAddr = (__ubuf__ int64_t*)tensorCumsumListLocalTensor.GetPhyAddr();

        if (THREAD_NUM / (curTensorNum_ * sliceSize_) > 0) {
            asc_vf_call<SimtDataCopy<T>>(
                dim3{curTensorNum_, THREAD_NUM / (curTensorNum_ * sliceSize_), sliceSize_}, xGmAddr_,
                startTensorIndex_, endTensorIndex_, startOffset_, endOffset_, tensorCumsumAddr, workspaceGmAddr,
                yGmAddr, sliceSize_);
        } else {
            asc_vf_call<SimtDataCopy<T>>(
                dim3{FIRST_DIM_THREAD_NUM, THREAD_NUM / (FIRST_DIM_THREAD_NUM * sliceSize_), sliceSize_},
                xGmAddr_, startTensorIndex_, endTensorIndex_, startOffset_, endOffset_, tensorCumsumAddr,
                workspaceGmAddr, yGmAddr, sliceSize_);
        }
    }
}
} // namespace DynamicStitch

#endif // DYNAMIC_STITCH_SIMT_H
