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
 * \file dynamic_stitch_indices_deduplicate.h
 * \brief
 */

#ifndef __DYNAMIC_STITCH_INDICES_DEDUPLICATE_H__
#define __DYNAMIC_STITCH_INDICES_DEDUPLICATE_H__

#include "kernel_operator.h"
#include "dynamic_stitch_tiling_def.h"
#include "dynamic_stitch_common.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"

namespace DynamicStitch {
using namespace AscendC;

template <typename T>
class DynamicStitchIndicesDeDuplicate {
public:
    __aicore__ inline DynamicStitchIndicesDeDuplicate(TPipe *pipe, const DynamicStitchTilingData *tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    const DynamicStitchTilingData *tilingData_;
    TBuf<TPosition::VECCALC> assistBuffer_;

    __gm__ int32_t *deDuplicateIndices_ = nullptr;
    __gm__ int32_t *writeBackIndices_ = nullptr;

    GlobalTensor<int32_t> wsGm_;

    GM_ADDR inTensorsPtr_ = nullptr;

    int64_t blockIdx_ = 0;
    int64_t curTensorNum_ = 0;
    int64_t curClrBlockWsSize_ = 0;
    int64_t curWriteBackBlockSize_ = 0;
    int64_t startTensorIndex_ = 0;
    int64_t endTensorIndex_ = 0;
    int64_t startOffset_ = 0;
    int64_t endOffset_ = 0;
};

template <typename T>
__aicore__ inline void DynamicStitchIndicesDeDuplicate<T>::Init(GM_ADDR indices, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    inTensorsPtr_ = indices;
    startTensorIndex_ = tilingData_->tensorStartList[blockIdx_];
    endTensorIndex_ = tilingData_->tensorEndList[blockIdx_];
    startOffset_ = tilingData_->tensorStartOffsetList[blockIdx_];
    endOffset_ = tilingData_->tensorEndOffsetList[blockIdx_];
    curTensorNum_ = endTensorIndex_ - startTensorIndex_ + 1;

    if (blockIdx_ == tilingData_->clrBlockNum - 1) {
        curClrBlockWsSize_ = tilingData_->clrTailBlockWsSize;
    } else {
        curClrBlockWsSize_ = tilingData_->clrBlockWsSize;
    }

    if (blockIdx_ == tilingData_->writeBackBlockNum - 1) {
        curWriteBackBlockSize_ = tilingData_->writeBackTailBlockSize;
    } else {
        curWriteBackBlockSize_ = tilingData_->writeBackBlockSize;
    }

    writeBackIndices_ = (__gm__ int32_t *)workspace;
    deDuplicateIndices_ = (__gm__ int32_t *)workspace + tilingData_->totalTensorSum;

    wsGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * tilingData_->clrBlockWsSize);
    pipe_->InitBuffer(assistBuffer_,
        AlignUp((MAX_LIST_TENSOR_CNT + 1) * sizeof(int64_t), Ops::Base::GetUbBlockSize()));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void DeduplicateIndices(GM_ADDR inTensorsPtr,
    __gm__ int32_t *deDuplicateIndices, int64_t startTensorIndex, int64_t endTensorIndex, int64_t startOffset,
    int64_t endOffset, __ubuf__ int64_t *tensorCumsumList)
{
    for (int tensorIndex = startTensorIndex + static_cast<int32_t>(threadIdx.x);
         tensorIndex <= endTensorIndex;
         tensorIndex += static_cast<int32_t>(blockDim.x)) {
        __gm__ int32_t *inputTensor = GetTensorSimtAddr<int32_t>(tensorIndex, inTensorsPtr);
        int64_t startIndex = 0;
        int64_t endIndex = tensorCumsumList[tensorIndex + 1] - tensorCumsumList[tensorIndex] - 1;
        if (tensorIndex == startTensorIndex) {
            startIndex = startOffset;
        }
        if (tensorIndex == endTensorIndex) {
            endIndex = endOffset;
        }

        for (int32_t index = startIndex + static_cast<int32_t>(threadIdx.y); index <= endIndex;
             index += static_cast<int32_t>(blockDim.y)) {
            int32_t dstIndex = inputTensor[index];
            int32_t dstValue = tensorCumsumList[tensorIndex] + index;
            asc_atomic_max(deDuplicateIndices + dstIndex, dstValue);
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void WriteBackIndices(
    __gm__ volatile int32_t *src, __gm__ int32_t *dst, int64_t startIndex, int64_t count, int64_t totalTensorSum)
{
    for (int index = static_cast<int32_t>(threadIdx.x); index < count;
         index += static_cast<int32_t>(blockDim.x)) {
        int32_t dstValue = index + startIndex;
        int32_t dstIndex = src[dstValue];
        if (dstIndex >= 0 && dstIndex < totalTensorSum) {
            dst[dstIndex] = dstValue;
        }
    }
}

template <typename T>
__aicore__ inline void DynamicStitchIndicesDeDuplicate<T>::Process()
{
    if (blockIdx_ < tilingData_->clrBlockNum) {
        InitGlobalMemory(wsGm_, curClrBlockWsSize_, -1);
    }
    SyncAll();

    if (blockIdx_ < tilingData_->usedCoreNum) {
        LocalTensor<int64_t> tensorCumsumListLocalTensor = assistBuffer_.Get<int64_t>();
        for (int i = startTensorIndex_; i <= endTensorIndex_ + 1; i++) {
            tensorCumsumListLocalTensor.SetValue(i, tilingData_->tensorCumsumList[i]);
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        asc_vf_call<DeduplicateIndices>(dim3{curTensorNum_, THREAD_NUM / curTensorNum_, 1},
            inTensorsPtr_,
            deDuplicateIndices_,
            startTensorIndex_,
            endTensorIndex_,
            startOffset_,
            endOffset_,
            (__ubuf__ int64_t *)tensorCumsumListLocalTensor.GetPhyAddr());
    }
    SyncAll();

    if (blockIdx_ < tilingData_->writeBackBlockNum) {
        asc_vf_call<WriteBackIndices>(dim3{THREAD_NUM, 1, 1},
            deDuplicateIndices_,
            writeBackIndices_,
            blockIdx_ * tilingData_->writeBackBlockSize,
            curWriteBackBlockSize_,
            tilingData_->totalTensorSum);
    }
    SyncAll();
}

}  // namespace DynamicStitch
#endif  // __DYNAMIC_STITCH_INDICES_DEDUPLICATE_H__