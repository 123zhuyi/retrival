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
 * \file dynamic_stitch_scatter_simd.h
 * \brief
 */

#ifndef __DYNAMIC_STITCH_SCATTER_SIMD_H__
#define __DYNAMIC_STITCH_SCATTER_SIMD_H__

#include "kernel_operator.h"
#include "dynamic_stitch_tiling_def.h"
#include "dynamic_stitch_common.h"
#include "kernel_utils.h"
#include "op_kernel/platform_util.h"

namespace DynamicStitch {
using namespace AscendC;

template <typename T>
class DynamicStitchScatterSimd {
public:
    __aicore__ inline DynamicStitchScatterSimd(TPipe *pipe, const DynamicStitchTilingData *tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t startIndex, int64_t count);
    __aicore__ inline void Scatter(GlobalTensor<T> inputDataGm, int64_t startOffset, int count);
    
private:
    const DynamicStitchTilingData *tilingData_;
    TPipe *pipe_;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> indicesInQue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> dataOutQue_;

    GlobalTensor<int32_t> wsGm_;
    GlobalTensor<T> yGm_;

    GM_ADDR dateTensorsPtr_ = nullptr;

    int64_t blockIdx_ = 0;
    int64_t curTensorNum_ = 0;
    int64_t startTensorIndex_ = 0;
    int64_t endTensorIndex_ = 0;
    int64_t startOffset_ = 0;
    int64_t endOffset_ = 0;
    int64_t maxIndicesNumInUb_ = 0;
};

template <typename T>
__aicore__ inline void DynamicStitchScatterSimd<T>::Init(
    GM_ADDR indices, GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    startTensorIndex_ = tilingData_->tensorStartList[blockIdx_];
    endTensorIndex_ = tilingData_->tensorEndList[blockIdx_];
    startOffset_ = tilingData_->tensorStartOffsetList[blockIdx_];
    endOffset_ = tilingData_->tensorEndOffsetList[blockIdx_];
    curTensorNum_ = endTensorIndex_ - startTensorIndex_ + 1;
    maxIndicesNumInUb_ = tilingData_->indicesBufferSize / sizeof(int32_t);

    dateTensorsPtr_ = x;
    wsGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    yGm_.SetGlobalBuffer((__gm__ T *)y);

    pipe_->InitBuffer(indicesInQue_, 1, tilingData_->indicesBufferSize);
    pipe_->InitBuffer(
        dataOutQue_, 1,
        AlignUp(tilingData_->ubFactor * sizeof(T), Ops::Base::GetUbBlockSize()));
}

template <typename T>
__aicore__ inline void DynamicStitchScatterSimd<T>::CopyInIndices(int64_t startIndex, int64_t count)
{
    LocalTensor<int32_t> indices = indicesInQue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<int32_t>(0)};
    DataCopyPad(indices, wsGm_[startIndex], dataCopyParams, dataCopyPadParams);
    indicesInQue_.EnQue<int32_t>(indices);
}

template <typename T>
__aicore__ inline void DynamicStitchScatterSimd<T>::Scatter(GlobalTensor<T> inputDataGm, int64_t startOffset, int count)
{
    LocalTensor<int32_t> indices = indicesInQue_.DeQue<int32_t>();
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    for (int i = 0; i < count; i++) {
        int32_t index = indices.GetValue(i);
        if (index < 0) {
            continue;
        }

        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(tilingData_->ubFactor * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
        for (int64_t sliceLoop = 0; sliceLoop < tilingData_->ubLoopTimes; sliceLoop++) {
            LocalTensor<T> data = dataOutQue_.AllocTensor<T>();
            if (sliceLoop == (tilingData_->ubLoopTimes - 1)) {
                dataCopyParams.blockLen = tilingData_->ubTailFactor * sizeof(T);
            }
            DataCopyPad(data,
                inputDataGm[(startOffset + i) * tilingData_->sliceSize + sliceLoop * tilingData_->ubFactor],
                dataCopyParams,
                dataCopyPadParams);
            dataOutQue_.EnQue<T>(data);
            data = dataOutQue_.DeQue<T>();
            DataCopyPad(yGm_[index * tilingData_->sliceSize + sliceLoop * tilingData_->ubFactor], data, dataCopyParams);
            dataOutQue_.FreeTensor(data);
        }
    }
    indicesInQue_.FreeTensor(indices);
}

template <typename T>
__aicore__ inline void DynamicStitchScatterSimd<T>::Process()
{
    if (blockIdx_ < tilingData_->usedCoreNum) {
        for (int i = 0; i < curTensorNum_; i++) {
            int64_t tensorIndex = startTensorIndex_ + i;
            int64_t indicesIndexStart = 0;
            int64_t startOffset = 0;
            int64_t endOffset =
                tilingData_->tensorCumsumList[tensorIndex + 1] - tilingData_->tensorCumsumList[tensorIndex];
            int64_t count = 0;
            if (i == 0) {
                startOffset = startOffset_;
            }

            if (i == curTensorNum_ - 1) {
                endOffset = endOffset_;
            }
            indicesIndexStart = startOffset + tilingData_->tensorCumsumList[tensorIndex];
            count = endOffset - startOffset + 1;

            GlobalTensor<T> inputDataGm_;
            inputDataGm_.SetGlobalBuffer(GetTensorAddr<T>(tensorIndex, dateTensorsPtr_));

            int64_t indicesLoops = Ops::Base::CeilDiv(count, maxIndicesNumInUb_);
            int64_t perLoopCount = Ops::Base::CeilDiv(count, indicesLoops);
            int64_t lastLoopCount = count - (indicesLoops - 1) * perLoopCount;

            for (int32_t indicesLoop = 0; indicesLoop < indicesLoops; indicesLoop++) {
                if (indicesLoop == indicesLoops - 1) {
                    CopyInIndices(indicesIndexStart + indicesLoop * perLoopCount, lastLoopCount);
                    Scatter(inputDataGm_, startOffset + indicesLoop * perLoopCount, lastLoopCount);
                } else {
                    CopyInIndices(indicesIndexStart + indicesLoop * perLoopCount, perLoopCount);
                    Scatter(inputDataGm_, startOffset + indicesLoop * perLoopCount, perLoopCount);
                }
            }
        }
    }
}

}  // namespace DynamicStitch
#endif  // __DYNAMIC_STITCH_SCATTER_SIMD_H__