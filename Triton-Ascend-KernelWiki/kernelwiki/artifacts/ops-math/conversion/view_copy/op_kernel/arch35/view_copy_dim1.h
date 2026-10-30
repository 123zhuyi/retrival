/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file view_copy_dim1.h
 * \brief
 */

#ifndef VIEW_COPY_DIM1_H_
#define VIEW_COPY_DIM1_H_

#include "view_copy_base.h"

namespace ViewCopy {
using namespace AscendC;

template <typename T>
class ViewCopyDim1 : public ViewCopyBase<T> {
public:
    __aicore__ inline ViewCopyDim1(TPipe &pipe, const ViewCopyTilingData *tilingData) :
        pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(
        GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
        GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
        GM_ADDR out);
    __aicore__ inline void Process();
    __aicore__ inline void InitCopyParams();

private:
    __aicore__ inline void ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM1> &dmaParam,
        const DataCopyExtParams &copyParams);
private:
    TPipe &pipe_;
    const ViewCopyTilingData *tilingData_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
    GlobalTensor<T> srcGm_;
    GlobalTensor<T> dstGm_;

    int64_t blockOffset_ = 0;
    MultiCopyParams<T, DIM1> dmaParam_;
    MultiCopyParams<T, DIM1> tailDmaParam_;
    DataCopyExtParams copyParams_;
    DataCopyExtParams tailCopyParams_;
};

template <typename T>
__aicore__ inline void ViewCopyDim1<T>::Init(
    GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
    GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset, GM_ADDR out)
{
    this->ParseTilingData(tilingData_);
    blockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    dstGm_.SetGlobalBuffer((__gm__ T *)(dst + tilingData_->dstStorageOffset * sizeof(T)));
    srcGm_.SetGlobalBuffer((__gm__ T *)(src + tilingData_->srcStorageOffset * sizeof(T)));

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_->bufferSize);
}

template <typename T>
__aicore__ inline void ViewCopyDim1<T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    int64_t loopSize = tilingData_->blockFactor;
    if (blockOffset_ >= tilingData_->fusedBlockDims) {
        return;
    } else if ((blockOffset_ + tilingData_->blockFactor) > tilingData_->fusedBlockDims) {
        loopSize = tilingData_->fusedBlockDims - blockOffset_;
    }
    InitCopyParams();

    int64_t globalLoopIdx = 0;
    int64_t globalLoopIdxModUo = 0;

    for (int64_t idx = 0; idx < loopSize; idx++) {
        globalLoopIdx = blockOffset_ + idx;
        globalLoopIdxModUo = globalLoopIdx % tilingData_->uo;
        if ((globalLoopIdxModUo) * tilingData_->ubFactor <= tilingData_->ubDimSize) {
            if((globalLoopIdxModUo + 1) * tilingData_->ubFactor > tilingData_->ubDimSize) {
                ProcessPerLoop(globalLoopIdx, tailDmaParam_, tailCopyParams_);
            } else {
                ProcessPerLoop(globalLoopIdx, dmaParam_, copyParams_);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopyDim1<T>::ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM1> &dmaParam,
    const DataCopyExtParams &copyParams)
{
    int64_t srcOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockSrcStride,
                                          tilingData_->blockFusedDimsNumber);
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    if (tilingData_->enableMovAlign != 0) {
        const auto& loopInfo = dmaParam.loopInfo;
        DataCopyExtParams copyParam;
        copyParam.blockCount = 1;
        copyParam.blockLen = loopInfo.loopSize[DIM0] * sizeof(T);
        copyParam.srcStride = 0;
        copyParam.dstStride = 0;
        CopyGmToUbCompact(srcLocal, srcGm_[srcOffset], copyParam);
    } else {
        DataCopy(srcLocal, srcGm_[srcOffset], dmaParam);
    }
    inQueue_.EnQue(srcLocal);
    int64_t dstOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockDstStride,
                                          tilingData_->blockFusedDimsNumber);
    LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
    DataCopyPad(dstGm_[dstOffset], dstLocal, copyParams);
    inQueue_.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void ViewCopyDim1<T>::InitCopyParams()
{
    dmaParam_ = {
        {
            {static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX])}, // src stride
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX])}, // dst stride
            {static_cast<uint32_t>(tilingData_->nddmaSize[DIM0_INDEX])}, // loop size
            {ZERO_U8}, // left pad
            {ZERO_U8} // right pad
        },
        0   //pad value
    };

    tailDmaParam_ = {
        {
            {static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX])}, // src stride
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX])}, // dst stride
            {static_cast<uint32_t>(this->tailNddmaSize_[DIM0_INDEX])}, // loop size
            {ZERO_U8}, // left pad
            {ZERO_U8} // right pad
        },
        0   //pad value
    };

    copyParams_ = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(tilingData_->ubDstSize[DIM0_INDEX] * sizeof(T)),
        static_cast<uint32_t>(0),
        0,
        0
    };
    tailCopyParams_ = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(this->tailUbDstSize_[DIM0_INDEX] * sizeof(T)),
        static_cast<uint32_t>(0),
        0,
        0
    };
}

}  // namespace ViewCopy

#endif  // VIEW_COPY_DIM1_H_