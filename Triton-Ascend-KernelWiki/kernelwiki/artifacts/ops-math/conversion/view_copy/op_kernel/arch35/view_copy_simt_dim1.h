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
 * \file view_copy_simt_dim1.h
 * \brief
 */

#ifndef VIEW_COPY_SIMT_DIM1_H_
#define VIEW_COPY_SIMT_DIM1_H_

#include "view_copy_base.h"

namespace ViewCopy {
using namespace AscendC;

template <typename T, typename U>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void CopyUbToGmDim1(__gm__ T* dst, __ubuf__ T* src, uint32_t batchSize, uint32_t srcStrideDim0, U dstStrideDim0)
{
    for (uint32_t i = threadIdx.x; i < batchSize; i += blockDim.x) {
        uint32_t srcIndex = (i * srcStrideDim0);
        U dstIndex = (i * dstStrideDim0);
        dst[dstIndex] = src[srcIndex];
    }
}

template <typename T>
class ViewCopySimtDim1 : public ViewCopyBase<T> {
public:
    __aicore__ inline ViewCopySimtDim1(TPipe &pipe, const ViewCopyTilingData *tilingData) :
        pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(
        GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
        GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
        GM_ADDR out);
    __aicore__ inline void Process();
    __aicore__ inline void InitCopyParams();

private:
    __aicore__ inline void ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM1> &dmaParam,
        int32_t batchSize);

private:
    TPipe &pipe_;
    const ViewCopyTilingData *tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;
    int64_t blockOffset_ = 0;

    DataCopyExtParams copyParams_;
    DataCopyExtParams tailCopyParams_;
    MultiCopyParams<T, DIM1> dmaParam_;
    MultiCopyParams<T, DIM1> tailDmaParam_;
};

template <typename T>
__aicore__ inline void ViewCopySimtDim1<T>::Init(
    GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
    GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
    GM_ADDR out)
{
    this->ParseTilingData(tilingData_);
    srcGm_.SetGlobalBuffer((__gm__ T *)(src + tilingData_->srcStorageOffset * sizeof(T)));
    dstGm_.SetGlobalBuffer((__gm__ T *)(dst + tilingData_->dstStorageOffset * sizeof(T)));

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_->bufferSize);
    blockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
}

template <typename T>
__aicore__ inline void ViewCopySimtDim1<T>::Process()
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
            if ((globalLoopIdxModUo + 1) * tilingData_->ubFactor > tilingData_->ubDimSize) {
                ProcessPerLoop(globalLoopIdx, tailDmaParam_, this->tailUbDstSize_[DIM0_INDEX]);
            } else {
                ProcessPerLoop(globalLoopIdx, dmaParam_, tilingData_->ubDstSize[DIM0_INDEX]);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopySimtDim1<T>::ProcessPerLoop(int64_t globalLoopIdx,
    const MultiCopyParams<T, DIM1> &dmaParam, int32_t batchSize)
{
    int64_t srcOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockSrcStride,
                                            tilingData_->blockFusedDimsNumber);
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    if (tilingData_->enableMovAlign != 0) {
        DataCopyExtParams copyParam;
        copyParam.blockCount = 1;
        copyParam.blockLen = dmaParam.loopInfo.loopSize[DIM0] * sizeof(T);
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
    __ubuf__ T* srcAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    __gm__ T* dstAddr = (__gm__ T*)(dstGm_.GetPhyAddr()) + dstOffset;
    uint32_t srcStrideDim0 = tilingData_->contiguousUbDstStride[DIM0_INDEX];
    if (tilingData_->enableDstInt64 != 0) {
        uint64_t dstStrideDim0 = static_cast<uint64_t>(tilingData_->ubDstStride[DIM0_INDEX]);
        asc_vf_call<CopyUbToGmDim1<T, uint64_t>>(dim3(THREAD_DIM), dstAddr, srcAddr, batchSize,
            srcStrideDim0, dstStrideDim0);
    } else {
        uint32_t dstStrideDim0 = static_cast<uint32_t>(tilingData_->ubDstStride[DIM0_INDEX]);
        asc_vf_call<CopyUbToGmDim1<T, uint32_t>>(dim3(THREAD_DIM), dstAddr, srcAddr, batchSize,
            srcStrideDim0, dstStrideDim0);
    }
    inQueue_.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void ViewCopySimtDim1<T>::InitCopyParams()
{
    dmaParam_ = {
        {
            {static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX])},
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX])},
            {static_cast<uint32_t>(tilingData_->nddmaSize[DIM0_INDEX])},
            {0},
            {0}
        },
        0
    };
    tailDmaParam_ = {
        {
            {static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX])},
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX])},
            {static_cast<uint32_t>(this->tailNddmaSize_[DIM0_INDEX])},
            {0},
            {0}
        },
        0
    };
}

}  // namespace ViewCopy

#endif  // VIEW_COPY_SIMT_DIM1_H_