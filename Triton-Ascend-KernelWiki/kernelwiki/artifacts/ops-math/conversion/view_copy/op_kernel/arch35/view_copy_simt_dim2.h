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
 * \file view_copy_simt_dim2.h
 * \brief
 */

#ifndef VIEW_COPY_SIMT_DIM2_H_
#define VIEW_COPY_SIMT_DIM2_H_

#include "view_copy_base.h"
#include "op_kernel/platform_util.h"

namespace ViewCopy {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, typename U>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_1024) __aicore__
    void CopyUbToGmDim2(__gm__ T* dst, __ubuf__ T* src, uint32_t batchSize, uint32_t srcStrideDim0,
    uint32_t srcStrideDim1, uint32_t ubDim1Stride, uint32_t magic0, uint32_t shift0, U dstStrideDim0, U dstStrideDim1)
{
    for (uint32_t i = threadIdx.x; i < batchSize; i += blockDim.x) {
        uint32_t idx = i;
        uint32_t srcIndex = 0;
        U dstIndex = 0;

        uint32_t t1 = __umulhi(idx, magic0);
        t1 = t1 + idx;
        uint32_t dim1Index = t1 >> shift0;
        uint32_t dim0Index = idx - dim1Index * ubDim1Stride;
        srcIndex += (dim1Index * srcStrideDim1);
        dstIndex += (dim1Index * dstStrideDim1);

        srcIndex += (dim0Index * srcStrideDim0);
        dstIndex += (dim0Index * dstStrideDim0);
        dst[dstIndex] = src[srcIndex];
    }
}

template <typename T>
class ViewCopySimtDim2 : public ViewCopyBase<T> {
public:
    __aicore__ inline ViewCopySimtDim2(TPipe &pipe, const ViewCopyTilingData *tilingData) :
        pipe_(pipe), tilingData_(tilingData) {};
    __aicore__ inline void Init(
        GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
        GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
        GM_ADDR out);
    __aicore__ inline void Process();
    __aicore__ inline void InitCopyParams();

private:
    __aicore__ inline void ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM2> &dmaParam,
        int32_t batchSize);

private:
    TPipe &pipe_;
    const ViewCopyTilingData *tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    DataCopyExtParams copyParams_;
    DataCopyExtParams tailCopyParams_;
    MultiCopyParams<T, DIM2> dmaParam_;
    MultiCopyParams<T, DIM2> tailDmaParam_;

    int64_t blockOffset_ = 0;
    int32_t batchSize_ = 1;
    int32_t tailBatchSize_ = 1;
    uint32_t dim1UbdstStride_ = 1;
    uint32_t shift_[1];
    uint32_t m_[1];
};

template <typename T>
__aicore__ inline void ViewCopySimtDim2<T>::Init(
    GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
    GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
    GM_ADDR out)
{
    this->ParseTilingData(tilingData_);

    blockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    srcGm_.SetGlobalBuffer((__gm__ T *)(src + tilingData_->srcStorageOffset * sizeof(T)));
    dstGm_.SetGlobalBuffer((__gm__ T *)(dst + tilingData_->dstStorageOffset * sizeof(T)));

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_->bufferSize);

    batchSize_ = tilingData_->ubDstSize[DIM0_INDEX] * tilingData_->ubDstSize[DIM1_INDEX];
    tailBatchSize_ = this->tailUbDstSize_[DIM0_INDEX] * this->tailUbDstSize_[DIM1_INDEX];

    dim1UbdstStride_ = tilingData_->ubDstSize[DIM0_INDEX];
    GetUintDivMagicAndShift<uint32_t>(m_[0], shift_[0], dim1UbdstStride_);
}

template <typename T>
__aicore__ inline void ViewCopySimtDim2<T>::Process()
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
    int64_t globalLoopIdx1 = 0;
    int64_t globalLoopIdxModUo1 = 0;
    for (int64_t idx = 0; idx < loopSize; idx++) {
        globalLoopIdx = blockOffset_ + idx;
        globalLoopIdxModUo = globalLoopIdx % tilingData_->uo;
        if ((globalLoopIdxModUo) * tilingData_->ubFactor <= tilingData_->ubDimSize) {
            if ((globalLoopIdxModUo + 1) * tilingData_->ubFactor > tilingData_->ubDimSize) {
                ProcessPerLoop(globalLoopIdx, tailDmaParam_, tailBatchSize_);
            } else {
                ProcessPerLoop(globalLoopIdx, dmaParam_, batchSize_);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopySimtDim2<T>::ProcessPerLoop(int64_t globalLoopIdx,
    const MultiCopyParams<T, DIM2> &dmaParam, int32_t batchSize)
{
    int64_t srcOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockSrcStride,
                                          tilingData_->blockFusedDimsNumber);
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    if (tilingData_->enableMovAlign != 0) {
        DataCopyExtParams copyParam;
        copyParam.blockCount = dmaParam.loopInfo.loopSize[DIM1];
        copyParam.blockLen = dmaParam.loopInfo.loopSize[DIM0] * sizeof(T);
        copyParam.srcStride = (dmaParam.loopInfo.loopSrcStride[DIM1] - dmaParam.loopInfo.loopSize[DIM0]) * sizeof(T);
        copyParam.dstStride = (dmaParam.loopInfo.loopDstStride[DIM1] -
            dmaParam.loopInfo.loopSize[DIM0]) * sizeof(T) / GetUbBlockSize();
        bool isCompact = dmaParam.loopInfo.loopDstStride[DIM1] == dmaParam.loopInfo.loopSize[DIM0];
        CopyGmToUbCompact(srcLocal, srcGm_[srcOffset], copyParam, isCompact);
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
    uint32_t srcStrideDim1 = tilingData_->contiguousUbDstStride[DIM1_INDEX];
    if (tilingData_->enableDstInt64 != 0) {
        uint64_t dstStrideDim0 = static_cast<uint64_t>(tilingData_->ubDstStride[DIM0_INDEX]);
        uint64_t dstStrideDim1 = static_cast<uint64_t>(tilingData_->ubDstStride[DIM1_INDEX]);
        asc_vf_call<CopyUbToGmDim2<T, uint64_t>>(dim3(THREAD_DIM_1024), dstAddr, srcAddr, batchSize,
            srcStrideDim0, srcStrideDim1, dim1UbdstStride_, m_[MS_IDX0], shift_[MS_IDX0], dstStrideDim0,
            dstStrideDim1);
    } else {
        uint32_t dstStrideDim0 = static_cast<uint32_t>(tilingData_->ubDstStride[DIM0_INDEX]);
        uint32_t dstStrideDim1 = static_cast<uint32_t>(tilingData_->ubDstStride[DIM1_INDEX]);
        asc_vf_call<CopyUbToGmDim2<T, uint32_t>>(dim3(THREAD_DIM_1024), dstAddr, srcAddr, batchSize,
            srcStrideDim0, srcStrideDim1, dim1UbdstStride_, m_[MS_IDX0], shift_[MS_IDX0], dstStrideDim0,
            dstStrideDim1);
    }

    inQueue_.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void ViewCopySimtDim2<T>::InitCopyParams()
{
    dmaParam_ = {
        {
            {static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
              static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX])},
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
              static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX]) },
            { static_cast<uint32_t>(tilingData_->nddmaSize[DIM0_INDEX]),
              static_cast<uint32_t>(tilingData_->nddmaSize[DIM1_INDEX])},
            {0, 0},
            {0, 0}
        },
        0
    };

    tailDmaParam_ = {
        {
            { static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
                static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX])},
            {static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX])},
            {static_cast<uint32_t>(this->tailNddmaSize_[DIM0_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM1_INDEX])},
            {0, 0},
            {0, 0}
        },
        0
    };
}

}  // namespace ViewCopy

#endif  // VIEW_COPY_SIMT_DIM2_H_