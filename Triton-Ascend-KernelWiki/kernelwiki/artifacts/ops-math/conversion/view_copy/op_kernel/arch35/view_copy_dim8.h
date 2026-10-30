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
 * \file view_copy_dim8.h
 * \brief
 */

#ifndef VIEW_COPY_DIM8_H_
#define VIEW_COPY_DIM8_H_

#include "view_copy_base.h"

namespace ViewCopy {
using namespace AscendC;

template <typename T>
class ViewCopyDim8 : public ViewCopyBase<T> {
public:
    __aicore__ inline ViewCopyDim8(TPipe& pipe, const ViewCopyTilingData* tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset, GM_ADDR src,
                                GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset, GM_ADDR out);
    __aicore__ inline void Process();
    __aicore__ inline void InitCopyParams();

private:
    __aicore__ inline void CustomNddma(LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                       const MultiCopyParams<T, 5>& dmaParam, const CustomCopyExtParams& extParams);
    __aicore__ inline void CopyOut(const GlobalTensor<T>& dst, const DataCopyExtParams& copyParams,
                                   const CustomCopyExtParams& extParams);
    __aicore__ inline void ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM5>& dmaParams,
                                          const DataCopyExtParams& copyParams, const int32_t* inLoopSize,
                                          const int32_t* outLoopSize);

private:
    TPipe& pipe_;
    const ViewCopyTilingData* tilingData_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    int64_t blockOffset_ = 0;
    CustomCopyExtParams outExtParams_;
    CustomCopyExtParams inExtParams_;
    MultiCopyParams<T, DIM5> dmaParam_;
    MultiCopyParams<T, DIM5> tailDmaParam_;
    DataCopyExtParams copyParams_;
    DataCopyExtParams tailCopyParams_;
};

template <typename T>
__aicore__ inline void ViewCopyDim8<T>::Init(GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
                                             GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
                                             GM_ADDR out)
{
    this->ParseTilingData(tilingData_);
    dstGm_.SetGlobalBuffer((__gm__ T*)(dst + tilingData_->dstStorageOffset * sizeof(T)));
    srcGm_.SetGlobalBuffer((__gm__ T*)(src + tilingData_->srcStorageOffset * sizeof(T)));

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_->bufferSize);
    InitCopyParams();
    blockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    inExtParams_ = {tilingData_->nddmaSize, tilingData_->nddmaStride, tilingData_->contiguousUbSrcStride};
    outExtParams_ = {tilingData_->ubDstSize, tilingData_->contiguousUbDstStride, tilingData_->ubDstStride};
}

template <typename T>
__aicore__ inline void ViewCopyDim8<T>::Process()
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

    int64_t globalLoopIdx = 0;
    int64_t globalLoopIdxModUo = 0;

    for (int64_t idx = 0; idx < loopSize; idx++) {
        globalLoopIdx = blockOffset_ + idx;
        globalLoopIdxModUo = globalLoopIdx % tilingData_->uo;
        if ((globalLoopIdxModUo)*tilingData_->ubFactor <= tilingData_->ubDimSize) {
            if ((globalLoopIdxModUo + 1) * tilingData_->ubFactor > tilingData_->ubDimSize) {
                ProcessPerLoop(globalLoopIdx, tailDmaParam_, tailCopyParams_, this->tailNddmaSize_,
                               this->tailUbDstSize_);
            } else {
                ProcessPerLoop(globalLoopIdx, dmaParam_, copyParams_, tilingData_->nddmaSize, tilingData_->ubDstSize);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopyDim8<T>::ProcessPerLoop(int64_t globalLoopIdx, const MultiCopyParams<T, DIM5>& dmaParams,
                                                       const DataCopyExtParams& copyParams, const int32_t* inLoopSize,
                                                       const int32_t* outLoopSize)
{
    int64_t srcOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockSrcStride,
                                          tilingData_->blockFusedDimsNumber);
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    inExtParams_.loopSize = inLoopSize;
    CustomNddma(srcLocal, srcGm_[srcOffset], dmaParams, inExtParams_);
    inQueue_.EnQue(srcLocal);
    int64_t dstOffset = this->GetGmOffset(globalLoopIdx, tilingData_->blockStride, tilingData_->blockDstStride,
                                          tilingData_->blockFusedDimsNumber);
    outExtParams_.loopSize = outLoopSize;
    CopyOut(dstGm_[dstOffset], copyParams, outExtParams_);
}

template <typename T>
__aicore__ inline void ViewCopyDim8<T>::CustomNddma(LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                                    const MultiCopyParams<T, DIM5>& dmaParam,
                                                    const CustomCopyExtParams& extParams)
{
    // srcSize (dim7, dim6, dim5, dim4, dim3, dim2, dim1, dim0)
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;

    for (int64_t loopDim7 = 0; loopDim7 < extParams.loopSize[DIM7_INDEX]; loopDim7++) {
        for (int64_t loopDim6 = 0; loopDim6 < extParams.loopSize[DIM6_INDEX]; loopDim6++) {
            for (int64_t loopDim5 = 0; loopDim5 < extParams.loopSize[DIM5_INDEX]; loopDim5++) {
                srcOffset = loopDim7 * extParams.srcStride[DIM7_INDEX] + loopDim6 * extParams.srcStride[DIM6_INDEX] +
                            loopDim5 * extParams.srcStride[DIM5_INDEX];
                dstOffset = loopDim7 * extParams.dstStride[DIM7_INDEX] + loopDim6 * extParams.dstStride[DIM6_INDEX] +
                            loopDim5 * extParams.dstStride[DIM5_INDEX];
                DataCopy(dst[dstOffset], src[srcOffset], dmaParam);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopyDim8<T>::CopyOut(const GlobalTensor<T>& dst, const DataCopyExtParams& copyParams,
                                                const CustomCopyExtParams& extParams)
{
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
    // loopSize (dim7, dim6, dim5, dim4, dim3, dim2, dim1, dim0)
    // 只处理last轴连续且较大的场景
    for (int64_t loopDim7 = 0; loopDim7 < extParams.loopSize[DIM7_INDEX]; loopDim7++) {
        for (int64_t loopDim6 = 0; loopDim6 < extParams.loopSize[DIM6_INDEX]; loopDim6++) {
            for (int64_t loopDim5 = 0; loopDim5 < extParams.loopSize[DIM5_INDEX]; loopDim5++) {
                for (int64_t loopDim4 = 0; loopDim4 < extParams.loopSize[DIM4_INDEX]; loopDim4++) {
                    for (int64_t loopDim3 = 0; loopDim3 < extParams.loopSize[DIM3_INDEX]; loopDim3++) {
                        for (int64_t loopDim2 = 0; loopDim2 < extParams.loopSize[DIM2_INDEX]; loopDim2++) {
                            srcOffset = loopDim7 * extParams.srcStride[DIM7_INDEX] +
                                        loopDim6 * extParams.srcStride[DIM6_INDEX] +
                                        loopDim5 * extParams.srcStride[DIM5_INDEX] +
                                        loopDim4 * extParams.srcStride[DIM4_INDEX] +
                                        loopDim3 * extParams.srcStride[DIM3_INDEX] +
                                        loopDim2 * extParams.srcStride[DIM2_INDEX];
                            dstOffset = loopDim7 * extParams.dstStride[DIM7_INDEX] +
                                        loopDim6 * extParams.dstStride[DIM6_INDEX] +
                                        loopDim5 * extParams.dstStride[DIM5_INDEX] +
                                        loopDim4 * extParams.dstStride[DIM4_INDEX] +
                                        loopDim3 * extParams.dstStride[DIM3_INDEX] +
                                        loopDim2 * extParams.dstStride[DIM2_INDEX];
                            DataCopyPad(dst[dstOffset], srcLocal[srcOffset], copyParams);
                        }
                    }
                }
            }
        }
    }
    inQueue_.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void ViewCopyDim8<T>::InitCopyParams()
{
    dmaParam_ = {
        {
            {// src stride
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM2_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM3_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM4_INDEX])},
            {
                // dst stride
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM4_INDEX]),
            },
            {
                // loop size
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->nddmaSize[DIM4_INDEX]),
            },
            {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}, // left pad
            {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}  // right pad
        },
        0                                                  // pad value
    };

    tailDmaParam_ = {
        {
            {// src stride
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM0_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM1_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM2_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM3_INDEX]),
             static_cast<uint64_t>(tilingData_->nddmaStride[DIM4_INDEX])},
            {
                // dst stride
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM0_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM1_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM2_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM3_INDEX]),
                static_cast<uint32_t>(tilingData_->contiguousUbSrcStride[DIM4_INDEX]),
            },
            {
                // loop size
                static_cast<uint32_t>(this->tailNddmaSize_[DIM0_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM1_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM2_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM3_INDEX]),
                static_cast<uint32_t>(this->tailNddmaSize_[DIM4_INDEX]),
            },
            {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}, // left pad
            {ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8, ZERO_U8}  // right pad
        },
        0                                                  // pad value
    };
    int64_t copyDstStride = (tilingData_->ubDstStride[DIM1_INDEX] - tilingData_->ubDstSize[DIM0_INDEX]) * sizeof(T);
    int64_t tailCopyDstStride = (tilingData_->ubDstStride[DIM1_INDEX] - this->tailUbDstSize_[DIM0_INDEX]) * sizeof(T);
    copyParams_ = {static_cast<uint16_t>(tilingData_->ubDstSize[DIM1_INDEX]),
                   static_cast<uint32_t>(tilingData_->ubDstSize[DIM0_INDEX] * sizeof(T)), static_cast<uint32_t>(0),
                   copyDstStride, 0};
    tailCopyParams_ = {static_cast<uint16_t>(this->tailUbDstSize_[DIM1_INDEX]),
                       static_cast<uint32_t>(this->tailUbDstSize_[DIM0_INDEX] * sizeof(T)), static_cast<uint32_t>(0),
                       tailCopyDstStride, 0};
}
} // namespace ViewCopy

#endif // VIEW_COPY_DIM8_H_