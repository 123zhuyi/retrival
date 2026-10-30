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
 * \file view_copy_base.h
 * \brief
 */

#ifndef VIEW_COPY_BASE_H_
#define VIEW_COPY_BASE_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"

namespace ViewCopy {
using namespace AscendC;
using namespace Ops::Base;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_DIM = 128;
#else
constexpr uint32_t THREAD_DIM = 512;
constexpr uint32_t THREAD_DIM_1024 = 1024;
#endif
constexpr int16_t TILING_ARRAY_LEN = 10;
constexpr int16_t NDDMA_ARRAY_LEN = 8;
constexpr uint8_t ZERO_U8 = 0;
constexpr int16_t DIM0 = 0;
constexpr int16_t DIM1 = 1;
constexpr int16_t DIM2 = 2;
constexpr int16_t DIM3 = 3;
constexpr int16_t DIM4 = 4;
constexpr int16_t DIM5 = 5;
constexpr int16_t DIM6 = 6;
constexpr int16_t DIM7 = 7;

constexpr int16_t DIM0_INDEX = 7;
constexpr int16_t DIM1_INDEX = 6;
constexpr int16_t DIM2_INDEX = 5;
constexpr int16_t DIM3_INDEX = 4;
constexpr int16_t DIM4_INDEX = 3;
constexpr int16_t DIM5_INDEX = 2;
constexpr int16_t DIM6_INDEX = 1;
constexpr int16_t DIM7_INDEX = 0;

constexpr int16_t BUFFER_NUM = 2;
constexpr int64_t MS_IDX0 = 0;
constexpr int64_t MS_IDX1 = 1;
constexpr int64_t MS_IDX2 = 2;
constexpr int64_t MS_IDX3 = 3;
constexpr int64_t MS_IDX4 = 4;
constexpr int64_t MS_IDX5 = 5;

constexpr uint32_t DIM4_PARAM_NUM = 8;
constexpr uint32_t DIM5_PARAM_NUM = 8;
constexpr uint32_t DIM7_PARAM_NUM = 8;
constexpr uint32_t DIM8_PARAM_NUM = 8;
constexpr uint32_t MS_DIM4_IDX_NUM = 4;
constexpr uint32_t DIM5_IDX_NUM = 5;
constexpr uint32_t MS_IDX_NUM = 7;
constexpr uint32_t DIM8_IDX_NUM = 8;

struct CustomCopyExtParams {
    const int32_t* loopSize;
    const int64_t* srcStride;
    const int64_t* dstStride;
};

template <typename T>
__aicore__ inline void CopyGmToUbCompact(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
    const DataCopyExtParams &params, bool isCompact=false)
{
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    if (isCompact) {
        DataCopyPad<T, PaddingMode::Compact>(dstLocal, srcGlobal, params, padExtParams);
    } else {
        DataCopyPad<T>(dstLocal, srcGlobal, params, padExtParams);
    }
}

template <typename T>
class ViewCopyBase {
public:
    __aicore__ inline ViewCopyBase(){};
    __aicore__ inline int64_t GetGmOffset(int64_t loopIdx, const int64_t *blockStride, const int64_t *blockSrcStride,
                                          int64_t blockFusedNumber);
public:
    int64_t tailUbFactor_;
    int32_t tailNddmaSize_[NDDMA_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1};
    int32_t tailUbDstSize_[NDDMA_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1};

protected:
    __aicore__ inline void ParseTilingData(const ViewCopyTilingData *tilingData);
    __aicore__ inline void CopyArray(const int64_t *src, int64_t *dst, int16_t size);
    __aicore__ inline void CopyArray(const int32_t *src, int32_t *dst, int16_t size);
    __aicore__ inline void CopyInDim3(const GlobalTensor<T> &src, const LocalTensor<T> &dstLocal,
        const MultiCopyParams<T, DIM3> &dmaParams, bool enableMovAlign);
    __aicore__ inline void CopyInDim4(const GlobalTensor<T> &src, const LocalTensor<T> &dstLocal,
        const MultiCopyParams<T, DIM4> &dmaParams, bool enableMovAlign);
    __aicore__ inline void CopyInDim5(const GlobalTensor<T> &src, const LocalTensor<T> &dstLocal,
        const MultiCopyParams<T, DIM5> &dmaParams, bool enableMovAlign);
};

template <typename T>
__aicore__ inline void ViewCopyBase<T>::ParseTilingData(const ViewCopyTilingData *tilingData)
{
    CopyArray(tilingData->nddmaSize, tailNddmaSize_, tilingData->nddmaSizeLen);
    CopyArray(tilingData->ubDstSize, tailUbDstSize_, tilingData->ubDstSizeLen);
    tailUbFactor_ = tilingData->ubDimSize - (tilingData->uo - 1) * tilingData->ubFactor;
    tailNddmaSize_[tilingData->srcUbDim] = tailNddmaSize_[tilingData->srcUbDim] / tilingData->ubFactor * tailUbFactor_;
    tailUbDstSize_[tilingData->dstUbDim] = tailUbDstSize_[tilingData->dstUbDim] / tilingData->ubFactor * tailUbFactor_;
}

template <typename T>
__aicore__ inline void ViewCopyBase<T>::CopyArray(const int32_t *src, int32_t *dst, int16_t size)
{
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

template <typename T>
__aicore__ inline int64_t ViewCopyBase<T>::GetGmOffset(int64_t loopIdx, const int64_t *blockStride,
    const int64_t *blockSrcStride, int64_t blockFusedNumber)
{
    int64_t offset = 0;
    int64_t curLoopIdx = loopIdx;
    for (int64_t idx=0; idx < blockFusedNumber; idx++){
        offset += (curLoopIdx / blockStride[idx] * blockSrcStride[idx]);
        curLoopIdx = curLoopIdx % blockStride[idx];
    }
    return offset;
}

template <typename T>
__aicore__ inline void ViewCopyBase<T>::CopyInDim3(const GlobalTensor<T> &src, const LocalTensor<T> &dstLocal,
    const MultiCopyParams<T, DIM3> &dmaParams, bool enableMovAlign)
{
    if (enableMovAlign) {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        const auto& loopInfo = dmaParams.loopInfo;
        DataCopyExtParams copyParams;
        copyParams.blockCount = loopInfo.loopSize[DIM1];
        copyParams.blockLen = loopInfo.loopSize[DIM0] * sizeof(T);
        copyParams.srcStride = (loopInfo.loopSrcStride[DIM1] - loopInfo.loopSize[DIM0]) * sizeof(T);
        copyParams.dstStride = (loopInfo.loopDstStride[DIM1] - loopInfo.loopSize[DIM0]) * sizeof(T) /
            GetUbBlockSize();
        bool isCompact = loopInfo.loopDstStride[DIM1] == loopInfo.loopSize[DIM0];

        for (uint32_t loopDim2 = 0; loopDim2 < loopInfo.loopSize[DIM2]; loopDim2++) {
            srcOffset = loopDim2 * loopInfo.loopSrcStride[DIM2];
            dstOffset = loopDim2 * loopInfo.loopDstStride[DIM2];
            CopyGmToUbCompact(dstLocal[dstOffset], src, copyParams, isCompact);
        }
    } else {
        DataCopy(dstLocal, src, dmaParams);
    }
}

template <typename T>
__aicore__ inline void ViewCopyBase<T>::CopyInDim4(const GlobalTensor<T> &src, const LocalTensor<T> &dstLocal,
    const MultiCopyParams<T, DIM4> &dmaParams, bool enableMovAlign)
{
    if (enableMovAlign) {
        const auto& loopInfo = dmaParams.loopInfo;
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        DataCopyExtParams copyParams;
        copyParams.blockCount = loopInfo.loopSize[DIM1];
        copyParams.blockLen = loopInfo.loopSize[DIM0] * sizeof(T);
        copyParams.srcStride = (loopInfo.loopSrcStride[DIM1] - loopInfo.loopSize[DIM0]) * sizeof(T);
        copyParams.dstStride = (loopInfo.loopDstStride[DIM1] -
            loopInfo.loopSize[DIM0]) * sizeof(T) / GetUbBlockSize();
        bool isCompact = loopInfo.loopDstStride[DIM1] == loopInfo.loopSize[DIM0];
        for (uint32_t loopDim3 = 0; loopDim3 < loopInfo.loopSize[DIM3]; loopDim3++) {
            for (uint32_t loopDim2 = 0; loopDim2 < loopInfo.loopSize[DIM2]; loopDim2++) {
                srcOffset = loopDim3 * loopInfo.loopSrcStride[DIM3] + loopDim2 * loopInfo.loopSrcStride[DIM2];
                dstOffset = loopDim3 * loopInfo.loopDstStride[DIM3] + loopDim2 * loopInfo.loopDstStride[DIM2];
                CopyGmToUbCompact(dstLocal[dstOffset], src, copyParams, isCompact);
            }
        }
    } else {
        DataCopy(dstLocal, src, dmaParams);
    }
}

template <typename T>
__aicore__ inline void ViewCopyBase<T>::CopyInDim5(const GlobalTensor<T> &src, const LocalTensor<T> &dstLocal,
    const MultiCopyParams<T, DIM5> &dmaParams, bool enableMovAlign)
{
    if (enableMovAlign) {
        const auto& loopInfo = dmaParams.loopInfo;
        DataCopyExtParams copyParams;
        copyParams.blockCount = loopInfo.loopSize[DIM1];
        copyParams.blockLen = loopInfo.loopSize[DIM0] * sizeof(T);
        copyParams.srcStride = (loopInfo.loopSrcStride[DIM1] - loopInfo.loopSize[DIM0]) * sizeof(T);
        copyParams.dstStride = (loopInfo.loopDstStride[DIM1] -
            loopInfo.loopSize[DIM0]) * sizeof(T) / GetUbBlockSize();
        bool isCompact = loopInfo.loopDstStride[DIM1] == loopInfo.loopSize[DIM0];
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for (uint32_t loopDim4 = 0; loopDim4 < loopInfo.loopSize[DIM4_INDEX]; loopDim4++) {
            for (uint32_t loopDim3 = 0; loopDim3 < loopInfo.loopSize[DIM3]; loopDim3++) {
                for (uint32_t loopDim2 = 0; loopDim2 < loopInfo.loopSize[DIM2]; loopDim2++) {
                    srcOffset = loopDim4 * loopInfo.loopSrcStride[DIM4] + loopDim3 * loopInfo.loopSrcStride[DIM3] +
                                loopDim2 * loopInfo.loopSrcStride[DIM2];
                    dstOffset = loopDim4 * loopInfo.loopDstStride[DIM4] + loopDim3 * loopInfo.loopDstStride[DIM3] +
                                loopDim2 * loopInfo.loopDstStride[DIM2];
                    CopyGmToUbCompact(dstLocal[dstOffset], src, copyParams, isCompact);
                }
            }
        }
    } else {
        DataCopy(dstLocal, src, dmaParams);
    }
}
}  // namespace ViewCopy

#endif  // VIEW_COPY_BASE_H_
