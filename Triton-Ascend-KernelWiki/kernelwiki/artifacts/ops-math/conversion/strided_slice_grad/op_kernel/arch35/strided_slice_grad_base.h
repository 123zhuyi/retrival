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
 * \file strided_slice_grad_base.h
 * \brief
 */

#ifndef STRIDED_SLICE_GRAD_BASE_H
#define STRIDED_SLICE_GRAD_BASE_H

#include "kernel_operator.h"

namespace StridedSliceGrad
{
using namespace AscendC;

constexpr int32_t DB_BUFFER = 2;
constexpr uint32_t THREAD_DIM = 512;
constexpr uint32_t MAX_SUPPORT_DIM = 8;
constexpr uint32_t ONE_BLOCK_BYTES = 32;
constexpr uint32_t NDDMA_MAX_DIMS = 5;

constexpr uint32_t DIM0 = 7;
constexpr uint32_t DIM1 = 6;
constexpr uint32_t DIM2 = 5;
constexpr uint32_t DIM3 = 4;
constexpr uint32_t DIM4 = 3;
constexpr uint32_t DIM5 = 2;
constexpr uint32_t DIM6 = 1;
constexpr uint32_t DIM7 = 0;

constexpr uint64_t TOTAL_DIGITS = 64;
constexpr uint64_t OFFSET = 32;

struct arrayParam{
    uint32_t m[MAX_SUPPORT_DIM];
    uint32_t shift[MAX_SUPPORT_DIM];
    int64_t fusedSliceInnerShape[MAX_SUPPORT_DIM];
    int64_t begin[MAX_SUPPORT_DIM];
    int64_t strides[MAX_SUPPORT_DIM];
    int64_t fusedOutputInnerShape[MAX_SUPPORT_DIM];
};

class StridedSliceGradBase
{
public:
    __aicore__ inline StridedSliceGradBase(){};

public:
    uint32_t usedCoreNum_;
    uint64_t bufferSize_;
    uint32_t inputDimNum_;
    uint32_t bytesForOneData_;
    uint32_t normalCoreProcessNum_;
    uint32_t tailCoreProcessNum_;

    uint32_t curCoreProcessNum_;

    // 高维->低维
    int64_t inputShape_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t outputShape_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t begin_[MAX_SUPPORT_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t end_[MAX_SUPPORT_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t strides_[MAX_SUPPORT_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t fusedOutputInnerShape_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t fusedSliceInnerShape_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t fusedSliceNoEndAxis_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t fusedOutputNoEndAxis_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};

protected:
    __aicore__ inline void ParseTilingData(const StridedSliceGradTilingData& tilingData);
    __aicore__ inline void CopyArray(const int64_t* src, int64_t* dst);
};

__aicore__ inline void StridedSliceGradBase::ParseTilingData(const StridedSliceGradTilingData& tilingData)
{
    usedCoreNum_ = tilingData.usedCoreNum;
    bufferSize_ = tilingData.bufferSize;
    inputDimNum_ = tilingData.inputDimNum;
    bytesForOneData_ = tilingData.bytesForOneData;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;

    CopyArray(tilingData.inputShape, inputShape_);
    CopyArray(tilingData.outputShape, outputShape_);
    CopyArray(tilingData.begin, begin_);
    CopyArray(tilingData.end, end_);
    CopyArray(tilingData.strides, strides_);
    CopyArray(tilingData.fusedOutputInnerShape, fusedOutputInnerShape_);
    CopyArray(tilingData.fusedSliceInnerShape, fusedSliceInnerShape_);

    for (uint32_t i = 0; i < MAX_SUPPORT_DIM - 1; i++) {
        fusedSliceNoEndAxis_[i] = fusedSliceInnerShape_[i] / inputShape_[DIM0];
        fusedOutputNoEndAxis_[i] = fusedOutputInnerShape_[i] / outputShape_[DIM0];
    }
}

__aicore__ inline void StridedSliceGradBase::CopyArray(const int64_t* src, int64_t* dst)
{
    for (uint32_t i = 0; i < MAX_SUPPORT_DIM; i++) {
        dst[i] = src[i];
    }
}

}  // namespace StridedSliceGrad

#endif  // STRIDED_SLICE_GRAD_BASE_H