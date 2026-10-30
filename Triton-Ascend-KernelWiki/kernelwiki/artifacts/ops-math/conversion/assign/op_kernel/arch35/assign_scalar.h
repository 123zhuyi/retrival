/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file assign_scalar.h
 * \brief
 */

#ifndef ASSIGN_SCALAR_H_
#define ASSIGN_SCALAR_H_

#include "kernel_operator.h"

namespace AssignScalar {
using namespace AscendC;
constexpr int64_t DB_BUFFER = 2;
template <typename T> class AssignScalarKernel {
public:
    __aicore__ inline AssignScalarKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TensorMoveTilingData &tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const TensorMoveTilingData &tilingData);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen, LocalTensor<T> yLocal);

private:
    TPipe pipe_;
    TQue<QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t totalCoreNum_;
    int64_t usedCoreNum_;
    int64_t ubFactor_;
    int64_t tailBlockTailUbFactor_;
    int64_t blockFactor_;
    int64_t tailBlockFactor_;
    int64_t bufferSize_;
    T scalar_;
};

template <typename T>
__aicore__ inline void AssignScalarKernel<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const TensorMoveTilingData &tilingData)
{
    blockIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);
    blockOffset_ = GetBlockIdx() * blockFactor_ * ubFactor_;
    xGm_.SetGlobalBuffer((__gm__ T *)(x));
    yGm_.SetGlobalBuffer((__gm__ T *)(y) + blockOffset_);
    scalar_ = xGm_.GetValue(0);
    bufferSize_ = ubFactor_ * sizeof(T);
    pipe_.InitBuffer(dataQueue_, DB_BUFFER, bufferSize_);
}

template <typename T>
__aicore__ inline void AssignScalarKernel<T>::ParseTilingData(const TensorMoveTilingData &tilingData)
{
    totalCoreNum_ = tilingData.totalCoreNum;
    usedCoreNum_ = tilingData.usedCoreNum;
    ubFactor_ = tilingData.ubFactor;
    tailBlockTailUbFactor_ = tilingData.tailBlockTailUbFactor;
    blockFactor_ = tilingData.blockFactor;
    tailBlockFactor_ = tilingData.tailBlockFactor;
}

template <typename T>
__aicore__ inline void AssignScalarKernel<T>::CopyOut(int64_t offset, int64_t dataLen, LocalTensor<T> yLocal)
{
  DataCopyExtParams outParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
 
  DataCopyPad(yGm_[offset], yLocal, outParams);
}

template <typename T> __aicore__ inline void AssignScalarKernel<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t loopSize = blockFactor_;
    if (blockIdx_ == usedCoreNum_ - 1) {
        loopSize = tailBlockFactor_;
    }
    int64_t offset = 0;
    int64_t dataLen = ubFactor_;
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    Duplicate(xLocal, scalar_, ubFactor_);
    dataQueue_.EnQue(xLocal);
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();

    for (int64_t idx = 0; idx < loopSize - 1; idx++) {
        offset = idx * ubFactor_;
        CopyOut(offset, dataLen, yLocal);
    }

    offset = (loopSize - 1) * ubFactor_;

    if (blockIdx_ == usedCoreNum_ - 1) {
        dataLen = tailBlockTailUbFactor_;
    }
    CopyOut(offset, dataLen, yLocal);

    dataQueue_.FreeTensor(yLocal);
}
} // namespace AssignScalar

#endif // ASSIGN_SCALAR_H_
