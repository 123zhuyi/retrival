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
 * \file matrix_diag_tensor_move.h
 * \brief
 */
#ifndef MATRIX_DIAG_TENSOR_MOVE_H
#define MATRIX_DIAG_TENSOR_MOVE_H
#include "kernel_operator.h"
#include "matrix_diag_tiling_data_struct.h"

namespace MatrixDiagAsc {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class MatrixDiagTensorMove {
public:
    __aicore__ inline MatrixDiagTensorMove(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MatrixDiagPureCopyTilingData *tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyToCache();

private:
    int64_t blockIdx_;

    // buffer
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inputQueue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    // tiling params
    const MatrixDiagPureCopyTilingData* tilingData_;
    int64_t mainLoopNum_ = 0;
    int64_t peerloopElements_ = 0;
    int64_t tailElements_ = 0;
    int64_t inputOffset_ = 0;
    uint32_t maxUbUsed_ = 0;
};


template <typename T>
__aicore__ inline void MatrixDiagTensorMove<T>::Init(GM_ADDR x, GM_ADDR y, const MatrixDiagPureCopyTilingData *tilingData,
                                                     TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    maxUbUsed_ = tilingData_->ubSize / BUFFER_NUM;
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    uint32_t curBlockFactor = (blockIdx_ < tilingData_->mainBlockCount) ?
                              tilingData_->mainBlockFactor : tilingData_->tailBlockFactor;
    peerloopElements_ = maxUbUsed_ / sizeof(T);
    mainLoopNum_ = curBlockFactor / peerloopElements_;  // main ub factor count
    tailElements_ = curBlockFactor - mainLoopNum_ * peerloopElements_;
    inputOffset_ = (blockIdx_ < tilingData_->mainBlockCount) ?
                    tilingData_->mainBlockFactor * blockIdx_ :
                    tilingData_->mainBlockFactor * tilingData_->mainBlockCount +
                    tilingData_->tailBlockFactor * (blockIdx_ - tilingData_->mainBlockCount);

    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(inputQueue_, BUFFER_NUM, peerloopElements_ * sizeof(T));
}

template <typename T>
__aicore__ inline void MatrixDiagTensorMove<T>::Process()
{
    CopyToCache();
}

template <typename T>
__aicore__ inline void MatrixDiagTensorMove<T>::CopyToCache()
{
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(peerloopElements_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(peerloopElements_ * sizeof(T)), 0, 0, 0};
    for (int64_t loopIdx = 0; loopIdx < mainLoopNum_; loopIdx++) {
        // CopyIn
        LocalTensor<T> input = inputQueue_.AllocTensor<T>();
        DataCopyPad(input, inputGM_[inputOffset_ + loopIdx * peerloopElements_], copyInParams, padParams);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T>();
        DataCopyPad(outputGM_[inputOffset_ + loopIdx * peerloopElements_], input, copyOutParams);
        inputQueue_.FreeTensor(input);
    }

    if (tailElements_ > 0) {
        copyInParams.blockLen = tailElements_ * sizeof(T);
        copyOutParams.blockLen = tailElements_ * sizeof(T);
        // CopyIn
        LocalTensor<T> input = inputQueue_.AllocTensor<T>();
        DataCopyPad(input, inputGM_[inputOffset_ + mainLoopNum_ * peerloopElements_], copyInParams, padParams);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T>();
        DataCopyPad(outputGM_[inputOffset_ + mainLoopNum_ * peerloopElements_], input, copyOutParams);
        inputQueue_.FreeTensor(input);
    }
}
}  // namespace MatrixDiag

#endif  // MATRIX_DIAG_TENSOR_MOVE_H