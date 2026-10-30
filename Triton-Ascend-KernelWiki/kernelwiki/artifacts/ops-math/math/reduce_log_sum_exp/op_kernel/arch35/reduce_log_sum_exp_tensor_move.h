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
 * \file reduce_log_sum_exp_tensor_move.h
 * \brief reduce_log_sum_exp tensor move kernel
 */

#ifndef REDUCE_LOG_SUM_EXP_TENSOR_MOVE_H
#define REDUCE_LOG_SUM_EXP_TENSOR_MOVE_H

#include "atvoss/reduce/reduce_tiling_data.h"

namespace ReduceLogSumExpTmpl
{
using namespace AscendC;
using namespace Ops::Base;

template <typename T>
class ReduceLogSumExpTensorMove
{
public:
    __aicore__ inline ReduceLogSumExpTensorMove(){};

    __aicore__ inline void Init(const ReduceOpTilingData* tilingData, TPipe* pipeIn, 
                                GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore();

private:
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    constexpr static int32_t BUFFER_NUM = 2;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> vecQue_;

    const ReduceOpTilingData* tiling_;
    int64_t tailLoopLen_ = 0;
    int64_t addrOffset_ = 0;

    uint64_t loopStartIdx_ = 0;
    uint64_t loopEndIdx_ = 0;

    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
};

template <typename T>
__aicore__ inline void ReduceLogSumExpTensorMove<T>::Init(const ReduceOpTilingData* tilingData, TPipe* pipeIn,
                                                          GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    tiling_ = tilingData;
    input_.SetGlobalBuffer((__gm__ T*)x);
    output_.SetGlobalBuffer((__gm__ T*)y);
    pipeIn->InitBuffer(vecQue_, BUFFER_NUM, tiling_->basicBlock);
}

template <typename T>
__aicore__ inline void ReduceLogSumExpTensorMove<T>::Process()
{
    int64_t blockIdx = GetBlockIdx();
    loopStartIdx_ = blockIdx * tiling_->factorACntPerCore;
    loopEndIdx_ = loopStartIdx_ + tiling_->factorACntPerCore;
    if (unlikely(loopEndIdx_ > tiling_->factorATotalCnt)) {
        loopEndIdx_ = tiling_->factorATotalCnt;
    }

    tailLoopLen_ = tiling_->outSize % tiling_->ubFactorA;
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void ReduceLogSumExpTensorMove<T>::ProcessPerCore()
{
    int64_t copyElementNum = tiling_->ubFactorA;
    for (int64_t loopIdx = loopStartIdx_; loopIdx < loopEndIdx_; loopIdx++) {
        if (loopIdx == tiling_->factorATotalCnt - 1 && tailLoopLen_ != 0) {
            copyElementNum = tailLoopLen_;
        }

        auto bindLocalIn = vecQue_.AllocTensor<T>();
        copyOutParams_.blockLen = copyElementNum * sizeof(T);
        DataCopyPad(bindLocalIn, input_[loopIdx * tiling_->ubFactorA], copyOutParams_, padParams_);
        vecQue_.EnQue(bindLocalIn);

        auto bindLocalOut = vecQue_.DeQue<T>();
        DataCopyPad(output_[loopIdx * tiling_->ubFactorA], bindLocalOut, copyOutParams_);

        vecQue_.FreeTensor(bindLocalOut);
    }
}

}  // namespace ReduceOpTmpl
 
#endif  // REDUCE_TENSOR_MOVE_H