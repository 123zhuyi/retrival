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
 * \file reduce_var_empty.h
 * \brief reduce_var_empty
 */
#ifndef REDUCE_VAR_EMPTY_H
#define REDUCE_VAR_EMPTY_H

namespace ReduceOpTmpl
{
using namespace AscendC;

template <typename T>
class ReduceVarEmpty
{
public:
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline explicit ReduceVarEmpty(T value)
    {
        value_ = value;
    };
    __aicore__ inline void InitTiling(const ReduceVarTilingData* tilingData)
    {
        tiling_ = tilingData;
    }
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR var, GM_ADDR mean);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore();

private:
    int64_t blockIdx_ = 0;
    TPipe *pipe_ = nullptr;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> varGM_;
    GlobalTensor<T> meanGM_;

    TQue<QuePosition::VECOUT, 1> varQue_;

    const ReduceVarTilingData* tiling_ = nullptr;
    int64_t bufferSize_ = 64 * 1024;
    int64_t tailLoopLen_ = 0;
    int64_t addrOffset_ = 0;

    uint64_t loopStartIdx_ = 0;
    uint64_t loopEndIdx_ = 0;
    T value_ = 0;

    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
};

template <typename T>
__aicore__ inline void ReduceVarEmpty<T>::Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR var, GM_ADDR mean)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipeIn;
    bufferSize_ = tiling_->reduceOpTiling.basicBlock == 0 ? bufferSize_ : tiling_->reduceOpTiling.basicBlock;

    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    varGM_.SetGlobalBuffer((__gm__ T*)var);

    if (mean != nullptr) {
        meanGM_.SetGlobalBuffer((__gm__ T*)mean);
    }

    pipe_->InitBuffer(varQue_, BUFFER_NUM, bufferSize_);
}

template <typename T>
__aicore__ inline void ReduceVarEmpty<T>::Process()
{
    if (tiling_->reduceOpTiling.outSize <= 0) {
        return;
    }

    loopStartIdx_ = blockIdx_ * tiling_->reduceOpTiling.factorACntPerCore;
    loopEndIdx_ = loopStartIdx_ + tiling_->reduceOpTiling.factorACntPerCore;
    if (unlikely(loopEndIdx_ > tiling_->reduceOpTiling.factorATotalCnt)) {
        loopEndIdx_ = tiling_->reduceOpTiling.factorATotalCnt;
    }

    tailLoopLen_ = tiling_->reduceOpTiling.outSize % tiling_->reduceOpTiling.ubFactorA;
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void ReduceVarEmpty<T>::ProcessPerCore()
{
    int64_t copyElementNum = tiling_->reduceOpTiling.ubFactorA;
    auto varLocalIn = varQue_.AllocTensor<T>();
    Duplicate<T>(varLocalIn, value_, copyElementNum);
    varQue_.EnQue(varLocalIn);
    auto varLocalOut = varQue_.DeQue<T>();

    for (int64_t loopIdx = loopStartIdx_; loopIdx < loopEndIdx_; loopIdx++) {
        if (loopIdx == tiling_->reduceOpTiling.factorATotalCnt - 1 && tailLoopLen_ != 0) {
            copyElementNum = tailLoopLen_;
        }

        copyOutParams_.blockLen = copyElementNum * sizeof(T);

        DataCopyPad(varGM_[loopIdx * tiling_->reduceOpTiling.ubFactorA], varLocalOut, copyOutParams_);
        if (tiling_->isMeanOut != 0) {
            DataCopyPad(meanGM_[loopIdx * tiling_->reduceOpTiling.ubFactorA], varLocalOut, copyOutParams_);
        }
    }
    varQue_.FreeTensor(varLocalOut);
}
}  // namespace ReduceOpTmpl
#endif  // REDUCE_VAR_EMPTY_H