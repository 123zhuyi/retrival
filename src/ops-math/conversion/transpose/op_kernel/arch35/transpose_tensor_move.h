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
 * \file transpose_tensor_move.h
 * \brief TilingKey=10000 TENSOR_MOVE 策略实现
 *
 * 适用场景：经过 RemoveAxisV2 + MergeAxisV2 后 reduced dim == 1，即输入融合后仅剩单一维度，
 * 等价于纯数据搬运，无需任何转置操作。
 *
 * 核心设计：
 *   - 使用 TQueBind<VECIN, VECOUT, 1> 双缓冲队列（BUFFER_NUM=2），UB 分两半交替使用
 *   - 每次 CopyIn 从 GM 搬运 inUbFactor 个元素到 UB，再 CopyOut 原样写回 GM
 *   - 双缓冲流水线：CopyIn 和 CopyOut 可重叠执行，提高带宽利用率
 *
 * 数据流：GM → DataCopyPad(UB) → DataCopyPad(GM)，纯顺序搬运无转置
 */
#ifndef TRANSPOSE_TENSOR_MOVE_H
#define TRANSPOSE_TENSOR_MOVE_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

/**
 * @brief TENSOR_MOVE 策略类，实现纯数据搬运
 *
 * 当 Transpose 经过轴消除和轴合并后只剩1维时，无需转置，只需将数据从
 * 输入 GM 搬运到输出 GM。使用双缓冲队列实现搬运流水线。
 *
 * @tparam T 数据元素类型（int8_t/int16_t/int32_t/int64_t 或复合类型）
 */
template <typename T>
class TransposeTensorMove : public TransposeBase<T> {
public:
    __aicore__ inline TransposeTensorMove(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore();

private:
    int64_t blockIdx_;

    // buffer
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> vecQue_; // 双缓冲绑定队列，VECIN→VECOUT 自动配对
    GlobalTensor<T> inputGM_;                                     // 输入 GM 张量
    GlobalTensor<T> outputGM_;                                    // 输出 GM 张量

    // tiling params
    const TransposeOpTilingData* tiling_;
    int64_t inputTailFactor_ = 0; // 最后一次 UB 循环的尾块元素数（不满 inUbFactor 时）
    int64_t blockLoopNum_ = 0;    // 当前核需要执行的 UB 循环总次数
    int64_t inLoopNum_ = 0;       // UB 循环次数（= CeilDiv(blockLoopNum_, inUbFactor)）
    int64_t srcOffset_ = 0;       // 当前核在全局数据中的起始偏移（元素数）

    // Datacopy params
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};  // Pad 扩展参数（不使用 Padding）
    DataCopyExtParams copyOutParamsMain_{1, 0, 0, 0, 0}; // CopyOut 参数：blockCount=1, blockLen 动态设置
};

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(vecQue_, BUFFER_NUM, tiling_->ubSize / BUFFER_NUM);
}

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    blockLoopNum_ = tiling_->blkFactor;
    srcOffset_ = blockIdx_ * tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blockLoopNum_ += 1;
        srcOffset_ += blockIdx_;
    } else {
        srcOffset_ += tiling_->blkTailFactor;
    }
    inLoopNum_ = Ops::Base::CeilDiv(blockLoopNum_, tiling_->inUbFactor);
    inputTailFactor_ = blockLoopNum_ % tiling_->inUbFactor;
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::ProcessPerCore()
{
    int64_t copyNum = tiling_->inUbFactor;
    for (int64_t loopIdx = 0; loopIdx < inLoopNum_; loopIdx++) {
        if (loopIdx == inLoopNum_ - 1 && inputTailFactor_ != 0) {
            copyNum = inputTailFactor_;
        }
        copyOutParamsMain_.blockLen = copyNum * sizeof(T);
        LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
        DataCopyPad(bindLocalIn, inputGM_[srcOffset_ + loopIdx * tiling_->inUbFactor], copyOutParamsMain_, padParams_);
        vecQue_.EnQue(bindLocalIn);
        LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
        DataCopyPad(outputGM_[srcOffset_ + loopIdx * tiling_->inUbFactor], bindLocalOut, copyOutParamsMain_);
        vecQue_.FreeTensor(bindLocalOut);
    }
}
} // namespace Transpose

#endif // TRANSPOSE_TENSOR_MOVE_H
