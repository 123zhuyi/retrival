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
 * \file strided_slice_move_align_last_dim.h
 * \brief
 */
#ifndef STRIDED_SLICE_MOVE_ALIGN_LAST_DIM_H
#define STRIDED_SLICE_MOVE_ALIGN_LAST_DIM_H

#include "strided_slice_base.h"

namespace StridedSlice {
using namespace AscendC;

template <typename T, typename U>
class StridedSliceMoveAlignLastDim : public StridedSliceBase<T, U> {
public:
    __aicore__ inline StridedSliceMoveAlignLastDim(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y,
                                const StridedSliceMALastDimTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerBlockInLastDim();
    __aicore__ inline void ProcessPerBlockNLastDim();

private:
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;

    DataCopyPadExtParams<T> padParams_ {false, 0, 0, 0};
    DataCopyExtParams copyParamsMain_ {1, 0, 0, 0, 0};
    DataCopyExtParams copyParamsTail_ {1, 0, 0, 0, 0};
};

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignLastDim<T, U>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                                      GM_ADDR y, const StridedSliceMALastDimTilingData* tilingData, TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();

    pipe_ = pipeIn;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);

    this->ParseBaseTilingDataV2(begin, &(tilingData->stridedSliceBaseTilingData), blockIdx_);

    pipe_->InitBuffer(vecQue_, DOUBLE_BUFFER, this->ubSize_ / DOUBLE_BUFFER);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignLastDim<T, U>::Process()
{
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    if (this->ubIndex_ != this->inputDims_ - 1) {
        return;
    }

    if (this->blkIndex_ != this->ubIndex_) {
        // ub切最后一根轴，blk切非最后一根轴
        ProcessPerBlockNLastDim();
    } else {
        // blk和ub都切最后一根轴时
        ProcessPerBlockInLastDim();
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignLastDim<T, U>::ProcessPerBlockInLastDim()
{
    // blk切最后一根轴，表示 blkSplitOutNum 个核处理一行
    // 该block处理的第几行
    int64_t blkProcessRowNum = blockIdx_ / this->blkSplitOutNum_;
    // 该行在input中的偏移地址
    int64_t inputGmAddr = this->GetInputGmAddr(blkProcessRowNum);
    // 该blk处理数据在该行中的偏移
    int64_t offsetGmAddr = (blockIdx_ % this->blkSplitOutNum_) * this->blkFactor_;
    inputGmAddr = inputGmAddr + offsetGmAddr;
    // 该行在output中的偏移地址
    int64_t outputGmAddr = this->GetOutputGmAddr(blkProcessRowNum) + offsetGmAddr;

    int64_t loopCnt = 0;
    this->GetLastDimSplitLoopCnt(loopCnt, blockIdx_);

    copyParamsMain_.blockLen = this->ubFactor_ * sizeof(T);
    for (int64_t idx = 0; idx < loopCnt; idx++) {
        LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
        DataCopyPad(inputLocal, inputGM_[inputGmAddr + idx * this->ubFactor_], copyParamsMain_, padParams_);

        vecQue_.EnQue(inputLocal);
        inputLocal = vecQue_.DeQue<T>();

        DataCopyPad(outputGM_[outputGmAddr + idx * this->ubFactor_], inputLocal, copyParamsMain_);
        vecQue_.FreeTensor(inputLocal);
    }

    if (this->ubTailFactor_ != 0) {
        LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
        copyParamsTail_.blockLen = this->ubTailFactor_ * sizeof(T);
        DataCopyPad(inputLocal, inputGM_[inputGmAddr + loopCnt * this->ubFactor_], copyParamsTail_, padParams_);

        vecQue_.EnQue(inputLocal);
        inputLocal = vecQue_.DeQue<T>();

        DataCopyPad(outputGM_[outputGmAddr + loopCnt * this->ubFactor_], inputLocal, copyParamsTail_);
        vecQue_.FreeTensor(inputLocal);
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignLastDim<T, U>::ProcessPerBlockNLastDim()
{
    // ub切最后一根轴，ubSplitOutNum表示最后一根轴的搬运次数及尾块处理
    int64_t ubSplitOutNum = this->outputShape_[this->ubIndex_] / this->ubFactor_;
    int64_t curCoreRowsNum = 0;
    int64_t curCoreRowsOffset = 0;
    this->GetHandleRowsNum(curCoreRowsNum, blockIdx_);
    this->GetProcessRowsOffset(curCoreRowsOffset, blockIdx_);

    copyParamsMain_.blockLen = this->ubFactor_ * sizeof(T);
    copyParamsTail_.blockLen = this->ubTailFactor_ * sizeof(T);

    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;
    for (int64_t idx = 0; idx < curCoreRowsNum; idx++) {
        inputGmAddr = this->GetInputGmAddr(curCoreRowsOffset + idx);
        outputGmAddr = this->GetOutputGmAddr(curCoreRowsOffset + idx);
        for (int64_t ubCnt = 0; ubCnt < ubSplitOutNum; ubCnt++) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopyPad(inputLocal, inputGM_[inputGmAddr + ubCnt * this->ubFactor_], copyParamsMain_, padParams_);

            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();

            DataCopyPad(outputGM_[outputGmAddr + ubCnt * this->ubFactor_], inputLocal, copyParamsMain_);
            vecQue_.FreeTensor(inputLocal);
        }

        if (this->ubTailFactor_ != 0) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopyPad(inputLocal, inputGM_[inputGmAddr + ubSplitOutNum * this->ubFactor_],
                        copyParamsTail_, padParams_);

            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();

            DataCopyPad(outputGM_[outputGmAddr + ubSplitOutNum * this->ubFactor_], inputLocal, copyParamsTail_);
            vecQue_.FreeTensor(inputLocal);
        }
    }
}

} // namespace StridedSlice

#endif // STRIDED_SLICE_MOVE_ALIGN_LAST_DIM_H