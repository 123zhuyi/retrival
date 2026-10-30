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
 * \file slice_nddma.h
 * \brief
 */
#ifndef SLICE_NDDMA_H
#define SLICE_NDDMA_H

#include "slice_base.h"

namespace Slice {
using namespace AscendC;

template <typename T, typename U, typename V = int8_t>
class SliceNDDMA : public SliceBase<T, U, V> {
public:
    __aicore__ inline SliceNDDMA(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y,
                                const SliceNDDMATilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseNDDMATilingData(
        GM_ADDR begin, const SliceNDDMATilingData* tilingData, int64_t blockIdx);
    __aicore__ inline void SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIMS>& loopInfo,
                                       MultiCopyLoopInfo<NDDMA_MAX_DIMS>& loopInfoTail);
    __aicore__ inline void SetCopyOutParams(DataCopyExtParams &copyOutParamsMain,
                                            DataCopyExtParams &copyOutParamsTail);
    __aicore__ inline void ProcessPerBlock();

private:
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;

    static constexpr MultiCopyConfig nddmaConfig_ = {false, 0, 0, false};

    // 计算中间变量
    int64_t ubSplitLoopsNum_ = 0; // ub切分轴上的循环次数
    int64_t curCoreLoopsNum_ = 0; // ub切分轴之外的循环次数
    int64_t curCoreRowsOffset_ = 0; // 当前核处理的output shape中的起始行数
};

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMA<T, U, V>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                                      GM_ADDR y, const SliceNDDMATilingData* tilingData,
                                                      TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipeIn;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);

    this->ParseNDDMATilingData(begin, tilingData, blockIdx_);

    pipe_->InitBuffer(vecQue_, DOUBLE_BUFFER, this->ubSize_ / DOUBLE_BUFFER);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMA<T, U, V>::ParseNDDMATilingData(GM_ADDR begin, const SliceNDDMATilingData* tilingData,
    int64_t blockIdx)
{
    this->ParseBaseTilingData(begin, &(tilingData->sliceBaseTilingData), blockIdx);
    this->ubOutLoopSteps_ = tilingData->ubOutLoopSteps;

    for (int32_t i = 0; i < STRIDED_SLICE_MAX_AXIS_NUM; i++) {
        this->nddmaLoopSize_[i] = tilingData->nddmaLoopSize[i];
        this->nddmaLoopSrcStride_[i] = tilingData->nddmaLoopSrcStride[i];
        this->nddmaLoopDstStride_[i] = tilingData->nddmaLoopDstStride[i];
    }
    this->nddmaTotalNum_ = tilingData->nddmaTotalNum;
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMA<T, U, V>::Process()
{
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    this->CalcProcessLoopsNum(curCoreLoopsNum_, ubSplitLoopsNum_, blockIdx_);
    this->GetProcessRowsOffset(curCoreRowsOffset_, blockIdx_);

    ProcessPerBlock();
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMA<T, U, V>::SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIMS>& loopInfo,
                                                         MultiCopyLoopInfo<NDDMA_MAX_DIMS>& loopInfoTail)
{
    int64_t inUbDims = this->inputDims_ - this->ubIndex_;
    for (int64_t i = 0; i < inUbDims; i++) {
        // ub main
        loopInfo.loopSize[i] = this->nddmaLoopSize_[NDDMA_MAX_DIMS - 1 - i];
        loopInfo.loopSrcStride[i] = this->nddmaLoopSrcStride_[NDDMA_MAX_DIMS - 1 - i];
        loopInfo.loopDstStride[i] = this->nddmaLoopDstStride_[NDDMA_MAX_DIMS - 1 - i];
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;

        loopInfoTail.loopSize[i] = this->nddmaLoopSize_[NDDMA_MAX_DIMS - 1 - i];
        loopInfoTail.loopSrcStride[i] = this->nddmaLoopSrcStride_[NDDMA_MAX_DIMS - 1 - i];
        loopInfoTail.loopDstStride[i] = this->nddmaLoopDstStride_[NDDMA_MAX_DIMS - 1 - i];
        loopInfoTail.loopLpSize[i] = 0;
        loopInfoTail.loopRpSize[i] = 0;
    }

    loopInfoTail.loopSize[inUbDims - 1] = this->ubTailFactor_;

    // 补维
    for (int64_t i = inUbDims; i < NDDMA_MAX_DIMS; i++) {
        // ub main
        loopInfo.loopSize[i] = 1;
        loopInfo.loopSrcStride[i] = 0;
        loopInfo.loopDstStride[i] = 0;
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
        // ub tail
        loopInfoTail.loopSize[i] = 1;
        loopInfoTail.loopSrcStride[i] = 0;
        loopInfoTail.loopDstStride[i] = 0;
        loopInfoTail.loopLpSize[i] = 0;
        loopInfoTail.loopRpSize[i] = 0;
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMA<T, U, V>::SetCopyOutParams(DataCopyExtParams &copyOutParamsMain,
                                                              DataCopyExtParams &copyOutParamsTail)
{
    copyOutParamsMain.blockCount = 1;
    copyOutParamsMain.blockLen = this->nddmaTotalNum_ * sizeof(T);
    copyOutParamsMain.dstStride = 0;
    copyOutParamsMain.srcStride = 0;

    if (this->ubTailFactor_ > 0) {
        copyOutParamsTail.blockCount = 1;
        copyOutParamsTail.blockLen = this->nddmaTotalNum_ / this->ubFactor_ * this->ubTailFactor_ * sizeof(T);
        copyOutParamsTail.dstStride = 0;
        copyOutParamsTail.srcStride = 0;
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMA<T, U, V>::ProcessPerBlock()
{
    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;
    int64_t handleRowsNum = 0;

    MultiCopyParams<T, NDDMA_MAX_DIMS> paramsMain;
    MultiCopyParams<T, NDDMA_MAX_DIMS> paramsTail;
    paramsMain.constantValue = 0;
    paramsTail.constantValue = 0;
    SetLoopInfo(paramsMain.loopInfo, paramsTail.loopInfo);

    DataCopyExtParams copyOutParamsMain;
    DataCopyExtParams copyOutParamsTail;
    SetCopyOutParams(copyOutParamsMain, copyOutParamsTail);

    for (int64_t idx = 0; idx < curCoreLoopsNum_; idx++) {
        inputGmAddr = this->GetInputGmAddr(curCoreRowsOffset_ + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        outputGmAddr = this->GetOutputGmAddr(curCoreRowsOffset_ + idx * this->rowsOffsetSteps_[this->ubIndex_]);

        for (int64_t loops = 0; loops < ubSplitLoopsNum_; loops++) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopy<T, NDDMA_MAX_DIMS, nddmaConfig_>(inputLocal,
                                                      inputGM_[inputGmAddr + loops * this->ubInLoopSteps_],
                                                      paramsMain);
            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + loops * this->ubOutLoopSteps_], inputLocal, copyOutParamsMain);
            vecQue_.FreeTensor(inputLocal);
        }
        if (this->ubTailFactor_ > 0) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopy<T, NDDMA_MAX_DIMS, nddmaConfig_>(inputLocal,
                inputGM_[inputGmAddr + ubSplitLoopsNum_ * this->ubInLoopSteps_], paramsTail);
            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + ubSplitLoopsNum_ * this->ubOutLoopSteps_],
                        inputLocal, copyOutParamsTail);
            vecQue_.FreeTensor(inputLocal);
        }
    }
}

} // namespace Slice

#endif // SLICE_NDDMA_H