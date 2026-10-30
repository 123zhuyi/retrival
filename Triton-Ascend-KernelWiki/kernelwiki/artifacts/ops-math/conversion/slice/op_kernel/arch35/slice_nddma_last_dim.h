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
 * \file slice_nddma_last_dim.h
 * \brief
 */
#ifndef SLICE_NDDMA_LAST_DIM_H
#define SLICE_NDDMA_LAST_DIM_H

#include "slice_base.h"

namespace Slice {
using namespace AscendC;

template <typename T, typename U, typename V = int8_t>
class SliceNDDMALastDim : public SliceBase<T, U, V> {
public:
    __aicore__ inline SliceNDDMALastDim(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y,
                                const SliceNDDMALastDimTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseNDDMALastDimTilingData(
        GM_ADDR begin, const SliceNDDMALastDimTilingData* tilingData, int64_t blockIdx);
    __aicore__ inline void ProcessPerBlockInLastDim();
    __aicore__ inline void ProcessPerBlock();
    __aicore__ inline void SetLoopInfo(MultiCopyLoopInfo<NDDMA_LAST_DIMS>& loopInfo,
                                       MultiCopyLoopInfo<NDDMA_LAST_DIMS>& loopInfoTail);

private:
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;

    static constexpr MultiCopyConfig nddmaConfig_ = {false, 0, 0, false};

    DataCopyExtParams copyOutParamsMain_ {1, 0, 0, 0, 0};
    DataCopyExtParams copyOutParamsTail_ {1, 0, 0, 0, 0};
};

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMALastDim<T, U, V>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                                      GM_ADDR y, const SliceNDDMALastDimTilingData* tilingData,
                                                      TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipeIn;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);

    this->ParseNDDMALastDimTilingData(begin, tilingData, blockIdx_);

    pipe_->InitBuffer(vecQue_, DOUBLE_BUFFER, this->ubSize_ / DOUBLE_BUFFER);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMALastDim<T, U, V>::ParseNDDMALastDimTilingData(GM_ADDR begin,
    const SliceNDDMALastDimTilingData* tilingData, int64_t blockIdx)
{
    this->ParseBaseTilingData(begin, &(tilingData->sliceBaseTilingData), blockIdx);

    for (int32_t i = 0; i < STRIDED_SLICE_MAX_AXIS_NUM; i++) {
        this->nddmaLoopSrcStride_[i] = tilingData->nddmaLoopSrcStride[i];
        this->nddmaLoopDstStride_[i] = tilingData->nddmaLoopDstStride[i];
    }
}
template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMALastDim<T, U, V>::Process()
{
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    if (this->ubIndex_ != this->inputDims_ - 1) {
        return;
    }

    if (this->blkIndex_ == this->ubIndex_) {
        // blk和ub都切最后一根轴时
        ProcessPerBlockInLastDim();
    } else {
        // ub切最后一根轴，blk切非最后一根轴
        ProcessPerBlock();
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMALastDim<T, U, V>::ProcessPerBlockInLastDim()
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

    MultiCopyParams<T, NDDMA_LAST_DIMS> paramsMain;
    MultiCopyParams<T, NDDMA_LAST_DIMS> paramsTail;
    paramsMain.constantValue = 0;
    paramsTail.constantValue = 0;
    SetLoopInfo(paramsMain.loopInfo, paramsTail.loopInfo);

    copyOutParamsMain_.blockLen = this->ubFactor_ * sizeof(T);
    for (int64_t loops = 0; loops < loopCnt; loops++) {
        LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
        DataCopy<T, NDDMA_LAST_DIMS, nddmaConfig_>(inputLocal,
                                                   inputGM_[inputGmAddr + loops * this->ubInLoopSteps_],
                                                   paramsMain);
        vecQue_.EnQue(inputLocal);
        inputLocal = vecQue_.DeQue<T>();
        DataCopyPad(outputGM_[outputGmAddr + loops * this->ubFactor_], inputLocal, copyOutParamsMain_);
        vecQue_.FreeTensor(inputLocal);
    }

    if (this->ubTailFactor_ > 0) {
        LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
        copyOutParamsTail_.blockLen = this->ubTailFactor_ * sizeof(T);
        DataCopy<T, NDDMA_LAST_DIMS, nddmaConfig_>(inputLocal,
                                                   inputGM_[inputGmAddr + loopCnt * this->ubInLoopSteps_],
                                                   paramsTail);
        vecQue_.EnQue(inputLocal);
        inputLocal = vecQue_.DeQue<T>();
        DataCopyPad(outputGM_[outputGmAddr + loopCnt * this->ubFactor_], inputLocal, copyOutParamsTail_);
        vecQue_.FreeTensor(inputLocal);
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMALastDim<T, U, V>::ProcessPerBlock()
{
    // ub切最后一根轴，ubSplitOutNum表示最后一根轴的搬运次数及尾块处理
    int64_t ubSplitOutNum = this->outputShape_[this->ubIndex_] / this->ubFactor_;
    int64_t curCoreRowsNum = 0;
    int64_t curCoreRowsOffset = 0;
    this->GetHandleRowsNum(curCoreRowsNum, blockIdx_);
    this->GetProcessRowsOffset(curCoreRowsOffset, blockIdx_);

    copyOutParamsMain_.blockLen = this->ubFactor_ * sizeof(T);
    copyOutParamsTail_.blockLen = this->ubTailFactor_ * sizeof(T);

    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;
    MultiCopyParams<T, NDDMA_LAST_DIMS> paramsMain;
    MultiCopyParams<T, NDDMA_LAST_DIMS> paramsTail;
    paramsMain.constantValue = 0;
    paramsTail.constantValue = 0;
    SetLoopInfo(paramsMain.loopInfo, paramsTail.loopInfo);

    int64_t outLoopSteps = this->ubFactor_;
    for (int64_t idx = 0; idx < curCoreRowsNum; idx++) {
        inputGmAddr = this->GetInputGmAddr(curCoreRowsOffset + idx);
        outputGmAddr = this->GetOutputGmAddr(curCoreRowsOffset + idx);
        for (int64_t ubCnt = 0; ubCnt < ubSplitOutNum; ubCnt++) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopy<T, NDDMA_LAST_DIMS, nddmaConfig_>(inputLocal, inputGM_[inputGmAddr + ubCnt * outLoopSteps],
                                                       paramsMain);
            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + ubCnt * this->ubFactor_], inputLocal, copyOutParamsMain_);
            vecQue_.FreeTensor(inputLocal);
        }

        if (this->ubTailFactor_ > 0) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            DataCopy<T, NDDMA_LAST_DIMS, nddmaConfig_>(inputLocal,
                                                       inputGM_[inputGmAddr + ubSplitOutNum * outLoopSteps],
                                                       paramsTail);
            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + ubSplitOutNum * this->ubFactor_], inputLocal,
                        copyOutParamsTail_);
            vecQue_.FreeTensor(inputLocal);
        }
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceNDDMALastDim<T, U, V>::SetLoopInfo(MultiCopyLoopInfo<NDDMA_LAST_DIMS>& loopInfo,
                                                                MultiCopyLoopInfo<NDDMA_LAST_DIMS>& loopInfoTail)
{
    // ub main
    loopInfo.loopSize[0] = this->ubFactor_;
    loopInfo.loopSrcStride[0] = this->nddmaLoopSrcStride_[NDDMA_MAX_DIMS - 1];
    loopInfo.loopDstStride[0] = this->nddmaLoopDstStride_[NDDMA_MAX_DIMS - 1];
    loopInfo.loopLpSize[0] = 0;
    loopInfo.loopRpSize[0] = 0;
    // ub tail
    loopInfoTail.loopSize[0] = this->ubTailFactor_;
    loopInfoTail.loopSrcStride[0] = this->nddmaLoopSrcStride_[NDDMA_MAX_DIMS - 1];
    loopInfoTail.loopDstStride[0] = this->nddmaLoopDstStride_[NDDMA_MAX_DIMS - 1];
    loopInfoTail.loopLpSize[0] = 0;
    loopInfoTail.loopRpSize[0] = 0;
}

} // namespace Slice

#endif // SLICE_NDDMA_LAST_DIM_H