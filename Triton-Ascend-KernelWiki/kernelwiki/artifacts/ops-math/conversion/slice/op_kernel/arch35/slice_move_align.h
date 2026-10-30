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
 * \file slice_move_align.h
 * \brief
 */
#ifndef SLICE_MOVE_ALIGN_H
#define SLICE_MOVE_ALIGN_H

#include "slice_base.h"

namespace Slice {
using namespace AscendC;

template <typename T, typename U, typename V = int8_t>
class SliceMoveAlign : public SliceBase<T, U, V> {
public:
    __aicore__ inline SliceMoveAlign(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR y,
                                const SliceMoveAlignTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseMoveAlignTilingData(
        GM_ADDR begin, const SliceMoveAlignTilingData* tilingData, int64_t blockIdx);
    __aicore__ inline void ProcessPerBlock(const DataCopyExtParams &copyOutParamsMain,
                                           const DataCopyExtParams &copyOutParamsTail);
    __aicore__ inline void ParseCopyInTilingData(const SliceMoveAlignTilingData* tilingData);
    __aicore__ inline void ParseLoopModeAndMoveAlignParams(LoopModeParams &loopMode,
        DataCopyExtParams &extParams, const SliceMoveAlignTilingData* tilingData) const;
    __aicore__ inline void SetCopyOutAlignParams(DataCopyExtParams &copyOutParams,
                                                 const DataCopyExtParams &copyInParam,
                                                 const LoopModeParams &loopMode);

private:
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> vecQue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;

    LoopModeParams loopMode_;
    LoopModeParams loopModeTail_;
    DataCopyExtParams copyInParam_;
    DataCopyExtParams copyInParamTail_;

    DataCopyPadExtParams<T> padParams_ {false, 0, 0, 0};
};

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlign<T, U, V>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                                      GM_ADDR y, const SliceMoveAlignTilingData* tilingData,
                                                      TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipeIn;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);

    this->ParseMoveAlignTilingData(begin, tilingData, blockIdx_);
    // gm -> ub move_align_v2 params
    ParseCopyInTilingData(tilingData);

    pipe_->InitBuffer(vecQue_, DOUBLE_BUFFER, this->ubSize_ / DOUBLE_BUFFER);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlign<T, U, V>::ParseMoveAlignTilingData(GM_ADDR begin,
    const SliceMoveAlignTilingData* tilingData, int64_t blockIdx)
{
    this->ParseBaseTilingData(begin, &(tilingData->sliceBaseTilingData), blockIdx);
    this->ubOutLoopSteps_ = tilingData->ubOutLoopSteps;
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlign<T, U, V>::Process()
{
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    // ub -> gm
    DataCopyExtParams copyOutParamsMain;
    DataCopyExtParams copyOutParamsTail;
    SetCopyOutAlignParams(copyOutParamsMain, copyInParam_, loopMode_);
    if (this->ubTailFactor_ > 0) {
        SetCopyOutAlignParams(copyOutParamsTail, copyInParamTail_, loopModeTail_);
    }

    ProcessPerBlock(copyOutParamsMain, copyOutParamsTail);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlign<T, U, V>::ParseLoopModeAndMoveAlignParams(LoopModeParams &loopMode,
    DataCopyExtParams &extParams, const SliceMoveAlignTilingData* tilingData) const
{
    loopMode.loop1Size = tilingData->moveAlignParams.loop1Size;
    loopMode.loop2Size = tilingData->moveAlignParams.loop2Size;
    loopMode.loop1SrcStride = tilingData->moveAlignParams.loop1SrcStride;
    loopMode.loop1DstStride = tilingData->moveAlignParams.loop1DstStride;
    loopMode.loop2SrcStride = tilingData->moveAlignParams.loop2SrcStride;
    loopMode.loop2DstStride = tilingData->moveAlignParams.loop2DstStride;

    extParams.blockCount = tilingData->moveAlignParams.blockCount;
    extParams.blockLen = tilingData->moveAlignParams.blockLen;
    extParams.srcStride = tilingData->moveAlignParams.srcStride - extParams.blockLen; // DataCopyPad接口要求是尾和头的间隔
    extParams.dstStride = tilingData->moveAlignParams.dstStride;
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlign<T, U, V>::ParseCopyInTilingData(const SliceMoveAlignTilingData* tilingData)
{
    /********************************************
    1、blk和ub切分不同轴时:
        (A, B, C, D, E) blk切B, ub切C
        主核和尾核的外循环次数不同，内循环(ub切分轴上的循环)次数都一样。
        所有核的单次搬运都只有两类参数：ub_factor(loopMode和copyInParam)/ub_tail_factor(loopModeTail和copyInParamTail)
    2、blk和ub切分相同轴时： blkFactor >= ubFactor
        (A, B, C, D, E) blk切C, ub切C
        C = 900   blkFactor=200  ubFactor=150
        则: blkTailFactor=900%200=100  ubTailFactor=blkFactor%ubFactor=50 ubTailTailFactor=blkTailFactor%ubFactor=100
        外循环次数都为1
        (1) 主核:
            内循环次数：主块 blkFactor / ubFactor， 尾块 ubTailFactor = blkFactor % ubFactor
            搬运的loop参数： 主块 ubFactor   尾块 ubTailFactor = blkFactor % ubFactor
        (2) 尾核：
            内循环次数：主块 blkTailFactor / ubFactor， 尾块 ubTailTailFactor = blkTailFactor % ubFactor
            搬运的loop参数： 主块 ubFactor   尾块 ubTailTailFactor = blkTailFactor % ubFactor
    ********************************************/
    // main loop
    ParseLoopModeAndMoveAlignParams(loopMode_, copyInParam_, tilingData);

    if (this->ubTailFactor_ > 0) {
        ParseLoopModeAndMoveAlignParams(loopModeTail_, copyInParamTail_, tilingData);
        int64_t inUbDims = this->inputDims_ - this->ubIndex_;
        if (inUbDims == DIMS_2) {
            copyInParamTail_.blockCount = this->ubTailFactor_;
        } else if (inUbDims == DIMS_3) {
            loopModeTail_.loop1Size = this->ubTailFactor_;
        } else {
            loopModeTail_.loop2Size = this->ubTailFactor_;
        }
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlign<T, U, V>::SetCopyOutAlignParams(DataCopyExtParams &copyOutParams,
                                                                       const DataCopyExtParams &copyInParam,
                                                                       const LoopModeParams &loopMode)
{
    // blockCount 为uint16，不需要考虑越界。因为最后一根轴只有大于128B时才选择move_align_v2模板
    // MTE2 nburst * burstLen 要求Block对齐, 在ub内可能存在pad
    int64_t inUbDims = this->inputDims_ - this->ubIndex_;
    if (inUbDims <= DIMS_2) {
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyInParam.blockCount * copyInParam.blockLen;
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
    } else if ((copyInParam.blockCount * copyInParam.blockLen) % BLOCK_SIZE_BYTE == 0) {
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = loopMode.loop1Size * loopMode.loop2Size *
                                 copyInParam.blockCount * copyInParam.blockLen;
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
    } else {
        copyOutParams.blockCount = loopMode.loop1Size * loopMode.loop2Size;
        copyOutParams.blockLen = copyInParam.blockCount * copyInParam.blockLen;
        copyOutParams.dstStride = 0; // 头和尾
        copyOutParams.srcStride = 0; // 头和尾. AscendC会将blockLen向32Bytes对齐, CopyUbufToGmAlignV2
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlign<T, U, V>::ProcessPerBlock(const DataCopyExtParams &copyOutParamsMain,
                                                                 const DataCopyExtParams &copyOutParamsTail)
{
    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;
    int64_t ubSplitLoopsNum = 0;  // ub切分轴上的循环次数
    int64_t ubOuterLoopsNum = 0;  // ub切分轴之外的循环次数
    int64_t rowsOffsetOutput = 0; // 当前核处理的output shape中的起始行数

    this->CalcProcessLoopsNum(ubOuterLoopsNum, ubSplitLoopsNum, blockIdx_);
    this->GetProcessRowsOffset(rowsOffsetOutput, blockIdx_);

    for (int64_t idx = 0; idx < ubOuterLoopsNum; idx++) {
        inputGmAddr = this->GetInputGmAddr(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        outputGmAddr = this->GetOutputGmAddr(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);

        for (int64_t loops = 0; loops < ubSplitLoopsNum; loops++) {
            // 1、启用loop  mode：调用SetLoopModePara接口，打开loop模式，调用datacopy
            // 2、关闭：调用ResetLoopModePara接口，关闭相应模式
            // 3、注意：loop模式如果打开了，需要手动关闭，否则默认就全部都是开的
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
            DataCopyPad<T, PaddingMode::Compact>(inputLocal, inputGM_[inputGmAddr + loops * this->ubInLoopSteps_],
                                                 copyInParam_, padParams_);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();

            DataCopyPad(outputGM_[outputGmAddr + loops * this->ubOutLoopSteps_], inputLocal, copyOutParamsMain);
            vecQue_.FreeTensor(inputLocal);
        }

        if (this->ubTailFactor_ > 0) {
            LocalTensor<T> inputLocal = vecQue_.AllocTensor<T>();
            SetLoopModePara(loopModeTail_, DataCopyMVType::OUT_TO_UB);
            DataCopyPad<T, PaddingMode::Compact>(inputLocal,
                                                 inputGM_[inputGmAddr + ubSplitLoopsNum * this->ubInLoopSteps_],
                                                 copyInParamTail_, padParams_);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

            vecQue_.EnQue(inputLocal);
            inputLocal = vecQue_.DeQue<T>();

            DataCopyPad(outputGM_[outputGmAddr + ubSplitLoopsNum * this->ubOutLoopSteps_],
                        inputLocal, copyOutParamsTail);
            vecQue_.FreeTensor(inputLocal);
        }
    }
}

} // namespace Slice

#endif // SLICE_MOVE_ALIGN_H