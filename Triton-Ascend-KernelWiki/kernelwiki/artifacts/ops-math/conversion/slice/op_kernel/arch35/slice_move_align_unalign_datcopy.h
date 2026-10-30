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
 * \file slice_move_align_unalign_datcopy.h
 * \brief
 */
#ifndef SLICE_MOVE_ALIGN_UNALIGN_DATCOPY_H
#define SLICE_MOVE_ALIGN_UNALIGN_DATCOPY_H

#include <cstdlib>
#include <type_traits>

#include "slice_base.h"

namespace Slice {
using namespace AscendC;

template <typename T, typename U, typename V = int8_t>
class SliceMoveAlignDataCopyUnalign : public SliceBase<T, U, V> {
public:
    __aicore__ inline SliceMoveAlignDataCopyUnalign(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y, const SliceMoveAlignGatherTilingData* tdPtr,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseMoveAlignGatherTilingData(GM_ADDR begin, const SliceMoveAlignGatherTilingData* tdPtr,
                                                          int64_t blockIdx);
    __aicore__ inline void ProcessPerBlock(const DataCopyExtParams& copyOutParamsMain,
                                           const DataCopyExtParams& copyOutParamsTail);
    __aicore__ inline void ParseCopyInTilingData();
    __aicore__ inline void ParseLoopModeAndMoveAlignParams(LoopModeParams& loopMode, DataCopyExtParams& extParams);
    __aicore__ inline void SetCopyOutAlignParams(DataCopyExtParams& copyOutParams, const DataCopyExtParams& copyInParam,
                                                 const LoopModeParams& loopMode, const uint32_t blockLen);
    __aicore__ inline void DataCopyUnalignGather(uint32_t blockCount);

private:
    TPipe* pipe_ = nullptr;
    const SliceMoveAlignGatherTilingData* tdPtr_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;
    uint8_t bufferCnt_ = 2; // enable db
    int64_t inUbDims_ = 0;

    LoopModeParams loopMode_;
    LoopModeParams loopModeTail_;
    DataCopyExtParams copyInParam_;
    DataCopyExtParams copyInParamTail_;

    uint32_t outputMainblockLen_ = 0;
    uint32_t outputTailblockLen_ = 0;
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    uint32_t lastOneDimOutputDim_ = 0;
    uint32_t vlCnt_ = Ops::Base::GetVRegSize() / sizeof(T);
};

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y,
                                                                    const SliceMoveAlignGatherTilingData* tdPtr,
                                                                    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    tdPtr_ = tdPtr;
    this->ParseMoveAlignGatherTilingData(begin, tdPtr_, blockIdx_);
    inUbDims_ = this->inputDims_ - this->ubIndex_;
    // gm -> ub move_align_v2 params
    ParseCopyInTilingData();

    // pipeBuffer
    pipe_ = pipe;
    pipe_->InitBuffer(inQue_, bufferCnt_, tdPtr_->ubSizeInput);
    pipe_->InitBuffer(outQue_, bufferCnt_, this->ubSize_);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::ParseMoveAlignGatherTilingData(
    GM_ADDR begin, const SliceMoveAlignGatherTilingData* tdPtr, int64_t blockIdx)
{
    this->ParseBaseTilingData(begin, &(tdPtr->sliceBaseTilingData), blockIdx);
    this->ubOutLoopSteps_ = tdPtr->ubOutLoopSteps;
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::Process()
{
    // for empty tensor set realCoreNum as 0, do nothing
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    // ub -> gm
    DataCopyExtParams copyOutParamsMain;
    DataCopyExtParams copyOutParamsTail;
    SetCopyOutAlignParams(copyOutParamsMain, copyInParam_, loopMode_, outputMainblockLen_);
    if (this->ubTailFactor_ > 0) {
        SetCopyOutAlignParams(copyOutParamsTail, copyInParamTail_, loopModeTail_, outputTailblockLen_);
    }

    ProcessPerBlock(copyOutParamsMain, copyOutParamsTail);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::ParseLoopModeAndMoveAlignParams(
    LoopModeParams& loopMode, DataCopyExtParams& extParams)
{
    loopMode.loop1Size = tdPtr_->moveAlignParams.loop1Size;
    loopMode.loop2Size = tdPtr_->moveAlignParams.loop2Size;
    loopMode.loop1SrcStride = tdPtr_->moveAlignParams.loop1SrcStride;
    loopMode.loop1DstStride = tdPtr_->moveAlignParams.loop1DstStride;
    loopMode.loop2SrcStride = tdPtr_->moveAlignParams.loop2SrcStride;
    loopMode.loop2DstStride = tdPtr_->moveAlignParams.loop2DstStride;

    extParams.blockCount = tdPtr_->moveAlignParams.blockCount;
    extParams.blockLen = tdPtr_->moveAlignParams.blockLen;
    extParams.srcStride = tdPtr_->moveAlignParams.srcStride;
    extParams.dstStride = 0;
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::ParseCopyInTilingData()
{
    outputMainblockLen_ = tdPtr_->outBlockLen;
    outputTailblockLen_ = outputMainblockLen_;
    // main loop
    ParseLoopModeAndMoveAlignParams(loopMode_, copyInParam_);

    if (this->ubTailFactor_ > 0) {
        ParseLoopModeAndMoveAlignParams(loopModeTail_, copyInParamTail_);
        if (inUbDims_ == DIMS_2) {
            copyInParamTail_.blockLen = copyInParam_.blockLen / this->ubFactor_ * this->ubTailFactor_;
            outputTailblockLen_ = outputMainblockLen_ / this->ubFactor_ * this->ubTailFactor_;
        } else if (inUbDims_ == DIMS_3) {
            copyInParamTail_.blockCount = this->ubTailFactor_;
        } else if (inUbDims_ == DIMS_4) {
            loopModeTail_.loop1Size = this->ubTailFactor_;
        }
    }
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::SetCopyOutAlignParams(
    DataCopyExtParams& copyOutParams, const DataCopyExtParams& copyInParam, const LoopModeParams& loopMode,
    const uint32_t blockLen)
{
    copyOutParams.blockCount = copyInParam.blockCount * loopMode.loop1Size;
    copyOutParams.blockLen = blockLen;
    copyOutParams.dstStride = 0;
    copyOutParams.srcStride = 0;
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::DataCopyUnalignGather(uint32_t blockCount)
{
    auto inTensor = inQue_.DeQue<T>();
    __local_mem__ T* srcPtr = (__local_mem__ T*)inTensor.GetPhyAddr();
    auto outTensor = outQue_.AllocTensor<T>();
    __local_mem__ T* dstPtr = (__local_mem__ T*)outTensor.GetPhyAddr();
    uint32_t lastOneDimBegin = this->begin_[this->inputDims_ - 1];
    uint32_t lastOneInputDim = tdPtr_->lastOneInputDim;
    uint32_t lastOneOutputDim = this->outputShape_[this->inputDims_ - 1];
    uint32_t perLoopInputPadElem = Ops::Base::CeilAlign(tdPtr_->moveAlignParams.blockLen, uint32_t(BLOCK_SIZE_BYTE)) /
                                   sizeof(T);
    uint32_t perLoopOutputPadElem = Ops::Base::CeilAlign(tdPtr_->outBlockLen, uint32_t(BLOCK_SIZE_BYTE)) / sizeof(T);
    uint32_t repeatTimes = tdPtr_->moveAlignParams.blockLen / sizeof(T) / lastOneInputDim;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vd0;
        Reg::UnalignReg u0;
        Reg::UnalignReg u1;

        for (uint16_t i = 0; i < (uint16_t)blockCount; i++) {
            uint32_t blockCountInputOffest = i * perLoopInputPadElem + lastOneDimBegin;
            uint32_t blockCountOutputOffest = i * perLoopOutputPadElem;
            __local_mem__ T* srcPtr1 = srcPtr + blockCountInputOffest;
            __local_mem__ T* dstPtr1 = dstPtr + blockCountOutputOffest;
            Reg::DataCopyUnAlignPre(u0, srcPtr1);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(vd0, u0, srcPtr1, lastOneInputDim);
                Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(dstPtr1, vd0, u1, lastOneOutputDim);
            }
            Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(dstPtr1, u1, 0);
        }
    }
    inQue_.FreeTensor(inTensor);
    outQue_.EnQue(outTensor);
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceMoveAlignDataCopyUnalign<T, U, V>::ProcessPerBlock(
    const DataCopyExtParams& copyOutParamsMain, const DataCopyExtParams& copyOutParamsTail)
{
    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;
    int64_t ubSplitLoopsNum = 0;  // ub切分轴上的循环次数
    int64_t ubOuterLoopsNum = 0;  // ub切分轴之外的循环次数
    int64_t rowsOffsetOutput = 0; // 当前核处理的output shape中的起始行数

    this->CalcProcessLoopsNum(ubOuterLoopsNum, ubSplitLoopsNum, blockIdx_);
    this->GetProcessRowsOffset(rowsOffsetOutput, blockIdx_);
    for (int64_t idx = 0; idx < ubOuterLoopsNum; idx++) {
        inputGmAddr = this->GetInputGmAddrWithoutLastDim(rowsOffsetOutput +
                                                         idx * this->rowsOffsetSteps_[this->ubIndex_]);
        outputGmAddr = this->GetOutputGmAddr(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        for (int64_t loops = 0; loops < ubSplitLoopsNum; loops++) {
            LocalTensor<T> inputLocal = inQue_.AllocTensor<T>();
            SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(inputLocal, inputGM_[inputGmAddr + loops * this->ubInLoopSteps_], copyInParam_, padParams_);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            inQue_.EnQue(inputLocal);

            // 模版准入条件已限制，blockCount < 65536
            DataCopyUnalignGather(copyOutParamsMain.blockCount);
            auto outTensor = outQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + loops * this->ubOutLoopSteps_], outTensor, copyOutParamsMain);
            outQue_.FreeTensor(outTensor);
        }
        if (this->ubTailFactor_ > 0) {
            LocalTensor<T> inputLocal = inQue_.AllocTensor<T>();
            SetLoopModePara(loopModeTail_, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(inputLocal, inputGM_[inputGmAddr + ubSplitLoopsNum * this->ubInLoopSteps_], copyInParamTail_,
                        padParams_);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            inQue_.EnQue(inputLocal);

            DataCopyUnalignGather(copyOutParamsTail.blockCount);
            auto outTensor = outQue_.DeQue<T>();
            DataCopyPad(outputGM_[outputGmAddr + ubSplitLoopsNum * this->ubOutLoopSteps_], outTensor,
                        copyOutParamsTail);
            outQue_.FreeTensor(outTensor);
        }
    }
}

} // namespace Slice

#endif // SLICE_MOVE_ALIGN_UNALIGN_DATCOPY_H
