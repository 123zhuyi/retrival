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
 * \file strided_slice_nddma_negative_strides_ub2ub.h
 * \brief negative strides implement of op strided_slice
 */
#ifndef STRIDED_SLICE_NDDMA_NEG_STRIDES_UB2UB_H
#define STRIDED_SLICE_NDDMA_NEG_STRIDES_UB2UB_H

#include <cstdlib>
#include <type_traits>

#include "strided_slice_base.h"

namespace StridedSlice
{
using namespace AscendC;

template <typename T, typename U>
class StridedSliceNDDMAUb2Ub : public StridedSliceBase<T, U>
{
public:
    __aicore__ inline StridedSliceNDDMAUb2Ub(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y,
        const StridedSliceNDDMATilingData *tdPtr, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo,
                                       MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfoTail);
    __aicore__ inline void SetCopyOutParams(DataCopyExtParams &copyOutParams,
                                            const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfo);
    __aicore__ inline void ProcessPerBlock();
    __aicore__ inline void ProcessWithDataCopyGather(int64_t inGmAddr, int64_t outGmAddr,
        const MultiCopyParams<T, NDDMA_MAX_DIMS_NEG> &nddmaParams, const DataCopyExtParams &copyOutParam);
    __aicore__ inline void ReorderSlice(const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfo, __local_mem__ T *outAddr);
    __aicore__ inline int64_t CalcRollbackOffset(const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfo);

private:
    TPipe *pipe_ = nullptr;
    const StridedSliceNDDMATilingData *tdPtr_ = nullptr;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    static constexpr MultiCopyConfig nddmaConfig_ = {false, 0, 0, false};

    int64_t blockIdx_ = 0;
    uint8_t bufferCnt_ = 2; // enable db
    uint32_t elemPerBlock_ = BLOCK_SIZE_BYTE / sizeof(T);
    int64_t inUbDims_ = 0;
    int64_t lastDimStride_ = 1;
};

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAUb2Ub<T, U>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y,
    const StridedSliceNDDMATilingData *tdPtr, TPipe *pipe)
{
    blockIdx_ = GetBlockIdx();

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    tdPtr_ = tdPtr;
    this->ParseNDDMATilingData(begin, tdPtr_, blockIdx_);
    inUbDims_ = this->inputDims_ - this->ubIndex_;
    lastDimStride_ = this->strides_[this->inputDims_ - 1];

    // pipeBuffer
    pipe_ = pipe;
    pipe_->InitBuffer(inOutQue_, bufferCnt_, tdPtr_->ubSizeInput);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAUb2Ub<T, U>::Process()
{
    // for empty tensor set realCoreNum as 0, do nothing
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    ProcessPerBlock();
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAUb2Ub<T, U>::ReorderSlice(const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfo,
                                                               __local_mem__ T *outAddr)
{
    // (axis0, axis1, axis2, axis3BA)
    uint32_t axis0 = loopInfo.loopSize[DIMS_3];
    uint32_t axis1 = loopInfo.loopSize[DIMS_2];
    uint32_t axis2 = loopInfo.loopSize[DIMS_1];
    uint32_t axis3 = loopInfo.loopSize[0];
    uint32_t axis3BA = Ops::Base::CeilAlign(axis3, elemPerBlock_);

    this->CalcReorderAxisInfo(axis0, axis1, axis2, axis3BA);
    this->Reorder4Output(outAddr);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAUb2Ub<T, U>::ProcessWithDataCopyGather(
    int64_t inGmAddr, int64_t outGmAddr, const MultiCopyParams<T, NDDMA_MAX_DIMS_NEG> &nddmaParams,
    const DataCopyExtParams &copyOutParam)
{
    auto inTensor = inOutQue_.AllocTensor<T>();
    DataCopy<T, NDDMA_MAX_DIMS_NEG, nddmaConfig_>(inTensor, inputGM_[inGmAddr], nddmaParams);
    inOutQue_.EnQue(inTensor);
    inTensor = inOutQue_.DeQue<T>();

    event_t eventID0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID0);
    WaitFlag<HardEvent::MTE2_V>(eventID0);

    ReorderSlice(nddmaParams.loopInfo, (__local_mem__ T *)inTensor.GetPhyAddr());

    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID1);
    WaitFlag<HardEvent::V_MTE3>(eventID1);

    // inTensor = inOutQue_.DeQue<T>();
    DataCopyPad(outputGM_[outGmAddr], inTensor, copyOutParam);
    inOutQue_.FreeTensor(inTensor);
}

template <typename T, typename U>
__aicore__ inline int64_t StridedSliceNDDMAUb2Ub<T, U>::CalcRollbackOffset(const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfo)
{
    int64_t backOffset = 0;

    if (lastDimStride_ < 0) {
        backOffset += (loopInfo.loopSize[0] - 1) * loopInfo.loopSrcStride[0];
    }
    if (inUbDims_ > DIMS_1 && this->strides_[this->inputDims_ - DIMS_2] < 0) {
        backOffset += (loopInfo.loopSize[DIMS_1] - 1) * loopInfo.loopSrcStride[DIMS_1];
    }
    if (inUbDims_ > DIMS_2 && this->strides_[this->inputDims_ - DIMS_3] < 0) {
        backOffset += (loopInfo.loopSize[DIMS_2] - 1) * loopInfo.loopSrcStride[DIMS_2];
    }
    if (inUbDims_ > DIMS_3 && this->strides_[this->inputDims_ - DIMS_4] < 0) {
        backOffset += (loopInfo.loopSize[DIMS_3] - 1) * loopInfo.loopSrcStride[DIMS_3];
    }

    return backOffset;
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAUb2Ub<T, U>::SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfo,
                                                         MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfoTail)
{
    for (int64_t i = 0; i < inUbDims_; i++) {
        // ub main
        loopInfo.loopSize[i] = this->nddmaLoopSize_[NDDMA_MAX_DIMS_NEG - 1 - i];
        loopInfo.loopSrcStride[i] = this->nddmaLoopSrcStride_[NDDMA_MAX_DIMS_NEG - 1 - i];
        loopInfo.loopDstStride[i] = this->nddmaLoopDstStride_[NDDMA_MAX_DIMS_NEG - 1 - i];
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;

        loopInfoTail.loopSize[i] = this->nddmaLoopSize_[NDDMA_MAX_DIMS_NEG - 1 - i];
        loopInfoTail.loopSrcStride[i] = this->nddmaLoopSrcStride_[NDDMA_MAX_DIMS_NEG - 1 - i];
        loopInfoTail.loopDstStride[i] = this->nddmaLoopDstStride_[NDDMA_MAX_DIMS_NEG - 1 - i];
        loopInfoTail.loopLpSize[i] = 0;
        loopInfoTail.loopRpSize[i] = 0;
    }

    loopInfoTail.loopSize[inUbDims_ - 1] = this->ubTailFactor_;

    // 补维
    for (int64_t i = inUbDims_; i < NDDMA_MAX_DIMS_NEG; i++) {
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

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAUb2Ub<T, U>::SetCopyOutParams(DataCopyExtParams &copyOutParams,
                                                              const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG> &loopInfo)
{
    if (loopInfo.loopSize[0] % elemPerBlock_ == 0) {
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = loopInfo.loopSize[0] * loopInfo.loopSize[1] * loopInfo.loopSize[2] *
                                     loopInfo.loopSize[3] * sizeof(T);
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
    } else {
        copyOutParams.blockCount = loopInfo.loopSize[1] * loopInfo.loopSize[2] * loopInfo.loopSize[3];
        copyOutParams.blockLen = loopInfo.loopSize[0] * sizeof(T);
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAUb2Ub<T, U>::ProcessPerBlock()
{
    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;

    MultiCopyParams<T, NDDMA_MAX_DIMS_NEG> paramsMain;
    MultiCopyParams<T, NDDMA_MAX_DIMS_NEG> paramsTail;
    paramsMain.constantValue = 0;
    paramsTail.constantValue = 0;
    SetLoopInfo(paramsMain.loopInfo, paramsTail.loopInfo);

    DataCopyExtParams copyOutParamsMain;
    DataCopyExtParams copyOutParamsTail;
    SetCopyOutParams(copyOutParamsMain, paramsMain.loopInfo);
    SetCopyOutParams(copyOutParamsTail, paramsTail.loopInfo);

    // 计算中间变量
    int64_t ubSplitLoopsNum = 0; // ub切分轴上的循环次数
    int64_t ubOuterLoopsNum = 0; // ub切分轴之外的循环次数
    int64_t rowsOffsetOutput = 0; // 当前核处理的output shape中的起始行数
    this->CalcProcessLoopsNum(ubOuterLoopsNum, ubSplitLoopsNum, blockIdx_);
    this->GetProcessRowsOffsetAll(rowsOffsetOutput, blockIdx_);

    int64_t negativeStrideOffset = CalcRollbackOffset(paramsMain.loopInfo);
    int64_t negativeStrideOffsetT = CalcRollbackOffset(paramsTail.loopInfo);

    for (int64_t idx = 0; idx < ubOuterLoopsNum; idx++) {
        inputGmAddr = this->GetInputGmAddrAll(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        outputGmAddr = this->GetOutputGmAddrAll(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        for (int64_t loops = 0; loops < ubSplitLoopsNum; loops++) {
            ProcessWithDataCopyGather(inputGmAddr + loops * this->ubInLoopSteps_ - negativeStrideOffset,
                                      outputGmAddr + loops * this->ubOutLoopSteps_, paramsMain,
                                      copyOutParamsMain);
        }
        if (this->ubTailFactor_ > 0) {
            ProcessWithDataCopyGather(inputGmAddr + ubSplitLoopsNum * this->ubInLoopSteps_ - negativeStrideOffsetT,
                                      outputGmAddr + ubSplitLoopsNum * this->ubOutLoopSteps_, paramsTail,
                                      copyOutParamsTail);
        }
    }
}

} // namespace StridedSlice

#endif // STRIDED_SLICE_NDDMA_NEG_STRIDES_UB2UB_H