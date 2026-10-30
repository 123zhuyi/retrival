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
 * \file slice_base.h
 * \brief
 */
#ifndef SLICE_BASE_H
#define SLICE_BASE_H

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "slice_struct.h"

namespace Slice
{
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr int32_t THREAD_DIM = 512;
constexpr int32_t HALF_THREAD_DIM = 512;
constexpr int32_t QUARTER_THREAD_DIM = 512;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256;
#else
constexpr int32_t THREAD_DIM = 2048;
constexpr int32_t HALF_THREAD_DIM = 1024;
constexpr int32_t QUARTER_THREAD_DIM = 512;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256;
#endif

constexpr int64_t STRIDED_SLICE_MAX_AXIS_NUM = 8;
constexpr int64_t MOVE_ALIGN_V2_MAX_DIMS = 4;
constexpr int64_t NDDMA_MAX_DIMS = 5;
constexpr int64_t NDDMA_MAX_DIMS_NEG = 4;
constexpr int64_t NDDMA_LAST_DIMS = 1;
constexpr int64_t BLOCK_SIZE_BYTE = Ops::Base::GetUbBlockSize();
constexpr int32_t BIT64_SIZE = 8;
constexpr uint16_t DIMS_1 = 1;
constexpr uint16_t DIMS_2 = 2;
constexpr uint16_t DIMS_3 = 3;
constexpr uint16_t DIMS_4 = 4;
constexpr uint16_t DIMS_5 = 5;
constexpr uint16_t DIMS_6 = 6;
constexpr uint16_t DIMS_7 = 7;
constexpr uint16_t DIMS_8 = 8;
constexpr int32_t BIT64 = 64;
constexpr int32_t BIT32 = 32;

constexpr uint16_t DIM0_INDEX = 0;
constexpr uint16_t DIM1_INDEX = 1;
constexpr uint16_t DIM2_INDEX = 2;
constexpr uint16_t DIM3_INDEX = 3;
constexpr uint16_t DIM4_INDEX = 4;
constexpr uint16_t DIM5_INDEX = 5;
constexpr uint16_t DIM6_INDEX = 6;
constexpr uint16_t DIM7_INDEX = 7;

constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t VL_SIZE_BYTE = Ops::Base::GetVRegSize();

template <typename T, typename U, typename V = int8_t>
class SliceBase
{
public:
    __aicore__ inline SliceBase(){};
    __aicore__ inline void ParseBaseTilingData(GM_ADDR begin, const SliceBaseTilingData *tilingData, int64_t blockIdx);
    __aicore__ inline int64_t GetInputGmAddr(int64_t rowIdx) const;
    __aicore__ inline int64_t GetInputGmAddrWithoutLastDim(int64_t rowIdx) const;
    __aicore__ inline int64_t GetOutputGmAddr(int64_t rowIdx) const;
    __aicore__ inline void GetProcessRowsOffset(int64_t &rowsOffset, int64_t blockIdx) const;
    __aicore__ inline void CalcProcessLoopsNum(int64_t &curCoreLoopsNum, int64_t &ubSplitLoopNum, int64_t blockIdx);
    __aicore__ inline void GetHandleRowsNum(int64_t &rowsNum, int64_t blockIdx) const;
    __aicore__ inline void GetLastDimSplitLoopCnt(int64_t &loopCnt, int64_t blockIdx);

protected:
    template <typename T1>
    __aicore__ inline T1 CeilDiv(T1 a, T1 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };

    template <typename T1>
    __aicore__ inline T1 CeilAlign(T1 a, T1 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b * b;
    };

protected:
    // Tiling Data
    int64_t ubSize_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t inputDims_ = 0;
    int64_t outputShape_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};
    int64_t begin_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};

    int64_t rowsOffsetSteps_[STRIDED_SLICE_MAX_AXIS_NUM] = {0}; // output每个维度包含的行数
    int64_t inputSteps_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};      // output每个维度跨度(个数)

    int64_t ubIndex_ = 0;
    int64_t ubFactor_ = 0;
    int64_t ubTailFactor_ = 0;

    int64_t blkIndex_ = 0;
    int64_t blkFactor_ = 0;
    int64_t blkTailFactor_ = 0;

    int64_t ubOutLoopSteps_ = 0;
    int64_t ubInLoopSteps_ = 0;
    int64_t nddmaTotalNum_ = 0;
    int64_t nddmaLoopSize_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};
    int64_t nddmaLoopSrcStride_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};
    int64_t nddmaLoopDstStride_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};

    int64_t blkSplitOutNum_ = 0; // 切分轴用来开多核的个数
};

template <typename T, typename U, typename V>
__aicore__ inline void SliceBase<T, U, V>::ParseBaseTilingData(GM_ADDR begin, const SliceBaseTilingData *tilingData,
                                                                int64_t blockIdx)
{
    ubSize_ = tilingData->ubSize;
    realCoreNum_ = tilingData->realCoreNum;

    inputDims_ = tilingData->inputDims;
    for (int32_t i = 0; i < inputDims_; i++) {
        outputShape_[i] = tilingData->outputShape[i];
        rowsOffsetSteps_[i] = tilingData->rowsOffsetSteps[i];
        inputSteps_[i] = tilingData->inputSteps[i];
        if (tilingData->isBeginConst) {
            begin_[i] = tilingData->begin[i];
        } else {
            begin_[i] = static_cast<int64_t>(((__gm__ U*)begin)[i]);
            if constexpr (IsSameType<V, fp4x2_e2m1_t>::value || IsSameType<V, fp4x2_e1m2_t>::value) {
                ascendc_assert(
                    !(begin_[inputDims_ - 1] & 1),
                    "When the input dtype is fp4, the last dimension of offset must be even.\n");
                begin_[inputDims_ - 1] /= 2;
            }
        }
    }

    blkIndex_ = tilingData->blkIndex;
    blkFactor_ = tilingData->blkFactor;
    blkTailFactor_ = tilingData->blkTailFactor;

    blkSplitOutNum_ = CeilDiv(outputShape_[blkIndex_], blkFactor_);

    ubIndex_ = tilingData->ubIndex;
    ubFactor_ = tilingData->ubFactor;
    ubTailFactor_ = tilingData->ubTailFactor;
    if ((blockIdx % blkSplitOutNum_ == blkSplitOutNum_ - 1) && (blkIndex_ == ubIndex_) && (blkTailFactor_ != 0)) {
        ubTailFactor_ = tilingData->ubTailTailFactor;
    }

    ubInLoopSteps_ = tilingData->ubInLoopSteps;
}


/*
  从最内轴向最外轴扩展
  根据每一维度的begin和stride计算第rowIdx行相对于inputGm中的偏移地址
*/
template <typename T, typename U, typename V>
__aicore__ inline int64_t SliceBase<T, U, V>::GetInputGmAddr(int64_t rowIdx) const
{
    int64_t inputGmAddr = begin_[inputDims_ - 1]; // 最内轴的begin
    int64_t curDim = 0;
    int64_t tmpBegin = 0;
    for (int64_t i = inputDims_ - 1; i > 0; i--) {
        curDim = outputShape_[i - 1];
        tmpBegin = begin_[i - 1] + rowIdx % curDim;
        inputGmAddr = inputGmAddr + inputSteps_[i] * tmpBegin;
        rowIdx = rowIdx / curDim;
    }
    return inputGmAddr;
}

template <typename T, typename U, typename V>
__aicore__ inline int64_t SliceBase<T, U, V>::GetInputGmAddrWithoutLastDim(int64_t rowIdx) const
{
    int64_t inputGmAddr = 0; // 最内轴的begin
    int64_t curDim = 0;
    int64_t tmpBegin = 0;
    for (int64_t i = inputDims_ - 1; i > 0; i--) {
        curDim = outputShape_[i - 1];
        tmpBegin = begin_[i - 1] + rowIdx % curDim;
        inputGmAddr = inputGmAddr + inputSteps_[i] * tmpBegin;
        rowIdx = rowIdx / curDim;
    }
    return inputGmAddr;
}

template <typename T, typename U, typename V>
__aicore__ inline int64_t SliceBase<T, U, V>::GetOutputGmAddr(int64_t rowIdx) const
{
    return rowIdx * outputShape_[inputDims_ - 1];
}

// 计算当前核处理的行的起始行号，即output中的第几行
template <typename T, typename U, typename V>
__aicore__ inline void SliceBase<T, U, V>::GetProcessRowsOffset(int64_t &rowsOffset, int64_t blockIdx) const
{
    // output: (6, 5, 4, 3)
    // blk切5，blkIndex=1, blkFactor=2;  blkSplitOutNum=ceil(5/2)=3
    // 包含完整切分轴个数对应的起始行位置
    rowsOffset = blockIdx / blkSplitOutNum_ * rowsOffsetSteps_[blkIndex_];
    // 当前核对应的切分轴内的起始行位置
    rowsOffset = rowsOffset + blockIdx % blkSplitOutNum_ * blkFactor_ * rowsOffsetSteps_[blkIndex_ + 1];
    return;
}

template <typename T, typename U, typename V>
__aicore__ inline void SliceBase<T, U, V>::CalcProcessLoopsNum(int64_t &curCoreLoopsNum, int64_t &ubSplitLoopNum,
                                                                int64_t blockIdx)
{
    if ((blockIdx % blkSplitOutNum_ == blkSplitOutNum_ - 1) && (blkTailFactor_ != 0)) {
        curCoreLoopsNum = blkTailFactor_;
    } else {
        curCoreLoopsNum = blkFactor_;
    }

    if (blkIndex_ == ubIndex_) {
        // blk和ub切同一根轴时，tiling保证 blk_factor >= ub_factor
        ubSplitLoopNum = curCoreLoopsNum / ubFactor_;
        // 切同一根轴时，外层循环次数肯定为1 !!!
        curCoreLoopsNum = 1;
        return;
    }

    for (int64_t i = blkIndex_ + 1; i < ubIndex_; i++) {
        curCoreLoopsNum = curCoreLoopsNum * outputShape_[i];
    }

    ubSplitLoopNum = outputShape_[ubIndex_] / ubFactor_;
    return;
}

// ub split last axis and blk split nlast axis
template <typename T, typename U, typename V>
__aicore__ inline void SliceBase<T, U, V>::GetHandleRowsNum(int64_t &rowsNum, int64_t blockIdx) const
{
    if ((blockIdx % blkSplitOutNum_ == blkSplitOutNum_ - 1) && (blkTailFactor_ != 0)) {
        rowsNum = blkTailFactor_;
    } else {
        rowsNum = blkFactor_;
    }

    for (int64_t i = blkIndex_ + 1; i < ubIndex_; i++) {
        rowsNum = rowsNum * outputShape_[i];
    }
}

// ub and blk all split last axis
template <typename T, typename U, typename V>
__aicore__ inline void SliceBase<T, U, V>::GetLastDimSplitLoopCnt(int64_t &loopCnt, int64_t blockIdx)
{
    if ((blockIdx % blkSplitOutNum_ == blkSplitOutNum_ - 1) && (blkTailFactor_ != 0)) {
        // blk/ub all split last axis
        loopCnt = blkTailFactor_ / ubFactor_;
        return;
    }
    loopCnt = blkFactor_ / ubFactor_;
}

} // namespace Slice

#endif // SLICE_BASE_H