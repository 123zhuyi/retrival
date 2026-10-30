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
 * \file strided_slice_base.h
 * \brief
 */
#ifndef STRIDED_SLICE_BASE_H
#define STRIDED_SLICE_BASE_H

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"

namespace StridedSlice {
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
constexpr int64_t UNCONST_BEGIN_VALUE = -10;

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

template <typename T, typename U>
class StridedSliceBase {
protected:
    using RT = std::conditional_t<sizeof(T) != sizeof(uint64_t), T, uint32_t>;

public:
    __aicore__ inline StridedSliceBase(){};
    __aicore__ inline void ParseBaseTilingData(GM_ADDR begin, const StridedSliceTilingData* tilingData,
                                               int64_t blockIdx);
    __aicore__ inline void ParseBaseTilingDataV2(GM_ADDR begin, const StridedSliceBaseTilingData* tilingData,
                                                 int64_t blockIdx);
    __aicore__ inline void ParseNDDMATilingData(GM_ADDR begin, const StridedSliceNDDMATilingData* tilingData,
                                                int64_t blockIdx);
    __aicore__ inline int64_t GetInputGmAddr(int64_t rowIdx) const;
    __aicore__ inline int64_t GetInputGmAddrAll(int64_t rowIdx) const;
    __aicore__ inline int64_t GetOutputGmAddr(int64_t rowIdx) const;
    __aicore__ inline int64_t GetOutputGmAddrAll(int64_t rowIdx) const;
    __aicore__ inline void GetProcessRowsOffset(int64_t& rowsOffset, int64_t blockIdx) const;
    __aicore__ inline void GetProcessRowsOffsetAll(int64_t& rowsOffset, int64_t blockIdx) const;
    __aicore__ inline void CalcProcessLoopsNum(int64_t& curCoreLoopsNum, int64_t& ubSplitLoopNum, int64_t blockIdx);
    __aicore__ inline void GetHandleRowsNum(int64_t& rowsNum, int64_t blockIdx) const;
    __aicore__ inline void GetLastDimSplitLoopCnt(int64_t& loopCnt, int64_t blockIdx);
    __aicore__ inline void CalcReorderAxisInfo(uint32_t axis0, uint32_t axis1, uint32_t axis2, uint32_t axis3BA);
    __aicore__ inline void CalcReorderAxisInfoDimsThree(uint32_t axis1, uint32_t axis2, uint32_t axis3BA);
    __aicore__ inline void CalcReorderAxisInfoDimsFour(uint32_t axis0, uint32_t axis1, uint32_t axis2,
                                                       uint32_t axis3BA);
    __aicore__ inline void Reorder4Output(__local_mem__ T* outAddr);
    __aicore__ inline void Reorder4OutputPhase1(__local_mem__ RT* reOutAddr, uint32_t rVLCnt);
    __aicore__ inline void Reorder4OutputPhase2(__local_mem__ RT* reOutAddr, uint32_t rVLCnt);

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
    int64_t strides_[STRIDED_SLICE_MAX_AXIS_NUM] = {0};

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
    bool isEnableB64_ = false;

    // for negative strides
    uint32_t uAxis0_ = 1;
    uint32_t uAxis1_ = 1;
    uint32_t uAxis2_ = 1;
    uint32_t uAxis0Offset_ = 0;
    uint32_t uAxis1Offset_ = 0;
    uint32_t uAxis2Offset_ = 0;
    uint32_t vlCnt_ = Ops::Base::GetVRegSize() / sizeof(T);
};

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::ParseBaseTilingData(GM_ADDR begin,
                                                                   const StridedSliceTilingData* tilingData,
                                                                   int64_t blockIdx)
{
    ubSize_ = tilingData->ubSize;
    realCoreNum_ = tilingData->realCoreNum;

    inputDims_ = tilingData->inputDims;
    for (int32_t i = 0; i < inputDims_; i++) {
        outputShape_[i] = tilingData->outputShape[i];
        strides_[i] = tilingData->strides[i];
        rowsOffsetSteps_[i] = tilingData->rowsOffsetSteps[i];
        inputSteps_[i] = tilingData->inputSteps[i];
        if (tilingData->begin[i] >= 0) {
            begin_[i] = tilingData->begin[i];
        } else {
            int64_t realIdx = UNCONST_BEGIN_VALUE - tilingData->begin[i];
            begin_[i] = static_cast<int64_t>(((__gm__ U*)begin)[realIdx]);
            int64_t curInShape = (i == inputDims_ - 1) ? tilingData->inputSteps[i] :
                                                         (tilingData->inputSteps[i] / tilingData->inputSteps[i + 1]);
            begin_[i] = begin_[i] < 0 ? (begin_[i] + curInShape) : begin_[i];
            if (begin_[i] < 0) {
                begin_[i] = 0;
            } else if (begin_[i] >= curInShape) {
                begin_[i] = curInShape - 1;
            }
        }
    }

    for (int32_t i = 0; i < STRIDED_SLICE_MAX_AXIS_NUM; i++) {
        nddmaLoopSize_[i] = tilingData->nddmaLoopSize[i];
        nddmaLoopSrcStride_[i] = tilingData->nddmaLoopSrcStride[i];
        nddmaLoopDstStride_[i] = tilingData->nddmaLoopDstStride[i];
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

    ubOutLoopSteps_ = tilingData->ubOutLoopSteps;
    ubInLoopSteps_ = tilingData->ubInLoopSteps;
    nddmaTotalNum_ = tilingData->nddmaTotalNum;
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::ParseBaseTilingDataV2(GM_ADDR begin,
                                                                     const StridedSliceBaseTilingData* tilingData,
                                                                     int64_t blockIdx)
{
    ubSize_ = tilingData->ubSize;
    realCoreNum_ = tilingData->realCoreNum;

    inputDims_ = tilingData->inputDims;
    for (int32_t i = 0; i < inputDims_; i++) {
        outputShape_[i] = tilingData->outputShape[i];
        strides_[i] = tilingData->strides[i];
        rowsOffsetSteps_[i] = tilingData->rowsOffsetSteps[i];
        inputSteps_[i] = tilingData->inputSteps[i];
        if (tilingData->begin[i] >= 0) {
            begin_[i] = tilingData->begin[i];
        } else {
            int64_t realIdx = UNCONST_BEGIN_VALUE - tilingData->begin[i];
            begin_[i] = static_cast<int64_t>(((__gm__ U*)begin)[realIdx]);
            int64_t curInShape = (i == inputDims_ - 1) ? tilingData->inputSteps[i] :
                                                         (tilingData->inputSteps[i] / tilingData->inputSteps[i + 1]);
            begin_[i] = begin_[i] < 0 ? (begin_[i] + curInShape) : begin_[i];
            if (begin_[i] < 0) {
                begin_[i] = 0;
            } else if (begin_[i] >= curInShape) {
                begin_[i] = curInShape - 1;
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
    ubOutLoopSteps_ = tilingData->ubOutLoopSteps;
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::ParseNDDMATilingData(GM_ADDR begin,
                                                                    const StridedSliceNDDMATilingData* tilingData,
                                                                    int64_t blockIdx)
{
    ParseBaseTilingDataV2(begin, &(tilingData->stridedSliceBaseTilingData), blockIdx);
    for (int32_t i = 0; i < STRIDED_SLICE_MAX_AXIS_NUM; i++) {
        nddmaLoopSize_[i] = tilingData->nddmaLoopSize[i];
        nddmaLoopSrcStride_[i] = tilingData->nddmaLoopSrcStride[i];
        nddmaLoopDstStride_[i] = tilingData->nddmaLoopDstStride[i];
    }
    nddmaTotalNum_ = tilingData->nddmaTotalNum;
}

/*
  从最内轴向最外轴扩展
  根据每一维度的begin和stride计算第rowIdx行相对于inputGm中的偏移地址
*/
template <typename T, typename U>
__aicore__ inline int64_t StridedSliceBase<T, U>::GetInputGmAddr(int64_t rowIdx) const
{
    int64_t inputGmAddr = begin_[inputDims_ - 1]; // 最内轴的begin
    int64_t curDim = 0;
    int64_t tmpBegin = 0;
    for (int64_t i = inputDims_ - 1; i > 0; i--) {
        curDim = outputShape_[i - 1];
        tmpBegin = begin_[i - 1] + rowIdx % curDim * strides_[i - 1];
        inputGmAddr = inputGmAddr + inputSteps_[i] * tmpBegin;
        rowIdx = rowIdx / curDim;
    }
    return inputGmAddr;
}

template <typename T, typename U>
__aicore__ inline int64_t StridedSliceBase<T, U>::GetInputGmAddrAll(int64_t rowIdx) const
{
    int64_t inputGmAddr = 0;
    int64_t curDim = 0;
    int64_t tmpBegin = 0;
    if (blkIndex_ != inputDims_ - 1) {
        rowIdx *= outputShape_[inputDims_ - 1];
    }
    inputGmAddr = begin_[inputDims_ - 1] + rowIdx % outputShape_[inputDims_ - 1] * strides_[inputDims_ - 1];
    rowIdx /= outputShape_[inputDims_ - 1];
    // from right to left
    for (int64_t i = inputDims_ - 1; i > 0; i--) {
        curDim = outputShape_[i - 1];
        tmpBegin = begin_[i - 1] + rowIdx % curDim * strides_[i - 1];
        inputGmAddr = inputGmAddr + inputSteps_[i] * tmpBegin;
        rowIdx = rowIdx / curDim;
    }
    return inputGmAddr;
}

template <typename T, typename U>
__aicore__ inline int64_t StridedSliceBase<T, U>::GetOutputGmAddr(int64_t rowIdx) const
{
    return rowIdx * outputShape_[inputDims_ - 1];
}

template <typename T, typename U>
__aicore__ inline int64_t StridedSliceBase<T, U>::GetOutputGmAddrAll(int64_t rowIdx) const
{
    int64_t res = rowIdx;
    if (blkIndex_ != inputDims_ - 1) {
        res = rowIdx * outputShape_[inputDims_ - 1];
    }
    return res;
}

// 计算当前核处理的行的起始行号，即output中的第几行
template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::GetProcessRowsOffset(int64_t& rowsOffset, int64_t blockIdx) const
{
    // output: (6, 5, 4, 3)
    // blk切5，blkIndex=1, blkFactor=2;  blkSplitOutNum=ceil(5/2)=3
    // 包含完整切分轴个数对应的起始行位置
    rowsOffset = blockIdx / blkSplitOutNum_ * rowsOffsetSteps_[blkIndex_];
    // 当前核对应的切分轴内的起始行位置
    rowsOffset = rowsOffset + blockIdx % blkSplitOutNum_ * blkFactor_ * rowsOffsetSteps_[blkIndex_ + 1];
    return;
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::GetProcessRowsOffsetAll(int64_t& rowsOffset, int64_t blockIdx) const
{
    // 包含完整切分轴个数对应的起始行位置
    rowsOffset = blockIdx / blkSplitOutNum_ * rowsOffsetSteps_[blkIndex_];
    // 当前核对应的切分轴内的起始行位置
    if (blkIndex_ != inputDims_ - 1) {
        rowsOffset = rowsOffset + blockIdx % blkSplitOutNum_ * blkFactor_ * rowsOffsetSteps_[blkIndex_ + 1];
    } else {
        rowsOffset = rowsOffset * outputShape_[inputDims_ - 1] + blockIdx % blkSplitOutNum_ * blkFactor_;
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::CalcProcessLoopsNum(int64_t& curCoreLoopsNum, int64_t& ubSplitLoopNum,
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
template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::GetHandleRowsNum(int64_t& rowsNum, int64_t blockIdx) const
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
template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::GetLastDimSplitLoopCnt(int64_t& loopCnt, int64_t blockIdx)
{
    if ((blockIdx % blkSplitOutNum_ == blkSplitOutNum_ - 1) && (blkTailFactor_ != 0)) {
        // blk/ub all split last axis
        loopCnt = blkTailFactor_ / ubFactor_;
        return;
    }
    loopCnt = blkFactor_ / ubFactor_;
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::CalcReorderAxisInfoDimsThree(uint32_t axis1, uint32_t axis2,
                                                                            uint32_t axis3BA)
{
    auto lastTwoStride = strides_[inputDims_ - DIMS_2];
    auto lastThreeStride = strides_[inputDims_ - DIMS_3];
    if (lastTwoStride < 0) {
        if (lastThreeStride < 0) {
            uAxis2_ = axis1 * axis2;
            uAxis2Offset_ = axis3BA;
        } else if (lastThreeStride > 0) {
            uAxis2_ = axis2;
            uAxis2Offset_ = axis3BA;
            uAxis1_ = axis1;
            uAxis1Offset_ = axis2 * axis3BA;
        }
    } else if (lastThreeStride < 0) {
        uAxis2_ = axis1;
        uAxis2Offset_ = axis2 * axis3BA;
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::CalcReorderAxisInfoDimsFour(uint32_t axis0, uint32_t axis1,
                                                                           uint32_t axis2, uint32_t axis3BA)
{
    auto lastTwoStride = strides_[inputDims_ - DIMS_2];
    auto lastThreeStride = strides_[inputDims_ - DIMS_3];
    auto lastFourStride = strides_[inputDims_ - DIMS_4];
    if (lastTwoStride > 0) {
        if (lastThreeStride < 0 && lastFourStride < 0) {
            uAxis2_ = axis0 * axis1;
            uAxis2Offset_ = axis2 * axis3BA;
        } else if (lastThreeStride > 0 && lastFourStride < 0) {
            uAxis2_ = axis0;
            uAxis2Offset_ = axis1 * axis2 * axis3BA;
        } else if (lastThreeStride < 0 && lastFourStride > 0) {
            uAxis2_ = axis1;
            uAxis2Offset_ = axis2 * axis3BA;
            uAxis1_ = axis0;
            uAxis1Offset_ = axis1 * axis2 * axis3BA;
        }
    } else {
        if (lastThreeStride < 0 && lastFourStride < 0) {
            uAxis2_ = axis0 * axis1 * axis2;
            uAxis2Offset_ = axis3BA;
        } else if (lastThreeStride < 0 && lastFourStride > 0) {
            uAxis2_ = axis1 * axis2;
            uAxis2Offset_ = axis3BA;
            uAxis1_ = axis0;
            uAxis1Offset_ = axis1 * axis2 * axis3BA;
        } else if (lastThreeStride > 0 && lastFourStride > 0) {
            uAxis2_ = axis2;
            uAxis2Offset_ = axis3BA;
            uAxis1_ = axis0 * axis1;
            uAxis1Offset_ = axis2 * axis3BA;
        } else if (lastThreeStride > 0 && lastFourStride < 0) {
            uAxis2_ = axis2;
            uAxis2Offset_ = axis3BA;
            // phase1 should inclue axis0
            uAxis1_ = axis0 * axis1;
            uAxis1Offset_ = axis2 * axis3BA;
            uAxis0_ = axis0;
            uAxis0Offset_ = axis1 * axis2 * axis3BA;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::CalcReorderAxisInfo(uint32_t axis0, uint32_t axis1, uint32_t axis2,
                                                                   uint32_t axis3BA)
{
    int64_t inUbDims_ = inputDims_ - ubIndex_;
    if (inUbDims_ == 1L) {
        return;
    }

    // N
    if (inUbDims_ == DIMS_2 && strides_[inputDims_ - DIMS_2] < 0) {
        uAxis2_ = axis2;
        uAxis2Offset_ = axis3BA;
        return;
    }
    // P N
    if (inUbDims_ == DIMS_3) {
        CalcReorderAxisInfoDimsThree(axis1, axis2, axis3BA);
        return;
    }
    // N P N
    if (inUbDims_ == DIMS_4) {
        CalcReorderAxisInfoDimsFour(axis0, axis1, axis2, axis3BA);
        return;
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::Reorder4OutputPhase1(__local_mem__ RT* reOutAddr, uint32_t rVLCnt)
{
    if (uAxis2_ <= 1) {
        return;
    }

    // P N
    uint16_t halfAxis2 = uAxis2_ / NUM_TWO;
    uint16_t axis3LpCnt = Ops::Base::CeilDiv(uAxis2Offset_, vlCnt_);
    uint32_t axis1Offset = uAxis1Offset_;
    uint32_t axis2Offset = uAxis2Offset_;
    if (sizeof(T) != sizeof(RT)) {
        axis1Offset *= NUM_TWO;
        axis2Offset *= NUM_TWO;
    }
    uint32_t maskValue = axis2Offset;
    uint32_t axis2EndBase = (uAxis2_ - 1) * axis2Offset;
    uint32_t axis2RBOffset = NUM_TWO * axis2Offset;
    uint16_t uAxis1 = uint16_t(uAxis1_);

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<RT> regData0;
        Reg::RegTensor<RT> regData1;
        Reg::RegTensor<RT> regTmp;
        for (uint16_t axis3LpIdx = 0; axis3LpIdx < axis3LpCnt; axis3LpIdx++) {
            mask = Reg::UpdateMask<RT>(maskValue);
            auto reOutAddr1 = reOutAddr + axis2EndBase;
            for (uint16_t axis2Idx = 0; axis2Idx < halfAxis2; axis2Idx++) {
                for (uint16_t axis1Idx = 0; axis1Idx < uAxis1; axis1Idx++) {
                    auto areg = Reg::CreateAddrReg<RT>(axis3LpIdx, rVLCnt, axis2Idx, axis2Offset, axis1Idx,
                                                       axis1Offset);
                    Reg::DataCopy(regData0, reOutAddr, areg);
                    Reg::DataCopy(regData1, reOutAddr1, areg);
                    Reg::Copy(regTmp, regData0);
                    Reg::Copy(regData0, regData1);
                    Reg::Copy(regData1, regTmp);
                    Reg::DataCopy(reOutAddr, regData0, areg, mask);
                    Reg::DataCopy(reOutAddr1, regData1, areg, mask);
                }
                reOutAddr1 -= axis2RBOffset;
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::Reorder4OutputPhase2(__local_mem__ RT* reOutAddr, uint32_t rVLCnt)
{
    if (uAxis0_ <= 1) {
        return;
    }

    // P
    uint16_t axis1LpCnt = Ops::Base::CeilDiv(uAxis0Offset_, vlCnt_);
    uint16_t halfAxis0 = uAxis0_ / NUM_TWO;
    uint32_t axis0Offset = uAxis0Offset_;
    if (sizeof(T) != sizeof(RT)) {
        axis0Offset *= NUM_TWO;
    }
    uint32_t axis0EndBase = (uAxis0_ - 1) * axis0Offset;
    auto reOutAddr1 = reOutAddr + axis0EndBase;
    uint32_t axis0RBOffset = NUM_TWO * axis0Offset;

    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<RT> regData0;
        Reg::RegTensor<RT> regData1;
        Reg::RegTensor<RT> regTmp;
        for (uint16_t axis0Idx = 0; axis0Idx < halfAxis0; axis0Idx++) {
            uint32_t maskValue = axis0Offset;
            for (uint16_t axis1LpIdx = 0; axis1LpIdx < axis1LpCnt; axis1LpIdx++) {
                mask = Reg::UpdateMask<RT>(maskValue);
                auto areg = Reg::CreateAddrReg<RT>(axis0Idx, axis0Offset, axis1LpIdx, rVLCnt);
                Reg::DataCopy(regData0, reOutAddr, areg);
                Reg::DataCopy(regData1, reOutAddr1, areg);
                Reg::Copy(regTmp, regData0);
                Reg::Copy(regData0, regData1);
                Reg::Copy(regData1, regTmp);
                Reg::DataCopy(reOutAddr, regData0, areg, mask);
                Reg::DataCopy(reOutAddr1, regData1, areg, mask);
            }
            reOutAddr1 -= axis0RBOffset;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceBase<T, U>::Reorder4Output(__local_mem__ T* outAddr)
{
    if (uAxis0_ * uAxis1_ * uAxis2_ == 1U) {
        return;
    }

    uint32_t rVLCnt = (sizeof(T) != sizeof(RT)) ? NUM_TWO * vlCnt_ : vlCnt_;
    auto reOutAddr = reinterpret_cast<__local_mem__ RT*>(outAddr);
    Reorder4OutputPhase1(reOutAddr, rVLCnt);
    Reorder4OutputPhase2(reOutAddr, rVLCnt);
}

} // namespace StridedSlice

#endif // STRIDED_SLICE_BASE_H
