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
 * \file one_axis_concat_all_align.h
 * \brief one axis concat all align
 */

#ifndef ONE_AXIS_CONCAT_ALL_ALIGN_
#define ONE_AXIS_CONCAT_ALL_ALIGN_

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename T, const bool SAMESHAPE, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatAllAlign {
public:
    __aicore__ inline OneAxisConcatAllAlign(const TILINGDATA& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t cols);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t rows, int64_t cols);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void CopyInSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t cols, SplitInfo& splitInfo);
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1PerLoop(
        int64_t loopIdxInCols, int64_t cols, int64_t loopSizeDim0, SplitInfo& splitInfo);
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1PerLoop(
        int64_t loopIdx, int64_t rows, int64_t cols, int64_t blockIdxInCol, SplitInfo& splitInfo);
    __aicore__ inline void CalcSplitInfo(int64_t endIdx, int64_t endOffset, int64_t length, SplitInfo& splitInfo);
    __aicore__ inline int64_t GetTensorDim1(int64_t idx);
    __aicore__ inline void UpdateSplitInfo();

private:
    TPipe& pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;
    const TILINGDATA& tilingData_;
    uint32_t blockIdx_{0};
    int64_t blockOffset_{0};
    int64_t numPerBlock_{BYTES_PER_BLOCK / sizeof(T)};
    uint32_t colsUsedCoreNum_{1};
    uint32_t rowsUsedCoreNum_{1};

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};

    int64_t endTensorIdx_{0};
    int64_t endTensorOffset_{0};
    int64_t startTensorIdx_{0};
    int64_t startTensorOffset_{0};
    ListTensorDesc inputList_;
    uint64_t buf_[ARRAY_SIZE];
    TensorDesc<T> desc_;
};

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        int64_t colsUsedCoreNum = GetBlockNum() / tilingData_.uoDim0;
        if (blockIdx_ % colsUsedCoreNum != 0) {
            startTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_ - 1];
            startTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx_ - 1];
        }
        endTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx_];
        endTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_];
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value) {
        blockOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + tilingData_.catDim1 * blockOffset_);
    } else {
        rowsUsedCoreNum_ = tilingData_.uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        blockOffset_ = (blockIdx_ / colsUsedCoreNum_) * tilingData_.ubFactorDim0;
        int64_t colOffset = (blockIdx_ % colsUsedCoreNum_) * tilingData_.blockFactor * tilingData_.ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_.catDim1 + colOffset);
    }

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_.ubFactorDim0 * tilingData_.ubFactorDim1 * sizeof(T));

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    desc_.SetShapeAddr(&buf_[0]);
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        UpdateSplitInfo();
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::UpdateSplitInfo()
{
    if (startTensorOffset_ == GetTensorDim1(startTensorIdx_)) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::GetTensorAddr(
    uint32_t index, int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        ProcessBlockSplitDim1();
    } else {
        if (tilingData_.ubSplitDim1 == 0) {
            ProcessBlockSplitDim0NoSplitDim1();
        } else {
            ProcessBlockSplitDim0SplitDim1();
        }
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = tilingData_.blockFactor;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSize = tilingData_.tailBlockFactor - 1;
    }
    for (int64_t i = 0; i < loopSize; i++) {
        CopyInNoSplitDim1(i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0, tilingData_.ubFactorDim1);
        CopyOut(i * tilingData_.ubFactorDim0 * tilingData_.catDim1, tilingData_.ubFactorDim0, tilingData_.ubFactorDim1);
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        CopyInNoSplitDim1(loopSize * tilingData_.ubFactorDim0, tilingData_.tailUbFactorDim0, tilingData_.ubFactorDim1);
        CopyOut(
            loopSize * tilingData_.ubFactorDim0 * tilingData_.catDim1, tilingData_.tailUbFactorDim0,
            tilingData_.ubFactorDim1);
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = tilingData_.blockFactor;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSizeDim0 = tilingData_.tailBlockFactor - 1;
    }
    SplitInfo splitInfo = {0, 0, 0, 0};
    int64_t lastTensorDim1 = GetTensorDim1(tilingData_.tensorNum - 1);
    for (int64_t i = 0; i < tilingData_.uoDim1 - 1; i++) {
        CalcSplitInfo(tilingData_.tensorNum - 1, lastTensorDim1, tilingData_.ubFactorDim1, splitInfo);
        ProcessBlockSplitDim0SplitDim1PerLoop(i, tilingData_.ubFactorDim1, loopSizeDim0, splitInfo);
    }

    CalcSplitInfo(tilingData_.tensorNum - 1, lastTensorDim1, tilingData_.tailUbFactorDim1, splitInfo);
    ProcessBlockSplitDim0SplitDim1PerLoop(
        tilingData_.uoDim1 - 1, tilingData_.tailUbFactorDim1, loopSizeDim0, splitInfo);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim1()
{
    int64_t loopSizeDim1 = tilingData_.blockFactor - 1;
    int64_t blockIdxInCol = blockIdx_ % colsUsedCoreNum_;
    int64_t blockIdxInRow = blockIdx_ / colsUsedCoreNum_;
    int64_t tailLoopCopyColsNum = tilingData_.ubFactorDim1;
    int64_t copyRowsNum = tilingData_.ubFactorDim0;

    if (blockIdxInCol == colsUsedCoreNum_ - 1) {
        loopSizeDim1 = tilingData_.tailBlockFactor - 1;
        tailLoopCopyColsNum = tilingData_.tailUbFactorDim1;
    }
    if (blockIdxInRow == rowsUsedCoreNum_ - 1) {
        copyRowsNum = tilingData_.tailUbFactorDim0;
    }

    SplitInfo splitInfo = {0, 0, startTensorIdx_, startTensorOffset_};
    for (int64_t i = 0; i < loopSizeDim1; i++) {
        ProcessBlockSplitDim1PerLoop(i, copyRowsNum, tilingData_.ubFactorDim1, blockIdxInCol, splitInfo);
    }

    ProcessBlockSplitDim1PerLoop(loopSizeDim1, copyRowsNum, tailLoopCopyColsNum, blockIdxInCol, splitInfo);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CalcSplitInfo(
    int64_t endIdx, int64_t endOffset, int64_t length, SplitInfo& splitInfo)
{
    splitInfo.startIdx = splitInfo.endIdx;
    splitInfo.startOffset = splitInfo.endOffset;

    if (splitInfo.startOffset == GetTensorDim1(splitInfo.startIdx)) {
        splitInfo.startIdx += 1;
        splitInfo.startOffset = 0;
    }

    if (splitInfo.startIdx == endIdx || splitInfo.startOffset + length <= GetTensorDim1(splitInfo.startIdx)) {
        splitInfo.endIdx = splitInfo.startIdx;
        splitInfo.endOffset = splitInfo.startOffset + length;
        return;
    }

    int64_t curOffset = GetTensorDim1(splitInfo.startIdx) - splitInfo.startOffset;

    for (int64_t i = splitInfo.startIdx + 1; i <= endIdx; i++) {
        curOffset += GetTensorDim1(i);
        if (curOffset >= length) {
            splitInfo.endIdx = i;
            splitInfo.endOffset = GetTensorDim1(i) + length - curOffset;
            return;
        }
    }
    splitInfo.endIdx = endIdx;
    splitInfo.endOffset = endOffset;
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim0SplitDim1PerLoop(
    int64_t loopIdxInCols, int64_t copyColsNum, int64_t loopSizeDim0, SplitInfo& splitInfo)
{
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        CopyInSplitDim1(i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0, copyColsNum, splitInfo);
        CopyOut(
            i * tilingData_.ubFactorDim0 * tilingData_.catDim1 + loopIdxInCols * tilingData_.ubFactorDim1,
            tilingData_.ubFactorDim0, copyColsNum);
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        CopyInSplitDim1(loopSizeDim0 * tilingData_.ubFactorDim0, tilingData_.tailUbFactorDim0, copyColsNum, splitInfo);
        CopyOut(
            loopSizeDim0 * tilingData_.ubFactorDim0 * tilingData_.catDim1 + loopIdxInCols * tilingData_.ubFactorDim1,
            tilingData_.tailUbFactorDim0, copyColsNum);
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim1PerLoop(
    int64_t loopIdx, int64_t rows, int64_t cols, int64_t blockIdxInCol, SplitInfo& splitInfo)
{
    CalcSplitInfo(endTensorIdx_, endTensorOffset_, cols, splitInfo);
    CopyInSplitDim1(0, rows, cols, splitInfo);
    CopyOut(loopIdx * tilingData_.ubFactorDim1, rows, cols);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CopyInNoSplitDim1(
    int64_t srcRowsOffset, int64_t rows, int64_t cols)
{
    int64_t curDim1Offset = 0;
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    for (int64_t i = 0; i < tilingData_.tensorNum; i++) {
        int64_t dim1 = GetTensorDim1(i);
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
        SetCopyInparam(rows, dim1, dim0stride - dim1, (cols - dim1) / numPerBlock_);
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
        curDim1Offset += dim1;
    }
    inQueue_.EnQue(srcLocal);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CopyInSplitDim1(
    int64_t srcRowsOffset, int64_t rows, int64_t cols, SplitInfo& splitInfo)
{
    int64_t curDim1Offset = 0;
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();

    int64_t dim1 = GetTensorDim1(splitInfo.startIdx);
    int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, dim1);
    int64_t copyColsNum = 0;
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    if (splitInfo.startIdx == splitInfo.endIdx) {
        copyColsNum = splitInfo.endOffset - splitInfo.startOffset;
    } else {
        copyColsNum = dim1 - splitInfo.startOffset;
    }
    srcGlobal_.SetGlobalBuffer(GetTensorAddr(splitInfo.startIdx, blockOffset_ * dim0stride + splitInfo.startOffset));
    SetCopyInparam(rows, copyColsNum, dim0stride - copyColsNum, (cols - copyColsNum) / numPerBlock_);
    DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
    curDim1Offset += copyColsNum;

    for (int64_t i = splitInfo.startIdx + 1; i < splitInfo.endIdx; i++) {
        dim1 = GetTensorDim1(i);
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
        SetCopyInparam(rows, dim1, dim0stride - dim1, (cols - dim1) / numPerBlock_);
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
        curDim1Offset += dim1;
    }

    if (splitInfo.startIdx != splitInfo.endIdx) {
        dim1 = GetTensorDim1(splitInfo.endIdx);
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.endIdx, dim1);
        copyColsNum = splitInfo.endOffset;
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(splitInfo.endIdx, blockOffset_ * dim0stride));
        SetCopyInparam(rows, copyColsNum, dim0stride - copyColsNum, (cols - copyColsNum) / numPerBlock_);
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
    }
    inQueue_.EnQue(srcLocal);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::SetCopyInparam(
    int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * sizeof(T);
    copyInParam_.srcStride = srcStride * sizeof(T);
    copyInParam_.dstStride = dstStride;
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CopyOut(
    int64_t dstOffset, int64_t rows, int64_t cols)
{
    LocalTensor<T> dstLocal = inQueue_.DeQue<T>();
    DataCopyExtParams copyOutParam{0, 0, 0, 0, 0};
    copyOutParam.blockCount = rows;
    copyOutParam.blockLen = cols * sizeof(T);
    copyOutParam.dstStride = (tilingData_.catDim1 - cols) * sizeof(T);
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
    inQueue_.FreeTensor(dstLocal);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline int64_t OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::GetTensorDim1(int64_t idx)
{
    if constexpr (SAMESHAPE) {
        return tilingData_.sameShapeTensorDim1;
    } else {
        if (idx < PRELOAD_DIM1_SIZE) {
            return tilingData_.arrays.preLoadDim1[idx];
        }
        int64_t dim1 = GetNonConDimSize<TILINGDATA, T>(tilingData_, idx, inputList_, desc_) * tilingData_.sameShapeTensorDim1;
        if (tilingData_.isFP4Type) {
            dim1 /= 2;
        }
        return dim1;
    }
}
} // namespace Concat

#endif // ONE_AXIS_CONCAT_ALL_ALIGN_