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
 * \file one_axis_concat_pure_copy.h
 * \brief one axis concat pure copy
 */

#ifndef ONE_AXIS_CONCAT_PURE_COPY_
#define ONE_AXIS_CONCAT_PURE_COPY_

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename TILINGDATA = ConcatTilingData>
class OneAxisConcatPureCopy {
public:
    __aicore__ inline OneAxisConcatPureCopy(const TILINGDATA& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ int8_t* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInSingleTensor(
        const GlobalTensor<int8_t> srcGm, int64_t copyRows, int64_t copyCols, int64_t srcOffset, int64_t tensorDim1);
    __aicore__ inline void CopyOutSingleTensor(int64_t copyRows, int64_t copyCols, int64_t dstOffset);
    __aicore__ inline void ProcessSingleTensor(
        int64_t tensorIdx, int64_t tensorDim1, int64_t totalRows, int64_t totalCols, int64_t globalSrcOffset,
        int64_t globalDstOffset);
    __aicore__ inline void ProcessNoSplitDim1();
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void ProcessSplitDim1();
    __aicore__ inline int64_t GetTensorDim1(int64_t idx);
    __aicore__ inline void UpdateSplitInfo();

private:
    TPipe& pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
    GlobalTensor<int8_t> dstGlobal_;
    GlobalTensor<int8_t> srcGlobal_;
    const TILINGDATA& tilingData_;
    uint32_t blockIdx_{0};
    int64_t blockOffset_{0};

    uint32_t colsUsedCoreNum_{1};
    uint32_t rowsUsedCoreNum_{1};
    uint32_t blockIdxInCol_{0};
    uint32_t blockIdxInRow_{0};

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};

    int64_t endTensorIdx_{0};
    int64_t endTensorOffset_{0};
    int64_t startTensorIdx_{0};
    int64_t startTensorOffset_{0};
    ListTensorDesc inputList_;
    uint64_t buf_[ARRAY_SIZE];
    TensorDesc<int8_t> desc_;
};

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        rowsUsedCoreNum_ = tilingData_.uoDim0;
        colsUsedCoreNum_ = tilingData_.uoDim1;
        blockIdxInCol_ = blockIdx_ % colsUsedCoreNum_;
        blockIdxInRow_ = blockIdx_ / colsUsedCoreNum_;
        if (blockIdxInCol_ != 0) {
            startTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_ - 1];
            startTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx_ - 1];
        }
        endTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx_];
        endTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_];
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value) {
        blockOffset_ = static_cast<int64_t>(blockIdx_) * tilingData_.ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ int8_t*)dst + tilingData_.catDim1 * tilingData_.dtypeSize * static_cast<int64_t>(blockOffset_));
    } else {
        blockOffset_ = blockIdxInRow_ * tilingData_.ubFactorDim0;
        int64_t colOffset = static_cast<int64_t>(blockIdxInCol_) * tilingData_.ubFactorDim1 * tilingData_.dtypeSize;
        dstGlobal_.SetGlobalBuffer(
            (__gm__ int8_t*)dst + blockOffset_ * tilingData_.catDim1 * tilingData_.dtypeSize + colOffset);
    }

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_.bufferSize * tilingData_.dtypeSize);

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    desc_.SetShapeAddr(&buf_[0]);
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        UpdateSplitInfo();
    }
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::UpdateSplitInfo()
{
    if (startTensorOffset_ == GetTensorDim1(startTensorIdx_)) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
}

template <typename TILINGDATA>
__aicore__ inline __gm__ int8_t* OneAxisConcatPureCopy<TILINGDATA>::GetTensorAddr(uint32_t index, int64_t offset)
{
    return inputList_.GetDataPtr<int8_t>(index) + offset;
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        ProcessSplitDim1();
    } else {
        ProcessNoSplitDim1();
    }
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::ProcessNoSplitDim1()
{
    int64_t totalRows = tilingData_.ubFactorDim0;
    if (blockIdx_ == GetBlockNum() - 1) {
        totalRows = tilingData_.tailUbFactorDim0;
    }
    int64_t totalColOffset = 0;
    for (int64_t i = 0; i < tilingData_.tensorNum; i++) {
        int64_t dim1 = GetTensorDim1(i);
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        int64_t globalSrcOffset = blockOffset_ * dim0stride;
        ProcessSingleTensor(i, dim0stride, totalRows, dim1, globalSrcOffset, totalColOffset);
        totalColOffset += dim1;
    }
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::ProcessSplitDim1()
{
    int64_t totalRows = tilingData_.ubFactorDim0;
    if (blockIdxInRow_ == rowsUsedCoreNum_ - 1) {
        totalRows = tilingData_.tailUbFactorDim0;
    }
    int64_t totalColOffset = 0;
    int64_t globalSrcOffset = 0;
    int64_t tensorDim1 = GetTensorDim1(startTensorIdx_);
    int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, startTensorIdx_, tensorDim1);
    globalSrcOffset = blockOffset_ * dim0stride + startTensorOffset_;
    if (startTensorIdx_ == endTensorIdx_) {
        ProcessSingleTensor(
            startTensorIdx_, dim0stride, totalRows, endTensorOffset_ - startTensorOffset_, globalSrcOffset,
            totalColOffset);
        return;
    }
    ProcessSingleTensor(
        startTensorIdx_, dim0stride, totalRows, tensorDim1 - startTensorOffset_, globalSrcOffset, totalColOffset);
    totalColOffset += tensorDim1 - startTensorOffset_;
    for (int64_t i = startTensorIdx_ + 1; i < endTensorIdx_; i++) {
        tensorDim1 = GetTensorDim1(i);
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, tensorDim1);
        globalSrcOffset = blockOffset_ * dim0stride;
        ProcessSingleTensor(i, dim0stride, totalRows, tensorDim1, globalSrcOffset, totalColOffset);
        totalColOffset += tensorDim1;
    }
    tensorDim1 = GetTensorDim1(endTensorIdx_);
    dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, endTensorIdx_, tensorDim1);
    globalSrcOffset = blockOffset_ * dim0stride;
    ProcessSingleTensor(endTensorIdx_, dim0stride, totalRows, endTensorOffset_, globalSrcOffset, totalColOffset);
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::ProcessSingleTensor(
    int64_t tensorIdx, int64_t tensorDim1, int64_t totalRows, int64_t totalCols, int64_t globalSrcOffset,
    int64_t globalDstOffset)
{
    if (tensorDim1 == 0 || totalRows == 0 || totalCols == 0) {
        return;
    }
    srcGlobal_.SetGlobalBuffer(GetTensorAddr(tensorIdx, globalSrcOffset * tilingData_.dtypeSize));
    int64_t colFactor = min(totalCols, static_cast<int64_t>(tilingData_.bufferSize));
    int64_t rowFactor = tilingData_.bufferSize / colFactor;
    int64_t colLoopSize = (totalCols + colFactor - 1) / colFactor;
    int64_t rowLoopSize = (totalRows + rowFactor - 1) / rowFactor;
    int64_t tailColFactor = totalCols - (colLoopSize - 1) * colFactor;
    int64_t tailRowFactor = totalRows - (rowLoopSize - 1) * rowFactor;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    for (int64_t m = 0; m < rowLoopSize; m++) {
        int64_t copyRows = m == rowLoopSize - 1 ? tailRowFactor : rowFactor;
        dstOffset = globalDstOffset + m * rowFactor * tilingData_.catDim1;
        for (int64_t n = 0; n < colLoopSize - 1; n++) {
            srcOffset = m * rowFactor * tensorDim1 + n * colFactor;
            CopyInSingleTensor(srcGlobal_, copyRows, colFactor, srcOffset, tensorDim1);
            CopyOutSingleTensor(copyRows, colFactor, dstOffset);
            dstOffset += colFactor;
        }
        srcOffset = m * rowFactor * tensorDim1 + (colLoopSize - 1) * colFactor;
        CopyInSingleTensor(srcGlobal_, copyRows, tailColFactor, srcOffset, tensorDim1);
        CopyOutSingleTensor(copyRows, tailColFactor, dstOffset);
    }
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::CopyInSingleTensor(
    const GlobalTensor<int8_t> srcGm, int64_t copyRows, int64_t copyCols, int64_t srcOffset, int64_t tensorDim1)
{
    LocalTensor<int8_t> srcLocal = inQueue_.AllocTensor<int8_t>();
    DataCopyPadExtParams<int8_t> padParam{false, 0, 0, 0};
    SetCopyInparam(copyRows, copyCols, tensorDim1 - copyCols, 0);
    DataCopyPad<int8_t, PaddingMode::Compact>(
        srcLocal, srcGm[srcOffset * tilingData_.dtypeSize], copyInParam_, padParam);
    inQueue_.EnQue(srcLocal);
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::CopyOutSingleTensor(
    int64_t copyRows, int64_t copyCols, int64_t dstOffset)
{
    LocalTensor<int8_t> dstLocal = inQueue_.DeQue<int8_t>();
    DataCopyExtParams copyOutParam{0, 0, 0, 0, 0};
    copyOutParam.blockCount = copyRows;
    copyOutParam.blockLen = copyCols * tilingData_.dtypeSize;
    copyOutParam.dstStride = (tilingData_.catDim1 - copyCols) * tilingData_.dtypeSize;
    copyOutParam.srcStride = 0;
    DataCopyPad<int8_t, PaddingMode::Compact>(dstGlobal_[dstOffset * tilingData_.dtypeSize], dstLocal, copyOutParam);
    inQueue_.FreeTensor(dstLocal);
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::SetCopyInparam(
    int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * tilingData_.dtypeSize;
    copyInParam_.srcStride = srcStride * tilingData_.dtypeSize;
    copyInParam_.dstStride = dstStride;
}

template <typename TILINGDATA>
__aicore__ inline int64_t OneAxisConcatPureCopy<TILINGDATA>::GetTensorDim1(int64_t idx)
{
    if (idx < PRELOAD_DIM1_SIZE) {
        return tilingData_.arrays.preLoadDim1[idx];
    }
    int64_t dim1 = GetNonConDimSize<TILINGDATA, int8_t>(tilingData_, idx, inputList_, desc_) * tilingData_.sameShapeTensorDim1;
    if (tilingData_.isFP4Type) {
        dim1 /= 2;
    }
    return dim1;
}
} // namespace Concat

#endif // ONE_AXIS_CONCAT_PURE_COPY_