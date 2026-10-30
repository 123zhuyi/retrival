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
 * \file one_axis_concat_no_align_same_shape_copy.h
 * \brief one axis concat no align same shape copy
 */

#ifndef ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_COPY
#define ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_COPY

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename T, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatNoAlignCopy {
public:
    __aicore__ inline OneAxisConcatNoAlignCopy(const TILINGDATA& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t startIdx, int64_t endIdx);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t rows, int64_t cols);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t dataLen);
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ComputeNoSplitDim1Copy(uint32_t rows, uint32_t cols, uint32_t tensorStride,
                                                  uint32_t inputCols, uint32_t dstColStride);
    __aicore__ inline void ComputeSplitDim1Copy(uint32_t rows, uint32_t cols, SplitInfo& splitInfo);
    __aicore__ inline void CopyInSingleTensor(const LocalTensor<T>& dstLocal, int64_t tensorIdx, int64_t srcOffset,
                                              const DataCopyExtParams& params);
    __aicore__ inline int64_t CopyInSplitDim1(int64_t endIdx, int64_t endOffset, int64_t srcRowsOffset, int64_t rows,
                                              SplitInfo& splitInfo);

private:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;
    const TILINGDATA& tilingData_;
    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t numPerBlock_ = BYTES_PER_BLOCK / sizeof(T);
    int64_t colsUsedCoreNum_ = 1;
    int64_t rowsUsedCoreNum_ = 1;

    DataCopyExtParams copyInParam_ = {0, 0, 0, 0, 0};

    int64_t startTensorIdx_ = 0;
    int64_t startTensorOffset_ = 0;
    int64_t endTensorIdx_ = 0;
    int64_t endTensorOffset_ = 0;
    ListTensorDesc inputList_;

    uint32_t vfLen_ = GetVRegSize() / sizeof(T);
};

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        int64_t colsUsedCoreNum = GetBlockNum() / tilingData_.uoDim0;
        if (blockIdx_ % colsUsedCoreNum != 0) {
            startTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_ % colsUsedCoreNum - 1];
            startTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx_ % colsUsedCoreNum - 1];
        }
        endTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_];
        endTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx_];
    }
    if (startTensorOffset_ == tilingData_.sameShapeTensorDim1) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        rowsUsedCoreNum_ = tilingData_.uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        blockOffset_ = (blockIdx_ / colsUsedCoreNum_) * tilingData_.ubFactorDim0;
        int64_t colOffset = (blockIdx_ % colsUsedCoreNum_) * tilingData_.blockFactor * tilingData_.ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_.catDim1 + colOffset);
    } else {
        blockOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_.catDim1);
    }

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_.bufferSize * sizeof(T));
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.bufferSize * sizeof(T));

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
}

template <typename T, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatNoAlignCopy<T, TILINGDATA>::GetTensorAddr(uint32_t index, int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        ProcessBlockSplitDim1();
    } else {
        if (0 == tilingData_.ubSplitDim1) {
            ProcessBlockSplitDim0NoSplitDim1();
        } else {
            ProcessBlockSplitDim0SplitDim1();
        }
    }
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ProcessBlockSplitDim1()
{
    int64_t blockIdxInCol = blockIdx_ % colsUsedCoreNum_;
    int64_t blockIdxInRow = blockIdx_ / colsUsedCoreNum_;
    int64_t copyRowsNum = tilingData_.ubFactorDim0;

    if (blockIdxInRow == rowsUsedCoreNum_ - 1) {
        copyRowsNum = tilingData_.tailUbFactorDim0;
    }
    int64_t curColOffset = 0;
    int64_t curColNum = 0;
    int64_t totalCols = 0;
    SplitInfo splitInfo = {0, 0, startTensorIdx_, startTensorOffset_};
    if (startTensorIdx_ == endTensorIdx_) {
        totalCols = endTensorOffset_ - startTensorOffset_;
    } else {
        totalCols = (tilingData_.sameShapeTensorDim1 - startTensorOffset_) + endTensorOffset_ +
                    (endTensorIdx_ - startTensorIdx_ - 1) * tilingData_.sameShapeTensorDim1;
    }
    while (curColOffset < totalCols) {
        curColNum = CopyInSplitDim1(endTensorIdx_, endTensorOffset_, 0, copyRowsNum, splitInfo);
        ComputeSplitDim1Copy(copyRowsNum, curColNum, splitInfo);
        CopyOut(curColOffset, copyRowsNum, curColNum);
        curColOffset += curColNum;
    }
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = tilingData_.blockFactor;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSizeDim0 = tilingData_.tailBlockFactor - 1;
    }

    int64_t curColOffset = 0;
    int64_t curColNum = 0;
    SplitInfo splitInfo = {0, 0, 0, 0};

    for (int64_t i = 0; i < loopSizeDim0; i++) {
        curColOffset = 0;
        curColNum = 0;
        splitInfo = {0, 0, 0, 0};
        while (curColOffset < tilingData_.catDim1) {
            curColNum = CopyInSplitDim1(tilingData_.tensorNum - 1, tilingData_.sameShapeTensorDim1,
                                        i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0, splitInfo);
            ComputeSplitDim1Copy(tilingData_.ubFactorDim0, curColNum, splitInfo);
            CopyOut(i * tilingData_.ubFactorDim0 * tilingData_.catDim1 + curColOffset, tilingData_.ubFactorDim0,
                    curColNum);
            curColOffset += curColNum;
        }
    }

    if (blockIdx_ == GetBlockNum() - 1) {
        curColOffset = 0;
        curColNum = 0;
        splitInfo = {0, 0, 0, 0};
        while (curColOffset < tilingData_.catDim1) {
            curColNum = CopyInSplitDim1(tilingData_.tensorNum - 1, tilingData_.sameShapeTensorDim1,
                                        loopSizeDim0 * tilingData_.ubFactorDim0, tilingData_.tailUbFactorDim0,
                                        splitInfo);
            ComputeSplitDim1Copy(tilingData_.tailUbFactorDim0, curColNum, splitInfo);
            CopyOut(loopSizeDim0 * tilingData_.ubFactorDim0 * tilingData_.catDim1 + curColOffset,
                    tilingData_.tailUbFactorDim0, curColNum);
            curColOffset += curColNum;
        }
    }
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = tilingData_.blockFactor;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSize = tilingData_.tailBlockFactor - 1;
    }
    int64_t tensorStride = (tilingData_.ubFactorDim0 * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) /
                           numPerBlock_ * numPerBlock_;

    for (int64_t i = 0; i < loopSize; i++) {
        CopyInNoSplitDim1(i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0, 0, tilingData_.tensorNum);
        ComputeNoSplitDim1Copy(tilingData_.ubFactorDim0, tilingData_.catDim1, tensorStride,
                               tilingData_.sameShapeTensorDim1, tilingData_.catDim1);
        CopyOut(i * tilingData_.ubFactorDim0 * tilingData_.catDim1, tilingData_.ubFactorDim0 * tilingData_.catDim1);
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        tensorStride = (tilingData_.tailUbFactorDim0 * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) /
                       numPerBlock_ * numPerBlock_;
        CopyInNoSplitDim1(loopSize * tilingData_.ubFactorDim0, tilingData_.tailUbFactorDim0, 0, tilingData_.tensorNum);
        ComputeNoSplitDim1Copy(tilingData_.tailUbFactorDim0, tilingData_.catDim1, tensorStride,
                               tilingData_.sameShapeTensorDim1, tilingData_.catDim1);
        CopyOut(loopSize * tilingData_.ubFactorDim0 * tilingData_.catDim1,
                tilingData_.tailUbFactorDim0 * tilingData_.catDim1);
    }
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ComputeNoSplitDim1Copy(uint32_t rows, uint32_t cols,
                                                                                       uint32_t tensorStride,
                                                                                       uint32_t inputCols,
                                                                                       uint32_t dstColStride)
{
    LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
    // tiling可保证每个tensor元素个数不超过uint16最大值
    uint16_t inputNum = cols / inputCols; // 每个cols包含完整的inputCols
    uint16_t size0 = rows;
    uint16_t size1 = inputCols / vfLen_;

    uint32_t main = vfLen_;
    uint32_t tail = inputCols - size1 * vfLen_;

    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __ubuf__ T* curDstAddr;
    __ubuf__ T* curSrcAddr;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vd0;
        AscendC::Reg::RegTensor<T> vd1;
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::UnalignReg u1;
        for (uint16_t i = 0; i < size0; i++) {
            for (uint16_t j = 0; j < inputNum; j++) {
                curSrcAddr = srcAddr + j * tensorStride + i * inputCols;
                curDstAddr = dstAddr + i * dstColStride + j * inputCols;
                AscendC::Reg::DataCopyUnAlignPre(u0, curSrcAddr);
                for (uint16_t k = 0; k < size1; k++) {
                    AscendC::Reg::DataCopyUnAlign(vd0, u0, curSrcAddr, main);
                    AscendC::Reg::DataCopyUnAlign(curDstAddr, vd0, u1, main);
                }
                AscendC::Reg::DataCopyUnAlign(vd0, u0, curSrcAddr, tail);
                AscendC::Reg::DataCopyUnAlign(curDstAddr, vd0, u1, tail);
                AscendC::Reg::DataCopyUnAlignPost(curDstAddr, u1, 0);
            }
        }
    }

    inQueue_.FreeTensor(srcLocal);
    outQueue_.EnQue(dstLocal);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ComputeSplitDim1Copy(uint32_t rows, uint32_t cols,
                                                                                     SplitInfo& splitInfo)
{
    LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();

    int64_t copyColsNum = 0;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    uint32_t rowStride = (cols + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    uint32_t tensorOffset = (rows * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;

    if (splitInfo.startIdx == splitInfo.endIdx) {
        copyColsNum = splitInfo.endOffset - splitInfo.startOffset;
    } else {
        copyColsNum = tilingData_.sameShapeTensorDim1 - splitInfo.startOffset;
    }
    Copy(dstLocal, srcLocal, rows, copyColsNum, rowStride);
    srcOffset += (copyColsNum * rows + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    dstOffset += copyColsNum;

    for (int64_t i = splitInfo.startIdx + 1; i < splitInfo.endIdx; i++) {
        Copy(dstLocal[dstOffset], srcLocal[srcOffset], rows, tilingData_.sameShapeTensorDim1, rowStride);
        dstOffset += tilingData_.sameShapeTensorDim1;
        srcOffset += tensorOffset;
    }

    if (splitInfo.startIdx != splitInfo.endIdx) {
        copyColsNum = splitInfo.endOffset;
        Copy(dstLocal[dstOffset], srcLocal[srcOffset], rows, copyColsNum, rowStride);
    }

    inQueue_.FreeTensor(srcLocal);
    outQueue_.EnQue(dstLocal);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows,
                                                                                  int64_t startIdx, int64_t endIdx)
{
    int64_t curDim1Offset = 0;
    int64_t tensorStride = (rows * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    SetCopyInparam(1, rows * tilingData_.sameShapeTensorDim1, 0, 0);
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParam = {false, 0, 0, 0};
    for (int64_t i = startIdx; i < endIdx; i++) {
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, tilingData_.sameShapeTensorDim1);
        if (tilingData_.isNonContiguous) {
            SetCopyInparam(rows, tilingData_.sameShapeTensorDim1, dim0stride - tilingData_.sameShapeTensorDim1, 0);
        } else {
            SetCopyInparam(1, rows * tilingData_.sameShapeTensorDim1, 0, 0);
        }
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
        DataCopyPad<T, PaddingMode::Compact>(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride],
                                             copyInParam_, padParam);
        curDim1Offset += tensorStride;
    }
    inQueue_.EnQue(srcLocal);
}

template <typename T, typename TILINGDATA>
__aicore__ inline int64_t OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyInSplitDim1(int64_t endIdx, int64_t endOffset,
                                                                                   int64_t srcRowsOffset, int64_t rows,
                                                                                   SplitInfo& splitInfo)
{
    int64_t curOffset = 0;
    int64_t limit = tilingData_.bufferSize - rows * numPerBlock_ - numPerBlock_;
    int64_t limitCols = limit / rows;
    int64_t copyCols = 0;
    int64_t colOffset = 0;
    int64_t dim0stride = 0;
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();

    splitInfo.startIdx = splitInfo.endIdx;
    splitInfo.startOffset = splitInfo.endOffset;

    if (splitInfo.startOffset == tilingData_.sameShapeTensorDim1) {
        splitInfo.startIdx += 1;
        splitInfo.startOffset = 0;
    }
    if (splitInfo.startIdx == endIdx) {
        copyCols = splitInfo.startOffset + limitCols > endOffset ? endOffset - splitInfo.startOffset : limitCols;
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, tilingData_.sameShapeTensorDim1);
        SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
        splitInfo.endOffset = splitInfo.startOffset + copyCols;
        splitInfo.endIdx = splitInfo.startIdx;
        CopyInSingleTensor(srcLocal, splitInfo.startIdx, srcRowsOffset * dim0stride + splitInfo.startOffset,
                           copyInParam_);
        inQueue_.EnQue(srcLocal);
        return copyCols;
    }
    if ((tilingData_.sameShapeTensorDim1 - splitInfo.startOffset) >= limitCols) {
        copyCols = limitCols;
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, tilingData_.sameShapeTensorDim1);
        SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
        splitInfo.endOffset = splitInfo.startOffset + copyCols;
        splitInfo.endIdx = splitInfo.startIdx;
        CopyInSingleTensor(srcLocal, splitInfo.startIdx, srcRowsOffset * dim0stride + splitInfo.startOffset,
                           copyInParam_);
        inQueue_.EnQue(srcLocal);
        return copyCols;
    }
    copyCols = tilingData_.sameShapeTensorDim1 - splitInfo.startOffset;
    dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, tilingData_.sameShapeTensorDim1);
    SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
    CopyInSingleTensor(srcLocal, splitInfo.startIdx, srcRowsOffset * dim0stride + splitInfo.startOffset, copyInParam_);
    colOffset += copyCols;
    curOffset += (copyCols * rows + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;

    int64_t tensorStride = (tilingData_.sameShapeTensorDim1 * rows + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    for (int64_t i = splitInfo.startIdx + 1; i < endIdx; i++) {
        if (curOffset + tensorStride >= limit) {
            copyCols = min((limit - curOffset) / rows, tilingData_.sameShapeTensorDim1);
            splitInfo.endIdx = i;
            splitInfo.endOffset = copyCols;
            dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, tilingData_.sameShapeTensorDim1);
            SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
            CopyInSingleTensor(srcLocal[curOffset], i, srcRowsOffset * dim0stride, copyInParam_);
            colOffset += copyCols;
            inQueue_.EnQue(srcLocal);
            return colOffset;
        }
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, tilingData_.sameShapeTensorDim1);
        SetCopyInparam(rows, tilingData_.sameShapeTensorDim1, dim0stride - tilingData_.sameShapeTensorDim1,
                       tilingData_.sameShapeTensorDim1);
        CopyInSingleTensor(srcLocal[curOffset], i, srcRowsOffset * dim0stride, copyInParam_);
        colOffset += tilingData_.sameShapeTensorDim1;
        curOffset += tensorStride;
    }

    if (curOffset + endOffset * rows < limit) {
        copyCols = endOffset;
    } else {
        copyCols = (limit - curOffset) / rows;
    }
    splitInfo.endIdx = endIdx;
    splitInfo.endOffset = copyCols;
    dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.endIdx, tilingData_.sameShapeTensorDim1);
    SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
    CopyInSingleTensor(srcLocal[curOffset], endIdx, srcRowsOffset * dim0stride, copyInParam_);
    colOffset += copyCols;
    inQueue_.EnQue(srcLocal);
    return colOffset;
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyInSingleTensor(const LocalTensor<T>& dstLocal,
                                                                                   int64_t tensorIdx, int64_t srcOffset,
                                                                                   const DataCopyExtParams& params)
{
    int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, tensorIdx, tilingData_.sameShapeTensorDim1);
    srcGlobal_.SetGlobalBuffer(GetTensorAddr(tensorIdx, blockOffset_ * dim0stride));
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyPad<T, PaddingMode::Compact>(dstLocal, srcGlobal_[srcOffset], params, padParams);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::SetCopyInparam(int64_t rows, int64_t cols,
                                                                               int64_t srcStride, int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * sizeof(T);
    copyInParam_.srcStride = srcStride * sizeof(T);
    copyInParam_.dstStride = dstStride * sizeof(T);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t rows, int64_t cols)
{
    LocalTensor<T> dstLocal = outQueue_.DeQue<T>();
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = rows;
    copyOutParam.blockLen = cols * sizeof(T);
    copyOutParam.dstStride = (tilingData_.catDim1 - cols) * sizeof(T);
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
    outQueue_.FreeTensor(dstLocal);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t dataLen)
{
    LocalTensor<T> dstLocal = outQueue_.DeQue<T>();
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = 1;
    copyOutParam.blockLen = dataLen * sizeof(T);
    copyOutParam.dstStride = 0;
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
    outQueue_.FreeTensor(dstLocal);
}

} // namespace Concat

#endif // ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_COPY
