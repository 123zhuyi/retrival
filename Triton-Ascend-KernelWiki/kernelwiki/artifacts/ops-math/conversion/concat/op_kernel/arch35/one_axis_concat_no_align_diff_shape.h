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
 * \file one_axis_concat_no_align_diff_shape.h
 * \brief one axis concat no align diff shape
 */

#ifndef ONE_AXIS_CONCAT_NO_ALIGN_DIFF_SHAPE
#define ONE_AXIS_CONCAT_NO_ALIGN_DIFF_SHAPE

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/math_util.h"

namespace Concat {
using namespace AscendC;
using namespace Ops::Base;
template <typename T, typename U, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatNoAlignDiffShape {
public:
    __aicore__ inline OneAxisConcatNoAlignDiffShape(const TILINGDATA& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows);
    __aicore__ inline void CopyOut(LocalTensor<T> dstLocal, int64_t dstOffset, uint16_t rows, int64_t cols);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t dataLen);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ComputeSplitDim1(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, uint32_t rows,
                                            uint32_t cols, uint32_t dstOffset, uint32_t curLoopHandleCols);
    __aicore__ inline void GenScatterIndex(U curLoopHandleCols, U curTensorStartCols, U curLoopHandleCurTensorCols,
                                           LocalTensor<U>& indexLocal);
    __aicore__ inline void ScatterSplitDim1(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, LocalTensor<U> indexLocal,
                                            uint32_t curLoopHandleRows, uint32_t curLoopHandleCols,
                                            uint32_t curLoopHandleCurTensorCols);

private:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TBuf<QuePosition::VECCALC> vciSequenceBuf_;
    GlobalTensor<T> dstGlobal_;
    const TILINGDATA& tilingData_;
    int64_t blockOffset_ = 0;
    int64_t numPerBlock_ = BYTES_PER_BLOCK / sizeof(T);
    int64_t rowsUsedCoreNum_ = 1;
    int64_t colsUsedCoreNum_ = 1;
    int64_t startTensorIdx_ = 0;
    int64_t startTensorOffset_ = 0;
    int64_t endTensorIdx_ = 0;
    int64_t endTensorOffset_ = 0;
    TensorDesc<T> desc_;
    ListTensorDesc inputList_;
    uint64_t buf_[ARRAY_SIZE];
    static constexpr uint32_t SCATTER_MAX_LEN = GetVRegSize() * DIGIT_FOUR / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL4 = GetVRegSize() * DIGIT_THREE / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL3 = GetVRegSize() * DIGIT_TWO / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL2 = GetVRegSize() / sizeof(U);
    static constexpr uint32_t COLS_SIZE_UNROLL1 = GetVRegSize() / DIGIT_TWO / sizeof(U);
};

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    int64_t blockIdx = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value) {
        blockOffset_ = blockIdx * tilingData_.blockFactor * tilingData_.ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_.catDim1);
    } else {
        rowsUsedCoreNum_ = tilingData_.uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        int64_t blockIdxInCols = blockIdx / colsUsedCoreNum_;
        int64_t blockIdxInRow = blockIdx - blockIdxInCols * colsUsedCoreNum_;
        if (blockIdxInRow != 0) {
            startTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdxInRow - 1];
            startTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdxInRow - 1];
        }
        endTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx];
        endTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx];
        blockOffset_ = blockIdxInCols * tilingData_.ubFactorDim0;
        int64_t colOffset = blockIdxInRow * tilingData_.blockFactor * tilingData_.ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_.catDim1 + colOffset);
    }

    pipe_.InitBuffer(indexBuf_, INDEX_SIZE);
    pipe_.InitBuffer(vciSequenceBuf_, INDEX_SIZE);
    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_.bufferSize * sizeof(T));
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.bufferSize * sizeof(T));

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    desc_.SetShapeAddr(&buf_[0]);
    int64_t startTensorDim1_ = GetNonConDimSize<TILINGDATA, T>(tilingData_, startTensorIdx_, inputList_, desc_) *
                               tilingData_.sameShapeTensorDim1;
    if (tilingData_.isFP4Type) {
        startTensorDim1_ /= 2;
    }
    if (startTensorOffset_ == startTensorDim1_) {
        startTensorOffset_ = 0;
        startTensorIdx_ += 1;
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::GetTensorAddr(uint32_t index,
                                                                                           int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        ProcessBlockSplitDim1();
    } else {
        if (tilingData_.ubSplitDim1 == 1) {
            ProcessBlockSplitDim0SplitDim1();
        } else {
            ProcessBlockSplitDim0NoSplitDim1();
        }
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessBlockSplitDim1()
{
    GlobalTensor<T> srcGlobal;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t totalCopyCols = 0;
    int64_t colsOffset = startTensorOffset_;
    int64_t tensorIdx = startTensorIdx_;
    int64_t tensorStride = 0;
    int64_t outTensorColsOffset = 0;
    uint16_t rows = static_cast<uint16_t>(tilingData_.ubFactorDim0);
    if (GetBlockIdx() / colsUsedCoreNum_ == rowsUsedCoreNum_ - 1) {
        rows = static_cast<uint16_t>(tilingData_.tailUbFactorDim0);
    }
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
    uint32_t curLoopHandleCols = static_cast<uint32_t>(tilingData_.ubFactorDim1);
    while (tensorIdx <= endTensorIdx_) {
        int64_t dim1Size = GetNonConDimSize<TILINGDATA, T>(tilingData_, tensorIdx, inputList_, desc_) *
                           tilingData_.sameShapeTensorDim1;
        if (tilingData_.isFP4Type) {
            dim1Size /= 2;
        }
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, tensorIdx, dim1Size);
        int64_t copyCols = dim1Size - colsOffset;
        if (tensorIdx == endTensorIdx_) {
            copyCols = endTensorOffset_ - colsOffset;
        }
        int64_t extraCols = totalCopyCols + copyCols - tilingData_.ubFactorDim1;
        bool isSplit = extraCols >= numPerBlock_ || (extraCols > 0 && totalCopyCols == 0);
        if (extraCols <= 0 || isSplit) {
            if (isSplit) {
                copyCols = tilingData_.ubFactorDim1 - totalCopyCols;
            }
            DataCopyExtParams copyInParam = {rows, static_cast<uint32_t>(copyCols * sizeof(T)),
                                             static_cast<int64_t>((dim0stride - copyCols) * sizeof(T)),
                                             static_cast<int64_t>(copyCols * sizeof(T)), 0};
            srcGlobal.SetGlobalBuffer(GetTensorAddr(tensorIdx, blockOffset_ * dim0stride + colsOffset));
            DataCopyPad<T, PaddingMode::Compact>(srcLocal[tensorStride], srcGlobal, copyInParam, padParams);
            ComputeSplitDim1(dstLocal, srcLocal[tensorStride], rows, copyCols, totalCopyCols, curLoopHandleCols);
            if (isSplit) {
                colsOffset += copyCols;
            } else {
                colsOffset = 0;
                tensorIdx++;
            }
            totalCopyCols += copyCols;
            tensorStride += CeilAlign(rows * copyCols, numPerBlock_);
        }
        if (tensorIdx > endTensorIdx_ || totalCopyCols == tilingData_.ubFactorDim1 ||
            (extraCols > 0 && extraCols < numPerBlock_)) {
            inQueue_.FreeTensor(srcLocal);
            srcLocal = inQueue_.AllocTensor<T>();
            CopyOut(dstLocal, outTensorColsOffset, rows, totalCopyCols);
            dstLocal = outQueue_.AllocTensor<T>();
            outTensorColsOffset += totalCopyCols;
            totalCopyCols = 0;
            tensorStride = 0;
        }
    }
    inQueue_.FreeTensor(srcLocal);
    outQueue_.FreeTensor(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = tilingData_.blockFactor;
    if (GetBlockIdx() == GetBlockNum() - 1) {
        loopSizeDim0 = tilingData_.tailBlockFactor;
    }

    GlobalTensor<T> srcGlobal;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    uint32_t curLoopHandleCols = static_cast<uint32_t>(tilingData_.ubFactorDim1);
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        int64_t totalCopyCols = 0;
        int64_t colsOffset = 0;
        int64_t tensorIdx = 0;
        int64_t tensorStride = 0;
        int64_t loopOffsetInCols = i * tilingData_.ubFactorDim0;
        int64_t outTensorColsOffset = loopOffsetInCols * tilingData_.catDim1;
        uint16_t rows = static_cast<uint16_t>(tilingData_.ubFactorDim0);
        if (GetBlockIdx() == GetBlockNum() - 1 && i == loopSizeDim0 - 1) {
            rows = static_cast<uint16_t>(tilingData_.tailUbFactorDim0);
        }
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
        while (tensorIdx < tilingData_.tensorNum) {
            int64_t dim1Size = GetNonConDimSize<TILINGDATA, T>(tilingData_, tensorIdx, inputList_, desc_) *
                               tilingData_.sameShapeTensorDim1;
            if (tilingData_.isFP4Type) {
                dim1Size /= 2;
            }
            int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, tensorIdx, dim1Size);
            int64_t copyCols = dim1Size - colsOffset;
            int64_t extraCols = totalCopyCols + copyCols - tilingData_.ubFactorDim1;
            bool isSplit = extraCols >= numPerBlock_ || (extraCols > 0 && totalCopyCols == 0);
            if (extraCols <= 0 || isSplit) {
                if (isSplit) {
                    copyCols = tilingData_.ubFactorDim1 - totalCopyCols;
                }
                DataCopyExtParams copyInParam = {rows, static_cast<uint32_t>(copyCols * sizeof(T)),
                                                 static_cast<int64_t>((dim0stride - copyCols) * sizeof(T)),
                                                 static_cast<int64_t>(copyCols * sizeof(T)), 0};
                srcGlobal.SetGlobalBuffer(
                    GetTensorAddr(tensorIdx, blockOffset_ * dim0stride + loopOffsetInCols * dim0stride + colsOffset));
                DataCopyPad<T, PaddingMode::Compact>(srcLocal[tensorStride], srcGlobal, copyInParam, padParams);
                ComputeSplitDim1(dstLocal, srcLocal[tensorStride], rows, copyCols, totalCopyCols, curLoopHandleCols);
                if (isSplit) {
                    colsOffset += copyCols;
                } else {
                    colsOffset = 0;
                    tensorIdx++;
                }
                totalCopyCols += copyCols;
                tensorStride += CeilAlign(rows * copyCols, numPerBlock_);
            }
            if (tensorIdx >= tilingData_.tensorNum || totalCopyCols == tilingData_.ubFactorDim1 ||
                (extraCols > 0 && extraCols < numPerBlock_)) {
                inQueue_.FreeTensor(srcLocal);
                srcLocal = inQueue_.AllocTensor<T>();
                CopyOut(dstLocal, outTensorColsOffset, rows, totalCopyCols);
                dstLocal = outQueue_.AllocTensor<T>();
                outTensorColsOffset += totalCopyCols;
                totalCopyCols = 0;
                tensorStride = 0;
            }
        }
        inQueue_.FreeTensor(srcLocal);
        outQueue_.FreeTensor(dstLocal);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = tilingData_.blockFactor;
    if (GetBlockIdx() == GetBlockNum() - 1) {
        loopSize = tilingData_.tailBlockFactor - 1;
    }

    for (int64_t i = 0; i < loopSize; i++) {
        CopyInNoSplitDim1(i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0);
        CopyOut(i * tilingData_.ubFactorDim0 * tilingData_.catDim1, tilingData_.ubFactorDim0 * tilingData_.catDim1);
    }
    if (GetBlockIdx() == GetBlockNum() - 1) {
        CopyInNoSplitDim1(loopSize * tilingData_.ubFactorDim0, tilingData_.tailUbFactorDim0);
        CopyOut(loopSize * tilingData_.ubFactorDim0 * tilingData_.catDim1,
                tilingData_.tailUbFactorDim0 * tilingData_.catDim1);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ScatterSplitDim1(
    LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, LocalTensor<U> indexLocal, uint32_t curLoopHandleRows,
    uint32_t curLoopHandleCols, uint32_t curLoopHandleCurTensorCols)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
    uint16_t regFactorDim0 = vfLen / curLoopHandleCurTensorCols;
    uint16_t size0 = curLoopHandleRows / regFactorDim0;
    uint16_t tailRegFactorDim0 = curLoopHandleRows - size0 * regFactorDim0;
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vd2;
        AscendC::Reg::RegTensor<T> src;
        AscendC::Reg::RegTensor<T> tmp;
        AscendC::Reg::RegTensor<T> dst0;
        AscendC::Reg::RegTensor<T> dst1;
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::UnalignReg u0;

        uint32_t num = (uint32_t)(regFactorDim0 * curLoopHandleCurTensorCols);
        uint32_t tailNum = (uint32_t)(tailRegFactorDim0 * curLoopHandleCurTensorCols);
        uint32_t pnum = num;
        uint32_t tailPnum = tailNum;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailNum);

        AscendC::Reg::DataCopy(vd0, indexAddr);
        AscendC::Reg::DataCopyUnAlignPre(u0, srcAddr);
        for (uint16_t i = 0; i < size0; i++) {
            AscendC::Reg::DataCopyUnAlign(vd2, u0, srcAddr, pnum);
            AscendC::Reg::Adds(vd1, vd0, (U)(i * curLoopHandleCols * regFactorDim0), p0);
            if constexpr (sizeof(T) == 1) {
                AscendC::Reg::Interleave(dst0, dst1, vd2, tmp);
                AscendC::Reg::DataCopyScatter(dstAddr, dst0, vd1, p0);
            } else {
                AscendC::Reg::DataCopyScatter(dstAddr, vd2, vd1, p0);
            }
        }
        AscendC::Reg::DataCopyUnAlign(vd2, u0, srcAddr, tailPnum);
        AscendC::Reg::Adds(vd1, vd0, (U)(size0 * curLoopHandleCols * regFactorDim0), p1);
        if constexpr (sizeof(T) == 1) {
            AscendC::Reg::Interleave(dst0, dst1, vd2, tmp);
            AscendC::Reg::DataCopyScatter(dstAddr, dst0, vd1, p1);
        } else {
            AscendC::Reg::DataCopyScatter(dstAddr, vd2, vd1, p1);
        }
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::ComputeSplitDim1(LocalTensor<T> dstLocal,
                                                                                         LocalTensor<T> srcLocal,
                                                                                         uint32_t rows, uint32_t cols,
                                                                                         uint32_t dstOffset,
                                                                                         uint32_t curLoopHandleCols)
{
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto vWaitMTE2EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(vWaitMTE2EventID);
    WaitFlag<HardEvent::MTE2_V>(vWaitMTE2EventID);
    if (cols > SCATTER_MAX_LEN) {
        Copy(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols > COLS_SIZE_UNROLL4) {
        ScatterConcat<T, U, DIGIT_FOUR>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols > COLS_SIZE_UNROLL3) {
        ScatterConcat<T, U, DIGIT_THREE>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols > COLS_SIZE_UNROLL2) {
        ScatterConcat<T, U, DIGIT_TWO>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else if (cols >= COLS_SIZE_UNROLL1 || rows == 1) {
        ScatterConcat<T, U, 1>(dstLocal[dstOffset], srcLocal, rows, cols, curLoopHandleCols);
    } else {
        GenScatterIndex(curLoopHandleCols, dstOffset, cols, indexLocal);
        ScatterSplitDim1(dstLocal, srcLocal, indexLocal, rows, curLoopHandleCols, cols);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::CopyInNoSplitDim1(int64_t srcRowsOffset,
                                                                                          int64_t rows)
{
    GlobalTensor<T> srcGlobal;
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
    uint32_t curDim1Offset = 0;
    int64_t tensorStride = 0;
    uint32_t curLoopHandleCols = static_cast<uint32_t>(tilingData_.catDim1);
    for (int64_t i = 0; i < tilingData_.tensorNum; i++) {
        int64_t dim1 = GetNonConDimSize<TILINGDATA, T>(tilingData_, i, inputList_, desc_) *
                       tilingData_.sameShapeTensorDim1;
        if (tilingData_.isFP4Type) {
            dim1 /= 2;
        }
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        srcGlobal.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
        DataCopyExtParams copyInParam = {static_cast<uint16_t>(rows), static_cast<uint32_t>(dim1 * sizeof(T)),
                                         static_cast<int64_t>((dim0stride - dim1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad<T, PaddingMode::Compact>(srcLocal[tensorStride], srcGlobal[srcRowsOffset * dim0stride], copyInParam,
                                             padParams);
        ComputeSplitDim1(dstLocal, srcLocal[tensorStride], rows, dim1, curDim1Offset, curLoopHandleCols);
        curDim1Offset += dim1;
        tensorStride += (rows * dim1 + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    }
    inQueue_.FreeTensor(srcLocal);
    outQueue_.EnQue(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::GenScatterIndex(U curLoopHandleCols,
                                                                                        U curTensorStartCols,
                                                                                        U curLoopHandleCurTensorCols,
                                                                                        LocalTensor<U>& indexLocal)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::RegTensor<U> vd10;
        uint32_t num = vfLen;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::RegTensor<U> v0;
        using regType = typename VciTypeGet<U>::T;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::Arange(tmp, 0);
        v0 = (AscendC::Reg::RegTensor<U>&)tmp;
        AscendC::Reg::Duplicate(v1, curLoopHandleCurTensorCols, p0);
        AscendC::Reg::Div(vd2, v0, v1, p0);
        AscendC::Reg::Muls(vd6, vd2, curLoopHandleCurTensorCols, p0);
        AscendC::Reg::Sub(vd7, v0, vd6, p0);
        AscendC::Reg::Muls(vd0, vd2, curLoopHandleCols, p0);
        AscendC::Reg::Add(vd3, vd0, vd7, p0);
        AscendC::Reg::Adds(vd10, vd3, curTensorStartCols, p0);
        AscendC::Reg::DataCopy(dstAddr, vd10, p0);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::CopyOut(LocalTensor<T> dstLocal,
                                                                                int64_t dstOffset, uint16_t rows,
                                                                                int64_t cols)
{
    auto mTE3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(mTE3WaitVEventID);
    WaitFlag<HardEvent::V_MTE3>(mTE3WaitVEventID);
    DataCopyExtParams copyOutParam = {rows, static_cast<uint32_t>(cols * sizeof(T)),
                                      (tilingData_.ubFactorDim1 - cols) / numPerBlock_,
                                      static_cast<int64_t>((tilingData_.catDim1 - cols) * sizeof(T)), 0};
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
    outQueue_.FreeTensor(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignDiffShape<T, U, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t dataLen)
{
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    LocalTensor<T> dstLocal = outQueue_.DeQue<T>();
    copyOutParam.blockCount = 1;
    copyOutParam.blockLen = dataLen * sizeof(T);
    copyOutParam.dstStride = 0;
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
    outQueue_.FreeTensor(dstLocal);
}
} // namespace Concat

#endif // ONE_AXIS_CONCAT_NO_ALIGN_DIFF_SHAPE
