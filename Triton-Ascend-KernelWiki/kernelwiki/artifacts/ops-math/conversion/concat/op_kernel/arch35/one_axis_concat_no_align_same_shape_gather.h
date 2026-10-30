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
 * \file one_axis_concat_no_align_same_shape_gather.h
 * \brief one axis concat no align same shape gather
 */

#ifndef ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_GATHER
#define ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_GATHER

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename T, typename U, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatNoAlignGather {
public:
    __aicore__ inline OneAxisConcatNoAlignGather(const TILINGDATA& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t startIdx, int64_t endIdx);
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t rows, int64_t cols);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t dataLen);
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void GenGatherIndex(uint32_t cols, uint32_t tensorStride, uint32_t inputCols,
                                          LocalTensor<U>& indexLocal);
    __aicore__ inline void ComputeNoSplitDim1(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols);
    __aicore__ inline void ComputeNoSplitDim1B8(uint32_t rows, uint32_t cols, uint32_t tensorStride,
                                                uint32_t inputCols);
    __aicore__ inline void ComputeNoSplitDim1Norm(uint32_t rows, uint32_t cols, uint32_t tensorStride,
                                                  uint32_t inputCols);
    template <const bool needColAlign = true>
    __aicore__ inline void ComputeSplitDim1(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols);
    template <const bool needColAlign = true>
    __aicore__ inline void ComputeSplitDim1B8(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols);
    template <const bool needColAlign = true>
    __aicore__ inline void ComputeSplitDim1Norm(uint32_t rows, uint32_t cols, uint32_t tensorStride,
                                                uint32_t inputCols);
    __aicore__ inline void ProcessPerLoop(int64_t srcRowsOffset, int64_t rows, const SplitLoopParam& params);

private:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;
    const TILINGDATA& tilingData_;
    int64_t blockOffset_ = 0;
    int64_t blockIdx_ = 0;

    int64_t numPerBlock_ = BYTES_PER_BLOCK / sizeof(T);
    int64_t colsUsedCoreNum_ = 1;
    int64_t rowsUsedCoreNum_ = 1;

    DataCopyExtParams copyInParam_ = {0, 0, 0, 0, 0};

    int64_t startTensorIdx_ = 0;
    int64_t startTensorOffset_ = 0;
    int64_t endTensorIdx_ = 0;
    ListTensorDesc inputList_;

    uint32_t vfLen_ = GetVRegSize() / sizeof(U);
};

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value) {
        int64_t colsUsedCoreNum = GetBlockNum() / tilingData_.uoDim0;
        if (blockIdx_ % colsUsedCoreNum != 0) {
            startTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_ - 1];
            startTensorOffset_ = tilingData_.arrays.endTensorOffset[blockIdx_ - 1];
        }
        endTensorIdx_ = tilingData_.arrays.endTensorIdx[blockIdx_];
    }
    if (startTensorOffset_ == tilingData_.sameShapeTensorDim1) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value) {
        blockOffset_ = blockIdx_ * tilingData_.ubFactorDim0 * tilingData_.blockFactor;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_.catDim1);
    } else {
        rowsUsedCoreNum_ = tilingData_.uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        blockOffset_ = tilingData_.ubFactorDim0 * (blockIdx_ / colsUsedCoreNum_);
        int64_t colOffset = (blockIdx_ % colsUsedCoreNum_) * tilingData_.blockFactor * tilingData_.ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_.catDim1 + colOffset);
    }

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, tilingData_.bufferSize * sizeof(T));
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.bufferSize * sizeof(T));
    pipe_.InitBuffer(indexBuf_, INDEX_SIZE);

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatNoAlignGather<T, U, TILINGDATA>::GetTensorAddr(uint32_t index, int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::Process()
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

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessBlockSplitDim1()
{
    int64_t blockIdxInCol = blockIdx_ % colsUsedCoreNum_;
    int64_t blockIdxInRow = blockIdx_ / colsUsedCoreNum_;
    int64_t copyRowsNum = tilingData_.ubFactorDim0;

    if (blockIdxInRow == rowsUsedCoreNum_ - 1) {
        copyRowsNum = tilingData_.tailUbFactorDim0;
    }
    int64_t handleTensorNum = (endTensorIdx_ - startTensorIdx_) + 1;
    int64_t perLoopDealTensorNum = (tilingData_.bufferSize - copyRowsNum * numPerBlock_) /
                                   (copyRowsNum * tilingData_.sameShapeTensorDim1 + numPerBlock_);
    int64_t tensorStride = (copyRowsNum * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) / numPerBlock_ *
                           numPerBlock_;
    int64_t dim1Loop = handleTensorNum / perLoopDealTensorNum;
    int64_t tailLoopDealTensorNum = handleTensorNum - dim1Loop * perLoopDealTensorNum;
    if (tailLoopDealTensorNum == 0) {
        dim1Loop -= 1;
        tailLoopDealTensorNum = perLoopDealTensorNum;
    }
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    GenGatherIndex(perLoopDealTensorNum * tilingData_.sameShapeTensorDim1, tensorStride,
                   tilingData_.sameShapeTensorDim1, indexLocal);
    SplitLoopParam param = {startTensorIdx_,       handleTensorNum, perLoopDealTensorNum,
                            tailLoopDealTensorNum, dim1Loop,        tensorStride};
    ProcessPerLoop(0, copyRowsNum, param);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessPerLoop(int64_t srcRowsOffset, int64_t rows,
                                                                                    const SplitLoopParam& params)
{
    for (int64_t i = 0; i < params.loopSize; i++) {
        CopyInNoSplitDim1(srcRowsOffset, rows, params.startTensorIdx + i * params.perLoopDealTensorNum,
                          params.startTensorIdx + (i + 1) * params.perLoopDealTensorNum);
        ComputeSplitDim1(rows, params.perLoopDealTensorNum * tilingData_.sameShapeTensorDim1, params.tensorStride,
                         tilingData_.sameShapeTensorDim1);
        CopyOut(srcRowsOffset * tilingData_.catDim1 + i * params.perLoopDealTensorNum * tilingData_.sameShapeTensorDim1,
                rows, params.perLoopDealTensorNum * tilingData_.sameShapeTensorDim1);
    }
    CopyInNoSplitDim1(srcRowsOffset, rows, params.startTensorIdx + params.loopSize * params.perLoopDealTensorNum,
                      params.startTensorIdx + params.handleTensorNum);
    ComputeSplitDim1(rows, params.tailLoopDealTensorNum * tilingData_.sameShapeTensorDim1, params.tensorStride,
                     tilingData_.sameShapeTensorDim1);
    CopyOut(srcRowsOffset * tilingData_.catDim1 +
                params.loopSize * params.perLoopDealTensorNum * tilingData_.sameShapeTensorDim1,
            rows, params.tailLoopDealTensorNum * tilingData_.sameShapeTensorDim1);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = tilingData_.blockFactor;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSizeDim0 = tilingData_.tailBlockFactor - 1;
    }

    int64_t perLoopDealTensorNum = (tilingData_.bufferSize - tilingData_.ubFactorDim0 * numPerBlock_) /
                                   (tilingData_.ubFactorDim0 * tilingData_.sameShapeTensorDim1 + numPerBlock_);
    int64_t tensorStride = (tilingData_.ubFactorDim0 * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) /
                           numPerBlock_ * numPerBlock_;
    int64_t dim1Loop = tilingData_.tensorNum / perLoopDealTensorNum;
    int64_t tailLoopDealTensorNum = tilingData_.tensorNum - dim1Loop * perLoopDealTensorNum;
    if (tailLoopDealTensorNum == 0) {
        dim1Loop -= 1;
        tailLoopDealTensorNum = perLoopDealTensorNum;
    }

    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    if (loopSizeDim0 > 0) {
        GenGatherIndex(perLoopDealTensorNum * tilingData_.sameShapeTensorDim1, tensorStride,
                       tilingData_.sameShapeTensorDim1, indexLocal);
    }
    SplitLoopParam param = {0,        tilingData_.tensorNum, perLoopDealTensorNum, tailLoopDealTensorNum,
                            dim1Loop, tensorStride};
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        ProcessPerLoop(i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0, param);
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        tensorStride = (tilingData_.tailUbFactorDim0 * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) /
                       numPerBlock_ * numPerBlock_;
        GenGatherIndex(perLoopDealTensorNum * tilingData_.sameShapeTensorDim1, tensorStride,
                       tilingData_.sameShapeTensorDim1, indexLocal);
        param.tensorStride = tensorStride;
        ProcessPerLoop(loopSizeDim0 * tilingData_.ubFactorDim0, tilingData_.tailUbFactorDim0, param);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = (blockIdx_ == GetBlockNum() - 1) ? tilingData_.tailBlockFactor - 1 : tilingData_.blockFactor;
    int64_t tensorStride = (tilingData_.ubFactorDim0 * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) /
                           numPerBlock_ * numPerBlock_;
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    if (loopSize > 0) {
        GenGatherIndex(tilingData_.catDim1, tensorStride, tilingData_.sameShapeTensorDim1, indexLocal);
    }
    if (tilingData_.catDim1 * sizeof(U) <= GetVRegSize()) {
        for (int64_t i = 0; i < loopSize; i++) {
            CopyInNoSplitDim1(i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0, 0, tilingData_.tensorNum);
            ComputeNoSplitDim1(tilingData_.ubFactorDim0, tilingData_.catDim1, tensorStride,
                               tilingData_.sameShapeTensorDim1);
            CopyOut(i * tilingData_.ubFactorDim0 * tilingData_.catDim1, tilingData_.ubFactorDim0 * tilingData_.catDim1);
        }
        tensorStride = (tilingData_.tailUbFactorDim0 * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) /
                       numPerBlock_ * numPerBlock_;
        if (blockIdx_ == GetBlockNum() - 1) {
            GenGatherIndex(tilingData_.catDim1, tensorStride, tilingData_.sameShapeTensorDim1, indexLocal);
            CopyInNoSplitDim1(loopSize * tilingData_.ubFactorDim0, tilingData_.tailUbFactorDim0, 0,
                              tilingData_.tensorNum);
            ComputeNoSplitDim1(tilingData_.tailUbFactorDim0, tilingData_.catDim1, tensorStride,
                               tilingData_.sameShapeTensorDim1);
            CopyOut(loopSize * tilingData_.ubFactorDim0 * tilingData_.catDim1,
                    tilingData_.tailUbFactorDim0 * tilingData_.catDim1);
        }
    } else {
        for (int64_t i = 0; i < loopSize; i++) {
            CopyInNoSplitDim1(i * tilingData_.ubFactorDim0, tilingData_.ubFactorDim0, 0, tilingData_.tensorNum);
            ComputeSplitDim1<false>(tilingData_.ubFactorDim0, tilingData_.catDim1, tensorStride,
                                    tilingData_.sameShapeTensorDim1);
            CopyOut(i * tilingData_.ubFactorDim0 * tilingData_.catDim1, tilingData_.ubFactorDim0 * tilingData_.catDim1);
        }
        tensorStride = (tilingData_.sameShapeTensorDim1 * tilingData_.tailUbFactorDim0 + numPerBlock_ - 1) /
                       numPerBlock_ * numPerBlock_;
        if (blockIdx_ == GetBlockNum() - 1) {
            GenGatherIndex(tilingData_.catDim1, tensorStride, tilingData_.sameShapeTensorDim1, indexLocal);
            CopyInNoSplitDim1(tilingData_.ubFactorDim0 * loopSize, tilingData_.tailUbFactorDim0, 0,
                              tilingData_.tensorNum);
            ComputeSplitDim1<false>(tilingData_.tailUbFactorDim0, tilingData_.catDim1, tensorStride,
                                    tilingData_.sameShapeTensorDim1);
            CopyOut(loopSize * tilingData_.ubFactorDim0 * tilingData_.catDim1,
                    tilingData_.tailUbFactorDim0 * tilingData_.catDim1);
        }
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::GenGatherIndex(uint32_t cols,
                                                                                    uint32_t tensorStride,
                                                                                    uint32_t inputCols,
                                                                                    LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::T;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> v3;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::RegTensor<U> vd8;
        AscendC::Reg::RegTensor<U> vd9;
        AscendC::Reg::RegTensor<U> vd10;
        uint32_t num = vfLen_;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::Arange(tmp, 0);
        if constexpr (sizeof(U) == sizeof(uint32_t) || sizeof(U) == sizeof(uint16_t)) {
            v0 = (AscendC::Reg::RegTensor<U>&)tmp;
        }
        AscendC::Reg::Duplicate(v1, (U)inputCols, p0);
        AscendC::Reg::Duplicate(v2, (U)tensorStride, p0);
        AscendC::Reg::Duplicate(v3, (U)cols, p0);

        AscendC::Reg::Div(vd2, v0, v3, p0);
        AscendC::Reg::Mul(vd6, vd2, v3, p0);
        AscendC::Reg::Sub(vd7, v0, vd6, p0);

        AscendC::Reg::Div(vd0, vd7, v1, p0);
        AscendC::Reg::Mul(vd1, vd0, v2, p0);

        AscendC::Reg::Div(vd3, vd7, v1, p0);
        AscendC::Reg::Mul(vd4, vd3, v1, p0);
        AscendC::Reg::Sub(vd5, vd7, vd4, p0);

        AscendC::Reg::Mul(vd8, vd2, v1, p0);

        AscendC::Reg::Add(vd9, vd1, vd5, p0);
        AscendC::Reg::Add(vd10, vd8, vd9, p0);
        AscendC::Reg::DataCopy(dstAddr, vd10, p0);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeNoSplitDim1(uint32_t rows, uint32_t cols,
                                                                                        uint32_t tensorStride,
                                                                                        uint32_t inputCols)
{
    if constexpr (sizeof(T) == 1) {
        ComputeNoSplitDim1B8(rows, cols, tensorStride, inputCols);
    } else {
        ComputeNoSplitDim1Norm(rows, cols, tensorStride, inputCols);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeNoSplitDim1Norm(uint32_t rows,
                                                                                            uint32_t cols,
                                                                                            uint32_t tensorStride,
                                                                                            uint32_t inputCols)
{
    LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
    // tiling可保证每个tensor元素个数不超过uint16最大值
    uint16_t regFactorDim0 = vfLen_ / cols;
    uint16_t size0 = rows / regFactorDim0;
    uint16_t tailRegFactorDim0 = rows - size0 * regFactorDim0;
    uint32_t stride = inputCols * regFactorDim0;
    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<T> vd2;
        AscendC::Reg::UnalignReg u0;

        uint32_t num = static_cast<uint32_t>(regFactorDim0 * cols);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim0 * cols);
        uint32_t main = num;
        uint32_t tail = tailNum;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailNum);

        AscendC::Reg::DataCopy(vd0, indexAddr);
        for (uint16_t i = 0; i < size0; i++) {
            AscendC::Reg::Adds(vd1, vd0, (U)(i * stride), p0);
            AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p0);
            AscendC::Reg::DataCopyUnAlign(dstAddr, vd2, u0, main);
            AscendC::Reg::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
        AscendC::Reg::Adds(vd1, vd0, (U)(size0 * stride), p1);
        AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p1);
        AscendC::Reg::DataCopyUnAlign(dstAddr, vd2, u0, tail);
        AscendC::Reg::DataCopyUnAlignPost(dstAddr, u0, 0);
    }

    inQueue_.FreeTensor(srcLocal);
    outQueue_.EnQue(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeNoSplitDim1B8(uint32_t rows, uint32_t cols,
                                                                                          uint32_t tensorStride,
                                                                                          uint32_t inputCols)
{
    LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
    // tiling可保证每个tensor元素个数不超过uint16最大值
    uint16_t regFactorDim0 = vfLen_ / cols;
    uint16_t size0 = rows / regFactorDim0;
    uint16_t tailRegFactorDim0 = rows - size0 * regFactorDim0;
    uint32_t stride = inputCols * regFactorDim0;
    auto dstAddr = reinterpret_cast<__ubuf__ uint8_t*>(dstLocal.GetPhyAddr());
    auto srcAddr = reinterpret_cast<__ubuf__ uint8_t*>(srcLocal.GetPhyAddr());
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> vd0;
        AscendC::Reg::RegTensor<uint16_t> vd1;
        AscendC::Reg::RegTensor<uint16_t> vd2;
        AscendC::Reg::RegTensor<uint8_t> vd3;
        AscendC::Reg::RegTensor<uint16_t> vd4;
        AscendC::Reg::RegTensor<uint8_t> vd5;

        AscendC::Reg::UnalignReg u0;

        uint32_t num = static_cast<uint32_t>(regFactorDim0 * cols);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim0 * cols);
        uint32_t main = num;
        uint32_t tail = tailNum;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailNum);

        AscendC::Reg::DataCopy(vd0, indexAddr);
        for (uint16_t i = 0; i < size0; i++) {
            AscendC::Reg::Adds(vd1, vd0, (U)(i * stride), p0);
            AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p0);
            AscendC::Reg::Pack(vd3, vd2);
            AscendC::Reg::DataCopyUnAlign(dstAddr, vd3, u0, main);
            AscendC::Reg::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
        AscendC::Reg::Adds(vd1, vd0, (U)(size0 * stride), p1);
        AscendC::Reg::DataCopyGather(vd4, srcAddr, vd1, p1);
        AscendC::Reg::Pack(vd5, vd4);
        AscendC::Reg::DataCopyUnAlign(dstAddr, vd5, u0, tail);
        AscendC::Reg::DataCopyUnAlignPost(dstAddr, u0, 0);
    }

    inQueue_.FreeTensor(srcLocal);
    outQueue_.EnQue(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
template <const bool needColAlign>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeSplitDim1(uint32_t rows, uint32_t cols,
                                                                                      uint32_t tensorStride,
                                                                                      uint32_t inputCols)
{
    if constexpr (sizeof(T) == 1) {
        ComputeSplitDim1B8<needColAlign>(rows, cols, tensorStride, inputCols);
    } else {
        ComputeSplitDim1Norm<needColAlign>(rows, cols, tensorStride, inputCols);
    }
}

template <typename T, typename U, typename TILINGDATA>
template <const bool needColAlign>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeSplitDim1Norm(uint32_t rows, uint32_t cols,
                                                                                          uint32_t tensorStride,
                                                                                          uint32_t inputCols)
{
    LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
    // tiling可保证每个tensor元素个数不超过uint16最大值
    uint16_t size0 = rows;
    uint16_t everyRegCalcTensorNum = vfLen_ / inputCols;
    uint16_t regFactorDim1 = everyRegCalcTensorNum * inputCols;
    uint16_t size1 = cols / regFactorDim1;
    uint16_t tailRegFactorDim1 = cols - size1 * regFactorDim1;

    uint32_t padCols = cols;
    if constexpr (needColAlign) {
        padCols = (cols + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    }
    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __ubuf__ T* curDstAddr = dstAddr;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<T> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<T> vd4;
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::UnalignReg u1;

        uint32_t num = static_cast<uint32_t>(regFactorDim1);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim1);
        uint32_t pnum = num;
        uint32_t tailPnum = tailNum;

        uint32_t main = num;
        uint32_t tail = tailNum;

        AscendC::Reg::DataCopy(vd0, indexAddr);
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(pnum);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailPnum);
        uint32_t stride = everyRegCalcTensorNum * tensorStride;
        uint32_t rowStride = 0;
        for (uint16_t i = 0; i < size0; i++) {
            main = num;
            curDstAddr = dstAddr + i * padCols;
            rowStride = i * inputCols;
            for (uint16_t j = 0; j < size1; j++) {
                AscendC::Reg::Adds(vd1, vd0, (U)(rowStride + j * stride), p0);
                AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p0);
                AscendC::Reg::DataCopyUnAlign(curDstAddr, vd2, u0, main);
                AscendC::Reg::DataCopyUnAlignPost(curDstAddr, u0, 0);
            }
            tail = tailNum;
            AscendC::Reg::Adds(vd3, vd0, (U)(rowStride + size1 * stride), p1);
            AscendC::Reg::DataCopyGather(vd4, srcAddr, vd3, p1);
            AscendC::Reg::DataCopyUnAlign(curDstAddr, vd4, u1, tail);
            AscendC::Reg::DataCopyUnAlignPost(curDstAddr, u1, 0);
        }
    }

    inQueue_.FreeTensor(srcLocal);
    outQueue_.EnQue(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
template <const bool needColAlign>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeSplitDim1B8(uint32_t rows, uint32_t cols,
                                                                                        uint32_t tensorStride,
                                                                                        uint32_t inputCols)
{
    LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
    LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
    // tiling可保证每个tensor元素个数不超过uint16最大值
    uint16_t everyRegCalcTensorNum = vfLen_ / inputCols;
    uint16_t size0 = rows;
    uint16_t regFactorDim1 = everyRegCalcTensorNum * inputCols;
    uint16_t size1 = cols / regFactorDim1;
    uint16_t tailRegFactorDim1 = cols - size1 * regFactorDim1;

    uint32_t padCols = cols;
    if constexpr (needColAlign) {
        padCols = (cols + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    }
    auto dstAddr = reinterpret_cast<__ubuf__ uint8_t*>(dstLocal.GetPhyAddr());
    auto srcAddr = reinterpret_cast<__ubuf__ uint8_t*>(srcLocal.GetPhyAddr());
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = reinterpret_cast<__ubuf__ U*>(indexLocal.GetPhyAddr());
    __ubuf__ uint8_t* curDstAddr;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> vd0;
        AscendC::Reg::RegTensor<uint16_t> vd1;
        AscendC::Reg::RegTensor<uint16_t> vd2;
        AscendC::Reg::RegTensor<uint8_t> vd3;
        AscendC::Reg::RegTensor<uint16_t> vd4;
        AscendC::Reg::RegTensor<uint16_t> vd5;
        AscendC::Reg::RegTensor<uint8_t> vd6;
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::UnalignReg u1;

        uint32_t num = static_cast<uint32_t>(regFactorDim1);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim1);
        uint32_t tailPnum = tailNum;
        uint32_t pnum = num;

        uint32_t main = num;
        uint32_t tail = tailNum;

        AscendC::Reg::DataCopy(vd0, indexAddr);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailPnum);
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(pnum);
        uint32_t stride = everyRegCalcTensorNum * tensorStride;
        uint32_t rowStride = 0;
        for (uint16_t i = 0; i < size0; i++) {
            main = num;
            curDstAddr = dstAddr + i * padCols;
            rowStride = inputCols * i;
            for (uint16_t j = 0; j < size1; j++) {
                AscendC::Reg::Adds(vd1, vd0, (U)(rowStride + j * stride), p0);
                AscendC::Reg::DataCopyGather(vd2, srcAddr, vd1, p0);
                AscendC::Reg::Pack(vd3, vd2);
                AscendC::Reg::DataCopyUnAlign(curDstAddr, vd3, u0, main);
                AscendC::Reg::DataCopyUnAlignPost(curDstAddr, u0, 0);
            }
            tail = tailNum;
            AscendC::Reg::Adds(vd4, vd0, (U)(rowStride + size1 * stride), p1);
            AscendC::Reg::DataCopyGather(vd5, srcAddr, vd4, p1);
            AscendC::Reg::Pack(vd6, vd5);
            AscendC::Reg::DataCopyUnAlign(curDstAddr, vd6, u1, tail);
            AscendC::Reg::DataCopyUnAlignPost(curDstAddr, u1, 0);
        }
    }

    inQueue_.FreeTensor(srcLocal);
    outQueue_.EnQue(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::CopyInNoSplitDim1(int64_t srcRowsOffset,
                                                                                       int64_t rows, int64_t startIdx,
                                                                                       int64_t endIdx)
{
    int64_t curDim1Offset = 0;
    int64_t tensorStride = (rows * tilingData_.sameShapeTensorDim1 + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParam = {false, 0, 0, 0};
    for (int64_t tensorIdx = startIdx; tensorIdx < endIdx; tensorIdx++) {
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, tensorIdx, tilingData_.sameShapeTensorDim1);
        if (tilingData_.isNonContiguous) {
            SetCopyInparam(rows, tilingData_.sameShapeTensorDim1, dim0stride - tilingData_.sameShapeTensorDim1, 0);
        } else {
            SetCopyInparam(1, rows * tilingData_.sameShapeTensorDim1, 0, 0);
        }
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(tensorIdx, blockOffset_ * dim0stride));
        DataCopyPad<T, PaddingMode::Compact>(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride],
                                             copyInParam_, padParam);
        curDim1Offset += tensorStride;
    }
    inQueue_.EnQue(srcLocal);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::SetCopyInparam(int64_t rows, int64_t cols,
                                                                                    int64_t srcStride,
                                                                                    int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * sizeof(T);
    copyInParam_.srcStride = srcStride * sizeof(T);
    copyInParam_.dstStride = dstStride * sizeof(T);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t rows,
                                                                             int64_t cols)
{
    LocalTensor<T> dstLocal = outQueue_.DeQue<T>();
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = rows;
    copyOutParam.blockLen = cols * sizeof(T);
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = (tilingData_.catDim1 - cols) * sizeof(T);
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
    outQueue_.FreeTensor(dstLocal);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t dataLen)
{
    LocalTensor<T> dstLocal = outQueue_.DeQue<T>();
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = 1;
    copyOutParam.blockLen = dataLen * sizeof(T);
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
    outQueue_.FreeTensor(dstLocal);
}

} // namespace Concat

#endif // ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_GATHER
