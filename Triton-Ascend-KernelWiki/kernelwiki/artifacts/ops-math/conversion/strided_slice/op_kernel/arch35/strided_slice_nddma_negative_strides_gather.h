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
 * \file strided_slice_nddma_negative_strides_gather.h
 * \brief negative strides implement of op strided_slice
 */
#ifndef STRIDED_SLICE_NDDMA_NEG_STRIDES_GATHER_H
#define STRIDED_SLICE_NDDMA_NEG_STRIDES_GATHER_H

#include <cstdlib>
#include <type_traits>

#include "strided_slice_base.h"

namespace StridedSlice {
using namespace AscendC;

template <typename T, typename U>
class StridedSliceNDDMAGather : public StridedSliceBase<T, U> {
public:
    __aicore__ inline StridedSliceNDDMAGather(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y, const StridedSliceNDDMATilingData* tdPtr,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

private:
    __aicore__ inline void SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo,
                                       MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfoTail);
    __aicore__ inline void SetCopyOutParams(DataCopyExtParams& copyOutParams,
                                            const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo);
    __aicore__ inline void ProcessPerBlock();
    __aicore__ inline void ProcessWithDataCopyGather(int64_t inGmAddr, int64_t outGmAddr,
                                                     const MultiCopyParams<T, NDDMA_MAX_DIMS_NEG>& nddmaParams,
                                                     const DataCopyExtParams& copyOutParam,
                                                     const LocalTensor<RangeType>& idxTensor, RangeType idxBaseOffset);
    __aicore__ inline void GatherSlice(const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo,
                                       const LocalTensor<RangeType>& idxTensor, RangeType idxBaseOffset);
    __aicore__ inline void GenGatherIndex(LocalTensor<RangeType>& idxTensor);
    __aicore__ inline void GatherInOrder(__local_mem__ RangeType* idxAddr, __local_mem__ T* inAddr,
                                         __local_mem__ T* outAddr, uint32_t outerAxis, uint32_t axis3,
                                         const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo,
                                         RangeType idxBaseOffset);
    __aicore__ inline int64_t CalcRollbackOffset(const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo);

private:
    TPipe* pipe_ = nullptr;
    const StridedSliceNDDMATilingData* tdPtr_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> idxBuf_;
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
__aicore__ inline void StridedSliceNDDMAGather<T, U>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y,
                                                           const StridedSliceNDDMATilingData* tdPtr, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    tdPtr_ = tdPtr;
    this->ParseNDDMATilingData(begin, tdPtr_, blockIdx_);
    inUbDims_ = this->inputDims_ - this->ubIndex_;
    lastDimStride_ = this->strides_[this->inputDims_ - 1];

    // pipeBuffer
    pipe_ = pipe;
    pipe_->InitBuffer(inQue_, bufferCnt_, tdPtr_->ubSizeInput);
    pipe_->InitBuffer(outQue_, bufferCnt_, this->ubSize_);
    pipe_->InitBuffer(idxBuf_, this->vlCnt_ * sizeof(T));
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAGather<T, U>::Process()
{
    // for empty tensor set realCoreNum as 0, do nothing
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    ProcessPerBlock();
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAGather<T, U>::GatherInOrder(
    __local_mem__ RangeType* idxAddr, __local_mem__ T* inAddr, __local_mem__ T* outAddr, uint32_t outerAxis,
    uint32_t axis3, const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo, RangeType idxBaseOffset)
{
    uint32_t axis3BA = Ops::Base::CeilAlign(axis3, elemPerBlock_);
    uint32_t maskBeg = axis3;
    uint32_t rVLCnt = this->vlCnt_;
    if constexpr (sizeof(T) == 1) {
        rVLCnt /= sizeof(int16_t);
        maskBeg = axis3 * sizeof(int16_t);
    }
    uint32_t maskValue = maskBeg;
    uint16_t axis3LpCnt = Ops::Base::CeilDiv(axis3, rVLCnt);
    uint16_t oAxis = uint16_t(outerAxis);
    RangeType axis3Offset = (lastDimStride_ < 0) ? (int32_t(rVLCnt) * -1) : int32_t(rVLCnt);
    RangeType axis3BAIn = axis3BA;

    __VEC_SCOPE__
    {
        Reg::RegTensor<RangeType> regIdx;
        Reg::RegTensor<RangeType> regIdxBK;
        Reg::RegTensor<T> regData;
        Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskData;

        Reg::DataCopy(regIdx, idxAddr);
        Reg::Adds(regIdx, regIdx, idxBaseOffset, maskIdx);

        for (uint16_t axis3LpIdx = 0; axis3LpIdx < axis3LpCnt; axis3LpIdx++) {
            maskData = Reg::UpdateMask<T>(maskValue);
            Reg::Copy(regIdxBK, regIdx);
            for (uint16_t oIdx = 0; oIdx < oAxis; oIdx++) {
                Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, inAddr, (Reg::RegTensor<IdxType>&)regIdx,
                                    maskData);
                if constexpr (sizeof(T) != 1) {
                    Reg::DataCopy(outAddr + oIdx * axis3BA + axis3LpIdx * rVLCnt, regData, maskData);
                } else {
                    __local_mem__ CastType* outAddrB16 = reinterpret_cast<__local_mem__ CastType*>(
                        outAddr + oIdx * axis3BA + axis3LpIdx * rVLCnt);
                    Reg::DataCopy<CastType, Reg::StoreDist::DIST_PACK_B16>(
                        outAddrB16, (Reg::RegTensor<CastType>&)regData, maskData);
                }
                Reg::Adds(regIdx, regIdx, axis3BAIn, maskIdx);
            }
            Reg::Copy(regIdx, regIdxBK);
            Reg::Adds(regIdx, regIdx, axis3Offset, maskIdx);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAGather<T, U>::GatherSlice(const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo,
                                                                  const LocalTensor<RangeType>& idxTensor,
                                                                  RangeType idxBaseOffset)
{
    // (axis0, axis1, axis2, axis3BA)
    uint32_t axis0 = loopInfo.loopSize[DIMS_3];
    uint32_t axis1 = loopInfo.loopSize[DIMS_2];
    uint32_t axis2 = loopInfo.loopSize[DIMS_1];
    uint32_t axis3 = loopInfo.loopSize[0];
    uint32_t axis3BA = Ops::Base::CeilAlign(axis3, elemPerBlock_);

    auto outTensor = outQue_.AllocTensor<T>();
    auto inTensor = inQue_.DeQue<T>();

    __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
    __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr();
    GatherInOrder(idxAddr, inAddr, outAddr, axis0 * axis1 * axis2, axis3, loopInfo, idxBaseOffset);
    inQue_.FreeTensor(inTensor);

    this->CalcReorderAxisInfo(axis0, axis1, axis2, axis3BA);
    this->Reorder4Output(outAddr);
    outQue_.EnQue(outTensor);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAGather<T, U>::ProcessWithDataCopyGather(
    int64_t inGmAddr, int64_t outGmAddr, const MultiCopyParams<T, NDDMA_MAX_DIMS_NEG>& nddmaParams,
    const DataCopyExtParams& copyOutParam, const LocalTensor<RangeType>& idxTensor, RangeType idxBaseOffset)
{
    auto inTensor = inQue_.AllocTensor<T>();
    DataCopy<T, NDDMA_MAX_DIMS_NEG, nddmaConfig_>(inTensor, inputGM_[inGmAddr], nddmaParams);
    inQue_.EnQue(inTensor);

    GatherSlice(nddmaParams.loopInfo, idxTensor, idxBaseOffset);

    auto outTensor = outQue_.DeQue<T>();
    DataCopyPad(outputGM_[outGmAddr], outTensor, copyOutParam);
    outQue_.FreeTensor(outTensor);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceNDDMAGather<T, U>::GenGatherIndex(LocalTensor<RangeType>& idxTensor)
{
    __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
    int32_t begIndex = this->nddmaLoopSize_[NDDMA_MAX_DIMS_NEG - 1] - 1; // last axis num in ub
    // last axis stride is -1 in ub
    __VEC_SCOPE__
    {
        Reg::MaskReg mask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
        Reg::RegTensor<RangeType> indexReg;
        Reg::RegTensor<RangeType> tmpReg;
        Reg::Arange<RangeType, Reg::IndexOrder::INCREASE_ORDER>(indexReg, 0);
        Reg::Duplicate(tmpReg, begIndex);
        Reg::Sub(indexReg, tmpReg, indexReg, mask);
        Reg::DataCopy(idxAddr, indexReg, mask);
    }
}

template <typename T, typename U>
__aicore__ inline int64_t StridedSliceNDDMAGather<T, U>::CalcRollbackOffset(
    const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo)
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
__aicore__ inline void StridedSliceNDDMAGather<T, U>::SetLoopInfo(MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo,
                                                                  MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfoTail)
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
__aicore__ inline void StridedSliceNDDMAGather<T, U>::SetCopyOutParams(
    DataCopyExtParams& copyOutParams, const MultiCopyLoopInfo<NDDMA_MAX_DIMS_NEG>& loopInfo)
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
__aicore__ inline void StridedSliceNDDMAGather<T, U>::ProcessPerBlock()
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
    int64_t ubSplitLoopsNum = 0;  // ub切分轴上的循环次数
    int64_t ubOuterLoopsNum = 0;  // ub切分轴之外的循环次数
    int64_t rowsOffsetOutput = 0; // 当前核处理的output shape中的起始行数
    this->CalcProcessLoopsNum(ubOuterLoopsNum, ubSplitLoopsNum, blockIdx_);
    this->GetProcessRowsOffsetAll(rowsOffsetOutput, blockIdx_);

    auto idxTensor = idxBuf_.Get<RangeType>();
    GenGatherIndex(idxTensor);

    int64_t negativeStrideOffset = CalcRollbackOffset(paramsMain.loopInfo);
    int64_t negativeStrideOffsetT = CalcRollbackOffset(paramsTail.loopInfo);

    for (int64_t idx = 0; idx < ubOuterLoopsNum; idx++) {
        inputGmAddr = this->GetInputGmAddrAll(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        outputGmAddr = this->GetOutputGmAddrAll(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        for (int64_t loops = 0; loops < ubSplitLoopsNum; loops++) {
            ProcessWithDataCopyGather(inputGmAddr + loops * this->ubInLoopSteps_ - negativeStrideOffset,
                                      outputGmAddr + loops * this->ubOutLoopSteps_, paramsMain, copyOutParamsMain,
                                      idxTensor, 0);
        }
        if (this->ubTailFactor_ > 0) {
            // gather idx starts with begin[-1],  idxBaseOffset < 0 if lastDimStride_ < 0 when ub split last axis
            RangeType idxBaseOffset = (lastDimStride_ < 0) ? ((int32_t)paramsTail.loopInfo.loopSize[0] -
                                                              (int32_t)paramsMain.loopInfo.loopSize[0]) :
                                                             0;
            ProcessWithDataCopyGather(inputGmAddr + ubSplitLoopsNum * this->ubInLoopSteps_ - negativeStrideOffsetT,
                                      outputGmAddr + ubSplitLoopsNum * this->ubOutLoopSteps_, paramsTail,
                                      copyOutParamsTail, idxTensor, idxBaseOffset);
        }
    }
}

} // namespace StridedSlice

#endif // STRIDED_SLICE_NDDMA_NEG_STRIDES_GATHER_H
