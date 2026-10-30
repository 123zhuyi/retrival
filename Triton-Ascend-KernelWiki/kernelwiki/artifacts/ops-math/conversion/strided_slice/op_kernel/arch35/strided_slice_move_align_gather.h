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
 * \file strided_slice_move_align_gather.h
 * \brief negative strides implement of op strided_slice
 */
#ifndef STRIDED_SLICE_MOVE_ALIGN_GATHER_H
#define STRIDED_SLICE_MOVE_ALIGN_GATHER_H

#include <cstdlib>
#include <type_traits>

#include "strided_slice_base.h"

namespace StridedSlice {
using namespace AscendC;

template <typename T, typename U>
class StridedSliceMoveAlignGather : public StridedSliceBase<T, U> {
public:
    __aicore__ inline StridedSliceMoveAlignGather(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y, const StridedSliceMAGatherTilingData* tdPtr,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

private:
    __aicore__ inline void ProcessPerBlock(const DataCopyExtParams& copyOutParamsMain,
                                           const DataCopyExtParams& copyOutParamsTail);
    __aicore__ inline void ParseCopyInTilingData();
    __aicore__ inline void ParseLoopModeAndMoveAlignParams(LoopModeParams& loopMode, DataCopyExtParams& extParams);
    __aicore__ inline void SetCopyOutAlignParams(DataCopyExtParams& copyOutParams, const DataCopyExtParams& copyInParam,
                                                 const LoopModeParams& loopMode);
    __aicore__ inline void ProcessWithDataCopyGather(int64_t inGmAddr, int64_t outGmAddr,
                                                     const LoopModeParams& loopMode,
                                                     const DataCopyExtParams& copyInParam,
                                                     const DataCopyExtParams& copyOutParam,
                                                     const LocalTensor<RangeType>& idxTensor);
    __aicore__ inline void GatherSlice(const LoopModeParams& loopMode, const DataCopyExtParams& copyInParam,
                                       const LocalTensor<RangeType>& idxTensor);
    __aicore__ inline void GenGatherIndex(LocalTensor<RangeType>& idxTensor);
    __aicore__ inline void GatherInOrder(__local_mem__ RangeType* idxAddr, __local_mem__ T* inAddr,
                                         __local_mem__ T* outAddr, uint32_t outerAxis, uint32_t axis3,
                                         const DataCopyExtParams& copyInParam);
    __aicore__ inline int64_t CalcRollbackOffset(const LoopModeParams& loopMode, const DataCopyExtParams& copyInParam);

private:
    TPipe* pipe_ = nullptr;
    const StridedSliceMAGatherTilingData* tdPtr_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> idxBuf_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;
    uint8_t bufferCnt_ = 2; // enable db
    uint32_t elemPerBlock_ = BLOCK_SIZE_BYTE / sizeof(T);
    int64_t inUbDims_ = 0;
    int64_t lastDimStride_ = 1;

    LoopModeParams loopMode_;
    LoopModeParams loopModeTail_;
    DataCopyExtParams copyInParam_;
    DataCopyExtParams copyInParamTail_;
};

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::Init(GM_ADDR x, GM_ADDR begin, GM_ADDR y,
                                                               const StridedSliceMAGatherTilingData* tdPtr, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    tdPtr_ = tdPtr;
    this->ParseBaseTilingDataV2(begin, &(tdPtr_->stridedSliceBaseTilingData), blockIdx_);
    inUbDims_ = this->inputDims_ - this->ubIndex_;
    lastDimStride_ = this->strides_[this->inputDims_ - 1];
    // gm -> ub move_align_v2 params
    ParseCopyInTilingData();

    // pipeBuffer
    pipe_ = pipe;
    pipe_->InitBuffer(inQue_, bufferCnt_, tdPtr_->ubSizeInput);
    pipe_->InitBuffer(outQue_, bufferCnt_, this->ubSize_);
    pipe_->InitBuffer(idxBuf_, this->vlCnt_ * sizeof(T));
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::Process()
{
    // for empty tensor set realCoreNum as 0, do nothing
    if (blockIdx_ >= this->realCoreNum_) {
        return;
    }

    // ub -> gm
    DataCopyExtParams copyOutParamsMain;
    DataCopyExtParams copyOutParamsTail;
    SetCopyOutAlignParams(copyOutParamsMain, copyInParam_, loopMode_);
    if (this->ubTailFactor_ > 0) {
        SetCopyOutAlignParams(copyOutParamsTail, copyInParamTail_, loopModeTail_);
    }

    ProcessPerBlock(copyOutParamsMain, copyOutParamsTail);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::ParseLoopModeAndMoveAlignParams(LoopModeParams& loopMode,
                                                                                          DataCopyExtParams& extParams)
{
    loopMode.loop1Size = tdPtr_->moveAlignParams.loop1Size;
    loopMode.loop2Size = tdPtr_->moveAlignParams.loop2Size;
    loopMode.loop1SrcStride = tdPtr_->moveAlignParams.loop1SrcStride;
    loopMode.loop1DstStride = tdPtr_->moveAlignParams.loop1DstStride;
    loopMode.loop2SrcStride = tdPtr_->moveAlignParams.loop2SrcStride;
    loopMode.loop2DstStride = tdPtr_->moveAlignParams.loop2DstStride;

    extParams.blockCount = tdPtr_->moveAlignParams.blockCount;
    extParams.blockLen = tdPtr_->moveAlignParams.blockLen;
    extParams.srcStride = (tdPtr_->moveAlignParams.srcStride > extParams.blockLen) ?
                              tdPtr_->moveAlignParams.srcStride - extParams.blockLen :
                              0;
    extParams.dstStride = 0;
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::ParseCopyInTilingData()
{
    // main loop
    ParseLoopModeAndMoveAlignParams(loopMode_, copyInParam_);

    if (this->ubTailFactor_ > 0) {
        ParseLoopModeAndMoveAlignParams(loopModeTail_, copyInParamTail_);
        if (inUbDims_ == DIMS_1) {
            copyInParamTail_.blockLen = ((this->ubTailFactor_ - 1) * std::abs(lastDimStride_) + 1) * sizeof(T);
        } else if (inUbDims_ == DIMS_2) {
            copyInParamTail_.blockCount = this->ubTailFactor_;
        } else if (inUbDims_ == DIMS_3) {
            loopModeTail_.loop1Size = this->ubTailFactor_;
        } else {
            // the loop2 is outer loop
            loopModeTail_.loop2Size = this->ubTailFactor_;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::SetCopyOutAlignParams(DataCopyExtParams& copyOutParams,
                                                                                const DataCopyExtParams& copyInParam,
                                                                                const LoopModeParams& loopMode)
{
    uint32_t blockLen = Ops::Base::CeilDiv(uint32_t(copyInParam.blockLen / sizeof(T)),
                                           uint32_t(std::abs(lastDimStride_))) *
                        sizeof(T);
    if (blockLen % BLOCK_SIZE_BYTE != 0) {
        copyOutParams.blockCount = loopMode.loop2Size * loopMode.loop1Size * copyInParam.blockCount;
        copyOutParams.blockLen = blockLen;
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
    } else {
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = loopMode.loop2Size * loopMode.loop1Size * copyInParam.blockCount * blockLen;
        copyOutParams.dstStride = 0;
        copyOutParams.srcStride = 0;
    }
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::GatherInOrder(__local_mem__ RangeType* idxAddr,
                                                                        __local_mem__ T* inAddr,
                                                                        __local_mem__ T* outAddr, uint32_t outerAxis,
                                                                        uint32_t axis3,
                                                                        const DataCopyExtParams& copyInParam)
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
    RangeType axis3Offset = int32_t(rVLCnt) * lastDimStride_;
    RangeType axis3BAIn = Ops::Base::CeilAlign(uint32_t(copyInParam.blockLen / sizeof(T)), elemPerBlock_);
    RangeType idxBaseOffset = (lastDimStride_ < 0) ? ((int32_t)copyInParam.blockLen - (int32_t)copyInParam_.blockLen) /
                                                         (int32_t)(sizeof(T)) :
                                                     0;

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
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::GatherSlice(const LoopModeParams& loopMode,
                                                                      const DataCopyExtParams& copyInParam,
                                                                      const LocalTensor<RangeType>& idxTensor)
{
    uint32_t axis0 = loopMode.loop2Size;
    uint32_t axis1 = loopMode.loop1Size;
    uint32_t axis2 = copyInParam.blockCount;
    uint32_t axis3 = Ops::Base::CeilDiv(int64_t(copyInParam.blockLen / sizeof(T)), std::abs(lastDimStride_));
    uint32_t axis3BA = Ops::Base::CeilAlign(axis3, elemPerBlock_);

    auto outTensor = outQue_.AllocTensor<T>();
    auto inTensor = inQue_.DeQue<T>();
    __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
    __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr();
    GatherInOrder(idxAddr, inAddr, outAddr, axis0 * axis1 * axis2, axis3, copyInParam);
    inQue_.FreeTensor(inTensor);
    this->CalcReorderAxisInfo(axis0, axis1, axis2, axis3BA);
    this->Reorder4Output(outAddr);
    outQue_.EnQue(outTensor);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::ProcessWithDataCopyGather(
    int64_t inGmAddr, int64_t outGmAddr, const LoopModeParams& loopMode, const DataCopyExtParams& copyInParam,
    const DataCopyExtParams& copyOutParam, const LocalTensor<RangeType>& idxTensor)
{
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    auto inTensor = inQue_.AllocTensor<T>();
    SetLoopModePara(loopMode, DataCopyMVType::OUT_TO_UB);
    DataCopyPad<T>(inTensor, inputGM_[inGmAddr], copyInParam, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inQue_.EnQue(inTensor);

    GatherSlice(loopMode, copyInParam, idxTensor);

    auto outTensor = outQue_.DeQue<T>();
    DataCopyPad(outputGM_[outGmAddr], outTensor, copyOutParam);
    outQue_.FreeTensor(outTensor);
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::GenGatherIndex(LocalTensor<RangeType>& idxTensor)
{
    int32_t r1DimSize = Ops::Base::CeilDiv(int64_t(tdPtr_->moveAlignParams.blockLen / sizeof(T)),
                                           std::abs(lastDimStride_));
    __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
    if (lastDimStride_ > 0) {
        __VEC_SCOPE__
        {
            Reg::MaskReg mask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> indexReg;
            Reg::Arange(indexReg, 0);
            Reg::Muls(indexReg, indexReg, RangeType(lastDimStride_), mask);
            Reg::DataCopy(idxAddr, indexReg, mask);
        }
    } else {
        int32_t begIndex = r1DimSize - 1;
        RangeType stride = std::abs(lastDimStride_);
        __VEC_SCOPE__
        {
            Reg::MaskReg mask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> indexReg;
            Reg::RegTensor<RangeType> tmpReg;
            Reg::Arange<RangeType, Reg::IndexOrder::INCREASE_ORDER>(indexReg, 0);
            Reg::Duplicate(tmpReg, begIndex);
            Reg::Sub(indexReg, tmpReg, indexReg, mask);
            Reg::Muls(indexReg, indexReg, stride, mask);
            Reg::DataCopy(idxAddr, indexReg, mask);
        }
    }
}

template <typename T, typename U>
__aicore__ inline int64_t StridedSliceMoveAlignGather<T, U>::CalcRollbackOffset(const LoopModeParams& loopMode,
                                                                                const DataCopyExtParams& copyInParam)
{
    int64_t backOffset = 0;

    if (lastDimStride_ < 0) {
        backOffset += copyInParam.blockLen / sizeof(T) - 1;
    }
    if (inUbDims_ > DIMS_1 && this->strides_[this->inputDims_ - DIMS_2] < 0) {
        backOffset += (copyInParam.blockCount - 1) * tdPtr_->moveAlignParams.srcStride / sizeof(T);
    }
    if (inUbDims_ > DIMS_2 && this->strides_[this->inputDims_ - DIMS_3] < 0) {
        backOffset += (loopMode.loop1Size - 1) * tdPtr_->moveAlignParams.loop1SrcStride / sizeof(T);
    }
    if (inUbDims_ > DIMS_3 && this->strides_[this->inputDims_ - DIMS_4] < 0) {
        backOffset += (loopMode.loop2Size - 1) * tdPtr_->moveAlignParams.loop2SrcStride / sizeof(T);
    }

    return backOffset;
}

template <typename T, typename U>
__aicore__ inline void StridedSliceMoveAlignGather<T, U>::ProcessPerBlock(const DataCopyExtParams& copyOutParamsMain,
                                                                          const DataCopyExtParams& copyOutParamsTail)
{
    int64_t inputGmAddr = 0;
    int64_t outputGmAddr = 0;
    int64_t ubSplitLoopsNum = 0;  // ub切分轴上的循环次数
    int64_t ubOuterLoopsNum = 0;  // ub切分轴之外的循环次数
    int64_t rowsOffsetOutput = 0; // 当前核处理的output shape中的起始行数

    this->CalcProcessLoopsNum(ubOuterLoopsNum, ubSplitLoopsNum, blockIdx_);
    this->GetProcessRowsOffsetAll(rowsOffsetOutput, blockIdx_);

    auto idxTensor = idxBuf_.Get<RangeType>();
    GenGatherIndex(idxTensor);

    int64_t negativeStrideOffset = CalcRollbackOffset(loopMode_, copyInParam_);
    int64_t negativeStrideOffsetT = CalcRollbackOffset(loopModeTail_, copyInParamTail_);

    for (int64_t idx = 0; idx < ubOuterLoopsNum; idx++) {
        inputGmAddr = this->GetInputGmAddrAll(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        outputGmAddr = this->GetOutputGmAddrAll(rowsOffsetOutput + idx * this->rowsOffsetSteps_[this->ubIndex_]);
        for (int64_t loops = 0; loops < ubSplitLoopsNum; loops++) {
            ProcessWithDataCopyGather(inputGmAddr + loops * this->ubInLoopSteps_ - negativeStrideOffset,
                                      outputGmAddr + loops * this->ubOutLoopSteps_, loopMode_, copyInParam_,
                                      copyOutParamsMain, idxTensor);
        }
        if (this->ubTailFactor_ > 0) {
            ProcessWithDataCopyGather(inputGmAddr + ubSplitLoopsNum * this->ubInLoopSteps_ - negativeStrideOffsetT,
                                      outputGmAddr + ubSplitLoopsNum * this->ubOutLoopSteps_, loopModeTail_,
                                      copyInParamTail_, copyOutParamsTail, idxTensor);
        }
    }
}

} // namespace StridedSlice

#endif // STRIDED_SLICE_MOVE_ALIGN_GATHER_H
