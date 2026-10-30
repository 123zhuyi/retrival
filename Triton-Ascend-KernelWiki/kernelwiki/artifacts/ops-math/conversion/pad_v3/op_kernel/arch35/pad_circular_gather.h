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
 * \file pad_circular_gather.h
 * \brief pad circular kernel
 */

#ifndef ASCENDC_PAD_CIRCULAR_GATHER_H_
#define ASCENDC_PAD_CIRCULAR_GATHER_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;

template <typename T, int32_t KEY>
class PadCircularGather {
private:
    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    using OutType = std::conditional_t<sizeof(T) != sizeof(uint64_t), T, uint32_t>;
    constexpr static uint32_t UB_AXES = (KEY / KEY_BASE) % KEY_BASE; // TilingKey倒数第二维为UB内轴个数
    constexpr static uint32_t VL_CNT = VL_SIZE / sizeof(T);
    constexpr static uint32_t BLOCK_NUM = UB_BLOCK / sizeof(T);
    constexpr static uint32_t MAX_DIM = 8;
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;
    constexpr static int32_t CONST4 = 4;

private:
    TPipe* pipe_ = nullptr;
    const PadACTilingData* tdPtr_ = nullptr;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQueFw_; // ub切分轴对应的原始输出
    TBuf<QuePosition::VECCALC> idxBuf_;

    int32_t blockIdx_{0};
    int32_t dimNum_{0};
    int32_t ubAxis_{0};
    int32_t ubFactor_{0};
    uint16_t vlSplitIn_{0};        // VL切分轴的factor
    bool lastThirdDimInVL_{false}; // 一次VL是否处理后面三根轴

    struct OutIndicesSet {
        uint64_t outIdx[CONST3];
        int32_t count = 0;
    };

public:
    __aicore__ inline PadCircularGather(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tdPtr_ = tilingData;
        dimNum_ = tdPtr_->dimNum;
        ubAxis_ = tdPtr_->ubAxis;
        ubFactor_ = tdPtr_->ubFactor;

        // 一次VL需要处理后面3根轴，当前支持切-3轴、-4轴
        if (UB_AXES >= CONST3 && tdPtr_->outStride[dimNum_ - CONST3] <= VL_CNT / CONST2) {
            lastThirdDimInVL_ = true;
            vlSplitIn_ = Std::min(uint64_t(VL_CNT / tdPtr_->outStride[dimNum_ - CONST3]),
                                  uint64_t(tdPtr_->outShape[dimNum_ - CONST3]));
        } else {
            vlSplitIn_ = Std::min(uint64_t(VL_CNT / tdPtr_->outStride[dimNum_ - CONST2]),
                                  uint64_t(tdPtr_->outShape[dimNum_ - CONST2]));
        }
        if constexpr (sizeof(T) == 1) {
            vlSplitIn_ /= sizeof(int16_t);
            if (vlSplitIn_ == 0) {
                vlSplitIn_ = 1;
            }
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQue_, BUF_NUM, tdPtr_->outTileSize);
        // 正向一份outTileSize; 上pad一个blocksize的临时空间
        pipe_->InitBuffer(outQueFw_, BUF_NUM, tdPtr_->outTileSize + UB_BLOCK);
        pipe_->InitBuffer(idxBuf_, VL_SIZE);
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdx = blockIdx_ * tdPtr_->ubPerCount;
        if (startIdx >= tdPtr_->ubTotalCount) {
            return;
        }

        uint32_t endIdx = (blockIdx_ + 1L) * tdPtr_->ubPerCount;
        endIdx = endIdx < tdPtr_->ubTotalCount ? endIdx : tdPtr_->ubTotalCount;

        LocalTensor<RangeType> idxTensor = idxBuf_.Get<RangeType>();
        if (lastThirdDimInVL_) {
            GenGatherIndexThreeDim(idxTensor);
        } else {
            GenGatherIndex(idxTensor);
        }

        uint64_t ubTailFactor = tdPtr_->inShape[ubAxis_] % ubFactor_;
        ubTailFactor = (ubTailFactor == 0) ? ubFactor_ : ubTailFactor;
        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            uint64_t inIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint16_t ubAxisInCopyNum = 0;

            CalcDimIdx(idx, inIndex);
            ubAxisInCopyNum = (inIndex[ubAxis_] + ubFactor_ > tdPtr_->inShape[ubAxis_]) ? ubTailFactor : ubFactor_;

            ProcessOneStep(inIndex, ubAxisInCopyNum, idxTensor);
        }
    }

private:
    __aicore__ inline void CalcDimIdx(uint32_t curIdx, uint64_t* inIndex)
    {
        for (int32_t i = ubAxis_; i >= 0; i--) {
            uint64_t factor = tdPtr_->inShape[i];
            if (i == ubAxis_) {
                factor = CeilDiv(factor, static_cast<uint64_t>(ubFactor_));
            }
            inIndex[i] = (i == ubAxis_ ? curIdx % factor * ubFactor_ : curIdx % factor);
            curIdx = curIdx / factor;
        }
    }

    __aicore__ inline void ProcessOneStep(const uint64_t* inIndex, uint16_t ubAxisInCopyNum,
                                          LocalTensor<RangeType>& idxTensor)
    {
        CopyIn(inIndex, ubAxisInCopyNum);
        GatherAndCopyOut(ubAxisInCopyNum, inIndex, idxTensor);
    }

    __aicore__ inline void CopyIn(const uint64_t* inIndex, uint16_t ubAxisInCopyNum)
    {
        uint32_t copyInNum = ubAxisInCopyNum * tdPtr_->inStride[ubAxis_];
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += inIndex[i] * tdPtr_->inStride[i];
        }

        LocalTensor<T> inLocal = inQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams = {1u, static_cast<uint32_t>(copyInNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(inLocal, inputGm_[inAddr], copyInParams, padParams);
        inQue_.EnQue(inLocal);
    }

    __aicore__ inline void GatherAndCopyOut(uint16_t ubAxisInCopyNum, const uint64_t* inIndex,
                                            LocalTensor<RangeType>& idxTensor)
    {
        // 最大支持5维input, ub内至少2根轴，所以最多4根轴
        OutIndicesSet totalOutIdx[CONST4];
        CalOutIndexValue(inIndex, totalOutIdx, CONST4);

        LocalTensor<T> inLocal = inQue_.DeQue<T>();
        LocalTensor<T> outLocalFw = outQueFw_.AllocTensor<T>();
        if constexpr (UB_AXES == CONST2) {
            GatherProcessUb2DFw(idxTensor, inLocal, outLocalFw, ubAxisInCopyNum);
        } else if constexpr (UB_AXES == CONST3) {
            GatherProcessUb3DFw(idxTensor, inLocal, outLocalFw, ubAxisInCopyNum);
        } else if constexpr (UB_AXES == CONST4) {
            GatherProcessUb4DFw(idxTensor, inLocal, outLocalFw, ubAxisInCopyNum);
        }
        inQue_.FreeTensor(inLocal);

        outQueFw_.EnQue(outLocalFw);
        LocalTensor<T> outLocal = outQueFw_.DeQue<T>();
        CopyOut(ubAxisInCopyNum, inIndex, totalOutIdx, outLocal);
        CopyOutPad(ubAxisInCopyNum, inIndex, totalOutIdx, outLocal);
        outQueFw_.FreeTensor(outLocal);
    }

    __aicore__ inline bool IsInLeftPad(int64_t inIdx, int32_t inAxis)
    {
        return (inIdx < (int64_t)tdPtr_->inShape[inAxis] &&
                inIdx > (int64_t)tdPtr_->inShape[inAxis] - (int64_t)tdPtr_->leftPad[inAxis] - 1);
    }

    __aicore__ inline bool IsInRightPad(int64_t inIdx, int32_t inAxis)
    {
        return inIdx < (int64_t)(tdPtr_->outShape[inAxis] - tdPtr_->inShape[inAxis] - tdPtr_->leftPad[inAxis]);
    }

    __aicore__ inline void CalOutIndexValue(const uint64_t* inIndex, OutIndicesSet* totalOutIdx, int32_t setMax)
    {
        // ub切分轴左侧，左/右pad以及原始的正向
        for (int32_t i = 0; i < ubAxis_; i++) {
            uint64_t iDimInIdx = inIndex[i];
            // 左pad
            if (IsInLeftPad(iDimInIdx, i)) {
                uint64_t inLeftPadStart = Std::max(iDimInIdx, (uint64_t)(tdPtr_->inShape[i] - tdPtr_->leftPad[i]));
                totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] + inLeftPadStart - tdPtr_->inShape[i];
                totalOutIdx[i].count++;
            }
            // 原始
            totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] + iDimInIdx;
            totalOutIdx[i].count++;
            // 右pad
            if (IsInRightPad(iDimInIdx, i)) {
                totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] + tdPtr_->inShape[i] + iDimInIdx;
                totalOutIdx[i].count++;
            }
        }

        // ub切分轴，正反向单独处理，此处不赋值

        // ub切分轴右侧，已经在ub中补好了所有的pad，此处置0
        for (int32_t i = ubAxis_ + 1; i < setMax; i++) {
            totalOutIdx[i].outIdx[0] = 0;
            totalOutIdx[i].count = 1;
        }
    }

    __aicore__ inline void CopyOut(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, OutIndicesSet* totalOutIdx,
                                   LocalTensor<T>& outLocal)
    {
        uint32_t copyOutNum = ubAxisInCopyNum * tdPtr_->outStride[ubAxis_];
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNum * sizeof(T)), 0, 0, 0};

        // ub切分轴，正向
        totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] + inIndex[ubAxis_];
        totalOutIdx[ubAxis_].count = 1;

        for (int32_t o0 = 0; o0 < totalOutIdx[0].count; o0++) {
            uint64_t o0Offset = totalOutIdx[0].outIdx[o0] * tdPtr_->outStride[0];
            for (int32_t o1 = 0; o1 < totalOutIdx[1].count; o1++) {
                uint64_t o1Offset = totalOutIdx[1].outIdx[o1] * tdPtr_->outStride[1];
                for (int32_t o2 = 0; o2 < totalOutIdx[CONST2].count; o2++) {
                    uint64_t o2Offset = totalOutIdx[CONST2].outIdx[o2] * tdPtr_->outStride[CONST2];
                    for (int32_t o3 = 0; o3 < totalOutIdx[CONST3].count; o3++) {
                        uint64_t o3Offset = totalOutIdx[CONST3].outIdx[o3] * tdPtr_->outStride[CONST3];
                        uint64_t outAddr = o0Offset + o1Offset + o2Offset + o3Offset;
                        DataCopyPad(outputGm_[outAddr], outLocal[BLOCK_NUM], copyOutParams);
                    }
                }
            }
        }
    }

    __aicore__ inline void CalcUbAxisPadNum(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, uint32_t& inLeftPadNum,
                                            uint32_t& inLeftPadStart, uint32_t& inRightPadNum,
                                            uint32_t& inRightPadStart)
    {
        uint64_t leftIdx = inIndex[ubAxis_] + ubAxisInCopyNum - 1;
        if (IsInLeftPad(leftIdx, ubAxis_)) {
            inLeftPadStart = Std::max((uint64_t)inIndex[ubAxis_],
                                      (uint64_t)(tdPtr_->inShape[ubAxis_] - tdPtr_->leftPad[ubAxis_]));
            inLeftPadNum = inIndex[ubAxis_] + ubAxisInCopyNum - inLeftPadStart;
            inLeftPadStart = inLeftPadStart - inIndex[ubAxis_];
        }

        int64_t rightPad = tdPtr_->outShape[ubAxis_] - tdPtr_->inShape[ubAxis_] - tdPtr_->leftPad[ubAxis_];
        if (IsInRightPad(inIndex[ubAxis_], ubAxis_)) {
            inRightPadStart = 0;
            inRightPadNum = Std::min((uint64_t)ubAxisInCopyNum, (uint64_t)rightPad - inIndex[ubAxis_]);
        }
    }

    __aicore__ inline void CopyOutLeftPad(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, LocalTensor<T>& outLocal,
                                          uint32_t inLeftPadNum, uint32_t inLeftPadStart, OutIndicesSet* totalOutIdx)
    {
        totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] + inLeftPadStart - tdPtr_->inShape[ubAxis_] +
                                         inIndex[ubAxis_];
        totalOutIdx[ubAxis_].count = 1;

        LocalTensor<T> outLocalReal = outLocal[BLOCK_NUM];
        LocalTensor<T> outLocalTmp = outLocal[0];

        uint32_t copyOutNum = inLeftPadNum * tdPtr_->outStride[ubAxis_];
        uint32_t copyStartOffset = inLeftPadStart * tdPtr_->outStride[ubAxis_];
        uint32_t alignRed = copyStartOffset % BLOCK_NUM;
        uint32_t alignOffset = 0;
        if (alignRed != 0) {
            __local_mem__ T* inAddrTmp = (__local_mem__ T*)outLocalReal.GetPhyAddr() + copyStartOffset;
            __local_mem__ T* outAddrTmp = (__local_mem__ T*)outLocalTmp.GetPhyAddr();

            alignOffset = BLOCK_NUM - alignRed;
            copyStartOffset = copyStartOffset + alignOffset;
            if (copyOutNum > alignOffset) {
                copyOutNum = (copyOutNum - alignOffset);
            } else {
                alignOffset = copyOutNum;
                copyOutNum = 0;
            }

            CopyTmpUnAlign(inAddrTmp, outAddrTmp, alignOffset);

            SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        }

        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNum * sizeof(T)), 0, 0, 0};
        DataCopyExtParams outParamAlign = {1u, static_cast<uint32_t>(alignOffset * sizeof(T)), 0, 0, 0};

        for (int32_t o0 = 0; o0 < totalOutIdx[0].count; o0++) {
            uint64_t o0Offset = totalOutIdx[0].outIdx[o0] * tdPtr_->outStride[0];
            for (int32_t o1 = 0; o1 < totalOutIdx[1].count; o1++) {
                uint64_t o1Offset = totalOutIdx[1].outIdx[o1] * tdPtr_->outStride[1];
                for (int32_t o2 = 0; o2 < totalOutIdx[CONST2].count; o2++) {
                    uint64_t o2Offset = totalOutIdx[CONST2].outIdx[o2] * tdPtr_->outStride[CONST2];
                    for (int32_t o3 = 0; o3 < totalOutIdx[CONST3].count; o3++) {
                        uint64_t o3Offset = totalOutIdx[CONST3].outIdx[o3] * tdPtr_->outStride[CONST3];
                        uint64_t outAddr = o0Offset + o1Offset + o2Offset + o3Offset;
                        if (copyOutNum > 0) {
                            DataCopyPad(outputGm_[outAddr + alignOffset], outLocalReal[copyStartOffset], copyOutParams);
                        }
                        if (alignOffset > 0) {
                            DataCopyPad(outputGm_[outAddr], outLocalTmp, outParamAlign);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutRightPad(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, LocalTensor<T>& outLocal,
                                           uint32_t inRightPadNum, uint32_t inRightPadStart, OutIndicesSet* totalOutIdx)
    {
        totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] + tdPtr_->inShape[ubAxis_] + inIndex[ubAxis_];
        totalOutIdx[ubAxis_].count = 1;

        uint32_t copyOutNum = inRightPadNum * tdPtr_->outStride[ubAxis_];
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNum * sizeof(T)), 0, 0, 0};
        uint32_t copyStartOffset = inRightPadStart * tdPtr_->outStride[ubAxis_];

        for (int32_t o0 = 0; o0 < totalOutIdx[0].count; o0++) {
            uint64_t o0Offset = totalOutIdx[0].outIdx[o0] * tdPtr_->outStride[0];
            for (int32_t o1 = 0; o1 < totalOutIdx[1].count; o1++) {
                uint64_t o1Offset = totalOutIdx[1].outIdx[o1] * tdPtr_->outStride[1];
                for (int32_t o2 = 0; o2 < totalOutIdx[CONST2].count; o2++) {
                    uint64_t o2Offset = totalOutIdx[CONST2].outIdx[o2] * tdPtr_->outStride[CONST2];
                    for (int32_t o3 = 0; o3 < totalOutIdx[CONST3].count; o3++) {
                        uint64_t o3Offset = totalOutIdx[CONST3].outIdx[o3] * tdPtr_->outStride[CONST3];
                        uint64_t outAddr = o0Offset + o1Offset + o2Offset + o3Offset;
                        DataCopyPad(outputGm_[outAddr], outLocal[BLOCK_NUM], copyOutParams);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyTmpUnAlign(__local_mem__ T* inAddrTmp, __local_mem__ T* outAddrTmp, uint32_t alignOffset)
    {
        uint32_t newCnt = (sizeof(T) != sizeof(OutType)) ? CONST2 * alignOffset : alignOffset;
        auto newInAddr = reinterpret_cast<__local_mem__ OutType*>(inAddrTmp);
        auto newOutAddr = reinterpret_cast<__local_mem__ OutType*>(outAddrTmp);

        __VEC_SCOPE__
        {
            Reg::RegTensor<OutType> regData;
            Reg::UnalignReg uReg;
            Reg::MaskReg maskIdx = Reg::UpdateMask<OutType>(newCnt);

            Reg::DataCopyUnAlignPre(uReg, newInAddr);
            Reg::DataCopyUnAlign(regData, uReg, newInAddr, newCnt);
            Reg::DataCopy(newOutAddr, regData, maskIdx);
        }
    }

    __aicore__ inline void CopyOutPad(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, OutIndicesSet* totalOutIdx,
                                      LocalTensor<T>& outLocal)
    {
        // 计算当前ub切分轴上，该次搬运的inputGm是否在左、右pad上，以及左右pad的起始位置
        uint32_t inLeftPadNum = 0;
        uint32_t inLeftPadStart = 0;
        uint32_t inRightPadNum = 0;
        uint32_t inRightPadStart = 0;

        CalcUbAxisPadNum(ubAxisInCopyNum, inIndex, inLeftPadNum, inLeftPadStart, inRightPadNum, inRightPadStart);

        if (inLeftPadNum != 0) {
            CopyOutLeftPad(ubAxisInCopyNum, inIndex, outLocal, inLeftPadNum, inLeftPadStart, totalOutIdx);
        }

        if (inRightPadNum != 0) {
            CopyOutRightPad(ubAxisInCopyNum, inIndex, outLocal, inRightPadNum, inRightPadStart, totalOutIdx);
        }
    }

    __aicore__ inline void GenGatherIndex(LocalTensor<RangeType>& idxTensor)
    {
        // 2*VL长度的索引，纯PAD索引占前VL，正常值索引占后VL
        uint32_t lastInDimSize = tdPtr_->inShape[dimNum_ - 1];
        int32_t lastLeftPadNum = tdPtr_->leftPad[dimNum_ - 1];
        uint32_t lastOutDimSize = tdPtr_->outShape[dimNum_ - 1];
        uint16_t lastDimsLeft = vlSplitIn_;
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> lineRange, lineRangeNew;
            Reg::MaskReg leftMask, rightMask;
            Reg::RegTensor<RangeType> leftPadIdxReg, rightPadIdxReg;
            Reg::UnalignReg uRegIn;

            // 先拼好-1轴的索引
            Reg::Arange(lineRange, 0);
            // 先拷出去，防止索引尾部脏数据
            Reg::DataCopy(idxAddr, lineRange, maskIdx);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

            Reg::Adds(lineRange, lineRange, ((RangeType)-1) * lastLeftPadNum, maskIdx);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(leftMask, lineRange, 0, maskIdx);
            Reg::Arange(leftPadIdxReg, lastInDimSize - lastLeftPadNum);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, leftPadIdxReg, leftMask);

            Reg::Arange(rightPadIdxReg, ((RangeType)-1) * (lastLeftPadNum + lastInDimSize));
            Reg::CompareScalar<RangeType, CMPMODE::GE>(rightMask, rightPadIdxReg, 0, maskIdx);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, rightPadIdxReg, rightMask);

            // -2轴有效输入的索引
            __local_mem__ RangeType* idxAddrTmp2 = idxAddr;
            for (uint16_t i = 0; i < lastDimsLeft; i++) {
                RangeType loopStride = (RangeType)lastInDimSize * i;
                Reg::Adds(lineRangeNew, lineRange, loopStride, maskIdx);
                Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeNew, uRegIn, lastOutDimSize);
            }
            Reg::DataCopyUnAlignPost(idxAddrTmp2, uRegIn, 0);
        }
    }

    __aicore__ inline void GenGatherIndexThreeDim(LocalTensor<RangeType>& idxTensor)
    {
        // 2*VL长度的索引，纯PAD索引占前VL，正常值索引占后VL
        uint32_t lastInDimSize = tdPtr_->inShape[dimNum_ - 1];
        uint16_t lastSecInDimSize = tdPtr_->inShape[dimNum_ - CONST2];
        int32_t outStride1 = tdPtr_->outStride[dimNum_ - CONST3];
        int32_t outStride2 = tdPtr_->outStride[dimNum_ - CONST2];
        int32_t inStride1 = tdPtr_->inStride[dimNum_ - CONST3];
        uint16_t lastTwoDimLoops = vlSplitIn_;
        // 切在-3轴上，-2轴上的数据都是从gather中获取到的，包含pad
        int32_t lastLeftPadNum = tdPtr_->leftPad[dimNum_ - 1];
        uint16_t last2LeftPadNum = tdPtr_->leftPad[dimNum_ - CONST2];
        uint16_t last2RightPadNum = tdPtr_->outShape[dimNum_ - CONST2] - lastSecInDimSize - last2LeftPadNum;
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> lineRange, lineRangeNew, lineRangeBk;
            Reg::MaskReg leftMask, rightMask;
            Reg::RegTensor<RangeType> leftPadIdxReg, rightPadIdxReg;
            Reg::UnalignReg uReg;

            // 先拼好-1轴的索引
            Reg::Arange(lineRange, 0);
            // 先拷出去，防止索引尾部脏数据
            Reg::DataCopy(idxAddr, lineRange, maskIdx);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

            Reg::Adds(lineRange, lineRange, ((RangeType)-1) * lastLeftPadNum, maskIdx);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(leftMask, lineRange, 0, maskIdx);
            Reg::Arange(leftPadIdxReg, lastInDimSize - lastLeftPadNum);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, leftPadIdxReg, leftMask);

            Reg::Arange(rightPadIdxReg, ((RangeType)-1) * (lastLeftPadNum + lastInDimSize));
            Reg::CompareScalar<RangeType, CMPMODE::GE>(rightMask, rightPadIdxReg, 0, maskIdx);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, rightPadIdxReg, rightMask);

            // -3轴有效输入的索引
            for (uint16_t i = 0; i < lastTwoDimLoops; i++) {
                Reg::Adds(lineRangeBk, lineRange, (RangeType)(i * inStride1), maskIdx);
                // -2轴leftpad行数, 索引取H轴下部
                __local_mem__ RangeType* idxAddrTmp = idxAddr + i * outStride1;
                for (uint16_t j = 0; j < last2LeftPadNum; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk,
                              (RangeType)((lastSecInDimSize - last2LeftPadNum + j) * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);

                // -2 轴inputshape, 索引递增
                __local_mem__ RangeType* idxAddrTmp1 = idxAddr + i * outStride1 + last2LeftPadNum * outStride2;
                for (uint16_t j = 0; j < lastSecInDimSize; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(j * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp1, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp1, uReg, 0);

                // -2轴rightpad行数, 索引取H轴上部
                __local_mem__ RangeType* idxAddrTmp2 = idxAddr + i * outStride1 + last2LeftPadNum * outStride2 +
                                                       lastSecInDimSize * outStride2;
                for (uint16_t j = 0; j < last2RightPadNum; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(j * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp2, uReg, 0);
            }
        }
    }

    __aicore__ inline void VlInCopyProc(uint16_t inLoops, uint16_t lastInLoops, RangeType idxOffset, uint32_t maskValue,
                                        uint32_t lastInMaskValue, __local_mem__ T* curInAddr,
                                        __local_mem__ T* curOutAddr, Reg::RegTensor<RangeType>& regIdx,
                                        uint32_t idxPadOffset)
    {
        Reg::RegTensor<T> regData;
        Reg::RegTensor<T> regDataT;
        Reg::RegTensor<RangeType> regIdxBk;
        Reg::RegTensor<RangeType> regNewIdx;
        Reg::UnalignReg uReg;
        uint32_t validMask = maskValue;
        if constexpr (sizeof(T) == 8) {
            validMask = maskValue * 2;
        }
        Reg::MaskReg maskIdx = Reg::UpdateMask<RangeType>(validMask);

        __local_mem__ T* outAddrTmp = curOutAddr;
        Reg::Adds(regIdxBk, regIdx, idxPadOffset, maskIdx);
        for (uint16_t cpIdx = 0; cpIdx < inLoops; cpIdx++) {
            Reg::Adds(regNewIdx, regIdxBk, cpIdx * idxOffset, maskIdx);
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, curInAddr, (Reg::RegTensor<IdxType>&)regNewIdx,
                                maskIdx);
            if constexpr (sizeof(T) != 1) {
                Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, maskValue);
            } else {
                Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, maskValue);
            }
        }
        Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
        for (uint16_t cpTailIdx = 0; cpTailIdx < lastInLoops; cpTailIdx++) {
            outAddrTmp = curOutAddr + inLoops * maskValue;
            Reg::Adds(regNewIdx, regIdxBk, inLoops * idxOffset, maskIdx);
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, curInAddr, (Reg::RegTensor<IdxType>&)regNewIdx,
                                maskIdx);
            if constexpr (sizeof(T) != 1) {
                Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, lastInMaskValue);
            } else {
                Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, lastInMaskValue);
            }
            Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
        }
    }

    __aicore__ inline void GatherProcessUb2DFw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        __local_mem__ RangeType* idxAddrFw = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr() + BLOCK_NUM;

        uint16_t vlSplitLoopIn = vlSplitIn_;
        RangeType idxOffset = tdPtr_->inStride[dimNum_ - CONST2] * vlSplitLoopIn;
        uint32_t maskValue = tdPtr_->outStride[dimNum_ - CONST2] * vlSplitLoopIn;

        uint16_t copyInPadLoops = static_cast<uint16_t>(ubAxisInCopyNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(ubAxisInCopyNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * tdPtr_->outStride[dimNum_ - CONST2];

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdx, idxAddrFw);
            VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr, outAddr,
                         regIdx, 0);
        }
    }

    __aicore__ inline void GatherProcessUb3DFw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        __local_mem__ RangeType* idxAddrFw = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr() + BLOCK_NUM;

        uint16_t vlSplitLoopIn = vlSplitIn_;
        uint32_t strideInVl = tdPtr_->inStride[dimNum_ - CONST2];
        uint32_t strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST3];
        uint32_t strideOutVl = tdPtr_->outStride[dimNum_ - CONST2];
        uint32_t strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST3];

        // 该次Ub内C轴左pad, VL切3维时退化为1
        uint16_t ubAxisInCopyLoops = ubAxisInCopyNum;

        // ub切-3，这个值不会很大，uint32_t足够
        uint32_t vlLeftPadNum = tdPtr_->leftPad[dimNum_ - CONST2];
        uint32_t vlInNum = tdPtr_->inShape[dimNum_ - CONST2];
        uint32_t vlRightPadNum = tdPtr_->outShape[dimNum_ - CONST2] - tdPtr_->leftPad[dimNum_ - CONST2] -
                                 tdPtr_->inShape[dimNum_ - CONST2];
        uint32_t leftPadInVlOffset = vlInNum == 0 ? 0 : (vlInNum - vlLeftPadNum) * strideInVl;

        if (lastThirdDimInVL_) {
            strideInVl = tdPtr_->inStride[dimNum_ - CONST3];
            strideInVlO1 = 1;
            strideOutVl = tdPtr_->outStride[dimNum_ - CONST3];
            strideOutVlO1 = 1;
            ubAxisInCopyLoops = 1;
            vlLeftPadNum = 0;
            vlInNum = ubAxisInCopyNum;
            vlRightPadNum = 0;
            leftPadInVlOffset = (vlInNum - vlLeftPadNum) * strideInVl;
        }

        RangeType idxOffset = strideInVl * vlSplitLoopIn;
        uint32_t maskValue = strideOutVl * vlSplitLoopIn;

        uint16_t leftPadLoops = static_cast<uint16_t>(vlLeftPadNum / vlSplitLoopIn);
        uint16_t lastLeftPadNum = static_cast<uint16_t>(vlLeftPadNum - leftPadLoops * vlSplitLoopIn);
        uint16_t lastLeftPadLoops = lastLeftPadNum == 0 ? 0 : 1;
        uint32_t lastLeftPadMaskValue = lastLeftPadNum * strideOutVl;

        uint16_t copyInPadLoops = static_cast<uint16_t>(vlInNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(vlInNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * strideOutVl;

        uint16_t rightPadLoops = static_cast<uint16_t>(vlRightPadNum / vlSplitLoopIn);
        uint16_t lastrightPadNum = static_cast<uint16_t>(vlRightPadNum - rightPadLoops * vlSplitLoopIn);
        uint16_t lastrightPadLoops = lastrightPadNum == 0 ? 0 : 1;
        uint32_t lastrightPadMaskValue = lastrightPadNum * strideOutVl;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdx, idxAddrFw);

            // 该次Ub内C轴上的copyIn, VL切3维时退化为0
            for (uint16_t uiIdx = 0; uiIdx < ubAxisInCopyLoops; uiIdx++) {
                // H轴左pad
                uint32_t idxPadOffset = uiIdx * strideInVlO1 + leftPadInVlOffset;
                __local_mem__ T* curInOutAddr = outAddr + uiIdx * strideOutVlO1;
                VlInCopyProc(leftPadLoops, lastLeftPadLoops, idxOffset, maskValue, lastLeftPadMaskValue, inAddr,
                             curInOutAddr, regIdx, idxPadOffset);

                idxPadOffset = uiIdx * strideInVlO1;
                curInOutAddr = outAddr + uiIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl;
                VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                             curInOutAddr, regIdx, idxPadOffset);

                curInOutAddr = outAddr + uiIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl + vlInNum * strideOutVl;
                VlInCopyProc(rightPadLoops, lastrightPadLoops, idxOffset, maskValue, lastrightPadMaskValue, inAddr,
                             curInOutAddr, regIdx, idxPadOffset);
            }
        }
    }

    __aicore__ inline void GatherProcessUb4DFw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr() + BLOCK_NUM;

        uint16_t vlSplitLoopIn = vlSplitIn_;
        uint16_t ubAxisLeftPadLoops = 0;              // ubAxisLeftPadNum; 当前不支持N轴的pad, 只会为0
        uint16_t ubAxisInCopyLoops = ubAxisInCopyNum; // ubAxisInCopyNum;
        uint16_t ubAxisRightPadLoops = 0;             // ubAxisRightPadNum; 当前不支持N轴的pad, 只会为0

        uint32_t strideInVl = tdPtr_->inStride[dimNum_ - CONST2];
        uint32_t strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST3];
        uint32_t strideOutVl = tdPtr_->outStride[dimNum_ - CONST2];
        uint32_t strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST3];

        // ub切-4，这个值不会很大，uint32_t足够
        uint16_t vlO1LeftPadNum = tdPtr_->leftPad[dimNum_ - CONST3];
        uint16_t vlO1InNum = tdPtr_->inShape[dimNum_ - CONST3];
        uint16_t vlO1RightPadNum = tdPtr_->outShape[dimNum_ - CONST3] - tdPtr_->leftPad[dimNum_ - CONST3] -
                                   tdPtr_->inShape[dimNum_ - CONST3];

        uint32_t vlLeftPadNum = tdPtr_->leftPad[dimNum_ - CONST2];
        uint32_t vlInNum = tdPtr_->inShape[dimNum_ - CONST2];
        uint32_t vlRightPadNum = tdPtr_->outShape[dimNum_ - CONST2] - tdPtr_->leftPad[dimNum_ - CONST2] -
                                 tdPtr_->inShape[dimNum_ - CONST2];
        uint32_t leftPadInVlOffset = (vlInNum - vlLeftPadNum) * strideInVl;
        uint32_t leftPadInVlO1Offset = (vlO1InNum - vlO1LeftPadNum) * strideInVlO1;

        if (lastThirdDimInVL_) {
            strideInVl = tdPtr_->inStride[dimNum_ - CONST3];
            strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST4];
            strideOutVl = tdPtr_->outStride[dimNum_ - CONST3];
            strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST4];

            vlLeftPadNum = tdPtr_->leftPad[dimNum_ - CONST3];
            vlInNum = tdPtr_->inShape[dimNum_ - CONST3];
            vlRightPadNum = tdPtr_->outShape[dimNum_ - CONST3] - tdPtr_->leftPad[dimNum_ - CONST3] -
                            tdPtr_->inShape[dimNum_ - CONST3];

            vlO1LeftPadNum = 0;
            vlO1InNum = 1;
            vlO1RightPadNum = 0;

            leftPadInVlOffset = (vlInNum - vlLeftPadNum) * strideInVl;
            leftPadInVlO1Offset = 0;
        }

        RangeType idxOffset = strideInVl * vlSplitLoopIn;
        uint32_t maskValue = strideOutVl * vlSplitLoopIn;

        uint32_t strideInN = tdPtr_->inStride[dimNum_ - CONST4];
        uint32_t strideOutN = tdPtr_->outStride[dimNum_ - CONST4];

        uint16_t leftPadLoops = static_cast<uint16_t>(vlLeftPadNum / vlSplitLoopIn);
        uint16_t lastLeftPadNum = static_cast<uint16_t>(vlLeftPadNum - leftPadLoops * vlSplitLoopIn);
        uint16_t lastLeftPadLoops = lastLeftPadNum == 0 ? 0 : 1;
        uint32_t lastLeftPadMaskValue = lastLeftPadNum * strideOutVl;

        uint16_t copyInPadLoops = static_cast<uint16_t>(vlInNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(vlInNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * strideOutVl;

        uint16_t rightPadLoops = static_cast<uint16_t>(vlRightPadNum / vlSplitLoopIn);
        uint16_t lastrightPadNum = static_cast<uint16_t>(vlRightPadNum - rightPadLoops * vlSplitLoopIn);
        uint16_t lastrightPadLoops = lastrightPadNum == 0 ? 0 : 1;
        uint32_t lastrightPadMaskValue = lastrightPadNum * strideOutVl;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdx, idxAddr);

            for (uint16_t nIdx = 0; nIdx < ubAxisInCopyLoops; nIdx++) {
                // C轴左pad, VL切3维时退化为1
                uint32_t curInOffset = nIdx * strideInN + leftPadInVlO1Offset;
                uint32_t curOutOffset = nIdx * strideOutN;
                for (uint16_t i = 0; i < vlO1LeftPadNum; i++) {
                    // H轴左pad, VL切3维时退化为C轴左pad
                    uint32_t idxPadOffset = curInOffset + i * strideInVlO1 + leftPadInVlOffset;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1;
                    VlInCopyProc(leftPadLoops, lastLeftPadLoops, idxOffset, maskValue, lastLeftPadMaskValue, inAddr,
                                 curPadOutAddr, regIdx, idxPadOffset);

                    // H轴输入个数, VL切3维时退化为C轴输入个数
                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + i * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    idxPadOffset = curInOffset + i * strideInVlO1;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    // H轴右pad, VL切3维时退化为C轴右pad
                    curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1 + vlLeftPadNum * strideOutVl +
                                    vlInNum * strideOutVl;
                    VlInCopyProc(rightPadLoops, lastrightPadLoops, idxOffset, maskValue, lastrightPadMaskValue, inAddr,
                                 curPadOutAddr, regIdx, idxPadOffset);
                }
                // C轴上的输入个数，VL切3维时退化为0
                curInOffset = nIdx * strideInN;
                curOutOffset = nIdx * strideOutN + vlO1LeftPadNum * strideOutVlO1;
                for (uint16_t i = 0; i < vlO1InNum; i++) {
                    // H轴左pad
                    uint32_t idxPadOffset = curInOffset + i * strideInVlO1 + leftPadInVlOffset;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1;
                    VlInCopyProc(leftPadLoops, lastLeftPadLoops, idxOffset, maskValue, lastLeftPadMaskValue, inAddr,
                                 curPadOutAddr, regIdx, idxPadOffset);

                    // H轴输入有效个数
                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + i * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    idxPadOffset = curInOffset + i * strideInVlO1;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    // H轴右pad
                    curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1 + vlLeftPadNum * strideOutVl +
                                    vlInNum * strideOutVl;
                    VlInCopyProc(rightPadLoops, lastrightPadLoops, idxOffset, maskValue, lastrightPadMaskValue, inAddr,
                                 curPadOutAddr, regIdx, idxPadOffset);
                }
                // C轴右pad，VL切3维时退化为0
                curInOffset = nIdx * strideInN;
                curOutOffset = nIdx * strideOutN + vlO1LeftPadNum * strideOutVlO1 + vlO1InNum * strideOutVlO1;
                for (uint16_t i = 0; i < vlO1RightPadNum; i++) {
                    uint32_t idxPadOffset = curInOffset + i * strideInVlO1 + leftPadInVlOffset;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1;
                    VlInCopyProc(leftPadLoops, lastLeftPadLoops, idxOffset, maskValue, lastLeftPadMaskValue, inAddr,
                                 curPadOutAddr, regIdx, idxPadOffset);

                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + i * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    idxPadOffset = curInOffset + i * strideInVlO1;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1 + vlLeftPadNum * strideOutVl +
                                    vlInNum * strideOutVl;
                    VlInCopyProc(rightPadLoops, lastrightPadLoops, idxOffset, maskValue, lastrightPadMaskValue, inAddr,
                                 curPadOutAddr, regIdx, idxPadOffset);
                }
            }
        }
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetWaitEvent(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }
};
} // namespace PadV3

#endif
