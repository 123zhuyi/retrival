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
 * \file pad_edge_gather.h
 * \brief pad gather kernel
 */

#ifndef ASCENDC_PAD_EDGE_GATHER_H_
#define ASCENDC_PAD_EDGE_GATHER_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "op_kernel/platform_util.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;

template <typename T, int32_t KEY>
class PadEdgeGather {
private:
    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    constexpr static uint32_t VL_RANGE_CNT = VL_SIZE / sizeof(RangeType);
    constexpr static uint32_t UB_AXES = (KEY / KEY_BASE) % KEY_BASE; // TilingKey倒数第二维为UB内轴个数
    constexpr static uint32_t VL_CNT = VL_SIZE / sizeof(T);
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);
    constexpr static uint32_t MAX_DIM = 8;
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;
    constexpr static int32_t CONST4 = 4;

public:
    __aicore__ inline PadEdgeGather(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tdPtrGather_ = tilingData;
        dimNum_ = tdPtrGather_->dimNum;
        ubAxis_ = tdPtrGather_->ubAxis;
        ubFactor_ = tdPtrGather_->ubFactor;

        if (dimNum_ >= CONST3 && tdPtrGather_->outStride[dimNum_ - CONST3] <= VL_CNT / CONST2) {
            // 一次VL需要处理后面3根轴
            lastThirdDimInVL_ = true;
            vlSplitInEdgeGather_ = Std::min(uint64_t(VL_CNT / tdPtrGather_->outStride[dimNum_ - CONST3]),
                                            uint64_t(tdPtrGather_->outShape[dimNum_ - CONST3]));
        } else {
            vlSplitInEdgeGather_ = Std::min(uint64_t(VL_CNT / tdPtrGather_->outStride[dimNum_ - CONST2]),
                                            uint64_t(tdPtrGather_->outShape[dimNum_ - CONST2]));
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQue_, BUF_NUM, tdPtrGather_->outTileSize);
        pipe_->InitBuffer(outQue_, BUF_NUM, tdPtrGather_->outTileSize);
        pipe_->InitBuffer(idxBuf_, VL_SIZE * CONST2); // 纯pad索引占前VL; 非纯pad索引占后VL
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdxEdgeGather = blockIdx_ * tdPtrGather_->ubPerCount;
        if (startIdxEdgeGather >= tdPtrGather_->ubTotalCount) {
            return;
        }

        uint32_t endIdxEdgeGather = (blockIdx_ + 1L) * tdPtrGather_->ubPerCount;
        endIdxEdgeGather = endIdxEdgeGather < tdPtrGather_->ubTotalCount ? endIdxEdgeGather :
                                                                           tdPtrGather_->ubTotalCount;

        LocalTensor<RangeType> idxTensor = idxBuf_.Get<RangeType>();
        if (lastThirdDimInVL_) {
            GenGatherIndexThreeDim(idxTensor);
        } else {
            GenGatherIndex(idxTensor);
        }

        uint64_t ubTailFactor = tdPtrGather_->outShape[ubAxis_] % ubFactor_;
        ubTailFactor = (ubTailFactor == 0) ? ubFactor_ : ubTailFactor;
        for (uint32_t idx = startIdxEdgeGather; idx < endIdxEdgeGather; idx++) {
            uint64_t inIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint64_t outIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint32_t ubAxisInCopyNum = 0;
            uint32_t ubAxisOutCopyNum = 0;
            uint32_t ubAxisLeftPadNum = 0;
            uint32_t ubAxisRightPadNum = 0;
            bool isAllDup = false;

            CalcDimIdx(idx, inIndex, outIndex, isAllDup);
            CalcUbSplitInAndOut(inIndex, outIndex, ubTailFactor, ubAxisInCopyNum, ubAxisOutCopyNum, ubAxisLeftPadNum,
                                ubAxisRightPadNum);

            ProcessOneStep(inIndex, outIndex, isAllDup, ubAxisInCopyNum, ubAxisOutCopyNum, ubAxisLeftPadNum,
                           ubAxisRightPadNum, idxTensor);
        }
    }

private:
    __aicore__ inline void CalcDimIdx(uint32_t curIdx, uint64_t* inIndex, uint64_t* outIndex, bool& isAllDup)
    {
        for (int32_t i = ubAxis_; i >= 0; i--) {
            uint64_t factorEdgeGather = tdPtrGather_->outShape[i];
            if (i == ubAxis_) {
                factorEdgeGather = CeilDiv(factorEdgeGather, static_cast<uint64_t>(ubFactor_));
            }
            if (factorEdgeGather != 0) {
                outIndex[i] = (i == ubAxis_ ? curIdx % factorEdgeGather * ubFactor_ : curIdx % factorEdgeGather);
            }
            if (outIndex[i] < tdPtrGather_->leftPad[i]) {
                inIndex[i] = 0;
            } else {
                inIndex[i] = (outIndex[i] >= tdPtrGather_->leftPad[i] + tdPtrGather_->inShape[i]) ?
                                 (tdPtrGather_->inShape[i] - 1) :
                                 (outIndex[i] - tdPtrGather_->leftPad[i]);
            }
            if (i != ubAxis_ && (outIndex[i] < tdPtrGather_->leftPad[i] ||
                                 outIndex[i] >= tdPtrGather_->leftPad[i] + tdPtrGather_->inShape[i])) {
                isAllDup = true;
            }
            if (factorEdgeGather != 0) {
                curIdx = curIdx / factorEdgeGather;
            }
        }
    }

    __aicore__ inline void CalcUbSplitInAndOut(const uint64_t* inIndex, const uint64_t* outIndex, uint64_t ubTailFactor,
                                               uint32_t& ubAxisInCopyNum, uint32_t& ubAxisOutCopyNum,
                                               uint32_t& ubAxisLeftPadNum, uint32_t& ubAxisRightPadNum)
    {
        ubAxisOutCopyNum = (outIndex[ubAxis_] + ubFactor_ > tdPtrGather_->outShape[ubAxis_]) ? ubTailFactor : ubFactor_;

        // 可优化: ubAxisRightPadNum 一定等于 ubAxisOutCopyNum - ubAxisLeftPadNum - ubAxisInCopyNum;
        if (outIndex[ubAxis_] < tdPtrGather_->leftPad[ubAxis_] + tdPtrGather_->inShape[ubAxis_]) {
            // outIndex 在右pad点的左侧
            if (outIndex[ubAxis_] + ubFactor_ <= tdPtrGather_->leftPad[ubAxis_]) {
                // 输出都在左pad点左侧
                ubAxisInCopyNum = 0;
                ubAxisLeftPadNum = ubAxisOutCopyNum;
                ubAxisRightPadNum = 0;
            } else if (outIndex[ubAxis_] + ubFactor_ <
                       tdPtrGather_->leftPad[ubAxis_] + tdPtrGather_->inShape[ubAxis_]) {
                // 输出都在右pad点左侧
                ubAxisInCopyNum = outIndex[ubAxis_] + ubFactor_ - inIndex[ubAxis_] - tdPtrGather_->leftPad[ubAxis_];
                ubAxisLeftPadNum = ubAxisOutCopyNum - ubAxisInCopyNum;
                ubAxisRightPadNum = 0;
            } else {
                // 输出跨过右pad点
                ubAxisInCopyNum = tdPtrGather_->inShape[ubAxis_] - inIndex[ubAxis_];
                ubAxisLeftPadNum = outIndex[ubAxis_] < tdPtrGather_->leftPad[ubAxis_] ?
                                       (tdPtrGather_->leftPad[ubAxis_] - outIndex[ubAxis_]) :
                                       0;
                ubAxisRightPadNum = ubAxisOutCopyNum - ubAxisLeftPadNum - ubAxisInCopyNum;
            }
        } else {
            // outIndex 在右pad点的右侧
            ubAxisInCopyNum = 0;
            ubAxisLeftPadNum = 0;
            ubAxisRightPadNum = ubAxisOutCopyNum;
        }
    }

    __aicore__ inline void ProcessOneStep(const uint64_t* inIndex, const uint64_t* outIndex, bool isAllDup,
                                          uint32_t ubAxisInCopyNum, uint32_t ubAxisOutCopyNum,
                                          uint32_t ubAxisLeftPadNum, uint32_t ubAxisRightPadNum,
                                          LocalTensor<RangeType>& idxTensor)
    {
        uint32_t dupCopyInNum = ubAxisInCopyNum;
        if (ubAxisInCopyNum == 0 || isAllDup) {
            // isAllDup时 ubAxisInCopyNum 有可能不是0
            dupCopyInNum = ubAxisInCopyNum == 0 ? 1 : ubAxisInCopyNum;
        }
        CopyIn(inIndex, dupCopyInNum);
        GatherCompute(outIndex, ubAxisInCopyNum, ubAxisLeftPadNum, ubAxisRightPadNum, idxTensor);
        CopyOut(outIndex, ubAxisOutCopyNum);
    }

    __aicore__ inline void CopyIn(const uint64_t* inIndex, uint32_t ubAxisInCopyNum)
    {
        uint32_t copyInNum = ubAxisInCopyNum * tdPtrGather_->inStride[ubAxis_];
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += inIndex[i] * tdPtrGather_->inStride[i];
        }

        LocalTensor<T> inLocal = inQue_.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams = {1u, static_cast<uint32_t>(copyInNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(inLocal, inputGm_[inAddr], copyInParams, padParams);

        inQue_.EnQue(inLocal);
    }

    __aicore__ inline void GatherCompute(const uint64_t* outIndex, uint32_t ubAxisInCopyNum, uint32_t ubAxisLeftPadNum,
                                         uint32_t ubAxisRightPadNum, LocalTensor<RangeType>& idxTensor)
    {
        LocalTensor<T> inLocal = inQue_.DeQue<T>();
        LocalTensor<T> outLocal = outQue_.AllocTensor<T>();

        if constexpr (UB_AXES == CONST2) {
            GatherProcessUb2D(idxTensor, inLocal, outLocal, ubAxisInCopyNum, ubAxisLeftPadNum, ubAxisRightPadNum);
        } else if constexpr (UB_AXES == CONST3) {
            GatherProcessUb3D(idxTensor, inLocal, outLocal, ubAxisInCopyNum, ubAxisLeftPadNum, ubAxisRightPadNum);
        } else if constexpr (UB_AXES == CONST4) {
            GatherProcessUb4D(idxTensor, inLocal, outLocal, ubAxisInCopyNum, ubAxisLeftPadNum, ubAxisRightPadNum);
        }

        inQue_.FreeTensor(inLocal);
        outQue_.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(const uint64_t* outIndex, uint32_t ubAxisOutCopyNum)
    {
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            outAddr += outIndex[i] * tdPtrGather_->outStride[i];
        }
        uint32_t copyOutNum = ubAxisOutCopyNum * tdPtrGather_->outStride[ubAxis_];
        LocalTensor<T> outLocal = outQue_.DeQue<T>();
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outAddr], outLocal, copyOutParams);

        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void GenGatherIndexThreeDim(LocalTensor<RangeType>& idxTensor)
    {
        // 2*VL长度的索引，纯PAD索引占前VL，正常值索引占后VL
        uint32_t lastInDimSize = tdPtrGather_->inShape[dimNum_ - 1];
        uint16_t lastSecInDimSize = tdPtrGather_->inShape[dimNum_ - CONST2];
        int32_t outStride1 = tdPtrGather_->outStride[dimNum_ - CONST3];
        int32_t outStride2 = tdPtrGather_->outStride[dimNum_ - CONST2];
        int32_t inStride1 = tdPtrGather_->inStride[dimNum_ - CONST3];
        uint16_t lastTwoDimLoops = vlSplitInEdgeGather_;
        // 切在-3轴上，-2轴上的数据都是从gather中获取到的，包含pad
        int32_t lastLeftPadNum = tdPtrGather_->leftPad[dimNum_ - 1];
        uint16_t last2LeftPadNum = tdPtrGather_->leftPad[dimNum_ - CONST2];
        uint16_t last2RightPadNum = tdPtrGather_->outShape[dimNum_ - CONST2] - lastSecInDimSize - last2LeftPadNum;
        RangeType lastDimIdx = lastInDimSize - 1;
        RangeType last2DimIdx = lastSecInDimSize - 1;

        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr2 = (__local_mem__ RangeType*)idxTensor.GetPhyAddr() + VL_RANGE_CNT;

        __VEC_SCOPE__
        {
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> lineRange;
            Reg::RegTensor<RangeType> lineRangeNew;
            Reg::RegTensor<RangeType> lineRangeBk;
            Reg::MaskReg leftMask;
            Reg::RegTensor<RangeType> leftPadIdxReg;
            Reg::MaskReg rightMask;
            Reg::RegTensor<RangeType> rightPadIdxReg;
            Reg::UnalignReg uReg;
            Reg::UnalignReg uRegIn;

            // 先拼好-1轴的索引
            Reg::Arange(lineRange, 0);
            Reg::Adds(lineRange, lineRange, ((RangeType)-1) * lastLeftPadNum, maskIdx);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(leftMask, lineRange, 0, maskIdx);
            Reg::Duplicate(leftPadIdxReg, 0);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, leftPadIdxReg, leftMask);
            Reg::CompareScalar<RangeType, CMPMODE::GT>(rightMask, lineRange, lastDimIdx, maskIdx);
            Reg::Duplicate(rightPadIdxReg, lastDimIdx);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, rightPadIdxReg, rightMask);

            // -3轴纯pad的索引, 包含末尾两根轴
            for (uint16_t i = 0; i < lastTwoDimLoops; i++) {
                // -2轴leftpad行数, 索引都一样
                __local_mem__ RangeType* idxAddrTmp = idxAddr + i * outStride1;
                for (uint16_t j = 0; j < last2LeftPadNum; j++) {
                    Reg::DataCopyUnAlign(idxAddrTmp, lineRange, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);

                // -2 轴inputshape, 索引递增
                __local_mem__ RangeType* idxAddrTmp1 = idxAddr + i * outStride1 + last2LeftPadNum * outStride2;
                for (uint16_t j = 0; j < lastSecInDimSize; j++) {
                    Reg::Adds(lineRangeNew, lineRange, (RangeType)(j * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp1, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp1, uReg, 0);

                // -2轴rightpad行数, 索引都一样
                Reg::Adds(lineRangeNew, lineRange, (RangeType)(last2DimIdx * lastInDimSize), maskIdx);
                __local_mem__ RangeType* idxAddrTmp2 = idxAddr + i * outStride1 + last2LeftPadNum * outStride2 +
                                                       lastSecInDimSize * outStride2;
                for (uint16_t j = 0; j < last2RightPadNum; j++) {
                    Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp2, uReg, 0);
            }

            // -3轴有效输入的索引
            for (uint16_t i = 0; i < lastTwoDimLoops; i++) {
                Reg::Adds(lineRangeBk, lineRange, (RangeType)(i * inStride1), maskIdx);
                // -2轴leftpad行数, 索引都一样
                __local_mem__ RangeType* idxAddrTmp = idxAddr2 + i * outStride1;
                for (uint16_t j = 0; j < last2LeftPadNum; j++) {
                    Reg::DataCopyUnAlign(idxAddrTmp, lineRangeBk, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);

                // -2 轴inputshape, 索引递增
                __local_mem__ RangeType* idxAddrTmp1 = idxAddr2 + i * outStride1 + last2LeftPadNum * outStride2;
                for (uint16_t j = 0; j < lastSecInDimSize; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(j * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp1, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp1, uReg, 0);

                // -2轴rightpad行数, 索引都一样
                Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(last2DimIdx * lastInDimSize), maskIdx);
                __local_mem__ RangeType* idxAddrTmp2 = idxAddr2 + i * outStride1 + last2LeftPadNum * outStride2 +
                                                       lastSecInDimSize * outStride2;
                for (uint16_t j = 0; j < last2RightPadNum; j++) {
                    Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp2, uReg, 0);
            }
        }
    }

    __aicore__ inline void GenGatherIndex(LocalTensor<RangeType>& idxTensor)
    {
        // 2*VL长度的索引，纯PAD索引占前VL，正常值索引占后VL
        uint32_t lastInDimSize = tdPtrGather_->inShape[dimNum_ - 1];
        int32_t lastLeftPadNum = tdPtrGather_->leftPad[dimNum_ - 1];
        uint32_t lastOutDimSize = tdPtrGather_->outShape[dimNum_ - 1];
        uint16_t lastDimsLeft = vlSplitInEdgeGather_;
        RangeType lastDimIdx = lastInDimSize - 1;
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr2 = (__local_mem__ RangeType*)idxTensor.GetPhyAddr() + VL_RANGE_CNT;

        __VEC_SCOPE__
        {
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> lineRange;
            Reg::RegTensor<RangeType> lineRangeNew;
            Reg::MaskReg leftMask;
            Reg::RegTensor<RangeType> leftPadIdxReg;
            Reg::MaskReg rightMask;
            Reg::RegTensor<RangeType> rightPadIdxReg;
            Reg::UnalignReg uReg;
            Reg::UnalignReg uRegIn;

            // 先拼好-1轴的索引
            Reg::Arange(lineRange, 0);
            // 先拷出去，防止索引尾部脏数据
            Reg::DataCopy(idxAddr, lineRange, maskIdx);
            Reg::DataCopy(idxAddr2, lineRange, maskIdx);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

            Reg::Adds(lineRange, lineRange, ((RangeType)-1) * lastLeftPadNum, maskIdx);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(leftMask, lineRange, 0, maskIdx);
            Reg::Duplicate(leftPadIdxReg, (RangeType)0);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, leftPadIdxReg, leftMask);

            Reg::CompareScalar<RangeType, CMPMODE::GT>(rightMask, lineRange, lastDimIdx, maskIdx);
            Reg::Duplicate(rightPadIdxReg, lastDimIdx);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, rightPadIdxReg, rightMask);

            // -2轴leftpad行数, 索引都一样
            __local_mem__ RangeType* idxAddrTmp = idxAddr;
            for (uint16_t i = 0; i < lastDimsLeft; i++) {
                Reg::DataCopyUnAlign(idxAddrTmp, lineRange, uReg, lastOutDimSize);
            }
            Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);

            // -2轴有效输入的索引
            __local_mem__ RangeType* idxAddrTmp2 = idxAddr2;
            for (uint16_t i = 0; i < lastDimsLeft; i++) {
                RangeType loopStride = (RangeType)lastInDimSize * i;
                Reg::Adds(lineRangeNew, lineRange, loopStride, maskIdx);
                Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeNew, uRegIn, lastOutDimSize);
            }
            Reg::DataCopyUnAlignPost(idxAddrTmp2, uRegIn, 0);
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
        Reg::MaskReg maskAll = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
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

    // gather一次，copy多次
    __aicore__ inline void VlPaddingCopyProc(uint16_t gatherLoops, uint16_t padLoops, uint16_t lastPadLoops,
                                             uint32_t maskValue, uint32_t lastPadMaskValue,
                                             __local_mem__ T* curPadInAddr, __local_mem__ T* curPadOutAddr,
                                             Reg::RegTensor<RangeType>& regIdxPad, uint32_t idxPadOffset)
    {
        Reg::RegTensor<T> regData;
        Reg::RegTensor<T> regDataT;
        Reg::RegTensor<RangeType> regNewIdx;
        Reg::UnalignReg uReg;
        Reg::MaskReg maskAll = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
        uint32_t validMask = maskValue;
        if constexpr (sizeof(T) == 8) {
            validMask = maskValue * 2;
        }
        Reg::MaskReg maskIdx = Reg::UpdateMask<RangeType>(validMask);

        for (uint16_t gIdx = 0; gIdx < gatherLoops; gIdx++) {
            Reg::Adds(regNewIdx, regIdxPad, idxPadOffset, maskIdx);
            // gather一次，copy多次
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, curPadInAddr, (Reg::RegTensor<IdxType>&)regNewIdx,
                                maskIdx);
            __local_mem__ T* outAddrTmp = curPadOutAddr;
            for (uint16_t pIdx = 0; pIdx < padLoops; pIdx++) {
                if constexpr (sizeof(T) != 1) {
                    Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, maskValue);
                } else {
                    Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                    Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, maskValue);
                }
            }
            Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
            for (uint16_t pTaiIdx = 0; pTaiIdx < lastPadLoops; pTaiIdx++) {
                outAddrTmp = curPadOutAddr + padLoops * maskValue;
                if constexpr (sizeof(T) != 1) {
                    Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, lastPadMaskValue);
                } else {
                    Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                    Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, lastPadMaskValue);
                }
                Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
            }
        }
    }

    __aicore__ inline void GatherProcessUb2D(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                             LocalTensor<T>& outTensor, uint32_t ubAxisInCopyNum,
                                             uint32_t ubAxisLeftPadNum, uint32_t ubAxisRightPadNum)
    {
        __local_mem__ RangeType* idxPadAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr = idxPadAddr + VL_RANGE_CNT;
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr();

        // ubAxisInCopyNum 可能是0
        __local_mem__ T* outAddrValidIn = outAddr + tdPtrGather_->outStride[dimNum_ - CONST2] * ubAxisLeftPadNum;
        __local_mem__ T* outAddrDupRight = outAddrValidIn + tdPtrGather_->outStride[dimNum_ - CONST2] * ubAxisInCopyNum;

        uint32_t vlSplitLoopIn = vlSplitInEdgeGather_;
        if constexpr (sizeof(T) == 1) {
            vlSplitLoopIn /= sizeof(int16_t);
        }
        if (vlSplitLoopIn == 0) {
            vlSplitLoopIn = 1;
        }

        RangeType idxOffset = tdPtrGather_->inStride[dimNum_ - CONST2] * vlSplitLoopIn;
        uint32_t maskValue = tdPtrGather_->outStride[dimNum_ - CONST2] * vlSplitLoopIn;
        uint32_t axisVlRightPadOffset = ubAxisInCopyNum == 0 ?
                                            0 :
                                            (ubAxisInCopyNum - 1) * tdPtrGather_->inStride[dimNum_ - CONST2];

        uint16_t leftGatherLoops = ubAxisLeftPadNum == 0 ? 0 : 1;
        uint16_t leftPadLoops = static_cast<uint16_t>(ubAxisLeftPadNum / vlSplitLoopIn);
        uint16_t lastLeftPadNum = static_cast<uint16_t>(ubAxisLeftPadNum - leftPadLoops * vlSplitLoopIn);
        uint16_t lastLeftPadLoops = lastLeftPadNum == 0 ? 0 : 1;
        uint32_t lastLeftPadMaskValue = lastLeftPadNum * tdPtrGather_->outStride[dimNum_ - CONST2];

        uint16_t copyInPadLoops = static_cast<uint16_t>(ubAxisInCopyNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(ubAxisInCopyNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * tdPtrGather_->outStride[dimNum_ - CONST2];

        uint16_t rightGatherLoops = ubAxisRightPadNum == 0 ? 0 : 1;
        uint16_t rightPadLoops = static_cast<uint16_t>(ubAxisRightPadNum / vlSplitLoopIn);
        uint16_t lastrightPadNum = static_cast<uint16_t>(ubAxisRightPadNum - rightPadLoops * vlSplitLoopIn);
        uint16_t lastrightPadLoops = lastrightPadNum == 0 ? 0 : 1;
        uint32_t lastrightPadMaskValue = lastrightPadNum * tdPtrGather_->outStride[dimNum_ - CONST2];

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdxPad;
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdxPad, idxPadAddr);
            Reg::DataCopy(regIdx, idxAddr);

            // 该次Ub内H轴左pad
            VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue, inAddr,
                              outAddr, regIdxPad, 0);

            // 该次Ub内H轴有效输入个数
            VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                         outAddrValidIn, regIdx, 0);

            // 该次Ub内H轴右pad
            VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue, lastrightPadMaskValue,
                              inAddr, outAddrDupRight, regIdxPad, axisVlRightPadOffset);
        }
    }

    // ub切-3轴，vl切-2轴或-3轴
    __aicore__ inline void GatherProcessUb3D(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                             LocalTensor<T>& outTensor, uint32_t ubAxisInCopyNum,
                                             uint32_t ubAxisLeftPadNum, uint32_t ubAxisRightPadNum)
    {
        __local_mem__ RangeType* idxPadAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr = idxPadAddr + VL_RANGE_CNT;
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr();

        uint32_t vlSplitLoopIn = vlSplitInEdgeGather_;
        if constexpr (sizeof(T) == 1) {
            vlSplitLoopIn /= sizeof(int16_t);
        }
        if (vlSplitLoopIn == 0) {
            vlSplitLoopIn = 1;
        }

        uint32_t strideInVl = tdPtrGather_->inStride[dimNum_ - CONST2];
        uint32_t strideInVlO1 = tdPtrGather_->inStride[dimNum_ - CONST3];
        uint32_t strideOutVl = tdPtrGather_->outStride[dimNum_ - CONST2];
        uint32_t strideOutVlO1 = tdPtrGather_->outStride[dimNum_ - CONST3];

        // 该次Ub内C轴左pad, VL切3维时退化为1
        uint16_t ubAxisLeftPadLoops = ubAxisLeftPadNum;
        uint16_t ubAxisInCopyLoops = ubAxisInCopyNum;
        uint16_t ubAxisRightPadLoops = ubAxisRightPadNum;

        // ub切-3，这个值不会很大，uint32_t足够
        uint32_t vlLeftPadNum = tdPtrGather_->leftPad[dimNum_ - CONST2];
        uint32_t vlInNum = tdPtrGather_->inShape[dimNum_ - CONST2];
        uint32_t vlRightPadNum = tdPtrGather_->outShape[dimNum_ - CONST2] - tdPtrGather_->leftPad[dimNum_ - CONST2] -
                                 tdPtrGather_->inShape[dimNum_ - CONST2];
        uint32_t rightPadInVlOffset = vlInNum == 0 ? 0 : (vlInNum - 1) * strideInVl;
        uint32_t rightPadInVlO1Offset = ubAxisInCopyNum == 0 ? 0 : (ubAxisInCopyNum - 1) * strideInVlO1;

        if (lastThirdDimInVL_) {
            strideInVl = tdPtrGather_->inStride[dimNum_ - CONST3];
            strideInVlO1 = 1;
            strideOutVl = tdPtrGather_->outStride[dimNum_ - CONST3];
            strideOutVlO1 = 1;
            ubAxisLeftPadLoops = 1;
            ubAxisInCopyLoops = 0;
            ubAxisRightPadLoops = 0;
            vlLeftPadNum = ubAxisLeftPadNum;
            vlInNum = ubAxisInCopyNum;
            vlRightPadNum = ubAxisRightPadNum;
            rightPadInVlOffset = vlInNum == 0 ? 0 : (vlInNum - 1) * strideInVl;
            rightPadInVlO1Offset = 0;
        }
        RangeType idxOffset = strideInVl * vlSplitLoopIn;
        uint32_t maskValue = strideOutVl * vlSplitLoopIn;

        uint16_t leftGatherLoops = vlLeftPadNum == 0 ? 0 : 1;
        uint16_t leftPadLoops = static_cast<uint16_t>(vlLeftPadNum / vlSplitLoopIn);
        uint16_t lastLeftPadNum = static_cast<uint16_t>(vlLeftPadNum - leftPadLoops * vlSplitLoopIn);
        uint16_t lastLeftPadLoops = lastLeftPadNum == 0 ? 0 : 1;
        uint32_t lastLeftPadMaskValue = lastLeftPadNum * strideOutVl;

        uint16_t copyInPadLoops = static_cast<uint16_t>(vlInNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(vlInNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * strideOutVl;

        uint16_t rightGatherLoops = vlRightPadNum == 0 ? 0 : 1;
        uint16_t rightPadLoops = static_cast<uint16_t>(vlRightPadNum / vlSplitLoopIn);
        uint16_t lastrightPadNum = static_cast<uint16_t>(vlRightPadNum - rightPadLoops * vlSplitLoopIn);
        uint16_t lastrightPadLoops = lastrightPadNum == 0 ? 0 : 1;
        uint32_t lastrightPadMaskValue = lastrightPadNum * strideOutVl;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdxPad;
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdxPad, idxPadAddr);
            Reg::DataCopy(regIdx, idxAddr);

            // 该次Ub内C轴左pad, VL切3维时退化为1
            for (uint16_t ulIdx = 0; ulIdx < ubAxisLeftPadLoops; ulIdx++) {
                // H轴左pad，VL切3维时退化为该次Ub内C轴左pad
                __local_mem__ T* curPadOutAddr = outAddr + ulIdx * strideOutVlO1;
                VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue,
                                  inAddr, curPadOutAddr, regIdxPad, 0);

                // H轴上的输入，VL切3维时退化为该次Ub内C轴有效输入
                __local_mem__ T* curOutAddr = outAddr + ulIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl;
                VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                             curOutAddr, regIdx, 0);

                // H轴右pad，VL切3维时退化为该次Ub内C轴右pad
                curPadOutAddr = outAddr + ulIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl + vlInNum * strideOutVl;
                VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue, lastrightPadMaskValue,
                                  inAddr, curPadOutAddr, regIdxPad, rightPadInVlOffset);
            }

            // 该次Ub内C轴上的copyIn, VL切3维时退化为0
            for (uint16_t uiIdx = 0; uiIdx < ubAxisInCopyLoops; uiIdx++) {
                // H轴左pad
                uint32_t idxPadOffset = uiIdx * strideInVlO1;
                __local_mem__ T* curInOutAddr = outAddr + ubAxisLeftPadLoops * strideOutVlO1 + uiIdx * strideOutVlO1;
                VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue,
                                  inAddr, curInOutAddr, regIdxPad, idxPadOffset);

                // copyInPad -- gather多次 copy多次
                curInOutAddr = outAddr + ubAxisLeftPadLoops * strideOutVlO1 + uiIdx * strideOutVlO1 +
                               vlLeftPadNum * strideOutVl;
                VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                             curInOutAddr, regIdx, idxPadOffset);

                // right pad -- gather一次，copy多次
                curInOutAddr = outAddr + ubAxisLeftPadLoops * strideOutVlO1 + uiIdx * strideOutVlO1 +
                               vlLeftPadNum * strideOutVl + vlInNum * strideOutVl;
                idxPadOffset = uiIdx * strideInVlO1 + rightPadInVlOffset;
                VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue, lastrightPadMaskValue,
                                  inAddr, curInOutAddr, regIdxPad, idxPadOffset);
            }

            // 该次Ub内C轴右pad, VL切3维时退化为0
            for (uint16_t urIdx = 0; urIdx < ubAxisRightPadLoops; urIdx++) {
                // H轴左pad
                uint32_t idxPadOffset = rightPadInVlO1Offset;
                __local_mem__ T* curPadOutAddr = outAddr + ubAxisLeftPadLoops * strideOutVlO1 +
                                                 ubAxisInCopyLoops * strideOutVlO1 + urIdx * strideOutVlO1;
                VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue,
                                  inAddr, curPadOutAddr, regIdxPad, idxPadOffset);

                // H轴上的输入
                curPadOutAddr = outAddr + ubAxisLeftPadLoops * strideOutVlO1 + ubAxisInCopyLoops * strideOutVlO1 +
                                urIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl;
                VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                             curPadOutAddr, regIdx, idxPadOffset);

                // H轴右pad
                curPadOutAddr = outAddr + ubAxisLeftPadLoops * strideOutVlO1 + ubAxisInCopyLoops * strideOutVlO1 +
                                urIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl + vlInNum * strideOutVl;
                idxPadOffset = rightPadInVlO1Offset + rightPadInVlOffset;
                VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue, lastrightPadMaskValue,
                                  inAddr, curPadOutAddr, regIdxPad, idxPadOffset);
            }
        }
    }

    // NCHW: ub切N轴，vl切H轴或C轴。当前仅支持CHW维的pad
    __aicore__ inline void GatherProcessUb4D(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                             LocalTensor<T>& outTensor, uint32_t ubAxisInCopyNum,
                                             uint32_t ubAxisLeftPadNum, uint32_t ubAxisRightPadNum)
    {
        __local_mem__ RangeType* idxPadAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr = idxPadAddr + VL_RANGE_CNT;
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr();

        uint32_t vlSplitLoopIn = vlSplitInEdgeGather_;
        if constexpr (sizeof(T) == 1) {
            vlSplitLoopIn /= sizeof(int16_t);
        }
        if (vlSplitLoopIn == 0) {
            vlSplitLoopIn = 1;
        }

        uint16_t ubAxisLeftPadLoops = 0;              // ubAxisLeftPadNum; 当前不支持N轴的pad, 只会为0
        uint16_t ubAxisInCopyLoops = ubAxisInCopyNum; // ubAxisInCopyNum;
        uint16_t ubAxisRightPadLoops = 0;             // ubAxisRightPadNum; 当前不支持N轴的pad, 只会为0

        uint32_t strideInVl = tdPtrGather_->inStride[dimNum_ - CONST2];
        uint32_t strideInVlO1 = tdPtrGather_->inStride[dimNum_ - CONST3];
        uint32_t strideOutVl = tdPtrGather_->outStride[dimNum_ - CONST2];
        uint32_t strideOutVlO1 = tdPtrGather_->outStride[dimNum_ - CONST3];

        // ub切-4，这个值不会很大，uint32_t足够
        uint16_t vlO1LeftPadNum = tdPtrGather_->leftPad[dimNum_ - CONST3];
        uint16_t vlO1InNum = tdPtrGather_->inShape[dimNum_ - CONST3];
        uint16_t vlO1RightPadNum = tdPtrGather_->outShape[dimNum_ - CONST3] - tdPtrGather_->leftPad[dimNum_ - CONST3] -
                                   tdPtrGather_->inShape[dimNum_ - CONST3];

        uint32_t vlLeftPadNum = tdPtrGather_->leftPad[dimNum_ - CONST2];
        uint32_t vlInNum = tdPtrGather_->inShape[dimNum_ - CONST2];
        uint32_t vlRightPadNum = tdPtrGather_->outShape[dimNum_ - CONST2] - tdPtrGather_->leftPad[dimNum_ - CONST2] -
                                 tdPtrGather_->inShape[dimNum_ - CONST2];
        uint32_t rightPadInVlOffset = (vlInNum - 1) * strideInVl;
        uint32_t rightPadInVlO1Offset = (vlO1InNum - 1) * strideInVlO1;
        if (lastThirdDimInVL_) {
            strideInVl = tdPtrGather_->inStride[dimNum_ - CONST3];
            strideInVlO1 = tdPtrGather_->inStride[dimNum_ - CONST4];
            strideOutVl = tdPtrGather_->outStride[dimNum_ - CONST3];
            strideOutVlO1 = tdPtrGather_->outStride[dimNum_ - CONST4];

            vlLeftPadNum = tdPtrGather_->leftPad[dimNum_ - CONST3];
            vlInNum = tdPtrGather_->inShape[dimNum_ - CONST3];
            vlRightPadNum = tdPtrGather_->outShape[dimNum_ - CONST3] - tdPtrGather_->leftPad[dimNum_ - CONST3] -
                            tdPtrGather_->inShape[dimNum_ - CONST3];
            rightPadInVlOffset = (vlInNum - 1) * strideInVl;
            rightPadInVlO1Offset = 0;

            vlO1LeftPadNum = 1;
            vlO1InNum = 0;
            vlO1RightPadNum = 0;
        }

        RangeType idxOffset = strideInVl * vlSplitLoopIn;
        uint32_t maskValue = strideOutVl * vlSplitLoopIn;

        uint32_t strideInN = tdPtrGather_->inStride[dimNum_ - CONST4];
        uint32_t strideOutN = tdPtrGather_->outStride[dimNum_ - CONST4];

        uint16_t leftGatherLoops4D = vlLeftPadNum == 0 ? 0 : 1;
        uint16_t leftPadLoops = static_cast<uint16_t>(vlLeftPadNum / vlSplitLoopIn);
        uint16_t lastLeftPadNum = static_cast<uint16_t>(vlLeftPadNum - leftPadLoops * vlSplitLoopIn);
        uint16_t lastLeftPadLoops4D = lastLeftPadNum == 0 ? 0 : 1;
        uint32_t lastLeftPadMaskValue = lastLeftPadNum * strideOutVl;

        uint16_t copyInPadLoops = static_cast<uint16_t>(vlInNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum4D = static_cast<uint16_t>(vlInNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum4D == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum4D * strideOutVl;

        uint16_t rightGatherLoops = vlRightPadNum == 0 ? 0 : 1;
        uint16_t rightPadLoops = static_cast<uint16_t>(vlRightPadNum / vlSplitLoopIn);
        uint16_t lastrightPadNum4D = static_cast<uint16_t>(vlRightPadNum - rightPadLoops * vlSplitLoopIn);
        uint16_t lastrightPadLoops = lastrightPadNum4D == 0 ? 0 : 1;
        uint32_t lastrightPadMaskValue = lastrightPadNum4D * strideOutVl;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdxPad;
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdxPad, idxPadAddr);
            Reg::DataCopy(regIdx, idxAddr);

            // 处理N轴上的输入, N轴没有pad
            for (uint16_t nIdx = 0; nIdx < ubAxisInCopyLoops; nIdx++) {
                // C轴左pad, VL切3维时退化为1
                uint32_t curInOffset = nIdx * strideInN;
                uint32_t curOutOffset = nIdx * strideOutN;
                for (uint16_t i = 0; i < vlO1LeftPadNum; i++) {
                    // H轴左pad, VL切3维时退化为C轴左pad
                    uint32_t idxPadOffset = curInOffset;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1;
                    VlPaddingCopyProc(leftGatherLoops4D, leftPadLoops, lastLeftPadLoops4D, maskValue,
                                      lastLeftPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset);

                    // H轴输入个数, VL切3维时退化为C轴输入个数
                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + i * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    // H轴右pad, VL切3维时退化为C轴右pad
                    idxPadOffset = curInOffset + rightPadInVlOffset;
                    curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1 + vlLeftPadNum * strideOutVl +
                                    vlInNum * strideOutVl;
                    VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue,
                                      lastrightPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset);
                }
                // C轴上的输入个数，VL切3维时退化为0
                // curInOffset 不变 nIdx * strideInN;
                curOutOffset = nIdx * strideOutN + vlO1LeftPadNum * strideOutVlO1;
                for (uint16_t i = 0; i < vlO1InNum; i++) {
                    // H轴左pad
                    uint32_t idxPadOffset = curInOffset + i * strideInVlO1;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1;
                    VlPaddingCopyProc(leftGatherLoops4D, leftPadLoops, lastLeftPadLoops4D, maskValue,
                                      lastLeftPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset);

                    // H轴输入有效个数
                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + i * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    // H轴右pad
                    idxPadOffset = curInOffset + i * strideInVlO1 + rightPadInVlOffset;
                    curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1 + vlLeftPadNum * strideOutVl +
                                    vlInNum * strideOutVl;
                    VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue,
                                      lastrightPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset);
                }
                // C轴右pad，VL切3维时退化为0
                curInOffset = nIdx * strideInN + rightPadInVlO1Offset;
                curOutOffset = nIdx * strideOutN + vlO1LeftPadNum * strideOutVlO1 + vlO1InNum * strideOutVlO1;
                for (uint16_t i = 0; i < vlO1RightPadNum; i++) {
                    uint32_t idxPadOffset = curInOffset;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1;
                    VlPaddingCopyProc(leftGatherLoops4D, leftPadLoops, lastLeftPadLoops4D, maskValue,
                                      lastLeftPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset);

                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + i * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    idxPadOffset = curInOffset + rightPadInVlOffset;
                    curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1 + vlLeftPadNum * strideOutVl +
                                    vlInNum * strideOutVl;
                    VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue,
                                      lastrightPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset);
                }
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    const PadACTilingData* tdPtrGather_ = nullptr;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> idxBuf_;

    int32_t blockIdx_{0};
    int32_t dimNum_{0};
    int32_t ubAxis_{0};
    int32_t ubFactor_{0};
    uint16_t vlSplitInEdgeGather_{0}; // VL切分轴的factor
    bool lastThirdDimInVL_{false};    // 一次VL是否处理后面三根轴
};
} // namespace PadV3

#endif
