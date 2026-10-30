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
 * \file pad_gather.h
 * \brief pad gather kernel
 */

#ifndef ASCENDC_PAD_GATHER_H_
#define ASCENDC_PAD_GATHER_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "op_kernel/platform_util.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;

struct PadGatherParam {
    uint32_t axisVlO2Gather = 1; // 本次ub需要处理的实际输入shape各轴的大小
    uint32_t axisVlO1Gather = 1;
    uint32_t axisVlGather = 1;
    uint32_t strideOutVlO2Gather = 1;
    uint32_t strideOutVlO1Gather = 1;
    uint32_t strideOutVlGather = 1;
    uint32_t strideInVlO2Gather = 1;
    uint32_t strideInVlO1Gather = 1;
    uint32_t strideInVlGather = 1;
};

template <typename T, typename U = int8_t>
class PadGather {
private:
    constexpr static uint32_t VL_CNT = VL_SIZE / sizeof(T);
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);
    constexpr static uint32_t MAX_DIM = 8;
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;

    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

public:
    __aicore__ inline PadGather(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData,
                                GM_ADDR constValue = nullptr)
    {
        blockIdxGather_ = GetBlockIdx();
        tdPtr_ = tilingData;
        dimNum_ = tdPtr_->dimNum;
        ubAxis_ = tdPtr_->ubAxis;
        ubFactor_ = tdPtr_->ubFactor;

        if (dimNum_ >= CONST3 && tdPtr_->outStride[dimNum_ - CONST3] <= VL_CNT / CONST2) {
            // 一次VL需要处理后面3根轴
            lastThirdDimInVL_ = true;
            vlSplitInGather_ = Std::min(uint64_t(VL_CNT / tdPtr_->outStride[dimNum_ - CONST3]),
                                        uint64_t(tdPtr_->outShape[dimNum_ - CONST3]));
        } else {
            vlSplitInGather_ = Std::min(uint64_t(VL_CNT / tdPtr_->outStride[dimNum_ - CONST2]),
                                        uint64_t(tdPtr_->outShape[dimNum_ - CONST2]));
        }

        if (constValue != nullptr) {
            constValueGMGather_.SetGlobalBuffer((__gm__ T*)constValue);
            constValue_ = constValueGMGather_(0);
            if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
                T padValue = constValueGMGather_(0);
                T low4bit = padValue & 0x0F;
                constValue_ = (low4bit << 4) | low4bit;
            }
        }
        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQue_, BUF_NUM, tdPtr_->outTileSize + VL_SIZE);
        pipe_->InitBuffer(outQue_, BUF_NUM, tdPtr_->outTileSize);
        pipe_->InitBuffer(idxBuf_, VL_SIZE);
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdxGather = blockIdxGather_ * tdPtr_->ubPerCount;
        if (startIdxGather >= tdPtr_->ubTotalCount) {
            return;
        }

        uint32_t endIdxGather = (blockIdxGather_ + 1L) * tdPtr_->ubPerCount;
        endIdxGather = endIdxGather < tdPtr_->ubTotalCount ? endIdxGather : tdPtr_->ubTotalCount;

        DupInputFrontVL();
        LocalTensor<RangeType> idxTensor = idxBuf_.Get<RangeType>();
        if (lastThirdDimInVL_) {
            GenGatherIndexThreeDim(idxTensor);
        } else {
            GenGatherIndex(idxTensor);
        }

        uint64_t ubTailFactor = tdPtr_->outShape[ubAxis_] % ubFactor_;
        ubTailFactor = (ubTailFactor == 0) ? ubFactor_ : ubTailFactor;
        for (uint32_t idx = startIdxGather; idx < endIdxGather; idx++) {
            uint64_t inIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint64_t outIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint32_t ubAxisInCopyNum = 0;
            uint32_t ubAxisOutCopyNum = 0;
            bool isAllDup = false;

            CalcDimIdx(idx, inIndex, outIndex, isAllDup);
            CalcUbSplitInAndOut(inIndex, outIndex, ubTailFactor, ubAxisInCopyNum, ubAxisOutCopyNum);

            ProcessOneStep(inIndex, outIndex, isAllDup, ubAxisInCopyNum, ubAxisOutCopyNum, idxTensor);
        }
    }

private:
    __aicore__ inline void CalcDimIdx(uint32_t curIdx, uint64_t* inIndex, uint64_t* outIndex, bool& isAllDup)
    {
        for (int32_t i = ubAxis_; i >= 0; i--) {
            uint64_t factorGather = tdPtr_->outShape[i];
            if (i == ubAxis_) {
                factorGather = CeilDiv(factorGather, static_cast<uint64_t>(ubFactor_));
            }
            if (factorGather != 0) {
                outIndex[i] = (i == ubAxis_ ? curIdx % factorGather * ubFactor_ : curIdx % factorGather);
            }
            inIndex[i] = outIndex[i] < tdPtr_->leftPad[i] ? 0 : outIndex[i] - tdPtr_->leftPad[i];
            if (i != ubAxis_ &&
                (outIndex[i] < tdPtr_->leftPad[i] || outIndex[i] >= tdPtr_->leftPad[i] + tdPtr_->inShape[i])) {
                isAllDup = true;
            }
            if (factorGather != 0) {
                curIdx = curIdx / factorGather;
            }
        }
    }

    __aicore__ inline void CalcUbSplitInAndOut(const uint64_t* inIndex, const uint64_t* outIndex, uint64_t ubTailFactor,
                                               uint32_t& ubAxisInCopyNum, uint32_t& ubAxisOutCopyNum)
    {
        if (outIndex[ubAxis_] < tdPtr_->leftPad[ubAxis_] + tdPtr_->inShape[ubAxis_]) {
            // outIndex 在右pad点的左侧
            if (outIndex[ubAxis_] + ubFactor_ <= tdPtr_->leftPad[ubAxis_]) {
                // 输出都在左pad点左侧
                ubAxisInCopyNum = 0;
            } else if (outIndex[ubAxis_] + ubFactor_ < tdPtr_->leftPad[ubAxis_] + tdPtr_->inShape[ubAxis_]) {
                // 输出都在右pad点左侧
                ubAxisInCopyNum = outIndex[ubAxis_] + ubFactor_ - inIndex[ubAxis_] -
                                  tdPtr_->leftPad[ubAxis_]; // ub非整切时是否正确？
            } else {
                // 输出跨过右pad点
                ubAxisInCopyNum = tdPtr_->inShape[ubAxis_] - inIndex[ubAxis_];
            }
        } else {
            // outIndex 在右pad点的右侧
            ubAxisInCopyNum = 0;
        }

        ubAxisOutCopyNum = (outIndex[ubAxis_] + ubFactor_ > tdPtr_->outShape[ubAxis_]) ? ubTailFactor : ubFactor_;
    }

    __aicore__ inline void ProcessOneStep(const uint64_t* inIndex, const uint64_t* outIndex, bool isAllDup,
                                          uint32_t ubAxisInCopyNum, uint32_t ubAxisOutCopyNum,
                                          LocalTensor<RangeType>& idxTensor)
    {
        if (ubAxisInCopyNum == 0 || isAllDup) {
            DoOnlyDup(outIndex, ubAxisOutCopyNum);
        } else {
            CopyIn(inIndex, ubAxisInCopyNum);
            GatherCompute(outIndex, ubAxisInCopyNum, idxTensor);
            CopyOut(outIndex, ubAxisOutCopyNum);
        }
    }

    __aicore__ inline void CopyIn(const uint64_t* inIndex, uint32_t ubAxisInCopyNum)
    {
        uint32_t copyInNumGather = ubAxisInCopyNum * tdPtr_->inStride[ubAxis_];
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += inIndex[i] * tdPtr_->inStride[i];
        }

        LocalTensor<T> inLocal = inQue_.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams = {1u, static_cast<uint32_t>(copyInNumGather * sizeof(T)), 0, 0, 0};
        DataCopyPad(inLocal[VL_CNT], inputGm_[inAddr], copyInParams, padParams);

        inQue_.EnQue(inLocal);
    }

    __aicore__ inline void GatherCompute(const uint64_t* outIndex, uint32_t ubAxisInCopyNum,
                                         LocalTensor<RangeType>& idxTensor)
    {
        LocalTensor<T> inLocal = inQue_.DeQue<T>();
        LocalTensor<T> outLocal = outQue_.AllocTensor<T>();
        Duplicate(outLocal, constValue_, tdPtr_->outTileSize / sizeof(T));

        uint32_t outUbStart = 0;
        PadGatherParam gatherParam;
        if (lastThirdDimInVL_) {
            GetGatherParamThreeDim(outIndex, ubAxisInCopyNum, outUbStart, gatherParam);
        } else {
            GetGatherParam(outIndex, ubAxisInCopyNum, outUbStart, gatherParam);
        }
        GatherProcess(gatherParam, idxTensor, inLocal, outLocal, outUbStart);

        inQue_.FreeTensor(inLocal);
        outQue_.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(const uint64_t* outIndex, uint32_t ubAxisOutCopyNum)
    {
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            outAddr += outIndex[i] * tdPtr_->outStride[i];
        }
        uint32_t copyOutNumGather = ubAxisOutCopyNum * tdPtr_->outStride[ubAxis_];
        LocalTensor<T> outLocal = outQue_.DeQue<T>();
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNumGather * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outAddr], outLocal, copyOutParams);

        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void DoOnlyDup(const uint64_t* outIndex, uint32_t ubAxisOutCopyNum)
    {
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            outAddr += outIndex[i] * tdPtr_->outStride[i];
        }
        uint32_t copyOutNumGather = ubAxisOutCopyNum * tdPtr_->outStride[ubAxis_];

        LocalTensor<T> outLocal = outQue_.AllocTensor<T>();
        Duplicate(outLocal, constValue_, tdPtr_->outTileSize / sizeof(T));
        outQue_.EnQue(outLocal);
        outLocal = outQue_.DeQue<T>();

        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNumGather * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outAddr], outLocal, copyOutParams);

        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void DupInputFrontVL()
    {
        LocalTensor<T> srcLocal = inQue_.AllocTensor<T>();
        Duplicate(srcLocal, constValue_, VL_CNT);
        LocalTensor<T> srcLocal1 = inQue_.AllocTensor<T>();
        Duplicate(srcLocal1, constValue_, VL_CNT);

        inQue_.FreeTensor(srcLocal);
        inQue_.FreeTensor(srcLocal1);
    }

    __aicore__ inline void GetGatherParam(const uint64_t* outIndex, uint32_t ubAxisInCopyNum, uint32_t& outUbStart,
                                          PadGatherParam& gatherParam)
    {
        // N C H W -- VL循环在H上
        // 3 2 1
        int32_t inUbDim = dimNum_ - ubAxis_;
        gatherParam.axisVlGather = tdPtr_->inShape[dimNum_ - DIM_INDEX_SECOND];
        gatherParam.strideOutVlGather = tdPtr_->outStride[dimNum_ - DIM_INDEX_SECOND];
        gatherParam.strideInVlGather = tdPtr_->inStride[dimNum_ - DIM_INDEX_SECOND];
        if (inUbDim == INUB_DIM_INDEX_FOURTH) {
            gatherParam.axisVlO2Gather = ubAxisInCopyNum;
            gatherParam.axisVlO1Gather = tdPtr_->inShape[dimNum_ - DIM_INDEX_THIRD];
            gatherParam.strideOutVlO2Gather = tdPtr_->outStride[dimNum_ - DIM_INDEX_FOURTH];
            gatherParam.strideOutVlO1Gather = tdPtr_->outStride[dimNum_ - DIM_INDEX_THIRD];
            gatherParam.strideInVlO2Gather = tdPtr_->inStride[dimNum_ - DIM_INDEX_FOURTH];
            gatherParam.strideInVlO1Gather = tdPtr_->inStride[dimNum_ - DIM_INDEX_THIRD];
        } else if (inUbDim == INUB_DIM_INDEX_THIRD) {
            gatherParam.axisVlO1Gather = ubAxisInCopyNum;
            gatherParam.strideOutVlO1Gather = tdPtr_->outStride[dimNum_ - DIM_INDEX_THIRD];
            gatherParam.strideInVlO1Gather = tdPtr_->inStride[dimNum_ - DIM_INDEX_THIRD];
        } else if (inUbDim == INUB_DIM_INDEX_SECOND) {
            gatherParam.axisVlGather = ubAxisInCopyNum;
        }

        for (int32_t i = dimNum_ - 2; i > ubAxis_; i--) {
            outUbStart += tdPtr_->leftPad[i] * tdPtr_->outStride[i];
        }
        if (ubAxisInCopyNum != ubFactor_) {
            // 前有pad
            if (outIndex[ubAxis_] < tdPtr_->leftPad[ubAxis_]) {
                uint32_t curUbLeftPad = tdPtr_->leftPad[ubAxis_] % ubFactor_;
                outUbStart += curUbLeftPad * tdPtr_->outStride[ubAxis_];
            }
        }
    }

    __aicore__ inline void GetGatherParamThreeDim(const uint64_t* outIndex, uint32_t ubAxisInCopyNum,
                                                  uint32_t& outUbStart, PadGatherParam& gatherParam)
    {
        //   N C HW -- VL循环在C上，为了复用同一份gather vf代码
        // 3 2 1
        int32_t inUbDim = dimNum_ - ubAxis_;
        gatherParam.axisVlGather = tdPtr_->inShape[dimNum_ - DIM_INDEX_THIRD];
        gatherParam.strideOutVlGather = tdPtr_->outStride[dimNum_ - DIM_INDEX_THIRD];
        gatherParam.strideInVlGather = tdPtr_->inStride[dimNum_ - DIM_INDEX_THIRD];
        if (inUbDim == INUB_DIM_INDEX_FOURTH) {
            gatherParam.axisVlO1Gather = ubAxisInCopyNum;
            gatherParam.strideOutVlO1Gather = tdPtr_->outStride[dimNum_ - DIM_INDEX_FOURTH];
            gatherParam.strideInVlO1Gather = tdPtr_->inStride[dimNum_ - DIM_INDEX_FOURTH];
        } else if (inUbDim == INUB_DIM_INDEX_THIRD) {
            gatherParam.axisVlGather = ubAxisInCopyNum;
        }

        for (int32_t i = dimNum_ - 3; i > ubAxis_; i--) {
            outUbStart += tdPtr_->leftPad[i] * tdPtr_->outStride[i];
        }
        if (ubAxisInCopyNum != ubFactor_) {
            // 前有pad
            if (outIndex[ubAxis_] < tdPtr_->leftPad[ubAxis_]) {
                uint32_t curUbLeftPad = tdPtr_->leftPad[ubAxis_] % ubFactor_;
                outUbStart += curUbLeftPad * tdPtr_->outStride[ubAxis_];
            }
        }
    }

    __aicore__ inline void GenGatherIndexThreeDim(LocalTensor<RangeType>& idxTensor)
    {
        uint32_t lastInDimSize = tdPtr_->inShape[dimNum_ - 1];
        uint16_t lastSecInDimSize = tdPtr_->inShape[dimNum_ - CONST2];
        int32_t outStride1 = tdPtr_->outStride[dimNum_ - CONST3];
        int32_t outStride2 = tdPtr_->outStride[dimNum_ - CONST2];
        int32_t inStride1 = tdPtr_->inStride[dimNum_ - CONST3];
        int32_t inStride2 = tdPtr_->inStride[dimNum_ - CONST2];
        int32_t validBeginIdx = VL_CNT;
        uint16_t lastTwoDimLoops = vlSplitInGather_;
        // 切在-3轴上，-2轴上的数据都是从gather中获取到的，包含pad
        int32_t leftPadNum = tdPtr_->leftPad[dimNum_ - CONST2] * tdPtr_->outStride[dimNum_ - CONST2] +
                             tdPtr_->leftPad[dimNum_ - 1];

        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg maskMain = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> indexReg;
            Reg::RegTensor<RangeType> validReg;
            Reg::UnalignReg uReg;

            Reg::Arange(indexReg, 0); // 0-128
            Reg::DataCopy(idxAddr, indexReg, maskMain);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

            for (uint16_t i = 0; i < lastTwoDimLoops; i++) {
                for (uint16_t j = 0; j < lastSecInDimSize; j++) {
                    __local_mem__ RangeType* idxAddrTmp = idxAddr + leftPadNum + i * outStride1 + j * outStride2;
                    Reg::Arange(validReg, validBeginIdx + i * inStride1 + j * inStride2);
                    Reg::DataCopyUnAlign(idxAddrTmp, validReg, uReg, lastInDimSize);
                    Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);
                }
            }
        }
    }

    __aicore__ inline void GenGatherIndex(LocalTensor<RangeType>& idxTensor)
    {
        uint32_t lastInDimSize = tdPtr_->inShape[dimNum_ - 1];
        int32_t lastLeftPadNum = tdPtr_->leftPad[dimNum_ - 1];
        int32_t allPadNum = tdPtr_->outShape[dimNum_ - 1] - lastInDimSize;
        int32_t beginIdx = VL_CNT;
        int32_t scatBeginIdx = lastLeftPadNum + (vlSplitInGather_ - 1) * allPadNum;
        uint32_t scatterNum = vlSplitInGather_ * lastInDimSize;
        uint16_t lastDimsLeft = vlSplitInGather_ - 1;
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();

        /*
        1. 生成 (0,1,2,,,,127) -> ub  假设 vlSplitInGather_=3
        2. 假如最終需要的是 (0,1,2, 128,129,130, 6,7,8,9,10,11, 131,132,133, 15,16,17,18,19,20, 134,135,136)
                            lp       start        rp   lp        start1        rp     lp       start2
        3. 先生成 128,129,130,131,132,133,134,135,136 的有效idx
           再生成scatter的索引 (15，16，17，18，19，20，21，22，23) -- scatBeginIdx
           再生成scatter的索引 (9, 10, 11, 12, 13, 14) --> merging
           再生成scatter的索引 (3, 4, 5) --> merging
           得到 (3, 4, 5, 12, 13, 14, 21，22，23)
        */
        __VEC_SCOPE__
        {
            Reg::MaskReg mask;
            Reg::MaskReg maskMain = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> indexReg;
            Reg::RegTensor<RangeType> validReg;
            Reg::RegTensor<RangeType> scatIdxReg;
            Reg::RegTensor<RangeType> tmpScatIdxReg;
            Reg::Arange(indexReg, 0); // b16:0-128; b64:0-32
            Reg::DataCopy(idxAddr, indexReg, maskMain);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

            Reg::Arange(validReg, beginIdx); // 128 129 ..
            Reg::Arange(scatIdxReg, scatBeginIdx);
            for (uint16_t i = 0; i < lastDimsLeft; i++) {
                uint32_t sreg0 = lastInDimSize * (lastDimsLeft - i);
                mask = Reg::UpdateMask<RangeType>(sreg0);
                Reg::Arange(tmpScatIdxReg, lastLeftPadNum + (lastDimsLeft - 1 - i) * allPadNum);
                Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(scatIdxReg, tmpScatIdxReg, mask);
            }

            mask = Reg::UpdateMask<RangeType>(scatterNum);
            Reg::DataCopyScatter(idxAddr, validReg, (Reg::RegTensor<IdxType>&)scatIdxReg, mask);
        }
    }

    __aicore__ inline void GatherProcess(const PadGatherParam& gatherParam, const LocalTensor<RangeType>& idxTensor,
                                         LocalTensor<T>& inTensor, LocalTensor<T>& outTensor, uint32_t outUbStart)
    {
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr() + outUbStart;

        RangeType validBegin = VL_CNT;
        uint32_t vlSplitLoopIn = vlSplitInGather_;
        if constexpr (sizeof(T) == 1) {
            vlSplitLoopIn /= sizeof(int16_t);
        }
        if (vlSplitLoopIn == 0) {
            vlSplitLoopIn = 1;
        }

        RangeType idxOffset = gatherParam.strideInVlGather * vlSplitLoopIn;
        uint32_t maskValue = gatherParam.strideOutVlGather * vlSplitLoopIn;

        uint16_t vlSplitLoopCnt = gatherParam.axisVlGather / vlSplitLoopIn;
        uint16_t vlSplitTailCnt = gatherParam.axisVlGather - vlSplitLoopCnt * vlSplitLoopIn;
        uint16_t vlSplitTailLoopCnt = vlSplitTailCnt == 0 ? 0 : 1;
        uint32_t maskValueTail = vlSplitTailCnt * gatherParam.strideOutVlGather;

        uint16_t axisVlO2 = gatherParam.axisVlO2Gather;
        uint16_t axisVlO1 = gatherParam.axisVlO1Gather;
        uint32_t strideOutVlO2 = gatherParam.strideOutVlO2Gather;
        uint32_t strideOutVlO1 = gatherParam.strideOutVlO1Gather;
        RangeType strideInVlO2 = gatherParam.strideInVlO2Gather;
        RangeType strideInVlO1 = gatherParam.strideInVlO1Gather;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdx;
            Reg::RegTensor<RangeType> regIdxBK;
            Reg::RegTensor<RangeType> regNewIdx;
            Reg::RegTensor<T> regData;
            Reg::RegTensor<T> regDataT;
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskData;
            Reg::MaskReg pregT;
            Reg::UnalignReg uReg;

            Reg::DataCopy(regIdx, idxAddr);

            Reg::Arange(regNewIdx, 0);
            Reg::CompareScalar<RangeType, CMPMODE::GE>(pregT, regIdx, validBegin, maskIdx);

            for (uint16_t nIdx = 0; nIdx < axisVlO2; nIdx++) {
                for (uint16_t cIdx = 0; cIdx < axisVlO1; cIdx++) {
                    __local_mem__ T* outAddrTmp = outAddr + nIdx * strideOutVlO2 + cIdx * strideOutVlO1;
                    RangeType addsScale = nIdx * strideInVlO2 + cIdx * strideInVlO1;
                    for (uint16_t hIdx = 0; hIdx < vlSplitLoopCnt; hIdx++) {
                        Reg::Adds(regIdxBK, regIdx, hIdx * idxOffset + addsScale, pregT);
                        Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(regNewIdx, regIdxBK, pregT);

                        Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, inAddr,
                                            (Reg::RegTensor<IdxType>&)regNewIdx, maskIdx);
                        if constexpr (sizeof(T) != 1) {
                            // Reg::DataCopy(outAddr + hIdx * maskValue, regData, maskData);
                            Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, maskValue);
                        } else {
                            Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                            Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, maskValue);
                        }
                    }
                    Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
                    for (uint16_t hTail = 0; hTail < vlSplitTailLoopCnt; hTail++) {
                        outAddrTmp = outAddr + nIdx * strideOutVlO2 + cIdx * strideOutVlO1 + vlSplitLoopCnt * maskValue;
                        Reg::Adds(regIdxBK, regIdx, vlSplitLoopCnt * idxOffset + addsScale, pregT);
                        Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(regNewIdx, regIdxBK, pregT);

                        Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, inAddr,
                                            (Reg::RegTensor<IdxType>&)regNewIdx, maskIdx);
                        if constexpr (sizeof(T) != 1) {
                            Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, maskValueTail);
                        } else {
                            Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                            Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, maskValueTail);
                        }
                        Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
                    }
                }
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    const PadACTilingData* tdPtr_ = nullptr;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<T> constValueGMGather_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> idxBuf_;

    int32_t blockIdxGather_{0};
    int32_t dimNum_{0};
    int32_t ubAxis_{0};
    int32_t ubFactor_{0};
    uint16_t vlSplitInGather_{0};  // VL切分轴的factor
    bool lastThirdDimInVL_{false}; // 一次VL是否处理后面三根轴

    T constValue_{0};
};
} // namespace PadV3

#endif
