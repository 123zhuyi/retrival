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
 * \file pad_scatter.h
 * \brief pad scatter kernel
 */

#ifndef ASCENDC_PAD_SCATTER_H_
#define ASCENDC_PAD_SCATTER_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "op_kernel/platform_util.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;

struct PadScatterParam {
    uint32_t axisVlO2 = 1; // 本次ub需要处理的实际输入shape各轴的大小
    uint32_t axisVlO1 = 1;
    uint32_t axisVl = 1;
    uint32_t strideOutVlO2 = 1;
    uint32_t strideOutVlO1 = 1;
    uint32_t strideOutVl = 1;
    uint32_t strideInVlO2 = 1;
    uint32_t strideInVlO1 = 1;
    uint32_t strideInVl = 1;
};

template <typename T, typename U = int8_t>
class PadScatter {
private:
    constexpr static uint32_t VL_CNT = VL_SIZE / sizeof(T);
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);
    constexpr static uint32_t MAX_DIM = 8;
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;
    constexpr static int32_t CONST4 = 4;

    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

public:
    __aicore__ inline PadScatter(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData,
                                GM_ADDR constValue = nullptr)
    {
        blockIdx_ = GetBlockIdx();
        tdPtr_ = tilingData;
        dimNum_ = tdPtr_->dimNum;
        ubAxis_ = tdPtr_->ubAxis;
        ubFactor_ = tdPtr_->ubFactor;

        if (dimNum_ - ubAxis_ >= CONST3 && tdPtr_->inStride[dimNum_ - CONST3] <= VL_CNT / CONST2) {
            // 一次VL需要处理后面3根轴
            lastThirdDimInVL_ = true;
            vlSplitIn_ = Std::min(uint64_t(VL_CNT / tdPtr_->inStride[dimNum_ - CONST3]),
                                  uint64_t(tdPtr_->inShape[dimNum_ - CONST3]));
            if (vlSplitIn_ == 1) {
                // 考虑到int8场景的gather，vlSplitIn_不能为1
                lastThirdDimInVL_ = false;
                vlSplitIn_ = tdPtr_->inShape[dimNum_ - CONST2];
            }
        } else {
            // 如果倒数第二根轴为1，则最后一根轴肯定 <= VL/2. vlSplitIn_不需要赋值为最后一根轴，也能保证b8场景的scatter
            vlSplitIn_ = Std::min(uint64_t(VL_CNT / tdPtr_->inStride[dimNum_ - CONST2]),
                                  uint64_t(tdPtr_->inShape[dimNum_ - CONST2]));
        }

        if (constValue != nullptr) {
            constValueGM_.SetGlobalBuffer((__gm__ T*)constValue);
            constValue_ = constValueGM_(0);
            if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
                T padValue = constValueGM_(0);
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
        uint32_t startIdxScatter = blockIdx_ * tdPtr_->ubPerCount;
        if (startIdxScatter >= tdPtr_->ubTotalCount) {
            return;
        }

        uint32_t endIdxScatter = (blockIdx_ + 1L) * tdPtr_->ubPerCount;
        endIdxScatter = endIdxScatter < tdPtr_->ubTotalCount ? endIdxScatter : tdPtr_->ubTotalCount;

        LocalTensor<RangeType> idxTensor = idxBuf_.Get<RangeType>();
        GenScatterIndex(idxTensor);

        uint64_t ubTailFactor = tdPtr_->outShape[ubAxis_] % ubFactor_;
        ubTailFactor = (ubTailFactor == 0) ? ubFactor_ : ubTailFactor;
        for (uint32_t idx = startIdxScatter; idx < endIdxScatter; idx++) {
            uint64_t inIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint64_t outIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint32_t ubAxisInCopyNum = 0;
            uint32_t ubAxisOutCopyNumScatter = 0;
            bool isAllDup = false;

            CalcDimIdx(idx, inIndex, outIndex, isAllDup);
            CalcUbSplitInAndOut(inIndex, outIndex, ubTailFactor, ubAxisInCopyNum, ubAxisOutCopyNumScatter);

            ProcessOneStep(inIndex, outIndex, isAllDup, ubAxisInCopyNum, ubAxisOutCopyNumScatter, idxTensor);
        }
    }

private:
    __aicore__ inline void CalcDimIdx(uint32_t curIdx, uint64_t* inIndex, uint64_t* outIndex, bool& isAllDup)
    {
        for (int32_t i = ubAxis_; i >= 0; i--) {
            uint64_t factorScatter = tdPtr_->outShape[i];
            if (i == ubAxis_) {
                factorScatter = CeilDiv(factorScatter, static_cast<uint64_t>(ubFactor_));
            }
            if (factorScatter != 0) {
                outIndex[i] = (i == ubAxis_ ? curIdx % factorScatter * ubFactor_ : curIdx % factorScatter);
            }
            inIndex[i] = outIndex[i] < tdPtr_->leftPad[i] ? 0 : outIndex[i] - tdPtr_->leftPad[i];
            if (i != ubAxis_ &&
                (outIndex[i] < tdPtr_->leftPad[i] || outIndex[i] >= tdPtr_->leftPad[i] + tdPtr_->inShape[i])) {
                isAllDup = true;
            }
            if (factorScatter != 0) {
                curIdx = curIdx / factorScatter;
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
            ScatterCompute(outIndex, ubAxisInCopyNum, idxTensor);
            CopyOut(outIndex, ubAxisOutCopyNum);
        }
    }

    __aicore__ inline void CopyIn(const uint64_t* inIndex, uint32_t ubAxisInCopyNum)
    {
        uint32_t copyInNumScatter = ubAxisInCopyNum * tdPtr_->inStride[ubAxis_];
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += inIndex[i] * tdPtr_->inStride[i];
        }

        LocalTensor<T> inLocal = inQue_.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams = {1u, static_cast<uint32_t>(copyInNumScatter * sizeof(T)), 0, 0, 0};
        DataCopyPad(inLocal, inputGm_[inAddr], copyInParams, padParams);

        inQue_.EnQue(inLocal);
    }

    __aicore__ inline void ScatterCompute(const uint64_t* outIndex, uint32_t ubAxisInCopyNum,
                                          LocalTensor<RangeType>& idxTensor)
    {
        LocalTensor<T> inLocal = inQue_.DeQue<T>();
        LocalTensor<T> outLocal = outQue_.AllocTensor<T>();
        Duplicate(outLocal, constValue_, tdPtr_->outTileSize / sizeof(T));

        uint32_t outUbStart = 0;
        PadScatterParam scatterParam;
        if (lastThirdDimInVL_) {
            GetScatterParamThreeDim(outIndex, ubAxisInCopyNum, outUbStart, scatterParam);
        } else {
            GetScatterParam(outIndex, ubAxisInCopyNum, outUbStart, scatterParam);
        }
        ScatterProcess(scatterParam, idxTensor, inLocal, outLocal, outUbStart);

        inQue_.FreeTensor(inLocal);
        outQue_.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(const uint64_t* outIndex, uint32_t ubAxisOutCopyNum)
    {
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            outAddr += outIndex[i] * tdPtr_->outStride[i];
        }
        uint32_t copyOutNumScatter = ubAxisOutCopyNum * tdPtr_->outStride[ubAxis_];
        LocalTensor<T> outLocal = outQue_.DeQue<T>();
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNumScatter * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outAddr], outLocal, copyOutParams);

        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void DoOnlyDup(const uint64_t* outIndex, uint32_t ubAxisOutCopyNum)
    {
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            outAddr += outIndex[i] * tdPtr_->outStride[i];
        }
        uint32_t copyOutNumScatter = ubAxisOutCopyNum * tdPtr_->outStride[ubAxis_];

        LocalTensor<T> outLocal = outQue_.AllocTensor<T>();
        Duplicate(outLocal, constValue_, tdPtr_->outTileSize / sizeof(T));
        outQue_.EnQue(outLocal);
        outLocal = outQue_.DeQue<T>();

        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNumScatter * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outAddr], outLocal, copyOutParams);

        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void GetScatterParam(const uint64_t* outIndex, uint32_t ubAxisInCopyNum, uint32_t& outUbStart,
                                           PadScatterParam& scatterParam)
    {
        // N C H W -- VL循环在H上
        // 3 2 1
        int32_t inUbDim = dimNum_ - ubAxis_;
        scatterParam.axisVl = tdPtr_->inShape[dimNum_ - CONST2];
        scatterParam.strideOutVl = tdPtr_->outStride[dimNum_ - CONST2];
        scatterParam.strideInVl = tdPtr_->inStride[dimNum_ - CONST2];
        if (inUbDim == CONST4) {
            scatterParam.axisVlO2 = ubAxisInCopyNum;
            scatterParam.axisVlO1 = tdPtr_->inShape[dimNum_ - CONST3];
            scatterParam.strideOutVlO2 = tdPtr_->outStride[dimNum_ - CONST4];
            scatterParam.strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST3];
            scatterParam.strideInVlO2 = tdPtr_->inStride[dimNum_ - CONST4];
            scatterParam.strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST3];
        } else if (inUbDim == CONST3) {
            scatterParam.axisVlO1 = ubAxisInCopyNum;
            scatterParam.strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST3];
            scatterParam.strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST3];
        } else if (inUbDim == CONST2) {
            scatterParam.axisVl = ubAxisInCopyNum;
        }

        for (int32_t i = dimNum_ - CONST2; i > ubAxis_; i--) {
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

    __aicore__ inline void GetScatterParamThreeDim(const uint64_t* outIndex, uint32_t ubAxisInCopyNum,
                                                   uint32_t& outUbStart, PadScatterParam& scatterParam)
    {
        //   N C HW -- VL循环在C上，为了复用同一份gather vf代码
        // 3 2 1
        int32_t inUbDim = dimNum_ - ubAxis_;
        scatterParam.axisVl = tdPtr_->inShape[dimNum_ - CONST3];
        scatterParam.strideOutVl = tdPtr_->outStride[dimNum_ - CONST3];
        scatterParam.strideInVl = tdPtr_->inStride[dimNum_ - CONST3];
        if (inUbDim == CONST4) {
            scatterParam.axisVlO1 = ubAxisInCopyNum;
            scatterParam.strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST4];
            scatterParam.strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST4];
        } else if (inUbDim == CONST3) {
            scatterParam.axisVl = ubAxisInCopyNum;
        }

        for (int32_t i = dimNum_ - CONST3; i > ubAxis_; i--) {
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

    __aicore__ inline void GenScatterIndex(LocalTensor<RangeType>& idxTensor)
    {
        int32_t startValue = lastThirdDimInVL_ ? (tdPtr_->leftPad[dimNum_ - CONST2] * tdPtr_->outShape[dimNum_ - 1] +
                                                  tdPtr_->leftPad[dimNum_ - 1]) :
                                                 (tdPtr_->leftPad[dimNum_ - 1]);
        uint16_t loop0 = lastThirdDimInVL_ ? vlSplitIn_ : 1;
        uint16_t loop1 = lastThirdDimInVL_ ? tdPtr_->inShape[dimNum_ - CONST2] : vlSplitIn_;
        int32_t lastDimIn = tdPtr_->inShape[dimNum_ - 1];
        int32_t inStride0 = lastThirdDimInVL_ ? tdPtr_->inStride[dimNum_ - CONST3] : 1; // 防止 dimNum_ < 3
        int32_t inStride1 = tdPtr_->inStride[dimNum_ - CONST2];
        int32_t outStride0 = lastThirdDimInVL_ ? tdPtr_->outStride[dimNum_ - CONST3] : 1;
        int32_t outStride1 = tdPtr_->outStride[dimNum_ - CONST2];

        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> indexReg;
            Reg::RegTensor<RangeType> validReg;
            Reg::UnalignReg uReg;

            for (uint16_t i = 0; i < loop0; i++) {
                for (uint16_t j = 0; j < loop1; j++) {
                    __local_mem__ RangeType* idxAddrTmp = idxAddr + i * inStride0 + j * inStride1;
                    Reg::Arange(validReg, startValue + i * outStride0 + j * outStride1);
                    Reg::DataCopyUnAlign(idxAddrTmp, validReg, uReg, lastDimIn);
                    Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);
                }
            }
        }
    }

    __aicore__ inline void ScatterProcess(const PadScatterParam& scatterParam, const LocalTensor<RangeType>& idxTensor,
                                          LocalTensor<T>& inTensor, LocalTensor<T>& outTensor, uint32_t outUbStart)
    {
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr();

        uint32_t vlSplitLoopIn = vlSplitIn_;
        if constexpr (sizeof(T) == 1) {
            vlSplitLoopIn /= sizeof(int16_t);
        }
        if (vlSplitLoopIn == 0) {
            vlSplitLoopIn = 1;
        }

        RangeType idxOffset = scatterParam.strideOutVl * vlSplitLoopIn;
        uint32_t maskValue = scatterParam.strideInVl * vlSplitLoopIn;

        uint16_t vlSplitLoopCnt = scatterParam.axisVl / vlSplitLoopIn;
        uint16_t vlSplitTailCnt = scatterParam.axisVl - vlSplitLoopCnt * vlSplitLoopIn;
        uint16_t vlSplitTailLoopCnt = vlSplitTailCnt == 0 ? 0 : 1;
        uint32_t maskValueTail = vlSplitTailCnt * scatterParam.strideInVl;

        uint16_t axisVlO2 = scatterParam.axisVlO2;
        uint16_t axisVlO1 = scatterParam.axisVlO1;
        RangeType strideOutVlO2 = scatterParam.strideOutVlO2;
        RangeType strideOutVlO1 = scatterParam.strideOutVlO1;
        uint32_t strideInVlO2 = scatterParam.strideInVlO2;
        uint32_t strideInVlO1 = scatterParam.strideInVlO1;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdx;
            Reg::RegTensor<RangeType> regIdxBK;
            Reg::RegTensor<RangeType> regNewIdx;
            Reg::RegTensor<T> regData;
            Reg::RegTensor<T> regDataT;
            uint32_t main = maskValue;
            uint32_t tail = maskValueTail;
            Reg::MaskReg maskMain = Reg::UpdateMask<CastType>(main);
            Reg::MaskReg maskTail = Reg::UpdateMask<CastType>(tail);
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskData;
            Reg::MaskReg pregT;
            Reg::UnalignReg uReg;

            Reg::DataCopy(regIdx, idxAddr);

            for (uint16_t nIdx = 0; nIdx < axisVlO2; nIdx++) {
                for (uint16_t cIdx = 0; cIdx < axisVlO1; cIdx++) {
                    __local_mem__ T* inAddrTmp = inAddr + nIdx * strideInVlO2 + cIdx * strideInVlO1;
                    RangeType addsScale = nIdx * strideOutVlO2 + cIdx * strideOutVlO1 + outUbStart;
                    Reg::DataCopyUnAlignPre(uReg, inAddrTmp);
                    for (uint16_t hIdx = 0; hIdx < vlSplitLoopCnt; hIdx++) {
                        Reg::Adds(regIdxBK, regIdx, (RangeType)(hIdx * idxOffset + addsScale), maskIdx);

                        Reg::DataCopyUnAlign(regData, uReg, inAddrTmp, maskValue); // maskValue 实际搬入的长度
                        if constexpr (sizeof(T) != 1) {
                            Reg::DataCopyScatter(outAddr, regData, (Reg::RegTensor<IdxType>&)regIdxBK, maskMain);
                        } else {
                            Reg::UnPack((Reg::RegTensor<CastType>&)regDataT, regData);
                            Reg::DataCopyScatter(outAddr, regDataT, (Reg::RegTensor<IdxType>&)regIdxBK, maskMain);
                        }
                    }

                    for (uint16_t hTail = 0; hTail < vlSplitTailLoopCnt; hTail++) {
                        inAddrTmp = inAddr + nIdx * strideInVlO2 + cIdx * strideInVlO1 + vlSplitLoopCnt * maskValue;
                        Reg::DataCopyUnAlignPre(uReg, inAddrTmp);
                        Reg::Adds(regIdxBK, regIdx, (RangeType)(vlSplitLoopCnt * idxOffset + addsScale), maskIdx);
                        Reg::DataCopyUnAlign(regData, uReg, inAddrTmp,
                                             maskValueTail); // maskValueTail 实际搬入的长度
                        if constexpr (sizeof(T) != 1) {
                            Reg::DataCopyScatter(outAddr, regData, (Reg::RegTensor<IdxType>&)regIdxBK, maskTail);
                        } else {
                            Reg::UnPack((Reg::RegTensor<CastType>&)regDataT, regData);
                            Reg::DataCopyScatter(outAddr, regDataT, (Reg::RegTensor<IdxType>&)regIdxBK, maskTail);
                        }
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
    GlobalTensor<T> constValueGM_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> idxBuf_;

    int32_t blockIdx_{0};
    int32_t dimNum_{0};
    int32_t ubAxis_{0};
    int32_t ubFactor_{0};
    uint16_t vlSplitIn_{0};        // VL切分轴的factor
    bool lastThirdDimInVL_{false}; // 一次VL是否处理后面三根轴

    T constValue_{0};
};
} // namespace PadV3

#endif
