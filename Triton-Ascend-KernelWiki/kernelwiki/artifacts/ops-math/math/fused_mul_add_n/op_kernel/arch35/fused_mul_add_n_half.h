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
 * \file fused_mul_add_n_half.h
 * \brief
 */
#ifndef ASCENDC_FUSED_MUL_ADD_N_HALF_H_
#define ASCENDC_FUSED_MUL_ADD_N_HALF_H_

#include "kernel_operator.h"

namespace FusedMulAddNOp {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::ToFloat;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

// input_x1 is T, input_x2 is T, input_x3 is T, output_y is T. T is half or bfloat16_t
template <typename T>
class FusedMulAddNHalf {
public:
    __aicore__ inline FusedMulAddNHalf(){};
    __aicore__ inline void Init(GM_ADDR inputX1, GM_ADDR inputX2, GM_ADDR inputX3, GM_ADDR outputY, GM_ADDR workspace,
                                const FusedMulAddNTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;

        inputGmInputX1_.SetGlobalBuffer((__gm__ T*)inputX1);
        inputGmInputX2_.SetGlobalBuffer((__gm__ T*)inputX2);
        inputGmInputX3_.SetGlobalBuffer((__gm__ T*)inputX3, 1);
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            inputX3Value_ = ToFloat<bfloat16_t>(inputGmInputX3_.GetValue(0));
        } else {
            inputX3Value_ = static_cast<float>(inputGmInputX3_.GetValue(0));
        }
        outputGmOutputY_.SetGlobalBuffer((__gm__ T*)outputY);

        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(T);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->ubLoopOfTailBlock :
                                                                                   tilingDataPtr_->ubLoopOfFormerBlock;
        int64_t tailExtent = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->ubTailOfTailBlock :
                                                                                    tilingDataPtr_->ubTailOfFormerBlock;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            int64_t i0Extent = ubLoopIdx == ubLoopNum - 1 ? tailExtent : tilingDataPtr_->ubFormer;
            CopyIn0(i0Extent, ubLoopIdx);
            CopyIn1(i0Extent, ubLoopIdx);
            Compute2(i0Extent, ubLoopIdx);
            CopyOut3(i0Extent, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<T>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(T);
        AscendC::DataCopyPad(bufferIn0_[0],
                             inputGmInputX1_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() +
                                             ubLoopIdx * tilingDataPtr_->ubFormer],
                             dataCopyExtParams, dataCopyPadExtParams);
        queIn0_.EnQue<T>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<T>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(T);
        AscendC::DataCopyPad(bufferIn1_[0],
                             inputGmInputX2_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() +
                                             ubLoopIdx * tilingDataPtr_->ubFormer],
                             dataCopyExtParams, dataCopyPadExtParams);
        queIn1_.EnQue<T>(bufferIn1_);
    }

    __aicore__ inline void Compute2(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<T>();
        bufferIn1_ = queIn1_.DeQue<T>();
        bufferOut0_ = queOut0_.AllocTensor<T>();
        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<float> vreg1;
            RegTensor<T> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;
            RegTensor<T> vreg5;
            MaskReg preg0;
            uint32_t size = i0Extent;
            uint16_t vfLoopNum = (i0Extent + (AscendC::VECTOR_REG_WIDTH / sizeof(float)) - 1) /
                                 (AscendC::VECTOR_REG_WIDTH / sizeof(float));
            __local_mem__ T* bufferIn0Addr = (__local_mem__ T*)bufferIn0_.GetPhyAddr();
            __local_mem__ T* bufferIn1Addr = (__local_mem__ T*)bufferIn1_.GetPhyAddr();
            __local_mem__ T* bufferOut0Addr = (__local_mem__ T*)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::Reg::UpdateMask<float>(size);
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
                AscendC::Reg::Cast<float, T, castTrait0>(vreg1, vreg0, preg0);
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg2, bufferIn1Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
                AscendC::Reg::Cast<float, T, castTrait0>(vreg3, vreg2, preg0);
                AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg1, vreg1, inputX3Value_,
                                                                                       preg0);
                AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg1, vreg3, preg0);
                AscendC::Reg::Cast<T, float, castTrait1>(vreg5, vreg4, preg0);
                AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)), vreg5, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queOut0_.EnQue<T>(bufferOut0_);
    }

    __aicore__ inline void CopyOut3(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<T>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(T);
        AscendC::DataCopyPad(outputGmOutputY_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() +
                                              ubLoopIdx * tilingDataPtr_->ubFormer],
                             bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const FusedMulAddNTilingData* tilingDataPtr_;
    GlobalTensor<T> inputGmInputX1_;
    GlobalTensor<T> inputGmInputX2_;
    GlobalTensor<T> inputGmInputX3_;
    GlobalTensor<T> outputGmOutputY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<T> bufferIn0_;
    LocalTensor<T> bufferIn1_;
    LocalTensor<T> bufferOut0_;
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};
    float inputX3Value_;
};

} // namespace FusedMulAddNOp
#endif // ASCENDC_FUSED_MUL_ADD_N_HALF_H_
