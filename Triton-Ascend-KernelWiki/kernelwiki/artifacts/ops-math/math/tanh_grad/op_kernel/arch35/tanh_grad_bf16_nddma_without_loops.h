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
 * \file tanh_grad_bf16_nddma_without_loops.h
 * \brief
 */
#ifndef ASCENDC_TANH_GRAD_BF16_NDDMA_WITHOUT_LOOPS_H_
#define ASCENDC_TANH_GRAD_BF16_NDDMA_WITHOUT_LOOPS_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"

namespace TanhGrad {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

// y is bfloat16, dy is bfloat16, z is bfloat16, max dims in ub is 5 and nddma does not need loops
class TanhGradBf16NddmaWithoutLoops {
public:
    __aicore__ inline TanhGradBf16NddmaWithoutLoops(){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace,
                                const TanhGradTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmY_.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        inputGmDy_.SetGlobalBuffer((__gm__ bfloat16_t*)dy);
        outputGmZ_.SetGlobalBuffer((__gm__ bfloat16_t*)z);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(bfloat16_t);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->blockTail :
                                                                                   tilingDataPtr_->blockFormer;
        int64_t axesIndices[Ops::Base::BROADCAST_MAX_DIMS] = {0};
        Ops::Base::BroadcastGetAxesIndices(axesIndices, tilingDataPtr_->blockFormer * AscendC::GetBlockIdx(),
                                           tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis,
                                           tilingDataPtr_->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            if (ubLoopIdx != 0) {
                Ops::Base::BroadcastUpdateAxesIndices(axesIndices, tilingDataPtr_->outputDims,
                                                      tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubOuter);
            }
            int64_t ubSplitSize = axesIndices[tilingDataPtr_->ubSplitAxis] == tilingDataPtr_->ubOuter - 1 ?
                                      tilingDataPtr_->ubTail :
                                      tilingDataPtr_->ubFormer;
            CopyIn0(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn1(ubSplitSize, axesIndices, ubLoopIdx);
            Compute2(ubSplitSize, axesIndices, ubLoopIdx);
            CopyOut3(ubSplitSize, axesIndices, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<bfloat16_t>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmY_, bufferIn0_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn0_.EnQue<bfloat16_t>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<bfloat16_t>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmDy_, bufferIn1_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<bfloat16_t>(bufferIn1_);
    }

    __aicore__ inline void Compute2(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<bfloat16_t>();
        bufferIn1_ = queIn1_.DeQue<bfloat16_t>();
        bufferOut0_ = queOut0_.AllocTensor<bfloat16_t>();
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> vreg0;
            RegTensor<float> vreg1;
            RegTensor<float> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;
            RegTensor<bfloat16_t> vreg5;
            RegTensor<float> vreg6;
            RegTensor<float> vreg7;
            RegTensor<bfloat16_t> vreg8;
            MaskReg preg0;
            uint32_t size = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis];
            uint16_t vfLoopNum = (ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] +
                                  (AscendC::VECTOR_REG_WIDTH / 4) - 1) /
                                 (AscendC::VECTOR_REG_WIDTH / 4);
            __local_mem__ bfloat16_t* bufferIn0Addr = (__local_mem__ bfloat16_t*)bufferIn0_.GetPhyAddr();
            __local_mem__ bfloat16_t* bufferIn1Addr = (__local_mem__ bfloat16_t*)bufferIn1_.GetPhyAddr();
            __local_mem__ bfloat16_t* bufferOut0Addr = (__local_mem__ bfloat16_t*)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::Reg::UpdateMask<float>(size);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::Cast<float, bfloat16_t, castTrait0>(vreg1, vreg0, preg0);
                AscendC::Reg::Duplicate<float, float>(vreg4, 1.0);
                AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg1,
                                                                                       static_cast<float>(-1), preg0);
                AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg2, vreg1, preg0);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg5, bufferIn1Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::Cast<float, bfloat16_t, castTrait0>(vreg6, vreg5, preg0);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg4, vreg6, preg0);
                AscendC::Reg::Cast<bfloat16_t, float, castTrait1>(vreg8, vreg7, preg0);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg8, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queOut0_.EnQue<bfloat16_t>(bufferOut0_);
    }

    __aicore__ inline void CopyOut3(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] *
                                     sizeof(bfloat16_t);
        int64_t gmOffset = Ops::Base::BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides,
                                                           tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmZ_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const TanhGradTilingData* tilingDataPtr_;
    GlobalTensor<bfloat16_t> inputGmY_;
    GlobalTensor<bfloat16_t> inputGmDy_;
    GlobalTensor<bfloat16_t> outputGmZ_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<bfloat16_t> bufferIn0_;
    LocalTensor<bfloat16_t> bufferIn1_;
    LocalTensor<bfloat16_t> bufferOut0_;
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};
};

} // namespace TanhGrad
#endif // ASCENDC_TANH_GRAD_BF16_NDDMA_WITHOUT_LOOPS_H_
