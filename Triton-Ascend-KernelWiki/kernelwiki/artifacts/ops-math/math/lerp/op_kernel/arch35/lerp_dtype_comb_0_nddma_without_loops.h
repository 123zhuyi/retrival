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
 * \file lerp_dtype_comb_0_nddma_without_loops.h
 * \brief
 */
#ifndef ASCENDC_LERP_DTYPE_COMB_0_NDDMA_WITHOUT_LOOPS_H_
#define ASCENDC_LERP_DTYPE_COMB_0_NDDMA_WITHOUT_LOOPS_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"

namespace Lerp {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

// start is bfloat16, end is bfloat16, weight is float32, y is bfloat16, max dims in ub is 5 and nddma does not need
// loops
class LerpDtypeComb0NddmaWithoutLoops {
public:
    __aicore__ inline LerpDtypeComb0NddmaWithoutLoops(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace,
                                const LerpTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmStart_.SetGlobalBuffer((__gm__ bfloat16_t*)start);
        inputGmEnd_.SetGlobalBuffer((__gm__ bfloat16_t*)end);
        inputGmWeight_.SetGlobalBuffer((__gm__ float*)weight);
        outputGmY_.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(float);
        int64_t BUFFER_SIZE_1 = tilingDataPtr_->elemNum * sizeof(bfloat16_t);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_1);
        pipePtr_->InitBuffer(queIn2_, DOUBLE_BUFFER, BUFFER_SIZE_1);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_1);
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
            CopyIn2(ubSplitSize, axesIndices, ubLoopIdx);
            Compute3(ubSplitSize, axesIndices, ubLoopIdx);
            CopyOut4(ubSplitSize, axesIndices, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<float>();
        if ((tilingDataPtr_->input2Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmWeight_, bufferIn0_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input2Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn0_.EnQue<float>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<bfloat16_t>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmEnd_, bufferIn1_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<bfloat16_t>(bufferIn1_);
    }

    __aicore__ inline void CopyIn2(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn2_ = queIn2_.AllocTensor<bfloat16_t>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmStart_, bufferIn2_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn2_.EnQue<bfloat16_t>(bufferIn2_);
    }

    __aicore__ inline void Compute3(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<float>();
        bufferIn1_ = queIn1_.DeQue<bfloat16_t>();
        bufferIn2_ = queIn2_.DeQue<bfloat16_t>();
        bufferOut0_ = queOut0_.AllocTensor<bfloat16_t>();
        __VEC_SCOPE__
        {
            RegTensor<float> vreg0;
            RegTensor<bfloat16_t> vreg1;
            RegTensor<float> vreg2;
            RegTensor<bfloat16_t> vreg3;
            RegTensor<float> vreg4;
            RegTensor<float> vreg5;
            RegTensor<float> vreg6;
            RegTensor<float> vreg7;
            RegTensor<float> vreg8;
            RegTensor<bfloat16_t> vreg9;
            MaskReg preg0;
            uint32_t size = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis];
            MaskReg preg73;
            preg0 = AscendC::Reg::CreateMask<float>();
            AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg6, static_cast<float>(1.0),
                                                                                        preg0);
            uint16_t vfLoopNum = (ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] +
                                  (AscendC::VECTOR_REG_WIDTH / 4) - 1) /
                                 (AscendC::VECTOR_REG_WIDTH / 4);
            __local_mem__ bfloat16_t* bufferIn1Addr = (__local_mem__ bfloat16_t*)bufferIn1_.GetPhyAddr();
            __local_mem__ bfloat16_t* bufferIn2Addr = (__local_mem__ bfloat16_t*)bufferIn2_.GetPhyAddr();
            __local_mem__ bfloat16_t* bufferOut0Addr = (__local_mem__ bfloat16_t*)bufferOut0_.GetPhyAddr();
            __local_mem__ float* bufferIn0Addr = (__local_mem__ float*)bufferIn0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::Reg::UpdateMask<float>(size);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(preg73, vreg0, static_cast<float>(0.5), preg0);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg1, bufferIn1Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::Cast<float, bfloat16_t, castTrait0>(vreg2, vreg1, preg0);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    vreg3, bufferIn2Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::Cast<float, bfloat16_t, castTrait0>(vreg4, vreg3, preg0);
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg2, vreg4, preg0);
                AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg4, vreg0, vreg5, preg0);
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg7, vreg0, vreg6, preg0);
                AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg7, vreg5, preg0);
                AscendC::Reg::Select<float>(vreg8, vreg4, vreg2, preg73);
                AscendC::Reg::Cast<bfloat16_t, float, castTrait1>(vreg9, vreg8, preg0);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg9, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queIn2_.FreeTensor(bufferIn2_);
        queOut0_.EnQue<bfloat16_t>(bufferOut0_);
    }

    __aicore__ inline void CopyOut4(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] *
                                     sizeof(bfloat16_t);
        int64_t gmOffset = Ops::Base::BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides,
                                                           tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmY_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const LerpTilingData* tilingDataPtr_;
    GlobalTensor<bfloat16_t> inputGmStart_;
    GlobalTensor<bfloat16_t> inputGmEnd_;
    GlobalTensor<float> inputGmWeight_;
    GlobalTensor<bfloat16_t> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn2_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<float> bufferIn0_;
    LocalTensor<bfloat16_t> bufferIn1_;
    LocalTensor<bfloat16_t> bufferIn2_;
    LocalTensor<bfloat16_t> bufferOut0_;
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};
};

} // namespace Lerp
#endif // ASCENDC_LERP_DTYPE_COMB_0_NDDMA_WITHOUT_LOOPS_H_
