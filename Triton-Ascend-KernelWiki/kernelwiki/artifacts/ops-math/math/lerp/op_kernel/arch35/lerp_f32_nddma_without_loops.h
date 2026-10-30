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
 * \file lerp_f32_nddma_without_loops.h
 * \brief
 */
#ifndef ASCENDC_LERP_F32_NDDMA_WITHOUT_LOOPS_H_
#define ASCENDC_LERP_F32_NDDMA_WITHOUT_LOOPS_H_

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

// start is float32, end is float32, weight is float32, y is float32, max dims in ub is 5 and nddma does not need loops
class LerpF32NddmaWithoutLoops {
public:
    __aicore__ inline LerpF32NddmaWithoutLoops(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace,
                                const LerpTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmStart_.SetGlobalBuffer((__gm__ float*)start);
        inputGmEnd_.SetGlobalBuffer((__gm__ float*)end);
        inputGmWeight_.SetGlobalBuffer((__gm__ float*)weight);
        outputGmY_.SetGlobalBuffer((__gm__ float*)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(float);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn2_, DOUBLE_BUFFER, BUFFER_SIZE_0);
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
        bufferIn1_ = queIn1_.AllocTensor<float>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmEnd_, bufferIn1_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<float>(bufferIn1_);
    }

    __aicore__ inline void CopyIn2(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn2_ = queIn2_.AllocTensor<float>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmStart_, bufferIn2_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn2_.EnQue<float>(bufferIn2_);
    }

    __aicore__ inline void Compute3(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<float>();
        bufferIn1_ = queIn1_.DeQue<float>();
        bufferIn2_ = queIn2_.DeQue<float>();
        bufferOut0_ = queOut0_.AllocTensor<float>();
        __VEC_SCOPE__
        {
            RegTensor<float> vreg0;
            RegTensor<float> vreg1;
            RegTensor<float> vreg2;
            RegTensor<float> vreg3;
            RegTensor<float> vreg4;
            RegTensor<float> vreg5;
            RegTensor<float> vreg6;
            MaskReg preg0;
            uint32_t size = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis];
            MaskReg preg26;
            preg0 = AscendC::Reg::CreateMask<float>();
            AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vreg4, static_cast<float>(1.0),
                                                                                        preg0);
            uint16_t vfLoopNum = (ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] +
                                  (AscendC::VECTOR_REG_WIDTH / 4) - 1) /
                                 (AscendC::VECTOR_REG_WIDTH / 4);
            __local_mem__ float* bufferIn0Addr = (__local_mem__ float*)bufferIn0_.GetPhyAddr();
            __local_mem__ float* bufferIn1Addr = (__local_mem__ float*)bufferIn1_.GetPhyAddr();
            __local_mem__ float* bufferIn2Addr = (__local_mem__ float*)bufferIn2_.GetPhyAddr();
            __local_mem__ float* bufferOut0Addr = (__local_mem__ float*)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::Reg::UpdateMask<float>(size);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(preg26, vreg0, static_cast<float>(0.5), preg0);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vreg1, bufferIn1Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vreg2, bufferIn2Addr + i * (AscendC::VECTOR_REG_WIDTH / 4));
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg3, vreg1, vreg2, preg0);
                AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg2, vreg0, vreg3, preg0);
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg5, vreg0, vreg4, preg0);
                AscendC::Reg::MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(vreg1, vreg5, vreg3, preg0);
                AscendC::Reg::Select<float>(vreg6, vreg2, vreg1, preg26);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / 4), vreg6, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queIn2_.FreeTensor(bufferIn2_);
        queOut0_.EnQue<float>(bufferOut0_);
    }

    __aicore__ inline void CopyOut4(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<float>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] *
                                     sizeof(float);
        int64_t gmOffset = Ops::Base::BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides,
                                                           tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmY_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const LerpTilingData* tilingDataPtr_;
    GlobalTensor<float> inputGmStart_;
    GlobalTensor<float> inputGmEnd_;
    GlobalTensor<float> inputGmWeight_;
    GlobalTensor<float> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn2_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<float> bufferIn0_;
    LocalTensor<float> bufferIn1_;
    LocalTensor<float> bufferIn2_;
    LocalTensor<float> bufferOut0_;
};

} // namespace Lerp
#endif // ASCENDC_LERP_F32_NDDMA_WITHOUT_LOOPS_H_
