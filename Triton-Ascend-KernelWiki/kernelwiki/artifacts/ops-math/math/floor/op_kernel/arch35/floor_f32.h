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
 * \file floor_f32.h
 * \brief
 */
#ifndef ASCENDC_FLOOR_F32_H_
#define ASCENDC_FLOOR_F32_H_

#include "kernel_operator.h"
const uint32_t FP32_SIGN1 = 0x80000000;
const uint32_t FP32_SIGN2 = 0x7FFFFFFF;
namespace Floor {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
// x is float32, y is float32
class FloorF32 {
public:
    __aicore__ inline FloorF32(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const FloorTilingData* tilingDataPtr,
                                TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmX_.SetGlobalBuffer((__gm__ float*)x);
        outputGmY_.SetGlobalBuffer((__gm__ float*)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(float);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
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
            Compute1(i0Extent, ubLoopIdx);
            CopyOut2(i0Extent, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<float>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(float);
        AscendC::DataCopyPad(
            bufferIn0_[0],
            inputGmX_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            dataCopyExtParams, dataCopyPadExtParams);
        queIn0_.EnQue<float>(bufferIn0_);
    }

    __aicore__ inline void Compute1(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<float>();
        bufferOut0_ = queOut0_.AllocTensor<float>();
        __VEC_SCOPE__
        {
            RegTensor<float> vreg0;
            RegTensor<float> vreg1;
            RegTensor<uint32_t> vreg2;
            RegTensor<uint32_t> vreg3;
            MaskReg preg0;
            uint32_t size = i0Extent;
            uint16_t vfLoopNum = (i0Extent + (AscendC::VECTOR_REG_WIDTH / sizeof(float)) - 1) /
                                 (AscendC::VECTOR_REG_WIDTH / sizeof(float));
            __local_mem__ float* bufferIn0Addr = (__local_mem__ float*)bufferIn0_.GetPhyAddr();
            __local_mem__ float* bufferOut0Addr = (__local_mem__ float*)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::Reg::UpdateMask<float>(size);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
                AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_FLOOR, AscendC::Reg::MaskMergeMode::ZEROING>(
                    vreg1, vreg0, preg0);
                AscendC::Reg::Duplicate(vreg2, FP32_SIGN1, preg0);
                AscendC::Reg::Duplicate(vreg3, FP32_SIGN2, preg0);
                AscendC::Reg::And(vreg2, vreg2, (RegTensor<uint32_t>&)vreg0, preg0);
                AscendC::Reg::And(vreg3, vreg3, (RegTensor<uint32_t>&)vreg1, preg0);
                AscendC::Reg::Or(vreg3, vreg2, vreg3, preg0);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)), (RegTensor<float>&)vreg3, preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queOut0_.EnQue<float>(bufferOut0_);
    }

    __aicore__ inline void CopyOut2(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<float>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(float);
        AscendC::DataCopyPad(
            outputGmY_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const FloorTilingData* tilingDataPtr_;
    GlobalTensor<float> inputGmX_;
    GlobalTensor<float> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<float> bufferIn0_;
    LocalTensor<float> bufferOut0_;
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_FLOOR};
};

} // namespace Floor
#endif // ASCENDC_FLOOR_F32_H_
