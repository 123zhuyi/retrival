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
 * \file floor_bf16.h
 * \brief
 */
#ifndef ASCENDC_FLOOR_BF16_H_
#define ASCENDC_FLOOR_BF16_H_

#include "kernel_operator.h"
const uint16_t BF16_SIGN1 = 0x8000;
const uint16_t BF16_SIGN2 = 0x7FFF;
namespace Floor {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
// x is bfloat16, y is bfloat16
class FloorBf16 {
public:
    __aicore__ inline FloorBf16(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const FloorTilingData* tilingDataPtr,
                                TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmX_.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        outputGmY_.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(bfloat16_t);
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
        bufferIn0_ = queIn0_.AllocTensor<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        AscendC::DataCopyPadExtParams<bfloat16_t> dataCopyPadExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(bfloat16_t);
        AscendC::DataCopyPad(
            bufferIn0_[0],
            inputGmX_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            dataCopyExtParams, dataCopyPadExtParams);
        queIn0_.EnQue<bfloat16_t>(bufferIn0_);
    }

    __aicore__ inline void Compute1(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<bfloat16_t>();
        bufferOut0_ = queOut0_.AllocTensor<bfloat16_t>();
        __VEC_SCOPE__
        {
            RegTensor<bfloat16_t> vreg0;
            RegTensor<bfloat16_t> vreg1;
            RegTensor<uint16_t> vreg2;
            RegTensor<uint16_t> vreg3;
            MaskReg preg0;
            uint32_t size = i0Extent;
            uint16_t vfLoopNum = (i0Extent + (AscendC::VECTOR_REG_WIDTH / sizeof(half)) - 1) /
                                 (AscendC::VECTOR_REG_WIDTH / sizeof(half));
            __local_mem__ bfloat16_t* bufferIn0Addr = (__local_mem__ bfloat16_t*)bufferIn0_.GetPhyAddr();
            __local_mem__ bfloat16_t* bufferOut0Addr = (__local_mem__ bfloat16_t*)bufferOut0_.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = AscendC::Reg::UpdateMask<bfloat16_t>(size);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_NORM>(
                    vreg0, bufferIn0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(half)));
                AscendC::Reg::Truncate<bfloat16_t, AscendC::RoundMode::CAST_FLOOR,
                                       AscendC::Reg::MaskMergeMode::ZEROING>(vreg1, vreg0, preg0);
                AscendC::Reg::Duplicate(vreg2, BF16_SIGN1, preg0);
                AscendC::Reg::Duplicate(vreg3, BF16_SIGN2, preg0);
                AscendC::Reg::And(vreg2, vreg2, (RegTensor<uint16_t>&)vreg0, preg0);
                AscendC::Reg::And(vreg3, vreg3, (RegTensor<uint16_t>&)vreg1, preg0);
                AscendC::Reg::Or(vreg3, vreg2, vreg3, preg0);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(
                    bufferOut0Addr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(half)), (RegTensor<bfloat16_t>&)vreg3,
                    preg0);
            }
        }
        queIn0_.FreeTensor(bufferIn0_);
        queOut0_.EnQue<bfloat16_t>(bufferOut0_);
    }

    __aicore__ inline void CopyOut2(int64_t i0Extent, int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = i0Extent * sizeof(bfloat16_t);
        AscendC::DataCopyPad(
            outputGmY_[tilingDataPtr_->blockFormer * AscendC::GetBlockIdx() + ubLoopIdx * tilingDataPtr_->ubFormer],
            bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const FloorTilingData* tilingDataPtr_;
    GlobalTensor<bfloat16_t> inputGmX_;
    GlobalTensor<bfloat16_t> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<bfloat16_t> bufferIn0_;
    LocalTensor<bfloat16_t> bufferOut0_;
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_FLOOR};
};

} // namespace Floor
#endif // ASCENDC_FLOOR_BF16_H_
