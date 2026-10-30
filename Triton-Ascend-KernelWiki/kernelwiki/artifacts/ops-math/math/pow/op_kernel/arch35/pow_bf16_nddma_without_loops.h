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
 * \file pow_bf16_nddma_without_loops.h
 * \brief
 */
#ifndef ASCENDC_POW_BF16_NDDMA_WITHOUT_LOOPS_H_
#define ASCENDC_POW_BF16_NDDMA_WITHOUT_LOOPS_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"

namespace Pow
{
using namespace Ops::Base;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

// input_x is bfloat16, input_y is bfloat16, output_z is bfloat16, max dims in ub is 5 and nddma does not need loops
class PowBf16NddmaWithoutLoops
{
public:
    __aicore__ inline PowBf16NddmaWithoutLoops(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR inputY, GM_ADDR outputZ, GM_ADDR workspace,
                                const PowTensorTensorTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmInputX_.SetGlobalBuffer((__gm__ bfloat16_t*)inputX);
        inputGmInputY_.SetGlobalBuffer((__gm__ bfloat16_t*)inputY);
        outputGmOutputZ_.SetGlobalBuffer((__gm__ bfloat16_t*)outputZ);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(bfloat16_t);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->blockTail
                                                                                 : tilingDataPtr_->blockFormer;
        int64_t axesIndices[BROADCAST_MAX_DIMS] = {0};
        BroadcastGetAxesIndices(axesIndices, tilingDataPtr_->blockFormer * AscendC::GetBlockIdx(),
                                tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis,
                                tilingDataPtr_->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            if (ubLoopIdx != 0) {
                BroadcastUpdateAxesIndices(axesIndices, tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis,
                                           tilingDataPtr_->ubOuter);
            }
            int64_t ubSplitSize = axesIndices[tilingDataPtr_->ubSplitAxis] == tilingDataPtr_->ubOuter - 1
                                      ? tilingDataPtr_->ubTail
                                      : tilingDataPtr_->ubFormer;
            CopyIn0(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn1(ubSplitSize, axesIndices, ubLoopIdx);
            Compute2(ubSplitSize, axesIndices, ubLoopIdx);
            CopyOut3(ubSplitSize, axesIndices, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<bfloat16_t>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            BroadcastNddmaWithoutLoop(inputGmInputX_, bufferIn0_, tilingDataPtr_->outputDims,
                                      tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides, axesIndices,
                                      tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen, ubSplitSize,
                                      tilingDataPtr_->ubFormer);
        }
        queIn0_.EnQue<bfloat16_t>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<bfloat16_t>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            BroadcastNddmaWithoutLoop(inputGmInputY_, bufferIn1_, tilingDataPtr_->outputDims,
                                      tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides, axesIndices,
                                      tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen, ubSplitSize,
                                      tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<bfloat16_t>(bufferIn1_);
    }

    __aicore__ inline void Compute2(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<bfloat16_t>();
        bufferIn1_ = queIn1_.DeQue<bfloat16_t>();
        bufferOut0_ = queOut0_.AllocTensor<bfloat16_t>();
        Power<bfloat16_t, false, pConfig_>(bufferOut0_, bufferIn0_, bufferIn1_,
                                          ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis]);
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queOut0_.EnQue<bfloat16_t>(bufferOut0_);
    }

    __aicore__ inline void CopyOut3(int64_t ubSplitSize, const int64_t (&axesIndices)[BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen =
            ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] * sizeof(bfloat16_t);
        int64_t gmOffset = BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides, tilingDataPtr_->ubSplitAxis,
                                                tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmOutputZ_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const PowTensorTensorTilingData* tilingDataPtr_;
    GlobalTensor<bfloat16_t> inputGmInputX_;
    GlobalTensor<bfloat16_t> inputGmInputY_;
    GlobalTensor<bfloat16_t> outputGmOutputZ_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<bfloat16_t> bufferIn0_;
    LocalTensor<bfloat16_t> bufferIn1_;
    LocalTensor<bfloat16_t> bufferOut0_;
    constexpr static PowerConfig pConfig_ = {PowerAlgo::DOUBLE_FLOAT_TECH};
};

}  // namespace Pow
#endif  // ASCENDC_POW_BF16_NDDMA_WITHOUT_LOOPS_H_
