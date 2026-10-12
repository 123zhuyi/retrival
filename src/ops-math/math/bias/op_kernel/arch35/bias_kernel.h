/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIAS_KERNEL_H_
#define BIAS_KERNEL_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"
#include "bias_tiling_data.h"

namespace BiasOp {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

template <typename T, bool NeedCast>
class BiasKernel {
public:
    __aicore__ inline BiasKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const BiasTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmX_.SetGlobalBuffer((__gm__ T*)x);
        inputGmBias_.SetGlobalBuffer((__gm__ T*)bias);
        outputGmY_.SetGlobalBuffer((__gm__ T*)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t bufferNum = tilingDataPtr_->elemNum;
        int64_t bufferSize = bufferNum * sizeof(T);
        pipePtr_->InitBuffer(queInX_, DOUBLE_BUFFER, bufferSize);
        pipePtr_->InitBuffer(queInBias_, DOUBLE_BUFFER, bufferSize);
        pipePtr_->InitBuffer(queOutY_, DOUBLE_BUFFER, bufferSize);
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
            CopyInX(ubSplitSize, axesIndices, ubLoopIdx);
            CopyInBias(ubSplitSize, axesIndices, ubLoopIdx);
            Compute(ubSplitSize);
            CopyOut(ubSplitSize, axesIndices);
        }
    }

private:
    __aicore__ inline void CopyInX(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferInX_ = queInX_.AllocTensor<T>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            if (tilingDataPtr_->shapeLen <= Ops::Base::NDDMA_MAX_DIMS) {
                Ops::Base::BroadcastNddmaWithoutLoop(inputGmX_, bufferInX_, tilingDataPtr_->outputDims,
                                                     tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides,
                                                     axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                     ubSplitSize, tilingDataPtr_->ubFormer);
            } else {
                Ops::Base::BroadcastNddmaWithLoop(inputGmX_, bufferInX_, tilingDataPtr_->outputDims,
                                                  tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides,
                                                  axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                  ubSplitSize, tilingDataPtr_->ubFormer);
            }
        }
        queInX_.EnQue<T>(bufferInX_);
    }

    __aicore__ inline void CopyInBias(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                      int64_t ubLoopIdx)
    {
        bufferInBias_ = queInBias_.AllocTensor<T>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            if (tilingDataPtr_->shapeLen <= Ops::Base::NDDMA_MAX_DIMS) {
                Ops::Base::BroadcastNddmaWithoutLoop(inputGmBias_, bufferInBias_, tilingDataPtr_->outputDims,
                                                     tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides,
                                                     axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                     ubSplitSize, tilingDataPtr_->ubFormer);
            } else {
                Ops::Base::BroadcastNddmaWithLoop(inputGmBias_, bufferInBias_, tilingDataPtr_->outputDims,
                                                  tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides,
                                                  axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                  ubSplitSize, tilingDataPtr_->ubFormer);
            }
        }
        queInBias_.EnQue<T>(bufferInBias_);
    }

    __aicore__ inline void Compute(int64_t ubSplitSize)
    {
        bufferInX_ = queInX_.DeQue<T>();
        bufferInBias_ = queInBias_.DeQue<T>();
        bufferOutY_ = queOutY_.AllocTensor<T>();
        __VEC_SCOPE__
        {
            uint32_t size = static_cast<uint32_t>(ubSplitSize *
                                                  tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis]);
            __ubuf__ T* xAddr = (__ubuf__ T*)bufferInX_.GetPhyAddr();
            __ubuf__ T* biasAddr = (__ubuf__ T*)bufferInBias_.GetPhyAddr();
            __ubuf__ T* yAddr = (__ubuf__ T*)bufferOutY_.GetPhyAddr();
            if constexpr (NeedCast) {
                MaskReg preg0 = AscendC::Reg::CreateMask<float>();
                constexpr int64_t regElemNum = AscendC::VECTOR_REG_WIDTH / sizeof(float);
                uint16_t vfLoopNum = (size + regElemNum - 1) / regElemNum;
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = AscendC::Reg::UpdateMask<float>(size);
                    RegTensor<T> vregX;
                    RegTensor<T> vregBias;
                    RegTensor<float> vregXf;
                    RegTensor<float> vregBiasf;
                    RegTensor<float> vregYf;
                    RegTensor<T> vregY;
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xAddr + i * regElemNum);
                    AscendC::Reg::Cast<float, T, castTraitIn>(vregXf, vregX, preg0);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBias,
                                                                                        biasAddr + i * regElemNum);
                    AscendC::Reg::Cast<float, T, castTraitIn>(vregBiasf, vregBias, preg0);
                    AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregYf, vregXf, vregBiasf, preg0);
                    AscendC::Reg::Cast<T, float, castTraitOut>(vregY, vregYf, preg0);
                    AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr + i * regElemNum, vregY,
                                                                                        preg0);
                }
            } else {
                MaskReg preg0 = AscendC::Reg::CreateMask<T>();
                constexpr int64_t regElemNum = AscendC::VECTOR_REG_WIDTH / sizeof(T);
                uint16_t vfLoopNum = (size + regElemNum - 1) / regElemNum;
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    preg0 = AscendC::Reg::UpdateMask<T>(size);
                    RegTensor<T> vregX;
                    RegTensor<T> vregBias;
                    RegTensor<T> vregY;
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(vregX, xAddr + i * regElemNum);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(vregBias, biasAddr + i * regElemNum);
                    AscendC::Reg::Add<T, AscendC::Reg::MaskMergeMode::ZEROING>(vregY, vregX, vregBias, preg0);
                    AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * regElemNum, vregY,
                                                                                        preg0);
                }
            }
        }
        queInX_.FreeTensor(bufferInX_);
        queInBias_.FreeTensor(bufferInBias_);
        queOutY_.EnQue<T>(bufferOutY_);
    }

    __aicore__ inline void CopyOut(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS])
    {
        bufferOutY_ = queOutY_.DeQue<T>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] *
                                     sizeof(T);
        int64_t gmOffset = Ops::Base::BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides,
                                                           tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmY_[gmOffset], bufferOutY_[0], dataCopyExtParams);
        queOutY_.FreeTensor(bufferOutY_);
    }

private:
    TPipe* pipePtr_;
    const BiasTilingData* tilingDataPtr_;
    GlobalTensor<T> inputGmX_;
    GlobalTensor<T> inputGmBias_;
    GlobalTensor<T> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queInX_;
    TQue<AscendC::QuePosition::VECIN, 1> queInBias_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOutY_;
    LocalTensor<T> bufferInX_;
    LocalTensor<T> bufferInBias_;
    LocalTensor<T> bufferOutY_;
    constexpr static AscendC::Reg::CastTrait castTraitIn = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTraitOut = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
};
} // namespace BiasOp

#endif // BIAS_KERNEL_H_
