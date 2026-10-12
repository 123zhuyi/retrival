/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file square_sum_all.h
 * \brief SquareSumAll RegBase kernel for Ascend 950 (DAV3510 / arch35).
 */

#ifndef SQUARE_SUM_ALL_ARCH35_H_
#define SQUARE_SUM_ALL_ARCH35_H_

#include "kernel_operator.h"
#include "square_sum_all_tiling_data.h"

namespace SquareSumAllOps {
using namespace AscendC;

class SquareSumAllKernel {
public:
    __aicore__ inline SquareSumAllKernel() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, GM_ADDR userWorkspace,
                                const SquareSumAllTilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        usedCoreNum_ = tilingData_->usedCoreNum;
        tileElements_ = tilingData_->tileElements;

        const int64_t prefixExtra = blockIdx_ < tilingData_->extraCoreCount ? blockIdx_ : tilingData_->extraCoreCount;
        coreStart_ = blockIdx_ * tilingData_->baseCoreElements + prefixExtra;
        coreElements_ = tilingData_->baseCoreElements + (blockIdx_ < tilingData_->extraCoreCount ? 1 : 0);

        x1Gm_.SetGlobalBuffer((__gm__ float*)x1, tilingData_->totalElements);
        x2Gm_.SetGlobalBuffer((__gm__ float*)x2, tilingData_->totalElements);
        y1Gm_.SetGlobalBuffer((__gm__ float*)y1, 1);
        y2Gm_.SetGlobalBuffer((__gm__ float*)y2, 1);

        const int64_t workspaceElementsPerRegion = usedCoreNum_ * FLOATS_PER_BLOCK;
        auto* workspace = (__gm__ float*)userWorkspace;
        partial1Gm_.SetGlobalBuffer(workspace, workspaceElementsPerRegion);
        partial2Gm_.SetGlobalBuffer(workspace + workspaceElementsPerRegion, workspaceElementsPerRegion);

        pipe_->InitBuffer(x1Queue_, DOUBLE_BUFFER, tileElements_ * sizeof(float));
        pipe_->InitBuffer(x2Queue_, DOUBLE_BUFFER, tileElements_ * sizeof(float));
        pipe_->InitBuffer(resultQueue_, 1, RESULT_LOCAL_BYTES);
        const int64_t partialElements = usedCoreNum_ * FLOATS_PER_BLOCK;
        const int64_t partialBufferElements = (partialElements / VECTOR_ELEMENTS +
                                               (partialElements % VECTOR_ELEMENTS != 0)) *
                                              VECTOR_ELEMENTS;
        pipe_->InitBuffer(partialQueue_, 1, partialBufferElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> resultLocal = resultQueue_.AllocTensor<float>();
        const int64_t loopCount = coreElements_ / tileElements_ + (coreElements_ % tileElements_ != 0);
        for (int64_t loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
            const int64_t tileOffset = loopIndex * tileElements_;
            const int64_t remaining = coreElements_ - tileOffset;
            const int64_t currentElements = remaining < tileElements_ ? remaining : tileElements_;
            CopyIn(coreStart_ + tileOffset, currentElements);

            LocalTensor<float> x1Local = x1Queue_.DeQue<float>();
            LocalTensor<float> x2Local = x2Queue_.DeQue<float>();
            ComputeTile(x1Local, x2Local, resultLocal, static_cast<uint32_t>(currentElements), loopIndex == 0);
            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
        }

        resultQueue_.EnQue(resultLocal);
        resultLocal = resultQueue_.DeQue<float>();
        WriteCorePartials(resultLocal);
        resultQueue_.FreeTensor(resultLocal);

        SyncAll();

        if (blockIdx_ == 0) {
            LocalTensor<float> finalLocal = resultQueue_.AllocTensor<float>();
            MergeOneRegion(partial1Gm_, finalLocal, Y1_LOCAL_OFFSET);
            MergeOneRegion(partial2Gm_, finalLocal, Y2_LOCAL_OFFSET);
            resultQueue_.EnQue(finalLocal);
            finalLocal = resultQueue_.DeQue<float>();
            CopyOut(finalLocal);
            resultQueue_.FreeTensor(finalLocal);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t elementCount)
    {
        LocalTensor<float> x1Local = x1Queue_.AllocTensor<float>();
        LocalTensor<float> x2Local = x2Queue_.AllocTensor<float>();
        const uint32_t tailElements = static_cast<uint32_t>(elementCount) % VECTOR_ELEMENTS;
        if (tailElements != 0) {
            const uint32_t tailOffset = static_cast<uint32_t>(elementCount) - tailElements;
            Duplicate(x1Local[tailOffset], 0.0f, VECTOR_ELEMENTS);
            Duplicate(x2Local[tailOffset], 0.0f, VECTOR_ELEMENTS);
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
        }
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(x1Local, x1Gm_[gmOffset], copyParams, padParams);
        DataCopyPad(x2Local, x2Gm_[gmOffset], copyParams, padParams);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void ComputeTile(LocalTensor<float> x1Local, LocalTensor<float> x2Local,
                                       LocalTensor<float> resultLocal, uint32_t elementCount, bool isFirstTile)
    {
        __ubuf__ float* x1Address = (__ubuf__ float*)x1Local.GetPhyAddr();
        __ubuf__ float* x2Address = (__ubuf__ float*)x2Local.GetPhyAddr();
        __ubuf__ float* resultAddress = (__ubuf__ float*)resultLocal.GetPhyAddr();
        const uint16_t fullVectorLoops = static_cast<uint16_t>(elementCount / VECTOR_ELEMENTS);
        uint32_t tailElements = elementCount % VECTOR_ELEMENTS;

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> x1Reg;
            Reg::RegTensor<float> x2Reg;
            Reg::RegTensor<float> squareReg;
            Reg::RegTensor<float> accumulator1;
            Reg::RegTensor<float> accumulator2;
            Reg::RegTensor<float> reduced1;
            Reg::RegTensor<float> reduced2;
            Reg::RegTensor<float> tailReduced1;
            Reg::RegTensor<float> tailReduced2;
            Reg::RegTensor<float> previous;
            Reg::MaskReg tailMask;
            Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg oneMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::Duplicate(accumulator1, 0.0f, allMask);
            Reg::Duplicate(accumulator2, 0.0f, allMask);
            Reg::Duplicate(tailReduced1, 0.0f, allMask);
            Reg::Duplicate(tailReduced2, 0.0f, allMask);

            for (uint16_t i = 0; i < fullVectorLoops; ++i) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x1Reg, x1Address + i * VECTOR_ELEMENTS);
                Reg::Mul(squareReg, x1Reg, x1Reg, allMask);
                Reg::Add(accumulator1, accumulator1, squareReg, allMask);

                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x2Reg, x2Address + i * VECTOR_ELEMENTS);
                Reg::Mul(squareReg, x2Reg, x2Reg, allMask);
                Reg::Add(accumulator2, accumulator2, squareReg, allMask);
            }

            if (tailElements > 0) {
                tailMask = Reg::UpdateMask<float>(tailElements);
                const uint32_t tailOffset = static_cast<uint32_t>(fullVectorLoops) * VECTOR_ELEMENTS;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x1Reg, x1Address + tailOffset);
                Reg::Mul(squareReg, x1Reg, x1Reg, tailMask);
                Reg::Reduce<Reg::ReduceType::SUM, float>(tailReduced1, squareReg, tailMask);

                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x2Reg, x2Address + tailOffset);
                Reg::Mul(squareReg, x2Reg, x2Reg, tailMask);
                Reg::Reduce<Reg::ReduceType::SUM, float>(tailReduced2, squareReg, tailMask);
            }

            Reg::Reduce<Reg::ReduceType::SUM, float>(reduced1, accumulator1, allMask);
            Reg::Reduce<Reg::ReduceType::SUM, float>(reduced2, accumulator2, allMask);
            Reg::Add(reduced1, reduced1, tailReduced1, oneMask);
            Reg::Add(reduced2, reduced2, tailReduced2, oneMask);
            if (!isFirstTile) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previous, resultAddress + Y1_LOCAL_OFFSET);
                Reg::Add(reduced1, reduced1, previous, oneMask);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previous, resultAddress + Y2_LOCAL_OFFSET);
                Reg::Add(reduced2, reduced2, previous, oneMask);
            }
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(resultAddress + Y1_LOCAL_OFFSET, reduced1,
                                                                           oneMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(resultAddress + Y2_LOCAL_OFFSET, reduced2,
                                                                           oneMask);
        }
    }

    __aicore__ inline void WriteCorePartials(LocalTensor<float> resultLocal)
    {
        DataCopyExtParams copyParams{1, sizeof(float), 0, 0, 0};
        const int64_t workspaceOffset = blockIdx_ * FLOATS_PER_BLOCK;
        DataCopyPad(partial1Gm_[workspaceOffset], resultLocal[Y1_LOCAL_OFFSET], copyParams);
        DataCopyPad(partial2Gm_[workspaceOffset], resultLocal[Y2_LOCAL_OFFSET], copyParams);
    }

    __aicore__ inline void MergeOneRegion(const GlobalTensor<float>& partialGm, LocalTensor<float> finalLocal,
                                          uint32_t finalOffset)
    {
        LocalTensor<float> partialLocal = partialQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(usedCoreNum_), sizeof(float), BLOCK_BYTES - sizeof(float), 0,
                                     0};
        DataCopyPadExtParams<float> padParams{true, 0, FLOATS_PER_BLOCK - 1, 0.0f};
        DataCopyPad(partialLocal, partialGm, copyParams, padParams);
        partialQueue_.EnQue(partialLocal);
        partialLocal = partialQueue_.DeQue<float>();
        ReducePartials(partialLocal, finalLocal, finalOffset);
        partialQueue_.FreeTensor(partialLocal);
    }

    __aicore__ inline void ReducePartials(LocalTensor<float> partialLocal, LocalTensor<float> finalLocal,
                                          uint32_t finalOffset)
    {
        __ubuf__ float* partialAddress = (__ubuf__ float*)partialLocal.GetPhyAddr();
        __ubuf__ float* finalAddress = (__ubuf__ float*)finalLocal.GetPhyAddr();
        const uint16_t partialElements = static_cast<uint16_t>(usedCoreNum_ * FLOATS_PER_BLOCK);
        const uint16_t fullVectorLoops = partialElements / VECTOR_ELEMENTS;
        uint32_t tailElements = partialElements % VECTOR_ELEMENTS;

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> total;
            Reg::RegTensor<float> current;
            Reg::RegTensor<float> tailCurrent;
            Reg::RegTensor<float> reduced;
            Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg oneMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::Duplicate(total, 0.0f, allMask);
            for (uint16_t i = 0; i < fullVectorLoops; ++i) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(current, partialAddress + i * VECTOR_ELEMENTS);
                Reg::Add(total, total, current, allMask);
            }
            if (tailElements != 0) {
                Reg::MaskReg tailMask = Reg::UpdateMask<float>(tailElements);
                Reg::Duplicate(tailCurrent, 0.0f, allMask);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(current,
                                                                partialAddress + fullVectorLoops * VECTOR_ELEMENTS);
                Reg::Add(tailCurrent, tailCurrent, current, tailMask);
                Reg::Add(total, total, tailCurrent, allMask);
            }
            Reg::Reduce<Reg::ReduceType::SUM, float>(reduced, total, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(finalAddress + finalOffset, reduced,
                                                                           oneMask);
        }
    }

    __aicore__ inline void CopyOut(LocalTensor<float> finalLocal)
    {
        DataCopyExtParams copyParams{1, sizeof(float), 0, 0, 0};
        DataCopyPad(y1Gm_, finalLocal[Y1_LOCAL_OFFSET], copyParams);
        DataCopyPad(y2Gm_, finalLocal[Y2_LOCAL_OFFSET], copyParams);
    }

private:
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t BLOCK_BYTES = 32;
    static constexpr int64_t FLOATS_PER_BLOCK = BLOCK_BYTES / sizeof(float);
    static constexpr uint16_t VECTOR_ELEMENTS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    static constexpr int64_t RESULT_LOCAL_BYTES = 2 * BLOCK_BYTES;
    static constexpr uint32_t Y1_LOCAL_OFFSET = 0;
    static constexpr uint32_t Y2_LOCAL_OFFSET = FLOATS_PER_BLOCK;
    static_assert(VECTOR_ELEMENTS == 64, "SquareSumAll tiling assumes a 256-byte FP32 vector register");

    const SquareSumAllTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t coreStart_ = 0;
    int64_t coreElements_ = 0;
    int64_t tileElements_ = 0;

    GlobalTensor<float> x1Gm_;
    GlobalTensor<float> x2Gm_;
    GlobalTensor<float> y1Gm_;
    GlobalTensor<float> y2Gm_;
    GlobalTensor<float> partial1Gm_;
    GlobalTensor<float> partial2Gm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Queue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Queue_;
    TQue<QuePosition::VECOUT, 1> resultQueue_;
    TQue<QuePosition::VECIN, 1> partialQueue_;
};
} // namespace SquareSumAllOps

#endif // SQUARE_SUM_ALL_ARCH35_H_
