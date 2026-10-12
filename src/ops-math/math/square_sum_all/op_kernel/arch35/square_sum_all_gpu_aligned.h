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
 * \file square_sum_all_gpu_aligned.h
 * \brief Deterministic SquareSumAll reduction topology empirically aligned with the observed A100 reference.
 */

#ifndef SQUARE_SUM_ALL_GPU_ALIGNED_ARCH35_H_
#define SQUARE_SUM_ALL_GPU_ALIGNED_ARCH35_H_

#include "kernel_operator.h"
#include "square_sum_all_tiling_data.h"

namespace SquareSumAllGpuAligned {
using namespace AscendC;
constexpr uint32_t REDUCTION_LANE_COUNT = 64;

// Reproduce the empirically best-matching 64-lane high-half reduction order explicitly:
// (0+32, 1+33, ...), then +16, +8, +4, +2 and +1.
__simd_callee__ inline void ReduceHighHalf64(Reg::RegTensor<float>& dst, Reg::RegTensor<float>& src)
{
    Reg::RegTensor<uint32_t> baseIndex;
    Reg::RegTensor<uint32_t> gatherIndex;
    Reg::RegTensor<uint32_t> validIndexMask;
    Reg::RegTensor<float> gathered;
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg reduceMask;
    uint32_t activeElements = 32;
    Reg::Arange((Reg::RegTensor<int32_t>&)baseIndex, static_cast<int32_t>(0));
    Reg::Duplicate(validIndexMask, REDUCTION_LANE_COUNT - 1, indexMask);

    Reg::Adds(gatherIndex, baseIndex, static_cast<uint32_t>(32), indexMask);
    Reg::And(gatherIndex, gatherIndex, validIndexMask, indexMask);
    Reg::Gather(gathered, src, gatherIndex);
    reduceMask = Reg::UpdateMask<float>(activeElements);
    Reg::Add(src, src, gathered, reduceMask);

    Reg::Adds(gatherIndex, baseIndex, static_cast<uint32_t>(16), indexMask);
    Reg::And(gatherIndex, gatherIndex, validIndexMask, indexMask);
    Reg::Gather(gathered, src, gatherIndex);
    activeElements = 16;
    reduceMask = Reg::UpdateMask<float>(activeElements);
    Reg::Add(src, src, gathered, reduceMask);

    Reg::Adds(gatherIndex, baseIndex, static_cast<uint32_t>(8), indexMask);
    Reg::And(gatherIndex, gatherIndex, validIndexMask, indexMask);
    Reg::Gather(gathered, src, gatherIndex);
    activeElements = 8;
    reduceMask = Reg::UpdateMask<float>(activeElements);
    Reg::Add(src, src, gathered, reduceMask);

    Reg::Adds(gatherIndex, baseIndex, static_cast<uint32_t>(4), indexMask);
    Reg::And(gatherIndex, gatherIndex, validIndexMask, indexMask);
    Reg::Gather(gathered, src, gatherIndex);
    activeElements = 4;
    reduceMask = Reg::UpdateMask<float>(activeElements);
    Reg::Add(src, src, gathered, reduceMask);

    Reg::Adds(gatherIndex, baseIndex, static_cast<uint32_t>(2), indexMask);
    Reg::And(gatherIndex, gatherIndex, validIndexMask, indexMask);
    Reg::Gather(gathered, src, gatherIndex);
    activeElements = 2;
    reduceMask = Reg::UpdateMask<float>(activeElements);
    Reg::Add(src, src, gathered, reduceMask);

    Reg::Adds(gatherIndex, baseIndex, static_cast<uint32_t>(1), indexMask);
    Reg::And(gatherIndex, gatherIndex, validIndexMask, indexMask);
    Reg::Gather(gathered, src, gatherIndex);
    activeElements = 1;
    reduceMask = Reg::UpdateMask<float>(activeElements);
    Reg::Add(dst, src, gathered, reduceMask);
}

class SquareSumAllGpuAlignedKernel {
public:
    __aicore__ inline SquareSumAllGpuAlignedKernel() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, GM_ADDR userWorkspace,
                                const SquareSumAllTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        totalElements_ = tilingData->totalElements;
        usedCoreNum_ = tilingData->usedCoreNum;
        partialCount_ = CeilDiv(totalElements_, CUDA_BLOCK_ELEMENTS);
        partialCount_ = partialCount_ < CUDA_MAX_BLOCKS ? partialCount_ : CUDA_MAX_BLOCKS;
        gridStride_ = partialCount_ * CUDA_BLOCK_ELEMENTS;
        chunkCount_ = CeilDiv(totalElements_, gridStride_);

        const int64_t basePartialsPerCore = partialCount_ / usedCoreNum_;
        const int64_t extraPartialCores = partialCount_ % usedCoreNum_;
        corePartialCount_ = basePartialsPerCore + (blockIdx_ < extraPartialCores ? 1 : 0);
        corePartialBegin_ = blockIdx_ * basePartialsPerCore +
                            (blockIdx_ < extraPartialCores ? blockIdx_ : extraPartialCores);

        x1Gm_.SetGlobalBuffer((__gm__ float*)x1, totalElements_);
        x2Gm_.SetGlobalBuffer((__gm__ float*)x2, totalElements_);
        y1Gm_.SetGlobalBuffer((__gm__ float*)y1, 1);
        y2Gm_.SetGlobalBuffer((__gm__ float*)y2, 1);

        const int64_t workspaceElementsPerRegion = partialCount_ * FLOATS_PER_BLOCK;
        auto* workspace = (__gm__ float*)userWorkspace;
        partial1Gm_.SetGlobalBuffer(workspace, workspaceElementsPerRegion);
        partial2Gm_.SetGlobalBuffer(workspace + workspaceElementsPerRegion, workspaceElementsPerRegion);

        const int64_t packedPartials = CeilDiv(partialCount_, CUDA_BLOCK_ELEMENTS) * CUDA_BLOCK_ELEMENTS;
        const int64_t partialLocalBytes = packedPartials * FLOATS_PER_BLOCK * sizeof(float);
        const int64_t fixedLocalBytes = partialLocalBytes + RESULT_LOCAL_BYTES;
        const int64_t batchCapacityByUb = (BATCH_UB_BUDGET_BYTES - fixedLocalBytes) / BATCH_BYTES_PER_PARTIAL;
        batchPartialCapacity_ = batchCapacityByUb > 0 ? batchCapacityByUb : 1;
        batchPartialCapacity_ = batchPartialCapacity_ < corePartialCount_ ? batchPartialCapacity_ : corePartialCount_;
        batchPartialCapacity_ = batchPartialCapacity_ > 0 ? batchPartialCapacity_ : 1;

        chunkInputElements_ = batchPartialCapacity_ * CUDA_BLOCK_ELEMENTS;
        accumulatorRegionElements_ = batchPartialCapacity_ * VECTOR_ELEMENTS;
        accumulatorBufferElements_ = ACCUMULATOR_REGION_COUNT * accumulatorRegionElements_;
        resultRegionElements_ = batchPartialCapacity_ * FLOATS_PER_BLOCK;
        batchResultBufferElements_ = 2 * resultRegionElements_;

        pipe_->InitBuffer(x1Queue_, INPUT_QUEUE_DEPTH, chunkInputElements_ * sizeof(float));
        pipe_->InitBuffer(x2Queue_, INPUT_QUEUE_DEPTH, chunkInputElements_ * sizeof(float));
        pipe_->InitBuffer(accumulatorBuffer_, accumulatorBufferElements_ * sizeof(float));
        pipe_->InitBuffer(batchResultQueue_, 1, batchResultBufferElements_ * sizeof(float));
        pipe_->InitBuffer(finalQueue_, 1, RESULT_LOCAL_BYTES);
        pipe_->InitBuffer(partialQueue_, 1, partialLocalBytes);
    }

    __aicore__ inline void Process()
    {
        const int64_t corePartialEnd = corePartialBegin_ + corePartialCount_;
        for (int64_t batchBegin = corePartialBegin_; batchBegin < corePartialEnd; batchBegin += batchPartialCapacity_) {
            const int64_t remaining = corePartialEnd - batchBegin;
            const int64_t batchCount = remaining < batchPartialCapacity_ ? remaining : batchPartialCapacity_;
            LocalTensor<float> accumulatorLocal = accumulatorBuffer_.Get<float>();

            for (int64_t chunk = 0; chunk < chunkCount_; ++chunk) {
                CopyBatchChunk(batchBegin, batchCount, chunk);
                LocalTensor<float> x1Local = x1Queue_.DeQue<float>();
                LocalTensor<float> x2Local = x2Queue_.DeQue<float>();
                AccumulateBatchChunk(x1Local, x2Local, accumulatorLocal, batchCount, chunk == 0);
                x1Queue_.FreeTensor(x1Local);
                x2Queue_.FreeTensor(x2Local);
            }

            LocalTensor<float> batchResultLocal = batchResultQueue_.AllocTensor<float>();
            FinalizeBatch(accumulatorLocal, batchResultLocal, batchCount);
            batchResultQueue_.EnQue(batchResultLocal);
            batchResultLocal = batchResultQueue_.DeQue<float>();
            WritePartialBatch(batchBegin, batchCount, batchResultLocal);
            batchResultQueue_.FreeTensor(batchResultLocal);
        }

        SyncAll();
        if (blockIdx_ == 0) {
            LocalTensor<float> finalLocal = finalQueue_.AllocTensor<float>();
            MergeRegion(partial1Gm_, finalLocal, Y1_LOCAL_OFFSET);
            MergeRegion(partial2Gm_, finalLocal, Y2_LOCAL_OFFSET);
            finalQueue_.EnQue(finalLocal);
            finalLocal = finalQueue_.DeQue<float>();
            CopyOut(finalLocal);
            finalQueue_.FreeTensor(finalLocal);
        }
    }

private:
    __aicore__ inline static int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return value / divisor + (value % divisor != 0);
    }

    __aicore__ inline void CopyBatchChunk(int64_t batchBegin, int64_t batchCount, int64_t chunk)
    {
        LocalTensor<float> x1Local = x1Queue_.AllocTensor<float>();
        LocalTensor<float> x2Local = x2Queue_.AllocTensor<float>();
        const int64_t gmOffset = batchBegin * CUDA_BLOCK_ELEMENTS + chunk * gridStride_;
        const int64_t requestedElements = batchCount * CUDA_BLOCK_ELEMENTS;
        const int64_t remainingElements = gmOffset < totalElements_ ? totalElements_ - gmOffset : 0;
        const int64_t copyElements = remainingElements < requestedElements ? remainingElements : requestedElements;

        if (copyElements < requestedElements) {
            Duplicate(x1Local, 0.0f, requestedElements);
            Duplicate(x2Local, 0.0f, requestedElements);
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
        }
        if (copyElements > 0) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyElements * sizeof(float)), 0, 0, 0};
            const uint8_t rightPadding = static_cast<uint8_t>((FLOATS_PER_BLOCK - copyElements % FLOATS_PER_BLOCK) %
                                                              FLOATS_PER_BLOCK);
            DataCopyPadExtParams<float> padParams{rightPadding != 0, 0, rightPadding, 0.0f};
            DataCopyPad(x1Local, x1Gm_[gmOffset], copyParams, padParams);
            DataCopyPad(x2Local, x2Gm_[gmOffset], copyParams, padParams);
        }
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void AccumulateBatchChunk(LocalTensor<float> x1Local, LocalTensor<float> x2Local,
                                                LocalTensor<float> accumulatorLocal, int64_t batchCount,
                                                bool firstChunk)
    {
        __ubuf__ float* x1Address = (__ubuf__ float*)x1Local.GetPhyAddr();
        __ubuf__ float* x2Address = (__ubuf__ float*)x2Local.GetPhyAddr();
        __ubuf__ float* accumulatorAddress = (__ubuf__ float*)accumulatorLocal.GetPhyAddr();
        const uint16_t vectorBatchCount = static_cast<uint16_t>(batchCount);
        const int64_t accumulatorRegionElements = accumulatorRegionElements_;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> xReg;
            Reg::RegTensor<float> x1Low;
            Reg::RegTensor<float> x1High;
            Reg::RegTensor<float> x2Low;
            Reg::RegTensor<float> x2High;
            Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            for (uint16_t batchSlot = 0; batchSlot < vectorBatchCount; ++batchSlot) {
                const int64_t inputOffset = static_cast<int64_t>(batchSlot) * CUDA_BLOCK_ELEMENTS;
                const int64_t accumulatorOffset = static_cast<int64_t>(batchSlot) * VECTOR_ELEMENTS;
                if (firstChunk) {
                    Reg::Duplicate(x1Low, 0.0f, allMask);
                    Reg::Duplicate(x1High, 0.0f, allMask);
                    Reg::Duplicate(x2Low, 0.0f, allMask);
                    Reg::Duplicate(x2High, 0.0f, allMask);
                } else {
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x1Low, accumulatorAddress + accumulatorOffset);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                        x1High, accumulatorAddress + accumulatorRegionElements + accumulatorOffset);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                        x2Low, accumulatorAddress + 2 * accumulatorRegionElements + accumulatorOffset);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                        x2High, accumulatorAddress + 3 * accumulatorRegionElements + accumulatorOffset);
                }
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(xReg, x1Address + inputOffset);
                Reg::MulAddDst(x1Low, xReg, xReg, allMask);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(xReg, x1Address + inputOffset + VECTOR_ELEMENTS);
                Reg::MulAddDst(x1High, xReg, xReg, allMask);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(xReg, x2Address + inputOffset);
                Reg::MulAddDst(x2Low, xReg, xReg, allMask);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(xReg, x2Address + inputOffset + VECTOR_ELEMENTS);
                Reg::MulAddDst(x2High, xReg, xReg, allMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(accumulatorAddress + accumulatorOffset, x1Low,
                                                                  allMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
                    accumulatorAddress + accumulatorRegionElements + accumulatorOffset, x1High, allMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
                    accumulatorAddress + 2 * accumulatorRegionElements + accumulatorOffset, x2Low, allMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
                    accumulatorAddress + 3 * accumulatorRegionElements + accumulatorOffset, x2High, allMask);
            }
        }
    }

    __aicore__ inline void FinalizeBatch(LocalTensor<float> accumulatorLocal, LocalTensor<float> resultLocal,
                                         int64_t batchCount)
    {
        __ubuf__ float* accumulatorAddress = (__ubuf__ float*)accumulatorLocal.GetPhyAddr();
        __ubuf__ float* resultAddress = (__ubuf__ float*)resultLocal.GetPhyAddr();
        const uint16_t vectorBatchCount = static_cast<uint16_t>(batchCount);
        const int64_t accumulatorRegionElements = accumulatorRegionElements_;
        const int64_t resultRegionElements = resultRegionElements_;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> x1Low;
            Reg::RegTensor<float> x1High;
            Reg::RegTensor<float> x2Low;
            Reg::RegTensor<float> x2High;
            Reg::RegTensor<float> reduced1;
            Reg::RegTensor<float> reduced2;
            Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg oneMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            for (uint16_t batchSlot = 0; batchSlot < vectorBatchCount; ++batchSlot) {
                const int64_t accumulatorOffset = static_cast<int64_t>(batchSlot) * VECTOR_ELEMENTS;
                const int64_t resultOffset = static_cast<int64_t>(batchSlot) * FLOATS_PER_BLOCK;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x1Low, accumulatorAddress + accumulatorOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                    x1High, accumulatorAddress + accumulatorRegionElements + accumulatorOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                    x2Low, accumulatorAddress + 2 * accumulatorRegionElements + accumulatorOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                    x2High, accumulatorAddress + 3 * accumulatorRegionElements + accumulatorOffset);
                Reg::Add(x1Low, x1Low, x1High, allMask);
                Reg::Add(x2Low, x2Low, x2High, allMask);
                ReduceHighHalf64(reduced1, x1Low);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(resultAddress + resultOffset, reduced1,
                                                                               oneMask);
                ReduceHighHalf64(reduced2, x2Low);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    resultAddress + resultRegionElements + resultOffset, reduced2, oneMask);
            }
        }
    }

    __aicore__ inline void WritePartialBatch(int64_t batchBegin, int64_t batchCount,
                                             LocalTensor<float> batchResultLocal)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(batchCount), sizeof(float), 0, BLOCK_BYTES - sizeof(float),
                                     0};
        const int64_t workspaceOffset = batchBegin * FLOATS_PER_BLOCK;
        DataCopyPad(partial1Gm_[workspaceOffset], batchResultLocal, copyParams);
        DataCopyPad(partial2Gm_[workspaceOffset], batchResultLocal[resultRegionElements_], copyParams);
    }

    __aicore__ inline void MergeRegion(const GlobalTensor<float>& partialGm, LocalTensor<float> finalLocal,
                                       uint32_t finalOffset)
    {
        LocalTensor<float> partialLocal = partialQueue_.AllocTensor<float>();
        const int64_t packedPartials = CeilDiv(partialCount_, CUDA_BLOCK_ELEMENTS) * CUDA_BLOCK_ELEMENTS;
        Duplicate(partialLocal, 0.0f, packedPartials * FLOATS_PER_BLOCK);
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        DataCopyExtParams copyParams{static_cast<uint16_t>(partialCount_), sizeof(float), BLOCK_BYTES - sizeof(float),
                                     0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, FLOATS_PER_BLOCK - 1, 0.0f};
        DataCopyPad(partialLocal, partialGm, copyParams, padParams);
        partialQueue_.EnQue(partialLocal);
        partialLocal = partialQueue_.DeQue<float>();
        ReducePackedPartials(partialLocal, finalLocal, finalOffset);
        partialQueue_.FreeTensor(partialLocal);
    }

    __aicore__ inline void ReducePackedPartials(LocalTensor<float> partialLocal, LocalTensor<float> finalLocal,
                                                uint32_t finalOffset)
    {
        __ubuf__ float* partialAddress = (__ubuf__ float*)partialLocal.GetPhyAddr();
        __ubuf__ float* finalAddress = (__ubuf__ float*)finalLocal.GetPhyAddr();
        const uint16_t groupCount = static_cast<uint16_t>(CeilDiv(partialCount_, CUDA_BLOCK_ELEMENTS));
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> current;
            Reg::RegTensor<float> low;
            Reg::RegTensor<float> high;
            Reg::RegTensor<float> reduced;
            Reg::RegTensor<int32_t> gatherIndex;
            Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg indexMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
            Reg::MaskReg oneMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
            Reg::Duplicate(low, 0.0f, allMask);
            Reg::Duplicate(high, 0.0f, allMask);
            Reg::Arange(gatherIndex, static_cast<int32_t>(0));
            Reg::Muls(gatherIndex, gatherIndex, static_cast<int32_t>(FLOATS_PER_BLOCK), indexMask);
            for (uint16_t group = 0; group < groupCount; ++group) {
                const int64_t offset = static_cast<int64_t>(group) * CUDA_BLOCK_ELEMENTS * FLOATS_PER_BLOCK;
                Reg::Gather(current, partialAddress + offset, (Reg::RegTensor<uint32_t>&)gatherIndex, allMask);
                Reg::Add(low, low, current, allMask);
                Reg::Gather(current, partialAddress + offset + VECTOR_ELEMENTS * FLOATS_PER_BLOCK,
                            (Reg::RegTensor<uint32_t>&)gatherIndex, allMask);
                Reg::Add(high, high, current, allMask);
            }
            Reg::Add(low, low, high, allMask);
            ReduceHighHalf64(reduced, low);
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
    static constexpr int64_t CUDA_BLOCK_ELEMENTS = 128;
    static constexpr int64_t CUDA_MAX_BLOCKS = 1728;
    static constexpr int64_t BLOCK_BYTES = 32;
    static constexpr int64_t FLOATS_PER_BLOCK = BLOCK_BYTES / sizeof(float);
    static constexpr int64_t VECTOR_ELEMENTS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    static constexpr int64_t INPUT_QUEUE_DEPTH = 2;
    static constexpr int64_t ACCUMULATOR_REGION_COUNT = 4;
    static constexpr int64_t RESULT_LOCAL_BYTES = 2 * VECTOR_ELEMENTS * sizeof(float);
    static constexpr int64_t BATCH_RESULT_BYTES_PER_PARTIAL = 2 * FLOATS_PER_BLOCK *
                                                              static_cast<int64_t>(sizeof(float));
    static constexpr int64_t BATCH_BYTES_PER_PARTIAL = 2 * INPUT_QUEUE_DEPTH * CUDA_BLOCK_ELEMENTS *
                                                           static_cast<int64_t>(sizeof(float)) +
                                                       ACCUMULATOR_REGION_COUNT * VECTOR_ELEMENTS *
                                                           static_cast<int64_t>(sizeof(float)) +
                                                       BATCH_RESULT_BYTES_PER_PARTIAL;
    static constexpr uint32_t Y1_LOCAL_OFFSET = 0;
    static constexpr uint32_t Y2_LOCAL_OFFSET = FLOATS_PER_BLOCK;
    static constexpr int64_t BATCH_UB_BUDGET_BYTES = 224 * 1024;
    static_assert(VECTOR_ELEMENTS == 64, "GPU-aligned reduction assumes a 256-byte FP32 vector register");
    static_assert(BATCH_UB_BUDGET_BYTES / BATCH_BYTES_PER_PARTIAL <= 4095,
                  "DataCopyPad blockCount must stay within the hardware limit");

    TPipe* pipe_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t totalElements_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t partialCount_ = 0;
    int64_t gridStride_ = 0;
    int64_t chunkCount_ = 0;
    int64_t corePartialBegin_ = 0;
    int64_t corePartialCount_ = 0;
    int64_t batchPartialCapacity_ = 1;
    int64_t chunkInputElements_ = 0;
    int64_t accumulatorRegionElements_ = 0;
    int64_t accumulatorBufferElements_ = 0;
    int64_t resultRegionElements_ = 0;
    int64_t batchResultBufferElements_ = 0;
    GlobalTensor<float> x1Gm_;
    GlobalTensor<float> x2Gm_;
    GlobalTensor<float> y1Gm_;
    GlobalTensor<float> y2Gm_;
    GlobalTensor<float> partial1Gm_;
    GlobalTensor<float> partial2Gm_;
    TQue<QuePosition::VECIN, INPUT_QUEUE_DEPTH> x1Queue_;
    TQue<QuePosition::VECIN, INPUT_QUEUE_DEPTH> x2Queue_;
    TBuf<TPosition::VECCALC> accumulatorBuffer_;
    TQue<QuePosition::VECOUT, 1> batchResultQueue_;
    TQue<QuePosition::VECOUT, 1> finalQueue_;
    TQue<QuePosition::VECIN, 1> partialQueue_;
};
} // namespace SquareSumAllGpuAligned

#endif // SQUARE_SUM_ALL_GPU_ALIGNED_ARCH35_H_
