/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "bernoulli_tiling_data.h"

using namespace AscendC;

template <typename T>
class BernoulliKernel {
    using StorageT = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
    static constexpr uint32_t kOutputBufferNum = 2;
    static constexpr uint32_t kMaskBufferNum = 1;

public:
    __aicore__ inline void Init(GM_ADDR mask, GM_ADDR output, const optiling::BernoulliTilingData& tiling)
    {
        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StorageT*>(output));
        total_ = tiling.totalElements;
        blockElements_ = tiling.blockElements;
        stageElements_ = tiling.stageElements;
        tileElements_ = tiling.tileElements;
        mode_ = tiling.mode;
        if (mode_ != optiling::BERNOULLI_MODE_RANDOM_ALIASED) {
            blockStart_ = static_cast<uint64_t>(GetBlockIdx()) * blockElements_;
            if (blockStart_ >= total_) {
                blockElements_ = 0;
                return;
            }
            blockElements_ = (blockElements_ < (total_ - blockStart_)) ? blockElements_ : (total_ - blockStart_);
        }
        pipe_.InitBuffer(outputQueue_, kOutputBufferNum, ((tileElements_ * sizeof(StorageT) + 31U) / 32U) * 32U);
        if (IsRandomMode()) {
            pipe_.InitBuffer(maskQueue_, kMaskBufferNum, ((stageElements_ / 8U + 31U) / 32U) * 32U);
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
                pipe_.InitBuffer(byteOutputTemp_, tileElements_ * sizeof(half));
            } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
                pipe_.InitBuffer(wideOutputTemp_, tileElements_ * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (mode_ == optiling::BERNOULLI_MODE_RANDOM_ALIASED) {
            ProcessRandomAliased();
            return;
        }
        for (uint64_t offset = 0; offset < blockElements_; offset += tileElements_) {
            const uint32_t count = static_cast<uint32_t>(
                tileElements_ < blockElements_ - offset ? tileElements_ : blockElements_ - offset);
            ComputeConstant(count);
            CopyOut(blockStart_ + offset, count);
        }
    }

private:
    __aicore__ inline bool IsRandomMode() const { return mode_ == optiling::BERNOULLI_MODE_RANDOM_ALIASED; }

    __aicore__ inline void ProcessRandomAliased()
    {
        const uint64_t waveElements = stageElements_ * static_cast<uint64_t>(GetBlockNum());
        const uint64_t stageCount = (total_ + waveElements - 1U) / waveElements;
        for (uint64_t stage = stageCount; stage > 0; --stage) {
            const uint64_t stageStart = (stage - 1U) * waveElements;
            const uint64_t outputStart = stageStart + static_cast<uint64_t>(GetBlockIdx()) * stageElements_;
            uint32_t count = 0;
            if (outputStart < total_) {
                count = static_cast<uint32_t>(stageElements_ < total_ - outputStart ? stageElements_ :
                                                                                      total_ - outputStart);
                CopyMaskIn(outputStart, count);
            }

            LocalTensor<uint8_t> maskLocal;
            if (count > 0) {
                // DeQue waits for this core's GM-to-UB copy. Multi-core runs
                // synchronize before overwrite; single-core runs keep the
                // dependency local and skip the barrier.
                maskLocal = maskQueue_.DeQue<uint8_t>();
            }
            if (GetBlockNum() > 1) {
                SyncAll<true>();
            }
            if (count > 0) {
                ComputeRandomStage(maskLocal, outputStart, count);
                maskQueue_.FreeTensor(maskLocal);
            }
        }
    }

    __aicore__ inline void ComputeRandomStage(const LocalTensor<uint8_t>& maskLocal, uint64_t outputStart,
                                              uint32_t count)
    {
        uint32_t copiedElements = 0;
        uint32_t pendingElements = 0;
        for (uint32_t offset = 0; offset < count; offset += static_cast<uint32_t>(tileElements_)) {
            const uint32_t tileCount = static_cast<uint32_t>(tileElements_ < count - offset ? tileElements_ :
                                                                                              count - offset);
            auto tileMask = maskLocal[offset / 8U];
            ComputeMask(tileMask, tileCount);

            // outputQueue_ owns two buffers. Keep one completed tile queued
            // while VECTOR prepares the next tile so its MTE3 write can run
            // concurrently with the next Duplicate/Select/Cast sequence.
            if (pendingElements > 0) {
                CopyOut(outputStart + copiedElements, pendingElements);
                copiedElements += pendingElements;
            }
            pendingElements = tileCount;
        }
        if (pendingElements > 0) {
            CopyOut(outputStart + copiedElements, pendingElements);
        }
    }

    __aicore__ inline void CopyMaskIn(uint64_t outputStart, uint32_t count)
    {
        const uint32_t maskBytes = (count + 7U) / 8U;
        const uint32_t alignedMaskBytes = ((maskBytes + 31U) / 32U) * 32U;
        auto maskLocal = maskQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams maskParams{1, maskBytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> maskPad{true, 0, static_cast<uint8_t>(alignedMaskBytes - maskBytes),
                                              static_cast<uint8_t>(0)};
        DataCopyPad(maskLocal, maskGm_[outputStart / 8U], maskParams, maskPad);
        maskQueue_.EnQue(maskLocal);
    }

    __aicore__ inline void ComputeMask(const LocalTensor<uint8_t>& maskLocal, uint32_t count)
    {
        const uint32_t processCount = ((count + 255U) / 256U) * 256U;
        auto outputLocal = outputQueue_.AllocTensor<StorageT>();
        if constexpr (std::is_same_v<T, float>) {
            Duplicate(outputLocal, 1.0f, processCount);
            Select(outputLocal, maskLocal, outputLocal, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, processCount);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            Duplicate(outputLocal, static_cast<int32_t>(1), processCount);
            auto outputFloat = outputLocal.template ReinterpretCast<float>();
            Select(outputFloat, maskLocal, outputFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, processCount);
        } else if constexpr (std::is_same_v<T, half>) {
            Duplicate(outputLocal, static_cast<half>(1.0), processCount);
            Select(outputLocal, maskLocal, outputLocal, static_cast<half>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   processCount);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            Duplicate(outputLocal, static_cast<int16_t>(1), processCount);
            auto outputHalf = outputLocal.template ReinterpretCast<half>();
            Select(outputHalf, maskLocal, outputHalf, static_cast<half>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   processCount);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            auto outputBits = outputLocal.template ReinterpretCast<uint16_t>();
            Duplicate(outputBits, static_cast<uint16_t>(0x3f80U), processCount);
            auto outputHalf = outputLocal.template ReinterpretCast<half>();
            Select(outputHalf, maskLocal, outputHalf, static_cast<half>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   processCount);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            auto outputHalf = byteOutputTemp_.Get<half>();
            Duplicate(outputHalf, static_cast<half>(1.0), processCount);
            Select(outputHalf, maskLocal, outputHalf, static_cast<half>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   processCount);
            Cast(outputLocal, outputHalf, RoundMode::CAST_NONE, processCount);
        } else if constexpr (std::is_same_v<T, bool>) {
            auto outputHalf = byteOutputTemp_.Get<half>();
            Duplicate(outputHalf, static_cast<half>(1.0), processCount);
            Select(outputHalf, maskLocal, outputHalf, static_cast<half>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   processCount);
            Cast(outputLocal, outputHalf, RoundMode::CAST_NONE, processCount);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            auto outputFloat = wideOutputTemp_.Get<float>();
            Duplicate(outputFloat, 1.0f, processCount);
            Select(outputFloat, maskLocal, outputFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, processCount);
            PipeBarrier<PIPE_V>();
            Cast(outputLocal, outputFloat, RoundMode::CAST_RINT, processCount);
        } else if constexpr (std::is_same_v<T, double>) {
            // 0x3ff0000000000000 is the bit pattern of double 1.0 and is
            // exactly representable as float. Convert the selected integer
            // value to int64 directly into the double output buffer.
            constexpr float kDoubleOneBits = 4607182418800017408.0f;
            auto outputFloat = wideOutputTemp_.Get<float>();
            Duplicate(outputFloat, kDoubleOneBits, processCount);
            Select(outputFloat, maskLocal, outputFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, processCount);
            PipeBarrier<PIPE_V>();
            auto outputBits = outputLocal.template ReinterpretCast<int64_t>();
            Cast(outputBits, outputFloat, RoundMode::CAST_RINT, processCount);
        }
        outputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void ComputeConstant(uint32_t count)
    {
        const uint32_t processCount = ((count + 255U) / 256U) * 256U;
        const bool isOne = mode_ == optiling::BERNOULLI_MODE_ONE;
        auto outputLocal = outputQueue_.AllocTensor<StorageT>();
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, bool>) {
            auto outputWords = outputLocal.template ReinterpretCast<uint16_t>();
            Duplicate(outputWords, static_cast<uint16_t>(isOne ? 0x0101U : 0U), processCount / 2U);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            auto outputBits = outputLocal.template ReinterpretCast<uint16_t>();
            Duplicate(outputBits, static_cast<uint16_t>(isOne ? 0x3f80U : 0U), processCount);
        } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
            auto outputWords = outputLocal.template ReinterpretCast<uint32_t>();
            const uint32_t wordCount = 2U * processCount;
            Duplicate(outputWords, static_cast<uint32_t>(0), wordCount);
            if (isOne) {
                PipeBarrier<PIPE_V>();
                uint64_t wordMask[2] = {std::is_same_v<T, int64_t> ? 0x5555555555555555ULL : 0xAAAAAAAAAAAAAAAAULL,
                                        0ULL};
                const uint8_t repeatTimes = static_cast<uint8_t>((wordCount + 63U) / 64U);
                const uint32_t oneWord = std::is_same_v<T, int64_t> ? 1U : 0x3ff00000U;
                Duplicate(outputWords, oneWord, wordMask, repeatTimes, 1, 8);
            }
        } else {
            Duplicate(outputLocal, static_cast<T>(isOne ? 1 : 0), processCount);
        }
        outputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset, uint32_t count)
    {
        auto outputLocal = outputQueue_.DeQue<StorageT>();
        DataCopyExtParams outputParams{1, static_cast<uint32_t>(count * sizeof(StorageT)), 0, 0, 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, outputParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    GlobalTensor<uint8_t> maskGm_;
    GlobalTensor<StorageT> outputGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, kMaskBufferNum> maskQueue_;
    TQue<QuePosition::VECOUT, kOutputBufferNum> outputQueue_;
    TBuf<QuePosition::VECCALC> byteOutputTemp_;
    TBuf<QuePosition::VECCALC> wideOutputTemp_;
    uint64_t total_ = 0;
    uint64_t blockStart_ = 0;
    uint64_t blockElements_ = 0;
    uint64_t stageElements_ = 0;
    uint64_t tileElements_ = 0;
    uint32_t mode_ = optiling::BERNOULLI_MODE_RANDOM_ALIASED;
};

extern "C" __global__ __aicore__ void bernoulli(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)x;
    (void)workspace;
    // Random alias mode uses SyncAll between descending output waves.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(optiling::BernoulliTilingData);
    GET_TILING_DATA_WITH_STRUCT(optiling::BernoulliTilingData, tilingData, tiling);
    BernoulliKernel<DTYPE_X> kernel;
    kernel.Init(mask, y, tilingData);
    kernel.Process();
}
