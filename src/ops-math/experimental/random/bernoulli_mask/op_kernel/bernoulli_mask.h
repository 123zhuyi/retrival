/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BERNOULLI_MASK_H
#define BERNOULLI_MASK_H

#include <type_traits>
#include "kernel_operator.h"
#include "bernoulli_mask_tiling_data.h"

namespace BernoulliMask {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint64_t BITS_PER_BYTE = 8;
constexpr uint64_t GM_ALIGN_BYTES = 32;
constexpr uint64_t MASK_ALIGN_ELEMENTS = 256;
// A double 1.0 is stored as two little-endian fp32 words [0, 0x3ff00000].
// 0x3ff00000 is exactly the fp32 encoding of 1.875f, which lets the vector
// unit build the high word without scalar bit manipulation.
constexpr float DOUBLE_ONE_HIGH_WORD = 1.875f;
constexpr uint64_t DOUBLE_WORK_PARTS = 4;

__aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T, bool IS_DOUBLE = false>
class KernelBernoulliMask {
public:
    __aicore__ inline KernelBernoulliMask() = default;

    __aicore__ inline void Init(GM_ADDR mask, GM_ADDR out, const optiling::BernoulliMaskTilingData* tilingData,
                                TPipe* pipe)
    {
        pipe_ = pipe;
        totalElements_ = tilingData->totalElements;
        elementsPerCore_ = tilingData->elementsPerCore;
        tileElements_ = tilingData->tileElements;
        maskAliasesOut_ = tilingData->maskAliasesOut != 0;

        const uint64_t blockIdx = GetBlockIdx();
        coreStart_ = blockIdx * elementsPerCore_;
        coreElements_ = coreStart_ < totalElements_ ? Min(elementsPerCore_, totalElements_ - coreStart_) : 0;

        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask),
                                (totalElements_ + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), totalElements_);

        const uint64_t maskBufferBytes = AlignUp((tileElements_ + BITS_PER_BYTE - 1) / BITS_PER_BYTE, GM_ALIGN_BYTES);
        const uint64_t outputBufferBytes = AlignUp(tileElements_ * sizeof(T), GM_ALIGN_BYTES);
        const uint64_t workBufferBytes = AlignUp(tileElements_ * sizeof(float), GM_ALIGN_BYTES);
        pipe_->InitBuffer(maskQueue_, BUFFER_NUM, maskBufferBytes);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, outputBufferBytes);
        pipe_->InitBuffer(workBuffer_, workBufferBytes);
        if constexpr (IS_DOUBLE) {
            doubleChunkElements_ = tileElements_ / DOUBLE_WORK_PARTS;
            BuildDoubleGatherOffsets();
        }
    }

    __aicore__ inline void Process()
    {
        if (maskAliasesOut_) {
            ProcessAliased();
        } else {
            ProcessRange(coreStart_, coreElements_);
        }
    }

private:
    __aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return (value + divisor - 1) / divisor; }

    __aicore__ inline void ProcessRange(uint64_t rangeStart, uint64_t rangeElements)
    {
        if (rangeElements == 0) {
            return;
        }
        const uint64_t loopCount = (rangeElements + tileElements_ - 1) / tileElements_;
        for (uint64_t loop = 0; loop < loopCount; ++loop) {
            const uint64_t offset = rangeStart + loop * tileElements_;
            const uint32_t count = static_cast<uint32_t>(Min(tileElements_, rangeElements - loop * tileElements_));
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

    __aicore__ inline void ProcessAliased()
    {
        const uint64_t blockIdx = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        uint64_t waveEnd = totalElements_;

        // The DSA packed mask initially occupies the beginning of out. For an
        // unconsumed prefix [0, waveEnd), output writes beginning at waveStart
        // are disjoint from every remaining mask byte when:
        //   sizeof(T) * waveStart >= ceil(waveEnd / 8).
        // Process that safe suffix on all cores, synchronize, then recurse on
        // the smaller prefix. Alignment keeps every mask read byte-aligned.
        while (waveEnd > MASK_ALIGN_ELEMENTS) {
            const uint64_t firstSafe = CeilDiv(CeilDiv(waveEnd, BITS_PER_BYTE), sizeof(T));
            const uint64_t waveStart = AlignUp(firstSafe, MASK_ALIGN_ELEMENTS);
            const uint64_t waveElements = waveEnd - waveStart;
            const uint64_t elementsPerCore = AlignUp(CeilDiv(waveElements, blockNum), MASK_ALIGN_ELEMENTS);
            const uint64_t coreStart = waveStart + blockIdx * elementsPerCore;
            const uint64_t coreElements = coreStart < waveEnd ? Min(elementsPerCore, waveEnd - coreStart) : 0;
            ProcessRange(coreStart, coreElements);
            SyncAll();
            waveEnd = waveStart;
        }

        // All mask bytes for higher output indices have been consumed. Core 0
        // copies the final small prefix into UB before its writes can overwrite
        // the aliased mask. Kernel completion provides the final global join.
        if (blockIdx == 0) {
            ProcessRange(0, waveEnd);
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();
        const uint32_t maskBytes = (count + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
        const uint8_t rightPadding = static_cast<uint8_t>(AlignUp(maskBytes, GM_ALIGN_BYTES) - maskBytes);
        DataCopyExtParams params{1, maskBytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{rightPadding != 0, 0, rightPadding, 0};
        DataCopyPad(maskLocal, maskGm_[offset / BITS_PER_BYTE], params, padParams);
        maskQueue_.EnQue(maskLocal);
    }

    __aicore__ inline void SelectHalf(const LocalTensor<uint8_t>& maskLocal, const LocalTensor<half>& selected,
                                      uint32_t count)
    {
        Duplicate(selected, static_cast<half>(1.0f), count);
        Select(selected, maskLocal, selected, static_cast<half>(0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }

    __aicore__ inline void SelectFloat(const LocalTensor<uint8_t>& maskLocal, const LocalTensor<float>& selected,
                                       uint32_t count)
    {
        Duplicate(selected, 1.0f, count);
        Select(selected, maskLocal, selected, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }

    __aicore__ inline void BuildDoubleGatherOffsets()
    {
        // The work buffer is split into:
        //   packed[0:2*S] | byteOffsets[0:2*S], S = tileElements / 4.
        // Gather offsets map the planar fp32 words
        //   [low_0 ... low_S-1 | high_0 ... high_S-1]
        // to interleaved fp64 storage
        //   [low_0, high_0, low_1, high_1, ...].
        const int32_t outputWords = static_cast<int32_t>(2 * doubleChunkElements_);
        const int32_t chunkElements = static_cast<int32_t>(doubleChunkElements_);
        LocalTensor<int32_t> scratch = workBuffer_.Get<int32_t>();
        LocalTensor<int32_t> halfIndex = scratch;
        LocalTensor<int32_t> byteOffsets = scratch[2 * doubleChunkElements_];
        CreateVecIndex(byteOffsets, static_cast<int32_t>(0), outputWords);
        ShiftRight(halfIndex, byteOffsets, static_cast<int32_t>(1), outputWords);
        Muls(byteOffsets, byteOffsets, static_cast<int32_t>(4 * chunkElements), outputWords);
        Muls(halfIndex, halfIndex, static_cast<int32_t>(4 - 8 * chunkElements), outputWords);
        Add(byteOffsets, byteOffsets, halfIndex, outputWords);
    }

    __aicore__ inline void SelectDouble(const LocalTensor<uint8_t>& maskLocal, const LocalTensor<T>& outLocal,
                                        uint32_t count)
    {
        LocalTensor<float> work = workBuffer_.Get<float>();
        LocalTensor<float> packed = work;
        LocalTensor<uint32_t> byteOffsets = work.template ReinterpretCast<uint32_t>()[2 * doubleChunkElements_];
        LocalTensor<float> outputWords = outLocal.template ReinterpretCast<float>();

        for (uint32_t done = 0; done < count; done += doubleChunkElements_) {
            const uint32_t chunk = static_cast<uint32_t>(
                Min(doubleChunkElements_, static_cast<uint64_t>(count - done)));
            Duplicate(packed, 0.0f, chunk);
            Duplicate(packed[doubleChunkElements_], DOUBLE_ONE_HIGH_WORD, chunk);
            Select(packed[doubleChunkElements_], maskLocal[done / BITS_PER_BYTE], packed[doubleChunkElements_], 0.0f,
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, chunk);
            PipeBarrier<PIPE_V>();
            Gather(outputWords[2 * done], packed, byteOffsets, static_cast<uint32_t>(0), 2 * chunk);
        }
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<uint8_t> maskLocal = maskQueue_.DeQue<uint8_t>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

        if constexpr (std::is_same_v<T, half>) {
            SelectHalf(maskLocal, outLocal, count);
        } else if constexpr (std::is_same_v<T, float>) {
            SelectFloat(maskLocal, outLocal, count);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t>) {
            LocalTensor<half> selected = workBuffer_.Get<half>();
            SelectHalf(maskLocal, selected, count);
            Cast(outLocal, selected, RoundMode::CAST_RINT, count);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<float> selected = workBuffer_.Get<float>();
            SelectFloat(maskLocal, selected, count);
            Cast(outLocal, selected, RoundMode::CAST_RINT, count);
        } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
            LocalTensor<float> selected = workBuffer_.Get<float>();
            SelectFloat(maskLocal, selected, count);
            Cast(outLocal, selected, RoundMode::CAST_TRUNC, count);
        } else if constexpr (IS_DOUBLE) {
            SelectDouble(maskLocal, outLocal, count);
        }

        outQueue_.EnQue(outLocal);
        maskQueue_.FreeTensor(maskLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm_[offset], outLocal, params);
        outQueue_.FreeTensor(outLocal);
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<TPosition::VECIN, BUFFER_NUM> maskQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> workBuffer_;
    GlobalTensor<uint8_t> maskGm_;
    GlobalTensor<T> outGm_;
    uint64_t totalElements_ = 0;
    uint64_t elementsPerCore_ = 0;
    uint64_t tileElements_ = 0;
    uint64_t coreStart_ = 0;
    uint64_t coreElements_ = 0;
    uint32_t doubleChunkElements_ = 0;
    bool maskAliasesOut_ = false;
};
} // namespace BernoulliMask

#endif // BERNOULLI_MASK_H
