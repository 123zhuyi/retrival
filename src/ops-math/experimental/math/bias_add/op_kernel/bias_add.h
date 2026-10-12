/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIAS_ADD_KERNEL_H_
#define BIAS_ADD_KERNEL_H_

#include "kernel_operator.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bias_add_tiling_data.h"
#include <type_traits>

namespace NsBiasAdd {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelBiasAdd {
public:
    __aicore__ inline KernelBiasAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        const uint32_t coreIdx = GetBlockIdx();
        uint64_t globalStart = 0;
        if (coreIdx < tilingData->tailBlockNum) {
            coreDataNum_ = tilingData->bigCoreDataNum;
            tileNum_ = tilingData->finalBigTileNum;
            tailDataNum_ = tilingData->bigTailDataNum;
            globalStart = tilingData->bigCoreDataNum * coreIdx;
        } else {
            coreDataNum_ = tilingData->smallCoreDataNum;
            tileNum_ = tilingData->finalSmallTileNum;
            tailDataNum_ = tilingData->smallTailDataNum;
            globalStart = tilingData->bigCoreDataNum * tilingData->tailBlockNum +
                          tilingData->smallCoreDataNum * (coreIdx - tilingData->tailBlockNum);
        }

        tileDataNum_ = tilingData->tileDataNum;
        biasCacheElems_ = tilingData->biasCacheElems;
        superCycleSize_ = static_cast<uint32_t>(tilingData->superCycleSize);
        kCycleCount_ = static_cast<uint32_t>(tilingData->kCycleCount);
        channelSize_ = tilingData->channelSize;
        innerSize_ = tilingData->innerSize;
        globalStart_ = globalStart;

        xGm_.SetGlobalBuffer((__gm__ T*)x + globalStart_, coreDataNum_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + globalStart_, coreDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ T*)bias, channelSize_);

        pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(biasBuffer_, tileDataNum_ * sizeof(T));
        if (biasCacheElems_ > 0) {
            pipe_.InitBuffer(biasCacheBuffer_, biasCacheElems_ * sizeof(T));
            LocalTensor<T> biasCache = biasCacheBuffer_.Get<T>();
            if (superCycleSize_ > 0 && kCycleCount_ > 0) {
                const uint32_t biasStart = static_cast<uint32_t>(globalStart_ % channelSize_);
                for (uint32_t i = 0; i < biasCacheElems_; ++i) {
                    const uint32_t biasIdx = static_cast<uint32_t>((biasStart + i) % channelSize_);
                    biasCache.SetValue(i, biasGm_.GetValue(biasIdx));
                }
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(biasCache, biasGm_[0], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
        }
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum_ == 0 || tileNum_ == 0) {
            return;
        }
        processDataNum_ = tileDataNum_;
        for (uint32_t i = 0; i < tileNum_; ++i) {
            if (i == tileNum_ - 1) {
                processDataNum_ = tailDataNum_;
            }
            if (processDataNum_ == 0) {
                continue;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

protected:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[progress * tileDataNum_], copyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void FillBias(LocalTensor<T>& biasLocal, uint32_t progress)
    {
        uint64_t globalIndex = globalStart_ + static_cast<uint64_t>(progress) * tileDataNum_;
        if (innerSize_ == 1) {
            uint32_t biasIndex = static_cast<uint32_t>(globalIndex % channelSize_);
            if (biasCacheElems_ > 0) {
                LocalTensor<T> biasCache = biasCacheBuffer_.Get<T>();
                for (uint32_t i = 0; i < processDataNum_; ++i) {
                    biasLocal.SetValue(i, biasCache.GetValue(biasIndex));
                    ++biasIndex;
                    if (biasIndex == channelSize_) {
                        biasIndex = 0;
                    }
                }
            } else {
                for (uint32_t i = 0; i < processDataNum_; ++i) {
                    biasLocal.SetValue(i, biasGm_.GetValue(biasIndex));
                    ++biasIndex;
                    if (biasIndex == channelSize_) {
                        biasIndex = 0;
                    }
                }
            }
            return;
        }

        uint32_t localOffset = 0;
        while (localOffset < processDataNum_) {
            const uint32_t biasIndex = static_cast<uint32_t>((globalIndex / innerSize_) % channelSize_);
            const uint32_t remainInChannel = static_cast<uint32_t>(innerSize_ - (globalIndex % innerSize_));
            const uint32_t remainLocal = processDataNum_ - localOffset;
            const uint32_t runLen = remainInChannel < remainLocal ? remainInChannel : remainLocal;
            const T biasValue = biasCacheElems_ > 0 ? biasCacheBuffer_.Get<T>().GetValue(biasIndex) :
                                                      biasGm_.GetValue(biasIndex);
            for (uint32_t j = 0; j < runLen; ++j) {
                biasLocal.SetValue(localOffset + j, biasValue);
            }
            localOffset += runLen;
            globalIndex += runLen;
        }
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        if (CanUseChannelCacheAdds()) {
            ComputeChannelCacheAdds(xLocal, yLocal, progress);
            xQueue_.FreeTensor(xLocal);
            yQueue_.EnQue(yLocal);
            return;
        }
        if (CanUseKCycleAdds()) {
            ComputeKCycleAdds(xLocal, yLocal, progress);
            xQueue_.FreeTensor(xLocal);
            yQueue_.EnQue(yLocal);
            return;
        }
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value ||
                      std::is_same<T, half>::value) {
            if (CanUseInnerSizeAdds()) {
                ComputeInnerSizeAdds(xLocal, yLocal, progress);
                xQueue_.FreeTensor(xLocal);
                yQueue_.EnQue(yLocal);
                return;
            }
        }
        if (CanUseSegmentAdds(progress)) {
            ComputeSegmentAdds(xLocal, yLocal, progress);
            xQueue_.FreeTensor(xLocal);
            yQueue_.EnQue(yLocal);
            return;
        }

        LocalTensor<T> biasLocal = biasBuffer_.Get<T>();
        FillBias(biasLocal, progress);
        Add(yLocal, xLocal, biasLocal, processDataNum_);
        biasBuffer_.FreeTensor(biasLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline bool CanUseInnerSizeAdds()
    {
        constexpr uint32_t alignElems = 32U / sizeof(T);
        return innerSize_ > 1 && innerSize_ >= alignElems * 2U && processDataNum_ >= alignElems * 4U;
    }

    __aicore__ inline void ComputeInnerSizeAdds(const LocalTensor<T>& xLocal, const LocalTensor<T>& yLocal,
                                                uint32_t progress)
    {
        constexpr uint32_t alignElems = 32U / sizeof(T);
        uint64_t globalIndex = globalStart_ + static_cast<uint64_t>(progress) * tileDataNum_;
        uint32_t localOffset = 0;
        while (localOffset < processDataNum_) {
            const uint32_t biasIndex = static_cast<uint32_t>((globalIndex / innerSize_) % channelSize_);
            const uint32_t remainInChannel = static_cast<uint32_t>(innerSize_ - (globalIndex % innerSize_));
            const uint32_t remainLocal = processDataNum_ - localOffset;
            const uint32_t runLen = remainInChannel < remainLocal ? remainInChannel : remainLocal;
            const T biasValue = biasGm_.GetValue(biasIndex);

            uint32_t consumed = 0;
            uint32_t headLen = (alignElems - (localOffset % alignElems)) % alignElems;
            if (headLen > runLen) {
                headLen = runLen;
            }
            for (uint32_t i = 0; i < headLen; ++i) {
                if constexpr (std::is_same<T, half>::value) {
                    yLocal.SetValue(localOffset + i,
                                    static_cast<half>(static_cast<float>(xLocal.GetValue(localOffset + i)) +
                                                      static_cast<float>(biasValue)));
                } else {
                    yLocal.SetValue(localOffset + i, xLocal.GetValue(localOffset + i) + biasValue);
                }
            }
            consumed += headLen;

            const uint32_t bodyLen = ((runLen - consumed) / alignElems) * alignElems;
            if (bodyLen > 0) {
                // Adds requires 32B-aligned source/destination start addresses; headLen above enforces it.
                Adds(yLocal[localOffset + consumed], xLocal[localOffset + consumed], biasValue,
                     static_cast<int32_t>(bodyLen));
                consumed += bodyLen;
            }

            while (consumed < runLen) {
                if constexpr (std::is_same<T, half>::value) {
                    yLocal.SetValue(localOffset + consumed,
                                    static_cast<half>(static_cast<float>(xLocal.GetValue(localOffset + consumed)) +
                                                      static_cast<float>(biasValue)));
                } else {
                    yLocal.SetValue(localOffset + consumed, xLocal.GetValue(localOffset + consumed) + biasValue);
                }
                ++consumed;
            }

            localOffset += runLen;
            globalIndex += runLen;
        }
    }

    __aicore__ inline bool CanUseSegmentAdds(uint32_t progress)
    {
        constexpr uint32_t alignElems = 32U / sizeof(T);
        const uint64_t globalIndex = globalStart_ + static_cast<uint64_t>(progress) * tileDataNum_;
        return innerSize_ > 1 && (innerSize_ % alignElems) == 0 && (tileDataNum_ % alignElems) == 0 &&
               (globalIndex % innerSize_) == 0;
    }

    __aicore__ inline void ComputeSegmentAdds(const LocalTensor<T>& xLocal, const LocalTensor<T>& yLocal,
                                              uint32_t progress)
    {
        uint64_t globalIndex = globalStart_ + static_cast<uint64_t>(progress) * tileDataNum_;
        uint32_t localOffset = 0;
        while (localOffset < processDataNum_) {
            const uint32_t biasIndex = static_cast<uint32_t>((globalIndex / innerSize_) % channelSize_);
            const uint32_t remainLocal = processDataNum_ - localOffset;
            const uint32_t runLen = innerSize_ < remainLocal ? static_cast<uint32_t>(innerSize_) : remainLocal;
            const T biasValue = biasGm_.GetValue(biasIndex);
            // Vector instructions require 32B-aligned src/dst start addresses.
            // This path is gated so each segment starts at an aligned localOffset;
            // non-aligned head/tail cases intentionally fall back to FillBias+Add.
            Adds(yLocal[localOffset], xLocal[localOffset], biasValue, static_cast<int32_t>(runLen));
            localOffset += runLen;
            globalIndex += runLen;
        }
    }

    __aicore__ inline bool CanUseChannelCacheAdds()
    {
        // innersize==1 path: pre-copied bias cache, per-channel-chunk Add.
        // Gate: bias cache allocated + tile/core channel-aligned + stride 32B-aligned.
        // Per-channel Add at offset=k*channelSize requires dst/src start 32B-aligned.
        // (Confirmed by RE on 16/41/42 where channelSize*sizeof(T) not 32B-multiple.)
        constexpr uint32_t kBlockBytes = 32U;
        const uint64_t strideBytes = static_cast<uint64_t>(channelSize_) * sizeof(T);
        return innerSize_ == 1 && superCycleSize_ == 0 && biasCacheElems_ >= channelSize_ && channelSize_ > 0 &&
               (strideBytes % kBlockBytes) == 0 && (tileDataNum_ % channelSize_) == 0 &&
               (globalStart_ % channelSize_) == 0;
    }

    __aicore__ inline void ComputeChannelCacheAdds(const LocalTensor<T>& xLocal, const LocalTensor<T>& yLocal,
                                                   uint32_t /*progress*/)
    {
        // Aligned channel-cache. The gate guarantees globalStart_ % C == 0 and
        // tileDataNum_ % C == 0, so every tile's broadcast bias pattern is identical
        // (biasIndex is always 0). Replicate bias[0:C] across one whole tile once per
        // core into biasBuffer_ (reused scratch, no extra UB), then do a single wide
        // Add per tile instead of one small Add per channel row. msprof showed this
        // path was scalar/control-bound (per-row loop), not bandwidth-bound; this
        // removes the per-row scalar loop. Channel stride is 32B-aligned, so the
        // replicate is an exact Copy-with-repeat (no GatherMask compaction needed).
        LocalTensor<T> biasFull = biasBuffer_.Get<T>();
        if (!channelFullHoisted_) {
            LocalTensor<T> biasCache = biasCacheBuffer_.Get<T>();
            constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
            const uint32_t cBlocks = channelSize_ / elemsPerBlock; // exact: strideBytes % 32 == 0
            const uint32_t rows = tileDataNum_ / channelSize_;
            SetMaskCount();
            SetVectorMask<T, MaskMode::COUNTER>(channelSize_);
            CopyRepeatParams rep{1, 1, static_cast<uint16_t>(cBlocks), 0};
            uint32_t doneRows = 0;
            while (doneRows < rows) {
                const uint32_t batch = (rows - doneRows) > 255U ? 255U : (rows - doneRows);
                Copy<T, false>(biasFull[doneRows * channelSize_], biasCache, MASK_PLACEHOLDER,
                               static_cast<uint8_t>(batch), rep);
                doneRows += batch;
            }
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();
            channelFullHoisted_ = true;
        }
        Add(yLocal, xLocal, biasFull, static_cast<int32_t>(processDataNum_));
    }

    __aicore__ inline bool CanUseKCycleAdds()
    {
        return innerSize_ == 1 && superCycleSize_ > 0 && biasCacheElems_ >= superCycleSize_ && channelSize_ > 0 &&
               (biasCacheElems_ % superCycleSize_) == 0 && (tileDataNum_ % superCycleSize_) == 0;
    }

    __aicore__ inline void ComputeKCycleAdds(const LocalTensor<T>& xLocal, const LocalTensor<T>& yLocal,
                                             uint32_t /*progress*/)
    {
        // K-cycle is memory-bandwidth-bound here (mte2+mte3 ~ 1.0, bus saturated), not
        // scalar-bound, so a hoist+single-Add rewrite was measured to give no benefit
        // (unlike the aligned ChannelCache path). Keep the simple per-superCycle Add loop.
        LocalTensor<T> biasCache = biasCacheBuffer_.Get<T>();
        const uint32_t chunk = biasCacheElems_;
        uint32_t off = 0;
        while (off + chunk <= processDataNum_) {
            Add(yLocal[off], xLocal[off], biasCache[0], static_cast<int32_t>(chunk));
            off += chunk;
        }
        if (off < processDataNum_) {
            Add(yLocal[off], xLocal[off], biasCache[0], static_cast<int32_t>(processDataNum_ - off));
        }
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[progress * tileDataNum_], yLocal, copyParams);
        yQueue_.FreeTensor(yLocal);
    }

protected:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TBuf<QuePosition::VECCALC> biasBuffer_;
    TBuf<QuePosition::VECCALC> biasCacheBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    uint64_t globalStart_ = 0;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = 0;
    uint32_t tailDataNum_ = 0;
    uint32_t processDataNum_ = 0;
    uint32_t biasCacheElems_ = 0;
    uint32_t superCycleSize_ = 0;
    uint32_t kCycleCount_ = 0;
    uint32_t channelSize_ = 1;
    uint32_t innerSize_ = 1;
    bool channelFullHoisted_ = false;
};

template <typename T>
class KernelBiasAddBroadcastUbTile {
public:
    __aicore__ inline KernelBiasAddBroadcastUbTile() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        const uint32_t coreIdx = GetBlockIdx();
        uint64_t globalStart = 0;
        if (coreIdx < tilingData->tailBlockNum) {
            coreDataNum_ = static_cast<uint32_t>(tilingData->bigCoreDataNum);
            tileNum_ = static_cast<uint32_t>(tilingData->finalBigTileNum);
            tailDataNum_ = static_cast<uint32_t>(tilingData->bigTailDataNum);
            globalStart = static_cast<uint64_t>(tilingData->bigCoreDataNum) * coreIdx;
        } else {
            coreDataNum_ = static_cast<uint32_t>(tilingData->smallCoreDataNum);
            tileNum_ = static_cast<uint32_t>(tilingData->finalSmallTileNum);
            tailDataNum_ = static_cast<uint32_t>(tilingData->smallTailDataNum);
            globalStart = static_cast<uint64_t>(tilingData->bigCoreDataNum) *
                              static_cast<uint64_t>(tilingData->tailBlockNum) +
                          static_cast<uint64_t>(tilingData->smallCoreDataNum) *
                              static_cast<uint64_t>(coreIdx - tilingData->tailBlockNum);
        }

        tileDataNum_ = static_cast<uint32_t>(tilingData->tileDataNum);
        channelSize_ = static_cast<uint32_t>(tilingData->channelSize);
        brcTmpBytes_ = static_cast<uint32_t>(tilingData->brcTmpBytes);
        globalStart_ = globalStart;

        xGm_.SetGlobalBuffer((__gm__ T*)x + globalStart_, coreDataNum_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + globalStart_, coreDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ T*)bias, channelSize_);

        pipe_.InitBuffer(xBuffer_, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(biasBuffer_, GetBiasAlignedElems() * sizeof(T));
        pipe_.InitBuffer(biasFullBuffer_, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(brcTmpBuffer_, brcTmpBytes_);
        evtXReuse_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
        xReusePending_ = false;
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum_ == 0U || tileNum_ == 0U || channelSize_ == 0U) {
            return;
        }

        LocalTensor<T> biasLocal = biasBuffer_.Get<T>();
        DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(biasLocal, biasGm_[0], biasCopyParams, padParams);
        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        // OneDim broadcast hoist: when every full tile starts on a channel
        // boundary (tile size and this core's global start are both multiples
        // of channelSize), the expanded bias pattern is identical for every
        // tile. Materialize it once per core and reuse across all tiles (the
        // tail tile reads the matching prefix), instead of re-running
        // Copy+GatherMask per tile. Pure structural reuse, no shape gate.
        biasFullHoisted_ = (channelSize_ != 0U) && (tileDataNum_ % channelSize_ == 0U) &&
                           (globalStart_ % channelSize_ == 0U);
        if (biasFullHoisted_) {
            LocalTensor<T> biasFull = biasFullBuffer_.Get<T>();
            LocalTensor<uint8_t> brcTmp = brcTmpBuffer_.Get<uint8_t>();
            processDataNum_ = tileDataNum_;
            ExpandBiasGatherMask(biasFull, biasLocal, brcTmp);
            PipeBarrier<PIPE_V>();
        }

        processDataNum_ = tileDataNum_;
        for (uint32_t i = 0; i < tileNum_; ++i) {
            if (i == tileNum_ - 1U) {
                processDataNum_ = tailDataNum_;
            }
            if (processDataNum_ == 0U) {
                continue;
            }
            ProcessTile(i, biasLocal);
        }
        if (xReusePending_) {
            WaitFlag<HardEvent::MTE3_MTE2>(evtXReuse_);
            xReusePending_ = false;
        }
    }

private:
    __aicore__ inline uint32_t GetBiasAlignedElems() const
    {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
        return ((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
    }

    __aicore__ inline void ProcessTile(uint32_t progress, LocalTensor<T>& biasLocal)
    {
        LocalTensor<T> xLocal = xBuffer_.Get<T>();
        LocalTensor<T> biasFull = biasFullBuffer_.Get<T>();
        LocalTensor<uint8_t> brcTmp = brcTmpBuffer_.Get<uint8_t>();

        // xBuffer_ is single-buffered and reused in-place (Add writes it, CopyOut reads it).
        // Gate this tile's CopyIn (MTE2 writes xLocal) on the previous tile's CopyOut
        // (MTE3 reads xLocal) to avoid a cross-tile WAR hazard: with many tiles, if MTE3
        // falls behind, tile N+1's load can overwrite xLocal before tile N's store reads it,
        // corrupting a small fraction of elements. Only manifests when tile count is high
        // (large NHWC tensors); the guard preserves CopyOut/next-compute overlap.
        if (xReusePending_) {
            WaitFlag<HardEvent::MTE3_MTE2>(evtXReuse_);
            xReusePending_ = false;
        }
        DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[static_cast<uint64_t>(progress) * tileDataNum_], xCopyParams, padParams);
        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        if (!biasFullHoisted_) {
            ExpandBiasGatherMask(biasFull, biasLocal, brcTmp);
            PipeBarrier<PIPE_V>();
        }
        Add(xLocal, xLocal, biasFull, processDataNum_);

        event_t eventVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(progress) * tileDataNum_], xLocal, yCopyParams);
        SetFlag<HardEvent::MTE3_MTE2>(evtXReuse_);
        xReusePending_ = true;
    }

    __aicore__ inline void ExpandBiasGatherMask(LocalTensor<T>& biasFull, LocalTensor<T>& biasLocal,
                                                LocalTensor<uint8_t>& tmpBuffer)
    {
        if ((processDataNum_ % channelSize_) != 0U) {
            for (uint32_t i = 0; i < processDataNum_; ++i) {
                biasFull.SetValue(i, biasLocal.GetValue((globalStart_ + i) % channelSize_));
            }
            return;
        }
        constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
        const uint32_t rows = processDataNum_ / channelSize_;
        const uint16_t cBlocks = static_cast<uint16_t>((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock);
        const uint32_t cAlignedElems = static_cast<uint32_t>(cBlocks) * elemsPerBlock;
        LocalTensor<T> tmpLocal = tmpBuffer.template ReinterpretCast<T>();

        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(cAlignedElems);
        CopyRepeatParams copyParams{1, 1, cBlocks, 0};
        Copy<T, false>(tmpLocal, biasLocal, MASK_PLACEHOLDER, static_cast<uint8_t>(rows), copyParams);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherParams{1, static_cast<uint16_t>(rows), cBlocks, 0};
        GatherMask(biasFull, tmpLocal, static_cast<uint8_t>(7), true, channelSize_, gatherParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xBuffer_;
    TBuf<QuePosition::VECCALC> biasBuffer_;
    TBuf<QuePosition::VECCALC> biasFullBuffer_;
    TBuf<QuePosition::VECCALC> brcTmpBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    uint64_t globalStart_ = 0;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = 0;
    uint32_t tailDataNum_ = 0;
    uint32_t processDataNum_ = 0;
    uint32_t channelSize_ = 1;
    uint32_t brcTmpBytes_ = 0;
    bool biasFullHoisted_ = false;
    event_t evtXReuse_ = EVENT_ID0;
    bool xReusePending_ = false;
};

class KernelBiasAddBf16BroadcastUbCastTile {
public:
    __aicore__ inline KernelBiasAddBf16BroadcastUbCastTile() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        const uint32_t coreIdx = GetBlockIdx();
        uint64_t globalStart = 0;
        if (coreIdx < tilingData->tailBlockNum) {
            coreDataNum_ = static_cast<uint32_t>(tilingData->bigCoreDataNum);
            tileNum_ = static_cast<uint32_t>(tilingData->finalBigTileNum);
            tailDataNum_ = static_cast<uint32_t>(tilingData->bigTailDataNum);
            globalStart = static_cast<uint64_t>(tilingData->bigCoreDataNum) * coreIdx;
        } else {
            coreDataNum_ = static_cast<uint32_t>(tilingData->smallCoreDataNum);
            tileNum_ = static_cast<uint32_t>(tilingData->finalSmallTileNum);
            tailDataNum_ = static_cast<uint32_t>(tilingData->smallTailDataNum);
            globalStart = static_cast<uint64_t>(tilingData->bigCoreDataNum) *
                              static_cast<uint64_t>(tilingData->tailBlockNum) +
                          static_cast<uint64_t>(tilingData->smallCoreDataNum) *
                              static_cast<uint64_t>(coreIdx - tilingData->tailBlockNum);
        }

        tileDataNum_ = static_cast<uint32_t>(tilingData->tileDataNum);
        channelSize_ = static_cast<uint32_t>(tilingData->channelSize);
        brcTmpBytes_ = static_cast<uint32_t>(tilingData->brcTmpBytes);
        globalStart_ = globalStart;

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x + globalStart_, coreDataNum_);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t*)y + globalStart_, coreDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ bfloat16_t*)bias, channelSize_);

        pipe_.InitBuffer(xBuffer_, tileDataNum_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(yBuffer_, tileDataNum_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(biasBuffer_, GetBiasAlignedBf16Elems() * sizeof(bfloat16_t));
        pipe_.InitBuffer(xCastBuffer_, tileDataNum_ * sizeof(float));
        pipe_.InitBuffer(biasCastBuffer_, GetBiasAlignedFloatElems() * sizeof(float));
        pipe_.InitBuffer(biasFullBuffer_, tileDataNum_ * sizeof(float));
        pipe_.InitBuffer(yCastBuffer_, tileDataNum_ * sizeof(float));
        pipe_.InitBuffer(brcTmpBuffer_, brcTmpBytes_);
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum_ == 0U || tileNum_ == 0U || channelSize_ == 0U) {
            return;
        }

        LocalTensor<bfloat16_t> biasLocal = biasBuffer_.Get<bfloat16_t>();
        LocalTensor<float> biasFloat = biasCastBuffer_.Get<float>();
        DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(biasLocal, biasGm_[0], biasCopyParams, padParams);
        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        Cast(biasFloat, biasLocal, RoundMode::CAST_NONE, channelSize_);
        PipeBarrier<PIPE_V>();

        // Cross-tile WAR guard on the reused yLocal output buffer: tile N's
        // CopyOut (MTE3 reads yLocal) must finish before tile N+1's final Cast
        // (V writes yLocal). Single-buffered + no inter-tile MTE3->V sync would
        // corrupt ~scale-dependent elements on many-tile shapes. Set after each
        // CopyOut, wait before the next tile's yLocal write (keeps CopyOut/compute
        // overlap, unlike a full per-tile barrier).
        evtYOut_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
        yOutPending_ = false;

        processDataNum_ = tileDataNum_;
        for (uint32_t i = 0; i < tileNum_; ++i) {
            if (i == tileNum_ - 1U) {
                processDataNum_ = tailDataNum_;
            }
            if (processDataNum_ == 0U) {
                continue;
            }
            ProcessTile(i, biasFloat);
        }
        if (yOutPending_) {
            WaitFlag<HardEvent::MTE3_V>(evtYOut_);
            yOutPending_ = false;
        }
    }

private:
    __aicore__ inline uint32_t GetBiasAlignedBf16Elems() const
    {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(bfloat16_t);
        return ((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
    }

    __aicore__ inline uint32_t GetBiasAlignedFloatElems() const
    {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(float);
        return ((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
    }

    __aicore__ inline void ProcessTile(uint32_t progress, LocalTensor<float>& biasFloat)
    {
        LocalTensor<bfloat16_t> xLocal = xBuffer_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = yBuffer_.Get<bfloat16_t>();
        LocalTensor<float> xCast = xCastBuffer_.Get<float>();
        LocalTensor<float> yCast = yCastBuffer_.Get<float>();
        LocalTensor<float> biasFull = biasFullBuffer_.Get<float>();
        LocalTensor<uint8_t> brcTmp = brcTmpBuffer_.Get<uint8_t>();

        DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[static_cast<uint64_t>(progress) * tileDataNum_], xCopyParams, padParams);
        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        Cast(xCast, xLocal, RoundMode::CAST_NONE, processDataNum_);
        PipeBarrier<PIPE_V>();
        ExpandBiasGatherMaskFloat(biasFull, biasFloat, brcTmp);
        PipeBarrier<PIPE_V>();
        Add(yCast, xCast, biasFull, processDataNum_);
        PipeBarrier<PIPE_V>();
        // Wait for the previous tile's CopyOut to finish reading yLocal before
        // overwriting it here (WAR guard, see Process()).
        if (yOutPending_) {
            WaitFlag<HardEvent::MTE3_V>(evtYOut_);
            yOutPending_ = false;
        }
        Cast(yLocal, yCast, RoundMode::CAST_RINT, processDataNum_);

        event_t eventVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(progress) * tileDataNum_], yLocal, yCopyParams);
        SetFlag<HardEvent::MTE3_V>(evtYOut_);
        yOutPending_ = true;
    }

    __aicore__ inline void ExpandBiasGatherMaskFloat(LocalTensor<float>& biasFull, LocalTensor<float>& biasFloat,
                                                     LocalTensor<uint8_t>& tmpBuffer)
    {
        if ((processDataNum_ % channelSize_) != 0U) {
            for (uint32_t i = 0; i < processDataNum_; ++i) {
                biasFull.SetValue(i, biasFloat.GetValue((globalStart_ + i) % channelSize_));
            }
            return;
        }
        constexpr uint32_t elemsPerBlock = 32U / sizeof(float);
        const uint32_t rows = processDataNum_ / channelSize_;
        const uint16_t cBlocks = static_cast<uint16_t>((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock);
        const uint32_t cAlignedElems = static_cast<uint32_t>(cBlocks) * elemsPerBlock;
        LocalTensor<float> tmpLocal = tmpBuffer.template ReinterpretCast<float>();

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(cAlignedElems);
        CopyRepeatParams copyParams{1, 1, cBlocks, 0};
        Copy<float, false>(tmpLocal, biasFloat, MASK_PLACEHOLDER, static_cast<uint8_t>(rows), copyParams);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherParams{1, static_cast<uint16_t>(rows), cBlocks, 0};
        GatherMask(biasFull, tmpLocal, static_cast<uint8_t>(7), true, channelSize_, gatherParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xBuffer_;
    TBuf<QuePosition::VECCALC> yBuffer_;
    TBuf<QuePosition::VECCALC> biasBuffer_;
    TBuf<QuePosition::VECCALC> xCastBuffer_;
    TBuf<QuePosition::VECCALC> biasCastBuffer_;
    TBuf<QuePosition::VECCALC> biasFullBuffer_;
    TBuf<QuePosition::VECCALC> yCastBuffer_;
    TBuf<QuePosition::VECCALC> brcTmpBuffer_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<bfloat16_t> biasGm_;
    GlobalTensor<bfloat16_t> yGm_;
    uint64_t globalStart_ = 0;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = 0;
    uint32_t tailDataNum_ = 0;
    uint32_t processDataNum_ = 0;
    uint32_t channelSize_ = 1;
    uint32_t brcTmpBytes_ = 0;
    event_t evtYOut_ = EVENT_ID0;
    bool yOutPending_ = false;
};

template <>
class KernelBiasAdd<bfloat16_t> {
public:
    __aicore__ inline KernelBiasAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        const uint32_t coreIdx = GetBlockIdx();
        uint64_t globalStart = 0;
        if (coreIdx < tilingData->tailBlockNum) {
            coreDataNum_ = tilingData->bigCoreDataNum;
            tileNum_ = tilingData->finalBigTileNum;
            tailDataNum_ = tilingData->bigTailDataNum;
            globalStart = tilingData->bigCoreDataNum * coreIdx;
        } else {
            coreDataNum_ = tilingData->smallCoreDataNum;
            tileNum_ = tilingData->finalSmallTileNum;
            tailDataNum_ = tilingData->smallTailDataNum;
            globalStart = tilingData->bigCoreDataNum * tilingData->tailBlockNum +
                          tilingData->smallCoreDataNum * (coreIdx - tilingData->tailBlockNum);
        }

        tileDataNum_ = tilingData->tileDataNum;
        biasCacheElems_ = tilingData->biasCacheElems;
        superCycleSize_ = static_cast<uint32_t>(tilingData->superCycleSize);
        kCycleCount_ = static_cast<uint32_t>(tilingData->kCycleCount);
        channelSize_ = tilingData->channelSize;
        innerSize_ = tilingData->innerSize;
        globalStart_ = globalStart;

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x + globalStart_, coreDataNum_);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t*)y + globalStart_, coreDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ bfloat16_t*)bias, channelSize_);

        pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileDataNum_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, tileDataNum_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(biasBuffer_, tileDataNum_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(xCastBuffer_, tileDataNum_ * sizeof(float));
        pipe_.InitBuffer(biasCastBuffer_, tileDataNum_ * sizeof(float));
        pipe_.InitBuffer(yCastBuffer_, tileDataNum_ * sizeof(float));
        if (CanUseInnerSizeAddsBf16()) {
            LocalTensor<bfloat16_t> biasLocal = biasBuffer_.Get<bfloat16_t>();
            LocalTensor<float> biasFloatCache = biasCastBuffer_.Get<float>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
            DataCopyPad(biasLocal, biasGm_[0], copyParams, padParams);
            PipeBarrier<PIPE_MTE2>();
            Cast(biasFloatCache, biasLocal, RoundMode::CAST_NONE, channelSize_);
            PipeBarrier<PIPE_V>();
            biasBuffer_.FreeTensor(biasLocal);
        }
        if (biasCacheElems_ > 0) {
            pipe_.InitBuffer(biasCacheBuffer_, biasCacheElems_ * sizeof(bfloat16_t));
            LocalTensor<bfloat16_t> biasCache = biasCacheBuffer_.Get<bfloat16_t>();
            if (superCycleSize_ > 0 && kCycleCount_ > 0) {
                const uint32_t biasStart = static_cast<uint32_t>(globalStart_ % channelSize_);
                for (uint32_t i = 0; i < biasCacheElems_; ++i) {
                    const uint32_t biasIdx = static_cast<uint32_t>((biasStart + i) % channelSize_);
                    biasCache.SetValue(i, biasGm_.GetValue(biasIdx));
                }
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
                DataCopyPad(biasCache, biasGm_[0], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
        }
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum_ == 0 || tileNum_ == 0) {
            return;
        }
        processDataNum_ = tileDataNum_;
        for (uint32_t i = 0; i < tileNum_; ++i) {
            if (i == tileNum_ - 1) {
                processDataNum_ = tailDataNum_;
            }
            if (processDataNum_ == 0) {
                continue;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<bfloat16_t> xLocal = xQueue_.AllocTensor<bfloat16_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[progress * tileDataNum_], copyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void FillBias(LocalTensor<bfloat16_t>& biasLocal, uint32_t progress)
    {
        uint64_t globalIndex = globalStart_ + static_cast<uint64_t>(progress) * tileDataNum_;
        if (biasCacheElems_ > 0) {
            LocalTensor<bfloat16_t> biasCache = biasCacheBuffer_.Get<bfloat16_t>();
            if (innerSize_ == 1) {
                uint32_t biasIndex = static_cast<uint32_t>(globalIndex % channelSize_);
                for (uint32_t i = 0; i < processDataNum_; ++i) {
                    biasLocal.SetValue(i, biasCache.GetValue(biasIndex));
                    ++biasIndex;
                    if (biasIndex == channelSize_) {
                        biasIndex = 0;
                    }
                }
                return;
            }

            uint32_t localOffset = 0;
            while (localOffset < processDataNum_) {
                const uint32_t biasIndex = static_cast<uint32_t>((globalIndex / innerSize_) % channelSize_);
                const uint32_t remainInChannel = static_cast<uint32_t>(innerSize_ - (globalIndex % innerSize_));
                const uint32_t remainLocal = processDataNum_ - localOffset;
                const uint32_t runLen = remainInChannel < remainLocal ? remainInChannel : remainLocal;
                const bfloat16_t biasValue = biasCache.GetValue(biasIndex);
                for (uint32_t j = 0; j < runLen; ++j) {
                    biasLocal.SetValue(localOffset + j, biasValue);
                }
                localOffset += runLen;
                globalIndex += runLen;
            }
        } else {
            if (innerSize_ == 1) {
                uint32_t biasIndex = static_cast<uint32_t>(globalIndex % channelSize_);
                for (uint32_t i = 0; i < processDataNum_; ++i) {
                    biasLocal.SetValue(i, biasGm_.GetValue(biasIndex));
                    ++biasIndex;
                    if (biasIndex == channelSize_) {
                        biasIndex = 0;
                    }
                }
                return;
            }

            uint32_t localOffset = 0;
            while (localOffset < processDataNum_) {
                const uint32_t biasIndex = static_cast<uint32_t>((globalIndex / innerSize_) % channelSize_);
                const uint32_t remainInChannel = static_cast<uint32_t>(innerSize_ - (globalIndex % innerSize_));
                const uint32_t remainLocal = processDataNum_ - localOffset;
                const uint32_t runLen = remainInChannel < remainLocal ? remainInChannel : remainLocal;
                const bfloat16_t biasValue = biasGm_.GetValue(biasIndex);
                for (uint32_t j = 0; j < runLen; ++j) {
                    biasLocal.SetValue(localOffset + j, biasValue);
                }
                localOffset += runLen;
                globalIndex += runLen;
            }
        }
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<bfloat16_t> xLocal = xQueue_.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = yQueue_.AllocTensor<bfloat16_t>();

        if (superCycleSize_ > 0 && biasCacheElems_ >= superCycleSize_ && (biasCacheElems_ % superCycleSize_) == 0 &&
            (tileDataNum_ % superCycleSize_) == 0) {
            LocalTensor<float> xCast = xCastBuffer_.Get<float>();
            LocalTensor<float> biasCast = biasCastBuffer_.Get<float>();
            LocalTensor<float> yCast = yCastBuffer_.Get<float>();
            LocalTensor<bfloat16_t> biasCache = biasCacheBuffer_.Get<bfloat16_t>();
            Cast(xCast, xLocal, RoundMode::CAST_NONE, processDataNum_);
            Cast(biasCast, biasCache, RoundMode::CAST_NONE, biasCacheElems_);
            const uint32_t chunk = biasCacheElems_;
            uint32_t off = 0;
            while (off + chunk <= processDataNum_) {
                Add(yCast[off], xCast[off], biasCast, static_cast<int32_t>(chunk));
                off += chunk;
            }
            if (off < processDataNum_) {
                Add(yCast[off], xCast[off], biasCast, static_cast<int32_t>(processDataNum_ - off));
            }
            Cast(yLocal, yCast, RoundMode::CAST_RINT, processDataNum_);
            yCastBuffer_.FreeTensor(yCast);
            biasCastBuffer_.FreeTensor(biasCast);
            xCastBuffer_.FreeTensor(xCast);
            xQueue_.FreeTensor(xLocal);
            yQueue_.EnQue(yLocal);
            return;
        }

        if (innerSize_ == 1 && biasCacheElems_ >= channelSize_ && channelSize_ > 0 &&
            (static_cast<uint64_t>(channelSize_) * sizeof(bfloat16_t)) % 32U == 0 && tileDataNum_ % channelSize_ == 0 &&
            globalStart_ % channelSize_ == 0) {
            LocalTensor<float> xCast = xCastBuffer_.Get<float>();
            LocalTensor<float> biasCast = biasCastBuffer_.Get<float>();
            LocalTensor<float> yCast = yCastBuffer_.Get<float>();
            ComputeChannelCached(xLocal, yLocal, xCast, biasCast, yCast, progress);
            yCastBuffer_.FreeTensor(yCast);
            biasCastBuffer_.FreeTensor(biasCast);
            xCastBuffer_.FreeTensor(xCast);
            xQueue_.FreeTensor(xLocal);
            yQueue_.EnQue(yLocal);
            return;
        }

        if (CanUseInnerSizeAddsBf16()) {
            LocalTensor<float> xCast = xCastBuffer_.Get<float>();
            LocalTensor<float> biasFloatCache = biasCastBuffer_.Get<float>();
            LocalTensor<float> yCast = yCastBuffer_.Get<float>();
            ComputeInnerSizeAddsBf16(xLocal, yLocal, biasFloatCache, xCast, yCast, progress);
            yCastBuffer_.FreeTensor(yCast);
            xCastBuffer_.FreeTensor(xCast);
            xQueue_.FreeTensor(xLocal);
            yQueue_.EnQue(yLocal);
            return;
        }

        LocalTensor<bfloat16_t> biasLocal = biasBuffer_.Get<bfloat16_t>();
        FillBias(biasLocal, progress);

        LocalTensor<float> xCast = xCastBuffer_.Get<float>();
        LocalTensor<float> biasCast = biasCastBuffer_.Get<float>();
        LocalTensor<float> yCast = yCastBuffer_.Get<float>();
        Cast(xCast, xLocal, RoundMode::CAST_NONE, processDataNum_);
        Cast(biasCast, biasLocal, RoundMode::CAST_NONE, processDataNum_);
        Add(yCast, xCast, biasCast, processDataNum_);
        Cast(yLocal, yCast, RoundMode::CAST_RINT, processDataNum_);

        yCastBuffer_.FreeTensor(yCast);
        biasCastBuffer_.FreeTensor(biasCast);
        xCastBuffer_.FreeTensor(xCast);
        biasBuffer_.FreeTensor(biasLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline bool CanUseInnerSizeAddsBf16()
    {
        return innerSize_ > 1 && channelSize_ > 0 && tileDataNum_ >= channelSize_;
    }

    __aicore__ inline void ComputeInnerSizeAddsBf16(const LocalTensor<bfloat16_t>& xLocal,
                                                    const LocalTensor<bfloat16_t>& yLocal,
                                                    const LocalTensor<float>& biasFloatCache,
                                                    const LocalTensor<float>& xCast, const LocalTensor<float>& yCast,
                                                    uint32_t progress)
    {
        constexpr uint32_t alignElems = 32U / sizeof(float);
        Cast(xCast, xLocal, RoundMode::CAST_NONE, processDataNum_);
        PipeBarrier<PIPE_V>();

        uint64_t globalIndex = globalStart_ + static_cast<uint64_t>(progress) * tileDataNum_;
        uint32_t localOffset = 0;
        while (localOffset < processDataNum_) {
            const uint32_t biasIndex = static_cast<uint32_t>((globalIndex / innerSize_) % channelSize_);
            const uint32_t remainInChannel = static_cast<uint32_t>(innerSize_ - (globalIndex % innerSize_));
            const uint32_t remainLocal = processDataNum_ - localOffset;
            const uint32_t runLen = remainInChannel < remainLocal ? remainInChannel : remainLocal;
            const float biasValue = biasFloatCache.GetValue(biasIndex);

            uint32_t consumed = 0;
            uint32_t headLen = (alignElems - (localOffset % alignElems)) % alignElems;
            if (headLen > runLen) {
                headLen = runLen;
            }
            for (uint32_t i = 0; i < headLen; ++i) {
                yCast.SetValue(localOffset + i, xCast.GetValue(localOffset + i) + biasValue);
            }
            consumed += headLen;

            const uint32_t bodyLen = ((runLen - consumed) / alignElems) * alignElems;
            if (bodyLen > 0) {
                Adds(yCast[localOffset + consumed], xCast[localOffset + consumed], biasValue,
                     static_cast<int32_t>(bodyLen));
                consumed += bodyLen;
            }

            while (consumed < runLen) {
                yCast.SetValue(localOffset + consumed, xCast.GetValue(localOffset + consumed) + biasValue);
                ++consumed;
            }

            localOffset += runLen;
            globalIndex += runLen;
        }
        PipeBarrier<PIPE_ALL>();
        Cast(yLocal, yCast, RoundMode::CAST_RINT, processDataNum_);
    }

    __aicore__ inline void ComputeChannelCached(const LocalTensor<bfloat16_t>& xLocal,
                                                const LocalTensor<bfloat16_t>& yLocal, const LocalTensor<float>& xCast,
                                                const LocalTensor<float>& biasCast, const LocalTensor<float>& yCast,
                                                uint32_t progress)
    {
        (void)progress;
        LocalTensor<bfloat16_t> biasCache = biasCacheBuffer_.Get<bfloat16_t>();
        Cast(xCast, xLocal, RoundMode::CAST_NONE, processDataNum_);

        uint32_t localOffset = 0;
        while (localOffset + channelSize_ <= processDataNum_) {
            Cast(biasCast[localOffset], biasCache[0], RoundMode::CAST_NONE, channelSize_);
            Add(yCast[localOffset], xCast[localOffset], biasCast[localOffset], static_cast<int32_t>(channelSize_));
            localOffset += channelSize_;
        }

        if (localOffset < processDataNum_) {
            const uint32_t tailLen = processDataNum_ - localOffset;
            Cast(biasCast[localOffset], biasCache[0], RoundMode::CAST_NONE, tailLen);
            Add(yCast[localOffset], xCast[localOffset], biasCast[localOffset], static_cast<int32_t>(tailLen));
        }
        Cast(yLocal, yCast, RoundMode::CAST_RINT, processDataNum_);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<bfloat16_t> yLocal = yQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(yGm_[progress * tileDataNum_], yLocal, copyParams);
        yQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TBuf<QuePosition::VECCALC> biasBuffer_;
    TBuf<QuePosition::VECCALC> xCastBuffer_;
    TBuf<QuePosition::VECCALC> biasCastBuffer_;
    TBuf<QuePosition::VECCALC> yCastBuffer_;
    TBuf<QuePosition::VECCALC> biasCacheBuffer_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<bfloat16_t> biasGm_;
    GlobalTensor<bfloat16_t> yGm_;
    uint64_t globalStart_ = 0;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = 0;
    uint32_t biasCacheElems_ = 0;
    uint32_t tailDataNum_ = 0;
    uint32_t processDataNum_ = 0;
    uint32_t superCycleSize_ = 0;
    uint32_t kCycleCount_ = 0;
    uint32_t channelSize_ = 1;
    uint32_t innerSize_ = 1;
};

template <typename T>
class KernelBiasAddTinyNoQueue {
public:
    __aicore__ inline KernelBiasAddTinyNoQueue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        processDataNum_ = static_cast<uint32_t>(tilingData->totalElements);
        channelSize_ = static_cast<uint32_t>(tilingData->channelSize);
        innerSize_ = static_cast<uint32_t>(tilingData->innerSize);

        xGm_.SetGlobalBuffer((__gm__ T*)x, processDataNum_);
        yGm_.SetGlobalBuffer((__gm__ T*)y, processDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ T*)bias, channelSize_);

        pipe_.InitBuffer(xBuffer_, processDataNum_ * sizeof(T));
        pipe_.InitBuffer(yBuffer_, processDataNum_ * sizeof(T));
        pipe_.InitBuffer(biasBuffer_, processDataNum_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (processDataNum_ == 0 || channelSize_ == 0) {
            return;
        }

        LocalTensor<T> xLocal = xBuffer_.Get<T>();
        LocalTensor<T> yLocal = yBuffer_.Get<T>();
        LocalTensor<T> biasLocal = biasBuffer_.Get<T>();

        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[0], copyInParams, padParams);

        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        FillBias(biasLocal);
        Add(yLocal, xLocal, biasLocal, processDataNum_);

        event_t eventVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[0], yLocal, copyOutParams);
    }

private:
    __aicore__ inline void FillBias(LocalTensor<T>& biasLocal)
    {
        uint64_t globalIndex = 0;
        if (innerSize_ == 1) {
            uint32_t biasIndex = 0;
            for (uint32_t i = 0; i < processDataNum_; ++i) {
                biasLocal.SetValue(i, biasGm_.GetValue(biasIndex));
                ++biasIndex;
                if (biasIndex == channelSize_) {
                    biasIndex = 0;
                }
            }
            return;
        }

        uint32_t localOffset = 0;
        while (localOffset < processDataNum_) {
            const uint32_t biasIndex = static_cast<uint32_t>((globalIndex / innerSize_) % channelSize_);
            const uint32_t remainInChannel = static_cast<uint32_t>(innerSize_ - (globalIndex % innerSize_));
            const uint32_t remainLocal = processDataNum_ - localOffset;
            const uint32_t runLen = remainInChannel < remainLocal ? remainInChannel : remainLocal;
            const T biasValue = biasGm_.GetValue(biasIndex);
            for (uint32_t j = 0; j < runLen; ++j) {
                biasLocal.SetValue(localOffset + j, biasValue);
            }
            localOffset += runLen;
            globalIndex += runLen;
        }
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xBuffer_;
    TBuf<QuePosition::VECCALC> yBuffer_;
    TBuf<QuePosition::VECCALC> biasBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    uint32_t processDataNum_ = 0;
    uint32_t channelSize_ = 1;
    uint32_t innerSize_ = 1;
};

class KernelBiasAddThinTinyNhwcBf16Runtime {
public:
    __aicore__ inline KernelBiasAddThinTinyNhwcBf16Runtime() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        processDataNum_ = static_cast<uint32_t>(tilingData->totalElements);
        channelSize_ = static_cast<uint32_t>(tilingData->channelSize);

        // Defensive: GetTmpFloatElems() below divides by channelSize_. Host GetShapeAttrsInfo
        // guarantees channelSize >= 1, so this never triggers in valid builds, but guard here so a
        // corrupted/unvalidated tiling cannot divide-by-zero during kernel init. Process() applies
        // the same guard before doing any work.
        if (channelSize_ == 0U) {
            return;
        }

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x, processDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ bfloat16_t*)bias, channelSize_);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t*)y, processDataNum_);

        pipe_.InitBuffer(xBuffer_, processDataNum_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(yBuffer_, processDataNum_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(xCastBuffer_, processDataNum_ * sizeof(float));
        pipe_.InitBuffer(yCastBuffer_, processDataNum_ * sizeof(float));
        pipe_.InitBuffer(biasFullFloatBuffer_, processDataNum_ * sizeof(float));
        pipe_.InitBuffer(biasBf16Buffer_, GetBiasBf16AlignedElems() * sizeof(bfloat16_t));
        pipe_.InitBuffer(biasFloatBuffer_, GetBiasFloatAlignedElems() * sizeof(float));
        pipe_.InitBuffer(tmpFloatBuffer_, GetTmpFloatElems() * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (processDataNum_ == 0U || channelSize_ == 0U || (processDataNum_ % channelSize_) != 0U) {
            return;
        }

        LocalTensor<bfloat16_t> xLocal = xBuffer_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = yBuffer_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> biasBf16 = biasBf16Buffer_.Get<bfloat16_t>();
        LocalTensor<float> xCast = xCastBuffer_.Get<float>();
        LocalTensor<float> yCast = yCastBuffer_.Get<float>();
        LocalTensor<float> biasFloat = biasFloatBuffer_.Get<float>();
        LocalTensor<float> biasFull = biasFullFloatBuffer_.Get<float>();
        LocalTensor<float> tmpFloat = tmpFloatBuffer_.Get<float>();

        DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[0], xCopyParams, padParams);
        DataCopyPad(biasBf16, biasGm_[0], biasCopyParams, padParams);

        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        Cast(biasFloat, biasBf16, RoundMode::CAST_NONE, channelSize_);
        PipeBarrier<PIPE_V>();
        ExpandBiasFloat(biasFull, biasFloat, tmpFloat);
        Cast(xCast, xLocal, RoundMode::CAST_NONE, processDataNum_);
        Add(yCast, xCast, biasFull, processDataNum_);
        Cast(yLocal, yCast, RoundMode::CAST_RINT, processDataNum_);

        event_t eventVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(yGm_[0], yLocal, copyOutParams);
    }

private:
    __aicore__ inline void ExpandBiasFloat(LocalTensor<float>& biasFull, LocalTensor<float>& biasC,
                                           LocalTensor<float>& tmp)
    {
        const uint32_t rows = processDataNum_ / channelSize_;
        const uint16_t cBlocks = static_cast<uint16_t>((channelSize_ + FLOAT_ELEMS_PER_BLOCK - 1U) /
                                                       FLOAT_ELEMS_PER_BLOCK);
        const uint32_t cAlignedElems = static_cast<uint32_t>(cBlocks) * FLOAT_ELEMS_PER_BLOCK;

        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(cAlignedElems);
        CopyRepeatParams copyParams{1, 1, cBlocks, 0};
        Copy<float, false>(tmp, biasC, MASK_PLACEHOLDER, static_cast<uint8_t>(rows), copyParams);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherParams{1, static_cast<uint16_t>(rows), cBlocks, 0};
        GatherMask(biasFull, tmp, static_cast<uint8_t>(7), true, channelSize_, gatherParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline uint32_t GetBiasBf16AlignedElems() const
    {
        return ((channelSize_ + BF16_ELEMS_PER_BLOCK - 1U) / BF16_ELEMS_PER_BLOCK) * BF16_ELEMS_PER_BLOCK;
    }

    __aicore__ inline uint32_t GetBiasFloatAlignedElems() const
    {
        return ((channelSize_ + FLOAT_ELEMS_PER_BLOCK - 1U) / FLOAT_ELEMS_PER_BLOCK) * FLOAT_ELEMS_PER_BLOCK;
    }

    __aicore__ inline uint32_t GetTmpFloatElems() const
    {
        return (processDataNum_ / channelSize_) * GetBiasFloatAlignedElems();
    }

private:
    static constexpr uint32_t BF16_ELEMS_PER_BLOCK = 32U / sizeof(bfloat16_t);
    static constexpr uint32_t FLOAT_ELEMS_PER_BLOCK = 32U / sizeof(float);
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xBuffer_;
    TBuf<QuePosition::VECCALC> yBuffer_;
    TBuf<QuePosition::VECCALC> xCastBuffer_;
    TBuf<QuePosition::VECCALC> yCastBuffer_;
    TBuf<QuePosition::VECCALC> biasBf16Buffer_;
    TBuf<QuePosition::VECCALC> biasFloatBuffer_;
    TBuf<QuePosition::VECCALC> biasFullFloatBuffer_;
    TBuf<QuePosition::VECCALC> tmpFloatBuffer_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<bfloat16_t> biasGm_;
    GlobalTensor<bfloat16_t> yGm_;
    uint32_t processDataNum_ = 0;
    uint32_t channelSize_ = 0;
};

template <typename T>
class KernelBiasAddThinTinyNhwcVectorBroadcast {
public:
    __aicore__ inline KernelBiasAddThinTinyNhwcVectorBroadcast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        processDataNum_ = static_cast<uint32_t>(tilingData->totalElements);
        channelSize_ = static_cast<uint32_t>(tilingData->channelSize);
        const uint32_t brcTmpBytes = static_cast<uint32_t>(tilingData->brcTmpBytes);
        brcTmpBytes_ = brcTmpBytes == 0U ? DEFAULT_BRC_TMP_BYTES : brcTmpBytes;

        xGm_.SetGlobalBuffer((__gm__ T*)x, processDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ T*)bias, channelSize_);
        yGm_.SetGlobalBuffer((__gm__ T*)y, processDataNum_);

        pipe_.InitBuffer(xBuffer_, processDataNum_ * sizeof(T));
        pipe_.InitBuffer(biasBuffer_, GetBiasAlignedElems() * sizeof(T));
        pipe_.InitBuffer(biasFullBuffer_, processDataNum_ * sizeof(T));
        pipe_.InitBuffer(brcTmpBuffer_, brcTmpBytes_);
    }

    __aicore__ inline void Process()
    {
        if (processDataNum_ == 0U || channelSize_ == 0U || (processDataNum_ % channelSize_) != 0U) {
            return;
        }
        LocalTensor<T> xLocal = xBuffer_.Get<T>();
        LocalTensor<T> biasLocal = biasBuffer_.Get<T>();
        LocalTensor<T> biasFull = biasFullBuffer_.Get<T>();
        LocalTensor<uint8_t> brcTmp = brcTmpBuffer_.Get<uint8_t>();

        DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[0], xCopyParams, padParams);
        DataCopyPad(biasLocal, biasGm_[0], biasCopyParams, padParams);

        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        ExpandBiasGatherMask(biasFull, biasLocal, brcTmp);
        PipeBarrier<PIPE_V>();
        Add(xLocal, xLocal, biasFull, processDataNum_);

        event_t eventVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[0], xLocal, copyOutParams);
    }

private:
    __aicore__ inline uint32_t GetBiasAlignedElems() const
    {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
        return ((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
    }

    __aicore__ inline void ExpandBiasGatherMask(LocalTensor<T>& biasFull, LocalTensor<T>& biasLocal,
                                                LocalTensor<uint8_t>& tmpBuffer)
    {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
        const uint32_t rows = processDataNum_ / channelSize_;
        const uint16_t cBlocks = static_cast<uint16_t>((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock);
        const uint32_t cAlignedElems = static_cast<uint32_t>(cBlocks) * elemsPerBlock;
        LocalTensor<T> tmpLocal = tmpBuffer.template ReinterpretCast<T>();

        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(cAlignedElems);
        CopyRepeatParams copyParams{1, 1, cBlocks, 0};
        Copy<T, false>(tmpLocal, biasLocal, MASK_PLACEHOLDER, static_cast<uint8_t>(rows), copyParams);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherParams{1, static_cast<uint16_t>(rows), cBlocks, 0};
        GatherMask(biasFull, tmpLocal, static_cast<uint8_t>(7), true, channelSize_, gatherParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    }

private:
    static constexpr uint32_t DEFAULT_BRC_TMP_BYTES = 8192U;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xBuffer_;
    TBuf<QuePosition::VECCALC> biasBuffer_;
    TBuf<QuePosition::VECCALC> biasFullBuffer_;
    TBuf<QuePosition::VECCALC> brcTmpBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    uint32_t processDataNum_ = 0;
    uint32_t channelSize_ = 0;
    uint32_t brcTmpBytes_ = DEFAULT_BRC_TMP_BYTES;
};

template <typename T>
class KernelBiasAddThinTinyNhwcVectorBroadcastOutplace {
public:
    __aicore__ inline KernelBiasAddThinTinyNhwcVectorBroadcastOutplace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        processDataNum_ = static_cast<uint32_t>(tilingData->totalElements);
        channelSize_ = static_cast<uint32_t>(tilingData->channelSize);
        const uint32_t brcTmpBytes = static_cast<uint32_t>(tilingData->brcTmpBytes);
        brcTmpBytes_ = brcTmpBytes == 0U ? DEFAULT_BRC_TMP_BYTES : brcTmpBytes;

        xGm_.SetGlobalBuffer((__gm__ T*)x, processDataNum_);
        biasGm_.SetGlobalBuffer((__gm__ T*)bias, channelSize_);
        yGm_.SetGlobalBuffer((__gm__ T*)y, processDataNum_);

        pipe_.InitBuffer(xBuffer_, processDataNum_ * sizeof(T));
        pipe_.InitBuffer(yBuffer_, processDataNum_ * sizeof(T));
        pipe_.InitBuffer(biasBuffer_, GetBiasAlignedElems() * sizeof(T));
        pipe_.InitBuffer(biasFullBuffer_, processDataNum_ * sizeof(T));
        pipe_.InitBuffer(brcTmpBuffer_, brcTmpBytes_);
    }

    __aicore__ inline void Process()
    {
        if (processDataNum_ == 0U || channelSize_ == 0U || (processDataNum_ % channelSize_) != 0U) {
            return;
        }
        LocalTensor<T> xLocal = xBuffer_.Get<T>();
        LocalTensor<T> yLocal = yBuffer_.Get<T>();
        LocalTensor<T> biasLocal = biasBuffer_.Get<T>();
        LocalTensor<T> biasFull = biasFullBuffer_.Get<T>();
        LocalTensor<uint8_t> brcTmp = brcTmpBuffer_.Get<uint8_t>();

        DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyExtParams biasCopyParams{1, static_cast<uint32_t>(channelSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[0], xCopyParams, padParams);
        DataCopyPad(biasLocal, biasGm_[0], biasCopyParams, padParams);

        event_t eventMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        ExpandBiasGatherMask(biasFull, biasLocal, brcTmp);
        PipeBarrier<PIPE_V>();
        Add(yLocal, xLocal, biasFull, processDataNum_);

        event_t eventVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(processDataNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[0], yLocal, copyOutParams);
    }

private:
    __aicore__ inline uint32_t GetBiasAlignedElems() const
    {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
        return ((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
    }

    __aicore__ inline void ExpandBiasGatherMask(LocalTensor<T>& biasFull, LocalTensor<T>& biasLocal,
                                                LocalTensor<uint8_t>& tmpBuffer)
    {
        constexpr uint32_t elemsPerBlock = 32U / sizeof(T);
        const uint32_t rows = processDataNum_ / channelSize_;
        const uint16_t cBlocks = static_cast<uint16_t>((channelSize_ + elemsPerBlock - 1U) / elemsPerBlock);
        const uint32_t cAlignedElems = static_cast<uint32_t>(cBlocks) * elemsPerBlock;
        LocalTensor<T> tmpLocal = tmpBuffer.template ReinterpretCast<T>();

        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(cAlignedElems);
        CopyRepeatParams copyParams{1, 1, cBlocks, 0};
        Copy<T, false>(tmpLocal, biasLocal, MASK_PLACEHOLDER, static_cast<uint8_t>(rows), copyParams);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherParams{1, static_cast<uint16_t>(rows), cBlocks, 0};
        GatherMask(biasFull, tmpLocal, static_cast<uint8_t>(7), true, channelSize_, gatherParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    }

private:
    static constexpr uint32_t DEFAULT_BRC_TMP_BYTES = 8192U;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xBuffer_;
    TBuf<QuePosition::VECCALC> yBuffer_;
    TBuf<QuePosition::VECCALC> biasBuffer_;
    TBuf<QuePosition::VECCALC> biasFullBuffer_;
    TBuf<QuePosition::VECCALC> brcTmpBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    uint32_t processDataNum_ = 0;
    uint32_t channelSize_ = 0;
    uint32_t brcTmpBytes_ = DEFAULT_BRC_TMP_BYTES;
};

template <>
class KernelBiasAddTinyNoQueue<bfloat16_t> {
public:
    __aicore__ inline KernelBiasAddTinyNoQueue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, const BiasAddTilingData* tilingData)
    {
        base_.Init(x, bias, y, tilingData);
    }

    __aicore__ inline void Process() { base_.Process(); }

private:
    KernelBiasAdd<bfloat16_t> base_;
};

} // namespace NsBiasAdd

#endif // BIAS_ADD_KERNEL_H_
