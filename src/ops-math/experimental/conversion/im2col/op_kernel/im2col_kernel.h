/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXPERIMENTAL_IM2COL_KERNEL_IMPL_H_
#define EXPERIMENTAL_IM2COL_KERNEL_IMPL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "im2col_tiling_data.h"
#include "im2col_tiling_key.h"

namespace NsIm2col {
using namespace AscendC;

template <typename T, uint32_t Path>
class Im2colKernel {
public:
    __aicore__ inline Im2colKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, const Im2colTilingHeader* td, TPipe* pipe)
    {
        pipe_ = pipe;
        n_ = td->n;
        c_ = td->c;
        h_ = td->h;
        w_ = td->w;
        kernelH_ = td->kernelH;
        kernelW_ = td->kernelW;
        strideH_ = td->strideH;
        strideW_ = td->strideW;
        dilationH_ = td->dilationH;
        dilationW_ = td->dilationW;
        padTop_ = td->padTop;
        padBottom_ = td->padBottom;
        padLeft_ = td->padLeft;
        padRight_ = td->padRight;
        outH_ = td->outH;
        outW_ = td->outW;
        totalRows_ = td->totalRows;
        tileElements_ = td->tileElements;
        fastGroup_ = td->fastGroup != IM2COL_TILING_FLAG_DISABLED;
        totalGroups_ = td->totalGroups;
        batchRows_ = td->batchRows;
        outRowStrideElements_ = td->outRowStrideElements;
        rawRowStrideElements_ = td->rawRowStrideElements;
        groupBatch_ = td->groupBatch;
        fastChannel_ = td->fastChannel != IM2COL_TILING_FLAG_DISABLED;
        channelIdentity_ = td->channelIdentity != IM2COL_TILING_FLAG_DISABLED;
        channelFlatGather_ = td->channelFlatGather != IM2COL_TILING_FLAG_DISABLED;
        channelContiguousRaw_ = td->channelContiguousRaw != IM2COL_TILING_FLAG_DISABLED;
        totalChannels_ = td->totalChannels;
        channelBatch_ = td->channelBatch;
        rawChannelStrideElements_ = td->rawChannelStrideElements;
        outputChannelElements_ = td->outputChannelElements;
        outputGroupStrideElements_ = td->outputGroupStrideElements;
        outputChannelStrideElements_ = td->outputChannelStrideElements;
        rawInputBaseElements_ = td->rawInputBaseElements;
        indexBufferBytes_ = td->indexBufferBytes;
        channelIndexTemplateElements_ = td->channelIndexTemplateElements;
        channelIndexTemplateValid_ = td->channelIndexTemplateValid != IM2COL_CHANNEL_INDEX_TEMPLATE_NONE;
        channelIndexTemplateInt16_ = td->channelIndexTemplateValid == IM2COL_CHANNEL_INDEX_TEMPLATE_INT16;
        channelIndexTemplateUint8_ = td->channelIndexTemplateValid == IM2COL_CHANNEL_INDEX_TEMPLATE_UINT8;

        const int64_t block = static_cast<int64_t>(GetBlockIdx());
        if (fastChannel_) {
            const int64_t channelsBeforeExtra = block < td->extraChannels ? block : td->extraChannels;
            channelBegin_ = block * td->baseChannelsPerCore + channelsBeforeExtra;
            channelEnd_ = channelBegin_ + td->baseChannelsPerCore +
                          (block < td->extraChannels ? EXTRA_WORK_ITEM_COUNT : 0);
            if (channelEnd_ > totalChannels_) {
                channelEnd_ = totalChannels_;
            }
        } else if (fastGroup_) {
            const int64_t groupsBeforeExtra = block < td->extraGroups ? block : td->extraGroups;
            groupBegin_ = block * td->baseGroupsPerCore + groupsBeforeExtra;
            groupEnd_ = groupBegin_ + td->baseGroupsPerCore + (block < td->extraGroups ? EXTRA_WORK_ITEM_COUNT : 0);
            if (groupEnd_ > totalGroups_) {
                groupEnd_ = totalGroups_;
            }
        } else {
            const int64_t rowsBeforeExtra = block < td->extraRows ? block : td->extraRows;
            rowBegin_ = block * td->baseRowsPerCore + rowsBeforeExtra;
            rowEnd_ = rowBegin_ + td->baseRowsPerCore + (block < td->extraRows ? EXTRA_WORK_ITEM_COUNT : 0);
            if (rowEnd_ > totalRows_) {
                rowEnd_ = totalRows_;
            }
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)x, td->totalInputElements);
        outputGm_.SetGlobalBuffer((__gm__ T*)y, td->totalOutputElements);
        if constexpr (Path == IM2COL_PATH_CHANNEL_TEMPLATE || Path == IM2COL_PATH_GATHER_BOOL) {
            indexTemplateGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(tiling) +
                                                   offsetof(Im2colTilingData, channelIndexTemplate)),
                IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS);
            indexTemplateInt16Gm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int16_t*>(reinterpret_cast<__gm__ uint8_t*>(tiling) +
                                                  offsetof(Im2colTilingData, channelIndexTemplate)),
                IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS);
            indexTemplateUint8Gm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t*>(tiling) + offsetof(Im2colTilingData, channelIndexTemplate),
                IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS);
        }
        if ((fastChannel_ && channelBegin_ >= channelEnd_) ||
            (!fastChannel_ && fastGroup_ && groupBegin_ >= groupEnd_) ||
            (!fastChannel_ && !fastGroup_ && rowBegin_ >= rowEnd_)) {
            return;
        }

        pipe_->InitBuffer(outBuf_, td->outBufferBytes);
        if ((fastChannel_ && !channelIdentity_) || Path != IM2COL_PATH_CONTIGUOUS_W) {
            pipe_->InitBuffer(rawBuf_, td->rawBufferBytes);
            pipe_->InitBuffer(indexBuf_, td->indexBufferBytes);
        }
        if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
            pipe_->InitBuffer(outWideBuf_, td->outWideBufferBytes);
            pipe_->InitBuffer(rawWideBuf_, td->rawWideBufferBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (fastChannel_) {
            ProcessChannels();
            return;
        }
        if (fastGroup_) {
            if constexpr (Path == IM2COL_PATH_CONTIGUOUS_W) {
                if (groupBatch_ > SINGLE_GROUP_BATCH) {
                    for (int64_t group = groupBegin_; group < groupEnd_; group += groupBatch_) {
                        int64_t groupStop = group + groupBatch_;
                        if (groupStop > groupEnd_) {
                            groupStop = groupEnd_;
                        }
                        ProcessContiguousGroupBatch(group, groupStop);
                    }
                    return;
                }
            }
            for (int64_t group = groupBegin_; group < groupEnd_; ++group) {
                ProcessGroup(group);
            }
            return;
        }
        if (rowBegin_ >= rowEnd_ || tileElements_ <= 0) {
            return;
        }
        for (int64_t row = rowBegin_; row < rowEnd_; ++row) {
            ProcessRow(row);
        }
    }

private:
    // AscendC data-movement alignment is expressed in 32-byte data blocks.
    static constexpr int64_t DATA_BLOCK_BYTES = 32;
    // Channel 0 owns the base index vector; later channels only add their GM offset.
    static constexpr int64_t FIRST_OFFSET_CHANNEL = 1;
    static constexpr int64_t TEMPLATE_BASE_KERNEL_ROWS = 2;
    static constexpr int64_t TEMPLATE_LAST_KERNEL_ROW = 4;
    // Compact uint8/int16 templates need extra in-place cast staging regions.
    static constexpr int64_t UINT8_TEMPLATE_BUFFER_REGIONS = 3;
    static constexpr int64_t INT16_TEMPLATE_BUFFER_REGIONS = 2;
    // Two output regions are alternated by the low bit of the kernel group index.
    static constexpr int64_t PING_PONG_BUFFER_MASK = 1;
    static constexpr int64_t EXTRA_WORK_ITEM_COUNT = 1;
    static constexpr int64_t SINGLE_GROUP_BATCH = 1;
    static constexpr uint16_t SINGLE_DATA_COPY_BLOCK_COUNT = 1;
    static constexpr int64_t INVALID_INDEX_GUARD_CHANNEL_COUNT = 1;
    static constexpr int64_t SINGLE_VALID_ELEMENT_COUNT = 1;

    // Host tiling validates stride and shape divisors before launching the kernel.
    // Keep these hot-path helpers branch-free; their divisor precondition is strictly positive.
    __aicore__ inline int64_t CeilDivPositive(int64_t value, int64_t divisor) const
    {
        return value <= 0 ? 0 : (value + divisor - 1) / divisor;
    }

    __aicore__ inline int64_t FloorDivPositiveDivisor(int64_t value, int64_t divisor) const
    {
        const int64_t quotient = value / divisor;
        return value % divisor < 0 ? quotient - 1 : quotient;
    }

    __aicore__ inline void DecodeRow(int64_t row, int64_t& ni, int64_t& ci, int64_t& khi, int64_t& kwi,
                                     int64_t& ohi) const
    {
        ohi = row % outH_;
        row /= outH_;
        kwi = row % kernelW_;
        row /= kernelW_;
        khi = row % kernelH_;
        row /= kernelH_;
        ci = row % c_;
        ni = row / c_;
    }

    __aicore__ inline void SyncMte2ToMte3()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(id);
        WaitFlag<HardEvent::MTE2_MTE3>(id);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(id);
        WaitFlag<HardEvent::V_MTE2>(id);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(id);
        WaitFlag<HardEvent::MTE3_MTE2>(id);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(id);
        WaitFlag<HardEvent::MTE3_V>(id);
    }

    __aicore__ inline void SyncSToV()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(id);
        WaitFlag<HardEvent::S_V>(id);
    }

    __aicore__ inline void SyncVToS()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(id);
        WaitFlag<HardEvent::V_S>(id);
    }

    __aicore__ inline int64_t AlignUpElements(int64_t elements) const
    {
        const int64_t bytes = elements * static_cast<int64_t>(sizeof(T));
        return ((bytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES) * DATA_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    }

    __aicore__ inline void BuildChannelIndex(int64_t activeChannels)
    {
        LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        LocalTensor<int32_t> indexInt = indexLocal.template ReinterpretCast<int32_t>();
        constexpr int64_t gatherElementBytes = Path == IM2COL_PATH_GATHER_BOOL ? static_cast<int64_t>(sizeof(half)) :
                                                                                 static_cast<int64_t>(sizeof(T));
        if constexpr (Path == IM2COL_PATH_CHANNEL_TEMPLATE) {
            if (channelFlatGather_ && !channelContiguousRaw_) {
                Duplicate<int32_t>(indexInt, 0, static_cast<uint32_t>(activeChannels * outputChannelStrideElements_));
                SyncVToMte2();
                DataCopy(indexLocal, indexTemplateGm_, static_cast<uint32_t>(channelIndexTemplateElements_));
                SyncMte2ToV();
                const int64_t outputGroupElements = outH_ * outW_;
                const int64_t firstKernelRowElements = kernelW_ * outputGroupElements;
                const int64_t baseKernelRowsElements = TEMPLATE_BASE_KERNEL_ROWS * firstKernelRowElements;
                const int32_t nextKernelRowsOffsetBytes = static_cast<int32_t>(
                    TEMPLATE_BASE_KERNEL_ROWS * dilationH_ * rawRowStrideElements_ * static_cast<int64_t>(sizeof(T)));
                Adds<int32_t>(indexInt[baseKernelRowsElements], indexInt, nextKernelRowsOffsetBytes,
                              static_cast<uint32_t>(baseKernelRowsElements));
                PipeBarrier<PIPE_V>();
                const int32_t lastKernelRowOffsetBytes = static_cast<int32_t>(
                    TEMPLATE_LAST_KERNEL_ROW * dilationH_ * rawRowStrideElements_ * static_cast<int64_t>(sizeof(T)));
                Adds<int32_t>(indexInt[TEMPLATE_LAST_KERNEL_ROW * firstKernelRowElements], indexInt,
                              lastKernelRowOffsetBytes, static_cast<uint32_t>(firstKernelRowElements));
                PipeBarrier<PIPE_V>();
                for (int64_t channel = FIRST_OFFSET_CHANNEL; channel < activeChannels; ++channel) {
                    const int32_t rawBaseBytes = static_cast<int32_t>(channel * rawChannelStrideElements_ *
                                                                      static_cast<int64_t>(sizeof(T)));
                    Adds<int32_t>(indexInt[channel * outputChannelStrideElements_], indexInt, rawBaseBytes,
                                  static_cast<uint32_t>(outputChannelStrideElements_));
                    PipeBarrier<PIPE_V>();
                }
                return;
            }
        }
        if (!channelContiguousRaw_) {
            Duplicate<int32_t>(indexInt, 0, static_cast<uint32_t>(activeChannels * outputChannelStrideElements_));
            SyncVToS();
            int64_t index = 0;
            for (int64_t oh = 0; oh < outH_; ++oh) {
                const int64_t rowOffset = oh * strideH_ * rawRowStrideElements_;
                for (int64_t ow = 0; ow < outW_; ++ow) {
                    const int64_t elementOffset = rowOffset + ow * strideW_;
                    indexLocal.SetValue(index++, static_cast<uint32_t>(elementOffset * gatherElementBytes));
                }
            }
            SyncSToV();
            for (int64_t channel = FIRST_OFFSET_CHANNEL; channel < activeChannels; ++channel) {
                const int32_t rawBaseBytes = static_cast<int32_t>(channel * rawChannelStrideElements_ *
                                                                  gatherElementBytes);
                Adds<int32_t>(indexInt[channel * outputChannelStrideElements_], indexInt, rawBaseBytes,
                              static_cast<uint32_t>(outputChannelStrideElements_));
                PipeBarrier<PIPE_V>();
            }
            return;
        }

        if constexpr (Path == IM2COL_PATH_CHANNEL_TEMPLATE || Path == IM2COL_PATH_GATHER_BOOL) {
            if (channelIndexTemplateValid_) {
                if (channelIndexTemplateUint8_) {
                    const int64_t indexBufferBytes = indexBufferBytes_;
                    LocalTensor<uint8_t>
                        compactIndex = indexBuf_.Get<uint8_t>()[indexBufferBytes - channelIndexTemplateElements_];
                    LocalTensor<half> compactHalf = indexBuf_.Get<
                        half>()[(indexBufferBytes - UINT8_TEMPLATE_BUFFER_REGIONS * channelIndexTemplateElements_) /
                                static_cast<int64_t>(sizeof(half))];
                    DataCopy(compactIndex, indexTemplateUint8Gm_, static_cast<uint32_t>(channelIndexTemplateElements_));
                    SyncMte2ToV();
                    Cast(compactHalf, compactIndex, RoundMode::CAST_NONE,
                         static_cast<uint32_t>(channelIndexTemplateElements_));
                    PipeBarrier<PIPE_V>();
                    Cast(indexInt, compactHalf, RoundMode::CAST_RINT,
                         static_cast<uint32_t>(channelIndexTemplateElements_));
                    PipeBarrier<PIPE_V>();
                    Muls<int32_t>(indexInt, indexInt, static_cast<int32_t>(gatherElementBytes),
                                  static_cast<uint32_t>(channelIndexTemplateElements_));
                    PipeBarrier<PIPE_V>();
                } else if (channelIndexTemplateInt16_) {
                    const int64_t indexBufferElements16 = indexBufferBytes_ / static_cast<int64_t>(sizeof(int16_t));
                    LocalTensor<int16_t>
                        compactIndex = indexBuf_.Get<int16_t>()[indexBufferElements16 - channelIndexTemplateElements_];
                    LocalTensor<half> compactHalf = indexBuf_.Get<
                        half>()[indexBufferElements16 - INT16_TEMPLATE_BUFFER_REGIONS * channelIndexTemplateElements_];
                    DataCopy(compactIndex, indexTemplateInt16Gm_, static_cast<uint32_t>(channelIndexTemplateElements_));
                    SyncMte2ToV();
                    Cast(compactHalf, compactIndex, RoundMode::CAST_NONE,
                         static_cast<uint32_t>(channelIndexTemplateElements_));
                    PipeBarrier<PIPE_V>();
                    Cast(indexInt, compactHalf, RoundMode::CAST_RINT,
                         static_cast<uint32_t>(channelIndexTemplateElements_));
                    PipeBarrier<PIPE_V>();
                } else {
                    DataCopy(indexLocal, indexTemplateGm_, static_cast<uint32_t>(channelIndexTemplateElements_));
                    SyncMte2ToV();
                }
                if (!channelIndexTemplateUint8_) {
                    if constexpr (Path != IM2COL_PATH_GATHER_BOOL) {
                        for (int64_t channel = FIRST_OFFSET_CHANNEL; channel < activeChannels; ++channel) {
                            const int32_t rawBaseBytes = static_cast<int32_t>(channel * rawChannelStrideElements_ *
                                                                              gatherElementBytes);
                            Adds<int32_t>(indexInt[channel * outputChannelStrideElements_], indexInt, rawBaseBytes,
                                          static_cast<uint32_t>(outputChannelStrideElements_));
                            PipeBarrier<PIPE_V>();
                            Maxs<int32_t>(indexInt[channel * outputChannelStrideElements_],
                                          indexInt[channel * outputChannelStrideElements_], 0,
                                          static_cast<uint32_t>(outputChannelStrideElements_));
                        }
                        Maxs<int32_t>(indexInt, indexInt, 0, static_cast<uint32_t>(outputChannelStrideElements_));
                        PipeBarrier<PIPE_V>();
                    }
                }
                return;
            }
        }
        const int32_t invalidIndex = static_cast<int32_t>(-(activeChannels + INVALID_INDEX_GUARD_CHANNEL_COUNT) * h_ *
                                                          w_ * gatherElementBytes);
        Duplicate<int32_t>(indexInt, invalidIndex, static_cast<uint32_t>(outputChannelStrideElements_));
        SyncVToS();
        int64_t index = 0;
        for (int64_t kh = 0; kh < kernelH_; ++kh) {
            for (int64_t kw = 0; kw < kernelW_; ++kw) {
                for (int64_t oh = 0; oh < outH_; ++oh) {
                    const int64_t inputH = oh * strideH_ + kh * dilationH_ - padTop_;
                    for (int64_t ow = 0; ow < outW_; ++ow) {
                        const int64_t inputW = ow * strideW_ + kw * dilationW_ - padLeft_;
                        if (inputH >= 0 && inputH < h_ && inputW >= 0 && inputW < w_) {
                            const int64_t elementOffset = rawInputBaseElements_ + inputH * w_ + inputW;
                            indexLocal.SetValue(index, static_cast<uint32_t>(elementOffset * gatherElementBytes));
                        }
                        ++index;
                    }
                }
            }
        }
        SyncSToV();
        for (int64_t channel = FIRST_OFFSET_CHANNEL; channel < activeChannels; ++channel) {
            const int32_t rawBaseBytes = static_cast<int32_t>(channel * rawChannelStrideElements_ * gatherElementBytes);
            Adds<int32_t>(indexInt[channel * outputChannelStrideElements_], indexInt, rawBaseBytes,
                          static_cast<uint32_t>(outputChannelStrideElements_));
            PipeBarrier<PIPE_V>();
            Maxs<int32_t>(indexInt[channel * outputChannelStrideElements_],
                          indexInt[channel * outputChannelStrideElements_], 0,
                          static_cast<uint32_t>(outputChannelStrideElements_));
        }
        Maxs<int32_t>(indexInt, indexInt, 0, static_cast<uint32_t>(outputChannelStrideElements_));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessChannelIdentityBatch(int64_t channelStart, int64_t channelStop)
    {
        LocalTensor<T> outLocal = outBuf_.Get<T>();
        const int64_t channels = channelStop - channelStart;
        DataCopyExtParams load{SINGLE_DATA_COPY_BLOCK_COUNT,
                               static_cast<uint32_t>(channels * outputChannelElements_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(outLocal, inputGm_[channelStart * outputChannelElements_], load, noPad);
        SyncMte2ToMte3();
        DataCopyExtParams store{SINGLE_DATA_COPY_BLOCK_COUNT,
                                static_cast<uint32_t>(channels * outputChannelElements_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[channelStart * outputChannelElements_], outLocal, store);
        if (channelStop < channelEnd_) {
            SyncMte3ToMte2();
        }
    }

    __aicore__ inline void ProcessChannelGatherBatch(int64_t channelStart, int64_t channelStop, bool buildIndex)
    {
        LocalTensor<T> outLocal = outBuf_.Get<T>();
        LocalTensor<T> rawLocal = rawBuf_.Get<T>();
        LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        const int64_t channels = channelStop - channelStart;

        if (channelContiguousRaw_) {
            const bool perChannelZeroBase = channelIndexTemplateUint8_ ||
                                            (Path == IM2COL_PATH_GATHER_BOOL && channelIndexTemplateValid_);
            DuplicateZeroNoSync(rawLocal,
                                perChannelZeroBase ? channels * rawChannelStrideElements_ : rawInputBaseElements_);
            SyncVToMte2();
            const uint32_t inputChannelBytes = static_cast<uint32_t>(h_ * w_ * static_cast<int64_t>(sizeof(T)));
            DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
            if (perChannelZeroBase) {
                const uint32_t alignedInputChannelBytes = (inputChannelBytes +
                                                           static_cast<uint32_t>(DATA_BLOCK_BYTES - 1)) &
                                                          ~static_cast<uint32_t>(DATA_BLOCK_BYTES - 1);
                const uint32_t rawChannelBytes = static_cast<uint32_t>(rawChannelStrideElements_ *
                                                                       static_cast<int64_t>(sizeof(T)));
                const uint32_t dstStrideBlocks = (rawChannelBytes - alignedInputChannelBytes) /
                                                 static_cast<uint32_t>(DATA_BLOCK_BYTES);
                DataCopyExtParams load{static_cast<uint16_t>(channels), inputChannelBytes, 0, dstStrideBlocks, 0};
                DataCopyPad(rawLocal[rawInputBaseElements_], inputGm_[channelStart * h_ * w_], load, noPad);
            } else {
                DataCopyExtParams load{SINGLE_DATA_COPY_BLOCK_COUNT,
                                       static_cast<uint32_t>(channels * inputChannelBytes), 0, 0, 0};
                DataCopyPad(rawLocal[rawInputBaseElements_], inputGm_[channelStart * h_ * w_], load, noPad);
            }
            if (buildIndex) {
                BuildChannelIndex(channels);
                if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
                    // The generic BOOL index is built by S/V and does not
                    // issue the template MTE2 copy used by CHANNEL_TEMPLATE.
                    if (!channelIndexTemplateValid_) {
                        SyncMte2ToV();
                    }
                }
            } else {
                SyncMte2ToV();
            }
            if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
                LocalTensor<half> rawWide = rawWideBuf_.Get<half>();
                LocalTensor<half> outWide = outWideBuf_.Get<half>();
                const int64_t rawElements = rawInputBaseElements_ + channels * rawChannelStrideElements_;
                Cast(rawWide, rawLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(rawElements));
                PipeBarrier<PIPE_V>();
                if (channelIndexTemplateValid_) {
                    for (int64_t channel = 0; channel < channels; ++channel) {
                        Gather(outWide[channel * outputChannelStrideElements_],
                               rawWide[channel * rawChannelStrideElements_], indexLocal, 0,
                               static_cast<uint32_t>(outputChannelStrideElements_));
                    }
                } else {
                    Gather(outWide, rawWide, indexLocal, 0,
                           static_cast<uint32_t>(channels * outputChannelStrideElements_));
                }
                PipeBarrier<PIPE_V>();
                Cast(outLocal, outWide, RoundMode::CAST_NONE,
                     static_cast<uint32_t>(channels * outputChannelStrideElements_));
            } else {
                if (channelIndexTemplateUint8_) {
                    for (int64_t channel = 0; channel < channels; ++channel) {
                        Gather(outLocal[channel * outputChannelStrideElements_],
                               rawLocal[channel * rawChannelStrideElements_], indexLocal, 0,
                               static_cast<uint32_t>(outputChannelStrideElements_));
                    }
                } else {
                    Gather(outLocal, rawLocal, indexLocal, 0,
                           static_cast<uint32_t>(channels * outputChannelStrideElements_));
                }
            }
            SyncVToMte3();
            DataCopyExtParams store{static_cast<uint16_t>(channels),
                                    static_cast<uint32_t>(outputChannelElements_ * static_cast<int64_t>(sizeof(T))), 0,
                                    0, 0};
            DataCopyPad(outputGm_[channelStart * outputChannelElements_], outLocal, store);
            if (channelStop < channelEnd_) {
                SyncMte3ToV();
            }
            return;
        }

        DuplicateZeroNoSync(rawLocal, channels * rawChannelStrideElements_);
        SyncVToMte2();
        DataCopyExtParams load{static_cast<uint16_t>(h_), static_cast<uint32_t>(w_ * static_cast<int64_t>(sizeof(T))),
                               0, 0, 0};
        DataCopyPadExtParams<T> pad{true, static_cast<uint8_t>(padLeft_), static_cast<uint8_t>(padRight_),
                                    static_cast<T>(0)};
        for (int64_t channel = 0; channel < channels; ++channel) {
            const int64_t rawStart = channel * rawChannelStrideElements_ + padTop_ * rawRowStrideElements_;
            const int64_t inputStart = (channelStart + channel) * h_ * w_;
            DataCopyPad(rawLocal[rawStart], inputGm_[inputStart], load, pad);
        }
        if (buildIndex) {
            BuildChannelIndex(channels);
        }
        SyncMte2ToV();

        if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
            LocalTensor<half> rawWide = rawWideBuf_.Get<half>();
            const int64_t rawElements = channels * rawChannelStrideElements_;
            Cast(rawWide, rawLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(rawElements));
            PipeBarrier<PIPE_V>();
        }

        if (channelFlatGather_) {
            if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
                LocalTensor<half> rawWide = rawWideBuf_.Get<half>();
                LocalTensor<half> outWide = outWideBuf_.Get<half>();
                Gather(outWide, rawWide, indexLocal, 0, static_cast<uint32_t>(channels * outputChannelStrideElements_));
                PipeBarrier<PIPE_V>();
                Cast(outLocal, outWide, RoundMode::CAST_NONE,
                     static_cast<uint32_t>(channels * outputChannelStrideElements_));
            } else {
                Gather(outLocal, rawLocal, indexLocal, 0,
                       static_cast<uint32_t>(channels * outputChannelStrideElements_));
            }
            SyncVToMte3();
            DataCopyExtParams store{static_cast<uint16_t>(channels),
                                    static_cast<uint32_t>(outputChannelElements_ * static_cast<int64_t>(sizeof(T))), 0,
                                    0, 0};
            DataCopyPad(outputGm_[channelStart * outputChannelElements_], outLocal, store);
            if (channelStop < channelEnd_) {
                SyncMte3ToV();
            }
            return;
        }

        const int64_t outputGroupElements = outH_ * outW_;
        const int64_t kernelArea = kernelH_ * kernelW_;
        for (int64_t kh = 0; kh < kernelH_; ++kh) {
            for (int64_t kw = 0; kw < kernelW_; ++kw) {
                const int64_t group = kh * kernelW_ + kw;
                LocalTensor<T>
                    groupOut = outLocal[(group & PING_PONG_BUFFER_MASK) * channelBatch_ * outputChannelStrideElements_];
                const uint32_t kernelOffsetBytes = static_cast<uint32_t>(
                    (kh * dilationH_ * rawRowStrideElements_ + kw * dilationW_) *
                    (Path == IM2COL_PATH_GATHER_BOOL ? static_cast<int64_t>(sizeof(half)) :
                                                       static_cast<int64_t>(sizeof(T))));
                if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
                    LocalTensor<half> rawWide = rawWideBuf_.Get<half>();
                    LocalTensor<half> wideGroupOut = outWideBuf_.Get<
                        half>()[(group & PING_PONG_BUFFER_MASK) * channelBatch_ * outputChannelStrideElements_];
                    Gather(wideGroupOut, rawWide, indexLocal, kernelOffsetBytes,
                           static_cast<uint32_t>(channels * outputChannelStrideElements_));
                    PipeBarrier<PIPE_V>();
                    Cast(groupOut, wideGroupOut, RoundMode::CAST_NONE,
                         static_cast<uint32_t>(channels * outputChannelStrideElements_));
                } else {
                    Gather(groupOut, rawLocal, indexLocal, kernelOffsetBytes,
                           static_cast<uint32_t>(channels * outputChannelStrideElements_));
                }
                SyncVToMte3();
                const uint32_t dstStrideBytes = static_cast<uint32_t>((outputChannelElements_ - outputGroupElements) *
                                                                      sizeof(T));
                DataCopyExtParams store{static_cast<uint16_t>(channels),
                                        static_cast<uint32_t>(outputGroupElements * sizeof(T)), 0, dstStrideBytes, 0};
                const int64_t outputStart = channelStart * outputChannelElements_ + group * outputGroupElements;
                DataCopyPad(outputGm_[outputStart], groupOut, store);
                const bool hasMoreWork = group + 1 < kernelArea || channelStop < channelEnd_;
                if (hasMoreWork && ((group & PING_PONG_BUFFER_MASK) != 0 || group + 1 == kernelArea)) {
                    SyncMte3ToV();
                }
            }
        }
    }

    __aicore__ inline void ProcessChannels()
    {
        if (channelBegin_ >= channelEnd_ || channelBatch_ <= 0) {
            return;
        }
        bool buildIndex = !channelIdentity_;
        for (int64_t channel = channelBegin_; channel < channelEnd_; channel += channelBatch_) {
            int64_t channelStop = channel + channelBatch_;
            if (channelStop > channelEnd_) {
                channelStop = channelEnd_;
            }
            if (channelIdentity_) {
                ProcessChannelIdentityBatch(channel, channelStop);
            } else {
                ProcessChannelGatherBatch(channel, channelStop, buildIndex);
                buildIndex = false;
            }
        }
    }

    __aicore__ inline void DecodeGroup(int64_t group, int64_t& ni, int64_t& ci, int64_t& khi, int64_t& kwi) const
    {
        kwi = group % kernelW_;
        group /= kernelW_;
        khi = group % kernelH_;
        group /= kernelH_;
        ci = group % c_;
        ni = group / c_;
    }

    __aicore__ inline void DuplicateZeroNoSync(LocalTensor<T>& outLocal, int64_t elements)
    {
        const uint32_t bytes = static_cast<uint32_t>(elements * static_cast<int64_t>(sizeof(T)));
        const uint32_t halfCount = (bytes + sizeof(uint16_t) - 1U) / sizeof(uint16_t);
        LocalTensor<uint16_t> zeroLocal = outLocal.template ReinterpretCast<uint16_t>();
        Duplicate<uint16_t>(zeroLocal, static_cast<uint16_t>(0), halfCount);
    }

    __aicore__ inline void StoreGroupBatch(LocalTensor<T>& outLocal, int64_t outputStart, int64_t rows, bool hasNext)
    {
        DataCopyExtParams store{static_cast<uint16_t>(rows),
                                static_cast<uint32_t>(outW_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGm_[outputStart], outLocal, store);
        if (!hasNext) {
            return;
        }
        if constexpr (Path == IM2COL_PATH_CONTIGUOUS_W) {
            SyncMte3ToMte2();
        } else {
            // The gather path reuses rawBuf_ as the next group's MTE2
            // destination.  Do not let that load overwrite data still being
            // consumed by the previous group's Vector Gather instructions.
            SyncVToMte2();
        }
        SyncMte3ToV();
    }

    __aicore__ inline void ProcessContiguousGroupBatch(int64_t groupStart, int64_t groupStop)
    {
        LocalTensor<T> outLocal = outBuf_.Get<T>();
        const int64_t groups = groupStop - groupStart;
        const int64_t bufferedRows = groups * outH_;
        DuplicateZeroNoSync(outLocal, bufferedRows * outRowStrideElements_);
        SyncVToMte2();

        bool hasValid = false;
        for (int64_t group = groupStart; group < groupStop; ++group) {
            int64_t ni = 0;
            int64_t ci = 0;
            int64_t khi = 0;
            int64_t kwi = 0;
            DecodeGroup(group, ni, ci, khi, kwi);

            int64_t verticalBegin = CeilDivPositive(padTop_ - khi * dilationH_, strideH_);
            int64_t verticalEnd = FloorDivPositiveDivisor(h_ - 1 + padTop_ - khi * dilationH_, strideH_);
            if (verticalBegin < 0) {
                verticalBegin = 0;
            }
            if (verticalEnd >= outH_) {
                verticalEnd = outH_ - 1;
            }
            if (verticalBegin > verticalEnd) {
                continue;
            }

            int64_t horizontalBegin = CeilDivPositive(padLeft_ - kwi * dilationW_, strideW_);
            int64_t horizontalEnd = FloorDivPositiveDivisor(w_ - 1 + padLeft_ - kwi * dilationW_, strideW_);
            if (horizontalBegin < 0) {
                horizontalBegin = 0;
            }
            if (horizontalEnd >= outW_) {
                horizontalEnd = outW_ - 1;
            }
            if (horizontalBegin > horizontalEnd) {
                continue;
            }

            hasValid = true;
            const int64_t validRows = verticalEnd - verticalBegin + 1;
            const int64_t validCount = horizontalEnd - horizontalBegin + 1;
            const int64_t inputH = verticalBegin * strideH_ + khi * dilationH_ - padTop_;
            const int64_t inputW = horizontalBegin + kwi * dilationW_ - padLeft_;
            const int64_t inputStart = ((ni * c_ + ci) * h_ + inputH) * w_ + inputW;
            const int64_t srcStrideElements = strideH_ * w_ - validCount;
            const int64_t localRow = (group - groupStart) * outH_ + verticalBegin;
            DataCopyExtParams load{static_cast<uint16_t>(validRows),
                                   static_cast<uint32_t>(validCount * static_cast<int64_t>(sizeof(T))),
                                   static_cast<uint32_t>(srcStrideElements * static_cast<int64_t>(sizeof(T))), 0, 0};
            DataCopyPadExtParams<T> pad{true, static_cast<uint8_t>(horizontalBegin),
                                        static_cast<uint8_t>(outW_ - horizontalEnd - 1), static_cast<T>(0)};
            DataCopyPad(outLocal[localRow * outRowStrideElements_], inputGm_[inputStart], load, pad);
        }

        if (hasValid) {
            SyncMte2ToMte3();
        } else {
            SyncVToMte3();
        }
        DataCopyExtParams store{static_cast<uint16_t>(bufferedRows),
                                static_cast<uint32_t>(outW_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGm_[groupStart * outH_ * outW_], outLocal, store);
        if (groupStop < groupEnd_) {
            SyncMte3ToMte2();
            SyncMte3ToV();
        }
    }

    __aicore__ inline void ProcessGroupContiguous(int64_t groupOutputStart, int64_t ni, int64_t ci, int64_t khi,
                                                  int64_t kwi, int64_t verticalBegin, int64_t verticalEnd,
                                                  int64_t horizontalBegin, int64_t horizontalEnd, int64_t tileStart,
                                                  int64_t rows, bool hasNext)
    {
        LocalTensor<T> outLocal = outBuf_.Get<T>();
        DuplicateZeroNoSync(outLocal, rows * outRowStrideElements_);

        const int64_t tileEnd = tileStart + rows - 1;
        const int64_t copyRowBegin = verticalBegin > tileStart ? verticalBegin : tileStart;
        const int64_t copyRowEnd = verticalEnd < tileEnd ? verticalEnd : tileEnd;
        const bool hasValid = verticalEnd >= 0 && horizontalEnd >= horizontalBegin && copyRowBegin <= copyRowEnd;
        if (hasValid) {
            SyncVToMte2();
            const int64_t validRows = copyRowEnd - copyRowBegin + 1;
            const int64_t validCount = horizontalEnd - horizontalBegin + 1;
            const int64_t inputH = copyRowBegin * strideH_ + khi * dilationH_ - padTop_;
            const int64_t inputW = horizontalBegin + kwi * dilationW_ - padLeft_;
            const int64_t inputStart = ((ni * c_ + ci) * h_ + inputH) * w_ + inputW;
            const int64_t srcStrideElements = strideH_ * w_ - validCount;
            DataCopyExtParams load{static_cast<uint16_t>(validRows),
                                   static_cast<uint32_t>(validCount * static_cast<int64_t>(sizeof(T))),
                                   static_cast<uint32_t>(srcStrideElements * static_cast<int64_t>(sizeof(T))), 0, 0};
            DataCopyPadExtParams<T> pad{true, static_cast<uint8_t>(horizontalBegin),
                                        static_cast<uint8_t>(outW_ - horizontalEnd - 1), static_cast<T>(0)};
            const int64_t localRow = copyRowBegin - tileStart;
            DataCopyPad(outLocal[localRow * outRowStrideElements_], inputGm_[inputStart], load, pad);
            SyncMte2ToMte3();
        } else {
            SyncVToMte3();
        }
        StoreGroupBatch(outLocal, groupOutputStart + tileStart * outW_, rows, hasNext);
    }

    __aicore__ inline void ProcessGroupGather(int64_t groupOutputStart, int64_t ni, int64_t ci, int64_t khi,
                                              int64_t kwi, int64_t verticalBegin, int64_t verticalEnd,
                                              int64_t horizontalBegin, int64_t horizontalEnd, int64_t tileStart,
                                              int64_t rows, bool hasNext)
    {
        LocalTensor<T> outLocal = outBuf_.Get<T>();
        LocalTensor<T> rawLocal = rawBuf_.Get<T>();
        LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        const int64_t tileEnd = tileStart + rows - 1;
        const int64_t copyRowBegin = verticalBegin > tileStart ? verticalBegin : tileStart;
        const int64_t copyRowEnd = verticalEnd < tileEnd ? verticalEnd : tileEnd;
        const bool hasValid = verticalEnd >= 0 && horizontalEnd >= horizontalBegin && copyRowBegin <= copyRowEnd;
        const int64_t validCount = hasValid ? horizontalEnd - horizontalBegin + 1 : 0;
        const int64_t validRows = hasValid ? copyRowEnd - copyRowBegin + 1 : 0;
        const int64_t rawSpan = hasValid ? (validCount - 1) * strideW_ + 1 : 0;
        const int64_t rawStrideElements = hasValid ? AlignUpElements(rawSpan) : rawRowStrideElements_;
        const bool partialHorizontal = hasValid && (horizontalBegin != 0 || horizontalEnd != outW_ - 1);
        if (partialHorizontal) {
            ProcessGroupGatherPartial(groupOutputStart, ni, ci, khi, kwi, copyRowBegin, horizontalBegin, tileStart,
                                      rows, validRows, validCount, rawSpan, rawStrideElements, hasNext);
            return;
        }

        if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
            LocalTensor<half> outWide = outWideBuf_.Get<half>();
            Duplicate<half>(outWide, static_cast<half>(0), static_cast<uint32_t>(rows * outRowStrideElements_));
            if (hasValid) {
                const int64_t inputH = copyRowBegin * strideH_ + khi * dilationH_ - padTop_;
                const int64_t inputW = horizontalBegin * strideW_ + kwi * dilationW_ - padLeft_;
                const int64_t inputStart = ((ni * c_ + ci) * h_ + inputH) * w_ + inputW;
                const int64_t srcStrideElements = strideH_ * w_ - rawSpan;
                DataCopyExtParams load{
                    static_cast<uint16_t>(validRows), static_cast<uint32_t>(rawSpan * static_cast<int64_t>(sizeof(T))),
                    static_cast<uint32_t>(srcStrideElements * static_cast<int64_t>(sizeof(T))), 0, 0};
                DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(rawLocal, inputGm_[inputStart], load, noPad);
                SyncMte2ToV();

                LocalTensor<half> rawWide = rawWideBuf_.Get<half>();
                Cast(rawWide, rawLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(validRows * rawStrideElements));
                PipeBarrier<PIPE_V>();
                ArithProgression<int32_t>(indexLocal.template ReinterpretCast<int32_t>(), 0,
                                          static_cast<int32_t>(strideW_ * sizeof(half)),
                                          static_cast<int32_t>(validCount));
                PipeBarrier<PIPE_V>();
                const int64_t localRowBegin = copyRowBegin - tileStart;
                for (int64_t row = 0; row < validRows; ++row) {
                    Gather(outWide[(localRowBegin + row) * outRowStrideElements_], rawWide, indexLocal,
                           static_cast<uint32_t>(row * rawStrideElements * sizeof(half)),
                           static_cast<uint32_t>(validCount));
                }
            }
            PipeBarrier<PIPE_V>();
            Cast(outLocal, outWide, RoundMode::CAST_NONE, static_cast<uint32_t>(rows * outRowStrideElements_));
        } else {
            DuplicateZeroNoSync(outLocal, rows * outRowStrideElements_);
            if (hasValid) {
                const int64_t inputH = copyRowBegin * strideH_ + khi * dilationH_ - padTop_;
                const int64_t inputW = horizontalBegin * strideW_ + kwi * dilationW_ - padLeft_;
                const int64_t inputStart = ((ni * c_ + ci) * h_ + inputH) * w_ + inputW;
                const int64_t srcStrideElements = strideH_ * w_ - rawSpan;
                DataCopyExtParams load{
                    static_cast<uint16_t>(validRows), static_cast<uint32_t>(rawSpan * static_cast<int64_t>(sizeof(T))),
                    static_cast<uint32_t>(srcStrideElements * static_cast<int64_t>(sizeof(T))), 0, 0};
                DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(rawLocal, inputGm_[inputStart], load, noPad);
                SyncMte2ToV();
                ArithProgression<int32_t>(indexLocal.template ReinterpretCast<int32_t>(), 0,
                                          static_cast<int32_t>(strideW_ * sizeof(T)), static_cast<int32_t>(validCount));
                PipeBarrier<PIPE_V>();
                const int64_t localRowBegin = copyRowBegin - tileStart;
                for (int64_t row = 0; row < validRows; ++row) {
                    Gather(outLocal[(localRowBegin + row) * outRowStrideElements_], rawLocal, indexLocal,
                           static_cast<uint32_t>(row * rawStrideElements * sizeof(T)),
                           static_cast<uint32_t>(validCount));
                }
            }
        }
        SyncVToMte3();
        StoreGroupBatch(outLocal, groupOutputStart + tileStart * outW_, rows, hasNext);
    }

    __aicore__ inline void ProcessGroupGatherPartial(int64_t groupOutputStart, int64_t ni, int64_t ci, int64_t khi,
                                                     int64_t kwi, int64_t copyRowBegin, int64_t horizontalBegin,
                                                     int64_t tileStart, int64_t rows, int64_t validRows,
                                                     int64_t validCount, int64_t rawSpan, int64_t rawStrideElements,
                                                     bool hasNext)
    {
        LocalTensor<T> outLocal = outBuf_.Get<T>();
        LocalTensor<T> rawLocal = rawBuf_.Get<T>();
        LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        const int64_t validOutStrideElements = AlignUpElements(validCount);
        const int64_t inputH = copyRowBegin * strideH_ + khi * dilationH_ - padTop_;
        const int64_t inputW = horizontalBegin * strideW_ + kwi * dilationW_ - padLeft_;
        const int64_t inputStart = ((ni * c_ + ci) * h_ + inputH) * w_ + inputW;
        const int64_t srcStrideElements = strideH_ * w_ - rawSpan;

        DuplicateZeroNoSync(outLocal, rows * outRowStrideElements_);
        DataCopyExtParams load{static_cast<uint16_t>(validRows),
                               static_cast<uint32_t>(rawSpan * static_cast<int64_t>(sizeof(T))),
                               static_cast<uint32_t>(srcStrideElements * static_cast<int64_t>(sizeof(T))), 0, 0};
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(rawLocal, inputGm_[inputStart], load, noPad);

        // A Gather destination must be 32-byte aligned.  When the valid
        // horizontal range starts inside the output row, first commit the
        // zero rows, then gather packed valid rows at aligned UB addresses
        // and overwrite only their GM segments with one strided MTE3 copy.
        SyncVToMte3();
        DataCopyExtParams zeroStore{static_cast<uint16_t>(rows),
                                    static_cast<uint32_t>(outW_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGm_[groupOutputStart + tileStart * outW_], outLocal, zeroStore);

        if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
            SyncMte2ToV();
            LocalTensor<half> rawWide = rawWideBuf_.Get<half>();
            LocalTensor<half> outWide = outWideBuf_.Get<half>();
            Cast(rawWide, rawLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(validRows * rawStrideElements));
            PipeBarrier<PIPE_V>();
            ArithProgression<int32_t>(indexLocal.template ReinterpretCast<int32_t>(), 0,
                                      static_cast<int32_t>(strideW_ * sizeof(half)), static_cast<int32_t>(validCount));
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < validRows; ++row) {
                Gather(outWide[row * validOutStrideElements], rawWide, indexLocal,
                       static_cast<uint32_t>(row * rawStrideElements * sizeof(half)),
                       static_cast<uint32_t>(validCount));
            }
            PipeBarrier<PIPE_V>();
            SyncMte3ToV();
            Cast(outLocal, outWide, RoundMode::CAST_NONE, static_cast<uint32_t>(validRows * validOutStrideElements));
        } else {
            SyncMte2ToV();
            SyncMte3ToV();
            ArithProgression<int32_t>(indexLocal.template ReinterpretCast<int32_t>(), 0,
                                      static_cast<int32_t>(strideW_ * sizeof(T)), static_cast<int32_t>(validCount));
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < validRows; ++row) {
                Gather(outLocal[row * validOutStrideElements], rawLocal, indexLocal,
                       static_cast<uint32_t>(row * rawStrideElements * sizeof(T)), static_cast<uint32_t>(validCount));
            }
        }

        SyncVToMte3();
        DataCopyExtParams validStore{static_cast<uint16_t>(validRows),
                                     static_cast<uint32_t>(validCount * static_cast<int64_t>(sizeof(T))), 0,
                                     static_cast<uint32_t>((outW_ - validCount) * static_cast<int64_t>(sizeof(T))), 0};
        const int64_t outputStart = groupOutputStart + copyRowBegin * outW_ + horizontalBegin;
        DataCopyPad(outputGm_[outputStart], outLocal, validStore);
        if (hasNext) {
            // rawBuf_ is also reused by the next group's MTE2 load.
            SyncVToMte2();
            SyncMte3ToV();
        }
    }

    __aicore__ inline void ProcessGroup(int64_t group)
    {
        int64_t ni = 0;
        int64_t ci = 0;
        int64_t khi = 0;
        int64_t kwi = 0;
        DecodeGroup(group, ni, ci, khi, kwi);

        int64_t verticalBegin = CeilDivPositive(padTop_ - khi * dilationH_, strideH_);
        int64_t verticalEnd = FloorDivPositiveDivisor(h_ - 1 + padTop_ - khi * dilationH_, strideH_);
        if (verticalBegin < 0) {
            verticalBegin = 0;
        }
        if (verticalEnd >= outH_) {
            verticalEnd = outH_ - 1;
        }
        if (verticalBegin > verticalEnd) {
            verticalEnd = -1;
        }

        int64_t horizontalBegin = CeilDivPositive(padLeft_ - kwi * dilationW_, strideW_);
        int64_t horizontalEnd = FloorDivPositiveDivisor(w_ - 1 + padLeft_ - kwi * dilationW_, strideW_);
        if (horizontalBegin < 0) {
            horizontalBegin = 0;
        }
        if (horizontalEnd >= outW_) {
            horizontalEnd = outW_ - 1;
        }
        if (horizontalBegin > horizontalEnd) {
            horizontalEnd = -1;
        }

        const int64_t groupOutputStart = group * outH_ * outW_;
        for (int64_t tileStart = 0; tileStart < outH_; tileStart += batchRows_) {
            int64_t rows = batchRows_;
            if (rows > outH_ - tileStart) {
                rows = outH_ - tileStart;
            }
            const bool hasNext = tileStart + rows < outH_ || group + 1 < groupEnd_;
            if constexpr (Path == IM2COL_PATH_CONTIGUOUS_W) {
                ProcessGroupContiguous(groupOutputStart, ni, ci, khi, kwi, verticalBegin, verticalEnd, horizontalBegin,
                                       horizontalEnd, tileStart, rows, hasNext);
            } else {
                ProcessGroupGather(groupOutputStart, ni, ci, khi, kwi, verticalBegin, verticalEnd, horizontalBegin,
                                   horizontalEnd, tileStart, rows, hasNext);
            }
        }
    }

    __aicore__ inline void ZeroTile(LocalTensor<T>& outLocal, int64_t elements)
    {
        const uint32_t bytes = static_cast<uint32_t>(elements * static_cast<int64_t>(sizeof(T)));
        const uint32_t halfCount = (bytes + sizeof(uint16_t) - 1U) / sizeof(uint16_t);
        LocalTensor<uint16_t> zeroLocal = outLocal.template ReinterpretCast<uint16_t>();
        Duplicate<uint16_t>(zeroLocal, static_cast<uint16_t>(0), halfCount);
        SyncVToMte3();
    }

    __aicore__ inline void CopyValidContiguous(LocalTensor<T>& outLocal, int64_t localStart, int64_t inputStart,
                                               int64_t validCount)
    {
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams params{SINGLE_DATA_COPY_BLOCK_COUNT,
                                 static_cast<uint32_t>(validCount * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outLocal[localStart], inputGm_[inputStart], params, pad);
    }

    __aicore__ inline void GatherValid(LocalTensor<T>& outLocal, int64_t localStart, int64_t inputStart,
                                       int64_t validCount)
    {
        if (validCount == SINGLE_VALID_ELEMENT_COUNT) {
            CopyValidContiguous(outLocal, localStart, inputStart, validCount);
            SyncMte2ToMte3();
            return;
        }

        LocalTensor<T> rawLocal = rawBuf_.Get<T>();
        LocalTensor<uint32_t> indexLocal = indexBuf_.Get<uint32_t>();
        const int64_t rawElements = (validCount - 1) * strideW_ + 1;
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams load{SINGLE_DATA_COPY_BLOCK_COUNT,
                               static_cast<uint32_t>(rawElements * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(rawLocal, inputGm_[inputStart], load, pad);
        SyncMte2ToV();

        if constexpr (Path == IM2COL_PATH_GATHER_BOOL) {
            LocalTensor<half> rawWide = rawWideBuf_.Get<half>();
            LocalTensor<half> outWide = outWideBuf_.Get<half>();
            Cast(rawWide, rawLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(rawElements));
            PipeBarrier<PIPE_V>();
            ArithProgression<int32_t>(indexLocal.template ReinterpretCast<int32_t>(), 0,
                                      static_cast<int32_t>(strideW_ * sizeof(half)), static_cast<int32_t>(validCount));
            PipeBarrier<PIPE_V>();
            Gather(outWide, rawWide, indexLocal, 0, static_cast<uint32_t>(validCount));
            PipeBarrier<PIPE_V>();
            Cast(outLocal[localStart], outWide, RoundMode::CAST_NONE, static_cast<uint32_t>(validCount));
        } else {
            ArithProgression<int32_t>(indexLocal.template ReinterpretCast<int32_t>(), 0,
                                      static_cast<int32_t>(strideW_ * sizeof(T)), static_cast<int32_t>(validCount));
            PipeBarrier<PIPE_V>();
            Gather(outLocal[localStart], rawLocal, indexLocal, 0, static_cast<uint32_t>(validCount));
        }
        SyncVToMte3();
    }

    __aicore__ inline void StoreTile(LocalTensor<T>& outLocal, int64_t outputStart, int64_t elements)
    {
        DataCopyExtParams store{SINGLE_DATA_COPY_BLOCK_COUNT,
                                static_cast<uint32_t>(elements * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGm_[outputStart], outLocal, store);
        // The next tile can be produced by either MTE2 or Vector. Make the
        // single reused output buffer visible to both pipelines.
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void ProcessRow(int64_t row)
    {
        int64_t ni = 0;
        int64_t ci = 0;
        int64_t khi = 0;
        int64_t kwi = 0;
        int64_t ohi = 0;
        DecodeRow(row, ni, ci, khi, kwi, ohi);

        const int64_t inputH = ohi * strideH_ + khi * dilationH_ - padTop_;
        int64_t validBegin = 0;
        int64_t validEnd = -1;
        if (inputH >= 0 && inputH < h_) {
            validBegin = CeilDivPositive(padLeft_ - kwi * dilationW_, strideW_);
            validEnd = FloorDivPositiveDivisor(w_ - 1 + padLeft_ - kwi * dilationW_, strideW_);
            if (validBegin < 0) {
                validBegin = 0;
            }
            if (validEnd >= outW_) {
                validEnd = outW_ - 1;
            }
            if (validBegin > validEnd) {
                validEnd = -1;
            }
        }

        const int64_t rowOutputStart = row * outW_;
        LocalTensor<T> outLocal = outBuf_.Get<T>();
        for (int64_t tileStart = 0; tileStart < outW_; tileStart += tileElements_) {
            int64_t tileCount = tileElements_;
            if (tileCount > outW_ - tileStart) {
                tileCount = outW_ - tileStart;
            }
            int64_t copyBegin = validBegin > tileStart ? validBegin : tileStart;
            int64_t tileLast = tileStart + tileCount - 1;
            int64_t copyEnd = validEnd < tileLast ? validEnd : tileLast;
            const bool hasValid = validEnd >= 0 && copyBegin <= copyEnd;
            const bool fullValid = hasValid && copyBegin == tileStart && copyEnd == tileLast;
            // Every local address supplied to MTE/Vector must remain
            // block-aligned on DAV_2201.  For a partially valid tile, store
            // the zero tile first and then overwrite the valid GM segment
            // from the aligned start of outLocal.
            if (!fullValid) {
                ZeroTile(outLocal, tileCount);
                StoreTile(outLocal, rowOutputStart + tileStart, tileCount);
            }
            if (hasValid) {
                const int64_t inputW = copyBegin * strideW_ + kwi * dilationW_ - padLeft_;
                const int64_t inputStart = ((ni * c_ + ci) * h_ + inputH) * w_ + inputW;
                const int64_t validCount = copyEnd - copyBegin + 1;
                if constexpr (Path == IM2COL_PATH_CONTIGUOUS_W) {
                    CopyValidContiguous(outLocal, 0, inputStart, validCount);
                    SyncMte2ToMte3();
                } else {
                    GatherValid(outLocal, 0, inputStart, validCount);
                }
                const int64_t outputStart = fullValid ? rowOutputStart + tileStart : rowOutputStart + copyBegin;
                StoreTile(outLocal, outputStart, validCount);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> outWideBuf_;
    TBuf<TPosition::VECCALC> rawWideBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<uint32_t> indexTemplateGm_;
    GlobalTensor<int16_t> indexTemplateInt16Gm_;
    GlobalTensor<uint8_t> indexTemplateUint8Gm_;

    int64_t n_ = 0;
    int64_t c_ = 0;
    int64_t h_ = 0;
    int64_t w_ = 0;
    int64_t kernelH_ = 0;
    int64_t kernelW_ = 0;
    int64_t strideH_ = 0;
    int64_t strideW_ = 0;
    int64_t dilationH_ = 0;
    int64_t dilationW_ = 0;
    int64_t padTop_ = 0;
    int64_t padBottom_ = 0;
    int64_t padLeft_ = 0;
    int64_t padRight_ = 0;
    int64_t outH_ = 0;
    int64_t outW_ = 0;
    int64_t totalRows_ = 0;
    int64_t totalGroups_ = 0;
    int64_t tileElements_ = 0;
    int64_t batchRows_ = 0;
    int64_t outRowStrideElements_ = 0;
    int64_t rawRowStrideElements_ = 0;
    int64_t groupBatch_ = SINGLE_GROUP_BATCH;
    int64_t totalChannels_ = 0;
    int64_t channelBatch_ = 0;
    int64_t rawChannelStrideElements_ = 0;
    int64_t outputChannelElements_ = 0;
    int64_t outputGroupStrideElements_ = 0;
    int64_t outputChannelStrideElements_ = 0;
    int64_t rowBegin_ = 0;
    int64_t rowEnd_ = 0;
    int64_t groupBegin_ = 0;
    int64_t groupEnd_ = 0;
    int64_t channelBegin_ = 0;
    int64_t channelEnd_ = 0;
    bool fastGroup_ = false;
    bool fastChannel_ = false;
    bool channelIdentity_ = false;
    bool channelFlatGather_ = false;
    bool channelContiguousRaw_ = false;
    int64_t rawInputBaseElements_ = 0;
    int64_t indexBufferBytes_ = 0;
    int64_t channelIndexTemplateElements_ = 0;
    bool channelIndexTemplateValid_ = false;
    bool channelIndexTemplateInt16_ = false;
    bool channelIndexTemplateUint8_ = false;
};

} // namespace NsIm2col

#endif // EXPERIMENTAL_IM2COL_KERNEL_IMPL_H_
