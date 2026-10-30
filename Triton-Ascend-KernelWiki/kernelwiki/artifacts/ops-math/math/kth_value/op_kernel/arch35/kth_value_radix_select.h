/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_RADIX_SELECT_H
#define KTH_VALUE_RADIX_SELECT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kth_value_tiling_data.h"
#include "../../sort/arch35/common/radix_sort_simd_utils.h"

// Radix select: narrows the kth element by histogramming one byte at a time (MSB to LSB),
// selecting one bucket per round instead of fully sorting. Supports multi-core per row.
namespace KthValue {
using namespace AscendC;
using namespace RadixSortCommon;

constexpr uint32_t RADIX_SELECT_FIND_THREADS = 128U;
constexpr uint32_t RADIX_SELECT_RESULT_WORDS = 8U;
constexpr uint32_t RADIX_SELECT_ACTIVE_INDEX_CAP = 4096U;
constexpr uint32_t RADIX_SELECT_ACTIVE_MODE_THRESHOLD = 512U;
constexpr uint32_t RADIX_SELECT_STATE_LEFT_K_IDX = 2U;
constexpr uint32_t RADIX_SELECT_STATE_SELECTED_IDX = 3U;
constexpr uint32_t RADIX_SELECT_STATE_SELECTED_COUNT_IDX = 4U;
constexpr uint32_t RADIX_SELECT_STATE_TARGET_CORE_IDX = 5U;
constexpr uint32_t RADIX_SELECT_STATE_TARGET_LEFT_K_IDX = 6U;

template <typename UT>
__simt_callee__ __aicore__ inline void CountPrefixMatchesInThread(__ubuf__ UT* keys, uint32_t begin, uint32_t end,
                                                                  UT prefixMask, UT prefixKey,
                                                                  __ubuf__ uint32_t* threadCounts)
{
    uint32_t localCount = 0;
    for (uint32_t i = begin; i < end; ++i) {
        localCount += ((keys[i] & prefixMask) == prefixKey) ? 1U : 0U;
    }
    threadCounts[threadIdx.x] = localCount;
}

template <typename UT>
__simt_callee__ __aicore__ inline int64_t LocatePrefixTarget(__ubuf__ UT* keys, uint32_t begin, uint32_t end,
                                                             UT prefixMask, UT prefixKey, uint64_t localTarget)
{
    uint64_t seen = 0;
    for (uint32_t i = begin; i < end; ++i) {
        if ((keys[i] & prefixMask) != prefixKey) {
            continue;
        }
        if (seen == localTarget) {
            return static_cast<int64_t>(i);
        }
        ++seen;
    }
    return -1;
}

__aicore__ inline void StoreRadixByteHistogram(__local_mem__ uint16_t* histogramPtr, Reg::RegTensor<uint16_t>& hist0,
                                               Reg::RegTensor<uint16_t>& hist1, Reg::MaskReg maskB16)
{
    Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramPtr, hist0, VF_LEN_B16, maskB16);
    Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramPtr, hist1, VF_LEN_B16, maskB16);
}

__aicore__ inline void AddSelectedByteHistogram(Reg::RegTensor<uint16_t>& hist0, Reg::RegTensor<uint16_t>& hist1,
                                                Reg::RegTensor<uint8_t>& bytes, Reg::RegTensor<uint8_t>& flagBytes,
                                                Reg::MaskReg histMask)
{
    Reg::MaskReg selectedMask;
    Reg::CompareScalar<uint8_t, CMPMODE::EQ>(selectedMask, flagBytes, 1, histMask);
    Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(hist0, bytes,
                                                                                                     selectedMask);
    Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(hist1, bytes,
                                                                                                     selectedMask);
}

template <typename VT>
__aicore__ inline void SelectPrefixFlag(Reg::RegTensor<VT>& flag, Reg::RegTensor<VT>& input, Reg::RegTensor<VT>& masked,
                                        Reg::RegTensor<VT>& prefixMaskReg, VT prefixKey, Reg::RegTensor<VT>& one,
                                        Reg::RegTensor<VT>& zero, Reg::MaskReg mask)
{
    Reg::MaskReg selected;
    Reg::And(masked, input, prefixMaskReg, mask);
    Reg::CompareScalar<VT, CMPMODE::EQ>(selected, masked, prefixKey, mask);
    Reg::Select(flag, one, zero, selected);
}

template <typename DstT, typename SrcT>
__aicore__ inline void DeInterleaveRadixBytes(Reg::RegTensor<DstT>& dst0, Reg::RegTensor<DstT>& dst1,
                                              Reg::RegTensor<SrcT>& src0, Reg::RegTensor<SrcT>& src1)
{
    Reg::DeInterleave(dst0, dst1, (Reg::RegTensor<DstT>&)src0, (Reg::RegTensor<DstT>&)src1);
}

__aicore__ inline void PackB32RadixBytes(Reg::RegTensor<uint8_t>& bytes, Reg::RegTensor<uint8_t>& highBytes,
                                         Reg::RegTensor<uint32_t>& shifted0, Reg::RegTensor<uint32_t>& shifted1,
                                         Reg::RegTensor<uint32_t>& shifted2, Reg::RegTensor<uint32_t>& shifted3)
{
    Reg::RegTensor<uint16_t> data16_0, data16_1, data16_2, data16_3;
    DeInterleaveRadixBytes(data16_0, data16_1, shifted0, shifted1);
    DeInterleaveRadixBytes(data16_2, data16_3, shifted2, shifted3);
    DeInterleaveRadixBytes(bytes, highBytes, data16_0, data16_2);
}

__aicore__ inline void PackB32FlagBytes(Reg::RegTensor<uint8_t>& flagBytes, Reg::RegTensor<uint8_t>& highFlags,
                                        Reg::RegTensor<uint32_t>& flag0, Reg::RegTensor<uint32_t>& flag1,
                                        Reg::RegTensor<uint32_t>& flag2, Reg::RegTensor<uint32_t>& flag3)
{
    Reg::RegTensor<uint16_t> flag16_0, flag16_1, flag16_2, flag16_3;
    DeInterleaveRadixBytes(flag16_0, flag16_1, flag0, flag1);
    DeInterleaveRadixBytes(flag16_2, flag16_3, flag2, flag3);
    DeInterleaveRadixBytes(flagBytes, highFlags, flag16_0, flag16_2);
}

__aicore__ inline void PackB64RadixBytes(Reg::RegTensor<uint8_t>& bytes, Reg::RegTensor<uint8_t>& highBytes,
                                         Reg::RegTensor<uint64_t>& shifted0, Reg::RegTensor<uint64_t>& shifted1,
                                         Reg::RegTensor<uint64_t>& shifted2, Reg::RegTensor<uint64_t>& shifted3,
                                         Reg::RegTensor<uint64_t>& shifted4, Reg::RegTensor<uint64_t>& shifted5,
                                         Reg::RegTensor<uint64_t>& shifted6, Reg::RegTensor<uint64_t>& shifted7)
{
    Reg::RegTensor<uint32_t> data32_0, data32_1, data32_2, data32_3;
    Reg::RegTensor<uint32_t> data32_4, data32_5, data32_6, data32_7;
    DeInterleaveRadixBytes(data32_0, data32_1, shifted0, shifted1);
    DeInterleaveRadixBytes(data32_2, data32_3, shifted2, shifted3);
    DeInterleaveRadixBytes(data32_4, data32_5, shifted4, shifted5);
    DeInterleaveRadixBytes(data32_6, data32_7, shifted6, shifted7);
    PackB32RadixBytes(bytes, highBytes, data32_0, data32_2, data32_4, data32_6);
}

__aicore__ inline void PackB64FlagBytes(Reg::RegTensor<uint8_t>& flagBytes, Reg::RegTensor<uint8_t>& highFlags,
                                        Reg::RegTensor<uint64_t>& flag0, Reg::RegTensor<uint64_t>& flag1,
                                        Reg::RegTensor<uint64_t>& flag2, Reg::RegTensor<uint64_t>& flag3,
                                        Reg::RegTensor<uint64_t>& flag4, Reg::RegTensor<uint64_t>& flag5,
                                        Reg::RegTensor<uint64_t>& flag6, Reg::RegTensor<uint64_t>& flag7)
{
    Reg::RegTensor<uint32_t> flag32_0, flag32_1, flag32_2, flag32_3;
    Reg::RegTensor<uint32_t> flag32_4, flag32_5, flag32_6, flag32_7;
    DeInterleaveRadixBytes(flag32_0, flag32_1, flag0, flag1);
    DeInterleaveRadixBytes(flag32_2, flag32_3, flag2, flag3);
    DeInterleaveRadixBytes(flag32_4, flag32_5, flag4, flag5);
    DeInterleaveRadixBytes(flag32_6, flag32_7, flag6, flag7);
    PackB32FlagBytes(flagBytes, highFlags, flag32_0, flag32_2, flag32_4, flag32_6);
}

// SIMT: each of 128 threads counts prefix matches in its slice, then thread 0 does
// prefix-sum to locate which thread owns the target, and searches within that slice.
template <typename UT>
__simt_vf__ LAUNCH_BOUND(RADIX_SELECT_FIND_THREADS) __aicore__
    void FindKthMatchInTile(__ubuf__ UT* keys, uint32_t count, UT prefixMask, UT prefixKey, uint64_t target,
                            __ubuf__ uint32_t* threadCounts, __ubuf__ int64_t* result)
{
    uint32_t tid = threadIdx.x;
    uint32_t elemsPerThread = (count + RADIX_SELECT_FIND_THREADS - 1U) / RADIX_SELECT_FIND_THREADS;
    uint32_t begin = tid * elemsPerThread;
    uint32_t end = begin + elemsPerThread;
    end = end < count ? end : count;
    CountPrefixMatchesInThread(keys, begin, end, prefixMask, prefixKey, threadCounts);
    asc_syncthreads();
    if (tid != 0U) {
        return;
    }
    uint64_t accumulated = 0;
    int64_t found = -1;
    for (uint32_t thread = 0; thread < RADIX_SELECT_FIND_THREADS; ++thread) {
        uint64_t next = accumulated + threadCounts[thread];
        if (target < next) {
            uint64_t localTarget = target - accumulated;
            uint32_t targetBegin = thread * elemsPerThread;
            uint32_t targetEnd = targetBegin + elemsPerThread;
            targetEnd = targetEnd < count ? targetEnd : count;
            found = LocatePrefixTarget(keys, targetBegin, targetEnd, prefixMask, prefixKey, localTarget);
            break;
        }
        accumulated = next;
    }
    uint64_t total = 0;
    for (uint32_t thread = 0; thread < RADIX_SELECT_FIND_THREADS; ++thread) {
        total += threadCounts[thread];
    }
    result[0] = found;
    result[1] = static_cast<int64_t>(total);
}

// SIMT: collect all prefix-matching indices into activeIndices via exclusive prefix-sum offsets.
// Aborts if total exceeds ACTIVE_INDEX_CAP (4096); result[1] signals success.
template <typename UT>
__simt_vf__ LAUNCH_BOUND(RADIX_SELECT_FIND_THREADS) __aicore__
    void CollectActiveIndices(__ubuf__ UT* keys, uint32_t count, UT prefixMask, UT prefixKey,
                              __ubuf__ uint32_t* threadCounts, __ubuf__ uint32_t* activeIndices,
                              __ubuf__ int64_t* result)
{
    uint32_t tid = threadIdx.x;
    uint32_t elemsPerThread = (count + RADIX_SELECT_FIND_THREADS - 1U) / RADIX_SELECT_FIND_THREADS;
    uint32_t begin = tid * elemsPerThread;
    uint32_t end = begin + elemsPerThread;
    end = end < count ? end : count;
    CountPrefixMatchesInThread(keys, begin, end, prefixMask, prefixKey, threadCounts);
    asc_syncthreads();
    if (tid == 0U) {
        uint32_t accumulated = 0;
        for (uint32_t thread = 0; thread < RADIX_SELECT_FIND_THREADS; ++thread) {
            uint32_t countInThread = threadCounts[thread];
            threadCounts[thread] = accumulated;
            accumulated += countInThread;
        }
        result[0] = static_cast<int64_t>(accumulated);
        result[1] = accumulated <= RADIX_SELECT_ACTIVE_INDEX_CAP ? 1 : 0;
    }
    asc_syncthreads();
    if (result[1] == 0) {
        return;
    }
    uint32_t writePos = threadCounts[tid];
    for (uint32_t i = begin; i < end; ++i) {
        if ((keys[i] & prefixMask) == prefixKey) {
            activeIndices[writePos] = i;
            ++writePos;
        }
    }
}

// Active-mode histogram: single-threaded scan over activeIndices only (count <= 512).
template <typename UT>
__simt_vf__ LAUNCH_BOUND(RADIX_SELECT_FIND_THREADS) __aicore__
    void BuildActiveHistogram(__ubuf__ UT* keys, __ubuf__ uint32_t* activeIndices, uint32_t activeCount, uint32_t shift,
                              __ubuf__ uint64_t* histogram)
{
    if (threadIdx.x != 0U) {
        return;
    }
    for (uint32_t bucket = 0; bucket < 256U; ++bucket) {
        histogram[bucket] = 0UL;
    }
    for (uint32_t i = 0; i < activeCount; ++i) {
        UT key = keys[activeIndices[i]];
        uint32_t bucket = static_cast<uint32_t>((key >> shift) & static_cast<UT>(0xFFU));
        histogram[bucket] += 1UL;
    }
}

// Active-mode compaction: filter activeIndices in-place by the expanded prefix.
template <typename UT>
__simt_vf__ LAUNCH_BOUND(RADIX_SELECT_FIND_THREADS) __aicore__
    void CompactActiveIndices(__ubuf__ UT* keys, __ubuf__ uint32_t* activeIndices, uint32_t activeCount, UT prefixMask,
                              UT prefixKey, __ubuf__ int64_t* result)
{
    if (threadIdx.x != 0U) {
        return;
    }
    uint32_t writePos = 0;
    for (uint32_t i = 0; i < activeCount; ++i) {
        uint32_t index = activeIndices[i];
        if ((keys[index] & prefixMask) == prefixKey) {
            activeIndices[writePos] = index;
            ++writePos;
        }
    }
    result[0] = static_cast<int64_t>(writePos);
}

// Selects one radix bucket per byte instead of scattering a fully sorted row.
// Cores are grouped by row. Each core builds the histogram for a contiguous axis
// slice; the leader reduces the group histogram and broadcasts the selected prefix.
// Optimizations: retained tile (avoid re-loading when slice fits in one tile),
// active mode (track matching indices when count <= 512 to skip full scans).
template <typename T, typename UT>
class KthValueRadixSelect {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const KthValueTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline LocalTensor<UT> MakeSortKeys(LocalTensor<T>& input, uint32_t count);
    __aicore__ inline void LoadTile(int64_t offset, uint32_t count, LocalTensor<T>& input);
    __aicore__ inline void CountByte(int64_t rowOffset, int64_t sliceStart, int64_t sliceCount, uint32_t shift,
                                     UT prefixMask, UT prefixKey, LocalTensor<uint64_t>& histogram);
    __aicore__ inline void ClearHistogram(LocalTensor<uint64_t>& histogram);
    __aicore__ inline void AccumulateHistogram(LocalTensor<uint64_t>& histogram, LocalTensor<uint64_t>& src);
    __aicore__ inline void AccumulateTileHistogram(LocalTensor<uint64_t>& histogram,
                                                   LocalTensor<uint16_t>& tileHistogram);
    __aicore__ inline void BuildTileHistogram(LocalTensor<UT>& keys, uint32_t count, uint32_t shift, UT prefixMask,
                                              UT prefixKey, LocalTensor<uint16_t>& tileHistogram);
    __aicore__ inline void BuildTileHistogramB8(LocalTensor<UT>& keys, uint32_t count, UT prefixMask, UT prefixKey,
                                                LocalTensor<uint16_t>& tileHistogram);
    __aicore__ inline void BuildTileHistogramB16(LocalTensor<UT>& keys, uint32_t count, uint32_t shift, UT prefixMask,
                                                 UT prefixKey, LocalTensor<uint16_t>& tileHistogram);
    __aicore__ inline void BuildTileHistogramB32(LocalTensor<UT>& keys, uint32_t count, uint32_t shift, UT prefixMask,
                                                 UT prefixKey, LocalTensor<uint16_t>& tileHistogram);
    __aicore__ inline void BuildTileHistogramB64(LocalTensor<UT>& keys, uint32_t count, uint32_t shift, UT prefixMask,
                                                 UT prefixKey, LocalTensor<uint16_t>& tileHistogram);
    __aicore__ inline void WriteOutput(int64_t row, T value, int64_t index);
    __aicore__ inline void SelectOneRow(int64_t row, uint32_t group, uint32_t coreInGroup, bool writeOutput);
    __aicore__ inline void UpdateActiveIndices(LocalTensor<UT>& retainedKeys, LocalTensor<uint32_t>& activeIndices,
                                               uint32_t retainedCount, UT prefixMask, UT prefixKey, bool& activeMode,
                                               uint32_t& activeCount);
    __aicore__ inline void StoreCoreHistogram(LocalTensor<uint64_t>& histogram);
    __aicore__ inline uint64_t LoadCoreHistogramBucket(uint32_t group, uint32_t coreInGroup, uint32_t bucket,
                                                       LocalTensor<uint64_t>& scratch);
    __aicore__ inline void ReduceGroupHistogram(uint32_t group, LocalTensor<uint64_t>& histogram,
                                                LocalTensor<uint64_t>& scratch);
    __aicore__ inline void StoreGroupState(uint32_t group, LocalTensor<uint64_t>& state);
    __aicore__ inline void LoadGroupState(uint32_t group, LocalTensor<uint64_t>& state);

    static constexpr uint32_t RADIX = 256U;
    static constexpr uint32_t BYTE_BITS = 8U;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> valuesGm_;
    GlobalTensor<int64_t> indicesGm_;
    GlobalTensor<uint64_t> coreHistogramGm_;
    GlobalTensor<uint64_t> groupStateGm_;
    TPipe* pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TBuf<TPosition::VECCALC> keyBuf_;
    TBuf<TPosition::VECCALC> histogramBuf_;
    TBuf<TPosition::VECCALC> tileHistogramBuf_;
    TBuf<TPosition::VECCALC> outputValueBuf_;
    TBuf<TPosition::VECCALC> outputIndexBuf_;
    TBuf<TPosition::VECCALC> findCountBuf_;
    TBuf<TPosition::VECCALC> findResultBuf_;
    TBuf<TPosition::VECCALC> groupReduceBuf_;
    TBuf<TPosition::VECCALC> activeIndexBuf_;

    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};
    uint32_t tileElems_{0};
    uint32_t rowsParallel_{0};
    uint32_t coresPerRow_{0};
    uint32_t sortLoopTimes_{0};
    int64_t kthIndex_{0};
    int64_t axisLen_{0};
    int64_t rowCount_{0};
};

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                                        const KthValueTilingData* tiling, TPipe* pipe)
{
    if (tiling == nullptr || pipe == nullptr) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    pipe_ = pipe;
    tileElems_ = tiling->numTileDataSize;
    rowsParallel_ = tiling->unsortedDimParallel;
    coresPerRow_ = tiling->lastDimNeedCore;
    sortLoopTimes_ = tiling->sortLoopTimes;
    kthIndex_ = tiling->kthIndex;
    axisLen_ = tiling->lastAxisNum;
    rowCount_ = tiling->unsortedDimNum;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));
    uint64_t histogramWords = static_cast<uint64_t>(blockNum_) * RADIX;
    coreHistogramGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(workspace), histogramWords);
    groupStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(workspace) + histogramWords,
                                  static_cast<uint64_t>(rowsParallel_) * RADIX_SELECT_RESULT_WORDS);
    pipe_->InitBuffer(inputQueue_, 1, tileElems_ * sizeof(T));
    pipe_->InitBuffer(keyBuf_, tileElems_ * sizeof(UT));
    pipe_->InitBuffer(histogramBuf_, RADIX * sizeof(uint64_t));
    pipe_->InitBuffer(tileHistogramBuf_, RADIX * sizeof(uint16_t));
    pipe_->InitBuffer(outputValueBuf_, Ops::Base::GetUbBlockSize());
    pipe_->InitBuffer(outputIndexBuf_, Ops::Base::GetUbBlockSize());
    pipe_->InitBuffer(findCountBuf_, RADIX_SELECT_FIND_THREADS * sizeof(uint32_t));
    pipe_->InitBuffer(findResultBuf_, RADIX_SELECT_RESULT_WORDS * sizeof(uint64_t));
    pipe_->InitBuffer(groupReduceBuf_, RADIX * sizeof(uint64_t));
    pipe_->InitBuffer(activeIndexBuf_, RADIX_SELECT_ACTIVE_INDEX_CAP * sizeof(uint32_t));
}

template <typename T, typename UT>
__aicore__ inline LocalTensor<UT> KthValueRadixSelect<T, UT>::MakeSortKeys(LocalTensor<T>& input, uint32_t count)
{
    LocalTensor<UT> keys = keyBuf_.Get<UT>();
    if constexpr (IsSameType<int8_t, T>::value) {
        TwiddleInB8<T, UT, 0>(input, keys, count);
    } else if constexpr (IsSameType<int16_t, T>::value) {
        TwiddleInB16<T, UT, 0>(input, keys, count);
    } else if constexpr (IsSameType<int32_t, T>::value) {
        TwiddleInB32<T, UT, 0>(input, keys, count);
    } else if constexpr (IsSameType<int64_t, T>::value) {
        TwiddleInB64<T, UT, 0>(input, keys, count);
    } else if constexpr (IsSameType<half, T>::value || IsSameType<bfloat16_t, T>::value) {
        TwiddleInFp16<T, UT, 0>(input, keys, count);
    } else if constexpr (IsSameType<float, T>::value) {
        TwiddleInFp32<T, UT, 0>(input, keys, count);
    } else {
        // Ascending unsigned integers already have lexicographically sortable bit patterns.
        keys = input.template ReinterpretCast<UT>();
    }
    return keys;
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::LoadTile(int64_t offset, uint32_t count, LocalTensor<T>& input)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(input, xGm_[offset], copyParams, padParams);
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::BuildTileHistogram(LocalTensor<UT>& keys, uint32_t count,
                                                                      uint32_t shift, UT prefixMask, UT prefixKey,
                                                                      LocalTensor<uint16_t>& tileHistogram)
{
    if constexpr (sizeof(UT) == sizeof(uint8_t)) {
        BuildTileHistogramB8(keys, count, prefixMask, prefixKey, tileHistogram);
    } else if constexpr (sizeof(UT) == sizeof(uint16_t)) {
        BuildTileHistogramB16(keys, count, shift, prefixMask, prefixKey, tileHistogram);
    } else if constexpr (sizeof(UT) == sizeof(uint32_t)) {
        BuildTileHistogramB32(keys, count, shift, prefixMask, prefixKey, tileHistogram);
    } else {
        BuildTileHistogramB64(keys, count, shift, prefixMask, prefixKey, tileHistogram);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::BuildTileHistogramB8(LocalTensor<UT>& keys, uint32_t count,
                                                                        UT prefixMask, UT prefixKey,
                                                                        LocalTensor<uint16_t>& tileHistogram)
{
    __local_mem__ UT* keyPtr = (__ubuf__ UT*)keys.GetPhyAddr();
    __local_mem__ uint16_t* histogramPtr = (__ubuf__ uint16_t*)tileHistogram.GetPhyAddr();
    uint32_t remain = count;
    uint16_t repeats = CeilDivision(count, VF_LEN_B8);
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> hist0, hist1;
        Reg::RegTensor<uint8_t> input, prefixMaskReg, masked;
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
        Reg::Duplicate(hist0, 0, maskB16);
        Reg::Duplicate(hist1, 0, maskB16);
        Reg::Duplicate(prefixMaskReg, prefixMask, maskB8);
        for (uint16_t i = 0; i < repeats; ++i) {
            Reg::MaskReg validMask = Reg::UpdateMask<uint8_t>(remain);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(input, keyPtr, VF_LEN_B8);
            Reg::And(masked, input, prefixMaskReg, validMask);
            Reg::MaskReg selectedMask;
            Reg::CompareScalar<uint8_t, CMPMODE::EQ>(selectedMask, masked, prefixKey, validMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(
                hist0, input, selectedMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(
                hist1, input, selectedMask);
        }
        StoreRadixByteHistogram(histogramPtr, hist0, hist1, maskB16);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::BuildTileHistogramB16(LocalTensor<UT>& keys, uint32_t count,
                                                                         uint32_t shift, UT prefixMask, UT prefixKey,
                                                                         LocalTensor<uint16_t>& tileHistogram)
{
    __local_mem__ UT* keyPtr = (__ubuf__ UT*)keys.GetPhyAddr();
    __local_mem__ uint16_t* histogramPtr = (__ubuf__ uint16_t*)tileHistogram.GetPhyAddr();
    uint32_t remain = count;
    uint16_t repeats = CeilDivision(count, VF_LEN_B8);
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> hist0, hist1, input0, input1, shifted0, shifted1;
        Reg::RegTensor<uint16_t> prefixMaskReg, masked0, masked1, flag0, flag1, one, zero;
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
        Reg::Duplicate(hist0, 0, maskB16);
        Reg::Duplicate(hist1, 0, maskB16);
        Reg::Duplicate(prefixMaskReg, prefixMask, maskB16);
        Reg::Duplicate(one, 1, maskB16);
        Reg::Duplicate(zero, 0, maskB16);
        for (uint16_t i = 0; i < repeats; ++i) {
            Reg::MaskReg histMask = Reg::UpdateMask<uint8_t>(remain);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(input0, keyPtr, VF_LEN_B16);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(input1, keyPtr, VF_LEN_B16);
            Reg::And(masked0, input0, prefixMaskReg, maskB16);
            Reg::And(masked1, input1, prefixMaskReg, maskB16);
            Reg::MaskReg selected0, selected1;
            Reg::CompareScalar<uint16_t, CMPMODE::EQ>(selected0, masked0, prefixKey, maskB16);
            Reg::CompareScalar<uint16_t, CMPMODE::EQ>(selected1, masked1, prefixKey, maskB16);
            Reg::Select(flag0, one, zero, selected0);
            Reg::Select(flag1, one, zero, selected1);
            Reg::ShiftRights<uint16_t, int16_t>(shifted0, input0, shift, maskB16);
            Reg::ShiftRights<uint16_t, int16_t>(shifted1, input1, shift, maskB16);
            Reg::RegTensor<uint8_t> bytes, highBytes, flagBytes, highFlags;
            Reg::DeInterleave(bytes, highBytes, (Reg::RegTensor<uint8_t>&)shifted0, (Reg::RegTensor<uint8_t>&)shifted1);
            Reg::DeInterleave(flagBytes, highFlags, (Reg::RegTensor<uint8_t>&)flag0, (Reg::RegTensor<uint8_t>&)flag1);
            AddSelectedByteHistogram(hist0, hist1, bytes, flagBytes, histMask);
        }
        StoreRadixByteHistogram(histogramPtr, hist0, hist1, maskB16);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::BuildTileHistogramB32(LocalTensor<UT>& keys, uint32_t count,
                                                                         uint32_t shift, UT prefixMask, UT prefixKey,
                                                                         LocalTensor<uint16_t>& tileHistogram)
{
    __local_mem__ UT* keyPtr = (__ubuf__ UT*)keys.GetPhyAddr();
    __local_mem__ uint16_t* histogramPtr = (__ubuf__ uint16_t*)tileHistogram.GetPhyAddr();
    uint32_t remain = count;
    uint16_t repeats = CeilDivision(count, VF_LEN_B8);
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> hist0, hist1;
        Reg::RegTensor<uint32_t> input0, input1, input2, input3, shifted0, shifted1, shifted2, shifted3;
        Reg::RegTensor<uint32_t> prefixMaskReg, masked, flag0, flag1, flag2, flag3, one, zero;
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
        Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t>();
        Reg::Duplicate(hist0, 0, maskB16);
        Reg::Duplicate(hist1, 0, maskB16);
        Reg::Duplicate(prefixMaskReg, prefixMask, maskB32);
        Reg::Duplicate(one, 1, maskB32);
        Reg::Duplicate(zero, 0, maskB32);
        for (uint16_t i = 0; i < repeats; ++i) {
            Reg::MaskReg histMask = Reg::UpdateMask<uint8_t>(remain);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input0, keyPtr, VF_LEN_B32);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input1, keyPtr, VF_LEN_B32);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input2, keyPtr, VF_LEN_B32);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input3, keyPtr, VF_LEN_B32);
            SelectPrefixFlag(flag0, input0, masked, prefixMaskReg, prefixKey, one, zero, maskB32);
            SelectPrefixFlag(flag1, input1, masked, prefixMaskReg, prefixKey, one, zero, maskB32);
            SelectPrefixFlag(flag2, input2, masked, prefixMaskReg, prefixKey, one, zero, maskB32);
            SelectPrefixFlag(flag3, input3, masked, prefixMaskReg, prefixKey, one, zero, maskB32);
            Reg::ShiftRights<uint32_t, int16_t>(shifted0, input0, shift, maskB32);
            Reg::ShiftRights<uint32_t, int16_t>(shifted1, input1, shift, maskB32);
            Reg::ShiftRights<uint32_t, int16_t>(shifted2, input2, shift, maskB32);
            Reg::ShiftRights<uint32_t, int16_t>(shifted3, input3, shift, maskB32);
            Reg::RegTensor<uint8_t> bytes, highBytes;
            Reg::RegTensor<uint8_t> flagBytes, highFlags;
            PackB32RadixBytes(bytes, highBytes, shifted0, shifted1, shifted2, shifted3);
            PackB32FlagBytes(flagBytes, highFlags, flag0, flag1, flag2, flag3);
            AddSelectedByteHistogram(hist0, hist1, bytes, flagBytes, histMask);
        }
        StoreRadixByteHistogram(histogramPtr, hist0, hist1, maskB16);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::BuildTileHistogramB64(LocalTensor<UT>& keys, uint32_t count,
                                                                         uint32_t shift, UT prefixMask, UT prefixKey,
                                                                         LocalTensor<uint16_t>& tileHistogram)
{
    __local_mem__ UT* keyPtr = (__ubuf__ UT*)keys.GetPhyAddr();
    uint16_t repeats = CeilDivision(count, VF_LEN_B8);
    uint32_t remain = count;
    __local_mem__ uint16_t* histogramPtr = (__ubuf__ uint16_t*)tileHistogram.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> hist0, hist1;
        Reg::RegTensor<uint64_t> input0, input1, input2, input3, input4, input5, input6, input7;
        Reg::RegTensor<uint64_t> shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6, shifted7;
        Reg::RegTensor<uint64_t> flag0, flag1, flag2, flag3, flag4, flag5, flag6, flag7;
        Reg::RegTensor<uint64_t> prefixMaskReg, masked, one, zero;
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
        Reg::MaskReg maskB64 = Reg::CreateMask<uint64_t>();
        Reg::Duplicate(hist0, 0, maskB16);
        Reg::Duplicate(hist1, 0, maskB16);
        Reg::Duplicate(prefixMaskReg, prefixMask, maskB64);
        Reg::Duplicate(one, 1, maskB64);
        Reg::Duplicate(zero, 0, maskB64);
        for (uint16_t i = 0; i < repeats; ++i) {
            Reg::MaskReg histMask = Reg::UpdateMask<uint8_t>(remain);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input0, keyPtr, VF_LEN_B64);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input1, keyPtr, VF_LEN_B64);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input2, keyPtr, VF_LEN_B64);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input3, keyPtr, VF_LEN_B64);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input4, keyPtr, VF_LEN_B64);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input5, keyPtr, VF_LEN_B64);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input6, keyPtr, VF_LEN_B64);
            Reg::DataCopy<uint64_t, Reg::PostLiteral::POST_MODE_UPDATE>(input7, keyPtr, VF_LEN_B64);
            SelectPrefixFlag(flag0, input0, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            SelectPrefixFlag(flag1, input1, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            SelectPrefixFlag(flag2, input2, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            SelectPrefixFlag(flag3, input3, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            SelectPrefixFlag(flag4, input4, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            SelectPrefixFlag(flag5, input5, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            SelectPrefixFlag(flag6, input6, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            SelectPrefixFlag(flag7, input7, masked, prefixMaskReg, prefixKey, one, zero, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted0, input0, shift, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted1, input1, shift, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted2, input2, shift, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted3, input3, shift, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted4, input4, shift, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted5, input5, shift, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted6, input6, shift, maskB64);
            Reg::ShiftRights<uint64_t, int16_t>(shifted7, input7, shift, maskB64);
            Reg::RegTensor<uint8_t> bytes, highBytes;
            Reg::RegTensor<uint8_t> flagBytes, highFlags;
            PackB64RadixBytes(bytes, highBytes, shifted0, shifted1, shifted2, shifted3, shifted4, shifted5, shifted6,
                              shifted7);
            PackB64FlagBytes(flagBytes, highFlags, flag0, flag1, flag2, flag3, flag4, flag5, flag6, flag7);
            AddSelectedByteHistogram(hist0, hist1, bytes, flagBytes, histMask);
        }
        StoreRadixByteHistogram(histogramPtr, hist0, hist1, maskB16);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::ClearHistogram(LocalTensor<uint64_t>& histogram)
{
    __local_mem__ int64_t* histogramPtr = (__ubuf__ int64_t*)histogram.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg maskB64 = Reg::CreateMask<int64_t>();
        Reg::RegTensor<int64_t> zero;
        Reg::Duplicate(zero, 0, maskB64);
        for (uint16_t i = 0; i < RADIX / VF_LEN_B64; ++i) {
            Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramPtr, zero, VF_LEN_B64, maskB64);
        }
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::AccumulateHistogram(LocalTensor<uint64_t>& histogram,
                                                                       LocalTensor<uint64_t>& src)
{
    __local_mem__ int64_t* histogramReadPtr = (__ubuf__ int64_t*)histogram.GetPhyAddr();
    __local_mem__ int64_t* histogramWritePtr = histogramReadPtr;
    __local_mem__ int64_t* srcPtr = (__ubuf__ int64_t*)src.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg maskB64 = Reg::CreateMask<int64_t>();
        Reg::RegTensor<int64_t> dstReg, srcReg;
        for (uint16_t i = 0; i < RADIX / VF_LEN_B64; ++i) {
            Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(dstReg, histogramReadPtr, VF_LEN_B64);
            Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(srcReg, srcPtr, VF_LEN_B64);
            Reg::Add(dstReg, dstReg, srcReg, maskB64);
            Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, dstReg, VF_LEN_B64, maskB64);
        }
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::AccumulateTileHistogram(LocalTensor<uint64_t>& histogram,
                                                                           LocalTensor<uint16_t>& tileHistogram)
{
    __local_mem__ uint16_t* tilePtr = (__ubuf__ uint16_t*)tileHistogram.GetPhyAddr();
    __local_mem__ int64_t* histogramReadPtr = (__ubuf__ int64_t*)histogram.GetPhyAddr();
    __local_mem__ int64_t* histogramWritePtr = histogramReadPtr;
    __VEC_SCOPE__
    {
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t>();
        Reg::MaskReg maskB64 = Reg::CreateMask<int64_t>();
        Reg::RegTensor<uint16_t> hist0, hist1, zero16;
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(hist0, tilePtr, VF_LEN_B16);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(hist1, tilePtr, VF_LEN_B16);
        Reg::Duplicate(zero16, 0, maskB16);

        Reg::RegTensor<uint32_t> hist32_0, hist32_1, hist32_2, hist32_3;
        Reg::Interleave((Reg::RegTensor<uint16_t>&)hist32_0, (Reg::RegTensor<uint16_t>&)hist32_1, hist0, zero16);
        Reg::Interleave((Reg::RegTensor<uint16_t>&)hist32_2, (Reg::RegTensor<uint16_t>&)hist32_3, hist1, zero16);

        Reg::RegTensor<int64_t> hist64_0, hist64_1, hist64_2, hist64_3;
        Reg::RegTensor<int64_t> hist64_4, hist64_5, hist64_6, hist64_7;
        Reg::Interleave((Reg::RegTensor<uint32_t>&)hist64_0, (Reg::RegTensor<uint32_t>&)hist64_1, hist32_0,
                        (Reg::RegTensor<uint32_t>&)zero16);
        Reg::Interleave((Reg::RegTensor<uint32_t>&)hist64_2, (Reg::RegTensor<uint32_t>&)hist64_3, hist32_1,
                        (Reg::RegTensor<uint32_t>&)zero16);
        Reg::Interleave((Reg::RegTensor<uint32_t>&)hist64_4, (Reg::RegTensor<uint32_t>&)hist64_5, hist32_2,
                        (Reg::RegTensor<uint32_t>&)zero16);
        Reg::Interleave((Reg::RegTensor<uint32_t>&)hist64_6, (Reg::RegTensor<uint32_t>&)hist64_7, hist32_3,
                        (Reg::RegTensor<uint32_t>&)zero16);

        Reg::RegTensor<int64_t> old0, old1, old2, old3, old4, old5, old6, old7;
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old0, histogramReadPtr, VF_LEN_B64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old1, histogramReadPtr, VF_LEN_B64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old2, histogramReadPtr, VF_LEN_B64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old3, histogramReadPtr, VF_LEN_B64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old4, histogramReadPtr, VF_LEN_B64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old5, histogramReadPtr, VF_LEN_B64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old6, histogramReadPtr, VF_LEN_B64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(old7, histogramReadPtr, VF_LEN_B64);
        Reg::Add(old0, old0, hist64_0, maskB64);
        Reg::Add(old1, old1, hist64_1, maskB64);
        Reg::Add(old2, old2, hist64_2, maskB64);
        Reg::Add(old3, old3, hist64_3, maskB64);
        Reg::Add(old4, old4, hist64_4, maskB64);
        Reg::Add(old5, old5, hist64_5, maskB64);
        Reg::Add(old6, old6, hist64_6, maskB64);
        Reg::Add(old7, old7, hist64_7, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old0, VF_LEN_B64, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old1, VF_LEN_B64, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old2, VF_LEN_B64, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old3, VF_LEN_B64, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old4, VF_LEN_B64, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old5, VF_LEN_B64, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old6, VF_LEN_B64, maskB64);
        Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramWritePtr, old7, VF_LEN_B64, maskB64);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::CountByte(int64_t rowOffset, int64_t sliceStart, int64_t sliceCount,
                                                             uint32_t shift, UT prefixMask, UT prefixKey,
                                                             LocalTensor<uint64_t>& histogram)
{
    ClearHistogram(histogram);
    for (int64_t start = 0; start < sliceCount; start += static_cast<int64_t>(tileElems_)) {
        int64_t remain = sliceCount - start;
        uint32_t count = remain < static_cast<int64_t>(tileElems_) ? static_cast<uint32_t>(remain) : tileElems_;
        LocalTensor<T> input = inputQueue_.AllocTensor<T>();
        LoadTile(rowOffset + sliceStart + start, count, input);
        inputQueue_.EnQue<T>(input);
        input = inputQueue_.DeQue<T>();
        LocalTensor<UT> keys = MakeSortKeys(input, count);
        LocalTensor<uint16_t> tileHistogram = tileHistogramBuf_.Get<uint16_t>();
        BuildTileHistogram(keys, count, shift, prefixMask, prefixKey, tileHistogram);
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        AccumulateTileHistogram(histogram, tileHistogram);
        inputQueue_.FreeTensor(input);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::StoreCoreHistogram(LocalTensor<uint64_t>& histogram)
{
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId);
    WaitFlag<HardEvent::S_MTE3>(eventId);
    DataCopyExtParams params{1, RADIX * static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
    DataCopyPad(coreHistogramGm_[static_cast<uint64_t>(blockIdx_) * RADIX], histogram, params);
    event_t doneEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(doneEvent);
    WaitFlag<HardEvent::MTE3_S>(doneEvent);
}

template <typename T, typename UT>
__aicore__ inline uint64_t KthValueRadixSelect<T, UT>::LoadCoreHistogramBucket(uint32_t group, uint32_t coreInGroup,
                                                                               uint32_t bucket,
                                                                               LocalTensor<uint64_t>& scratch)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
    DataCopyPadExtParams<uint64_t> padParams{false, 0, 0, 0};
    uint64_t coreOffset = (static_cast<uint64_t>(group) * coresPerRow_ + coreInGroup) * RADIX;
    DataCopyPad(scratch, coreHistogramGm_[coreOffset + bucket], params, padParams);
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
    return scratch.GetValue(0);
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::ReduceGroupHistogram(uint32_t group,
                                                                        LocalTensor<uint64_t>& histogram,
                                                                        LocalTensor<uint64_t>& scratch)
{
    ClearHistogram(histogram);
    DataCopyExtParams params{1, RADIX * static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
    DataCopyPadExtParams<uint64_t> padParams{false, 0, 0, 0};
    for (uint32_t core = 0; core < coresPerRow_; ++core) {
        uint64_t coreOffset = (static_cast<uint64_t>(group) * coresPerRow_ + core) * RADIX;
        DataCopyPad(scratch, coreHistogramGm_[coreOffset], params, padParams);
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        AccumulateHistogram(histogram, scratch);
        event_t accSyncEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(accSyncEvent);
        WaitFlag<HardEvent::V_MTE2>(accSyncEvent);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::StoreGroupState(uint32_t group, LocalTensor<uint64_t>& state)
{
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId);
    WaitFlag<HardEvent::S_MTE3>(eventId);
    DataCopyExtParams params{1, RADIX_SELECT_RESULT_WORDS * static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
    DataCopyPad(groupStateGm_[static_cast<uint64_t>(group) * RADIX_SELECT_RESULT_WORDS], state, params);
    event_t doneEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(doneEvent);
    WaitFlag<HardEvent::MTE3_S>(doneEvent);
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::LoadGroupState(uint32_t group, LocalTensor<uint64_t>& state)
{
    DataCopyExtParams params{1, RADIX_SELECT_RESULT_WORDS * static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
    DataCopyPadExtParams<uint64_t> padParams{false, 0, 0, 0};
    DataCopyPad(state, groupStateGm_[static_cast<uint64_t>(group) * RADIX_SELECT_RESULT_WORDS], params, padParams);
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::WriteOutput(int64_t row, T value, int64_t index)
{
    LocalTensor<T> outputValue = outputValueBuf_.Get<T>();
    LocalTensor<int64_t> outputIndex = outputIndexBuf_.Get<int64_t>();
    outputValue.SetValue(0, value);
    outputIndex.SetValue(0, index);
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId);
    WaitFlag<HardEvent::S_MTE3>(eventId);
    DataCopyExtParams valueParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyExtParams indexParams{1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(valuesGm_[row], outputValue, valueParams);
    DataCopyPad(indicesGm_[row], outputIndex, indexParams);
}

// Enter or update active mode: if already active, compact existing indices;
// otherwise collect from full data. Stay active only if count in (0, 512].
template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::UpdateActiveIndices(LocalTensor<UT>& retainedKeys,
                                                                       LocalTensor<uint32_t>& activeIndices,
                                                                       uint32_t retainedCount, UT prefixMask,
                                                                       UT prefixKey, bool& activeMode,
                                                                       uint32_t& activeCount)
{
    LocalTensor<int64_t> activeResult = findResultBuf_.Get<int64_t>();
    if (activeMode) {
        asc_vf_call<CompactActiveIndices<UT>>(
            dim3(RADIX_SELECT_FIND_THREADS), reinterpret_cast<__ubuf__ UT*>(retainedKeys.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(activeIndices.GetPhyAddr()), activeCount, prefixMask, prefixKey,
            reinterpret_cast<__ubuf__ int64_t*>(activeResult.GetPhyAddr()));
    } else {
        LocalTensor<uint32_t> threadCounts = findCountBuf_.Get<uint32_t>();
        asc_vf_call<CollectActiveIndices<UT>>(
            dim3(RADIX_SELECT_FIND_THREADS), reinterpret_cast<__ubuf__ UT*>(retainedKeys.GetPhyAddr()), retainedCount,
            prefixMask, prefixKey, reinterpret_cast<__ubuf__ uint32_t*>(threadCounts.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(activeIndices.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int64_t*>(activeResult.GetPhyAddr()));
    }
    event_t activeEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(activeEvent);
    WaitFlag<HardEvent::V_S>(activeEvent);
    activeCount = static_cast<uint32_t>(activeResult.GetValue(0));
    activeMode = activeCount > 0U && activeCount <= RADIX_SELECT_ACTIVE_MODE_THRESHOLD;
}

// Core algorithm: process one row by narrowing prefix byte-by-byte (MSB to LSB).
// Each round: histogram current byte of prefix-matching elements, find which bucket
// contains the kth element, extend prefix. Multi-core: each core handles a slice,
// leader reduces histograms and broadcasts the selected bucket via workspace GM.
template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::SelectOneRow(int64_t row, uint32_t group, uint32_t coreInGroup,
                                                                bool writeOutput)
{
    LocalTensor<uint64_t> histogram = histogramBuf_.Get<uint64_t>();
    LocalTensor<uint64_t> reduceScratch = groupReduceBuf_.Get<uint64_t>();
    uint64_t leftK = static_cast<uint64_t>(kthIndex_); // remaining elements to skip
    UT prefixMask = static_cast<UT>(0);                // bits already determined
    UT prefixKey = static_cast<UT>(0);                 // values of those bits
    int64_t rowOffset = row * axisLen_;
    int64_t sliceSize = Ops::Base::CeilDiv(axisLen_, static_cast<int64_t>(coresPerRow_));
    int64_t sliceStart = static_cast<int64_t>(coreInGroup) * sliceSize;
    int64_t sliceRemain = axisLen_ - sliceStart;
    int64_t sliceCount = sliceStart < axisLen_ ? (sliceSize < sliceRemain ? sliceSize : sliceRemain) : 0;
    uint32_t finalTargetCore = coreInGroup;
    uint64_t finalTargetLeftK = leftK;
    bool keepFinalTile = sliceCount > 0 && sliceCount <= static_cast<int64_t>(tileElems_);
    bool hasRetainedTile = false;
    bool activeMode = false;
    uint32_t activeCount = 0;
    uint32_t retainedCount = 0;
    LocalTensor<T> retainedInput;
    LocalTensor<UT> retainedKeys;
    LocalTensor<uint32_t> activeIndices = activeIndexBuf_.Get<uint32_t>();

    for (int32_t byte = static_cast<int32_t>(sizeof(UT)) - 1; byte >= 0; --byte) {
        uint32_t shift = static_cast<uint32_t>(byte) * BYTE_BITS;
        if (keepFinalTile) {
            if (!hasRetainedTile) {
                retainedCount = static_cast<uint32_t>(sliceCount);
                retainedInput = inputQueue_.AllocTensor<T>();
                LoadTile(rowOffset + sliceStart, retainedCount, retainedInput);
                inputQueue_.EnQue<T>(retainedInput);
                retainedInput = inputQueue_.DeQue<T>();
                retainedKeys = MakeSortKeys(retainedInput, retainedCount);
                hasRetainedTile = true;
            }
            if (activeMode) {
                asc_vf_call<BuildActiveHistogram<UT>>(
                    dim3(RADIX_SELECT_FIND_THREADS), reinterpret_cast<__ubuf__ UT*>(retainedKeys.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(activeIndices.GetPhyAddr()), activeCount, shift,
                    reinterpret_cast<__ubuf__ uint64_t*>(histogram.GetPhyAddr()));
                event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
            } else {
                ClearHistogram(histogram);
                event_t clearEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(clearEvent);
                WaitFlag<HardEvent::V_S>(clearEvent);
                LocalTensor<uint16_t> tileHistogram = tileHistogramBuf_.Get<uint16_t>();
                BuildTileHistogram(retainedKeys, retainedCount, shift, prefixMask, prefixKey, tileHistogram);
                event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
                AccumulateTileHistogram(histogram, tileHistogram);
            }
        } else {
            CountByte(rowOffset, sliceStart, sliceCount, shift, prefixMask, prefixKey, histogram);
        }
        event_t histSyncEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(histSyncEvent);
        WaitFlag<HardEvent::V_S>(histSyncEvent);
        if (coresPerRow_ == 1U) {
            uint64_t accumulated = 0;
            uint32_t selected = 0;
            for (; selected < RADIX; ++selected) {
                uint64_t count = histogram.GetValue(selected);
                if (leftK < accumulated + count) {
                    leftK -= accumulated;
                    break;
                }
                accumulated += count;
            }
            if (selected >= RADIX) {
                if (hasRetainedTile) {
                    inputQueue_.FreeTensor(retainedInput);
                }
                return;
            }
            uint64_t selectedCount = histogram.GetValue(selected);
            UT byteMask = static_cast<UT>(static_cast<UT>(0xFFU) << shift);
            prefixMask = static_cast<UT>(prefixMask | byteMask);
            prefixKey = static_cast<UT>(prefixKey | (static_cast<UT>(selected) << shift));
            if (keepFinalTile && selectedCount <= RADIX_SELECT_ACTIVE_MODE_THRESHOLD) {
                UpdateActiveIndices(retainedKeys, activeIndices, retainedCount, prefixMask, prefixKey, activeMode,
                                    activeCount);
            }
            if (selectedCount == 1UL) {
                break;
            }
            continue;
        }
        uint32_t selected = 0;
        uint64_t selectedCount = 0;
        StoreCoreHistogram(histogram);
        SyncAll();
        LocalTensor<uint64_t> state = findResultBuf_.Get<uint64_t>();
        if (coreInGroup == 0U) {
            ReduceGroupHistogram(group, histogram, reduceScratch);
            event_t reduceSyncEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(reduceSyncEvent);
            WaitFlag<HardEvent::V_S>(reduceSyncEvent);
            uint64_t accumulated = 0;
            for (; selected < RADIX; ++selected) {
                uint64_t count = histogram.GetValue(selected);
                if (leftK < accumulated + count) {
                    leftK -= accumulated;
                    break;
                }
                accumulated += count;
            }
            if (byte == 0 && selected < RADIX) {
                uint64_t coreAccumulated = 0;
                uint32_t targetCore = coresPerRow_;
                uint64_t targetLeftK = leftK;
                for (uint32_t core = 0; core < coresPerRow_; ++core) {
                    uint64_t count = LoadCoreHistogramBucket(group, core, selected, reduceScratch);
                    if (leftK < coreAccumulated + count) {
                        targetCore = core;
                        targetLeftK = leftK - coreAccumulated;
                        break;
                    }
                    coreAccumulated += count;
                }
                state.SetValue(RADIX_SELECT_STATE_TARGET_CORE_IDX, targetCore);
                state.SetValue(RADIX_SELECT_STATE_TARGET_LEFT_K_IDX, targetLeftK);
            }
            state.SetValue(RADIX_SELECT_STATE_LEFT_K_IDX, leftK);
            state.SetValue(RADIX_SELECT_STATE_SELECTED_IDX, selected);
            state.SetValue(RADIX_SELECT_STATE_SELECTED_COUNT_IDX,
                           selected < RADIX ? histogram.GetValue(selected) : 0UL);
            StoreGroupState(group, state);
        }
        SyncAll();
        LoadGroupState(group, state);
        selected = static_cast<uint32_t>(state.GetValue(RADIX_SELECT_STATE_SELECTED_IDX));
        if (selected >= RADIX) {
            if (hasRetainedTile) {
                inputQueue_.FreeTensor(retainedInput);
            }
            return;
        }
        leftK = state.GetValue(RADIX_SELECT_STATE_LEFT_K_IDX);
        selectedCount = state.GetValue(RADIX_SELECT_STATE_SELECTED_COUNT_IDX);
        if (byte == 0) {
            finalTargetCore = static_cast<uint32_t>(state.GetValue(RADIX_SELECT_STATE_TARGET_CORE_IDX));
            finalTargetLeftK = state.GetValue(RADIX_SELECT_STATE_TARGET_LEFT_K_IDX);
        }
        UT byteMask = static_cast<UT>(static_cast<UT>(0xFFU) << shift);
        prefixMask = static_cast<UT>(prefixMask | byteMask);
        prefixKey = static_cast<UT>(prefixKey | (static_cast<UT>(selected) << shift));
        if (keepFinalTile && selectedCount <= RADIX_SELECT_ACTIVE_MODE_THRESHOLD) {
            UpdateActiveIndices(retainedKeys, activeIndices, retainedCount, prefixMask, prefixKey, activeMode,
                                activeCount);
        }
    }

    if (coresPerRow_ > 1U) {
        if (coreInGroup != finalTargetCore) {
            if (hasRetainedTile) {
                inputQueue_.FreeTensor(retainedInput);
            }
            return;
        }
        leftK = finalTargetLeftK;
    }

    // Final location: three paths depending on retained tile and active mode
    // Path A: active mode + retained tile → O(1) lookup from activeIndices
    if (hasRetainedTile && activeMode && leftK < static_cast<uint64_t>(activeCount)) {
        uint32_t found = activeIndices.GetValue(static_cast<uint32_t>(leftK));
        if (writeOutput) {
            WriteOutput(row, retainedInput.GetValue(found), sliceStart + found);
        }
        inputQueue_.FreeTensor(retainedInput);
        return;
    }

    if (hasRetainedTile) {
        LocalTensor<uint32_t> threadCounts = findCountBuf_.Get<uint32_t>();
        LocalTensor<int64_t> findResult = findResultBuf_.Get<int64_t>();
        asc_vf_call<FindKthMatchInTile<UT>>(
            dim3(RADIX_SELECT_FIND_THREADS), reinterpret_cast<__ubuf__ UT*>(retainedKeys.GetPhyAddr()), retainedCount,
            prefixMask, prefixKey, leftK, reinterpret_cast<__ubuf__ uint32_t*>(threadCounts.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int64_t*>(findResult.GetPhyAddr()));
        event_t findEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(findEvent);
        WaitFlag<HardEvent::V_S>(findEvent);
        int64_t found = findResult.GetValue(0);
        if (found >= 0 && writeOutput) {
            WriteOutput(row, retainedInput.GetValue(static_cast<uint32_t>(found)), sliceStart + found);
        }
        inputQueue_.FreeTensor(retainedInput);
        return;
    }

    uint64_t matched = 0;
    for (int64_t start = 0; start < sliceCount; start += static_cast<int64_t>(tileElems_)) {
        int64_t remain = sliceCount - start;
        uint32_t count = remain < static_cast<int64_t>(tileElems_) ? static_cast<uint32_t>(remain) : tileElems_;
        LocalTensor<T> input = inputQueue_.AllocTensor<T>();
        LoadTile(rowOffset + sliceStart + start, count, input);
        inputQueue_.EnQue<T>(input);
        input = inputQueue_.DeQue<T>();
        LocalTensor<UT> keys = MakeSortKeys(input, count);
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        LocalTensor<uint32_t> threadCounts = findCountBuf_.Get<uint32_t>();
        LocalTensor<int64_t> findResult = findResultBuf_.Get<int64_t>();
        uint64_t tileTarget = leftK >= matched ? leftK - matched : 0UL;
        asc_vf_call<FindKthMatchInTile<UT>>(
            dim3(RADIX_SELECT_FIND_THREADS), reinterpret_cast<__ubuf__ UT*>(keys.GetPhyAddr()), count, prefixMask,
            prefixKey, tileTarget, reinterpret_cast<__ubuf__ uint32_t*>(threadCounts.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int64_t*>(findResult.GetPhyAddr()));
        event_t findEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(findEvent);
        WaitFlag<HardEvent::V_S>(findEvent);
        int64_t found = findResult.GetValue(0);
        uint64_t tileMatches = static_cast<uint64_t>(findResult.GetValue(1));
        if (found >= 0 && leftK >= matched) {
            if (writeOutput) {
                WriteOutput(row, input.GetValue(static_cast<uint32_t>(found)), sliceStart + start + found);
            }
            inputQueue_.FreeTensor(input);
            break;
        }
        matched += tileMatches;
        inputQueue_.FreeTensor(input);
    }
}

template <typename T, typename UT>
__aicore__ inline void KthValueRadixSelect<T, UT>::Process()
{
    if (tileElems_ == 0U || blockNum_ == 0U || rowsParallel_ == 0U || coresPerRow_ == 0U || sortLoopTimes_ == 0U) {
        return;
    }
    uint32_t group = blockIdx_ / coresPerRow_;
    uint32_t coreInGroup = blockIdx_ % coresPerRow_;
    for (uint32_t loop = 0; loop < sortLoopTimes_; ++loop) {
        int64_t logicalRow = static_cast<int64_t>(loop) * rowsParallel_ + group;
        bool writeOutput = logicalRow < rowCount_;
        int64_t row = writeOutput ? logicalRow : 0;
        SelectOneRow(row, group, coreInGroup, writeOutput);
    }
}
} // namespace KthValue

#endif
