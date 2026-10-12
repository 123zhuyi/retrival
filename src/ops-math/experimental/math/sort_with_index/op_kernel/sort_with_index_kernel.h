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
 * \file sort_with_index_kernel.h
 * \brief SortWithIndex device kernel (ascend910b, DAV_2201).
 *
 * SortWithIndex<VALUE_DT, INDEX_DT, SIZE_MODE>
 *   VALUE_DT  : value dtype (x/y). One of {half, float, bfloat16_t, int32_t}.
 *   INDEX_DT  : index dtype (index/sorted_index). One of {int32_t, int64_t}.
 *   SIZE_MODE : 0 SINGLE_TILE / 1 MRGSORT (large axis core-internal merge) / 2 EMPTY (no-op).
 *
 * Value data path (DESIGN 3.3/3.4, all validated by iteration-1 probes):
 *   - half  : sorts directly in the half domain (SortT == half). No Cast.
 *   - float : sorts directly in the float domain (SortT == float). No Cast.
 *   - bf16  : Cast(f32, bf16, CAST_NONE) -> Sort<float> -> Cast(bf16, f32, CAST_RINT). Lossless.
 *   - int32 : Cast(f32, int32, CAST_RINT) -> Sort<float> -> Cast(int32, f32, CAST_RINT). Exact only
 *             for |x| <= 2^24 (documented limitation, spec/REQUIREMENTS).
 *   SortT = half for VALUE_DT==half, else float. The Cast paths use a dedicated float key buffer.
 *
 * Index-follow scheme (DESIGN 3.5): the Sort index channel carries a POSITION p (0..N-1, uint32),
 * NOT the raw index values. After Extract gives the permutation sortedPos, the real index values
 * are collected via Gather.
 *   - int32 : single Gather(idxLocal, srcOffset = sortedPos * 4 Bytes).
 *   - int64 : Gather is unavailable for int64 on A2, so the int64 row is viewed as [N x 2] int32
 *             (lo/hi interleaved); two Gathers (offset = sortedPos*8 for lo, sortedPos*8+4 for hi)
 *             then an interleaved SetValue writeback reassembles the int64 result (probe-validated).
 *
 * Ordering: Sort is descending. Ascending (default, descending=false) is realized by Muls(-1) on the
 * sort key before Sort and Muls(-1) on the extracted value after. Padding uses an INFINITE per-dtype
 * sentinel (+inf ascending / -inf descending; I1 iteration-3 correctness fix) so the padded tail sorts
 * away from the valid region AND never displaces a real +-Inf element. (Iteration-2 used a FINITE
 * sentinel +3e38 < +Inf; probe_special_values found that with ascending + padding + a real +Inf in the
 * row, the smaller +3e38 sentinel took +Inf's valid rank and +Inf was dropped from the output -- a
 * correctness bug. The inf sentinel sorts at the same extreme as a real +-Inf, so CopyOut trims only
 * the "+Inf + sentinel" mixed tail and the valid region is never stolen. inf-as-sort-key stability in
 * the single tile is NPU-validated by probe_inf_sentinel.) CopyIn whole-region Duplicate(sentinel) +
 * DataCopyPad(isPad=true, rightPadding=padLen != 0) keeps the seam from being dummy-filled with the
 * first element (probe finding #1/#3). Only the first N elements are written back.
 *
 * SIZE_MODE=1 (MRGSORT, DESIGN 3.6): single row N > tileLen. realSortLen is rounded up to 32*(power of
 * 4) (host), so the row splits into numRuns power-of-4 sorted runs (Sort<SortT,false> emits one
 * sorted run per 32 elements); MergeRuns does in-core ping-pong 4-way MrgSort until the whole row is
 * sorted. No SyncAll (rows never cross cores). pos follows through; Gather writes back the index.
 */
#ifndef SORT_WITH_INDEX_KERNEL_H
#define SORT_WITH_INDEX_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sort_with_index_tiling_data.h"
#include "sort_with_index_tiling_key.h"
#include <cfloat>

namespace NsSortWithIndex {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ELEMENT_16 = 16U; // Concat: 16 elements per repeat
constexpr uint32_t ELEMENT_32 = 32U; // Sort/Extract: 32 elements per repeat (one sorted run)
// half/float proposal record = 8 Bytes/element. As a SortT(half/float) view that is:
//   half  -> 4 elements/record (4*2B = 8B); float -> 2 elements/record (2*4B = 8B).
// Buffers that hold proposals (concat/sorted) MUST be sized realSortLen * PROPOSAL_ELEM_PER_REC.
constexpr uint32_t PROPOSAL_HALF_PER_ELEM = 4U;  // half proposal: 4 half elements per record
constexpr uint32_t PROPOSAL_FLOAT_PER_ELEM = 2U; // float proposal: 2 float elements per record
constexpr uint32_t MRG_WAYS = 4U;                // MrgSort fixed 4-way (numRuns is a power of 4)

// I1 (iteration-3 correctness fix): the ascending padding sentinel must be >= the largest possible
// ascending value, INCLUDING +Inf. Iteration-2 used finite +-3e38/+-65504/+-2^30 (< +Inf); with
// ascending + padding + a real +Inf the smaller sentinel stole +Inf's valid rank and dropped +Inf
// (probe_special_values finding #2). Using +-Inf as the sentinel makes the padded tail sort at the
// same extreme as a real +-Inf, so CopyOut trims only the mixed "+Inf + sentinel" tail and never the
// valid region. inf-as-sort-key stability in the SINGLE TILE (no MrgSort merge) is NPU-validated by
// probe_inf_sentinel. The half/float/bf16 paths carry the inf sentinel as the sort key directly; the
// int32-value path writes the inf sentinel into the FLOAT key buffer only (the in-row int32 sentinel
// is a placeholder 0 and is never cast to int32 since only the valid sliceLen elements are cast back),
// so Cast(int32) never sees inf.
constexpr float SENTINEL_INF = __builtin_huge_valf(); // +inf (float); half cast yields half +inf

template <typename VALUE_DT, typename INDEX_DT, uint32_t SIZE_MODE>
class SortWithIndex {
public:
    __aicore__ inline SortWithIndex()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex, GM_ADDR workspace,
        const SortWithIndexTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(uint32_t rowIdx);

    __aicore__ inline LocalTensor<typename std::conditional<std::is_same<VALUE_DT, half>::value, half, float>::type>
    MergeRuns(
        LocalTensor<typename std::conditional<std::is_same<VALUE_DT, half>::value, half, float>::type> bufA,
        LocalTensor<typename std::conditional<std::is_same<VALUE_DT, half>::value, half, float>::type> bufB,
        uint32_t numRuns, uint32_t runLen);

    // SortT: half stays half; bf16 / float / int32 sort through float.
    using SortT = typename std::conditional<std::is_same<VALUE_DT, half>::value, half, float>::type;
    // SortT-view elements per 8B proposal record (4 for half, 2 for float).
    static constexpr uint32_t kProposalPerElem =
        std::is_same<SortT, half>::value ? PROPOSAL_HALF_PER_ELEM : PROPOSAL_FLOAT_PER_ELEM;
    // Value paths: "direct" = sort key is the value reinterpreted as SortT (half / float).
    //              "cast"   = value Cast to/from a separate float key buffer (bf16 / int32).
    static constexpr bool kValueDirect = std::is_same<VALUE_DT, half>::value || std::is_same<VALUE_DT, float>::value;
    static constexpr bool kValueIsBf16 = std::is_same<VALUE_DT, bfloat16_t>::value;
    static constexpr bool kValueIsInt32 = std::is_same<VALUE_DT, int32_t>::value;
    static constexpr bool kIndexIsInt64 = std::is_same<INDEX_DT, int64_t>::value;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueIdx;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueIdx;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> keyBuf;       // float key carrier (Cast paths only)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sortedKeyBuf; // extracted sorted key (Cast paths)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> posBuf;       // position channel (uint32)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sortedPosBuf; // permutation p (uint32)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> offsetBuf;    // byte offset for Gather (uint32)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> offsetHiBuf;  // hi byte offset for int64 Gather
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gatherLoBuf;  // int64 lo dword Gather result
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gatherHiBuf;  // int64 hi dword Gather result
    AscendC::TBuf<AscendC::QuePosition::VECCALC> propABuf;     // Sort proposal (value + pos) / merge A
    AscendC::TBuf<AscendC::QuePosition::VECCALC> propBBuf;     // merge B (ping-pong, MRGSORT)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> concatTmpBuf; // Concat tmp
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sortTmpBuf;   // Sort tmp

    AscendC::GlobalTensor<VALUE_DT> xGm;
    AscendC::GlobalTensor<INDEX_DT> indexGm;
    AscendC::GlobalTensor<VALUE_DT> yGm;
    AscendC::GlobalTensor<INDEX_DT> sortedIndexGm;

    uint32_t coreRowNum = 0;
    uint32_t sliceLen = 0;
    uint32_t realSortLen = 0;
    uint32_t align8 = 0;
    uint32_t padLen = 0;
    uint32_t dupCount = 0;
    uint32_t concatRepeat = 0;
    uint32_t sortRepeat = 0;
    uint32_t extractRepeat = 0;
    uint32_t numRuns0 = 0; // initial sorted-run count for MRGSORT (= realSortLen / 32)
    bool descending = false;
};

template <typename VALUE_DT, typename INDEX_DT, uint32_t SIZE_MODE>
__aicore__ inline void SortWithIndex<VALUE_DT, INDEX_DT, SIZE_MODE>::Init(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex, GM_ADDR workspace,
    const SortWithIndexTilingData* tilingData)
{
    // EMPTY: nothing to do.
    if constexpr (SIZE_MODE == SORT_WITH_INDEX_SIZE_MODE_EMPTY) {
        coreRowNum = 0;
        return;
    }

    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    const uint32_t coreIdx = AscendC::GetBlockIdx();
    const uint32_t validCoreNum = tilingData->validCoreNum;
    if (coreIdx >= validCoreNum) {
        coreRowNum = 0;
        return;
    }

    const uint32_t bigCoreNum = tilingData->bigCoreNum;
    const uint32_t bigCoreRowNum = tilingData->bigCoreRowNum;
    const uint32_t smallCoreRowNum = tilingData->smallCoreRowNum;

    sliceLen = tilingData->sliceLen;
    realSortLen = tilingData->realSortLen;
    align8 = tilingData->align8;
    padLen = tilingData->padLen;
    dupCount = tilingData->dupCount;
    descending = tilingData->descending;

    // Row range owned by this core. The first 'bigCoreNum' cores own one extra row each.
    uint32_t rowStart;
    if (coreIdx < bigCoreNum) {
        coreRowNum = bigCoreRowNum;
        rowStart = coreIdx * bigCoreRowNum;
    } else {
        coreRowNum = smallCoreRowNum;
        rowStart = bigCoreNum * bigCoreRowNum + (coreIdx - bigCoreNum) * smallCoreRowNum;
    }
    if (coreRowNum == 0) {
        return;
    }

    concatRepeat = realSortLen / ELEMENT_16;
    sortRepeat = realSortLen / ELEMENT_32; // Sort/Extract granularity: 32 elements per repeat
    extractRepeat = realSortLen / ELEMENT_32;
    numRuns0 = realSortLen / ELEMENT_32; // each sorted run is 32 elements

    const uint64_t elemStart = static_cast<uint64_t>(rowStart) * sliceLen;
    const uint64_t coreElems = static_cast<uint64_t>(coreRowNum) * sliceLen;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ VALUE_DT*>(x) + elemStart, coreElems);
    indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ INDEX_DT*>(index) + elemStart, coreElems);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ VALUE_DT*>(y) + elemStart, coreElems);
    sortedIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ INDEX_DT*>(sortedIndex) + elemStart, coreElems);

    // ---- I/O queues (sized by the real dtype) ----
    pipe.InitBuffer(inQueueX, BUFFER_NUM, realSortLen * sizeof(VALUE_DT));
    pipe.InitBuffer(inQueueIdx, BUFFER_NUM, realSortLen * sizeof(INDEX_DT));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, realSortLen * sizeof(VALUE_DT));
    pipe.InitBuffer(outQueueIdx, BUFFER_NUM, realSortLen * sizeof(INDEX_DT));

    // ---- float key carriers (Cast paths bf16/int32 only) ----
    if constexpr (!kValueDirect) {
        pipe.InitBuffer(keyBuf, realSortLen * sizeof(float));
        pipe.InitBuffer(sortedKeyBuf, realSortLen * sizeof(float));
    }

    // ---- position / permutation / Gather offset (uint32) ----
    pipe.InitBuffer(posBuf, realSortLen * sizeof(uint32_t));
    pipe.InitBuffer(sortedPosBuf, realSortLen * sizeof(uint32_t));
    pipe.InitBuffer(offsetBuf, realSortLen * sizeof(uint32_t));
    if constexpr (kIndexIsInt64) {
        pipe.InitBuffer(offsetHiBuf, realSortLen * sizeof(uint32_t));
        pipe.InitBuffer(gatherLoBuf, realSortLen * sizeof(int32_t));
        pipe.InitBuffer(gatherHiBuf, realSortLen * sizeof(int32_t));
    }

    // ---- proposal + tmp (8B/record => realSortLen * kProposalPerElem SortT elements) ----
    // probe_mrgsort_across_dtype finding #2: on DAV_2201 Sort -> Sort32 (no tmp param) and Concat ->
    // concat=src (no tmp); the Sort/Concat tmp & concat buffers are NEVER read/written, they are only
    // API formal-parameter placeholders. They MUST be sized at the proposal record size
    // (realSortLen * kProposalPerElem = 8B/elem), NOT the sort_v2 9x head-room. With float (4B) sort
    // keys the old 9x sizing (72B/elem for both tmp+concat) overflows the 192KB UB at large realSortLen
    // (e.g. float32 N=1024 -> realSortLen=2048 silently outputs all zeros). Proposal-size suffices.
    pipe.InitBuffer(propABuf, realSortLen * kProposalPerElem * sizeof(SortT));
    if constexpr (SIZE_MODE == SORT_WITH_INDEX_SIZE_MODE_MRGSORT) {
        pipe.InitBuffer(propBBuf, realSortLen * kProposalPerElem * sizeof(SortT)); // ping-pong B
    }
    pipe.InitBuffer(concatTmpBuf, realSortLen * kProposalPerElem * sizeof(SortT));
    pipe.InitBuffer(sortTmpBuf, realSortLen * kProposalPerElem * sizeof(SortT));
}

template <typename VALUE_DT, typename INDEX_DT, uint32_t SIZE_MODE>
__aicore__ inline void SortWithIndex<VALUE_DT, INDEX_DT, SIZE_MODE>::CopyIn(uint32_t rowIdx)
{
    AscendC::LocalTensor<VALUE_DT> xLocal = inQueueX.AllocTensor<VALUE_DT>();
    AscendC::LocalTensor<INDEX_DT> idxLocal = inQueueIdx.AllocTensor<INDEX_DT>();

    // Value sentinel in the VALUE_DT domain. For int32-value the in-row sentinel is just a placeholder
    // (0): the real float sort-key sentinel (+-inf, I1) is written by Duplicate over the float key
    // buffer in Compute (the int32 sentinel never becomes a sort key, so Cast never sees inf). For
    // half / float / bf16 the value sentinel IS the sort-key sentinel (carried through reinterpret /
    // Cast). I1: ascending sentinel = +inf, descending = -inf (>= / <= any real value incl. +-Inf).
    VALUE_DT padVal;
    if constexpr (kValueIsInt32) {
        padVal = static_cast<VALUE_DT>(0);
    } else {
        // half / float / bf16: cast +-inf into the value dtype (half/bf16/float all have +-inf).
        padVal = descending ? static_cast<VALUE_DT>(-SENTINEL_INF) : static_cast<VALUE_DT>(SENTINEL_INF);
    }

    // (1) Pre-fill the whole 32-aligned region with the sentinel so any unaligned-tail seam reads a
    //     valid finite key (prevents 507035 AIV exception on non-aligned N). (2) DataCopyPad with
    //     isPad=true + rightPadding=padLen (!= 0 whenever N is not 16-aligned) makes the framework
    //     dummy-fill use the sentinel (not the first element), so the seam never overwrites a real
    //     extreme value (probe-validated: critical for multi-block / non-32-aligned N correctness).
    AscendC::Duplicate(xLocal, padVal, realSortLen);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopyExtParams xCopy{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sliceLen * sizeof(VALUE_DT)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<VALUE_DT> xPad{true, 0, static_cast<uint8_t>(padLen), padVal};
    AscendC::DataCopyPad(xLocal, xGm[static_cast<uint64_t>(rowIdx) * sliceLen], xCopy, xPad);

    // index row: copy sliceLen valid elements; tail padding is irrelevant (Gather drops it).
    AscendC::DataCopyExtParams idxCopy{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sliceLen * sizeof(INDEX_DT)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<INDEX_DT> idxPad{false, 0, 0, static_cast<INDEX_DT>(0)};
    AscendC::DataCopyPad(idxLocal, indexGm[static_cast<uint64_t>(rowIdx) * sliceLen], idxCopy, idxPad);

    inQueueX.EnQue(xLocal);
    inQueueIdx.EnQue(idxLocal);
}

// In-core multi-round 4-way MrgSort: merge numRuns sorted runs of length runLen into one sorted run.
// Host guarantees numRuns is a power of 4 (realSortLen rounded to 32 * power-of-4), so every round is
// strictly 4-way with equal run lengths. Returns the buffer holding the fully sorted proposal.
template <typename VALUE_DT, typename INDEX_DT, uint32_t SIZE_MODE>
__aicore__ inline AscendC::LocalTensor<
    typename std::conditional<std::is_same<VALUE_DT, half>::value, half, float>::type>
SortWithIndex<VALUE_DT, INDEX_DT, SIZE_MODE>::MergeRuns(
    AscendC::LocalTensor<typename std::conditional<std::is_same<VALUE_DT, half>::value, half, float>::type> bufA,
    AscendC::LocalTensor<typename std::conditional<std::is_same<VALUE_DT, half>::value, half, float>::type> bufB,
    uint32_t numRuns, uint32_t runLen)
{
    AscendC::LocalTensor<SortT> src = bufA;
    AscendC::LocalTensor<SortT> dst = bufB;
    uint32_t sortedNumDummy[MRG_WAYS];

    while (numRuns > 1U) {
        const uint32_t groupStride = runLen * kProposalPerElem; // SortT-view stride of one run
        const uint32_t outRuns = numRuns / MRG_WAYS;            // power-of-4 => exact division
        for (uint32_t g = 0; g < outRuns; ++g) {
            const uint32_t b0 = (MRG_WAYS * g + 0U) * groupStride;
            const uint32_t b1 = (MRG_WAYS * g + 1U) * groupStride;
            const uint32_t b2 = (MRG_WAYS * g + 2U) * groupStride;
            const uint32_t b3 = (MRG_WAYS * g + 3U) * groupStride;
            AscendC::MrgSortSrcList<SortT> sortList(src[b0], src[b1], src[b2], src[b3]);
            uint16_t elementCountList[MRG_WAYS] = {
                static_cast<uint16_t>(runLen), static_cast<uint16_t>(runLen), static_cast<uint16_t>(runLen),
                static_cast<uint16_t>(runLen)};
            AscendC::MrgSort<SortT, false>(dst[b0], sortList, elementCountList, sortedNumDummy, 0b1111, 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        runLen = runLen * MRG_WAYS;
        numRuns = outRuns;
        AscendC::LocalTensor<SortT> t = src;
        src = dst;
        dst = t; // ping-pong
    }
    return src; // fully sorted proposal lives in src
}

template <typename VALUE_DT, typename INDEX_DT, uint32_t SIZE_MODE>
__aicore__ inline void SortWithIndex<VALUE_DT, INDEX_DT, SIZE_MODE>::Compute()
{
    AscendC::LocalTensor<VALUE_DT> xLocal = inQueueX.DeQue<VALUE_DT>();
    AscendC::LocalTensor<INDEX_DT> idxLocal = inQueueIdx.DeQue<INDEX_DT>();
    AscendC::LocalTensor<VALUE_DT> yLocal = outQueueY.AllocTensor<VALUE_DT>();
    AscendC::LocalTensor<INDEX_DT> sortedIdxLocal = outQueueIdx.AllocTensor<INDEX_DT>();

    AscendC::LocalTensor<uint32_t> posLocal = posBuf.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> sortedPosLocal = sortedPosBuf.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    AscendC::LocalTensor<SortT> propA = propABuf.Get<SortT>();
    AscendC::LocalTensor<SortT> concatTmpLocal = concatTmpBuf.Get<SortT>();
    AscendC::LocalTensor<SortT> sortTmpLocal = sortTmpBuf.Get<SortT>();

    // RC1 (perf-opt, KEPT): for the int32-index single-tile path, feed the caller's index DIRECTLY into
    // the Sort index channel (reinterpret int32->uint32, bit-exact) and Extract it straight back as
    // sorted_index -- no separate position channel (CreateVecIndex), no offset Muls, no Gather. The
    // separate position channel + Gather the baseline used was redundant: AscendC Sort<SortT,true> on
    // 910B (DAV_2201) breaks score ties by INPUT ORDER (positional stability), NOT by the value carried
    // in the index channel -- NPU-PROVEN by a controlled tie probe (ALL_SAME + reversed index, and a
    // two-group WITH_TIES + arbitrary index: both stable=true ascending returned sorted_index = index[p]
    // with p in original-position order, byte-identical to the baseline position+Gather kernel) plus the
    // ATK WITH_TIES/ALL_SAME/INT_TIES/INT_ALL_SAME + PERM-index + stable cases (extreme2 58/58, p1 sweep
    // 160/160, exact per-row index check; fault-injection confirmed that check has teeth). So fusing the
    // caller index into the channel does not change tie ordering. (int64 index is not 910b-exposed and
    // keeps the position+double-Gather path; MRGSORT keeps the position channel too -- see below.)
    static constexpr bool kFuseIndex = !kIndexIsInt64 && (SIZE_MODE == SORT_WITH_INDEX_SIZE_MODE_SINGLE);
    // The Sort index channel: caller index (fused) or the position channel (baseline). The Extract dest:
    // sorted_index directly (fused) or sortedPos (baseline). Sized realSortLen (uint32 view).
    AscendC::LocalTensor<uint32_t> sortIdxChannel;
    AscendC::LocalTensor<uint32_t> extractIdxDest;
    if constexpr (kFuseIndex) {
        sortIdxChannel = idxLocal.template ReinterpretCast<uint32_t>();
        extractIdxDest = sortedIdxLocal.template ReinterpretCast<uint32_t>();
    } else {
        // baseline: position channel p = [0..realSortLen-1] (uint32). CreateVecIndex has no native
        // uint32, so generate via int32 view then reinterpret (values < 2^31, bit-identical).
        AscendC::LocalTensor<int32_t> posInt = posLocal.template ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(posInt, static_cast<int32_t>(0), realSortLen);
        sortIdxChannel = posLocal;
        extractIdxDest = sortedPosLocal;
    }

    // 2. build the float/half sort key.
    AscendC::LocalTensor<SortT> sortInput;
    if constexpr (kValueDirect) {
        // half / float: the value buffer IS the sort key (reinterpret; for float it is identity).
        sortInput = xLocal.template ReinterpretCast<SortT>();
    } else {
        // bf16 / int32: Cast to a dedicated float key buffer.
        AscendC::LocalTensor<float> keyF32 = keyBuf.Get<float>();
        if constexpr (kValueIsBf16) {
            // bf16 -> float is lossless (CAST_NONE). The whole realSortLen region (incl. the padded
            // tail bf16 +-inf sentinel, I1) Casts cleanly to float +-inf.
            AscendC::Cast(keyF32, xLocal, AscendC::RoundMode::CAST_NONE, realSortLen);
        } else { // int32 value
            // The int32 in-row sentinel is a placeholder (0). Fill the float key with the real +-inf
            // sentinel (I1) first, then Cast only the valid sliceLen elements (|x| <= 2^24 exact). The
            // inf sentinel lives ONLY in the float key buffer's padded tail; value writeback casts back
            // only the valid sliceLen elements (Cast int32 below), so Cast(int32) never sees inf.
            const float keyPad = descending ? -SENTINEL_INF : SENTINEL_INF;
            AscendC::Duplicate(keyF32, keyPad, realSortLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(keyF32, xLocal, AscendC::RoundMode::CAST_RINT, sliceLen);
        }
        AscendC::PipeBarrier<PIPE_V>();
        sortInput = keyF32;
    }

    // 3. Concat builds the proposal (A2: Concat output reuses the source view). Ascending via Muls(-1).
    AscendC::LocalTensor<SortT> concatLocal;
    AscendC::Concat(concatLocal, sortInput, concatTmpLocal, concatRepeat);
    sortInput.SetSize(realSortLen);
    sortTmpLocal.SetSize(realSortLen);
    if (!descending) {
        AscendC::Muls(concatLocal, concatLocal, static_cast<SortT>(-1), realSortLen);
    }

    // 4. sort + extract sorted value + sorted position (the permutation p).
    AscendC::LocalTensor<SortT> sortedKey;
    if constexpr (kValueDirect) {
        sortedKey = yLocal.template ReinterpretCast<SortT>(); // half/float: extract straight into y
    } else {
        sortedKey = sortedKeyBuf.Get<SortT>(); // bf16/int32: sorted float, Cast later
    }

    if constexpr (SIZE_MODE == SORT_WITH_INDEX_SIZE_MODE_MRGSORT) {
        // Large axis: per-32 sorted runs, then in-core 4-way MrgSort merge. (kFuseIndex is false here:
        // MRGSORT always uses the position channel + Gather.)
        propA.SetSize(realSortLen * kProposalPerElem);
        AscendC::Sort<SortT, false>(propA, concatLocal, sortIdxChannel, sortTmpLocal, sortRepeat);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<SortT> propB = propBBuf.Get<SortT>();
        AscendC::LocalTensor<SortT> merged = MergeRuns(propA, propB, numRuns0, ELEMENT_32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract(sortedKey, extractIdxDest, merged, extractRepeat);
    } else {
        // Single tile: one complete sort across all repeats. With kFuseIndex the index channel carries
        // the caller index directly and Extract writes sorted_index straight out.
        propA.SetSize(realSortLen);
        AscendC::Sort<SortT, true>(propA, concatLocal, sortIdxChannel, sortTmpLocal, sortRepeat);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract(sortedKey, extractIdxDest, propA, extractRepeat);
    }
    if (!descending) {
        AscendC::Muls(sortedKey, sortedKey, static_cast<SortT>(-1), realSortLen);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // 5. value writeback. Direct paths already extracted into yLocal (reinterpret). Cast paths Cast the
    //    sorted float key back to VALUE_DT (bf16 round-trip is bitwise; int32 RINT exact for |x|<=2^24).
    if constexpr (!kValueDirect) {
        if constexpr (kValueIsBf16) {
            AscendC::Cast(yLocal, sortedKey, AscendC::RoundMode::CAST_RINT, realSortLen);
        } else { // int32 value: only the valid sliceLen elements are written back.
            AscendC::Cast(yLocal, sortedKey, AscendC::RoundMode::CAST_RINT, sliceLen);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 6. index follow: sorted_index[k] = index[p[k]].
    if constexpr (kFuseIndex) {
        // RC1 fuse: the Sort index channel already carried the caller index, so Extract above wrote
        // sorted_index directly. Nothing to do here -- no position channel, no offset Muls, no Gather.
        // (Correctness is gated on Sort breaking ties by INPUT ORDER, not by the channel value.)
    } else if constexpr (!kIndexIsInt64) {
        // int32 index (MRGSORT only, when not fused): Gather via the position permutation. Muls has no
        // uint32 overload; compute on the int32 view (positions 0..N-1 are small positives, bit-identical
        // between int32/uint32). Gather only the first sliceLen valid elements.
        AscendC::LocalTensor<int32_t> offsetInt = offsetLocal.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> sortedPosInt = sortedPosLocal.template ReinterpretCast<int32_t>();
        AscendC::Muls(offsetInt, sortedPosInt, static_cast<int32_t>(sizeof(int32_t)), realSortLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(sortedIdxLocal, idxLocal, offsetLocal, static_cast<uint32_t>(0), sliceLen);
        // RC3 (perf-opt): the only real tail dependency is V(Gather writes sortedIdxLocal) -> MTE3
        // (CopyOut DataCopyPad reads it), which the EnQue/DeQue queue sync covers. The old
        // PipeBarrier<PIPE_ALL> drained MTE+VEC+SCALAR maximally (pure serialization tax, ~0.3us,
        // V2 ablation). Scope it to PIPE_V (the minimal barrier the dataflow requires), proven
        // bitwise-correct by the V2 ablation.
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::LocalTensor<int32_t> offsetInt = offsetLocal.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> sortedPosInt = sortedPosLocal.template ReinterpretCast<int32_t>();
        // int64 index: double int32-view Gather. idxLocal viewed as [N x 2] int32 (lo at offset
        // sortedPos*8, hi at sortedPos*8+4); two Gathers then interleaved SetValue writeback.
        AscendC::LocalTensor<uint32_t> offsetHi = offsetHiBuf.Get<uint32_t>();
        AscendC::LocalTensor<int32_t> offsetHiInt = offsetHi.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> gatherLo = gatherLoBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> gatherHi = gatherHiBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> idxI32 = idxLocal.template ReinterpretCast<int32_t>();

        AscendC::Muls(offsetInt, sortedPosInt, static_cast<int32_t>(8), realSortLen);
        AscendC::Adds(offsetHiInt, offsetInt, static_cast<int32_t>(4), realSortLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(gatherLo, idxI32, offsetLocal, static_cast<uint32_t>(0), sliceLen);
        AscendC::Gather(gatherHi, idxI32, offsetHi, static_cast<uint32_t>(0), sliceLen);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<int32_t> sortedIdxI32 = sortedIdxLocal.template ReinterpretCast<int32_t>();
        for (uint32_t k = 0; k < sliceLen; ++k) {
            sortedIdxI32.SetValue(2U * k, gatherLo.GetValue(k));
            sortedIdxI32.SetValue(2U * k + 1U, gatherHi.GetValue(k));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    outQueueY.EnQue(yLocal);
    outQueueIdx.EnQue(sortedIdxLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueIdx.FreeTensor(idxLocal);
}

template <typename VALUE_DT, typename INDEX_DT, uint32_t SIZE_MODE>
__aicore__ inline void SortWithIndex<VALUE_DT, INDEX_DT, SIZE_MODE>::CopyOut(uint32_t rowIdx)
{
    AscendC::LocalTensor<VALUE_DT> yLocal = outQueueY.DeQue<VALUE_DT>();
    AscendC::LocalTensor<INDEX_DT> sortedIdxLocal = outQueueIdx.DeQue<INDEX_DT>();

    // Only the first sliceLen valid elements are written back (padding dropped).
    AscendC::DataCopyExtParams yCopy{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sliceLen * sizeof(VALUE_DT)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[static_cast<uint64_t>(rowIdx) * sliceLen], yLocal, yCopy);
    AscendC::DataCopyExtParams idxCopy{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sliceLen * sizeof(INDEX_DT)), 0, 0, 0};
    AscendC::DataCopyPad(sortedIndexGm[static_cast<uint64_t>(rowIdx) * sliceLen], sortedIdxLocal, idxCopy);

    outQueueY.FreeTensor(yLocal);
    outQueueIdx.FreeTensor(sortedIdxLocal);
}

template <typename VALUE_DT, typename INDEX_DT, uint32_t SIZE_MODE>
__aicore__ inline void SortWithIndex<VALUE_DT, INDEX_DT, SIZE_MODE>::Process()
{
    if constexpr (SIZE_MODE == SORT_WITH_INDEX_SIZE_MODE_EMPTY) {
        return; // empty tensor / N==0 / rowNum==0: nothing to compute.
    }
    if (coreRowNum == 0) {
        return;
    }
    for (uint32_t row = 0; row < coreRowNum; ++row) {
        CopyIn(row);
        Compute();
        CopyOut(row);
    }
}

} // namespace NsSortWithIndex
#endif // SORT_WITH_INDEX_KERNEL_H
