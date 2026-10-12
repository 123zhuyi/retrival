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
 * \file arg_min_with_value_tiling.cpp
 * \brief Host tiling for ArgMinWithValue (A2). Flattens the reduce axis to firstDim x axisSize x lastDim,
 *        picks one of three patterns (COPY / LAST / NLAST), splits the flattened output across the cores,
 *        and sizes each pattern's UB tile so its buffers fit. No magic tile constants — every size is
 *        derived from the actual shape and dtype.
 */
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/broadcast/broadcast_tiling.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/arg_min_with_value_tiling_data.h"
#include "../op_kernel/arg_min_with_value_tiling_key.h"

namespace optiling {
constexpr uint32_t PIECE_AXIS = 4096;  // LAST: max elements reduced per piece (must match kernel)
constexpr uint32_t WORKSET_CAP = 8192; // cap on the per-tile element working set (leaves ample UB headroom)
constexpr uint32_t UB_MARGIN = 16384;  // keep clear of the UB top (guards against tighter eval-env budgets)

struct ArgMinWithValueCompileInfo {};

static uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b * b; }

static uint32_t Gcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a == 0 ? 1 : a;
}

// Default 32B-aligned uneven output-split: distribute outSize elements across up to coreNum cores in
// outAlign-sized blocks so no two cores share a 32B output block, using ALL coreNum cores via an uneven
// split (the first bigCores cores get one extra block) rather than a uniform ceil-round that would
// silently collapse the core count (e.g. 1024/40 -> perCore 32 -> only 32 cores engaged).
static void ComputeOutputSplit(uint64_t outSize, uint32_t outAlign, uint32_t coreNum, uint32_t* used, uint32_t* perCore,
                               uint32_t* bigCores)
{
    *used = 1;
    *perCore = static_cast<uint32_t>(outSize);
    *bigCores = 0;
    if (outSize > 0) {
        uint32_t totalBlk = (static_cast<uint32_t>(outSize) + outAlign - 1) / outAlign; // # of outAlign-blocks
        *used = totalBlk < coreNum ? totalBlk : coreNum;                                // cores = min(blocks, N)
        if (*used == 0)
            *used = 1;
        uint32_t baseBlk = totalBlk / *used; // base blocks per core
        *bigCores = totalBlk % *used;        // first bigCores cores get baseBlk+1 blocks
        *perCore = baseBlk * outAlign;       // BASE per-core size; big cores get perCore+outAlign
    }
}

// Exact number of per-output kernel tiles in the slowest core for a candidate core cap.  NLAST restarts its
// inner-tile loop at every firstDim plane, so outSize/coreNum alone misses both plane boundaries and tile tails.
static uint64_t MaxNLastTiles(uint64_t outSize, uint64_t lastDim, uint32_t innerTile, uint32_t outAlign,
                              uint32_t coreCap, uint32_t* usedOut = nullptr)
{
    uint32_t used, perCore, bigCores;
    ComputeOutputSplit(outSize, outAlign, coreCap, &used, &perCore, &bigCores);
    if (usedOut != nullptr)
        *usedOut = used;
    uint64_t maxTiles = 0;
    for (uint32_t core = 0; core < used; ++core) {
        uint64_t start = core < bigCores ? static_cast<uint64_t>(core) * (perCore + outAlign) :
                                           static_cast<uint64_t>(bigCores) * (perCore + outAlign) +
                                               static_cast<uint64_t>(core - bigCores) * perCore;
        uint64_t len = perCore + (core < bigCores ? outAlign : 0u);
        if (start >= outSize)
            len = 0;
        else if (len > outSize - start)
            len = outSize - start;
        uint64_t first = len == 0 ? 0 : lastDim - start % lastDim;
        if (first > len)
            first = len;
        uint64_t tiles = (first + innerTile - 1u) / innerTile;
        len -= first;
        tiles += (len / lastDim) * ((lastDim + innerTile - 1u) / innerTile);
        uint64_t tail = len % lastDim;
        tiles += (tail + innerTile - 1u) / innerTile;
        if (tiles > maxTiles)
            maxTiles = tiles;
    }
    return maxTiles;
}

static ge::graphStatus ArgTilingFunc(gert::TilingContext* context)
{
    auto td = context->GetTilingData<ArgMinWithValueTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(ArgMinWithValueTilingData), 0, sizeof(ArgMinWithValueTilingData)) != EOK,
                OP_LOGE(context, "memset tiling failed"), return ge::GRAPH_FAILED);

    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = plat.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0 || ubSize == 0, OP_LOGE(context, "platform info invalid"), return ge::GRAPH_FAILED);
    const uint32_t sysWorkspaceSize = plat.GetLibApiWorkSpaceSize();

    // Logical input shape. The framework makes the kernel input contiguous, so origin == storage here.
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t dimNum = xShape.GetDimNum();
    uint32_t typeLen = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLen);
    OP_CHECK_IF(typeLen == 0, OP_LOGE(context, "dtype len 0"), return ge::GRAPH_FAILED);
    const bool useCast = typeLen < sizeof(float);
    const bool isFp16 = context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16;
    const uint32_t outAlign = (typeLen == 2u) ? 16u : 8u;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t dim = *(attrs->GetAttrPointer<int64_t>(0));

    // Flatten to firstDim x axisSize x lastDim.
    uint64_t firstDim = 1, axisSize = 1, lastDim = 1;
    if (dimNum > 0) {
        if (dim < 0)
            dim += static_cast<int64_t>(dimNum);
        OP_CHECK_IF(dim < 0 || dim >= static_cast<int64_t>(dimNum), OP_LOGE(context, "dimension out of range"),
                    return ge::GRAPH_FAILED);
        for (uint32_t i = 0; i < static_cast<uint32_t>(dim); ++i)
            firstDim *= static_cast<uint64_t>(xShape.GetDim(i));
        axisSize = static_cast<uint64_t>(xShape.GetDim(static_cast<uint32_t>(dim)));
        for (uint32_t i = static_cast<uint32_t>(dim) + 1; i < dimNum; ++i)
            lastDim *= static_cast<uint64_t>(xShape.GetDim(i));
    }
    uint64_t outSize = firstDim * lastDim;

    // Pick the pattern.
    uint32_t mode = ARG_MODE_NLAST;
    if (outSize == 0 || axisSize <= 1)
        mode = ARG_MODE_COPY;
    else if (lastDim == 1)
        mode = ARG_MODE_LAST;

    // Route and blockDim overview (later steps may override the earlier output split):
    //
    //   [firstDim, axisSize, lastDim], outSize = firstDim * lastDim
    //        |
    //        +-- outSize == 0 || axisSize <= 1 ---------- COPY
    //        +-- outSize > 0 && axisSize > 1 && lastDim == 1
    //        |                                            LAST
    //        `-- outSize > 0 && axisSize > 1 && lastDim > 1
    //                                                     NLAST
    //
    //   base blockDim = 32B-aligned output split, capped at coreNum
    //        |
    //        +-- COPY:  bytes < 64KB -> 1; otherwise keep base
    //        |
    //        +-- LAST:  native direct ------------------> LAST_DIRECT
    //        |          one-row/core long -------------> LAST_LONG[_PACKED]
    //        |          one output + >=8 useful slices -> SPLIT1
    //        |          few rows + long axis -----------> SPLIT2 (cap partial producers)
    //        |          otherwise: optional fine row split, select PACK/TINY/SEG/PIECE,
    //        |                     then score full vs half launch wave for SEG/PIECE
    //        |
    //        `-- NLAST: batch-capable ------------------> LCM split vs whole-plane split
    //                   |                                  TREE: full grid (smaller width/treeAc)
    //                   |                                  BATCH: full grid iff
    //                   |                                  savedWork >= extraBlocks*768 and
    //                   |                                  (axis>=16 || fullPpc>1); else half grid
    //                   +-- firstDim==1 + one tile -----> axis split when it adds parallelism
    //                   `-- ordinary output path -------> tile-count choice among 1/2/4 waves
    //
    //   The chosen schedule/gather flag forms the tiling key; final used is passed to SetBlockDim().
    // ============================ blockDim (usedCoreNum) STRATEGY ============================
    // usedCoreNum maximizes useful parallelism for each (mode, outSize, axisSize, dtype), bounded by the kernel's
    // split capabilities and the per-core-work versus launch/combine-overhead trade-off. The kernel offers three
    // split shapes with different cost profiles:
    //   [output-split] (mode 0)  partition OUTPUT ROWS; each core reduces (rows/used) rows x FULL axis with
    //                            CONTIGUOUS loads and NO cross-core combine. Cores <= ceil(outSize/outAlign).
    //                            Preferred when rows suffice; few output rows inherently cap the useful core count.
    //   [axis-split-1D] (mode 1) outSize==1 only: PIECE_AXIS-granular slices + a core-0 serial combine. Finer
    //                            slices add cores but also increase combine and SyncAll overhead.
    //   [axis-split-2D] (mode 2) every core reduces ALL rows x a STRIDED slice + SyncAll + per-row combine. The
    //                            Strided loads and the combine restrict it to very few rows with a large axis.
    // Decision flow, in source order: output-split (default, all modes) -> COPY 1-core floor -> single-output
    // axis-split-1D -> multi-output aspect axis-split-2D -> NLAST work-selected whole-plane re-split -> LAST cap.
    // NLAST batch compares the LCM-aligned grid with a whole-plane grid. TREE always evaluates the full
    // AIV grid because narrower per-core batches increase treeAc and remove reduction chunks. Scalar-loop BATCH
    // uses the full grid only when (halfPpc-fullPpc)*axis*alignedInner >= extraBlocks*768; an axis shorter than 16
    // must also retain multiple planes/core. Otherwise the half grid avoids startup for too little saved work.
    // This makes 20/40/80-plane transitions gradual without forcing large blockDim on tiny reductions.
    // LAST's final cap compares the generated SEG/PIECE work on full and half grids: SEG scores aligned rows times
    // chunk-count squared; PIECE scores aligned rows times native/cast chunk count. Thus its blockDim changes with
    // actual kernel work rather than a shape-product threshold. Too few output rows can still cap the contiguous
    // output split below coreNum.
    // ========================================================================================

    // Split the flattened output across cores (see ComputeOutputSplit above). Align so no two cores share a 32B
    // output block. Mode-specific cost models may narrow this default when launch or combine overhead dominates.
    uint32_t used, perCore, bigCores;
    ComputeOutputSplit(outSize, outAlign, coreNum, &used, &perCore, &bigCores);

    // Direct LAST is a separate generated kernel, not a runtime sub-route of ArgLast. Admit every native-width
    // row batch that one MTE2 + one vcmin repeat can consume from the fixed UB layout. Output splitting is 32B
    // aligned, so each core writes its values/indices directly without a workspace gather. The same kernel also
    // owns the single-row long-axis domain, whose fixed 32KB input window supports axis <= 8192.
    constexpr uint32_t DIRECT_AXIS_CAP = 8192u;
    const uint32_t nativeLanes = isFp16 ? 128u : 64u;
    bool directLong = false;
    if (mode == ARG_MODE_LAST) {
        const uint32_t maxRows = perCore + (bigCores != 0u ? outAlign : 0u);
        const uint32_t directWidth = RoundUp(static_cast<uint32_t>(axisSize), outAlign);
        const uint64_t rawBytes = static_cast<uint64_t>(maxRows) * directWidth * typeLen;
        const uint64_t calcBytes = static_cast<uint64_t>(maxRows) * directWidth * sizeof(float);
        const bool nativeBatch = axisSize <= nativeLanes && maxRows <= 255u && rawBytes <= 32u * 1024u &&
                                 (!useCast || calcBytes <= 32u * 1024u);
        // ReduceOneLong owns one fixed 32KB row window per core.  One row/core avoids the aligned output-split,
        // but outSize>coreNum/2 launches a second AIV wave.  Admit that wave only after each row has enough chunks
        // to amortize it: 4 native-half chunks, 4 cast-to-fp32 chunks, or 32 direct-fp32 chunks. These thresholds
        // derive from the three generated compute paths; shorter axes retain the ordinary 32-byte row split.
        const uint32_t secondWaveMinAxis = isFp16 ? 512u : (typeLen == sizeof(float) ? 2048u : 256u);
        const bool oneLongRowPerCore = axisSize > nativeLanes && axisSize <= DIRECT_AXIS_CAP && outSize <= coreNum &&
                                       (outSize <= coreNum / 2u || axisSize >= secondWaveMinAxis);
        if (nativeBatch || oneLongRowPerCore) {
            mode = ARG_MODE_LAST_DIRECT;
        }
        if (oneLongRowPerCore) {
            used = static_cast<uint32_t>(outSize);
            perCore = 1u;
            bigCores = 0u;
        }
        // ComputeOutputSplit rounds a single output up to outAlign, so perCore is 8/16 even though
        // the kernel's effective oLen is one.  Route by effective rows, not the aligned split span.
        directLong = mode == ARG_MODE_LAST_DIRECT && axisSize > nativeLanes &&
                     (outSize == 1u || (perCore == 1u && bigCores == 0u));
    }

    uint32_t schedule = mode == ARG_MODE_COPY ?
                            ARG_SCH_COPY :
                            (mode == ARG_MODE_LAST_DIRECT ?
                                 (directLong ?
                                      ((!isFp16 && axisSize <= 2048u) ? ARG_SCH_LAST_LONG_PACKED : ARG_SCH_LAST_LONG) :
                                      ARG_SCH_LAST_DIRECT) :
                                 ARG_SCH_NLAST_OUTPUT);

    // ===== COPY override: below a byte-size floor, avoid multi-core launch overhead. =====
    // COPY is pure GM->UB->GM movement. Small transfers use one core; larger transfers retain the multi-core split
    // for bandwidth. With one core, perCore=outSize and no two cores can share a 32-byte output block.
    if (mode == ARG_MODE_COPY && outSize > 0) {
        constexpr uint64_t COPY_MULTICORE_MIN_BYTES = 65536u;
        if (static_cast<uint64_t>(outSize) * typeLen < COPY_MULTICORE_MIN_BYTES) {
            perCore = static_cast<uint32_t>(outSize);
            used = 1;
            bigCores = 0;
        }
    }

    // ===== 1D axis-split override (splitAxis=1): single output + huge axis -> engage all cores. =====
    // Single output, large axis: split the reduce axis across cores instead (the output-split would use 1 core).
    // Align each core's slice to PIECE_AXIS so its pieces coincide with the single-core path's pieces -> the
    // cross-core strict-first combine reproduces the single-core result exactly. Split only when at least one full
    // AIV dispatch group is useful; otherwise the serial workspace combine dominates. Route by candidate
    // parallelism rather than an axis-length literal so PIECE/SPLIT1 has no artificial boundary cliff.
    constexpr uint32_t SPLIT_MIN_CORES = 8u;
    uint32_t splitAxis = 0, axisPerCore = 0;
    if (mode == ARG_MODE_LAST && outSize == 1) {
        const uint32_t candidatePer = RoundUp((static_cast<uint32_t>(axisSize) + coreNum - 1) / coreNum, PIECE_AXIS);
        const uint32_t candidateUsed = (static_cast<uint32_t>(axisSize) + candidatePer - 1) / candidatePer;
        if (candidateUsed >= SPLIT_MIN_CORES) {
            splitAxis = 1;
            axisPerCore = candidatePer;
            used = candidateUsed;
            perCore = 1;
            bigCores = 0; // splitAxis uses axisPerCore (ProcessSplit), not the uneven output-split
        }
    }

    // ===== 2D axis-split override (splitAxis=2): multi-output + huge axis + few rows -> engage more cores. =====
    // Multi-output, large axis, few rows: the 32-byte-aligned output split leaves cores idle. Also split the reduce
    // axis across cores so more cores engage: each core reduces [outSize x slice],
    // then a cross-core strict-first combine folds the slices (earlier slice = smaller axis index = wins ties).
    // Use this only when output splitting severely underutilizes the cores and each axis slice remains large enough
    // for efficient MTE2 transfers. Many-row moderate axes retain output splitting and its contiguous row loads.
    constexpr uint32_t SPLIT2D_MIN_AXIS = 2048;
    constexpr uint32_t SPLIT2D_MIN_SLICE_BYTES = 1024;
    // The ordinary PIECE path keeps its running global index in fp32.  Above 2^24 it cannot represent every
    // int32 axis position, so correctness requires the existing SPLIT2 path even when output-split already uses
    // many/all cores: each slice stays fp32-exact and DrainToWs adds the global slice base in int32.
    constexpr uint64_t FP32_EXACT_INDEX_LIMIT = 1ull << 24;
    const bool exactIndexSplit = axisSize > FP32_EXACT_INDEX_LIMIT;
    // SPLIT2 makes every core process every output row, publish used*outSize partials, execute a full-core barrier,
    // and fold all slices. Gate on useful output cores rather than a shape literal; the >2^24 correctness split
    // remains mandatory regardless of performance.
    if (mode == ARG_MODE_LAST && !splitAxis && outSize > 1 && (used < coreNum / 8 || exactIndexSplit) &&
        axisSize >= SPLIT2D_MIN_AXIS) {
        uint32_t per = RoundUp((static_cast<uint32_t>(axisSize) + coreNum - 1) / coreNum, 64u);
        uint32_t u2 = (static_cast<uint32_t>(axisSize) + per - 1) / per;
        // Cross-core SyncAll + workspace combine is a fixed floor.  A 1KB slice pays off only when 2-byte rows
        // fill at least one complete AIV wave; for fewer rows or fp32, require 2KB/core. With only two rows, require
        // enough pieces to amortize the combine.
        const uint32_t minSliceBytes = (typeLen == 2u && outSize >= coreNum) ? SPLIT2D_MIN_SLICE_BYTES :
                                                                               2u * SPLIT2D_MIN_SLICE_BYTES;
        const bool enoughSplitWork = exactIndexSplit ||
                                     (per * typeLen >= minSliceBytes && (outSize > 2u || axisSize >= 12u * PIECE_AXIS));
        if ((u2 > used || exactIndexSplit) && enoughSplitWork) {
            // CombineMulti distributes output in 8-row groups, and every active combine core folds every axis
            // partial for its rows.  More than eight partial producers per active output group only lengthens
            // that serial fold.  Cap producers to 8 * output-groups, then re-slice the full axis for that cap.
            uint32_t splitCoreCap = RoundUp(static_cast<uint32_t>(outSize < coreNum ? outSize : coreNum), 8u);
            if (splitCoreCap > coreNum)
                splitCoreCap = coreNum;
            if (exactIndexSplit) {
                const uint32_t exactCores = static_cast<uint32_t>((axisSize + FP32_EXACT_INDEX_LIMIT - 1u) /
                                                                  FP32_EXACT_INDEX_LIMIT);
                if (splitCoreCap < exactCores)
                    splitCoreCap = exactCores;
                // SyncAll cannot span more logical blocks than physical AIVs.  If even coreNum slices cannot
                // keep every local index fp32-exact, the split kernel retains exact int32 indices across pieces.
                if (splitCoreCap > coreNum)
                    splitCoreCap = coreNum;
            }
            per = RoundUp((static_cast<uint32_t>(axisSize) + splitCoreCap - 1u) / splitCoreCap, 64u);
            u2 = (static_cast<uint32_t>(axisSize) + per - 1u) / per;
            splitAxis = 2;
            axisPerCore = per;
            used = u2;
            bigCores = 0; // 2D split uses axisPerCore + per-row workspace, not the uneven output-split
        }
    }

    // ===== Fine output split (LAST only): do not let 32B output alignment collapse a compute-heavy reduction. =====
    // The raw align-copy commits exactly blockLen bytes, so disjoint row ranges may be split at element granularity.
    // Balance those ranges across every useful core.  Rebalancing an already-full grid adds no launch wave, but its
    // fine MTE3 ranges still need enough benefit: require 8KB saved on the slowest core. Expanding a sparse grid
    // adds launch overhead, so require 16KB saved.
    uint32_t gatherOut = 0;
    const uint32_t fineUsed = outSize == 0 ? 1u : (outSize < coreNum ? static_cast<uint32_t>(outSize) : coreNum);
    const uint64_t alignedRows = perCore + (bigCores != 0u ? outAlign : 0u);
    const uint64_t fineRows = outSize == 0 ? 0u : (outSize + fineUsed - 1u) / fineUsed;
    constexpr uint64_t FINE_REBALANCE_MIN_SAVED_BYTES = 8u * 1024u;
    constexpr uint64_t FINE_EXPAND_MIN_SAVED_BYTES = 16u * 1024u;
    const uint64_t savedCoreBytes = alignedRows > fineRows ? (alignedRows - fineRows) * axisSize * typeLen : 0u;
    const bool rebalanceFullGrid = used == fineUsed && fineRows < alignedRows &&
                                   savedCoreBytes >= FINE_REBALANCE_MIN_SAVED_BYTES;
    const bool expandGrid = used < fineUsed && savedCoreBytes >= FINE_EXPAND_MIN_SAVED_BYTES;
    if (mode == ARG_MODE_LAST && !splitAxis && outSize > 1 && (rebalanceFullGrid || expandGrid)) {
        used = fineUsed;
        perCore = static_cast<uint32_t>(outSize) / used;
        bigCores = static_cast<uint32_t>(outSize) % used; // first bigCores get one row, not one 32B block
        gatherOut = 1;
    }

    // ===== Per-mode UB tile sizing: pick the tile shape (rowTile / innerTile+axisTile / nlBf) that fits ubBudget.
    // =====
    const uint64_t ubBudget = (ubSize > UB_MARGIN) ? (ubSize - UB_MARGIN) : ubSize;
    uint32_t rowTile = 1, innerTile = 64, axisTile = 1, apiTmpSize = 0;
    uint32_t nlBf = 1, nlIPad = 0; // NLAST batch: planes/group + per-plane UB stride (1 = per-output path)

    if (mode == ARG_MODE_LAST_DIRECT) {
        rowTile = static_cast<uint32_t>(outSize);
    } else if (mode == ARG_MODE_COPY) {
        // COPY is bandwidth-bound and uses one input tile plus one zero-index tile. Consume the available UB in
        // one large transfer instead of imposing WORKSET_CAP and issuing several small MTE commands. The raw
        // align-copy blockLen is 21-bit on A2/A3; the UB-sized tile is far below that instruction limit.
        auto bytes = [&](uint32_t t) -> uint64_t {
            return RoundUp(t * typeLen, 32u) + static_cast<uint64_t>(RoundUp(t, 8u)) * sizeof(int32_t);
        };
        uint32_t hi = perCore + (bigCores != 0u ? outAlign : 0u);
        const uint32_t ubElemCap = static_cast<uint32_t>(ubBudget / (typeLen + sizeof(int32_t)));
        if (hi > ubElemCap)
            hi = ubElemCap;
        if (hi == 0u)
            hi = 1u;
        uint32_t lo = 1u;
        rowTile = 1u;
        while (lo <= hi) {
            uint32_t mid = lo + (hi - lo) / 2u;
            if (bytes(mid) <= ubBudget) {
                rowTile = mid;
                lo = mid + 1u;
            } else {
                hi = mid - 1u;
            }
        }
    } else if (mode == ARG_MODE_LAST) {
        // 2D-split: each core reduces an axisPerCore slice (small -> fast seg/tiny path); plain: the full axis.
        const uint32_t axis = (splitAxis == 2) ? axisPerCore : static_cast<uint32_t>(axisSize);
        const uint32_t tinyLim = isFp16 ? 128u : 64u;
        const bool tiny = axis <= tinyLim;
        // SPLIT2 enters ComputeReduce, whose per-piece SEG fast path stops at 256.  The ordinary generated SEG
        // schedule supports 2-byte axes through 512, but treating a 257..512 SPLIT2 slice the same way would omit
        // PieceArgmin scratch from the host UB budget while the kernel still allocates and consumes it.
        const bool seg = axis > tinyLim && (axis <= 256u || (splitAxis != 2u && typeLen == 2u && axis <= 512u));
        const uint32_t pAxis = axis < PIECE_AXIS ? axis : PIECE_AXIS;
        const uint32_t blk = 32u / typeLen; // elements per 32B = load alignment (16 for 2-byte, 8 for fp32)
        const bool micro = (axis == 2);     // axis==2 -> ComputeMicro packed load, 2 elems/row not W=RoundUp(2,blk)
        const bool deint = (axis == 4);     // axis==4 -> ComputeDeinterleave packed load + GatherMask stride-4
        const bool deint3 = (axis == 3) &&
                            ((uint64_t)outSize * axis >= 8192u); // de-interleave only when the load is at least 16KB
        const bool deint5 = (axis == 5) &&
                            ((uint64_t)outSize * axis >= 8192u); // axis==5 de-interleave (gated like axis=3)
        const bool deint678 = (axis == 6 || axis == 7 || (axis == 9 && outSize >= 16384u) ||
                               (axis == 8 && typeLen < 4u && outSize >= 4096u)) &&
                              ((uint64_t)outSize * axis >=
                               8192u); // axis 6/7/8 de-interleave; axis=8 deint only for large outSize (contiguous-load
                                       // wins); small outSize -> tiny (8-way tournament too heavy when compute-bound)
        if (splitAxis == 1u)
            schedule = ARG_SCH_LAST_SPLIT1;
        else if (splitAxis == 2u)
            schedule = ARG_SCH_LAST_SPLIT2;
        else if (micro)
            schedule = ARG_SCH_LAST_PACK2;
        else if (deint3)
            schedule = ARG_SCH_LAST_PACK3;
        else if (deint)
            schedule = ARG_SCH_LAST_PACK4;
        else if (deint5)
            schedule = ARG_SCH_LAST_PACK5;
        else if (deint678)
            schedule = ARG_SCH_LAST_PACKN;
        else if (tiny)
            schedule = ARG_SCH_LAST_TINY;
        else if (seg)
            schedule = ARG_SCH_LAST_SEG;
        else
            schedule = ARG_SCH_LAST_PIECE;
        const uint32_t W = micro ? 2u :
                                   (deint678 ? axis :
                                               (deint5 ? 5u :
                                                         (deint3 ? 3u :
                                                                   (deint ? 4u :
                                                                            ((tiny || seg) ? RoundUp(axis, blk) :
                                                                                             RoundUp(pAxis, 512))))));
        const uint32_t nc = tiny ? 1u : (seg ? (W + 63u) / 64u : W / 64u);
        // Match the kernel's useHalf_ / noSrcBuf_ EXACTLY (UB accounting below depends on it). native-half allocs
        // half chunk scratch (useHalfHost) and drops srcBuf when EVERY piece is half (noSrcBuf) -> frees UB for a
        // larger R (occupancy). A mismatch overflows UB -> eval failure.
        const bool useHalfHost = isFp16 && !tiny && !seg && (W % 1024u == 0u);
        const uint32_t lastP = (axis % PIECE_AXIS == 0u) ? PIECE_AXIS : (axis % PIECE_AXIS);
        uint32_t finalSliceP = lastP;
        if (splitAxis == 2u) {
            const uint64_t finalSliceBase = static_cast<uint64_t>(used - 1u) * axisPerCore;
            const uint32_t finalSliceAxis = static_cast<uint32_t>(axisSize - finalSliceBase);
            finalSliceP = (finalSliceAxis % PIECE_AXIS == 0u) ? PIECE_AXIS : (finalSliceAxis % PIECE_AXIS);
        }
        // Every core shares rowTile.  The final split2 core can have a shorter slice whose tail falls back to
        // fp32 even when full slices are entirely native-half, so size UB for the worst slice.
        // SPLIT1 always converts its local piece through srcBuf even when every full-axis piece is native-half.
        // Keep the host budget identical to ArgLast::Init's `!noSrcBuf_ || SPLIT1` allocation rule.
        const bool noSrcBuf = splitAxis != 1u && useHalfHost && (RoundUp(lastP, 512u) % 1024u == 0u) &&
                              (RoundUp(finalSliceP, 512u) % 1024u == 0u);
        auto broadcastTmp = [&](uint32_t) -> uint32_t {
            return 0u;
        }; // level-2 reduction plus Gather needs no Broadcast
        auto bytes = [&](uint32_t R, uint32_t apiTmp) -> uint64_t {
            uint32_t R8 = RoundUp(R, 64);
            uint32_t rnc = RoundUp(R * nc, 64); // 64-aligned chunk scratch (matches kernel)
            uint64_t loadElems = (micro || deint || deint3 || deint5 || deint678) ?
                                     (RoundUp(axis * R, 128u) + 128u) :
                                     static_cast<uint64_t>(R) * W; // micro/deint packed
            uint64_t total = 2ull * loadElems * typeLen;           // inQ
            if (micro)
                total += static_cast<uint64_t>(R8) * 4;   // inQ
            total += 2ull * R8 * typeLen + 2ull * R8 * 4; // outVal + outIdx
            if (useCast && !noSrcBuf)
                total += loadElems * 4; // srcBuf (skipped for all-piece native-half)
            total += 2ull * rnc * 4;    // redBuf
            total += 2ull * rnc * 4;    // cminBuf + cidxBuf
            if (deint || deint3 || deint5 || deint678)
                total += 10ull * rnc * 4; // col2/col3 + tmpA/tmpB(2*rnc) + 4 const-index tensors
            if (deint678)
                total += 10ull * rnc * 4 + 2ull * RoundUp(axis * R, 64u) * 4 +
                         9ull * (RoundUp(axis * R, 256u) / 8 + 32); // col4-7+const4-7 + genBuf+gen2 + 8 masks
            if (deint5)
                total += 2ull * rnc * 4 + 2ull * RoundUp(5u * R, 64u) * 4 +
                         5ull * (RoundUp(5u * R, 256u) / 8 + 32); // col4+const4 + genBuf+gen2Buf + 5 masks
            if (deint3)
                total += 2ull * RoundUp(3u * R, 64u) * 4 +
                         3ull * (RoundUp(3u * R, 256u) / 8 + 32); // genBuf+gen2Buf + 3 bit-masks
            total += RoundUp(rnc, 256) / 8 + 32;                  // maskBuf
            if (!tiny)
                total += 4ull * R8 * 4; // accVal/accIdx/pieceVal/pieceIdx
            if (!tiny && !seg) {
                total += 3ull * R8 * 4; // wchBuf + offsBuf + glocBuf for index reconstruction
                if (useHalfHost)
                    total += 4ull * rnc; // cminHBuf + cidxHBuf (half)
            }
            (void)apiTmp;
            return total;
        };
        // Tiny splits vcmin/vcmax internally into <=255-repeat chunks, so let one tile consume the full per-core
        // row range whenever UB permits. The only remaining descriptor limit is MTE2 blockCount<=4095. This avoids
        // repeating the complete MTE2 -> reduce -> MTE3 chain merely because R*W crossed the generic workset cap.
        // Seg/piece still have an R<=255 second-stage reduce and keep their bounded working-set policy.
        const uint32_t maxCoreRows = perCore + (bigCores != 0u ? (gatherOut ? 1u : outAlign) : 0u);
        uint32_t hi = tiny ? (maxCoreRows < 4095u ? maxCoreRows : 4095u) :
                             (static_cast<uint32_t>(outSize) < 255u ? static_cast<uint32_t>(outSize) : 255u);
        if (!tiny) {
            const uint32_t wsCap = noSrcBuf ? WORKSET_CAP * 8u : WORKSET_CAP * 4u;
            if (static_cast<uint64_t>(hi) * W > wsCap)
                hi = wsCap / W > 0 ? wsCap / W : 1;
        }
        if (hi == 0)
            hi = 1;
        uint32_t lo = 1;
        rowTile = 1;
        while (lo <= hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t bt = broadcastTmp(mid);
            if (bytes(mid, bt) <= ubBudget) {
                rowTile = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (micro || deint || deint3 || deint5 || deint678) {
            rowTile = (rowTile / blk) * blk;
            if (rowTile == 0)
                rowTile = blk;
        }
        apiTmpSize = broadcastTmp(rowTile);
        OP_CHECK_IF(bytes(rowTile, apiTmpSize) > ubBudget, OP_LOGE(context, "LAST UB layout does not fit"),
                    return ge::GRAPH_FAILED);
    } else if (mode == ARG_MODE_NLAST) {
        // fp16 with axis<=2048 reduces natively in fp16 (no cast scratch). 128-align so both lane widths fit.
        const bool nativeNLast = isFp16 && axisSize <= 2048;
        innerTile = RoundUp(static_cast<uint32_t>(lastDim < 1024 ? lastDim : 1024), 128);
        auto bytes = [&](uint32_t iLen, uint32_t aCap) -> uint64_t {
            uint64_t total = 2ull * aCap * iLen * typeLen;    // inQ
            total += 2ull * iLen * typeLen + 2ull * iLen * 4; // outVal + outIdx
            if (useCast && !nativeNLast)
                total += static_cast<uint64_t>(aCap) * iLen * 4; // srcBuf
            total += 3ull * iLen * 4;                            // curV/idxA/idxB
            total += RoundUp(iLen, 256) / 8 + 32;                // maskBuf
            return total;
        };
        while (innerTile > 64 && bytes(innerTile, 1) > ubBudget)
            innerTile -= 64;
        OP_CHECK_IF(bytes(innerTile, 1) > ubBudget, OP_LOGE(context, "NLAST UB layout does not fit"),
                    return ge::GRAPH_FAILED);
        uint32_t hi = static_cast<uint32_t>(axisSize);
        uint32_t capByWs = WORKSET_CAP / innerTile;
        if (capByWs == 0)
            capByWs = 1;
        if (hi > capByWs)
            hi = capByWs;
        uint32_t lo = 1;
        axisTile = 1;
        while (lo <= hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (bytes(innerTile, mid) <= ubBudget) {
                axisTile = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        // Small-lastDim batch: reduce several consecutive firstDim planes per group so one extremum combine
        // covers them all, amortizing the scalar axis loop across nlBf planes.
        // Each plane pads to iPad (32B); only worth it when >=2 planes fit one group (i.e. lastDim is small).
        const uint32_t blk = 32u / typeLen;
        const uint32_t iPad = RoundUp(static_cast<uint32_t>(lastDim), blk);
        constexpr uint32_t TARGET_W = 1024;           // proven steady-state group width
        constexpr uint32_t NATIVE_TWO_PLANE_W = 1152; // permits two native-half planes through padded width 576
        const uint32_t batchWidthLimit = nativeNLast ? NATIVE_TWO_PLANE_W : TARGET_W;
        auto batchBytes = [&](uint32_t W) -> uint64_t {
            uint64_t total = 4ull * W * typeLen; // inQ(2x) + outVal(2x)
            total += 2ull * W * 4;               // outIdx(2x)
            if (useCast && !nativeNLast)
                total += static_cast<uint64_t>(W) * 4; // srcBuf
            total += 3ull * W * 4;                     // curV/idxA/idxB
            total += RoundUp(W, 256) / 8 + 32;         // maskBuf
            return total;
        };
        if (firstDim >= 2 && iPad <= batchWidthLimit / 2u) {
            uint32_t bf = TARGET_W / iPad;
            if (bf < 2u)
                bf = 2u;
            if (bf > static_cast<uint32_t>(firstDim))
                bf = static_cast<uint32_t>(firstDim);
            while (bf > 1 && batchBytes(RoundUp(bf * iPad, 128)) > ubBudget)
                bf--;
            if (bf >= 2) {
                nlBf = bf;
                nlIPad = iPad;
                axisTile = 1; // batch path loads one axis row (bf planes) per iteration
            }
        }
    }

    // ===== NLAST batch override: re-split by whole planes so the batch path engages more cores. =====
    // NLAST batch assigns whole planes to each core. StoreOut writes each plane with an exact-length raw MTE3,
    // so adjacent cores writing adjacent planes never RMW a shared 32B block. Two ways to split planes
    // across cores -- take whichever engages MORE cores:
    //   old: lcm(lastDim, outAlign)-aligned chunks. Best when lastDim shares factors with 8 (small firstDim can
    //        get 1 plane/core); for lastDim coprime to 8, lcm == lastDim*8 collapses onto firstDim/8 cores.
    //   new: whole-plane chunks. TREE benefits from the smallest per-core width because it raises treeAc and removes
    //        reduction chunks. BATCH uses the full grid only when the saved slowest-core work amortizes extra blocks.
    // The selected whole-plane candidate replaces the LCM grid only when it engages at least as many cores.
    if (nlBf > 1) {
        uint32_t planeAlign = static_cast<uint32_t>(lastDim) / Gcd(static_cast<uint32_t>(lastDim), outAlign) * outAlign;
        uint32_t oldPerCore = RoundUp((static_cast<uint32_t>(outSize) + coreNum - 1) / coreNum, planeAlign);
        uint32_t oldUsed = (static_cast<uint32_t>(outSize) + oldPerCore - 1) / oldPerCore;
        const uint32_t halfCap = coreNum > 1u ? coreNum / 2u : 1u;
        const uint32_t halfPpc = (static_cast<uint32_t>(firstDim) + halfCap - 1u) / halfCap;
        const uint32_t halfUsed = (static_cast<uint32_t>(firstDim) + halfPpc - 1u) / halfPpc;
        const uint32_t fullPpc = (static_cast<uint32_t>(firstDim) + coreNum - 1u) / coreNum;
        const uint32_t fullUsed = (static_cast<uint32_t>(firstDim) + fullPpc - 1u) / fullPpc;
        const bool treeSchedule = axisSize >= 64u || (axisSize >= 16u && lastDim % (32u / typeLen) == 0u);
        // One extra logical block must save at least three quarters of a steady-state batch width of element visits.
        // For axis<16, also require multiple planes/core: one-plane scalar batches do not amortize the extra blocks.
        constexpr uint64_t MIN_SAVED_BATCH_WORK_PER_BLOCK = 768u; // 3 * TARGET_W / 4
        const uint64_t savedBatchWork = static_cast<uint64_t>(halfPpc - fullPpc) * axisSize * nlIPad;
        const uint32_t extraBlocks = fullUsed > halfUsed ? fullUsed - halfUsed : 0u;
        const bool batchParallelWork = axisSize >= 16u || fullPpc > 1u;
        const bool fullGridWins = treeSchedule || (extraBlocks != 0u && batchParallelWork &&
                                                   savedBatchWork >= static_cast<uint64_t>(extraBlocks) *
                                                                         MIN_SAVED_BATCH_WORK_PER_BLOCK);
        const uint32_t planeCoreCap = fullGridWins ? coreNum : halfCap;
        uint32_t ppc = (static_cast<uint32_t>(firstDim) + planeCoreCap - 1u) / planeCoreCap;
        uint32_t newPerCore = ppc * static_cast<uint32_t>(lastDim);
        uint32_t newUsed = (static_cast<uint32_t>(firstDim) + ppc - 1) / ppc;
        if (newUsed >= oldUsed) { // whole-plane engages >= cores (fixes coprime-lastDim collapse)
            perCore = newPerCore;
            used = newUsed;
            // Keep the generated batch/tree schedule when one plane is assigned per core. The kernel uses
            // bn=1 for that core and coreBn=1 shrinks all dynamic UB allocations; nlBf remains a schedule flag.
            if (ppc > 1u && nlBf > ppc)
                nlBf = ppc;
        } else { // lcm split already engages more cores (small firstDim, lastDim shares factors with 8)
            perCore = oldPerCore;
            used = oldUsed;
        }
        if (used == 0)
            used = 1;
        bigCores = 0; // NLAST batch uses a uniform plane-based perCore
    }

    // ===== NLAST axis-split override: large axis + one-tile output + firstDim==1 -> split AXIS across cores. =====
    // The ordinary NLAST path already vectorizes across all lastDim output columns, so splitting by output column
    // would discard that SIMD parallelism. Instead, split the axis while every core retains all output columns,
    // then combine the partial values and indices across cores. Scope this route to firstDim==1 so RowBase's plane
    // stride is always zero and no separate full-axis value is required for GM addressing.
    // ProcessAxisSplit reduces every output column in one call, so its structural limit is the selected innerTile,
    // not the physical core count.  Comparing candidate `u` with the default output-split `used` makes the route
    // self-limiting: once output splitting exposes at least as much parallelism, it wins without SyncAll/combine.
    // This removes the artificial lastDim 39->40 cliff while preserving the one-tile kernel contract.
    if (mode == ARG_MODE_NLAST && nlBf <= 1 && firstDim == 1 && outSize > 1 && lastDim <= innerTile) {
        uint32_t per = RoundUp((static_cast<uint32_t>(axisSize) + coreNum - 1) / coreNum, 64u);
        uint32_t u = (static_cast<uint32_t>(axisSize) + per - 1) / per;
        if (u > used) {
            splitAxis = 3; // NLAST axis-split (LAST uses 1/2; mode context disambiguates in the kernel)
            axisPerCore = per;
            used = u;
            perCore = static_cast<uint32_t>(outSize); // every core still handles ALL outSize columns
            bigCores = 0;
        }
    }

    // ===== NLAST launch-wave cap: compare the kernel's real tile loops for 1/2/4 AIV waves. =====
    // A shape-product threshold creates cliffs because adding one output may add a whole per-core tile. Count the
    // exact slowest-core tiles (including firstDim plane restarts), then spend a launch-overhead budget on
    // the extra serial tiles of a smaller grid. Cast has twice the vector lane work and therefore a tighter budget.
    if (mode == ARG_MODE_NLAST && nlBf <= 1 && splitAxis == 0) {
        const uint32_t halfCap = coreNum > 1u ? coreNum / 2u : 1u;
        const uint32_t quarterCap = coreNum > 3u ? coreNum / 4u : 1u;
        uint32_t fullUsed;
        const uint64_t fullTiles = MaxNLastTiles(outSize, lastDim, innerTile, outAlign, coreNum, &fullUsed);
        const uint64_t halfTiles = MaxNLastTiles(outSize, lastDim, innerTile, outAlign, halfCap);
        const bool nativeNLast = isFp16 && axisSize <= 2048u;
        const uint64_t axisCost = axisSize + 4u; // Compare/select rows plus per-tile init/load/drain floor.
        const uint64_t halfExtra = halfTiles > fullTiles ? halfTiles - fullTiles : 0u;
        const uint64_t halfBudget = nativeNLast ? 20u : 12u;
        const bool halfWins = fullUsed <= halfCap || halfExtra == 0u ||
                              (axisCost <= halfBudget && halfExtra <= halfBudget / axisCost);
        uint32_t cap = coreNum;
        if (halfWins) {
            cap = halfCap;
            const uint64_t quarterTiles = MaxNLastTiles(outSize, lastDim, innerTile, outAlign, quarterCap);
            const uint64_t quarterExtra = quarterTiles > halfTiles ? quarterTiles - halfTiles : 0u;
            const uint64_t quarterBudget = nativeNLast ? 12u : 8u;
            if (quarterTiles <= 4u &&
                (quarterExtra == 0u || (axisCost <= quarterBudget && quarterExtra <= quarterBudget / axisCost)))
                cap = quarterCap;
        }
        if (cap >= 1u && used > cap) {
            uint32_t totalBlk = (static_cast<uint32_t>(outSize) + outAlign - 1) / outAlign;
            used = totalBlk < cap ? totalBlk : cap;
            if (used == 0)
                used = 1;
            uint32_t baseBlk = totalBlk / used;
            bigCores = totalBlk % used;
            perCore = baseBlk * outAlign;
        }
    }

    // ===== LAST launch-wave cap: compare generated-schedule work against the second-wave launch cost. =====
    // Compare the aligned half-grid row span using the generated reduction strategy: SEG-A pays GatherMask, SEG-B
    // uses two reductions per chunk, while PIECE batches chunks and can use native 128-lane half.
    if (mode == ARG_MODE_LAST && splitAxis == 0 && outSize > 0 && !gatherOut) {
        const uint32_t halfCap = coreNum > 1u ? coreNum / 2u : 1u;
        if (used > halfCap) {
            uint32_t halfUsed, halfPer, halfBig;
            ComputeOutputSplit(outSize, outAlign, halfCap, &halfUsed, &halfPer, &halfBig);
            const uint64_t halfRows = halfPer + (halfBig != 0u ? outAlign : 0u);
            bool halfWins = false;
            if (schedule == ARG_SCH_LAST_SEG) {
                const uint64_t w = RoundUp(static_cast<uint32_t>(axisSize), outAlign);
                const uint64_t nc = (w + 63u) / 64u;
                const uint64_t score = halfRows * nc * nc;
                const uint64_t budget = nc < 4u ? 320u : (typeLen == 2u ? 1536u : 1024u);
                halfWins = score <= budget;
            } else if (schedule == ARG_SCH_LAST_PIECE) {
                const uint32_t pAxis = axisSize < PIECE_AXIS ? static_cast<uint32_t>(axisSize) : PIECE_AXIS;
                const uint64_t w = RoundUp(pAxis, 512u);
                const uint64_t nc = isFp16 && w % 1024u == 0u ? w / 128u : w / 64u;
                halfWins = halfRows * nc <= 448u;
            }
            if (halfWins) {
                used = halfUsed;
                perCore = halfPer;
                bigCores = halfBig;
            }
        }
    }

    if (mode == ARG_MODE_NLAST) {
        if (splitAxis == 3u) {
            schedule = ARG_SCH_NLAST_SPLIT;
        } else if (nlBf > 1u) {
            const bool useTree = axisSize >= 64u || (axisSize >= 16u && lastDim % (32u / typeLen) == 0u);
            schedule = useTree ? ARG_SCH_NLAST_TREE : ARG_SCH_NLAST_BATCH;
        } else {
            schedule = ARG_SCH_NLAST_OUTPUT;
        }
    }

    td->tilingMode = schedule;
    td->firstDim = static_cast<uint32_t>(firstDim);
    td->axisSize = static_cast<uint32_t>(axisSize);
    td->lastDim = static_cast<uint32_t>(lastDim);
    td->outSize = static_cast<uint32_t>(outSize);
    td->usedCoreNum = used;
    td->perCore = perCore;
    td->bigCores = bigCores;
    td->rowTile = rowTile;
    td->innerTile = innerTile;
    td->axisTile = axisTile;
    td->apiTmpSize = apiTmpSize;
    td->splitAxis = splitAxis;
    td->axisPerCore = axisPerCore;
    td->nlBf = nlBf;
    td->nlIPad = nlIPad;
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    constexpr uint32_t SLOT_BYTES = 32; // per-core partial slot (value + index), must match the kernel
    // splitAxis==1 (single out): used*32B*2. splitAxis==2 (LAST 2D, per-core x per-row) / splitAxis==3 (NLAST
    // axis-split, per-core x per-column): used * RoundUp(outSize,8) float values + same-size int32 indices (each
    // core's block is 32B-aligned so cores never share a 32B GM block).
    const size_t valStride2d = RoundUp(static_cast<uint32_t>(outSize), 8u);
    const size_t userWorkspace = (splitAxis == 2 || splitAxis == 3) ?
                                     static_cast<size_t>(used) * valStride2d * sizeof(float) * 2 :
                                     (splitAxis ? static_cast<size_t>(used) * SLOT_BYTES * 2 : 0);
    // Only cross-core split kernels consume GetUserWorkspace(). Their generated wrapper keeps
    // SetSysWorkspaceForce, so reserve the framework prefix before the user partials. Every other tiling key
    // compiles SetSysWorkspaceForce to a no-op and never reads workspace; report zero for those paths as well.
    ws[0] = splitAxis ? userWorkspace + static_cast<size_t>(sysWorkspaceSize) : 0;
    context->SetBlockDim(used);

    context->SetTilingKey(GET_TPL_TILING_KEY(schedule, gatherOut != 0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForArgMinWithValue([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ArgMinWithValue)
    .Tiling(ArgTilingFunc)
    .TilingParse<ArgMinWithValueCompileInfo>(TilingParseForArgMinWithValue);
} // namespace optiling
