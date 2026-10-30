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
 * \file sort_with_index_tiling.cpp
 * \brief SortWithIndex host tiling (ascend910b, DAV_2201).
 *
 * Full 3-D TilingKey dispatch (VALUE_DT/INDEX_DT via DTYPE_* macros + SIZE_MODE via
 * ASCENDC_TPL schMode). SIZE_MODE selection:
 *   - 0 SINGLE_TILE : realSortLen = ceil(N,32)*32 <= single-tile budget; one full Sort per row.
 *   - 1 MRGSORT     : large axis. realSortLen = 32 * (power of 4) >= ceil(N,32)*32 so the run count
 *                     (realSortLen/32) is a power of 4 -> strict 4-way in-core MrgSort merge.
 *   - 2 EMPTY       : empty tensor / N==0 / rowNum==0.
 * bytesPerElem is computed from the value/index dtypes (int32-value & int64-index are the heaviest)
 * to derive the per-row UB budget; the single-tile limit is the min of that budget and the Sort
 * full-sort hardware cap (32*255), floored to a multiple of 32.
 *
 * Slicing model: sort is along the last axis only. Rows = prod(shape[:-1]); each row is one
 * contiguous sort slice of length N = shape[-1]. Rows are distributed across cores (big/small);
 * rows never cross cores.
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "sort_with_index_tiling_compile_info.h"
#include "../op_kernel/sort_with_index_tiling_data.h"
#include "../op_kernel/sort_with_index_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32U; // Sort granularity (realSortLen = ceil(N, 32) * 32)
constexpr uint32_t ELEMENT_16 = 16U; // DataCopyPad tail alignment (16 elements = 32B for half)
constexpr uint32_t WS_SYS_SIZE = 512U;
constexpr uint32_t MRG_WAYS = 4U;                // MrgSort fixed 4-way (numRuns must be a power of 4)
constexpr uint32_t SORT_FULLSORT_CAP = 8160U;    // Sort full-sort hardware cap = 32 * 255
constexpr uint32_t PROPOSAL_BYTES_PER_ELEM = 8U; // proposal record = 8B/element (half & float)

// Size (Bytes) of a value dtype as it sits in UB I/O buffers.
static uint32_t ValueDtypeBytes(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        default:
            return 4U;
    }
}

// Size (Bytes) of an index dtype.
static uint32_t IndexDtypeBytes(ge::DataType dt)
{
    return (dt == ge::DT_INT64) ? 8U : 4U;
}

// Get platform info (ubSize, coreNum). Core number is taken at runtime, never hard-coded.
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// Round up to a multiple of 32 whose 32-block count is a power of 4 (>= ceil(N,32)*32). Keeps the
// MrgSort run count (realSortLen/32) a power of 4 so every merge round is strictly 4-way.
static uint32_t ComputeMrgsortRealLen(uint32_t sliceLen)
{
    const uint32_t base = ((sliceLen + BLOCK_SIZE - 1U) / BLOCK_SIZE) * BLOCK_SIZE; // ceil(N,32)*32
    uint32_t blocks = base / BLOCK_SIZE;
    uint32_t pow4 = 1U;
    while (pow4 < blocks) {
        pow4 *= MRG_WAYS;
    }
    return pow4 * BLOCK_SIZE;
}

// Resolved shape + attributes for one tiling call.
struct SortShapeInfo {
    uint32_t sliceLen; // last-axis length N
    uint32_t rowNum;   // product of all preceding dims
    uint32_t normAxis; // normalized sort axis (always rank-1)
    bool descending;
    bool stable;
};

// Resolve shape (sliceLen/rowNum), attributes (axis/descending/stable) and validate the axis.
// Only last-axis sort is supported, so the normalized axis must equal rank-1.
static ge::graphStatus ResolveShapeAndAxis(gert::TilingContext* context, SortShapeInfo& info)
{
    const auto* storageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape& shape = storageShape->GetStorageShape();
    const uint32_t rank = shape.GetDimNum();

    // sliceLen = last-axis length; rowNum = product of all preceding dims.
    info.sliceLen = (rank > 0) ? static_cast<uint32_t>(shape.GetDim(rank - 1)) : 1U;
    info.rowNum = 1U;
    for (uint32_t i = 0; i + 1 < rank; ++i) {
        info.rowNum *= static_cast<uint32_t>(shape.GetDim(i));
    }

    int32_t axis = -1;
    info.descending = false;
    info.stable = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* axisPtr = attrs->GetInt(0);
        if (axisPtr != nullptr) {
            axis = static_cast<int32_t>(*axisPtr);
        }
        const bool* descPtr = attrs->GetBool(1);
        if (descPtr != nullptr) {
            info.descending = *descPtr;
        }
        const bool* stablePtr = attrs->GetBool(2);
        if (stablePtr != nullptr) {
            info.stable = *stablePtr;
        }
    }

    // axis range check + normalize. Only the last dim is supported.
    const int32_t signedRank = static_cast<int32_t>(rank == 0 ? 1 : rank);
    if (axis < -signedRank || axis >= signedRank) {
        OP_LOGE(context, "axis value is out of range");
        return ge::GRAPH_FAILED;
    }
    const int32_t normAxis = axis < 0 ? axis + signedRank : axis;
    if (normAxis != signedRank - 1) {
        OP_LOGE(context, "only last-axis sort is supported (axis must be -1 or rank-1)");
        return ge::GRAPH_FAILED;
    }
    info.normAxis = static_cast<uint32_t>(normAxis);
    return ge::GRAPH_SUCCESS;
}

// Per-element UB footprint (Bytes) over the realSortLen-sized buffers. int32-value (Cast path ->
// +8B float key buffers) and int64-index (+12B double-view Gather) are the heaviest. propB(8B) is
// only allocated in MrgSort, so it does NOT count here (it is added on top in SelectSizeMode).
// The Sort/Concat tmp+concat buffers are never read on DAV_2201 (Sort->Sort32 no tmp, Concat->
// concat=src), so they are sized at the proposal record size (8B/elem each).
static uint32_t ComputeBytesPerElem(ge::DataType valueDt, ge::DataType indexDt)
{
    const uint32_t vBytes = ValueDtypeBytes(valueDt);
    const uint32_t iBytes = IndexDtypeBytes(indexDt);
    const bool valueCastPath = (valueDt == ge::DT_BF16 || valueDt == ge::DT_INT32);
    uint32_t bytesPerElem = 2U * vBytes                     // inQueueX + outQueueY
                            + 2U * iBytes                   // inQueueIdx + outQueueIdx
                            + 12U                           // posBuf + sortedPosBuf + offsetBuf (3 * uint32)
                            + PROPOSAL_BYTES_PER_ELEM       // propABuf (8B/elem)
                            + 2U * PROPOSAL_BYTES_PER_ELEM; // concatTmp + sortTmp (8B/elem each)
    if (valueCastPath) {
        bytesPerElem += 2U * 4U; // keyBuf + sortedKeyBuf (float)
    }
    if (indexDt == ge::DT_INT64) {
        bytesPerElem += 3U * 4U; // offsetHi + gatherLo + gatherHi
    }
    return bytesPerElem;
}

// Compute the single-tile + MrgSort UB budgets, then select SIZE_MODE and realSortLen.
// Single-tile UB budget = ~85% of UB. Single-tile does NOT allocate propB (MrgSort-only), so its
// real per-row footprint at the documented N-upper-bounds is ~78-80% of UB; the 85% budget keeps
// the actual allocation safely below UB. The MrgSort budget stays at 75% because it additionally
// holds propB and needs the headroom.
static ge::graphStatus SelectSizeMode(
    gert::TilingContext* context, uint32_t sliceLen, uint64_t ubSize, uint32_t bytesPerElem, uint32_t& sizeMode,
    uint32_t& realSortLen)
{
    if (bytesPerElem == 0U) {
        OP_LOGE(context, "bytesPerElem is 0");
        return ge::GRAPH_FAILED;
    }
    // Single-tile budget: floor to a multiple of 32; cap by the Sort full-sort hardware limit (32*255).
    uint32_t ubBudgetElems = static_cast<uint32_t>((ubSize * 17U / 20U) / bytesPerElem); // 85%
    ubBudgetElems = (ubBudgetElems / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t singleTileLimit = ubBudgetElems < SORT_FULLSORT_CAP ? ubBudgetElems : SORT_FULLSORT_CAP;
    if (singleTileLimit < BLOCK_SIZE) {
        singleTileLimit = BLOCK_SIZE; // always allow at least one 32-block
    }

    // MrgSort additionally allocates propB (ping-pong, proposal size = 8B/elem). The largest MrgSort
    // realSortLen (a 32*power-of-4 value) that still fits UB is bounded by this.
    const uint32_t mrgBytesPerElem = bytesPerElem + PROPOSAL_BYTES_PER_ELEM;               // + propB (8B/elem)
    uint32_t mrgBudgetElems = static_cast<uint32_t>((ubSize * 3U / 4U) / mrgBytesPerElem); // 75%
    // Largest 32*power-of-4 <= mrgBudgetElems (the only realSortLen values MrgSort can emit).
    uint32_t mrgRealLenCap = BLOCK_SIZE; // 32 (1 block) always fits
    while (mrgRealLenCap * MRG_WAYS <= mrgBudgetElems) {
        mrgRealLenCap *= MRG_WAYS;
    }

    const uint32_t singleRealLen = ((sliceLen + BLOCK_SIZE - 1U) / BLOCK_SIZE) * BLOCK_SIZE; // ceil(N,32)*32
    if (singleRealLen <= singleTileLimit) {
        sizeMode = SORT_WITH_INDEX_SIZE_MODE_SINGLE;
        realSortLen = singleRealLen;
        return ge::GRAPH_SUCCESS;
    }

    sizeMode = SORT_WITH_INDEX_SIZE_MODE_MRGSORT;
    realSortLen = ComputeMrgsortRealLen(sliceLen); // 32 * power-of-4 >= ceil(N,32)*32
    // In-core MrgSort UB ceiling: a single core holds the whole row's proposal buffers. If the
    // required realSortLen exceeds what UB can hold, the in-core merge cannot run (GM-workspace
    // tiling is a future enhancement). Fail tiling explicitly instead of overflowing UB on board:
    // an over-budget realSortLen (e.g. 8192) makes propA+propB+tmp+concat+I/O exceed the 192KB UB,
    // which causes a 507035 "MPU address access invalid" aivec exception on board.
    OP_CHECK_IF(
        realSortLen > mrgRealLenCap,
        OP_LOGE(
            context,
            "sort axis N=%u (realSortLen=%u) exceeds the in-core MrgSort UB ceiling "
            "(max realSortLen=%u for this dtype). Large-axis GM-workspace tiling is not "
            "yet supported.",
            sliceLen, realSortLen, mrgRealLenCap),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Compute padding (padLen/align8/dupCount), distribute rows across cores (big/small) and fill all
// scalar TilingData fields. Returns the launched core count in workCoreNum (== SetBlockDim value).
//
// padLen pads the DataCopyPad tail to the next 32B boundary IN THE VALUE DTYPE (elemsPer32B =
// 32/sizeof(VALUE_DT): 16 for 2-byte fp16/bf16, 8 for 4-byte fp32/int32). When blockLen
// (= N*sizeof) is not 32B-aligned, DataCopyPad with rightPadding==0 dummy-fills the seam with the
// FIRST element instead of the sentinel, corrupting the sorted result (small N can lose the real
// extremes). padLen = (elemsPer32B - N%elemsPer32B) % elemsPer32B guarantees rightPadding != 0
// whenever N is not 32B-aligned, so the framework dummy-fill writes the sentinel. The dtype-
// dependent elemsPer32B also keeps rightPadding * sizeof <= 32B (the DataCopyPad rightPadding
// hardware limit; e.g. fp32 N=1: padLen=7 -> 28B; fp32 N=4: padLen=4 -> 16B). A fixed 16-element
// pad would make rightPadding 60B/48B for small N on 4-byte dtypes, above the 32B limit, which
// triggers a 507035 AIV MPU access exception.
static ge::graphStatus FillCoreAndShapeFields(
    gert::TilingContext* context, SortWithIndexTilingData* tiling, const SortShapeInfo& info, uint32_t vBytes,
    uint32_t realSortLen, int64_t coreNum, uint32_t& workCoreNum)
{
    if (vBytes == 0U) {
        OP_LOGE(context, "value dtype size is 0");
        return ge::GRAPH_FAILED;
    }
    const uint32_t elemsPer32B = BLOCK_SIZE / vBytes; // 16 (2-byte) or 8 (4-byte)
    const uint32_t padLen = (elemsPer32B - (info.sliceLen % elemsPer32B)) % elemsPer32B;
    const uint32_t align8 = info.sliceLen + padLen; // 32B-aligned upper bound of the DataCopyPad tail
    const uint32_t dupCount = realSortLen - align8;

    // multi-core row distribution (big/small core). std::max(1U, ...) keeps workCoreNum >= 1 so the
    // divisions below are well-defined; coreNum is already > 0 (GetPlatformInfo rejects coreNum<=0).
    workCoreNum = std::max(1U, std::min(static_cast<uint32_t>(coreNum), std::max(1U, info.rowNum)));
    if (workCoreNum == 0U) {
        OP_LOGE(context, "workCoreNum is 0");
        return ge::GRAPH_FAILED;
    }
    const uint32_t smallCoreRowNum = info.rowNum / workCoreNum;
    const uint32_t remainder = info.rowNum % workCoreNum; // big cores take one extra row
    const uint32_t bigCoreRowNum = smallCoreRowNum + (remainder == 0U ? 0U : 1U);
    const uint32_t bigCoreNum = remainder; // first 'remainder' cores are big
    const uint32_t smallCoreNum = workCoreNum - bigCoreNum;

    tiling->smallCoreRowNum = smallCoreRowNum;
    tiling->bigCoreRowNum = bigCoreRowNum;
    tiling->smallCoreNum = smallCoreNum;
    tiling->bigCoreNum = bigCoreNum;
    tiling->validCoreNum = workCoreNum;
    tiling->rowNum = info.rowNum;
    tiling->sliceLen = info.sliceLen;
    tiling->realSortLen = realSortLen;
    tiling->align8 = align8;
    tiling->padLen = padLen;
    tiling->dupCount = dupCount;
    // tileLen = per-32-element sorted-run length carrier; tileCntPerRow = number of 32-element runs.
    // Single tile merges in one full Sort (kernel uses Sort<true>); MrgSort merges realSortLen/32 runs.
    tiling->tileLen = BLOCK_SIZE;
    tiling->tileCntPerRow = realSortLen / BLOCK_SIZE;
    tiling->axis = info.normAxis;
    tiling->descending = info.descending;
    tiling->stable = info.stable;
    return ge::GRAPH_SUCCESS;
}

// Fill tiling data + select cores. Returns the chosen SIZE_MODE in sizeMode.
static ge::graphStatus FillTiling(
    gert::TilingContext* context, SortWithIndexTilingData* tiling, int64_t coreNum, uint64_t ubSize, uint32_t& sizeMode)
{
    SortShapeInfo info;
    OP_CHECK_IF(
        ResolveShapeAndAxis(context, info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ResolveShapeAndAxis error"),
        return ge::GRAPH_FAILED);

    // empty / degenerate: nothing to sort.
    if (info.rowNum == 0U || info.sliceLen == 0U) {
        sizeMode = SORT_WITH_INDEX_SIZE_MODE_EMPTY;
        tiling->validCoreNum = 1U;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    const auto* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto* idxDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxDesc);
    const ge::DataType valueDt = xDesc->GetDataType();
    const ge::DataType indexDt = idxDesc->GetDataType();
    const uint32_t vBytes = ValueDtypeBytes(valueDt);

    const uint32_t bytesPerElem = ComputeBytesPerElem(valueDt, indexDt);

    uint32_t realSortLen = 0U;
    OP_CHECK_IF(
        SelectSizeMode(context, info.sliceLen, ubSize, bytesPerElem, sizeMode, realSortLen) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SelectSizeMode error"), return ge::GRAPH_FAILED);

    uint32_t workCoreNum = 1U;
    OP_CHECK_IF(
        FillCoreAndShapeFields(context, tiling, info, vBytes, realSortLen, coreNum, workCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "FillCoreAndShapeFields error"), return ge::GRAPH_FAILED);

    context->SetBlockDim(workCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SortWithIndexTilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. tiling data buffer
    SortWithIndexTilingData* tiling = context->GetTilingData<SortWithIndexTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SortWithIndexTilingData), 0, sizeof(SortWithIndexTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 3. fill tiling + core selection
    uint32_t sizeMode = SORT_WITH_INDEX_SIZE_MODE_SINGLE;
    OP_CHECK_IF(
        FillTiling(context, tiling, coreNum, ubSize, sizeMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "FillTiling error"), return ge::GRAPH_FAILED);

    // 4. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 5. tiling key (schMode = SIZE_MODE)
    context->SetTilingKey(GET_TPL_TILING_KEY(sizeMode));
    context->GetRawTilingData()->SetDataSize(sizeof(SortWithIndexTilingData));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSortWithIndex([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SortWithIndex)
    .Tiling(SortWithIndexTilingFunc)
    .TilingParse<SortWithIndexCompileInfo>(TilingParseForSortWithIndex);
} // namespace optiling
