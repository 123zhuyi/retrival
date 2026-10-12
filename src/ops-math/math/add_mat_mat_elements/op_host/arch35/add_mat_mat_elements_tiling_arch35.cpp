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
 * \file add_mat_mat_elements_tiling_arch35.cpp
 * \brief AddMatMatElements TilingFunc (arch35, Ascend950)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/add_mat_mat_elements_tiling_data.h"
#include "../../op_kernel/arch35/add_mat_mat_elements_tiling_key.h"
#include "add_mat_mat_elements_tiling_arch35.h"

#include <algorithm>
#include <cinttypes>
#include <set>

namespace optiling {

constexpr int64_t P = ADD_MAT_MAT_ELEMENTS_PHYS_NODES;
constexpr int64_t RANK_MAX = ADD_MAT_MAT_ELEMENTS_RANK_MAX;
constexpr int64_t RANK_4 = ADD_MAT_MAT_ELEMENTS_RANK_4;
constexpr int64_t RANK_8 = ADD_MAT_MAT_ELEMENTS_RANK_8;
constexpr int64_t MIN_RANK = 1;
constexpr int64_t MAX_RANK = 8;
constexpr size_t WS_SIZE = 0U;

// Right-align pad leading 1s to effectiveRANK, write to inputShapes[slot][RANK_MAX]
// max_bro_shape = c.shape (output anchor, c does not participate in broadcast)
static void PadAndSqueeze(const int64_t* cShape, int64_t cRank, const int64_t* aShape, int64_t aRank,
                          const int64_t* bShape, int64_t bRank, int64_t effectiveRANK, int64_t inputShapes[][RANK_MAX],
                          int64_t* maxBroShape)
{
    for (int64_t s = 0; s < ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS; s++) {
        for (int64_t d = 0; d < RANK_MAX; d++) {
            inputShapes[s][d] = 1;
        }
    }
    const int64_t* shapes[3] = {cShape, aShape, bShape};
    int64_t ranks[3] = {cRank, aRank, bRank};
    for (int64_t s = 0; s < 3; s++) {
        for (int64_t d = 0; d < ranks[s]; d++) {
            inputShapes[s][effectiveRANK - ranks[s] + d] = shapes[s][d];
        }
    }
    for (int64_t d = 0; d < RANK_MAX; d++) {
        maxBroShape[d] = inputShapes[0][d];
    }
}

static ge::graphStatus CheckBroadcastShape(const int64_t inShapes[][RANK_MAX], int64_t effectiveRANK)
{
    for (int64_t slot = 1; slot <= 2; slot++) {
        for (int64_t d = 0; d < effectiveRANK; d++) {
            int64_t cd = inShapes[0][d];
            int64_t sd = inShapes[slot][d];
            if (sd != 1 && sd != cd) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// input dim == 1 → stride = 0 (broadcast axis); else stride = product(higher dims)
static void CalcBroadcastStrides(const int64_t inShapes[][RANK_MAX], int64_t effectiveRANK,
                                 int64_t inStrides[][RANK_MAX], int64_t outStrides[][RANK_MAX])
{
    for (int64_t slot = 0; slot < ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS; slot++) {
        int64_t acc = 1;
        for (int64_t d = RANK_MAX - 1; d >= 0; d--) {
            if (d >= effectiveRANK) {
                inStrides[slot][d] = 0;
                continue;
            }
            if (inShapes[slot][d] == 1) {
                inStrides[slot][d] = 0;
            } else {
                inStrides[slot][d] = acc;
                acc *= inShapes[slot][d];
            }
        }
    }
    for (int64_t d = 0; d < RANK_MAX; d++) {
        outStrides[0][d] = inStrides[0][d];
    }
}

// From inner axis outward, find max split that fits perBufElems
static void FindSplitAxis(int64_t perBufElems, const int64_t* broShape, int64_t effectiveRANK,
                          AddMatMatElementsSplitResult& split)
{
    int64_t innerProd = 1;
    for (int64_t axis = effectiveRANK - 1; axis >= 0; axis--) {
        innerProd *= broShape[axis];
        if (innerProd > perBufElems) {
            innerProd /= broShape[axis];
            split.axis = axis + 1;
            split.a_i = perBufElems / innerProd;
            split.a_o = broShape[axis] / split.a_i;
            split.a_i_tail = broShape[axis] % split.a_i;
            return;
        }
    }
    split.axis = 0;
    split.a_i = broShape[0];
    split.a_o = 1;
    split.a_i_tail = 0;
}

static void MultiCoreSplit(int64_t totalTiles, int64_t coreNum, AddMatMatElementsMultiCoreResult& mc)
{
    mc.total_tiles = totalTiles;
    mc.num_cores = (totalTiles < coreNum) ? totalTiles : coreNum;
    if (mc.num_cores <= 0) {
        mc.num_cores = 1;
    }
    mc.tiles_main = totalTiles / mc.num_cores;
    mc.cores_tail = totalTiles % mc.num_cores;
}

static void FillAndLogTilingData(AddMatMatElementsTilingData* td, const AddMatMatElementsSplitResult& split,
                                 const AddMatMatElementsMultiCoreResult& mc, int64_t perBufBytes, int64_t effectiveRank,
                                 const int64_t inShapes[][RANK_MAX], const int64_t inStrides[][RANK_MAX],
                                 const int64_t outStrides[][RANK_MAX], const int64_t* maxBroShape)
{
    td->split = split;
    td->multicore = mc;
    td->rank = effectiveRank;
    td->per_buf_bytes = perBufBytes;
    for (int64_t d = 0; d < RANK_MAX; d++) {
        td->max_bro_shape[d] = maxBroShape[d];
    }
    td->num_inputs = ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS;
    td->num_outputs = ADD_MAT_MAT_ELEMENTS_MAX_OUTPUT_SLOTS;
    for (int64_t s = 0; s < ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS; s++) {
        for (int64_t d = 0; d < RANK_MAX; d++) {
            td->input_shapes[s][d] = inShapes[s][d];
            td->input_strides[s][d] = inStrides[s][d];
        }
    }
    for (int64_t d = 0; d < RANK_MAX; d++) {
        td->output_shapes[0][d] = maxBroShape[d];
        td->output_strides[0][d] = outStrides[0][d];
    }
}

static ge::graphStatus ValidateInputDtypes(gert::TilingContext* context)
{
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT};
    ge::DataType refDtype = ge::DT_UNDEFINED;
    for (size_t i = 0; i < 5; i++) {
        auto desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        auto dtype = desc->GetDataType();
        if (supportedDtypes.count(dtype) == 0) {
            OP_LOGE(context, "AddMatMatElements: input %zu dtype %d not supported (FP16/FP32 only)", i,
                    static_cast<int>(dtype));
            return ge::GRAPH_FAILED;
        }
        if (i == 0) {
            refDtype = dtype;
        } else if (dtype != refDtype) {
            OP_LOGE(context, "AddMatMatElements: input %zu dtype mismatch (%d vs %d)", i, static_cast<int>(dtype),
                    static_cast<int>(refDtype));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputShapes(gert::TilingContext* context)
{
    auto cShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, cShape);
    int64_t cRank = cShape->GetStorageShape().GetDimNum();
    if (cRank < MIN_RANK || cRank > MAX_RANK) {
        OP_LOGE(context, "AddMatMatElements: rank(c)=%ld out of range [1,8]", cRank);
        return ge::GRAPH_FAILED;
    }

    auto aShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShape);
    int64_t aRank = aShape->GetStorageShape().GetDimNum();
    if (aRank > cRank) {
        OP_LOGE(context, "AddMatMatElements: rank(a)=%ld > rank(c)=%ld", aRank, cRank);
        return ge::GRAPH_FAILED;
    }

    auto bShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, bShape);
    int64_t bRank = bShape->GetStorageShape().GetDimNum();
    if (bRank > cRank) {
        OP_LOGE(context, "AddMatMatElements: rank(b)=%ld > rank(c)=%ld", bRank, cRank);
        return ge::GRAPH_FAILED;
    }

    auto betaShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    int64_t betaRank = betaShape->GetStorageShape().GetDimNum();
    if (betaRank != 1 || betaShape->GetStorageShape().GetDim(0) != 1) {
        OP_LOGE(context, "AddMatMatElements: beta shape must be (1,)");
        return ge::GRAPH_FAILED;
    }

    auto alphaShape = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, alphaShape);
    int64_t alphaRank = alphaShape->GetStorageShape().GetDimNum();
    if (alphaRank != 1 || alphaShape->GetStorageShape().GetDim(0) != 1) {
        OP_LOGE(context, "AddMatMatElements: alpha shape must be (1,)");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputs(gert::TilingContext* context)
{
    OP_CHECK_IF(ValidateInputDtypes(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AddMatMatElements: dtype validation failed"), return ge::GRAPH_FAILED);
    return ValidateInputShapes(context);
}

static ge::graphStatus GetPlatformParams(gert::TilingContext* context, int64_t& coreNum, uint64_t& ubSize)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "AddMatMatElements: coreNum=%ld invalid", coreNum),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "AddMatMatElements: ubSize=0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareInputShapes(gert::TilingContext* context, int64_t inShapes[][RANK_MAX],
                                          int64_t* maxBroShape, int64_t& cRank, int64_t& totalElems)
{
    auto cShape = context->GetInputShape(0);
    auto aShape = context->GetInputShape(1);
    auto bShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bShape);

    cRank = cShape->GetStorageShape().GetDimNum();
    int64_t aRank = aShape->GetStorageShape().GetDimNum();
    int64_t bRank = bShape->GetStorageShape().GetDimNum();
    totalElems = cShape->GetStorageShape().GetShapeSize();
    int64_t effectiveRank = (cRank <= RANK_4) ? RANK_4 : RANK_8;

    auto cDims = cShape->GetStorageShape();
    auto aDims = aShape->GetStorageShape();
    auto bDims = bShape->GetStorageShape();
    int64_t cShapeArr[RANK_MAX] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t aShapeArr[RANK_MAX] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t bShapeArr[RANK_MAX] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int64_t d = 0; d < cRank; d++) {
        cShapeArr[d] = cDims.GetDim(d);
    }
    for (int64_t d = 0; d < aRank; d++) {
        aShapeArr[d] = aDims.GetDim(d);
    }
    for (int64_t d = 0; d < bRank; d++) {
        bShapeArr[d] = bDims.GetDim(d);
    }
    PadAndSqueeze(cShapeArr, cRank, aShapeArr, aRank, bShapeArr, bRank, effectiveRank, inShapes, maxBroShape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ResetTilingData(gert::TilingContext* context, AddMatMatElementsTilingData* td)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(AddMatMatElementsTilingData), 0, sizeof(AddMatMatElementsTilingData)) != EOK,
                OP_LOGE(context, "AddMatMatElements: reset tiling data failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = WS_SIZE;
    }
}

static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, ge::DataType dtype, int64_t effectiveRank)
{
    SetWorkspaceSize(context);
    context->SetBlockDim(1);
    auto* td = context->GetTilingData<AddMatMatElementsTilingData>();
    OP_CHECK_IF(ResetTilingData(context, td) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AddMatMatElements: reset empty tensor tiling data failed"), return ge::GRAPH_FAILED);
    td->rank = effectiveRank;
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype), static_cast<uint32_t>(effectiveRank));
    OP_LOGI(context, "AddMatMatElements: empty tensor, short-circuit");
    return ge::GRAPH_SUCCESS;
}

static int64_t CalcTotalTiles(const AddMatMatElementsSplitResult& split, const int64_t* maxBroShape)
{
    if (split.axis == 0) {
        return 1;
    }
    int64_t outerProduct = 1;
    for (int64_t d = 0; d < split.axis - 1; d++) {
        outerProduct *= maxBroShape[d];
    }
    int64_t numTilesAlongAxis = split.a_o + (split.a_i_tail > 0 ? 1 : 0);
    return std::max<int64_t>(outerProduct * numTilesAlongAxis, 1);
}

ge::graphStatus AddMatMatElementsTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(ValidateInputs(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "AddMatMatElements: validation failed"),
                return ge::GRAPH_FAILED);

    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    OP_CHECK_IF(GetPlatformParams(context, coreNum, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AddMatMatElements: get platform info failed"), return ge::GRAPH_FAILED);

    int64_t inShapes[ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS][RANK_MAX];
    int64_t maxBroShape[RANK_MAX];
    int64_t cRank = 0;
    int64_t totalElems = 0;
    OP_CHECK_IF(PrepareInputShapes(context, inShapes, maxBroShape, cRank, totalElems) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AddMatMatElements: prepare input shapes failed"), return ge::GRAPH_FAILED);
    int64_t effectiveRank = (cRank <= RANK_4) ? RANK_4 : RANK_8;

    OP_CHECK_IF(CheckBroadcastShape(inShapes, effectiveRank) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AddMatMatElements: a/b cannot broadcast to c.shape"), return ge::GRAPH_FAILED);

    auto cDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, cDesc);
    ge::DataType dtype = cDesc->GetDataType();
    int64_t dtypeSize = (dtype == ge::DT_FLOAT16) ? 2 : 4;

    if (totalElems == 0) {
        // A zero block dimension is rejected as INVALID_TILING by the runtime.
        // Launch one no-op block and let total_tiles == 0 short-circuit the kernel.
        return HandleEmptyTensor(context, dtype, effectiveRank);
    }

    int64_t inStrides[ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS][RANK_MAX];
    int64_t outStrides[ADD_MAT_MAT_ELEMENTS_MAX_OUTPUT_SLOTS][RANK_MAX];
    CalcBroadcastStrides(inShapes, effectiveRank, inStrides, outStrides);

    int64_t perBufBytes = static_cast<int64_t>((ubSize / P) & ~31UL);
    int64_t perBufElems = perBufBytes / dtypeSize;
    AddMatMatElementsSplitResult split;
    FindSplitAxis(perBufElems, maxBroShape, effectiveRank, split);

    int64_t totalTiles = CalcTotalTiles(split, maxBroShape);
    AddMatMatElementsMultiCoreResult mc;
    MultiCoreSplit(totalTiles, coreNum, mc);

    auto* td = context->GetTilingData<AddMatMatElementsTilingData>();
    OP_CHECK_IF(ResetTilingData(context, td) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AddMatMatElements: reset tiling data failed"), return ge::GRAPH_FAILED);
    FillAndLogTilingData(td, split, mc, perBufBytes, effectiveRank, inShapes, inStrides, outStrides, maxBroShape);

    SetWorkspaceSize(context);

    uint32_t rankSlot = static_cast<uint32_t>(effectiveRank);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype), rankSlot);

    context->SetBlockDim(static_cast<uint32_t>(mc.num_cores));

    OP_LOGI(context, "AddMatMatElements: rankSlot=%u, cRank=%ld, totalElems=%ld, totalTiles=%ld, numCores=%ld",
            rankSlot, cRank, totalElems, totalTiles, mc.num_cores);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAddMatMatElements([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddMatMatElements)
    .Tiling(AddMatMatElementsTilingFunc)
    .TilingParse<AddMatMatElementsCompileInfo>(TilingParseForAddMatMatElements);

} // namespace optiling
