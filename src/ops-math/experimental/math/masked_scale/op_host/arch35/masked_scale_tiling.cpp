// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include "log/log.h"
#include <algorithm>
#include <limits>
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/masked_scale_tiling_data.h"
#include "../../op_kernel/arch35/masked_scale_tiling_key.h"

namespace optiling {

constexpr uint32_t INPUT_SELF_INDEX = 0U;
constexpr uint32_t INPUT_MASK_INDEX = 1U;
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t ATTR_SCALE_INDEX = 0U;
constexpr uint32_t CORE_MIN_BYTES = 4096U;
constexpr uint32_t BLOCK_FORMER_ALIGN = 512U;
constexpr uint32_t UB_RESERVE_BYTES = 32U * 1024U;
constexpr uint32_t MIN_UB_FORMER = 256U;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;

struct MaskedScaleMeta {
    uint64_t dim0;
    ge::DataType selfDtype;
    ge::DataType maskDtype;
    uint32_t selfBytes;
    uint32_t maskBytes;
    float scale;
};

static uint64_t CeilDiv(uint64_t value, uint64_t factor) { return factor == 0U ? 0U : (value + factor - 1U) / factor; }

static uint64_t AlignUp(uint64_t value, uint64_t align) { return align == 0U ? value : CeilDiv(value, align) * align; }

static uint64_t AlignDown(uint64_t value, uint64_t align) { return align == 0U ? value : value / align * align; }

static ge::graphStatus GetDtypeBytes(gert::TilingContext* context, ge::DataType dtype, uint32_t& bytes)
{
    if (!ge::TypeUtils::GetDataTypeLength(dtype, bytes) || bytes == 0U) {
        OP_LOGE(context, "MaskedScale: unsupported dtype length, dtype=%d", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static bool IsSupportedSelfDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16;
}

static bool IsSupportedMaskDtype(ge::DataType dtype)
{
    return dtype == ge::DT_UINT8 || dtype == ge::DT_INT8 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT;
}

static uint32_t GetBranchKey(ge::DataType selfDtype)
{
    if (selfDtype == ge::DT_FLOAT16) {
        return 0U;
    }
    if (selfDtype == ge::DT_FLOAT) {
        return 1U;
    }
    return 2U;
}

static uint64_t GetTilingKey(ge::DataType selfDtype, ge::DataType maskDtype)
{
    if (selfDtype == ge::DT_FLOAT16) {
        if (maskDtype == ge::DT_UINT8) {
            return MASKED_SCALE_KEY_FP16_UINT8;
        }
        if (maskDtype == ge::DT_INT8) {
            return MASKED_SCALE_KEY_FP16_INT8;
        }
        if (maskDtype == ge::DT_FLOAT16) {
            return MASKED_SCALE_KEY_FP16_FP16;
        }
        return MASKED_SCALE_KEY_FP16_FP32;
    }
    if (selfDtype == ge::DT_FLOAT) {
        if (maskDtype == ge::DT_UINT8) {
            return MASKED_SCALE_KEY_FP32_UINT8;
        }
        if (maskDtype == ge::DT_INT8) {
            return MASKED_SCALE_KEY_FP32_INT8;
        }
        if (maskDtype == ge::DT_FLOAT16) {
            return MASKED_SCALE_KEY_FP32_FP16;
        }
        return MASKED_SCALE_KEY_FP32_FP32;
    }
    if (maskDtype == ge::DT_UINT8) {
        return MASKED_SCALE_KEY_BF16_UINT8;
    }
    if (maskDtype == ge::DT_INT8) {
        return MASKED_SCALE_KEY_BF16_INT8;
    }
    if (maskDtype == ge::DT_FLOAT16) {
        return MASKED_SCALE_KEY_BF16_FP16;
    }
    return MASKED_SCALE_KEY_BF16_FP32;
}

static uint32_t GetSubCaseKey(ge::DataType selfDtype, ge::DataType maskDtype)
{
    return GetBranchKey(selfDtype) * 10U + (maskDtype == ge::DT_UINT8   ? 0U :
                                            maskDtype == ge::DT_INT8    ? 1U :
                                            maskDtype == ge::DT_FLOAT16 ? 2U :
                                                                          3U);
}

static uint64_t CalcUbFormer(uint64_t ubSize, uint32_t selfBytes, uint32_t maskBytes, ge::DataType selfDtype)
{
    const uint64_t ubBudget = ubSize > UB_RESERVE_BYTES ? ubSize - UB_RESERVE_BYTES : ubSize / 2U;
    const bool fp16Branch = selfDtype == ge::DT_FLOAT16;
    const uint32_t computeBytes = fp16Branch ? sizeof(uint16_t) : sizeof(float);
    const uint32_t bytesPerElem = selfBytes + maskBytes + selfBytes + sizeof(uint16_t) + 3U * sizeof(float);

    uint64_t maxElems = bytesPerElem == 0U ? MIN_UB_FORMER : ubBudget / bytesPerElem;
    const uint64_t alignElems = 256U / computeBytes;
    uint64_t ubFormer = AlignDown(maxElems, alignElems);
    if (ubFormer == 0U) {
        ubFormer = MIN_UB_FORMER;
    }
    return ubFormer;
}

static ge::graphStatus CheckShapeInfo(gert::TilingContext* context, uint64_t& dim0)
{
    auto selfShape = context->GetInputShape(INPUT_SELF_INDEX);
    auto maskShape = context->GetInputShape(INPUT_MASK_INDEX);
    auto yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const int64_t dim0Raw = selfShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(dim0Raw < 0, OP_LOGE(context, "MaskedScale: negative shape size"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        maskShape->GetStorageShape().GetShapeSize() != dim0Raw || yShape->GetStorageShape().GetShapeSize() != dim0Raw,
        OP_LOGE(context, "MaskedScale: self/mask/y shape size mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(dim0Raw) > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "MaskedScale: shape size exceeds uint32_t range"), return ge::GRAPH_FAILED);

    dim0 = static_cast<uint64_t>(dim0Raw);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillDtypeAndAttr(gert::TilingContext* context, MaskedScaleMeta& meta)
{
    auto selfDesc = context->GetInputDesc(INPUT_SELF_INDEX);
    auto maskDesc = context->GetInputDesc(INPUT_MASK_INDEX);
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    const ge::DataType selfDtype = selfDesc->GetDataType();
    const ge::DataType maskDtype = maskDesc->GetDataType();
    const ge::DataType yDtype = yDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedSelfDtype(selfDtype), OP_LOGE(context, "MaskedScale: unsupported self dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedMaskDtype(maskDtype), OP_LOGE(context, "MaskedScale: unsupported mask dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDtype != selfDtype, OP_LOGE(context, "MaskedScale: y dtype must equal self dtype"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetDtypeBytes(context, selfDtype, meta.selfBytes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "MaskedScale: get self dtype bytes failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDtypeBytes(context, maskDtype, meta.maskBytes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "MaskedScale: get mask dtype bytes failed"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scale = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);

    meta.selfDtype = selfDtype;
    meta.maskDtype = maskDtype;
    meta.scale = *scale;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMeta(gert::TilingContext* context, MaskedScaleMeta& meta)
{
    OP_CHECK_IF(CheckShapeInfo(context, meta.dim0) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "MaskedScale: check shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillDtypeAndAttr(context, meta) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "MaskedScale: fill dtype and attr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void FillEmptyTiling(gert::TilingContext* context, MaskedScaleTilingData* tiling, const MaskedScaleMeta& meta)
{
    const uint32_t blockDimWhenEmpty = 1U;
    tiling->dim0 = 0U;
    tiling->coreNum = blockDimWhenEmpty;
    tiling->blockFormer = 0U;
    tiling->blockNum = blockDimWhenEmpty;
    tiling->ubFormer = MIN_UB_FORMER;
    tiling->selfDtype = static_cast<uint32_t>(meta.selfDtype);
    tiling->maskDtype = static_cast<uint32_t>(meta.maskDtype);
    tiling->branchKey = GetBranchKey(meta.selfDtype);
    tiling->subCaseKey = GetSubCaseKey(meta.selfDtype, meta.maskDtype);
    tiling->scaleFloat = meta.scale;
    tiling->bufferNum = 1U;
    context->SetBlockDim(blockDimWhenEmpty);
}

static ge::graphStatus FillNormalTiling(gert::TilingContext* context, MaskedScaleTilingData* tiling, uint64_t ubSize,
                                        uint32_t platformCoreNum, const MaskedScaleMeta& meta)
{
    const uint64_t gmBytesPerElem = static_cast<uint64_t>(meta.selfBytes) + meta.maskBytes + meta.selfBytes;
    uint64_t coreNum = std::min<uint64_t>(platformCoreNum,
                                          std::max<uint64_t>(1U, CeilDiv(meta.dim0 * gmBytesPerElem, CORE_MIN_BYTES)));
    uint64_t blockFormer = AlignUp(CeilDiv(meta.dim0, coreNum), BLOCK_FORMER_ALIGN);
    uint64_t blockNum = CeilDiv(meta.dim0, blockFormer);
    coreNum = std::min(coreNum, blockNum);
    blockFormer = AlignUp(CeilDiv(meta.dim0, coreNum), BLOCK_FORMER_ALIGN);
    blockNum = CeilDiv(meta.dim0, blockFormer);
    OP_CHECK_IF(blockFormer > std::numeric_limits<uint32_t>::max() || blockNum > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "MaskedScale: tiling value exceeds uint32_t range"), return ge::GRAPH_FAILED);

    const uint64_t ubFormer = std::min(CalcUbFormer(ubSize, meta.selfBytes, meta.maskBytes, meta.selfDtype),
                                       blockFormer);
    if (ubFormer == 0U) {
        OP_LOGE(context, "MaskedScale: ubFormer is zero");
        return ge::GRAPH_FAILED;
    }

    tiling->dim0 = static_cast<uint32_t>(meta.dim0);
    tiling->coreNum = static_cast<uint32_t>(coreNum);
    tiling->blockFormer = static_cast<uint32_t>(blockFormer);
    tiling->blockNum = static_cast<uint32_t>(blockNum);
    tiling->ubFormer = static_cast<uint32_t>(ubFormer);
    tiling->selfDtype = static_cast<uint32_t>(meta.selfDtype);
    tiling->maskDtype = static_cast<uint32_t>(meta.maskDtype);
    tiling->branchKey = GetBranchKey(meta.selfDtype);
    tiling->subCaseKey = GetSubCaseKey(meta.selfDtype, meta.maskDtype);
    tiling->scaleFloat = meta.scale;
    tiling->bufferNum = 1U;
    context->SetBlockDim(static_cast<uint32_t>(blockNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillTilingData(gert::TilingContext* context, MaskedScaleTilingData* tiling, uint64_t ubSize,
                                      uint32_t platformCoreNum)
{
    MaskedScaleMeta meta{};
    OP_CHECK_IF(GetMeta(context, meta) != ge::GRAPH_SUCCESS, OP_LOGE(context, "MaskedScale: get meta failed"),
                return ge::GRAPH_FAILED);
    if (meta.dim0 == 0U) {
        FillEmptyTiling(context, tiling, meta);
        return ge::GRAPH_SUCCESS;
    }
    return FillNormalTiling(context, tiling, ubSize, platformCoreNum, meta);
}

static ge::graphStatus MaskedScaleTilingFunc(gert::TilingContext* context)
{
    auto tiling = context->GetTilingData<MaskedScaleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->Init();

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t platformCoreNum = static_cast<uint32_t>(std::max<int64_t>(1, ascendcPlatform.GetCoreNum()));

    OP_CHECK_IF(FillTilingData(context, tiling, ubSize, platformCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "MaskedScale: FillTilingData failed"), return ge::GRAPH_FAILED);

    auto selfDesc = context->GetInputDesc(INPUT_SELF_INDEX);
    auto maskDesc = context->GetInputDesc(INPUT_MASK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskDesc);
    context->SetTilingKey(GetTilingKey(selfDesc->GetDataType(), maskDesc->GetDataType()));

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaskedScale([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MaskedScaleCompileInfo {};

IMPL_OP_OPTILING(MaskedScale)
    .Tiling(MaskedScaleTilingFunc)
    .TilingParse<MaskedScaleCompileInfo>(TilingParseForMaskedScale);
} // namespace optiling
