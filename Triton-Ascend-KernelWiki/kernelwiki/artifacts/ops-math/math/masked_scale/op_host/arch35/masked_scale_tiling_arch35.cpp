/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_scale_tiling_arch35.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "masked_scale_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/masked_scale_dag.h"
#include "../../op_kernel/arch35/masked_scale_tiling_struct.h"

namespace optiling {
using namespace Ops::Base;

const uint64_t VALUE_ATTR_IDX = 0;
const uint64_t INPUT_X_IDX = 0;
const uint64_t INPUT_MASK_IDX = 1;
const uint64_t OUTPUT_Y_IDX = 0;
const uint64_t TILING_KEY_X_FP16 = 1;
const uint64_t TILING_KEY_X_BF16 = 2;
const uint64_t TILING_KEY_X_FP32 = 3;
const uint64_t TILING_KEY_MASK_UINT8 = 10;
const uint64_t TILING_KEY_X_HALF_MASK_UINT8 = 11;
const uint64_t TILING_KEY_X_BF16_MASK_UINT8 = 12;
const uint64_t TILING_KEY_X_FP32_MASK_UINT8 = 13;
const uint64_t TILING_KEY_MASK_INT8 = 20;
const uint64_t TILING_KEY_X_HALF_MASK_INT8 = 21;
const uint64_t TILING_KEY_X_BF16_MASK_INT8 = 22;
const uint64_t TILING_KEY_X_FP32_MASK_INT8 = 23;
const uint64_t TILING_KEY_MASK_FP16 = 30;
const uint64_t TILING_KEY_X_HALF_MASK_HALF = 31;
const uint64_t TILING_KEY_X_BF16_MASK_HALF = 32;
const uint64_t TILING_KEY_X_FP32_MASK_HALF = 33;
const uint64_t TILING_KEY_MASK_FP32 = 40;
const uint64_t TILING_KEY_X_HALF_MASK_FP32 = 41;
const uint64_t TILING_KEY_X_BF16_MASK_FP32 = 42;
const uint64_t TILING_KEY_X_FP32_MASK_FP32 = 43;

MaskedScaleNs::MaskedScaleTilingData *maskedScaleTilingData = nullptr;

static ge::graphStatus Tiling4DoTilingDag(gert::TilingContext* tilingContext, uint64_t tilingKey)
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    maskedScaleTilingData = tilingContext->GetTilingData<MaskedScaleNs::MaskedScaleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, maskedScaleTilingData);
    if (tilingKey == TILING_KEY_X_HALF_MASK_UINT8) { // 11: half uint8
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastTwo<half, uint8_t>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_BF16_MASK_UINT8) { // 12: bfloat16 uint8
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastTwo<half, uint8_t>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_FP32_MASK_UINT8) { // 13: fp32 uint8
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastTwo<float, uint8_t>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_HALF_MASK_INT8) { // 21: half int8
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastTwo<half, int8_t>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_BF16_MASK_INT8) { // 22: bfloat16 int8
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastTwo<half, int8_t>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_FP32_MASK_INT8) { // 23: fp32 int8
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastTwo<float, int8_t>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_HALF_MASK_HALF) { // 31: half half
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastOne<half, half>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_BF16_MASK_HALF) { // 32: bfloat16 half
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastOne<half, half>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_FP32_MASK_HALF) { // 33: fp32 half
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastOne<float, half>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_HALF_MASK_FP32) { // 41: half fp32
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastOne<half, float>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_BF16_MASK_FP32) { // 42: bfloat16 fp32
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastOne<half, float>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else if (tilingKey == TILING_KEY_X_FP32_MASK_FP32) { // 43: fp32 fp32
        return elewiseBaseTiling.DoTiling<MaskedScaleOp::MaskedScaleMaskCastOne<float, float>::OpDag>(
            maskedScaleTilingData->baseTiling);
    } else {
        return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus Tiling4MaskedScale(gert::TilingContext *tilingContext)
{
  OP_LOGD(tilingContext->GetNodeName(), "Tiling4MaskedScale rt2.0 is running.");
  auto xDesc = tilingContext->GetInputDesc(INPUT_X_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xDesc);
  auto maskDesc = tilingContext->GetInputDesc(INPUT_MASK_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(tilingContext, maskDesc);
  auto xDtype = xDesc->GetDataType();
  auto maskDtype = maskDesc->GetDataType();
  uint64_t tilingKey = 0;
  if (xDtype == ge::DT_FLOAT16) {
    tilingKey += TILING_KEY_X_FP16;
  } else if (xDtype == ge::DT_BF16) {
    tilingKey += TILING_KEY_X_BF16;
  } else if (xDtype == ge::DT_FLOAT) {
    tilingKey += TILING_KEY_X_FP32;
  } else {
    OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype), "FLOAT16, BF16, FLOAT");
    return ge::GRAPH_FAILED;
  }
  if (maskDtype == ge::DT_UINT8) {
    tilingKey += TILING_KEY_MASK_UINT8;
  } else if (maskDtype == ge::DT_INT8) {
    tilingKey += TILING_KEY_MASK_INT8;
  } else if (maskDtype == ge::DT_FLOAT16) {
    tilingKey += TILING_KEY_MASK_FP16;
  } else if (maskDtype == ge::DT_FLOAT) {
    tilingKey += TILING_KEY_MASK_FP32;
  } else {
    OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "mask", ge::TypeUtils::DataTypeToSerialString(maskDtype), "UINT8, INT8, FLOAT16, FLOAT");
    return ge::GRAPH_FAILED;
  }

OP_CHECK_IF(Tiling4DoTilingDag(tilingContext, tilingKey) != ge::GRAPH_SUCCESS,
                   OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling DoTiling failed"),
                   return ge::GRAPH_FAILED);

  float scale = *tilingContext->GetAttrs()->GetAttrPointer<float>(VALUE_ATTR_IDX);
  maskedScaleTilingData->scale = scale;
  OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : scale=%f.", scale);

  tilingContext->SetTilingKey(tilingKey);
  OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu.", tilingKey);
  tilingContext->SetBlockDim(maskedScaleTilingData->baseTiling.blockNum);

  size_t sysWorkspaceSize = static_cast<size_t>(16) * 1024 * 1024;
  size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
  currentWorkspace[0] = sysWorkspaceSize;

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMaskedScale(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<MaskedScaleCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaskedScale).Tiling(Tiling4MaskedScale)
                             .TilingParse<MaskedScaleCompileInfo>(TilingPrepareForMaskedScale);
}  // namespace optiling
