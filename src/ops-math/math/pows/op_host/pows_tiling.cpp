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
 * \file pows_tiling.cpp
 * \brief
 */
#include "pows_tiling.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static const uint32_t INPUT_IDX = 0;
static const uint32_t DIM_0 = 0;
static const uint32_t DIM_1 = 1;
static const uint32_t DIM_2 = 2;
static const uint32_t DIM_3 = 3;
static const uint32_t ATTR_DIM_INDEX = 0;
static const uint32_t ATTR_APPROXIMATE_INDEX = 1;
static const uint32_t ATTR_ACTIVATE_LEFT_INDEX = 2;
static const uint32_t FP16_DTYPE_BYTES = 2;
static const uint32_t FP32_DTYPE_BYTES = 4;
static const uint32_t FP16_COEXISTING_NUM = 7;
static const uint32_t FP32_COEXISTING_NUM = 3;
static const uint32_t WORK_SPACE_SIZE = 32;
static const uint32_t SPLIT_FACTOR = 2;
static const uint32_t SPLIT_ERROR_STATUS = 10000;
static const int64_t APPROXIMATE_USING_TANH = 1;
static const int64_t APPROXIMATE_USING_ERF = 0;
static const int64_t BYTES_ONE_BLOCK = 32;
static const int64_t MULTI_CORE_SHAPE_SIZE_LIMIT = 4096;
static const int64_t BUFFER_SIZE_ALIGN_LENGTH = 256;

inline static ge::graphStatus SetTilingDataForPows(gert::TilingContext* context, PowsTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void GetTilingDataNonCut(PowsTilingData& tilingData, TilingParam& tilingParam)
{
    tilingData.set_numPerCore(tilingParam.x);
    tilingData.set_realCoreNum(1);
    tilingData.set_mainCoreLoopNum(1);
    tilingData.set_mainCoreTailLength(0);
    tilingData.set_dataLength(tilingParam.x);
    tilingData.set_tailCoreLoopNum(0);
    tilingData.set_tailCoreTailLength(0);
    tilingData.set_bufSize(tilingParam.bufSize / BUFFER_SIZE_ALIGN_LENGTH * BUFFER_SIZE_ALIGN_LENGTH);
    tilingData.set_blockSize(tilingParam.blockSize);
}
static void GetTilingDataBigCase(PowsTilingData& tilingData, TilingParam& tilingParam)
{
    int64_t bufSizeAlign = tilingParam.bufSize / BUFFER_SIZE_ALIGN_LENGTH * BUFFER_SIZE_ALIGN_LENGTH;
    int64_t blockFactor = Ops::Base::CeilDiv(tilingParam.x, tilingParam.coreNum);
    // block factor align upper
    int64_t blockFactorAlign =
        (blockFactor + tilingParam.blockSize - 1) / tilingParam.blockSize * tilingParam.blockSize;
    int64_t realCoreNum = Ops::Base::CeilDiv(tilingParam.x, blockFactorAlign);

    tilingData.set_numPerCore(blockFactorAlign);
    tilingData.set_realCoreNum(realCoreNum);

    // ub factor align down
    int64_t ubFactorAlign = bufSizeAlign / tilingParam.blockSize * tilingParam.blockSize;
    ubFactorAlign = ubFactorAlign > blockFactorAlign ? blockFactorAlign : ubFactorAlign;
    tilingData.set_mainCoreLoopNum(blockFactorAlign / ubFactorAlign);
    tilingData.set_mainCoreTailLength(blockFactorAlign % ubFactorAlign);
    tilingData.set_dataLength(ubFactorAlign);

    if (tilingParam.x % blockFactorAlign != 0) {
        int64_t tailCoreTotalNum = tilingParam.x - blockFactorAlign * (realCoreNum - 1);
        tilingData.set_tailCoreLoopNum(tailCoreTotalNum / ubFactorAlign);
        tilingData.set_tailCoreTailLength(tailCoreTotalNum % ubFactorAlign);
    } else {
        tilingData.set_tailCoreLoopNum(0);
        tilingData.set_tailCoreTailLength(0);
    }
    tilingData.set_bufSize(bufSizeAlign);
    tilingData.set_blockSize(tilingParam.blockSize);
}

static void GetTilingData(PowsTilingData& tilingData, TilingParam& tilingParam)
{
    if (tilingParam.x < MULTI_CORE_SHAPE_SIZE_LIMIT) {
        GetTilingDataNonCut(tilingData, tilingParam);
    } else {
        GetTilingDataBigCase(tilingData, tilingParam);
    }
}

static void GetFp16TilingData(PowsTilingData& tilingData, TilingParam& tilingParam)
{
    tilingParam.bufSize = tilingParam.ubSize / (FP16_DTYPE_BYTES * FP16_COEXISTING_NUM);
    tilingParam.blockSize = BYTES_ONE_BLOCK / FP16_DTYPE_BYTES;
    GetTilingData(tilingData, tilingParam);
    tilingData.set_tilingKey(static_cast<int64_t>(PowsTilingKey::TILINGKEY_101));
}

static void GetBf16TilingData(PowsTilingData& tilingData, TilingParam& tilingParam)
{
    tilingParam.bufSize = tilingParam.ubSize / (FP16_DTYPE_BYTES * FP16_COEXISTING_NUM);
    tilingParam.blockSize = BYTES_ONE_BLOCK / FP16_DTYPE_BYTES;
    GetTilingData(tilingData, tilingParam);
    int64_t tilingkey = static_cast<int64_t>(PowsTilingKey::TILINGKEY_201);
    tilingData.set_tilingKey(tilingkey);
}

static void GetFp32TilingData(PowsTilingData& tilingData, TilingParam& tilingParam)
{
    tilingParam.bufSize = tilingParam.ubSize / (FP32_DTYPE_BYTES * FP32_COEXISTING_NUM);
    tilingParam.blockSize = BYTES_ONE_BLOCK / FP32_DTYPE_BYTES;
    GetTilingData(tilingData, tilingParam);
    int64_t tilingkey = static_cast<int64_t>(PowsTilingKey::TILINGKEY_301);
    tilingData.set_tilingKey(tilingkey);
}

static ge::graphStatus CheckInputParams(const gert::TilingContext* context)
{
    auto input = context->GetInputTensor(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    int32_t typeSize = ge::GetSizeByDataType(dtype);

    OP_CHECK_IF(
        dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x1",
            ge::TypeUtils::DataTypeToSerialString(dtype).c_str(), "fp16, fp32 or bf16"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (typeSize <= 0), OP_LOGE(context, "typeSize is invalid %d, please check.", typeSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Pows(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4Pows enter.");

    auto compileInfo = context->GetCompiledInfo<PowsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "TilingPrepare4Pows fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "TilingPrepare4Pows fail to get ub size."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepare4Pows exit.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTillingParam(const gert::TilingContext* context, TilingParam& tilingParam)
{
    auto inputShape = context->GetInputTensor(INPUT_IDX)->GetStorageShape();
    // fuse dims
    int64_t x{1};
    for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
        x *= inputShape.GetDim(i);
    }

    auto compileInfo = reinterpret_cast<const PowsCompileInfo*>(context->GetCompileInfo());
    tilingParam.x = x;
    tilingParam.coreNum = compileInfo->totalCoreNum;
    tilingParam.ubSize = compileInfo->ubSizePlatForm;

    OP_LOGI(
        context, "tilingParm is x: %ld, coreNum: %ld, ubSize: %ld", tilingParam.x, tilingParam.coreNum,
        tilingParam.ubSize);
    return ge::GRAPH_SUCCESS;
}

static void GetTillingData(ge::DataType dtype, TilingParam& tilingParam, PowsTilingData& tilingData)
{
    if (dtype == ge::DT_FLOAT16) {
        GetFp16TilingData(tilingData, tilingParam);
    } else if (dtype == ge::DT_BF16) {
        GetBf16TilingData(tilingData, tilingParam);
    } else {
        GetFp32TilingData(tilingData, tilingParam);
    }
}

static ge::graphStatus Tiling4Pows(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4Pows enter.");
    OP_CHECK_IF(
        CheckInputParams(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "InputParams not valid."),
        return ge::GRAPH_FAILED);

    TilingParam tilingParam;
    OP_CHECK_IF(
        GetTillingParam(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Get Tiling Param Failed."),
        return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    PowsTilingData tilingData;

    GetTillingData(dtype, tilingParam, tilingData);

    OP_CHECK_IF(
        SetTilingDataForPows(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "PowsSetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE + tilingParam.coreNum * BYTES_ONE_BLOCK;

    OP_LOGI(
        context,
        "tilingData is bufSize: %ld, tilingKey: %ld, numPerCore: %ld, realCoreNum: %ld, \
           mainCoreLoopNum: %ld, mainCoreTailLength: %ld, tailCoreloopNum: %ld, tailCoreTailLength: %ld, \
           dataLength: %ld, blockSize: %ld",
        tilingData.get_bufSize(), tilingData.get_tilingKey(), tilingData.get_numPerCore(), tilingData.get_realCoreNum(),
        tilingData.get_mainCoreLoopNum(), tilingData.get_mainCoreTailLength(), tilingData.get_tailCoreLoopNum(),
        tilingData.get_tailCoreTailLength(), tilingData.get_dataLength(), tilingData.get_blockSize());

    OP_LOGD(context, "Tiling4Pows exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Pows).Tiling(Tiling4Pows).TilingParse<PowsCompileInfo>(TilingPrepare4Pows);
} // namespace optiling
