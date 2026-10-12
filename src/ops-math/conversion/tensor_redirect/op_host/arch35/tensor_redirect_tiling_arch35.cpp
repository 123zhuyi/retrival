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
 * \file tensor_redirect_tiling_arch35.cpp
 * \brief
 */

#include "tensor_redirect_tiling_arch35.h"

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/tensor_redirect_tiling_data.h"
#include "../../op_kernel/arch35/tensor_redirect_tiling_key.h"

using namespace ge;

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_OUTPUT_X = 0;
constexpr int64_t UB_FACTOR_MIN_BETY = 2048; // UB 单块下界（字节）
constexpr int64_t N_BUFFER = 2;              // double buffer
constexpr int64_t ONE_BLK_BYTE = 32;         // ubblock_size
constexpr size_t MIN_RANK = 1;               // spec inputs[0].rank_range
constexpr size_t MAX_RANK = 8;

static int64_t GetRemainder(int64_t uValue, int64_t dValue)
{
    if (dValue == 0) {
        return uValue;
    }
    return uValue % dValue;
}

// dtype 校验
static ge::graphStatus CheckTensorRedirectDtype(const gert::TilingContext* context)
{
    static const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT8,  ge::DT_INT32,
                                                          ge::DT_UINT8,   ge::DT_INT64,  ge::DT_INT16, ge::DT_UINT16,
                                                          ge::DT_UINT64,  ge::DT_UINT32, ge::DT_BF16};

    auto inputXPtr = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(supportedDtype.count(xDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    "TensorRedirect", "x", Ops::Base::ToString(xDtype).c_str(),
                    "dtype must be one of [DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, "
                    "DT_INT16, DT_UINT16, DT_UINT64, DT_UINT32, DT_BF16]"),
                return ge::GRAPH_FAILED);

    auto outputXPtr = context->GetOutputDesc(INDEX_OUTPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputXPtr);
    auto yDtype = outputXPtr->GetDataType();
    OP_CHECK_IF(yDtype != xDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    "TensorRedirect", "x and output_x",
                    (Ops::Base::ToString(xDtype) + " and " + Ops::Base::ToString(yDtype)).c_str(),
                    "x and output_x must have the same dtype"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// rank + shape 校验
static ge::graphStatus CheckTensorRedirectShape(const gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    // CheckRank: 1 <= rank(x) <= 8，否则 shape_mismatch
    size_t xRank = xShape.GetDimNum();
    OP_CHECK_IF(xRank < MIN_RANK || xRank > MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("TensorRedirect", "x", std::to_string(xRank).c_str(),
                                                         "rank must be within [1, 8]"),
                return ge::GRAPH_FAILED);

    // CheckDim: Tiling 是外部输入边界，concrete shape 的每一维必须非负；
    // -1/-2 等动态占位符或非法负值不能进入切分计算
    for (size_t i = 0; i < xRank; ++i) {
        OP_CHECK_IF(
            xShape.GetDim(i) < 0,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("TensorRedirect", "x", std::to_string(xShape.GetDim(i)).c_str(),
                                                     "every dim of the concrete shape must be non-negative"),
            return ge::GRAPH_FAILED);
    }

    auto yShapePtr = context->GetOutputShape(INDEX_OUTPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    // CheckShape: output_x.shape == x.shape（逐维严格相等）
    OP_CHECK_IF(xShape != yShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "TensorRedirect", "x and output_x",
                    (Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(yShape)).c_str(),
                    "x and output_x must have the same shape"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 多核切分：由 ubFactor 反推 uo/usedCoreNum/blockFactor
static void CalcBlockFactor(TensorRedirectTilingParam& tilingParam, int64_t numel)
{
    tilingParam.uo = CeilDiv(numel, tilingParam.ubFactor);
    tilingParam.tailBlockTailUbFactor = GetRemainder(numel, tilingParam.ubFactor);

    int64_t coreData = CeilDiv(tilingParam.uo, tilingParam.totalCoreNum);
    tilingParam.usedCoreNum = CeilDiv(tilingParam.uo, coreData);
    tilingParam.blockFactor = CeilDiv(tilingParam.uo, tilingParam.usedCoreNum);
    tilingParam.tailBlockFactor = tilingParam.uo - (tilingParam.usedCoreNum - 1) * tilingParam.blockFactor;
    if (tilingParam.tailBlockTailUbFactor == 0) {
        tilingParam.tailBlockTailUbFactor = tilingParam.ubFactor;
    }
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, TensorRedirectTilingParam& tilingParam,
                                int64_t numel)
{
    // maxUbAvailable = ubSize / N_BUFFER / bytesForOneData
    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / tilingParam.bytesForOneData;
    OP_CHECK_IF(maxUbAvailable <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "maxUbAvailable", "0",
                                                      "ubSize too small for N_BUFFER * bytesForOneData"),
                return ge::GRAPH_FAILED);

    tilingParam.ubFactor = (numel >= maxUbAvailable) ? maxUbAvailable : numel;
    CalcBlockFactor(tilingParam, numel);

    // 核已用满，或每核多次循环 -> 无需提核优化
    if (tilingParam.usedCoreNum == tilingParam.totalCoreNum || tilingParam.blockFactor > 1) {
        return ge::GRAPH_SUCCESS;
    }

    // 单核守卫：totalCoreNum==1 时跳过提核优化
    if (tilingParam.totalCoreNum <= 1) {
        return ge::GRAPH_SUCCESS;
    }

    // 小 shape 提核优化：核未用满且每核仅一次循环时，缩小 ubFactor 以摊到更多核
    if (GetRemainder(numel, tilingParam.totalCoreNum) == 0) {
        tilingParam.ubFactor = FloorDiv(numel, tilingParam.totalCoreNum);
    } else {
        tilingParam.ubFactor = FloorDiv(numel, tilingParam.totalCoreNum - 1);
    }
    // 32B 对齐
    tilingParam.ubFactor = CeilAlign(tilingParam.ubFactor, ONE_BLK_BYTE / tilingParam.bytesForOneData);
    // 下界钳制
    int64_t ubFactorMin = UB_FACTOR_MIN_BETY / tilingParam.bytesForOneData;
    tilingParam.ubFactor = tilingParam.ubFactor < ubFactorMin ? ubFactorMin : tilingParam.ubFactor;
    CalcBlockFactor(tilingParam, numel);
    return ge::GRAPH_SUCCESS;
}

// 获取平台参数（核数/UB 容量），CompileInfo 优先，缺失时回退查 platform
static ge::graphStatus GetPlatformParams(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    auto compileInfo = reinterpret_cast<const TensorRedirectCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->coreNum > 0 && compileInfo->ubSize > 0) {
        coreNum = compileInfo->coreNum; // GE 图路径
        ubSize = compileInfo->ubSize;
        return ge::GRAPH_SUCCESS;
    }

    auto platformInfoPtr = context->GetPlatformInfo(); // ACLNN 单算子路径回退
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum = ascendcPlatform.GetCoreNumAiv(); // AIV_ONLY -> 取 AIV 核数
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "TensorRedirect: failed to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizeTmp = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTmp);
    ubSize = static_cast<int64_t>(ubSizeTmp);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "TensorRedirect: failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    auto compileInfo = reinterpret_cast<const TensorRedirectCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->libApiWorkspaceSize > 0) {
        workspaces[0] = static_cast<size_t>(compileInfo->libApiWorkspaceSize);
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaces[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

// 空 Tensor 防护：numel==0 前置拦截
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context)
{
    TensorRedirectTilingData* tiling = context->GetTilingData<TensorRedirectTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TensorRedirectTilingData), 0, sizeof(TensorRedirectTilingData)) != EOK,
                OP_LOGE(context, "TensorRedirect: memset tiling data error (empty tensor)."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetWorkspace(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "TensorRedirect: SetWorkspace error."),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(1);
    ASCENDC_TPL_SEL_PARAM(context, TPL_SCH_MODE_0);
    return ge::GRAPH_SUCCESS; // 不下发有效计算
}

static ge::graphStatus Tiling4TensorRedirect(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4TensorRedirect running begin.");

    OP_CHECK_IF(CheckTensorRedirectDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TensorRedirect: the dtype check failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorRedirectShape(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TensorRedirect: the shape check failed."), return ge::GRAPH_FAILED);

    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    // 1D 线性展平，不解释 stride/rank
    int64_t numel = xShapePtr->GetStorageShape().GetShapeSize();

    // 溢出防护：GetShapeSize() 在维度乘积溢出 int64_t 时返回 kInvalidDimValue，不会自行报错。
    // 必须在 numel == 0 判断之前拦截，否则负的 numel 会穿透到 DoTiling 产生 usedCoreNum == 0。
    OP_CHECK_IF(numel < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("TensorRedirect", "x",
                                                      Ops::Base::ToString(xShapePtr->GetStorageShape()).c_str(),
                                                      "the product of all dims overflows int64_t"),
                return ge::GRAPH_FAILED);

    // 空 Tensor 早返回
    if (numel == 0) {
        OP_LOGD(context, "TensorRedirect: empty tensor, skip kernel computation.");
        return HandleEmptyTensor(context);
    }

    TensorRedirectTilingParam tilingParam;
    OP_CHECK_IF(GetPlatformParams(context, tilingParam.totalCoreNum, tilingParam.ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TensorRedirect: failed to get platform params."), return ge::GRAPH_FAILED);

    auto inputXPtr = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    tilingParam.bytesForOneData = ge::GetSizeByDataType(inputXPtr->GetDataType());
    OP_CHECK_IF(tilingParam.bytesForOneData <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "bytesForOneData", "0",
                                                      "failed to get the size of dtype"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(DoTiling(context, tilingParam, numel) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TensorRedirect: DoTiling failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetWorkspace(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "TensorRedirect: SetWorkspace error."),
                return ge::GRAPH_FAILED);

    // TilingData 缓冲区容量由 kernel 侧 REGISTER_TILING_DEFAULT 决定；
    // GetTilingData<T>() 内部已调用 SetDataSize(sizeof(T))，此处无需手工设置。
    TensorRedirectTilingData* tiling = context->GetTilingData<TensorRedirectTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TensorRedirectTilingData), 0, sizeof(TensorRedirectTilingData)) != EOK,
                OP_LOGE(context, "TensorRedirect: memset tiling data error."), return ge::GRAPH_FAILED);

    tiling->usedCoreNum = tilingParam.usedCoreNum;
    tiling->blockFactor = tilingParam.blockFactor;
    tiling->tailBlockFactor = tilingParam.tailBlockFactor;
    tiling->ubFactor = tilingParam.ubFactor;
    tiling->tailBlockTailUbFactor = tilingParam.tailBlockTailUbFactor;

    context->SetBlockDim(tilingParam.usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, TPL_SCH_MODE_0);

    OP_LOGD(context,
            "TensorRedirect tilingData: usedCoreNum:%ld, ubFactor:%ld, tailBlockTailUbFactor:%ld, "
            "blockFactor:%ld, tailBlockFactor:%ld",
            tiling->usedCoreNum, tiling->ubFactor, tiling->tailBlockTailUbFactor, tiling->blockFactor,
            tiling->tailBlockFactor);
    return ge::GRAPH_SUCCESS;
}

// CompileInfo 来自 platform
static ge::graphStatus TilingPrepare4TensorRedirect(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TensorRedirect", "tiling parse context is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<TensorRedirectCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv(); // AIV_ONLY -> 取 AIV 核数
    OP_CHECK_IF(compileInfo->coreNum <= 0, OP_LOGE(context, "TensorRedirect: failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(compileInfo->ubSize <= 0, OP_LOGE(context, "TensorRedirect: failed to get ub size."),
                return ge::GRAPH_FAILED);
    compileInfo->libApiWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TensorRedirect)
    .Tiling(Tiling4TensorRedirect)
    .TilingParse<TensorRedirectCompileInfo>(TilingPrepare4TensorRedirect);

} // namespace optiling
