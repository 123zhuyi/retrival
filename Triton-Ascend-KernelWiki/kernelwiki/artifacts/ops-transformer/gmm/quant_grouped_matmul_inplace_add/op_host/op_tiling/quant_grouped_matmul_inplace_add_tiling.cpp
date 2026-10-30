/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "register/op_impl_registry.h"
#include "quant_grouped_matmul_inplace_add_tiling.h"
#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace QuantGroupedMatmulInplaceAdd;
using namespace optiling::GmmConstant;
namespace optiling {

void QuantGroupedInplaceAddTiling::Reset()
{
    tilingData_ = QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams();
    return;
}

bool QuantGroupedInplaceAddTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    }
    inputParams_.transA = true;
    return true;
}

bool QuantGroupedInplaceAddTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "scaleDesc is nullptr."), return false);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "yDesc is nullptr."), return false);
    inputParams_.cDtype = yDesc->GetDataType();
    return CheckDtype();
}

bool QuantGroupedInplaceAddTiling::CheckDtype()
{
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_FLOAT,
               OP_LOGE(inputParams_.opName, "Input yRef dtype should be DT_FLOAT, actual dtype is %s.",
                                         ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
               return false);
    if (inputParams_.aDtype == ge::DT_HIFLOAT8 && inputParams_.bDtype == ge::DT_HIFLOAT8) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT,
                   OP_LOGE(
                       inputParams_.opName, "With DT_HIFLOAT8 inputs, scale2 dtype should be DT_FLOAT, actual dtype is %s.",
                       ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                   return false);
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                   OP_LOGE(
                       inputParams_.opName, "With DT_HIFLOAT8 inputs, scale1 dtype should be DT_FLOAT, actual dtype is %s.",
                       ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                   return false);
    } else if ((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2) &&
               (inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_FLOAT8_E5M2)) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0,
                   OP_LOGE(
                       inputParams_.opName,
                       "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, scale2 dtype should be DT_FLOAT8_E8M0, actual dtype is %s.",
                       ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                   return false);
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                   OP_LOGE(
                       inputParams_.opName,
                       "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, scale1 dtype should be DT_FLOAT8_E8M0, actual dtype is %s.",
                       ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                   return false);
    } else {
        OP_LOGE(inputParams_.opName, "Quant case with x1 dtype %s and x2 dtype %s is not supported.",
                  ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str());
        return false;
    }
    return true;
}

bool QuantGroupedInplaceAddTiling::SetQuantModeForQGmmInplaceAdd()
{
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }
    inputParams_.bQuantMode = optiling::QuantMode::PERCHANNEL_MODE;
    inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    return true;
}

bool QuantGroupedInplaceAddTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();

    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."), return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(scaleDimNum < 1,
               OP_LOGE(inputParams_.opName,
                                         "The dimension of scale1 should be positive integer, actual is %zu",
                                         scaleDimNum),
               return false);
    auto x1ScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "x1ScaleStorageShape is nullptr."), return false);
    const gert::Shape &x1ScaleShape = x1ScaleStorageShape->GetOriginShape();
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."),
               return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!SetQuantModeForQGmmInplaceAdd(),
               OP_LOGE(inputParams_.opName, "SetQuantModeForQGmmInplaceAdd failed."), return false);
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        OP_CHECK_IF(!CheckShapeForMxQuant(x1ScaleShape, wScaleShape),
                   OP_LOGE(inputParams_.opName, "CheckShapeForMxQuant failed."), return false);
    } else {
        OP_CHECK_IF(!CheckShapeForTCQuant(x1ScaleShape, wScaleShape),
                   OP_LOGE(inputParams_.opName, "CheckShapeForTCQuant failed."), return false);
    }

    SetKernelType();
    OP_CHECK_IF(!CheckCoreNum(),
                OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);   
    return true;
}

bool QuantGroupedInplaceAddTiling::CheckCoreNum() const
{
    auto aicNum = context_->GetCompileInfo<GMMCompileInfo>()->aicNum;
    auto aivNum = context_->GetCompileInfo<GMMCompileInfo>()->aivNum;
    OP_CHECK_IF(aicNum == 0,
               OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.", aicNum),
               return false);
    OP_CHECK_IF(aivNum != GmmConstant::CORE_RATIO * aicNum,
               OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.", aicNum, aivNum),
               return false);
    return true;
}

bool QuantGroupedInplaceAddTiling::CheckShapeForMxQuant(const gert::Shape &x1ScaleShape,
                                                        const gert::Shape &x2ScaleShape)
{
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
               OP_LOGE(inputParams_.opName,
                                         "The dimension of scale2 should be 3 in mx quant mode, but actual is %zu",
                                         x2ScaleDimNum),
               return false);
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
               OP_LOGE(inputParams_.opName,
                                         "The dim num of scale1 should be 3 in mx quant mode, but \
actual is %zu",
                                         x1ScaleDimNum),
               return false);
    auto xScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto xScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto xScaleMDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    auto wScaleNDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto expectedKDimValue = inputParams_.kSize / MXFP_BASEK_FACTOR + inputParams_.groupNum;
    OP_CHECK_IF(xScaleLastDim != MXFP_MULTI_BASE_SIZE || xScaleKDim != expectedKDimValue ||
                   xScaleMDim != inputParams_.mSize,
               OP_LOGE(inputParams_.opName, "In mx quant mode, the expected shape of scale1 is \
(%lu,%lu,%lu), but the actual is (%lu,%lu,%lu).",
                                         expectedKDimValue, inputParams_.mSize, MXFP_MULTI_BASE_SIZE, xScaleKDim,
                                         xScaleMDim, xScaleLastDim),
               return false);
    OP_CHECK_IF(wScaleLastDim != MXFP_MULTI_BASE_SIZE || wScaleKDim != expectedKDimValue ||
                   wScaleNDim != inputParams_.nSize,
               OP_LOGE(
                   inputParams_.opName, "In mx quant mode, the expected shape of scale2 is (%lu,%lu,%lu), \
but the actual is (%lu,%lu,%lu).",
                   expectedKDimValue, inputParams_.nSize, MXFP_MULTI_BASE_SIZE, wScaleKDim, wScaleNDim, wScaleLastDim),
               return false);
    return true;
}

bool QuantGroupedInplaceAddTiling::CheckShapeForTCQuant(const gert::Shape &x1ScaleShape,
                                                        const gert::Shape &x2ScaleShape)
{
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != 1 && x1ScaleDimNum != 2, // Max dim num in T-C quant: 2
               OP_LOGE(inputParams_.opName,
                                         "The dimension of scale1 should be 1 or 2 in T-C quant mode, but \
actual is %zu",
                                         x1ScaleDimNum),
               return false);
    auto lastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto firstDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    if (x1ScaleDimNum == 1) {
        OP_CHECK_IF(
            firstDim != inputParams_.groupNum,
            OP_LOGE(inputParams_.opName,
                                      "In T-C quant mode, the expected shape of scale1 is (%lu, ) or (%lu, 1), \
but the actual is (%lu, ).",
                                      inputParams_.groupNum, inputParams_.groupNum, firstDim),
            return false);
    } else {
        OP_CHECK_IF(
            firstDim != inputParams_.groupNum || lastDim != 1,
            OP_LOGE(inputParams_.opName,
                                      "In T-C quant mode, the expected shape of scale1 is (%lu, ) or (%lu, 1), \
but the actual is (%lu, %lu).",
                                      inputParams_.groupNum, inputParams_.groupNum, firstDim, lastDim),
            return false);
    }

    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != 2, // Max dim num in T-C quant: 2
               OP_LOGE(inputParams_.opName,
                                         "The dimension of scale2 should be 2 in T-C quant mode, but \
actual is %zu",
                                         x2ScaleDimNum),
               return false);
    lastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    firstDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    OP_CHECK_IF(firstDim != inputParams_.groupNum || lastDim != inputParams_.nSize,
               OP_LOGE(inputParams_.opName,
                                         "In T-C quant mode, the expected shape of scale2 is (%lu, %lu), \
but the actual is (%lu, %lu).",
                                         inputParams_.groupNum, inputParams_.nSize, firstDim, lastDim),
               return false);

    return true;
}

ge::graphStatus QuantGroupedInplaceAddTiling::DoOpTiling()
{
    tilingData_.quantGmmInplaceAddParams.groupNum = inputParams_.groupNum;
    tilingData_.quantGmmInplaceAddParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedInplaceAddTiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS,
               OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.M = inputParams_.mSize;
    tilingData_.mmTilingData.N = inputParams_.nSize;
    tilingData_.mmTilingData.Ka = inputParams_.kSize;
    tilingData_.mmTilingData.Kb = inputParams_.kSize;
    tilingData_.mmTilingData.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.mmTilingData.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.mmTilingData.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.mmTilingData.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.depthA1 = basicTiling_.depthA1;
    tilingData_.mmTilingData.depthB1 = basicTiling_.depthB1;
    tilingData_.mmTilingData.stepM = basicTiling_.stepM;
    tilingData_.mmTilingData.stepN = basicTiling_.stepN;
    tilingData_.mmTilingData.stepKa = basicTiling_.stepKa;
    tilingData_.mmTilingData.stepKb = basicTiling_.stepKb;
    tilingData_.mmTilingData.isBias = 0;
    tilingData_.mmTilingData.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.mmTilingData.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        tilingData_.mmTilingData.mxTypePara =
            (SCALER_FACTOR_MIN << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_MIN << SCALER_FACTOR_M_BIT);
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.mxTypePara +=
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.mmTilingData.mxTypePara +=
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedInplaceAddTiling::PostTiling()
{
    context_->SetBlockDim(aicoreParams_.aicNum);
    OP_CHECK_IF(sizeof(tilingData_) % sizeof(uint64_t) != 0,
               OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8",
                                         sizeof(tilingData_)),
               return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

void QuantGroupedInplaceAddTiling::PrintQuantParams()
{
    int32_t enable = CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }
    QuantGroupedMatmulInplaceAdd::QGmmInplaceAddParams &params = tilingData_.quantGmmInplaceAddParams;
    std::ostringstream oss;
    oss << "QGMMIAParams: groupNum = " << params.groupNum
        << ", groupListType = " << static_cast<uint32_t>(params.groupListType);
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
}

ASCENDC_EXTERN_C ge::graphStatus TilingGMMInplaceAdd(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGMMInplaceAdd(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    OP_CHECK_IF((compileInfoPtr->aicNum == 0 || compileInfoPtr->aivNum == 0 || compileInfoPtr->ubSize == 0 ||
                compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 ||
                compileInfoPtr->l0BSize == 0 || compileInfoPtr->l2Size == 0),
               OP_LOGE(context->GetNodeName(),
                                           "Platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, "
                                           "l0CSize=%lu, l0ASize=%lu, l0BSize=%lu, l2Size=%lu",
                                           compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize,
                                           compileInfoPtr->l1Size, compileInfoPtr->l0CSize, compileInfoPtr->l0ASize,
                                           compileInfoPtr->l0BSize, compileInfoPtr->l2Size),
               return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
              static_cast<int>(compileInfoPtr->socVersion));
    return ge::GRAPH_SUCCESS;
}
uint64_t QuantGroupedInplaceAddTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
        static_cast<uint64_t>(inputParams_.kernelType));
}

REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatmulInplaceAdd, QuantGroupedInplaceAddTiling, 0);
IMPL_OP_OPTILING(QuantGroupedMatmulInplaceAdd)
    .Tiling(TilingGMMInplaceAdd)
    .TilingParse<GMMCompileInfo>(TilingPrepareForGMMInplaceAdd); // register into the framework
} // namespace optiling