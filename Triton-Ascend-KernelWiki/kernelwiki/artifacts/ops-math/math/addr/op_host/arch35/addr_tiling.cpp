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
 * \file addr_tiling.cpp
 * \brief addr_tiling source file
 */

#include "addr_tiling.h"
#include <graph/utils/type_utils.h>
#include "../../op_kernel/arch35/addr_struct.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_host/util/fp16.h"
#include "util/bfloat16.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace ge;
using namespace ops;

namespace optiling {

constexpr static uint64_t ADDR_COMMON_TILING_PRIORITY = 0;
constexpr static uint32_t INPUT_IDX_X1 = 0;
constexpr static uint32_t INPUT_IDX_X2 = 1;
constexpr static uint32_t INPUT_IDX_X3 = 2;
constexpr static uint32_t INPUT_IDX_BETA = 3;
constexpr static uint32_t INPUT_IDX_ALPHA = 4;
constexpr static uint32_t OUTPUT_IDX_Y = 0;
constexpr static int32_t nTwo = 2;
constexpr static uint16_t FP16_BF16_ZERO = 0x0000;
constexpr static uint16_t FP16_ONE = 0x3C00;
constexpr static uint16_t BF16_ONE = 0x3F80;
constexpr static uint64_t TILINGKET_ZERO = 0;
constexpr static uint64_t TILINGKET_NOT_ZERO = 1;

static ge::graphStatus TilingPrepareForAddr(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AddrCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddrTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool AddrTiling::IsCapable()
{
    return true;
}

bool AddrTiling::CheckDtype()
{
    auto x1Desc = context_->GetInputDesc(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    x1Dtype_ = x1Desc->GetDataType();
    auto x2Desc = context_->GetInputDesc(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    x2Dtype_ = x2Desc->GetDataType();
    auto x3Desc = context_->GetInputDesc(INPUT_IDX_X3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x3Desc);
    x3Dtype_ = x3Desc->GetDataType();
    auto betaDesc = context_->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, betaDesc);
    betaDtype_ = betaDesc->GetDataType();
    auto alphaDesc = context_->GetInputDesc(INPUT_IDX_ALPHA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alphaDesc);
    alphaDtype_ = alphaDesc->GetDataType();
    auto yDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    yDtype_ = yDesc->GetDataType();
    if (x1Dtype_ != x2Dtype_ || x2Dtype_ != x3Dtype_ || x3Dtype_ != betaDtype_ || betaDtype_ != alphaDtype_ ||
        alphaDtype_ != yDtype_) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(x1Dtype_) + ", " +
            ge::TypeUtils::DataTypeToSerialString(x2Dtype_) + ", " +
            ge::TypeUtils::DataTypeToSerialString(x3Dtype_) + ", " +
            ge::TypeUtils::DataTypeToSerialString(betaDtype_) + ", " +
            ge::TypeUtils::DataTypeToSerialString(alphaDtype_) + " and " +
            ge::TypeUtils::DataTypeToSerialString(yDtype_);
        std::string reasonMsg = "Dtypes of x1, x2, x3, beta, alpha and y must be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(),
            "x1, x2, x3, beta, alpha and y", dtypeMsg.c_str(), reasonMsg.c_str());
        return false;
    }
    return true;
}

bool AddrTiling::CheckBroadcast()
{
    int64_t n = x2Shape_.GetDim(0);
    int64_t m = x3Shape_.GetDim(0);
    // 校验x1
    size_t x1DimNum = x1Shape_.GetDimNum();
    if (x1DimNum == 1) {
        int64_t inM = x1Shape_.GetDim(0);
        if (inM == 1 || inM == m) {
            return true;
        }
        std::string shapeMsg = "[" + std::to_string(inM) + "] vs x2: [" +
            std::to_string(n) + "], x3: [" + std::to_string(m) + "]";
        std::string reasonMsg = "x1 shape is not broadcastable to outer product of x2 and x3";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(),
            "x1, x2 and x3", shapeMsg.c_str(), reasonMsg.c_str());
    } else if (x1DimNum == static_cast<size_t>(nTwo)) {
        int64_t inN = x1Shape_.GetDim(0);
        int64_t inM = x1Shape_.GetDim(1);
        if ((inN == 1 || inN == n) && (inM == 1 || inM == m)) {
            return true;
        }
        std::string shapeMsg = "[" + std::to_string(inN) + ", " + std::to_string(inM) +
            "] vs x2: [" + std::to_string(n) + "], x3: [" + std::to_string(m) + "]";
        std::string reasonMsg = "x1 shape is not broadcastable to outer product of x2 and x3";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(),
            "x1, x2 and x3", shapeMsg.c_str(), reasonMsg.c_str());
    } else {
        std::string dimMsg = std::to_string(x1DimNum);
        std::string reasonMsg = "x1 dim should be 1 or 2";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x1",
            dimMsg.c_str(), reasonMsg.c_str());
    }
    return false;
}

bool AddrTiling::CheckShapes()
{
    auto x1StorageShape = context_->GetInputShape(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1StorageShape);
    x1Shape_ = EnsureNotScalar(x1StorageShape->GetStorageShape());
    auto x2StorageShape = context_->GetInputShape(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2StorageShape);
    x2Shape_ = EnsureNotScalar(x2StorageShape->GetStorageShape());
    auto x3StorageShape = context_->GetInputShape(INPUT_IDX_X3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x3StorageShape);
    x3Shape_ = EnsureNotScalar(x3StorageShape->GetStorageShape());
    auto betaStorageShape = context_->GetInputShape(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, betaStorageShape);
    betaShape_ = EnsureNotScalar(betaStorageShape->GetStorageShape());
    auto alphaStorageShape = context_->GetInputShape(INPUT_IDX_ALPHA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alphaStorageShape);
    alphaShape_ = EnsureNotScalar(alphaStorageShape->GetStorageShape());
    auto yStorageShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
    yShape_ = EnsureNotScalar(yStorageShape->GetStorageShape());
    // 校验空tensor
    if (x1Shape_.GetShapeSize() <= 0L || x2Shape_.GetShapeSize() <= 0L || x3Shape_.GetShapeSize() <= 0L ||
        yShape_.GetShapeSize() <= 0L) {
        std::string reasonMsg = "x1, x2, x3 and y do not support empty tensors";
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "x1, x2, x3 and y", "0", reasonMsg.c_str());
        return false;
    }
    // 校验x2，x3为1维
    if (x2Shape_.GetDimNum() != 1) {
        std::string dimMsg = std::to_string(x2Shape_.GetDimNum());
        std::string correctMsg = "1";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x2", dimMsg.c_str(), correctMsg.c_str());
        return false;
    }
    if (x3Shape_.GetDimNum() != 1) {
        std::string dimMsg = std::to_string(x3Shape_.GetDimNum());
        std::string correctMsg = "1";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x3", dimMsg.c_str(), correctMsg.c_str());
        return false;
    }
    // 校验x1满足broadcast
    if (!CheckBroadcast()) {
        return false;
    }
    // 校验y
    if (yShape_.GetDimNum() != nTwo ||
            (yShape_.GetDim(0) != x2Shape_.GetDim(0) || yShape_.GetDim(1) != x3Shape_.GetDim(0))) {
        std::string shapeMsg = "[" + std::to_string(yShape_.GetDim(0)) + ", " + std::to_string(yShape_.GetDim(1)) + "]";
        std::string correctMsg = "[" + std::to_string(x2Shape_.GetDim(0)) + ", " + std::to_string(x3Shape_.GetDim(0)) + "]";
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "y", shapeMsg.c_str(), correctMsg.c_str());
        return false;
    }
    return true;
}

template <typename T>
ge::graphStatus AddrTiling::GetConstData(uint32_t inputIdx, bool isEmpty, T emptyValue, T& data)
{
    if (isEmpty) {
        data = emptyValue;
    } else {
        auto tensor = context_->GetInputTensor(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, tensor);
        const T* value = tensor->GetData<T>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, value);
        data = value[0];
    }
    return ge::GRAPH_SUCCESS;
}

template <typename OpDag1, typename OpDag2, typename OpDag3, typename T>
ge::graphStatus AddrTiling::BroadcastTiling(T beta, T alpha)
{
    bool isBetaZero = (beta == 0);
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    vector<gert::Shape> inputShapes;
    if (isBetaZero) {
        inputShapes.push_back(x2ReShape_);
        inputShapes.push_back(x3Shape_);

        BroadcastBaseTiling<OpDag2> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(alpha);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TILINGKET_ZERO, TILINGKET_NOT_ZERO);
    } else {
        inputShapes.push_back(x1Shape_);
        inputShapes.push_back(x2ReShape_);
        inputShapes.push_back(x3Shape_);

        BroadcastBaseTiling<OpDag3> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(beta);
        brcBaseTiling.SetScalar(alpha);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        ret = brcBaseTiling.DoTiling();
        tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TILINGKET_NOT_ZERO, TILINGKET_NOT_ZERO);
    }
    return ret;
}

ge::graphStatus AddrTiling::DoOpTiling4Float(bool betaEmpty, bool alphaEmpty)
{
    float beta = 0;
    float alpha = 0;
    if (GetConstData<float>(INPUT_IDX_BETA, betaEmpty, float(1), beta) != ge::GRAPH_SUCCESS ||
            GetConstData<float>(INPUT_IDX_ALPHA, alphaEmpty, float(1), alpha) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context_->GetNodeName(), "beta = %f, alpha = %f", beta, alpha);
    return BroadcastTiling<
        AddrOp::AddrWithoutAlphaCommon<float>::OpDag, AddrOp::AddrWithoutBetaCommon<float>::OpDag,
        AddrOp::AddrWithBetaWithAlphaCommon<float>::OpDag, float>(beta, alpha);
}

ge::graphStatus AddrTiling::DoOpTiling4Half(bool betaEmpty, bool alphaEmpty)
{
    float beta = 0;
    float alpha = 0;
    uint16_t betaTmp = 0;
    uint16_t alphaTmp = 0;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (x1Dtype_ == ge::DT_FLOAT16) {
        if (GetConstData<uint16_t>(INPUT_IDX_BETA, betaEmpty, FP16_ONE, betaTmp) != ge::GRAPH_SUCCESS ||
                GetConstData<uint16_t>(INPUT_IDX_ALPHA, alphaEmpty, FP16_ONE, alphaTmp) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        beta = float(*(reinterpret_cast<fp16_t*>(&betaTmp)));
        alpha = float(*(reinterpret_cast<fp16_t*>(&alphaTmp)));
    } else {
        if (GetConstData<uint16_t>(INPUT_IDX_BETA, betaEmpty, BF16_ONE, betaTmp) != ge::GRAPH_SUCCESS ||
                GetConstData<uint16_t>(INPUT_IDX_ALPHA, alphaEmpty, BF16_ONE, alphaTmp) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        beta = float(*(reinterpret_cast<bfloat16*>(&betaTmp)));
        alpha = float(*(reinterpret_cast<bfloat16*>(&alphaTmp)));
    }
    OP_LOGI(context_->GetNodeName(), "beta = %f, alpha = %f", beta, alpha);
    ret = BroadcastTiling<
        AddrOp::AddrWithoutAlphaBf16Fp16<half>::OpDag, AddrOp::AddrWithoutBetaBf16Fp16<half>::OpDag,
        AddrOp::AddrWithBetaWithAlphaBf16Fp16<half>::OpDag, float>(beta, alpha);
    return ret;
}

ge::graphStatus AddrTiling::DoOpTiling()
{
    if (!CheckDtype()) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckShapes()) {
        return ge::GRAPH_FAILED;
    }
    x2ReShape_.AppendDim(x2Shape_.GetDim(0));
    x2ReShape_.AppendDim(1);

    bool betaEmpty = (betaShape_.GetShapeSize() == 0);
    bool alphaEmpty = (alphaShape_.GetShapeSize() == 0);

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (x1Dtype_ == ge::DT_FLOAT) {
        ret = DoOpTiling4Float(betaEmpty, alphaEmpty);
    } else if (x1Dtype_ == ge::DT_FLOAT16 || x1Dtype_ == ge::DT_BF16) {
        ret = DoOpTiling4Half(betaEmpty, alphaEmpty);
    } else if (x1Dtype_ == ge::DT_BOOL || x1Dtype_ == ge::DT_INT8) {
        int8_t tmpBeta = 0;
        int8_t tmpAlpha = 0;
        if (GetConstData<int8_t>(INPUT_IDX_BETA, betaEmpty, int8_t(1), tmpBeta) != ge::GRAPH_SUCCESS ||
                GetConstData<int8_t>(INPUT_IDX_ALPHA, alphaEmpty, int8_t(1), tmpAlpha) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int32_t beta = static_cast<int32_t>(tmpBeta);
        int32_t alpha = static_cast<int32_t>(tmpAlpha);
        OP_LOGI(context_->GetNodeName(), "beta = %d, alpha = %d", beta, alpha);
        ret = BroadcastTiling<
            AddrOp::AddrWithoutAlphaInt8::OpDag, AddrOp::AddrWithoutBetaInt8::OpDag,
            AddrOp::AddrWithBetaWithAlphaInt8::OpDag, int32_t>(beta, alpha);
    } else if (x1Dtype_ == ge::DT_UINT8) {
        uint8_t tmpBeta = 0;
        uint8_t tmpAlpha = 0;
        if (GetConstData<uint8_t>(INPUT_IDX_BETA, betaEmpty, uint8_t(1), tmpBeta) != ge::GRAPH_SUCCESS ||
                GetConstData<uint8_t>(INPUT_IDX_ALPHA, alphaEmpty, uint8_t(1), tmpAlpha) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int32_t beta = static_cast<int32_t>(tmpBeta);
        int32_t alpha = static_cast<int32_t>(tmpAlpha);
        OP_LOGI(context_->GetNodeName(), "beta = %d, alpha = %d", beta, alpha);
        ret = BroadcastTiling<
            AddrOp::AddrWithoutAlphaUint8::OpDag, AddrOp::AddrWithoutBetaUint8::OpDag,
            AddrOp::AddrWithBetaWithAlphaUint8::OpDag, int32_t>(beta, alpha);
    } else {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(x1Dtype_);
        std::string reasonMsg = "Input dtype is only support fp16, bf16, fp32, uint8, int8, bool";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x1", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus AddrTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AddrTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus AddrTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddrTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddrTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAddr(gert::TilingContext* context)
{
    OP_LOGD("AddrTiling", "Enter TilingForAddr");
    if (context == nullptr) {
        OP_LOGE("AddrTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const AddrCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD("AddrTiling", "Enter ascendc AddrTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(Addr)
    .Tiling(TilingForAddr)
    .TilingParse<AddrCompileInfo>(TilingPrepareForAddr)
    .TilingInputsDataDependency({INPUT_IDX_BETA, INPUT_IDX_ALPHA});

REGISTER_OPS_TILING_TEMPLATE(Addr, AddrTiling, ADDR_COMMON_TILING_PRIORITY);
} // namespace optiling
