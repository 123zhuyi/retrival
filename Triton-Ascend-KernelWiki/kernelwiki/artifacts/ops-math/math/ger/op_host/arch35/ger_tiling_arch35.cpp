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
 * \file ger_tiling_arch35.cpp
 * \brief ger_tiling source file
 */

#include <graph/utils/type_utils.h>
#include <unordered_map>
#include <functional>
#include "ger_tiling_arch35.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/ger/op_kernel/arch35/ger_dag.h"
#include "math/ger/op_kernel/arch35/ger_struct.h"

namespace optiling
{
using namespace AscendC;
using namespace ge;
using namespace GerDag;
using namespace Ops::Base;

constexpr static uint64_t GER_COMMON_TILING_PRIORITY = 0;
constexpr static std::size_t HASH_PRIME = 31;
constexpr static std::size_t HASH_INIT = 17;
constexpr static std::size_t HASH_SHIFT_3 = 3;
constexpr static std::size_t HASH_SHIFT_5 = 5;
constexpr static uint32_t INPUT_IDX_X1 = 0;
constexpr static uint32_t INPUT_IDX_X2 = 1;
constexpr static uint32_t OUTPUT_IDX_Y = 0;
constexpr static int32_t NTWO = 2;

// 封装分块操作逻辑，将参数类型改为 gert::TilingContext*
template <typename OpDag>
ge::graphStatus DoGerTiling(gert::TilingContext* context, uint64_t& tilingKey,
                            int64_t extraSize = 0, int64_t extraBufferNum = 0)
{
    BroadcastBaseTiling<OpDag> brcBaseTiling(context);
    auto x1StorageShape = context->GetInputShape(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1StorageShape);
    gert::Shape x1Shape_ = EnsureNotScalar(x1StorageShape->GetStorageShape());
    auto x2StorageShape = context->GetInputShape(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2StorageShape);
    gert::Shape x2Shape_ = EnsureNotScalar(x2StorageShape->GetStorageShape());

    gert::Shape x1ReShape_;
    x1ReShape_.AppendDim(x1Shape_.GetDim(0));
    x1ReShape_.AppendDim(1);

    vector<gert::Shape> inputShapes;
    inputShapes.push_back(x1ReShape_);
    inputShapes.push_back(x2Shape_);

    brcBaseTiling.SetOpInputStorageShapes(inputShapes);
    OP_CHECK_IF((brcBaseTiling.DoTiling(extraSize, extraBufferNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "Broadcast template do base tiling failed."),
        return ge::GRAPH_FAILED);
    tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    return ge::GRAPH_SUCCESS;
}

// 定义数据类型组合结构体
struct DtypeCombination {
    ge::DataType input0;
    ge::DataType input1;
    ge::DataType output;

    // 重载相等运算符，用于哈希表的查找
    bool operator==(const DtypeCombination& other) const {
        return input0 == other.input0 && input1 == other.input1 && output == other.output;
    }
};

// 自定义哈希函数
struct DtypeCombinationHash {
    std::size_t operator()(const DtypeCombination& comb) const {
        const std::size_t prime = HASH_PRIME;
        std::size_t hash = HASH_INIT;
        hash = hash * prime + std::hash<ge::DataType>()(comb.input0);
        hash = hash * prime + (std::hash<ge::DataType>()(comb.input1) << HASH_SHIFT_3);
        hash = hash * prime + (std::hash<ge::DataType>()(comb.output) << HASH_SHIFT_5);
        return hash;
    }
};

using TilingFunc = std::function<ge::graphStatus(GerTiling*)>;

const std::unordered_map<DtypeCombination, TilingFunc, DtypeCombinationHash> GER_DTYPE_MAP = {
    {{ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
     [](GerTiling* tiling) {
         OP_LOGD("GerTiling", "Enter bf16 branch.");
         return DoGerTiling<GerOp<bfloat16_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
     [](GerTiling* tiling) {
         OP_LOGD("GerTiling", "Enter fp16 branch.");
         return DoGerTiling<GerOp<half>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
     [](GerTiling* tiling) {
         OP_LOGD("GerTiling", "Enter no cast branch.");
         return DoGerTiling<GerOp<float>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }}};

ge::graphStatus GerTiling::GetShapeAttrsInfo() {
    return ge::GRAPH_SUCCESS;
}

bool GerTiling::IsCapable() {
    return true;
}

bool GerTiling::CheckShapes()
{
    auto x1StorageShape = context_->GetInputShape(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1StorageShape);
    gert::Shape x1Shape_ = EnsureNotScalar(x1StorageShape->GetStorageShape());
    auto x2StorageShape = context_->GetInputShape(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2StorageShape);
    gert::Shape x2Shape_ = EnsureNotScalar(x2StorageShape->GetStorageShape());
    auto yStorageShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorageShape);
    gert::Shape yShape_ = EnsureNotScalar(yStorageShape->GetStorageShape());
    // 校验空tensor
    if (x1Shape_.GetShapeSize() <= 0L || x2Shape_.GetShapeSize() <= 0L || yShape_.GetShapeSize() <= 0L) {
        std::string sizesStr = std::to_string(x1Shape_.GetShapeSize()) + ", " +
                               std::to_string(x2Shape_.GetShapeSize()) + " and " +
                               std::to_string(yShape_.GetShapeSize());
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "x1, x2 and y",
            sizesStr.c_str(), "The shape sizes of x1, x2 and y should be greater than 0");
        return false;
    }
    // 校验x1，x2为1维
    OP_CHECK_IF(x1Shape_.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x1", std::to_string(x1Shape_.GetDimNum()).c_str(), "1"),
        return false);
    OP_CHECK_IF(x2Shape_.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x2", std::to_string(x2Shape_.GetDimNum()).c_str(), "1"),
        return false);

    // 校验y
    OP_CHECK_IF(
        yShape_.GetDimNum() != NTWO ||
            (yShape_.GetDim(0) != x1Shape_.GetDim(0) || yShape_.GetDim(1) != x2Shape_.GetDim(0)),
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "y", Ops::Base::ToString(yShape_).c_str(),
            ("[" + std::to_string(x1Shape_.GetDim(0)) + "," + std::to_string(x2Shape_.GetDim(0)) + "]").c_str()),
        return false);
    return true;
}

ge::graphStatus GerTiling::DoOpTiling() {
    OP_CHECK_IF(!CheckShapes(), OP_LOGE(context_->GetNodeName(), "CheckShapes error!"), return ge::GRAPH_FAILED);

    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDType = outputDesc->GetDataType();

    OP_LOGD("GerTiling", "Input0DType is: %s, input1DType is: %s, outputDtype is: %s.",
            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(input1DType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputDType).c_str());

    DtypeCombination key = {input0DType, input1DType, outputDType};
    auto it = GER_DTYPE_MAP.find(key);
    if (it != GER_DTYPE_MAP.end()) {
        return it->second(this);
    }
    std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                            ge::TypeUtils::DataTypeToSerialString(input1DType) + " and " +
                            ge::TypeUtils::DataTypeToSerialString(outputDType);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2 and y", dtypesStr.c_str(),
        "The dtypes of x1, x2 and y must be the same, and must be float16, bfloat16 or float");
    return ge::GRAPH_FAILED;
}

ge::graphStatus GerTiling::DoLibApiTiling() {
    return ge::GRAPH_SUCCESS;
}

uint64_t GerTiling::GetTilingKey() const {
    return tilingKey_;
}

ge::graphStatus GerTiling::GetWorkspaceSize() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GerTiling::PostTiling() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GerTiling::GetPlatformInfo() {
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const BroadcastCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"), return ge::GRAPH_FAILED);
        ubSize_ = compileInfoPtr->ubSize;
        OP_LOGD(context_->GetNodeName(), "Get ubSize form compileInfo is: %ld", ubSize_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
        OP_LOGD(context_->GetNodeName(), "Get ubSize form ascendcPlatform is: %ld", ubSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGer(gert::TilingContext* context) {
    OP_LOGD("GerTiling", "Enter TilingForGer");
    if (context == nullptr) {
        OP_LOGE("GerTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Enter ascendc GerTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForGer([[maybe_unused]] gert::TilingParseContext *context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Ger).Tiling(TilingForGer).TilingParse<BroadcastCompileInfo>(TilingPrepareForGer);

REGISTER_OPS_TILING_TEMPLATE(Ger, GerTiling, GER_COMMON_TILING_PRIORITY);
}  // namespace optiling