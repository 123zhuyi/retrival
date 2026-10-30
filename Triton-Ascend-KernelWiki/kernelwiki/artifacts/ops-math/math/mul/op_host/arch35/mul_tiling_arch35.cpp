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
 * \file mul_tiling_arch35.cpp
 * \brief mul_tiling source file
 */

#include <graph/utils/type_utils.h>
#include <unordered_map>
#include <functional>
#include "mul_tiling_arch35.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/mul/op_kernel/arch35/mul_dag.h"
#include "math/mul/op_kernel/arch35/mul_struct.h"

namespace optiling
{
using namespace AscendC;
using namespace ge;
using namespace MulDag;
using namespace Ops::Base;

constexpr static uint64_t MUL_COMMON_TILING_PRIORITY = 0;
constexpr static int64_t DCACHE_SIZE = 32 * 1024;
constexpr static std::size_t HASH_PRIME = 31;
constexpr static std::size_t HASH_INIT = 17;
constexpr static std::size_t HASH_SHIFT_3 = 3;
constexpr static std::size_t HASH_SHIFT_5 = 5;

// 封装分块操作逻辑，将参数类型改为 gert::TilingContext*
template <typename OpDag>
ge::graphStatus DoTiling(gert::TilingContext* context, uint64_t& tilingKey,
                         int64_t extraSize = 0, int64_t extraBufferNum = 0)
{
    BroadcastBaseTiling<OpDag> brcBaseTiling(context);
    OP_CHECK_IF((brcBaseTiling.DoTiling(extraSize, extraBufferNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "Broadcast template do base tiling failed."),
        return ge::GRAPH_FAILED);
    tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    return ge::GRAPH_SUCCESS;
}

namespace {
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

using TilingFunc = std::function<ge::graphStatus(MulTiling*)>;

const std::unordered_map<DtypeCombination, TilingFunc, DtypeCombinationHash> DTYPE_MAP = {
    {{ge::DT_INT8, ge::DT_INT8, ge::DT_INT8},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter int8 branch.");
         return DoTiling<MulInt8Op::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter uint8 branch.");
         return DoTiling<MulUint8Op::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter bool branch.");
         return DoTiling<MulBoolOp::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter mix float branch.");
         return DoTiling<MulMixFpOp<bfloat16_t, float, float>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter mix float branch.");
         return DoTiling<MulMixFpOp<float, bfloat16_t, float>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter mix float branch.");
         return DoTiling<MulMixFpOp<half, float, float>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter mix float branch.");
         return DoTiling<MulMixFpOp<float, half, float>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter bf16 branch.");
         return DoTiling<MulXfp16Op<bfloat16_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter fp16 branch.");
         return DoTiling<MulXfp16Op<half>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter no cast branch.");
         return DoTiling<MulOp<float>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_INT32, ge::DT_INT32, ge::DT_INT32},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter no cast branch.");
         return DoTiling<MulOp<int32_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_INT64, ge::DT_INT64, ge::DT_INT64},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter no cast branch.");
         return DoTiling<MulOp<int64_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_INT16, ge::DT_INT16, ge::DT_INT16},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter no cast branch.");
         return DoTiling<MulOp<int16_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter double branch.");
         return DoTiling<MulDoubleOp<double>::OpDag>(tiling->GetContext(), tiling->tilingKey_, DCACHE_SIZE, 0);
     }},
    {{ge::DT_COMPLEX32, ge::DT_COMPLEX32, ge::DT_COMPLEX32},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter complex32 branch.");
         return DoTiling<MulComplex32Op<int32_t, int64_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_COMPLEX64, ge::DT_COMPLEX64, ge::DT_COMPLEX64},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter complex64 branch.");
         return DoTiling<MulOp<int64_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }}};
}  // 匿名命名空间结束

ge::graphStatus MulTiling::GetShapeAttrsInfo() {
    return ge::GRAPH_SUCCESS;
}

bool MulTiling::IsCapable() {
    return true;
}

ge::graphStatus MulTiling::DoOpTiling() {
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDType = outputDesc->GetDataType();

    OP_LOGD("MulTiling", "Input0DType is: %s, input1DType is: %s, outputDtype is: %s.",
            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(input1DType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputDType).c_str());

    DtypeCombination key = {input0DType, input1DType, outputDType};
    auto it = DTYPE_MAP.find(key);
    if (it != DTYPE_MAP.end()) {
        return it->second(this);
    }
    std::string dtypeMsg = Ops::Base::ToString(input0DType) + ", " + Ops::Base::ToString(input1DType) + " and " +
                           Ops::Base::ToString(outputDType);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        context_->GetNodeName(), "x1, x2 and y", dtypeMsg.c_str(),
        "The dtypes of x1, x2 and y must be within the range (all int8, all uint8, all bool, all fp16, all bf16, "
        "all float, all int32, all int64, all int16, all double, all complex32, all complex64, (bf16, float, float), "
        "(float, bf16, float), (fp16, float, float) or (float, fp16, float))");
    return ge::GRAPH_FAILED;
}

ge::graphStatus MulTiling::DoLibApiTiling() {
    return ge::GRAPH_SUCCESS;
}

uint64_t MulTiling::GetTilingKey() const {
    return tilingKey_;
}

ge::graphStatus MulTiling::GetWorkspaceSize() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulTiling::PostTiling() {
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulTiling::GetPlatformInfo() {
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

ge::graphStatus TilingForMul(gert::TilingContext* context) {
    OP_LOGD("MulTiling", "Enter TilingForMul");
    if (context == nullptr) {
        OP_LOGE("MulTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Enter ascendc MulTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMul([[maybe_unused]] gert::TilingParseContext *context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Mul).Tiling(TilingForMul).TilingParse<BroadcastCompileInfo>(TilingPrepareForMul);

REGISTER_OPS_TILING_TEMPLATE(Mul, MulTiling, MUL_COMMON_TILING_PRIORITY);
}  // namespace optiling