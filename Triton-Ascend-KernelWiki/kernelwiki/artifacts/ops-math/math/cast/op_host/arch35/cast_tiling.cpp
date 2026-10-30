/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cast_tiling.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"
#include "cast_tiling.h"
#include "math/cast/op_kernel/arch35/cast_struct.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/platform_util.h"

namespace optiling {
using namespace ge;

constexpr int64_t CAST_PACK2 = 2;
constexpr int64_t CAST_PACK4 = 4;

constexpr int64_t B2_BITS = 2;
constexpr int64_t B4_BITS = 4;
constexpr int64_t B7_BITS = 7;
constexpr int64_t B8_BITS = 8;
constexpr int64_t B12_BITS = 12;
constexpr int64_t B13_BITS = 13;
constexpr int64_t B16_BITS = 16;
constexpr int64_t B32_BITS = 32;
constexpr int64_t B64_BITS = 64;

constexpr int64_t PER_CORE_MIN_UB_BIT = 4 * 1024 * 8;
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
constexpr int32_t SIMT_RESERVED_SIZE = 32 * 1024;

constexpr int64_t UB_ALIGN_RESERVE_TYPE1 = 32 * 6;
constexpr int64_t UB_ALIGN_RESERVE_TYPE2 = 32 * 5;
constexpr int64_t UB_ALIGN_RESERVE_TYPE3 = 32 * 5;
constexpr int64_t UB_ALIGN_RESERVE_TYPE4 = 32 * 4;

bool CastTiling::IsCapable()
{
    return true;
}

ge::graphStatus CastTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<CastCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "compile_info", "nullptr", "not nullptr"),
                        return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }

    vlBitSize_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_) * B8_BITS);
    return ge::GRAPH_SUCCESS;
}

ge::DataType CastTiling::TransAclToGeDataType(int32_t aclType)
{
    switch (aclType) {
        case 0:
            return ge::DT_FLOAT;
        case 1: // 1 DT_FLOAT16
            return ge::DT_FLOAT16;
        case 2: // 2 DT_INT8
            return ge::DT_INT8;
        case 3: // 3 DT_INT32
            return ge::DT_INT32;
        case 4: // 4 DT_UINT8
            return ge::DT_UINT8;
        case 6: // 6 DT_INT16
            return ge::DT_INT16;
        case 7: // 7 DT_UINT16
            return ge::DT_UINT16;
        case 8: // 8 DT_UINT32
            return ge::DT_UINT32;
        case 9: // 9 DT_INT64
            return ge::DT_INT64;
        case 10: // 10 DT_UINT64
            return ge::DT_UINT64;
        case 11: // 11 DT_DOUBLE
            return ge::DT_DOUBLE;
        case 12: // 12 DT_BOOL
            return ge::DT_BOOL;
        case 16: // 16 DT_COMPLEX64
            return ge::DT_COMPLEX64;
        case 27: // 27 DT_BF16
            return ge::DT_BF16;
        case 29: // 29 DT_INT4
            return ge::DT_INT4;
        case 33: // 33 DT_COMPLEX32
            return ge::DT_COMPLEX32;
        case 34: // 34 DT_HIFLOAT8
            return ge::DT_HIFLOAT8;
        case 35: // 35 DT_FLOAT8_E5M2
            return ge::DT_FLOAT8_E5M2;
        case 36: // 36 DT_FLOAT8_E4M3FN
            return ge::DT_FLOAT8_E4M3FN;
        case 40: // 40 DT_FLOAT4_E2M1
            return ge::DT_FLOAT4_E2M1;
        case 41: // 41 DT_FLOAT4_E1M2
            return ge::DT_FLOAT4_E1M2;
        default:
            return ge::DT_MAX;
    }
}

ge::graphStatus CastTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF((context_ == nullptr),
        OP_LOGE_FOR_INVALID_VALUE("Cast", "context", "nullptr", "not nullptr"),
        return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDtype = xDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::DataType yDtype = yDesc->GetDataType();

    // 判断属性和目的类型一致
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const int32_t *dstTypePtr = runtimeAttrs->GetAttrPointer<int32_t>(0);
    OP_CHECK_IF((dstTypePtr == nullptr),
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "dst_type_attr", "nullptr", "not nullptr"),
        return ge::GRAPH_FAILED);
    ge::DataType dstDtype = TransAclToGeDataType(*dstTypePtr);
    OP_CHECK_IF((dstDtype == ge::DT_MAX),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dst_type", ge::TypeUtils::DataTypeToSerialString(dstDtype), "The dtype of dst_type must be in the supported dtype list"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((dstDtype != yDtype),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "dst_type, y", std::string(ge::TypeUtils::DataTypeToSerialString(dstDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(yDtype)), "The dtypes of dst_type and y must be the same"),
        return ge::GRAPH_FAILED);

    // 表驱动，也校验了是否是支持的转换
    constexpr int arraySize = sizeof(castMap) / sizeof(CastMapSt);
    auto it = std::find_if(castMap, castMap + arraySize, [xDtype, yDtype](const CastMapSt &v)
    {
        return v.srcType_ == xDtype && v.dstType_ == yDtype;
    });
    if (it != castMap + arraySize) {
        policy_ = *it;
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x, y", std::string(ge::TypeUtils::DataTypeToSerialString(xDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(yDtype)), "Dtype conversion from x to y is not supported");
        return ge::GRAPH_FAILED;
    }

    auto outputShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outShape = outputShape->GetStorageShape();
    auto inputShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto inShape = inputShape->GetStorageShape();
    size_t xDimNum = inShape.GetDimNum();
    if (dstDtype == ge::DT_INT4 && (inShape.GetDim(xDimNum - 1) % CAST_PACK2)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x[last_dim]", std::to_string(inShape.GetDim(xDimNum - 1)), "when dst_type is DT_INT4, last dim must be divisible by 2");
        return ge::GRAPH_FAILED;
    }
    if (!Ops::Base::IsSameElewiseShape(outShape, inShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x, y", (Ops::Base::ToString(inShape) + ", " + Ops::Base::ToString(outShape)).c_str(), "The shapes of x and y must be the same");
        return ge::GRAPH_FAILED;
    }

    shapeSize_ = inShape.GetShapeSize();
    OP_CHECK_IF(shapeSize_ <= 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(shapeSize_), "x does not support empty tensor"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool CastTiling::IsSimt()
{
    if (policy_.id_ != CAST_TEMPLATE_DIRECT_CAST) {
        return false;
    }
    if (policy_.dstType_ == DT_DOUBLE && policy_.srcType_ == DT_INT64) {
        return true;
    }
    return false;
}

int64_t CastTiling::GetUbFormer(int64_t inputTypeBitSize, int64_t outputTypeBitSize)
{
    int64_t alignInputNum = vlBitSize_ / inputTypeBitSize;
    OP_CHECK_IF(alignInputNum == 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "vlBitSize", std::to_string(vlBitSize_), "greater than 0"),
        return 0);
    if (IsSimt()) {
        OP_LOGI(context_->GetNodeName(), "is SIMT, ub reserve 32k");
        ubSize_ = ubSize_ - SIMT_RESERVED_SIZE;
        context_->SetLocalMemorySize(ubSize_);
    }
    if (policy_.id_ == CAST_TEMPLATE_DIRECT_CAST || policy_.id_ == CAST_TEMPLATE_THROUGH ||
            policy_.id_ == CAST_TEMPLATE_MIRCRO_INOUT || policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST ||
            policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_INTER || policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_DEINTER ||
            policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER || policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_CAST ||
            policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST || policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_DEINTER_CAST ||
            policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER_CAST || policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST_CAST ||
            policy_.id_ == CAST_TEMPLATE_MIRCRO_DEINTER_SHIFT) {
        OP_CHECK_IF(ubSize_ <= UB_ALIGN_RESERVE_TYPE4,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "ubSize", std::to_string(ubSize_), "greater than UB_ALIGN_RESERVE_TYPE4"),
            return 0);
        int64_t ubCap = ((ubSize_ - UB_ALIGN_RESERVE_TYPE4) * B4_BITS) /
            (inputTypeBitSize + outputTypeBitSize);
        return ubCap / alignInputNum * alignInputNum;
    } else if (policy_.id_ == CAST_TEMPLATE_DST_BOOL) {
        OP_CHECK_IF(ubSize_ <= UB_ALIGN_RESERVE_TYPE1,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "ubSize", std::to_string(ubSize_), "greater than UB_ALIGN_RESERVE_TYPE1"),
            return 0);
        int64_t ubCap = ((ubSize_ - UB_ALIGN_RESERVE_TYPE1) * B4_BITS) / (inputTypeBitSize + B13_BITS);
        return ubCap / alignInputNum * alignInputNum;
    } else if (policy_.id_ == CAST_TEMPLATE_SRC_UINT1) {
        OP_CHECK_IF(ubSize_ <= UB_ALIGN_RESERVE_TYPE2,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "ubSize", std::to_string(ubSize_), "greater than UB_ALIGN_RESERVE_TYPE2"),
            return 0);
        int64_t ubCap = ((ubSize_ - UB_ALIGN_RESERVE_TYPE2) * B4_BITS) /
            (outputTypeBitSize * B12_BITS / B8_BITS + 1);
        return ubCap / alignInputNum * alignInputNum;
    } else if (policy_.id_ == CAST_TEMPLATE_TWO_CAST) {
        int64_t midTypeBitSize = GetDtypeBitSize(policy_.midType_);
        OP_CHECK_IF(midTypeBitSize == 0,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "midType", "0", "not 0"),
            return 0);
        OP_CHECK_IF(ubSize_ <= UB_ALIGN_RESERVE_TYPE3,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "ubSize", std::to_string(ubSize_), "greater than UB_ALIGN_RESERVE_TYPE3"),
            return 0);
        int64_t ubCap = ((ubSize_ - UB_ALIGN_RESERVE_TYPE3) * B4_BITS) /
            (inputTypeBitSize + outputTypeBitSize + midTypeBitSize);
        return ubCap / alignInputNum * alignInputNum;
    }
    return 0;
}

int64_t CastTiling::GetDtypeBitSize(uint8_t dtype)
{
    if (dtype == CAST_TPL_UINT1) {
        return 1;
    } else if (dtype == CAST_TPL_BOOL || dtype == CAST_TPL_INT8 || dtype == CAST_TPL_UINT8 ||
        dtype == CAST_TPL_FLOAT8_E4M3FN || dtype == CAST_TPL_FLOAT8_E5M2 || dtype == CAST_TPL_HIFLOAT8) {
        return B8_BITS;
    } else if (dtype == CAST_TPL_UINT16 || dtype == CAST_TPL_INT16 || dtype == CAST_TPL_FLOAT16 || dtype == CAST_TPL_BF16) {
        return B16_BITS;
    } else if (dtype == CAST_TPL_COMPLEX32 || dtype == CAST_TPL_FLOAT || dtype == CAST_TPL_INT32 || dtype == CAST_TPL_UINT32) {
        return B32_BITS;
    } else if (dtype == CAST_TPL_COMPLEX64 || dtype == CAST_TPL_INT64 || dtype == CAST_TPL_DOUBLE) {
        return B64_BITS;
    } else if (dtype == CAST_TPL_FLOAT4_E2M1 || dtype == CAST_TPL_FLOAT4_E1M2 || dtype == CAST_TPL_INT4) {
        return B4_BITS;
    }
    return 0;
}

int64_t CastTiling::GetGeDtypeBitSize(ge::DataType dtype)
{
    if (dtype == DT_UINT1) {
        return 1;
    } else if (dtype == DT_BOOL || dtype == DT_INT8 || dtype == DT_UINT8 ||
        dtype == DT_FLOAT8_E4M3FN || dtype == DT_FLOAT8_E5M2 || dtype == DT_HIFLOAT8) {
        return B8_BITS;
    } else if (dtype == DT_UINT16 || dtype == DT_INT16 || dtype == DT_FLOAT16 || dtype == DT_BF16) {
        return B16_BITS;
    } else if (dtype == DT_COMPLEX32 || dtype == DT_FLOAT || dtype == DT_INT32 || dtype == DT_UINT32) {
        return B32_BITS;
    } else if (dtype == DT_COMPLEX64 || dtype == DT_INT64 || dtype == DT_DOUBLE) {
        return B64_BITS;
    } else if (dtype == DT_FLOAT4_E2M1 || dtype == DT_FLOAT4_E1M2 || dtype == DT_INT4) {
        return B4_BITS;
    }
    return 0;
}

int64_t CastTiling::GetUbCopyStep(uint8_t inType, uint8_t outType,
    uint8_t copyType, int64_t &oneLoopCopyInBitSize)
{
    if (copyType == CAST_MODE_REG_COPYIN_NORM) {
        int64_t inSize = GetDtypeBitSize(inType);
        OP_CHECK_IF(inSize == 0,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "inSize", std::to_string(inSize), "not 0"), return -1);
        oneLoopCopyInBitSize = vlBitSize_;
        return oneLoopCopyInBitSize / inSize;
    } else if (copyType == CAST_MODE_REG_COPYIN_DS_B8) {
        oneLoopCopyInBitSize = vlBitSize_ * CAST_PACK2;
        return oneLoopCopyInBitSize / B8_BITS;
    } else if (copyType == CAST_MODE_REG_COPYIN_DS_B16) {
        oneLoopCopyInBitSize = vlBitSize_ * CAST_PACK2;
        return oneLoopCopyInBitSize / B16_BITS;
    } else if (copyType == CAST_MODE_REG_COPYIN_UNPACK_B8) {
        oneLoopCopyInBitSize = vlBitSize_ / CAST_PACK2;
        return oneLoopCopyInBitSize / B8_BITS;
    } else if (copyType == CAST_MODE_REG_COPYIN_UNPACK_B16) {
        oneLoopCopyInBitSize = vlBitSize_ / CAST_PACK2;
        return oneLoopCopyInBitSize / B16_BITS;
    } else if (copyType == CAST_MODE_REG_COPYIN_UNPACK_B32) {
        oneLoopCopyInBitSize = vlBitSize_ / CAST_PACK2;
        return oneLoopCopyInBitSize / B32_BITS;
    } else if (copyType == CAST_MODE_REG_COPYIN_UNPACK4_B8) {
        oneLoopCopyInBitSize = vlBitSize_ / CAST_PACK4;
        return oneLoopCopyInBitSize / B8_BITS;
    } else if (copyType == CAST_MODE_REG_COPYOUT_NORM) {
        int64_t outSize = GetDtypeBitSize(outType);
        OP_CHECK_IF(outSize == 0,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "outSize", std::to_string(outSize), "not 0"), return -1);
        return vlBitSize_ / outSize;
    } else if (copyType == CAST_MODE_REG_COPYOUT_PACK_B16) {
        return vlBitSize_ / B16_BITS / CAST_PACK2;
    } else if (copyType == CAST_MODE_REG_COPYOUT_PACK_B32) {
        return vlBitSize_ / B32_BITS / CAST_PACK2;
    } else if (copyType == CAST_MODE_REG_COPYOUT_PACK_B64) {
        return vlBitSize_ / B64_BITS / CAST_PACK2;
    } else if (copyType == CAST_MODE_REG_COPYOUT_PACK4_B32) {
        return vlBitSize_ / B32_BITS / CAST_PACK4;
    }
    return 0;
}

ge::graphStatus CastTiling::DoOpTiling()
{
    int64_t inputTypeBitSize = GetGeDtypeBitSize(policy_.srcType_);
    OP_CHECK_IF(inputTypeBitSize == 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "inputTypeBitSize", std::to_string(inputTypeBitSize), "not 0"),
        return ge::GRAPH_FAILED);

    int64_t outputTypeBitSize = GetGeDtypeBitSize(policy_.dstType_);
    OP_CHECK_IF(outputTypeBitSize == 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "outputTypeBitSize", std::to_string(outputTypeBitSize), "not 0"),
        return ge::GRAPH_FAILED);

    uint64_t ubFormer = GetUbFormer(inputTypeBitSize, outputTypeBitSize);
    OP_CHECK_IF(ubFormer == 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "ubSize", std::to_string(ubSize_), "greater than 0"),
        return ge::GRAPH_FAILED);

    int64_t coreNum = (shapeSize_ * inputTypeBitSize + PER_CORE_MIN_UB_BIT - 1) /
        PER_CORE_MIN_UB_BIT;
    if (coreNum > coreNum_) {
        coreNum = coreNum_;
    }
    OP_CHECK_IF(coreNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "coreNum", std::to_string(coreNum)+","+std::to_string(coreNum_), "in the range [1, sys_core_num]"),
        return ge::GRAPH_FAILED);

    int64_t blockFormer = ((shapeSize_ + coreNum - 1) / coreNum + B7_BITS) / B8_BITS * B8_BITS;
    int64_t blockNum = (shapeSize_ + blockFormer - 1) / blockFormer;
    int64_t blockTail = shapeSize_ - (blockNum - 1) * blockFormer;

    int64_t ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
    int64_t ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    tilingData_.set_blockNum(blockNum);
    tilingData_.set_ubFormer(ubFormer);
    tilingData_.set_blockFormer(blockFormer);
    tilingData_.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    tilingData_.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    tilingData_.set_ubTailOfFormerBlock(ubTailOfFormerBlock);
    tilingData_.set_ubTailOfTailBlock(ubTailOfTailBlock);

    int64_t oneLoopCopyInBitSize = 0;
    int64_t inStep = GetUbCopyStep(policy_.srcMapType_, policy_.dstMapType_,
        policy_.regCopyInMode_, oneLoopCopyInBitSize);
    OP_CHECK_IF(inStep == -1,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "inStep", std::to_string(inStep), "not -1"),
        return ge::GRAPH_FAILED);
    tilingData_.set_regCopyInStep(inStep);
    int64_t noUse = 0;
    int64_t outStep = GetUbCopyStep(policy_.srcMapType_, policy_.dstMapType_, policy_.regCopyOutMode_, noUse);
    OP_CHECK_IF(outStep == -1,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "outStep", std::to_string(outStep), "not -1"),
        return ge::GRAPH_FAILED);
    tilingData_.set_regCopyOutStep(outStep);

    int64_t ubFormerRegLoop = 0;
    int64_t ubTailOfFormerRegLoop = 0;
    int64_t ubTailOfTailRegLoop = 0;
    if (oneLoopCopyInBitSize != 0) {
        if (policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_DEINTER || policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_DEINTER_CAST ||
            policy_.id_ == CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER_CAST || policy_.id_ == CAST_TEMPLATE_MIRCRO_DEINTER_SHIFT) {
            // once load two reg len
            int64_t doubleCopyInBitSize = oneLoopCopyInBitSize + oneLoopCopyInBitSize;
            ubFormerRegLoop = (ubFormer * inputTypeBitSize + doubleCopyInBitSize - 1) / doubleCopyInBitSize;
            ubTailOfFormerRegLoop = (ubTailOfFormerBlock * inputTypeBitSize + doubleCopyInBitSize - 1) / doubleCopyInBitSize;
            ubTailOfTailRegLoop = (ubTailOfTailBlock * inputTypeBitSize + doubleCopyInBitSize - 1) / doubleCopyInBitSize;
        } else {
            ubFormerRegLoop = (ubFormer * inputTypeBitSize + oneLoopCopyInBitSize - 1) / oneLoopCopyInBitSize;
            ubTailOfFormerRegLoop = (ubTailOfFormerBlock * inputTypeBitSize + oneLoopCopyInBitSize - 1) / oneLoopCopyInBitSize;
            ubTailOfTailRegLoop = (ubTailOfTailBlock * inputTypeBitSize + oneLoopCopyInBitSize - 1) / oneLoopCopyInBitSize;
        }
    }
    tilingData_.set_ubFormerRegLoop(ubFormerRegLoop);
    tilingData_.set_ubTailOfFormerRegLoop(ubTailOfFormerRegLoop);
    tilingData_.set_ubTailOfTailRegLoop(ubTailOfTailRegLoop);
    
    OP_LOGD(context_->GetNodeName(),
        "cast do tiling finish. coreNum: %ld ubSize: %ld vlBit: %ld "
        "blockNum: %ld ubFormer: %ld blockFormer: %ld ubLoopOfFormerBlock: %ld "
        "ubLoopOfTailBlock: %ld ubTailOfFormerBlock: %ld ubTailOfTailBlock: %ld inStep: %ld outStep: %ld "
        "ubFormerRegLoop: %ld ubTailOfFormerRegLoop: %ld ubTailOfTailRegLoop: %ld oneLoopCopyInBitSize: %ld",
        coreNum_, ubSize_, vlBitSize_, blockNum, ubFormer, blockFormer, ubLoopOfFormerBlock,
        ubLoopOfTailBlock, ubTailOfFormerBlock, ubTailOfTailBlock, inStep, outStep,
        ubFormerRegLoop, ubTailOfFormerRegLoop, ubTailOfTailRegLoop, oneLoopCopyInBitSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CastTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t CastTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(0);
    return tilingKey;
}

ge::graphStatus CastTiling::GetWorkspaceSize()
{
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CastTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tiling_data_size", std::to_string(tilingData_.GetDataSize()), "The value of tiling_data_size must be less than or equal to capacity"),
        return ge::GRAPH_FAILED);

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                        context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData_.get_blockNum());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForCast(gert::TilingContext *context)
{
    OP_LOGD("CastTiling", "Enter TilingForCast");
    OP_CHECK_IF(context == nullptr,
        OP_LOGE_FOR_INVALID_VALUE("Cast", "context", "nullptr", "not nullptr"),
        return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompileInfo<CastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("CastTiling", "Enter new CastTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForCast(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<CastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Cast).Tiling(TilingForCast)
    .TilingParse<CastCompileInfo>(TilingPrepareForCast);

REGISTER_OPS_TILING_TEMPLATE(Cast, CastTiling, 1);
}
