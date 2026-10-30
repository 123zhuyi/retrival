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
 * \file stride_add_tiling.cpp
 * \brief StrideAdd tiling implementation
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../op_kernel/arch35/stride_add_tiling_data.h"
#include "../../op_kernel/arch35/stride_add_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
constexpr int64_t MIN_PER_CORE_ELEMENTS = 1024;

// 输入索引
constexpr size_t INPUT_X1_IDX = 0;
constexpr size_t INPUT_X2_IDX = 1;

// NC1HWC0 物理 shape 维度索引
constexpr size_t DIM_N = 0;
constexpr size_t DIM_C1 = 1;
constexpr size_t DIM_H = 2;
constexpr size_t DIM_W = 3;
constexpr size_t DIM_C0 = 4;
constexpr size_t SHAPE_DIM_NUM = 5;
constexpr int64_t EXPECTED_C0 = 16;

// 属性索引
constexpr size_t ATTR_X1_C1_OFFSET_IDX = 0;
constexpr size_t ATTR_X2_C1_OFFSET_IDX = 1;
constexpr size_t ATTR_C1_LEN_IDX = 2;

struct StrideAddCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus StrideAddTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取输入 shape 和属性
    auto x1Shape = context->GetInputShape(INPUT_X1_IDX);
    auto x2Shape = context->GetInputShape(INPUT_X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto x1StorageShape = x1Shape->GetStorageShape();
    auto x2StorageShape = x2Shape->GetStorageShape();

    int64_t N = x1StorageShape.GetDim(DIM_N);
    int64_t C1_x1 = x1StorageShape.GetDim(DIM_C1);
    int64_t H = x1StorageShape.GetDim(DIM_H);
    int64_t W = x1StorageShape.GetDim(DIM_W);
    int64_t C0 = x1StorageShape.GetDim(DIM_C0);
    int64_t C1_x2 = x2StorageShape.GetDim(DIM_C1);

    // 获取属性（指针判空+解引用模式）
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto x1OffsetPtr = attrs->GetAttrPointer<int32_t>(ATTR_X1_C1_OFFSET_IDX);
    auto x2OffsetPtr = attrs->GetAttrPointer<int32_t>(ATTR_X2_C1_OFFSET_IDX);
    auto c1LenPtr = attrs->GetAttrPointer<int32_t>(ATTR_C1_LEN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1OffsetPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2OffsetPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, c1LenPtr);
    int32_t x1_c1_offset = *x1OffsetPtr;
    int32_t x2_c1_offset = *x2OffsetPtr;
    int32_t c1_len = *c1LenPtr;

    // 3. 参数校验
    OP_CHECK_IF(x1_c1_offset < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x1_c1_offset",
            std::to_string(x1_c1_offset).c_str(), "x1_c1_offset must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2_c1_offset < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x2_c1_offset",
            std::to_string(x2_c1_offset).c_str(), "x2_c1_offset must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(c1_len <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "c1_len",
            std::to_string(c1_len).c_str(), "c1_len must be greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1_c1_offset + c1_len > C1_x1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x1_c1_offset",
            std::to_string(x1_c1_offset).c_str(), "x1_c1_offset + c1_len must not exceed x1's C1 dimension"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2_c1_offset + c1_len > C1_x2,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x2_c1_offset",
            std::to_string(x2_c1_offset).c_str(), "x2_c1_offset + c1_len must not exceed x2's C1 dimension"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1StorageShape.GetDim(DIM_N) != x2StorageShape.GetDim(DIM_N),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2",
            (Ops::Base::ToString(x1StorageShape) + ", " + Ops::Base::ToString(x2StorageShape)),
            "x1 and x2 N dimension must be same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1StorageShape.GetDim(DIM_H) != x2StorageShape.GetDim(DIM_H),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2",
            (Ops::Base::ToString(x1StorageShape) + ", " + Ops::Base::ToString(x2StorageShape)),
            "x1 and x2 H dimension must be same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1StorageShape.GetDim(DIM_W) != x2StorageShape.GetDim(DIM_W),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2",
            (Ops::Base::ToString(x1StorageShape) + ", " + Ops::Base::ToString(x2StorageShape)),
            "x1 and x2 W dimension must be same"),
        return ge::GRAPH_FAILED);
    // dtype 一致性校验
    auto x1Desc = context->GetInputDesc(INPUT_X1_IDX);
    auto x2Desc = context->GetInputDesc(INPUT_X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    OP_CHECK_IF(x1Desc->GetDataType() != x2Desc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x1, x2",
            std::string(ge::TypeUtils::DataTypeToSerialString(x1Desc->GetDataType())) + ", " +
            std::string(ge::TypeUtils::DataTypeToSerialString(x2Desc->GetDataType())),
            "The dtypes of x1 and x2 must be the same"),
        return ge::GRAPH_FAILED);
    // rank/format/C0 校验
    OP_CHECK_IF(x1StorageShape.GetDimNum() != SHAPE_DIM_NUM || x2StorageShape.GetDimNum() != SHAPE_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x1, x2",
            (std::to_string(x1StorageShape.GetDimNum()) + ", " + std::to_string(x2StorageShape.GetDimNum())),
            "x1 and x2 must be 5D NC1HWC0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1StorageShape.GetDim(DIM_C0) != x2StorageShape.GetDim(DIM_C0),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2",
            (Ops::Base::ToString(x1StorageShape) + ", " + Ops::Base::ToString(x2StorageShape)),
            "x1 and x2 C0 must be same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1StorageShape.GetDim(DIM_C0) != EXPECTED_C0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x1",
            Ops::Base::ToString(x1StorageShape).c_str(),
            "current implementation only supports C0=16"),
        return ge::GRAPH_FAILED);

    // 4. 计算 tiling 参数
    int64_t totalElements = N * c1_len * H * W * C0;
    int64_t hwC0Size = H * W * C0;
    int64_t x1NStride = C1_x1 * hwC0Size;
    int64_t x2NStride = C1_x2 * hwC0Size;

    // 空 tensor: 先初始化 needCoreNum=1/perCoreElements=0，再 if(totalElements>0) 分支计算
    int64_t needCoreNum = 1;
    int64_t perCoreElements = 0;
    if (totalElements > 0) {
        needCoreNum = Ops::Base::CeilDiv(totalElements, MIN_PER_CORE_ELEMENTS);
        needCoreNum = std::min(needCoreNum, coreNum);
        needCoreNum = std::max(needCoreNum, static_cast<int64_t>(1));
        perCoreElements = Ops::Base::CeilDiv(totalElements, needCoreNum);
    }

    // 5. 填充 tiling data
    StrideAddTilingData* tiling = context->GetTilingData<StrideAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(StrideAddTilingData), 0, sizeof(StrideAddTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalElements = totalElements;
    tiling->perCoreElements = perCoreElements;
    tiling->hwC0Size = hwC0Size;
    tiling->c1Len = c1_len;
    tiling->x1NStride = x1NStride;
    tiling->x2NStride = x2NStride;
    tiling->x1C1Offset = x1_c1_offset;
    tiling->x2C1Offset = x2_c1_offset;
    tiling->needCoreNum = needCoreNum;

    // 6. 设置核数和 workspace
    context->SetBlockDim(needCoreNum);
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
        OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize),
        return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u, STATIC_UB_ESTIMATE=%u",
            ubSize, DCACHE_SIZE, STATIC_UB_ESTIMATE),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());

    // 7. 设置 tiling key（单模式，dtype 由 DTYPE_X1 宏自动实例化）
    uint64_t tilingKey = GET_TPL_TILING_KEY(STRIDE_ADD_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForStrideAdd([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StrideAdd).Tiling(StrideAddTilingFunc).TilingParse<StrideAddCompileInfo>(TilingParseForStrideAdd);

} // namespace optiling