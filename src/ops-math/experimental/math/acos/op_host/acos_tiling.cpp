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
 * \file acos_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/acos_tiling_data.h"
#include "../op_kernel/acos_tiling_key.h"

namespace optiling {

constexpr int32_t BUFFER_NUM = 2;
static const int64_t MULTI_CORE_SHAPE_SIZE_LIMIT = 2048; // 2k x 8字节 * OP_COEXISTING_NUM小于192k
static const uint32_t OP_CALC_TENSOR_NUM = 6;            // 算子计算过程中需要用到6个float临时计算Tensor数量
static const uint32_t OP_INOUT_TENSOR_NUM = 4;           // 算子计算过程中需要用到4个T类型Tensor数量
static std::map<ge::DataType, uint> dataSizeMap = {
    {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 2}};
static const uint32_t BYTE_ALIGN = 256; // kernel侧计算的时候必须用256字节对齐
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

struct AcosCompileInfo {};
// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 计算生成
static void CalcTilingData(
    const uint64_t totalLength, const ge::DataType dtype_x, const uint64_t coreNum, uint64_t ubSize,
    AcosTilingData& tiling)
{
    auto it = dataSizeMap.find(dtype_x);
    if (it == dataSizeMap.end()) {
        return;
    }
    auto dataSize = it->second;
    if (dataSize == 0) {
        return;
    }
    auto elementAlgin = BYTE_ALIGN / dataSize;
    if (coreNum == 0) {
        return;
    }
    uint64_t tileBufferLen = ubSize / (dataSize * OP_INOUT_TENSOR_NUM + sizeof(float) * OP_CALC_TENSOR_NUM);
    // tileBufferLen必须保证占用字节数是256整数倍
    tileBufferLen = (tileBufferLen + elementAlgin - 1) / elementAlgin * elementAlgin;
    if (totalLength <= MULTI_CORE_SHAPE_SIZE_LIMIT) {
        tiling.formerCoreNum = 1;
        tiling.tailCoreNum = 0;
        tiling.formerCoreDataNum = totalLength;
        tiling.tailCoreDataNum = 0;
        tiling.formerCoreLoopCount = (totalLength + tileBufferLen - 1) / tileBufferLen;
        tiling.formerCoreFormerDataNum = totalLength > tileBufferLen ? tileBufferLen : totalLength;
        tiling.formerCoreTailDataNum = totalLength % tileBufferLen == 0 ? tileBufferLen : totalLength % tileBufferLen;
        tiling.tailCoreLoopCount = 0;
        tiling.tailCoreFormerDataNum = 0;
        tiling.tailCoreTailDataNum = 0;
        return;
    } else {
        tiling.formerCoreNum = totalLength % coreNum == 0 ? coreNum : totalLength % coreNum;
        tiling.tailCoreNum = coreNum - tiling.formerCoreNum;
        tiling.formerCoreDataNum = (totalLength + coreNum - 1) / coreNum;
        tiling.tailCoreDataNum = tiling.tailCoreNum == 0 ? 0 : totalLength / coreNum;
        tiling.formerCoreLoopCount = (tiling.formerCoreDataNum + tileBufferLen - 1) / tileBufferLen;
        tiling.formerCoreFormerDataNum =
            tiling.formerCoreDataNum > tileBufferLen ? tileBufferLen : tiling.formerCoreDataNum;
        tiling.formerCoreTailDataNum =
            tiling.formerCoreDataNum % tileBufferLen == 0 ? tileBufferLen : tiling.formerCoreDataNum % tileBufferLen;
        tiling.tailCoreLoopCount = (tiling.tailCoreDataNum + tileBufferLen - 1) / tileBufferLen;
        tiling.tailCoreFormerDataNum = tiling.tailCoreDataNum > tileBufferLen ? tileBufferLen : tiling.tailCoreDataNum;
        tiling.tailCoreTailDataNum =
            tiling.tailCoreDataNum % tileBufferLen == 0 ? tileBufferLen : tiling.tailCoreDataNum % tileBufferLen;
    }
}

// tiling 分发入口
static ge::graphStatus AcosTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    uint32_t D_T_X = static_cast<int>(dtype_x);
    uint64_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    AcosTilingData* tiling = context->GetTilingData<AcosTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    CalcTilingData(totalLength, dtype_x, coreNum, ubSize, *tiling);
    context->SetBlockDim(tiling->formerCoreNum + tiling->tailCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, D_T_X); // 模板参数tilingkey配置
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAcos([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
// tiling注册入口.
IMPL_OP_OPTILING(Acos).Tiling(AcosTilingFunc).TilingParse<AcosCompileInfo>(TilingParseForAcos);
} // namespace optiling
