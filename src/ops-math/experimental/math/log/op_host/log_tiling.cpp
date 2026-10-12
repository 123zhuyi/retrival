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
 * \file log_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/log_tiling_data.h"
#include "../op_kernel/log_tiling_key.h"

namespace optiling {

const uint64_t BUFFER_NUM = 2;

struct LogCompileInfo {};

// tiling 分发入口
static ge::graphStatus LogTilingFunc(gert::TilingContext* context)
{
    uint64_t ubLength = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t bigCoreLoopNum = 0;
    uint64_t bigCoreTailDataNum = 0;
    uint64_t blockSize = 0;

    blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    // 获取输入数据信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    uint64_t inputDataNum = inputX->GetStorageShape().GetShapeSize();
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, dataTypeLength);
    uint64_t inputLength = inputDataNum * dataTypeLength;

    // 计算UB可用空间

    uint64_t ubPartNum = (dataType == ge::DT_BF16) ? 3 : 2;
    uint64_t ubPartLength = ubLength / ubPartNum / BUFFER_NUM;
    uint64_t ubPartBlockNum = ubPartLength / blockSize;
    uint64_t ubPartDataNum = (ubPartBlockNum * blockSize) / dataTypeLength;

    // 32B对齐处理
    uint64_t inputLengthAlign32 = (((inputLength + blockSize - 1) / blockSize) * blockSize);

    // 核心数分配逻辑
    if (ubPartDataNum >= inputDataNum) {
        coreNum = 1;
    } else {
        coreNum = (coreNum < inputLengthAlign32 / blockSize) ? coreNum : inputLengthAlign32 / blockSize;
    }

    // 计算核心数据分配
    uint64_t everyCoreInputBlockNum = inputLengthAlign32 / blockSize / coreNum;
    uint64_t tailBlockNum = (inputLengthAlign32 / blockSize) % coreNum;

    // 小核数据计算
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * blockSize / dataTypeLength;
    uint64_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
    smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? smallCoreLoopNum : smallCoreLoopNum + 1;
    uint64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
    smallCoreTailDataNum = (smallCoreTailDataNum == 0) ? ubPartDataNum : smallCoreTailDataNum;

    // 大核数据计算
    uint64_t tilingKey = 0;
    if (tailBlockNum != 0) {
        everyCoreInputBlockNum += 1;
        bigCoreDataNum = everyCoreInputBlockNum * blockSize / dataTypeLength;
        bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
        bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? bigCoreLoopNum : bigCoreLoopNum + 1;
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
        bigCoreTailDataNum = (bigCoreTailDataNum == 0) ? ubPartDataNum : bigCoreTailDataNum;
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    }
    // 设置tiling参数
    LogTilingData* tiling = context->GetTilingData<LogTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LogTilingData), 0, sizeof(LogTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->ubPartDataNum = ubPartDataNum;
    tiling->smallCoreTailDataNum = smallCoreTailDataNum;
    tiling->bigCoreTailDataNum = bigCoreTailDataNum;
    tiling->smallCoreLoopNum = smallCoreLoopNum;
    tiling->bigCoreLoopNum = bigCoreLoopNum;
    tiling->tailBlockNum = tailBlockNum;

    context->SetBlockDim(coreNum);

    // 处理log特有属性
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* base = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, base);
    if (*base == -1.0f) {
        tiling->base = -1.0f;
    } else if (*base > 0.0f && *base != 1.0f) {
        tiling->base = 1.0f / std::log(*base);
    } else {
        tiling->base = 1.0f;
    }

    const float* scale = attrs->GetAttrPointer<float>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    tiling->scale = *scale;

    const float* shift = attrs->GetAttrPointer<float>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shift);
    tiling->shift = *shift;

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLog([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Log).Tiling(LogTilingFunc).TilingParse<LogCompileInfo>(TilingParseForLog);
} // namespace optiling
