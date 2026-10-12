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
 * \file ones_like_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_util.h"
#include "experimental/math/ones_like/op_kernel/ones_like_tiling_data.h"
#include "experimental/math/ones_like/op_kernel/ones_like_tiling_key.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define DATA_SIZE_ONE (1)
#define DATA_SIZE_TWO (2)
#define DATA_SIZE_FOUR (4)
#define MIN_CHUNK_SIZE (4096)
#define UB_USAGE_DIV (2)
namespace optiling {

struct OnesLikeCompileInfo {};

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

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = Ops::Base::EnsureNotScalar(outY->GetStorageShape());

    int dimNum = outShapeY.GetDimNum();
    totalIdx = 1;
    for (int i = 0; i < dimNum; i++) {
        totalIdx *= outShapeY.GetDim(i);
    }

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8,
                                                   ge::DT_INT8,  ge::DT_BF16,    ge::DT_BOOL};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus OnesLikeTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    int64_t totalIdx;
    ge::DataType dataType = ge::DT_FLOAT;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    OnesLikeTilingData* tiling = context->GetTilingData<OnesLikeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(OnesLikeTilingData), 0, sizeof(OnesLikeTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 5、计算tiling信息
    int blockNum = coreNum;
    int typeByte = GetSizeByDataType(dataType);
    if (totalIdx * typeByte < MIN_CHUNK_SIZE * coreNum) {
        blockNum = (totalIdx * typeByte + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE;
    }
    if (typeByte == 0) {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    uint32_t maxTileLen = ubSize / typeByte / UB_USAGE_DIV;
    uint32_t blockSize = totalIdx / blockNum;

    tiling->formerNum = (totalIdx % blockNum) == 0 ? blockNum : (totalIdx % blockNum);
    tiling->tailNum = blockNum - tiling->formerNum;
    tiling->tailLength = blockSize;
    tiling->formerLength = blockSize + (tiling->tailNum == 0 ? 0 : 1);

    tiling->formerTileLength = MIN(maxTileLen, tiling->formerLength);
    tiling->formerTileNum = (tiling->formerLength + tiling->formerTileLength - 1) / tiling->formerTileLength;
    tiling->formerLastTileLength = tiling->formerLength - (tiling->formerTileNum - 1) * tiling->formerTileLength;

    tiling->tailTileLength = MIN(maxTileLen, tiling->tailLength);
    tiling->tailTileNum = (tiling->tailLength + tiling->tailTileLength - 1) / tiling->tailTileLength;
    tiling->tailLastTileLength = tiling->tailLength - (tiling->tailTileNum - 1) * tiling->tailTileLength;

    context->SetBlockDim(blockNum);
    uint64_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForOnesLike([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(OnesLike).Tiling(OnesLikeTilingFunc).TilingParse<OnesLikeCompileInfo>(TilingParseForOnesLike);
} // namespace optiling
