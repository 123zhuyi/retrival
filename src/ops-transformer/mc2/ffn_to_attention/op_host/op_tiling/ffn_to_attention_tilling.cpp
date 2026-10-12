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
 * \file ffn_to_attention_tilling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "../../op_kernel/ffn_to_attention_tiling.h"
#include "../../op_kernel/ffn_to_attention_tilling_key.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
constexpr char FFN_TO_ATTN_WIN_TYPE_ENV[] = "ASCEND_FFN_TO_ATTN_WIN_TYPE";
 
constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_SESSION_IDS_INDEX = 1;
constexpr uint32_t INPUT_MICRO_BATCH_IDS_INDEX = 2;
constexpr uint32_t INPUT_TOKEN_IDS_INDEX = 3;
constexpr uint32_t INPUT_EXPERT_OFFSETS_INDEX = 4;
constexpr uint32_t INPUT_ACTUAL_TOKEN_NUM_INDEX = 5;
constexpr uint32_t INPUT_ATTN_RANK_TABLE_INDEX = 6;
 
constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_TOKEN_INFO_TABLE_SHAPE_INDEX = 2;
constexpr uint32_t ATTR_TOKEN_DATA_SHAPE = 3;

constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint64_t MB_SIZE = 1024 * 1024;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t SCALE_SIZE = 128;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_WORLD_SIZE = 768L; // 384 * 2
constexpr int64_t MIN_WORLD_SIZE = 2;

constexpr uint32_t TOKEN_DATA_SHAPE_MICRO_BATCH_NUM_INDEX = 0;
constexpr uint32_t TOKEN_DATA_SHAPE_BS_INDEX = 1;
constexpr uint32_t TOKEN_DATA_SHAPE_HS_INDEX = 3;
constexpr uint32_t TOKEN_INFO_TABLE_SHAPE_EXPERT_NUM_INDEX = 2;

constexpr uint32_t TOKEN_INFO_TABLE_DIM_NUM = 3;
constexpr uint32_t TOKEN_DATA_DIM_NUM = 4;

constexpr uint32_t TOKEN_DATA_SIZE = 2; // float/bfloat16, 占2字节
constexpr uint32_t TOKEN_INFO_SIZE = 4; // int32, 占4字节
 
static void PrintTilingDataInfo(const char *nodeName, FFNToAttentionTilingData &tilingData)
{
    OP_LOGD(nodeName, "H is %u.", tilingData.ffnToAttentionInfo.H);
    OP_LOGD(nodeName, "A is %u.", tilingData.ffnToAttentionInfo.A);
    OP_LOGD(nodeName, "microBatchNum is %u.", tilingData.ffnToAttentionInfo.microBatchNum);
    OP_LOGD(nodeName, "BS is %u.", tilingData.ffnToAttentionInfo.BS);
    OP_LOGD(nodeName, "expertNumPerToken is %u.", tilingData.ffnToAttentionInfo.expertNumPerToken);
    OP_LOGD(nodeName, "HS is %u.", tilingData.ffnToAttentionInfo.HS);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.ffnToAttentionInfo.aivNum);
    OP_LOGD(nodeName, "worldSize is %u.", tilingData.ffnToAttentionInfo.worldSize);
    OP_LOGD(nodeName, "isInputRankTable is %u.", tilingData.ffnToAttentionInfo.isInputRankTable);
    OP_LOGD(nodeName, "windowType is %u.", tilingData.ffnToAttentionInfo.windowType);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.ffnToAttentionInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.ffnToAttentionInfo.totalWinSize);
}
 
static bool CheckAndSetAttrs(gert::TilingContext* context, const char *nodeName, FFNToAttentionTilingData &tilingData, std::string &group)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "GetAttrs returned nullptr!"), return false);
 
    auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);
    auto tokenInfoTableDimNum = attrs->GetListInt(ATTR_TOKEN_INFO_TABLE_SHAPE_INDEX)->GetSize();
    auto tokenInfoTableShape = attrs->GetListInt(ATTR_TOKEN_INFO_TABLE_SHAPE_INDEX)->GetData();
    auto tokenDataDimNum = attrs->GetListInt(ATTR_TOKEN_DATA_SHAPE)->GetSize();
    auto tokenDataShape = attrs->GetListInt(ATTR_TOKEN_DATA_SHAPE)->GetData();

    // 当前仅对必选属性进行校空
    OP_TILING_CHECK((groupPtr == nullptr) || (strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
    (strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
    OP_LOGE(nodeName, "group is null."), return false);
    OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE(nodeName, "world_size is nullptr!"), return false);
    OP_TILING_CHECK(tokenInfoTableDimNum != TOKEN_INFO_TABLE_DIM_NUM, OP_LOGE(nodeName, "token_info_table_shape is not equal %u!", TOKEN_INFO_TABLE_DIM_NUM), return false);
    OP_TILING_CHECK(tokenDataDimNum != TOKEN_DATA_DIM_NUM, OP_LOGE(nodeName, "token_data_shape dims is not equal %u!", TOKEN_DATA_DIM_NUM), return false);
    
    OP_TILING_CHECK(tokenInfoTableShape[INDEX_ZERO] != tokenDataShape[INDEX_ZERO],
        OP_LOGE(nodeName, "token_info_table_shape dims0=%ld is not equal token_data_shape dims0=%ld!",
        tokenInfoTableShape[INDEX_ZERO], tokenDataShape[INDEX_ZERO]), return false);
    OP_TILING_CHECK(tokenInfoTableShape[INDEX_ONE] != tokenDataShape[INDEX_ONE],
        OP_LOGE(nodeName, "token_info_table_shape dims1=%ld is not equal token_data_shape dims1=%ld!",
        tokenInfoTableShape[INDEX_ONE], tokenDataShape[INDEX_ONE]), return false);
    OP_TILING_CHECK(tokenInfoTableShape[INDEX_TWO] != tokenDataShape[INDEX_TWO],
        OP_LOGE(nodeName, "token_info_table_shape dims2=%ld is not equal token_data_shape dims2=%ld!",
        tokenInfoTableShape[INDEX_TWO], tokenDataShape[INDEX_TWO]), return false);
    // 判断是否满足其他限制
    int64_t worldSize = *worldSizePtr;
    OP_TILING_CHECK((worldSize < MIN_WORLD_SIZE) || (worldSize > MAX_WORLD_SIZE),
        OP_LOGE(nodeName, "WorldSize is invalid, only support [%ld, %ld], but got WorldSize=%ld.",
        MIN_WORLD_SIZE, MAX_WORLD_SIZE, worldSize), return false);
    OP_TILING_CHECK(tokenInfoTableShape[INDEX_ZERO] != 1, OP_LOGE(nodeName, "MircoBatchNum dims is not equal 1!, but %u ", 
    tokenInfoTableShape[INDEX_ZERO]), return false);
    OP_TILING_CHECK((tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX] > BS_UPPER_BOUND) || (tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX] <= 0),
    OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between [1, %ld], but got xDim0=%ld.", BS_UPPER_BOUND,
            tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX]), return false);
    OP_TILING_CHECK((tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX] < H_MIN ) || (tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX] > H_MAX + SCALE_SIZE), 
    OP_LOGE(nodeName,"HS should be in [%ld, %ld], but got %ld.",
        H_MIN, H_MAX + SCALE_SIZE, tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX]), return false); 

    tilingData.ffnToAttentionInfo.worldSize = *worldSizePtr;
    tilingData.ffnToAttentionInfo.microBatchNum = tokenDataShape[TOKEN_DATA_SHAPE_MICRO_BATCH_NUM_INDEX];
    tilingData.ffnToAttentionInfo.BS = tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX];
    tilingData.ffnToAttentionInfo.HS = tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX];
    tilingData.ffnToAttentionInfo.expertNumPerToken = tokenInfoTableShape[TOKEN_INFO_TABLE_SHAPE_EXPERT_NUM_INDEX];
    
    OP_LOGD(nodeName, "group = %s", groupPtr);
    group = string(groupPtr);
 
    return true;
}

static bool CheckInputDim0Dim1(gert::TilingContext* context, const char *nodeName, FFNToAttentionTilingData &tilingData)
{
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *sessionIdsShape = context->GetInputShape(INPUT_SESSION_IDS_INDEX);
    const gert::StorageShape *microBatchIdsShape = context->GetInputShape(INPUT_MICRO_BATCH_IDS_INDEX);
    const gert::StorageShape *tokenIdsShape = context->GetInputShape(INPUT_TOKEN_IDS_INDEX);
    const gert::StorageShape *expertOffsetsShape = context->GetInputShape(INPUT_EXPERT_OFFSETS_INDEX);
    const gert::StorageShape *actualTokenNumShape = context->GetInputShape(INPUT_ACTUAL_TOKEN_NUM_INDEX);
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(INPUT_ATTN_RANK_TABLE_INDEX);
    // 校验输入x的维度Y和H
    const uint64_t xDim0 = xShape->GetStorageShape().GetDim(INDEX_ZERO);
    const uint64_t xDim1 = xShape->GetStorageShape().GetDim(INDEX_ONE);
    const int64_t MAX_INT64 = std::numeric_limits<int64_t>::max();
    OP_TILING_CHECK((xDim0 > static_cast<uint64_t>(MAX_INT64)), OP_LOGE(nodeName,"xShape dims0(Y) exceeds int64_t maximum value %ld, but got %ld.",
        MAX_INT64, xDim1), return false); 
    OP_TILING_CHECK((xDim1 < H_MIN) || (xDim1 > H_MAX), OP_LOGE(nodeName,"xShape dims1(H) should be in [%ld, %ld], but got %ld.",
        H_MIN, H_MAX, xDim1), return false); 
    
    // 校验输入sessionIds的维度Y
    const int64_t sessionIdsDim0 = sessionIdsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(xDim0 != sessionIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to sessionIdShape's dim0, "
        "xShape's dim0 is %ld, sessionIdShape's dim0 is %ld.", xDim0, sessionIdsDim0), return false);
    // 校验输入microBatchIds的维度Y
    const int64_t microBatchIdsDim0 = microBatchIdsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(xDim0 != microBatchIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to microBatchIdShape's dim0, "
        "xShape's dim0 is %ld, microBatchIdShape's dim0 is %ld.", xDim0, microBatchIdsDim0), return false);
    // 校验输入tokenIds的维度Y
    const int64_t tokenIdsDim0 = tokenIdsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(xDim0 != tokenIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to tokenIdShape's dim0, "
        "xShape's dim0 is %ld, tokenIdShape's dim0 is %ld.", xDim0, tokenIdsDim0), return false);
    // 校验输入expertOffsets的维度Y
    const int64_t expertOffsetsDim0 = expertOffsetsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(xDim0 != expertOffsetsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to expertOffsetShape's dim0, "
        "xShape's dim0 is %ld, expertOffsetShape's dim0 is %ld.", xDim0, expertOffsetsDim0), return false);
    // 校验输入actualTokenNum的维度1
    const uint64_t actualTokenNumDim0 = actualTokenNumShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(actualTokenNumDim0 != 1, OP_LOGE(nodeName, "actualTokenNumShape's dim0 not equal to 1, "
        "but actualTokenNumShape's dim0 is %ld.", actualTokenNumDim0), return false);
    tilingData.ffnToAttentionInfo.H = xDim1;
    return true;
}

static bool CheckInputDim(gert::TilingContext* context, const char *nodeName, FFNToAttentionTilingData &tilingData)
{   
    auto attrs = context->GetAttrs();
    auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);
    int64_t worldSize = *worldSizePtr;
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *sessionIdsShape = context->GetInputShape(INPUT_SESSION_IDS_INDEX);
    const gert::StorageShape *microBatchIdsShape = context->GetInputShape(INPUT_MICRO_BATCH_IDS_INDEX);
    const gert::StorageShape *tokenIdsShape = context->GetInputShape(INPUT_TOKEN_IDS_INDEX);
    const gert::StorageShape *expertOffsetsShape = context->GetInputShape(INPUT_EXPERT_OFFSETS_INDEX);
    const gert::StorageShape *actualTokenNumShape = context->GetInputShape(INPUT_ACTUAL_TOKEN_NUM_INDEX);
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(INPUT_ATTN_RANK_TABLE_INDEX);
    bool isInputRankTable = (attnRankTableShape != nullptr);
 
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE(nodeName, "x is nullptr!"), return false);
    OP_TILING_CHECK(sessionIdsShape == nullptr, OP_LOGE(nodeName, "sessionIds is nullptr!"), return false);
    OP_TILING_CHECK(microBatchIdsShape == nullptr, OP_LOGE(nodeName, "microBatchIds is nullptr!"), return false);
    OP_TILING_CHECK(tokenIdsShape == nullptr, OP_LOGE(nodeName, "tokenIds is nullptr!"), return false);
    OP_TILING_CHECK(expertOffsetsShape == nullptr, OP_LOGE(nodeName, "expertOffsets is nullptr!"), return false);
    OP_TILING_CHECK(actualTokenNumShape == nullptr, OP_LOGE(nodeName, "actualTokenNum is nullptr!"), return false);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "x dim must be equal to 2, cur xDim=%lu!", xShape->GetStorageShape().GetDimNum()), return false);
    OP_TILING_CHECK(sessionIdsShape->GetStorageShape().GetDimNum() != ONE_DIM, 
        OP_LOGE(nodeName, "sessionIds dim must be equal to 1, cur sessionIdsDim=%lu!", sessionIdsShape->GetStorageShape().GetDimNum()), return false);
    OP_TILING_CHECK(microBatchIdsShape->GetStorageShape().GetDimNum() != ONE_DIM, 
        OP_LOGE(nodeName, "microBatchIds dim must be equal to 1, cur microBatchIdsDim=%lu!", microBatchIdsShape->GetStorageShape().GetDimNum()), return false);
    OP_TILING_CHECK(tokenIdsShape->GetStorageShape().GetDimNum() != ONE_DIM, 
        OP_LOGE(nodeName, "tokenIds dim must be equal to 1, cur tokenIdsDim=%lu!", tokenIdsShape->GetStorageShape().GetDimNum()), return false);
    OP_TILING_CHECK(expertOffsetsShape->GetStorageShape().GetDimNum() != ONE_DIM, 
        OP_LOGE(nodeName, "expertOffsets dim must be equal to 1, cur expertOffsetsDim=%lu!", expertOffsetsShape->GetStorageShape().GetDimNum()), return false);
    OP_TILING_CHECK(actualTokenNumShape->GetStorageShape().GetDimNum() != ONE_DIM, 
        OP_LOGE(nodeName, "actualTokenNum dim must be equal to 1, cur actualTokenNumDim=%lu!", actualTokenNumShape->GetStorageShape().GetDimNum()), return false);

    OP_TILING_CHECK(!CheckInputDim0Dim1(context, nodeName, tilingData), OP_LOGE(nodeName,  "Check Inputsdim0ordim1 failed!"), return false);

    if (isInputRankTable) {
        OP_TILING_CHECK(attnRankTableShape->GetStorageShape().GetDimNum() != ONE_DIM, 
            OP_LOGE(nodeName, "attnRankTable dim must be equal to 1, cur attnRankTableDim=%lu!", attnRankTableShape->GetStorageShape().GetDimNum()), return false);
        const int32_t attnRankTableDim0 = attnRankTableShape->GetStorageShape().GetDim(INDEX_ZERO);
        tilingData.ffnToAttentionInfo.A = attnRankTableDim0;

        OP_TILING_CHECK(attnRankTableDim0 > worldSize - 1, 
            OP_LOGE(nodeName, "attnRankTable dim0 must be less than or equal to %lu, cur attnRankTableDim0=%lu!", worldSize - 1, attnRankTableShape->GetStorageShape().GetDimNum()), return false);
    }
    tilingData.ffnToAttentionInfo.isInputRankTable = isInputRankTable;
    return true;
}

static bool CheckInputDataType(gert::TilingContext* context, const char *nodeName)
{   
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(INPUT_ATTN_RANK_TABLE_INDEX);
    bool isInputRankTable = (attnRankTableShape != nullptr);

    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);
    
    auto sessionIdDesc = context->GetInputDesc(INPUT_SESSION_IDS_INDEX);
    OP_TILING_CHECK(sessionIdDesc == nullptr, OP_LOGE(nodeName, "sessionIdDesc is null."), return false);
    OP_TILING_CHECK(sessionIdDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName, 
        "sessionId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(sessionIdDesc->GetDataType()).c_str()), return false);
    
    auto microBatchIdDesc = context->GetInputDesc(INPUT_MICRO_BATCH_IDS_INDEX);
    OP_TILING_CHECK(microBatchIdDesc == nullptr, OP_LOGE(nodeName, "microBatchIdDesc is null."), return false);
    OP_TILING_CHECK(microBatchIdDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName, 
        "microBatchId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(microBatchIdDesc->GetDataType()).c_str()), return false);
    
    auto tokenIdDesc = context->GetInputDesc(INPUT_TOKEN_IDS_INDEX);
    OP_TILING_CHECK(tokenIdDesc == nullptr, OP_LOGE(nodeName, "tokenIdDesc is null."), return false);
    OP_TILING_CHECK(tokenIdDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName, 
        "tokenId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(tokenIdDesc->GetDataType()).c_str()), return false);

    auto expertOffsetDesc = context->GetInputDesc(INPUT_EXPERT_OFFSETS_INDEX);
    OP_TILING_CHECK(expertOffsetDesc == nullptr, OP_LOGE(nodeName, "expertOffsetDesc is null."), return false);
    OP_TILING_CHECK(expertOffsetDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
        "expertOffset dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(expertOffsetDesc->GetDataType()).c_str()), return false);
    
    auto atcualTokenNumDesc = context->GetInputDesc(INPUT_ACTUAL_TOKEN_NUM_INDEX);
    OP_TILING_CHECK(atcualTokenNumDesc == nullptr, OP_LOGE(nodeName, "atcualTokenNumDesc is null."), return false);
    OP_TILING_CHECK(atcualTokenNumDesc->GetDataType() != ge::DT_INT64, OP_LOGE(nodeName,
        "atcualTokenNum dataType is invalid, dataType should be int64, but is %s.",
        Ops::Base::ToString(atcualTokenNumDesc->GetDataType()).c_str()), return false);
    
    if (isInputRankTable) {
        auto attnRankTableDesc = context->GetInputDesc(INPUT_ATTN_RANK_TABLE_INDEX);
         OP_TILING_CHECK(attnRankTableDesc == nullptr, OP_LOGE(nodeName, "atcualTokenNumDesc is null."), return false);
         OP_TILING_CHECK(attnRankTableDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
            "attnRankTable dataType is invalid, dataType should be int32, but is %s.",
            Ops::Base::ToString(attnRankTableDesc->GetDataType()).c_str()), return false);      
    }

    return true;
}

static bool CheckInputFormat(gert::TilingContext* context, const char *nodeName)
{   
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(INPUT_ATTN_RANK_TABLE_INDEX);
    bool isInputRankTable = (attnRankTableShape != nullptr);

    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "x format is invalid."), return false);
    
    auto sessionIdDesc = context->GetInputDesc(INPUT_SESSION_IDS_INDEX);
    OP_TILING_CHECK(sessionIdDesc == nullptr, OP_LOGE(nodeName, "sessionIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(sessionIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "sessionId format is invalid."), return false);
    
    auto microBatchIdDesc = context->GetInputDesc(INPUT_MICRO_BATCH_IDS_INDEX);
    OP_TILING_CHECK(microBatchIdDesc == nullptr, OP_LOGE(nodeName, "microBatchIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(microBatchIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "microBatchId format is invalid."), return false);
    
    auto tokenIdDesc = context->GetInputDesc(INPUT_TOKEN_IDS_INDEX);
    OP_TILING_CHECK(tokenIdDesc == nullptr, OP_LOGE(nodeName, "tokenIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tokenIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "tokenId format is invalid."), return false);

    auto expertOffsetDesc = context->GetInputDesc(INPUT_EXPERT_OFFSETS_INDEX);
    OP_TILING_CHECK(expertOffsetDesc == nullptr, OP_LOGE(nodeName, "expertOffsetDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertOffsetDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ, 
        OP_LOGE(nodeName, "expertOffset format is invalid."), return false);
    
    auto atcualTokenNumDesc = context->GetInputDesc(INPUT_ACTUAL_TOKEN_NUM_INDEX);
    OP_TILING_CHECK(atcualTokenNumDesc == nullptr, OP_LOGE(nodeName, "atcualTokenNumDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(atcualTokenNumDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ, 
        OP_LOGE(nodeName, "atcualTokenNum format is invalid."), return false);

    if (isInputRankTable) {
        auto attnRankTableDesc = context->GetInputDesc(INPUT_ATTN_RANK_TABLE_INDEX);
        OP_TILING_CHECK(attnRankTableDesc == nullptr, OP_LOGE(nodeName, "attnRankTableDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(attnRankTableDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ, 
            OP_LOGE(nodeName, "attnRankTable format is invalid."), return false);
    }
    return true;
}

static bool CheckInputAndSetTilingData(gert::TilingContext* context, const char *nodeName, FFNToAttentionTilingData &tilingData)
{
    // 校验输入数据dim、format、dataType
    OP_TILING_CHECK(!CheckInputDim(context, nodeName, tilingData), OP_LOGE(nodeName, 
                "Check Inputs dim failed!"),
                return false);

    OP_TILING_CHECK(!CheckInputDataType(context, nodeName), OP_LOGE(nodeName,
                    "Check Inputs dataType is invalid."), return false);
    OP_TILING_CHECK(!CheckInputFormat(context, nodeName), OP_LOGE(nodeName,
                    "Check Inputs  format is invalid."), return false);
    return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
 
static void CalWinSize(FFNToAttentionTilingData &tilingData, uint64_t &neededSize, uint64_t &viableWindowSize)
{
    uint64_t microBatchNum = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.microBatchNum);
    uint64_t BS = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.BS);
    uint64_t expertNumPerToken = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.expertNumPerToken);
    uint64_t HS = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.HS);
 
    uint64_t tokenDataNeedWinSize = microBatchNum * BS * expertNumPerToken * HS * TOKEN_DATA_SIZE;
    uint64_t tokenInfoTableNeedWinSize = microBatchNum * BS * expertNumPerToken * TOKEN_INFO_SIZE;
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    neededSize = (tokenDataNeedWinSize + tokenInfoTableNeedWinSize) / MB_SIZE + 1;
    viableWindowSize = maxWindowSize / MB_SIZE;
    tilingData.ffnToAttentionInfo.totalWinSize = maxWindowSize;
 
    return;
}
 
static ge::graphStatus SetHcommCfg(gert::TilingContext *context, FFNToAttentionTilingData &tilingData, const std::string group)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "ffnToAttention group = %s", group.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus FFNToAttentionTilingFunc(gert::TilingContext* context)
{
    FFNToAttentionTilingData *tilingData = context->GetTilingData<FFNToAttentionTilingData>();
    std::string group = "";
    const char *nodeName = context->GetNodeName();
 
    // Function that get check and set Attrs
    OP_TILING_CHECK(!CheckAndSetAttrs(context, nodeName, *tilingData, group),
                    OP_LOGE(nodeName, "Check and set attributes failed!"),
                    return ge::GRAPH_FAILED);
 
    // Function that check input dim 、format、datatype
    OP_TILING_CHECK(!CheckInputAndSetTilingData(context, nodeName, *tilingData),
                    OP_LOGE(nodeName, "Check Inputs and Outputs failed!"), return ge::GRAPH_FAILED);
 
    // Check window Size
    uint64_t neededSize = 0;
    uint64_t viableWindowSize = 0;
    CalWinSize(*tilingData, neededSize, viableWindowSize);
    OP_TILING_CHECK(neededSize > viableWindowSize, 
                    OP_LOGE(nodeName, "needed size:%lu is greater than viable window size:%lu.", neededSize, viableWindowSize),
                    return ge::GRAPH_FAILED);
 
    // Set WorkSpace
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
 
    // Set HcommCfg
    OP_TILING_CHECK(SetHcommCfg(context, *tilingData, group) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "setHcommCfg failed."),
        return ge::GRAPH_FAILED);
 
    // Set TilingKey
    bool rankTableMode = tilingData->ffnToAttentionInfo.isInputRankTable;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A3);
    OP_LOGD(nodeName, "cur case tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
 
    // Set numBlocks
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->ffnToAttentionInfo.totalUbSize = ubSize;
    tilingData->ffnToAttentionInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
 
    PrintTilingDataInfo(nodeName, *tilingData);
    OP_LOGD("FFNToAttention", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}
 
struct FFNToAttentionCompileInfo {};
ge::graphStatus TilingParseForFFNToAttention(gert::TilingParseContext *context) { 
    (void)context;
	return ge::GRAPH_SUCCESS; 
}
 
IMPL_OP_OPTILING(FFNToAttention)
    .Tiling(FFNToAttentionTilingFunc)
    .TilingParse<FFNToAttentionCompileInfo>(TilingParseForFFNToAttention);
}  // end of namespace optiling