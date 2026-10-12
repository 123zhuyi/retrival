/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "mc2_log.h"

namespace fallback {
using namespace ge;
using namespace gert;

const char *moeDistributeCombineInfo = "MoeDistributeCombineFallback";

namespace {
struct OpInput {
    const gert::Tensor *expandX;
    const gert::Tensor *expertIds;
    const gert::Tensor *expandIdx;
    const gert::Tensor *epSendCounts;
    const gert::Tensor *expertScales;
};

struct OpOptionalInput {
    const gert::Tensor *tpSendCounts;
    const gert::Tensor *xActiveMask;
    const gert::Tensor *activationScale;
    const gert::Tensor *weightScale;
    const gert::Tensor *groupList;
    const gert::Tensor *expandScales;
};

struct OpOutput {
    const gert::Tensor *y; 
};

struct OpAttrs {
    const char *groupEp;
    const int64_t *epWorldSize;
    const int64_t *epRankId;
    const int64_t *moeExpertNum;
    const char *groupTp;
    const int64_t *tpWorldSize;
    const int64_t *tpRankId;
    const int64_t *expertShardType;
    const int64_t *sharedExpertNum;
    const int64_t *sharedExpertRankNum;
    const int64_t *globalBs;
    const int64_t *outDtype;
    const int64_t *commQuantMode;
    const int64_t *groupListType;
};

// 获取必选输入
graphStatus MoeDistributeCombineGetOpInput(OpExecuteContext* host_api_ctx, OpInput &opInput)
{
    opInput.expandX = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
    OP_CHECK_IF(opInput.expandX == nullptr, OP_LOGE(moeDistributeCombineInfo,"expandX is null"), return ge::GRAPH_FAILED);

    opInput.expertIds = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
    OP_CHECK_IF(opInput.expertIds == nullptr, OP_LOGE(moeDistributeCombineInfo,"expertIds is null"), return ge::GRAPH_FAILED);

    opInput.expandIdx = host_api_ctx->GetInputTensor(static_cast<size_t>(2));
    OP_CHECK_IF(opInput.expandIdx == nullptr, OP_LOGE(moeDistributeCombineInfo,"expandIdx is null"), return ge::GRAPH_FAILED);

    opInput.epSendCounts = host_api_ctx->GetInputTensor(static_cast<size_t>(3));
    OP_CHECK_IF(opInput.epSendCounts == nullptr, OP_LOGE(moeDistributeCombineInfo,"epSendCounts is null"), return ge::GRAPH_FAILED);

    opInput.expertScales = host_api_ctx->GetInputTensor(static_cast<size_t>(4));
    OP_CHECK_IF(opInput.expertScales == nullptr, OP_LOGE(moeDistributeCombineInfo,"expertScales is null"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取可选输入
graphStatus MoeDistributeCombineGetOpOptionalInput(OpExecuteContext* host_api_ctx, OpOptionalInput &opOptionalInput)
{
    opOptionalInput.tpSendCounts = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(5));
    OP_CHECK_IF(opOptionalInput.tpSendCounts == nullptr, OP_LOGE(moeDistributeCombineInfo,"tpSendCounts is null"), return ge::GRAPH_FAILED);

    opOptionalInput.xActiveMask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(6));
    opOptionalInput.activationScale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(7));
    opOptionalInput.weightScale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(8));
    opOptionalInput.groupList = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(9));
    opOptionalInput.expandScales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(10));
    return ge::GRAPH_SUCCESS;
}

// 获取输出
graphStatus MoeDistributeCombineGetOpOutput(OpExecuteContext* host_api_ctx, OpOutput &opOutput)
{
    opOutput.y = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
    OP_CHECK_IF(opOutput.y == nullptr, OP_LOGE(moeDistributeCombineInfo,"y is null"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性
graphStatus MoeDistributeCombineGetOpAttrs(OpExecuteContext* host_api_ctx, OpAttrs &opAttrs)
{
    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(moeDistributeCombineInfo,"attrs is null"), return ge::GRAPH_FAILED);

    opAttrs.groupEp = attrs->GetStr(static_cast<size_t>(0));
    OP_CHECK_IF(opAttrs.groupEp == nullptr, OP_LOGE(moeDistributeCombineInfo,"groupEp is null"), return ge::GRAPH_FAILED);

    opAttrs.epWorldSize = attrs->GetInt(static_cast<size_t>(1));
    OP_CHECK_IF(opAttrs.epWorldSize == nullptr, OP_LOGE(moeDistributeCombineInfo,"epWorldSize is null"), return ge::GRAPH_FAILED);

    opAttrs.epRankId = attrs->GetInt(static_cast<size_t>(2));
    OP_CHECK_IF(opAttrs.epRankId == nullptr, OP_LOGE(moeDistributeCombineInfo,"epRankId is null"), return ge::GRAPH_FAILED);

    opAttrs.moeExpertNum = attrs->GetInt(static_cast<size_t>(3));
    OP_CHECK_IF(opAttrs.moeExpertNum == nullptr, OP_LOGE(moeDistributeCombineInfo,"moeExpertNum is null"), return ge::GRAPH_FAILED);

    opAttrs.groupTp = attrs->GetStr(static_cast<size_t>(4));
    OP_CHECK_IF(opAttrs.groupTp == nullptr, OP_LOGE(moeDistributeCombineInfo,"groupTp is null"), return ge::GRAPH_FAILED);

    opAttrs.tpWorldSize = attrs->GetInt(static_cast<size_t>(5));
    OP_CHECK_IF(opAttrs.tpWorldSize == nullptr, OP_LOGE(moeDistributeCombineInfo,"tpWorldSize is null"), return ge::GRAPH_FAILED);

    opAttrs.tpRankId = attrs->GetInt(static_cast<size_t>(6));
    OP_CHECK_IF(opAttrs.tpRankId == nullptr, OP_LOGE(moeDistributeCombineInfo,"tpRankId is null"), return ge::GRAPH_FAILED);

    opAttrs.expertShardType = attrs->GetInt(static_cast<size_t>(7));
    OP_CHECK_IF(opAttrs.expertShardType == nullptr, OP_LOGE(moeDistributeCombineInfo,"expertShardType is null"), return ge::GRAPH_FAILED);

    opAttrs.sharedExpertNum = attrs->GetInt(static_cast<size_t>(8));
    OP_CHECK_IF(opAttrs.sharedExpertNum == nullptr, OP_LOGE(moeDistributeCombineInfo,"sharedExpertNum is null"), return ge::GRAPH_FAILED);

    opAttrs.sharedExpertRankNum = attrs->GetInt(static_cast<size_t>(9));
    OP_CHECK_IF(opAttrs.sharedExpertRankNum == nullptr, OP_LOGE(moeDistributeCombineInfo,"sharedExpertRankNum is null"), return ge::GRAPH_FAILED);

    opAttrs.globalBs = attrs->GetInt(static_cast<size_t>(10));
    OP_CHECK_IF(opAttrs.globalBs == nullptr, OP_LOGE(moeDistributeCombineInfo,"globalBs is null"), return ge::GRAPH_FAILED);

    opAttrs.outDtype = attrs->GetInt(static_cast<size_t>(11));
    OP_CHECK_IF(opAttrs.outDtype == nullptr, OP_LOGE(moeDistributeCombineInfo,"out_dtype is null"), return ge::GRAPH_FAILED);

    opAttrs.commQuantMode = attrs->GetInt(static_cast<size_t>(12));
    OP_CHECK_IF(opAttrs.commQuantMode == nullptr, OP_LOGE(moeDistributeCombineInfo,"comm_quant_mode is null"), return ge::GRAPH_FAILED);

    opAttrs.groupListType = attrs->GetInt(static_cast<size_t>(13));
    OP_CHECK_IF(opAttrs.groupListType == nullptr, OP_LOGE(moeDistributeCombineInfo,"group_list_type is null"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

static graphStatus MoeDistributeCombineExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_LOGD(moeDistributeCombineInfo, "start to fallback for moeDistributeCombine");
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(moeDistributeCombineInfo,"host_api_ctx is null"), return ge::GRAPH_FAILED);

    OpInput opInput;
    OpOptionalInput opOptionalInput;
    OpOutput opOutput;
    OpAttrs opAttrs;

    OP_CHECK_IF(MoeDistributeCombineGetOpInput(host_api_ctx, opInput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo,"get input failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeCombineGetOpOptionalInput(host_api_ctx, opOptionalInput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo,"get optional input failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeCombineGetOpOutput(host_api_ctx, opOutput) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo,"get output failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MoeDistributeCombineGetOpAttrs(host_api_ctx, opAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE(moeDistributeCombineInfo,"get attrs failed"), return ge::GRAPH_FAILED);

    const auto apiRet = EXEC_OPAPI_CMD(aclnnMoeDistributeCombine,
        opInput.expandX, opInput.expertIds, opInput.expandIdx, opInput.epSendCounts,
        opInput.expertScales, opOptionalInput.tpSendCounts, opOptionalInput.xActiveMask,
        opOptionalInput.activationScale, opOptionalInput.weightScale, opOptionalInput.groupList,
        opOptionalInput.expandScales, opAttrs.groupEp, *opAttrs.epWorldSize, *opAttrs.epRankId,
        *opAttrs.moeExpertNum, opAttrs.groupTp, *opAttrs.tpWorldSize, *opAttrs.tpRankId,
        *opAttrs.expertShardType, *opAttrs.sharedExpertNum, *opAttrs.sharedExpertRankNum,
        *opAttrs.globalBs, *opAttrs.outDtype, *opAttrs.commQuantMode,
        *opAttrs.groupListType, opOutput.y);
    OP_CHECK_IF(apiRet != ge::GRAPH_SUCCESS, OP_LOGE(moeDistributeCombineInfo,"aclnn api error code %u", apiRet), return apiRet);
    return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeCombine).OpExecuteFunc(MoeDistributeCombineExecuteFunc);

}  // namespace fallback