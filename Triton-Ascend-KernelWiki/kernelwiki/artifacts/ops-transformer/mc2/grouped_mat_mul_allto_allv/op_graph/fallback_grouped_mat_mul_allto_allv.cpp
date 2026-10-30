/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fallback_grouped_mat_mul_allto_allv.cpp
 * \brief fallback function of op GroupedMatMulAlltoAllv
 */
#include "mc2_log.h"
#include "fallback/fallback.h"
#include "op_mc2.h"

namespace fallback
{
// 输入参数和属性的校验
static ge::graphStatus CheckInputsAndAttrs(
    const gert::Tensor* gmmX,
    const gert::Tensor* gmmWeight,
    const char* group,
    const bool* transGmmWeight,
    const int64_t* epWorldSize)
{
    OPS_ERR_IF(gmmX == nullptr,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "gmmX is null"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmmWeight == nullptr,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "gmmWeight is null"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(group == nullptr,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "group is null"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(epWorldSize == nullptr,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "epWorldSize is null"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(transGmmWeight == nullptr,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "transGmmWeight is null"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 解析 sendCounts 和 recvCounts
static ge::graphStatus ParseSendRecvCounts(
    const gert::TypedContinuousVector<int64_t>* sendCounts,
    const gert::TypedContinuousVector<int64_t>* recvCounts,
    std::vector<int64_t>& actSendCountsSeqArray,
    std::vector<int64_t>& actRecvCountsSeqArray)
{
    OPS_ERR_IF(sendCounts == nullptr,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "sendCounts is null"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(recvCounts == nullptr,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "recvCounts is null"), return ge::GRAPH_FAILED);

    const size_t sendLen = static_cast<size_t>(sendCounts->GetSize());
    const int64_t* actSendSeqData = sendCounts->GetData();
    for (size_t i = 0UL; i < sendLen; i++) {
        actSendCountsSeqArray.push_back(actSendSeqData[i]);
    }

    const size_t recvLen = static_cast<size_t>(recvCounts->GetSize());
    const int64_t* actRecvSeqData = recvCounts->GetData();
    for (size_t i = 0UL; i < recvLen; i++) {
        actRecvCountsSeqArray.push_back(actRecvSeqData[i]);
    }

    return ge::GRAPH_SUCCESS;
}

// 校验输出参数
static ge::graphStatus CheckOutputTensors(const gert::Tensor* y, const gert::Tensor*& mmY)
{
    OPS_ERR_IF(y == nullptr, OP_LOGE("GroupedMatMulAlltoAllvFallback", "y is null"), return ge::GRAPH_FAILED);
    if ((mmY != nullptr) && (mmY->GetStorageShape().GetDimNum() == 0)) {
        mmY = nullptr;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupedMatMulAlltoAllvExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D("GroupedMatMulAlltoAllvFallback", "Start GroupedMatMulAlltoAllvFallback.");
    OPS_ERR_IF(host_api_ctx == nullptr, OP_LOGE("GroupedMatMulAlltoAllvFallback", "host_api_ctx is null"),
               return ge::GRAPH_FAILED);

    // 获取输入参数
    const gert::Tensor* gmmX = host_api_ctx->
        GetInputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvInputIdx::K_GMM_X));
    const gert::Tensor* gmmWeight = host_api_ctx->
        GetInputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvInputIdx::K_GMM_WEIGHT));
    const gert::Tensor* sendCountsTensor = host_api_ctx->
        GetOptionalInputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvInputIdx::K_SEND_COUNTS_TENSOR));
    const gert::Tensor* recvCountsTensor = host_api_ctx->
        GetOptionalInputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvInputIdx::K_RECV_COUNTS_TENSOR));
    const gert::Tensor* mmX = host_api_ctx->
        GetOptionalInputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvInputIdx::K_MM_X));
    const gert::Tensor* mmWeight = host_api_ctx->
        GetOptionalInputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvInputIdx::K_MM_WEIGHT));

    // 获取属性
    const gert::RuntimeAttrs* attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE("GroupedMatMulAlltoAllvFallback", "attrs is null"), return ge::GRAPH_FAILED);
    const char* group = attrs->GetStr(static_cast<size_t>(ops::GroupedMatMulAlltoAllvAttrIdx::K_GROUP));
    const int64_t* epWorldSize = attrs->
        GetInt(static_cast<size_t>(ops::GroupedMatMulAlltoAllvAttrIdx::K_EP_WORLD_SIZE));
    const gert::TypedContinuousVector<int64_t>* sendCounts = attrs->
        GetListInt(static_cast<size_t>(ops::GroupedMatMulAlltoAllvAttrIdx::K_SEND_COUNTS));
    const gert::TypedContinuousVector<int64_t>* recvCounts = attrs->
        GetListInt(static_cast<size_t>(ops::GroupedMatMulAlltoAllvAttrIdx::K_RECV_COUNTS));
    const bool* transGmmWeight = attrs->
        GetBool(static_cast<size_t>(ops::GroupedMatMulAlltoAllvAttrIdx::K_TRANS_GMM_WEIGHT));
    const bool* transMmWeight = attrs->
        GetBool(static_cast<size_t>(ops::GroupedMatMulAlltoAllvAttrIdx::K_TRANS_MM_WEIGHT));

    // 输入参数和属性的校验
    ge::graphStatus ret = CheckInputsAndAttrs(gmmX, gmmWeight, group, transGmmWeight, epWorldSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 解析 sendCounts 和 recvCounts
    std::vector<int64_t> actRecvCountsSeqArray;
    std::vector<int64_t> actSendCountsSeqArray;
    ret = ParseSendRecvCounts(sendCounts, recvCounts,
                              actSendCountsSeqArray, actRecvCountsSeqArray);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 获取并校验输出参数
    auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvOutputIdx::K_Y));
    auto mmY = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::GroupedMatMulAlltoAllvOutputIdx::K_MM_Y));
    ret = CheckOutputTensors(y, mmY);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 计算
    const auto apiRet = EXEC_OPAPI_CMD(aclnnGroupedMatMulAlltoAllv,
                                       gmmX, gmmWeight, sendCountsTensor, recvCountsTensor, mmX, mmWeight,
                                       group, *epWorldSize, actSendCountsSeqArray, actRecvCountsSeqArray,
                                       *transGmmWeight, *transMmWeight, y, mmY);
    OPS_ERR_IF(apiRet != ge::GRAPH_SUCCESS,
        OP_LOGE("GroupedMatMulAlltoAllvFallback", "Aclnn api error code %u", apiRet), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(GroupedMatMulAlltoAllv).OpExecuteFunc(GroupedMatMulAlltoAllvExecuteFunc);
}  // namespace fallback