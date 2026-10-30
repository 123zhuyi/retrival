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
 * \file fallback_matmul_allto_all.cpp
 * \brief 动态shape图回调aclnn
 */
#include "fallback/fallback.h"
#include "op_mc2.h"
#include "mc2_log.h"

namespace fallback {

constexpr size_t INDEX_IN_X1 = 0;
constexpr size_t INDEX_IN_X2 = 1;
constexpr size_t INDEX_IN_X1_SCALE = 3;
constexpr size_t INDEX_IN_X2_SCALE = 4;
constexpr size_t INDEX_IN_BIAS = 2;
constexpr size_t INDEX_IN_COMM_SCALE = 5;
constexpr size_t INDEX_IN_X1_OFFSET = 6;
constexpr size_t INDEX_IN_X2_OFFSET = 7;
constexpr size_t INDEX_ATTR_GROUP = 0;
constexpr size_t INDEX_ATTR_COMMON_QUANT_MODE = 6;
constexpr size_t INDEX_ATTR_COMMON_QUANT_DTYPE = 7;
constexpr size_t INDEX_ATTR_WORLD_SIZE = 1;
constexpr size_t INDEX_ATTR_ALL2ALL_AXES = 2;
constexpr size_t INDEX_ATTR_Y_DTYPE = 3;
constexpr size_t INDEX_ATTR_X1_QUANT_MODE = 4;
constexpr size_t INDEX_ATTR_X2_QUANT_MODE = 5;
constexpr size_t INDEX_ATTR_TRANS_X2 = 9;
constexpr size_t INDEX_ATTR_TRANS_X1 = 8;
constexpr size_t INDEX_ATTR_GROUP_SIZE = 10;
constexpr size_t INDEX_OUT = 0;
constexpr uint64_t X1_QUANT_MODE_NUM = 3;
constexpr uint64_t X2_QUANT_MODE_NUM = 2;
    
const char* MatmulAlltoAllInfo = "MatmulAlltoAllFallback";

// 公共输入参数结构体
struct CommonMatmulParas {
    /* const aclTensor *会导致Release重载方法匹配不上，造成内存泄漏 */
    aclTensor* x1_acl;
    aclTensor* x2_acl;
    const gert::Tensor* bias;
};

// Attr参数结构体
struct AttrParas {
    aclTensor* commScaleOptional = nullptr;
    aclTensor* x1OffsetOptional = nullptr;
    aclTensor* x2OffsetOptional = nullptr;
    const char* group;
    const gert::TypedContinuousVector<int64_t>* alltoAllAxesOptional;
    int64_t commQuantMode;
    int64_t commQuantDtype;
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize = 0;
};

// 量化输入参数结构体
struct QuantMatmulParas {
    aclTensor* x1_scale_acl = nullptr;
    aclTensor* x2_scale_acl = nullptr;
};

/**
 * @brief 获取公共Matmul中输入参数
 * @param host_api_ctx
 * @param para
 */
inline ge::graphStatus GetCommonMatmulInputPara(const gert::OpExecuteContext* host_api_ctx, CommonMatmulParas& para)
{
    const auto x2 = host_api_ctx->GetInputTensor(INDEX_IN_X2);
    OPS_CHECK(x2 == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x2 is null"), return ge::GRAPH_FAILED);

    const auto x1 = host_api_ctx->GetInputTensor(INDEX_IN_X1);
    OPS_CHECK(x1 == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x1 is null"), return ge::GRAPH_FAILED);

    para.bias = host_api_ctx->GetOptionalInputTensor(INDEX_IN_BIAS);

    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "Attrs is null"), return ge::GRAPH_FAILED);

    para.x1_acl = ConvertMmType(x1, false);
    OPS_CHECK(para.x1_acl == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x1_acl is null"), return ge::GRAPH_FAILED);

    const bool* transX2Ptr = attrs->GetBool(static_cast<size_t>(INDEX_ATTR_TRANS_X2));
    const bool x2Trans = (transX2Ptr != nullptr ? *transX2Ptr : false);
    para.x2_acl = ConvertMmType(x2, x2Trans);
    OPS_CHECK(para.x2_acl == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x2_acl is null"), return ge::GRAPH_FAILED);

    return ge::SUCCESS;
}

static ge::graphStatus ParseRecvCounts(
    const gert::TypedContinuousVector<int64_t>* sendCounts,
    std::vector<int64_t>& actSendCountsSeqArray)
{
    const size_t sendLens = static_cast<size_t>(sendCounts->GetSize());
    const int64_t* actSendSeqData = sendCounts->GetData();
    for (size_t i = 0UL; i < sendLens; i++) {
        actSendCountsSeqArray.push_back(actSendSeqData[i]);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取attr参数值
 * @param host_api_ctx
 * @param para
 */
inline ge::graphStatus GetAttrPara(const gert::OpExecuteContext* host_api_ctx, AttrParas& para)
{
    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "attrs is null."), return ge::GRAPH_FAILED);
    para.group = attrs->GetStr(INDEX_ATTR_GROUP);
    OPS_CHECK(para.group == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "group is null."), return ge::GRAPH_FAILED);

    const bool* transX2Ptr = attrs->GetBool(INDEX_ATTR_TRANS_X2);
    const bool transX2 = (transX2Ptr != nullptr ? *transX2Ptr : false);
    const auto commScaleOptional = host_api_ctx->GetOptionalInputTensor(INDEX_IN_COMM_SCALE);
    if (commScaleOptional != nullptr) {
        para.commScaleOptional = ConvertMmType(commScaleOptional, transX2);
        OPS_CHECK(
            para.commScaleOptional == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "commScaleOptional is null"),
            return ge::GRAPH_FAILED);
    }
    const auto x2OffsetOptional = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X2_OFFSET);
    if (x2OffsetOptional != nullptr) {
        para.x2OffsetOptional = ConvertMmType(x2OffsetOptional, false);
        OPS_CHECK(
            para.x2OffsetOptional == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x2OffsetOptional is null"),
            return ge::GRAPH_FAILED);
    }
    const auto x1OffsetOptional = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X1_OFFSET);
    if (x1OffsetOptional != nullptr) {
        para.x1OffsetOptional = ConvertMmType(x1OffsetOptional, false);
        OPS_CHECK(
            para.x1OffsetOptional == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x1OffsetOptional is null"),
            return ge::GRAPH_FAILED);
    }
    const int64_t* comm_quant_mode_ptr = attrs->GetInt(INDEX_ATTR_COMMON_QUANT_MODE);
    para.commQuantMode = (comm_quant_mode_ptr != nullptr ? *comm_quant_mode_ptr : 0);
    const int64_t* comm_quant_dtype_ptr = attrs->GetInt(INDEX_ATTR_COMMON_QUANT_DTYPE);
    para.commQuantDtype = (comm_quant_dtype_ptr != nullptr ? *comm_quant_dtype_ptr : static_cast<uint64_t>(ge::DataType::DT_UNDEFINED));
    const bool* transX1Ptr = attrs->GetBool(INDEX_ATTR_TRANS_X1);
    para.transposeX1 = (transX1Ptr != nullptr ? *transX1Ptr : false);
    para.transposeX2 = (transX2Ptr != nullptr ? *transX2Ptr : false);
    const int64_t* groupSize_ptr = attrs->GetInt(INDEX_ATTR_GROUP_SIZE);
    para.groupSize = (groupSize_ptr != nullptr ? *groupSize_ptr : 0);
    
    return ge::SUCCESS;
}

/**
 * @brief 获取量化Matmul输入参数
 * @param host_api_ctx
 * @param para
 */
inline ge::graphStatus GetQuantMatmulPara(const gert::OpExecuteContext* host_api_ctx, QuantMatmulParas& para)
{
    const auto x1_scale = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X1_SCALE);
    OPS_CHECK(x1_scale == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x1scale is null"), return ge::GRAPH_FAILED);
    para.x1_scale_acl = ConvertMmType(x1_scale, false);
    OPS_CHECK(para.x1_scale_acl == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x1_scale_acl is null"), return ge::GRAPH_FAILED);

    const auto x2_scale = host_api_ctx->GetOptionalInputTensor(INDEX_IN_X2_SCALE);
    OPS_CHECK(x2_scale == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x2scale is null"), return ge::GRAPH_FAILED);
    const auto attrs = host_api_ctx->GetAttrs();
    const bool* transX2Ptr = attrs->GetBool(INDEX_ATTR_TRANS_X2);
    const bool x2Trans = (transX2Ptr != nullptr ? *transX2Ptr : false);
    para.x2_scale_acl = ConvertMmType(x2_scale, x2Trans);
    OPS_CHECK(para.x2_scale_acl == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x2_scale_acl is null"), return ge::GRAPH_FAILED);

    return ge::SUCCESS;
}

/**
 * @brief 校验MatmulAlltoAll执行函数
 * @param host_api_ctx
 */
static ge::graphStatus MatmulAlltoAllExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OPS_LOG_D(MatmulAlltoAllInfo, "Start to fallback for matmul_allto_all.");
 	OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(MatmulAlltoAllInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);
    CommonMatmulParas mm_para;
    ge::graphStatus retPara = GetCommonMatmulInputPara(host_api_ctx, mm_para);
    OPS_CHECK(
        retPara != ge::SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get common matmul input paras."),
        return ge::GRAPH_FAILED);
    AttrParas attr_para;
    OPS_CHECK(GetAttrPara(host_api_ctx, attr_para) != ge::SUCCESS,
              OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get attr paras."), return ge::GRAPH_FAILED);

    const auto output = host_api_ctx->GetOutputTensor(INDEX_OUT);
    OPS_CHECK(output == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "output is null"), return ge::GRAPH_FAILED);
    
    const auto attrs = host_api_ctx->GetAttrs();
    OPS_CHECK(attrs == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const int64_t* x1QuantMode_ptr = attrs->GetInt(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t x1QuantMode = (x1QuantMode_ptr != nullptr ? *x1QuantMode_ptr : 0);
    const int64_t* x2QuantMode_ptr = attrs->GetInt(INDEX_ATTR_X2_QUANT_MODE);
    const int64_t x2QuantMode = (x2QuantMode_ptr != nullptr ? *x2QuantMode_ptr : 0);

    const auto alltoAllAxesOptional = attrs->GetListInt(INDEX_ATTR_ALL2ALL_AXES);
    std::vector<int64_t> actSeqArray;
    if(alltoAllAxesOptional != nullptr) {
        OPS_CHECK(alltoAllAxesOptional == nullptr,
            OP_LOGE(host_api_ctx->GetNodeName(), "alltoAllAxesOptional is null."),
            return ge::GRAPH_FAILED);
        ParseRecvCounts(alltoAllAxesOptional, actSeqArray);
    }

    if (x1QuantMode == 0 && x2QuantMode == 0) {
        const auto ret = EXEC_OPAPI_CMD(aclnnMatmulAlltoAll, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, actSeqArray, attr_para.group, attr_para.transposeX1,
                                        attr_para.transposeX2, output);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                   OPS_LOG_E(MatmulAlltoAllInfo, "Aclnn matmul allto all api error code %d", ret),
                   return ge::GRAPH_FAILED);
    } else if (x1QuantMode == X1_QUANT_MODE_NUM && x2QuantMode == X2_QUANT_MODE_NUM) {
        QuantMatmulParas quant_matmul_para;
        retPara = GetQuantMatmulPara(host_api_ctx, quant_matmul_para);
        OPS_CHECK(
            retPara != ge::SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "Failed to get quant matmul input paras."),
            return ge::GRAPH_FAILED);
        const auto ret = EXEC_OPAPI_CMD(aclnnQuantMatmulAlltoAll, mm_para.x1_acl, mm_para.x2_acl, mm_para.bias, quant_matmul_para.x1_scale_acl, quant_matmul_para.x2_scale_acl,
                                        attr_para.commScaleOptional, attr_para.x1OffsetOptional, attr_para.x2OffsetOptional, actSeqArray, attr_para.group,
                                        x1QuantMode, x2QuantMode, attr_para.commQuantMode, attr_para.commQuantDtype, attr_para.groupSize, attr_para.transposeX1, attr_para.transposeX2,
                                        output);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                   OPS_LOG_E(MatmulAlltoAllInfo, "Aclnn quant matmul allto all api error code %d", ret),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatmulAlltoAll).OpExecuteFunc(MatmulAlltoAllExecuteFunc);
}