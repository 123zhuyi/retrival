/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_sparse_attention_grad.h"

#include "block_sparse_attention_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include <acl/acl.h>
#include <algorithm>
#include <unordered_map>
#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {


static bool CheckDataType(const aclTensor *query,
                          const aclTensor *key,
                          const aclTensor *value)
{
    const DataType qDtype = query->GetDataType();
    const DataType kDtype = key->GetDataType();
    const DataType vDtype = value->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> validKvType = {
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
    };

    auto iter = validKvType.find(qDtype);
    if (iter == validKvType.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported query datatype %d.", static_cast<int>(qDtype));
        return false;
    }

    if (std::find(iter->second.begin(), iter->second.end(), kDtype) == iter->second.end() ||
        std::find(iter->second.begin(), iter->second.end(), vDtype) == iter->second.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Key/Value datatype mismatch with query.");
        return false;
    }

    return true;
}

static aclnnStatus CheckMandatoryTensors(const aclTensor *dout,
                                         const aclTensor *query,
                                         const aclTensor *key,
                                         const aclTensor *value,
                                         const aclTensor *attentionOut,
                                         const aclTensor *softmaxLse,
                                         const aclTensor *blockSparseMaskOptional)
{
    CHECK_RET(dout != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseBlockShape(const aclIntArray *blockShapeOptional)
{
    if (blockShapeOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "blockShapeOptional is null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    uint64_t size = blockShapeOptional->Size();
    if (size < 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional must contain at least two elements [x, y].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t *data = blockShapeOptional->GetData();
    if (data == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional data is null.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (data[0] <= 0 || data[1] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional values must be positive, got [%ld, %ld].", data[0], data[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateParams(const aclTensor *dout,
                                  const aclTensor *query,
                                  const aclTensor *key,
                                  const aclTensor *value,
                                  const aclTensor *attentionOut,
                                  const aclTensor *softmaxLse,
                                  const aclTensor *blockSparseMaskOptional,
                                  const aclTensor *attenMaskOptional,
                                  const aclIntArray *actualSeqLengthsOptional,
                                  const aclIntArray *actualSeqLengthsKvOptional,
                                  char *qInputLayout,
                                  char *kvInputLayout,
                                  const aclIntArray *blockShapeOptional,
                                  int64_t numKeyValueHeads,
                                  int64_t maskType,
                                  int64_t preTokens,
                                  int64_t nextTokens)
{
    CHECK_RET(CheckMandatoryTensors(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    if (!CheckDataType(query, key, value)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (attenMaskOptional!=nullptr){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "attenMaskOptional currently only supports nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maskType != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskType only supports 0, got %ld.", maskType);
        return ACLNN_ERR_PARAM_INVALID;
    }

       if (preTokens !=2147483647 || nextTokens !=2147483647 ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "preTokens and nextTokens must be 2147483647, got [%ld,%ld].", preTokens,nextTokens);
        return ACLNN_ERR_PARAM_INVALID;
    }
    
    if (qInputLayout == nullptr || kvInputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input layout strings are null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    std::string qLayout(qInputLayout);
    std::string kvLayout(kvInputLayout);
    
    if (qLayout == "TND" && actualSeqLengthsOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, " actualSeqLengthsOptional  is mandatory when qInputLayout is TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kvLayout == "TND" && actualSeqLengthsKvOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, " actualSeqLengthsKvOptional  is mandatory when kvInputLayout is TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }    

    // 验证Q layout
    if (qLayout != "TND" && qLayout != "BNSD") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qInputLayout only supports TND or BNSD, got %s.", qLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    
    // 验证KV layout
    if (kvLayout != "TND" && kvLayout != "BNSD") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvInputLayout only supports TND or BNSD, got %s.", kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    
    // 验证Q和KV格式一致性：如果其中一个是BNSD，另一个也必须是BNSD
    bool qIsBNSD = (qLayout == "BNSD");
    bool kvIsBNSD = (kvLayout == "BNSD");
    if (qIsBNSD != kvIsBNSD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
                "Q and KV layouts must match: if one is BNSD, the other must also be BNSD. "
                "Q layout: %s, KV layout: %s", qLayout.c_str(), kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ParseBlockShape(blockShapeOptional);
}

static aclnnStatus MakeContiguous(const aclTensor *&dout,
                                  const aclTensor *&query,
                                  const aclTensor *&key,
                                  const aclTensor *&value,
                                  const aclTensor *&attentionOut,
                                  const aclTensor *&softmaxLse,
                                  const aclTensor *&blockSparseMaskOptional,
                                  const aclTensor *&attenMaskOptional,
                                  aclOpExecutor *executor)
{
    dout = l0op::Contiguous(dout, executor);
    CHECK_RET(dout != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    attentionOut = l0op::Contiguous(attentionOut, executor);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    softmaxLse = l0op::Contiguous(softmaxLse, executor);
    CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executor);
    CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (attenMaskOptional != nullptr) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static string ConvertLayoutString(char *layoutStr)
{
    return op::ToString(layoutStr).GetString();
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *dout,
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    char *qInputLayout,
    char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    aclTensor *dq,
    aclTensor *dk,
    aclTensor *dv,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    aclnnStatus ret = ValidateParams(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional,
                                     attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                     qInputLayout, kvInputLayout, blockShapeOptional, 
                                     numKeyValueHeads, maskType, preTokens,nextTokens);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    
    L2_DFX_PHASE_1(aclnnBlockSparseAttentionGrad,
                   DFX_IN(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
                          actualSeqLengthsOptional, actualSeqLengthsKvOptional, qInputLayout, qInputLayout, numKeyValueHeads,
                          maskType, scaleValue, preTokens, nextTokens),
                   DFX_OUT(dq, dk, dv));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    ret = MakeContiguous(dout, query, key, value, attentionOut, softmaxLse,blockSparseMaskOptional, attenMaskOptional, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    string qInputLayoutStr = ConvertLayoutString(qInputLayout);
    string kvInputLayoutStr = ConvertLayoutString(kvInputLayout);
    
    auto outputs = l0op::BlockSparseAttentionGrad(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, 
                                                  attenMaskOptional, blockShapeOptional,actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                                  const_cast<char*>(qInputLayoutStr.c_str()), 
                                                  const_cast<char*>(kvInputLayoutStr.c_str()), numKeyValueHeads,
                                                  maskType, scaleValue, preTokens, nextTokens, executorImpl);
    if (outputs[0] == nullptr || outputs[1] == nullptr || outputs[2] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockSparseAttentionGrad returned nullptr outputs.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult0 = l0op::ViewCopy(outputs[0], dq, executorImpl);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(outputs[1], dk, executorImpl);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult2 = l0op::ViewCopy(outputs[2], dv, executorImpl);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockSparseAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

