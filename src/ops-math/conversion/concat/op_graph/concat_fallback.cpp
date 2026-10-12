/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_graph/op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t kConcatDAttrDim = 0;
constexpr size_t kConcatOut = 0;

ge::graphStatus ExecuteOpLaunch(gert::OpExecuteLaunchContext* context)
{
    auto params = reinterpret_cast<OpApiParams*>(context->GetOpApiParams());
    auto workspaceSizes = context->GetWorkspaceSizes();
    auto workspaceAddrs = context->GetWorkspaceAddrs();
    OP_CHECK_IF((workspaceSizes->GetSize() == 0) || (workspaceAddrs->GetSize() == 0),
                OP_LOGE("aclnnfallback", "no workspace addrs"), return ge::GRAPH_FAILED);
    auto workspaceSize = workspaceSizes->GetData()[0];
    auto workspaceAddr = workspaceAddrs->GetData()[0]->GetAddr();

    auto aclStream = context->GetStream();
    auto opApiFunc = params->op_api_func;
    OP_CHECK_IF(opApiFunc == nullptr, OP_LOGE("aclnnfallback", "opApiFunc nullptr"), return ge::GRAPH_FAILED);
    auto opApiRet = opApiFunc(workspaceAddr, workspaceSize, params->executor, aclStream);
    for (auto& av : params->converted_params) {
        if (av.deleter != nullptr) {
            av.deleter(av.pointer);
        }
    }
    params->converted_params.clear();
    if (opApiRet != 0) {
        OP_LOGE("aclnnfallback", "call %s allocate workspace failed opApiRet: %d", context->GetNodeName(), opApiRet);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static graphStatus ConcatExecuteFuncD(OpExecutePrepareContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE("aclnnfallback", "hostApiCtx is null"), return GRAPH_FAILED);

    auto inputNum = hostApiCtx->GetComputeNodeInputNum();
    std::vector<const gert::Tensor*> geTenserList;
    for (size_t i = 0; i < inputNum; i++) {
        auto geT = hostApiCtx->GetInputTensor(i);
        geTenserList.push_back(geT);
    }
    auto outGe = hostApiCtx->GetOutputTensor(kConcatOut);
    OP_CHECK_IF(outGe == nullptr, OP_LOGE("aclnnfallback", "outGe is null"), return GRAPH_FAILED);

    auto attrs = hostApiCtx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    const int64_t* concatDim = attrs->GetAttrPointer<int64_t>(kConcatDAttrDim);

    OP_CHECK_IF(concatDim == nullptr, OP_LOGE("aclnnfallback", "concatDim is null"), return GRAPH_FAILED);

    OP_LOGI("aclnnfallback", "concatDim: %ld", *concatDim);

    auto apiRet = CANN_OPS_OPB_ASYN_EXEC_ACLNN(hostApiCtx, aclnnCat, geTenserList, *concatDim, outGe);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet failed:%u", apiRet), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

static graphStatus ConcatExecuteFunc(OpExecutePrepareContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE("aclnnfallback", "hostApiCtx is nullptr"), return GRAPH_FAILED);

    auto inputNum = hostApiCtx->GetComputeNodeInputNum();
    OP_CHECK_IF(inputNum <= 1, OP_LOGE("aclnnfallback", "inputNum <=1"), return GRAPH_FAILED);

    std::vector<const gert::Tensor*> geTenserList;
    for (size_t i = 0; i < inputNum - 1; i++) {
        auto geT = hostApiCtx->GetInputTensor(i);
        geTenserList.push_back(geT);
    }
    auto outGe = hostApiCtx->GetOutputTensor(kConcatOut);
    OP_CHECK_IF(outGe == nullptr, OP_LOGE("aclnnfallback", "outGe is nullptr"), return GRAPH_FAILED);

    auto geT = hostApiCtx->GetInputTensor(inputNum - 1);
    OP_CHECK_IF(geT == nullptr, OP_LOGE("aclnnfallback", "geT is nullptr"), return GRAPH_FAILED);

    const int64_t* concatDim = geT->GetData<int64_t>();
    OP_CHECK_IF(concatDim == nullptr, OP_LOGE("aclnnfallback", "concatDim is nullptr"), return GRAPH_FAILED);

    OP_LOGI("aclnnfallback", "concatDim: %ld", *concatDim);

    auto apiRet = CANN_OPS_OPB_ASYN_EXEC_ACLNN(hostApiCtx, aclnnCat, geTenserList, *concatDim, outGe);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet failed:%u", apiRet), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(Concat).Op2StageExecuteFuncs(ConcatExecuteFunc, ExecuteOpLaunch).HostInputs({1});
IMPL_OP(ConcatV2).Op2StageExecuteFuncs(ConcatExecuteFunc, ExecuteOpLaunch).HostInputs({1});

IMPL_OP(ConcatD).Op2StageExecuteFuncs(ConcatExecuteFuncD, ExecuteOpLaunch);
IMPL_OP(ConcatV2D).Op2StageExecuteFuncs(ConcatExecuteFuncD, ExecuteOpLaunch);

} // namespace fallback

#ifdef __cplusplus
}
#endif
