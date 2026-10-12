/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_weight_quant_preprocess.h"

#include "weight_quant_preprocess_registry.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace {

static aclnnStatus ExecuteDataFlow(QuantContext& ctx, const std::vector<DataFlowEntry>& entries)
{
    const DataFlowEntry* matched = nullptr;
    for (const auto& entry : entries) {
        if (entry.judge(ctx)) {
            matched = &entry;
            break;
        }
    }

    OP_CHECK(matched != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[NpuArch=%u] dataflow is UNKNOWN.", static_cast<uint32_t>(ctx.npuArch)),
             return ACLNN_ERR_PARAM_INVALID);

    for (const auto& check : matched->checks) {
        auto ret = check(ctx);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    for (const auto& process : matched->processes) {
        auto ret = process(ctx);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnWeightQuantPreprocessGetWorkspaceSize(const aclTensor* weight, const aclTensor* weightScale,
                                                       const aclTensor* weightOffsetOptional,
                                                       const aclTensor* biasOptional, const aclDataType xDtype,
                                                       const aclDataType xScaleDtype, const int64_t kGroupSize,
                                                       aclTensor* outWeight, aclTensor* outWeightScale,
                                                       aclTensor* outWeightOffsetOptional, aclTensor* outBiasOptional,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnWeightQuantPreprocess,
                   DFX_IN(weight, weightScale, weightOffsetOptional, biasOptional, xDtype, xScaleDtype, kGroupSize),
                   DFX_OUT(outWeight, outWeightScale, outWeightOffsetOptional, outBiasOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    auto archIt = NPU_DATA_FLOW_REGISTRY_MAP.find(npuArch);
    OP_CHECK(archIt != NPU_DATA_FLOW_REGISTRY_MAP.end(),
             OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Unsupported NPU architecture: %u.", static_cast<uint32_t>(npuArch)),
             return ACLNN_ERR_RUNTIME_ERROR);
    OP_CHECK(weight != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Weight must not be nullptr."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(weightScale != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "WeightScale must not be nullptr."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(outWeight != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "outWeight must not be nullptr."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(outWeightScale != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "outWeightScale must not be nullptr."),
             return ACLNN_ERR_PARAM_NULLPTR);

    QuantContext ctx{weight,
                     weightScale,
                     weightOffsetOptional,
                     biasOptional,
                     ToOpDataType(xDtype),
                     ToOpDataType(xScaleDtype),
                     kGroupSize,
                     npuArch,
                     QuantDataFlow::UNKNOWN,
                     outWeight,
                     outWeightScale,
                     outWeightOffsetOptional,
                     outBiasOptional,
                     uniqueExecutor.get()};

    auto ret = ExecuteDataFlow(ctx, archIt->second);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnWeightQuantPreprocess(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnWeightQuantPreprocess);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
